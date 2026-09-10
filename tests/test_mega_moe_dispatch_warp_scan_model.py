#!/usr/bin/env python3
"""Host-only model and source contracts for direct-dispatch warp-scan restore."""

from __future__ import annotations

import random
import unittest
from pathlib import Path


NUM_SOURCES = 8
NUM_WARPS = 4
WARP_SIZE = 32
EXPERTS_PER_RANK = 56
MAX_TOKENS = 48
MAX_ROUTES = MAX_TOKENS * 16

REPO_ROOT = Path(__file__).resolve().parents[1]


def _legacy_restore(counts, routes):
    restored = {}
    for source in range(NUM_SOURCES):
        prefix = 0
        for expert, count in enumerate(counts[source]):
            restored[(source, expert)] = tuple(
                routes[source][prefix:prefix + count]
            )
            prefix += count
        if prefix != len(routes[source]):
            raise AssertionError("count vector does not cover route payload")
    return restored


def _exclusive_scan(values):
    prefix = []
    carry = 0
    for value in values:
        prefix.append(carry)
        carry += value
    return prefix, carry


def _warp_scan_restore(counts, routes):
    """Model four warps, two source waves, and two full-warp scans."""
    restored = {}
    visits = []
    for source_wave in range(2):
        for warp in range(NUM_WARPS):
            source = warp + source_wave * NUM_WARPS
            first_counts = counts[source][:WARP_SIZE]
            second_counts = (
                counts[source][WARP_SIZE:]
                + [0] * (2 * WARP_SIZE - EXPERTS_PER_RANK)
            )
            first_prefix, first_total = _exclusive_scan(first_counts)
            second_prefix, second_total = _exclusive_scan(second_counts)
            for lane in range(WARP_SIZE):
                expert = lane
                begin = first_prefix[lane]
                count = first_counts[lane]
                restored[(source, expert)] = tuple(
                    routes[source][begin:begin + count]
                )
                visits.append((source, expert))

                expert = lane + WARP_SIZE
                if expert < EXPERTS_PER_RANK:
                    begin = first_total + second_prefix[lane]
                    count = second_counts[lane]
                    restored[(source, expert)] = tuple(
                        routes[source][begin:begin + count]
                    )
                    visits.append((source, expert))
            if first_total + second_total != len(routes[source]):
                raise AssertionError("count vector does not cover route payload")
    return restored, visits


def _random_packets(seed):
    rng = random.Random(seed)
    counts = []
    routes = []
    for source in range(NUM_SOURCES):
        remaining = rng.randrange(MAX_ROUTES + 1)
        source_counts = []
        for expert in range(EXPERTS_PER_RANK):
            count = rng.randrange(min(MAX_TOKENS, remaining) + 1)
            source_counts.append(count)
            remaining -= count
        count_sum = sum(source_counts)
        source_routes = list(range(source * MAX_ROUTES,
                                   source * MAX_ROUTES + count_sum))
        rng.shuffle(source_routes)
        counts.append(source_counts)
        routes.append(source_routes)
    return counts, routes


class TestMegaMoeDispatchWarpScanModel(unittest.TestCase):
    def test_seeded_packets_exactly_match_legacy_restore(self):
        for seed in range(256):
            with self.subTest(seed=seed):
                counts, routes = _random_packets(seed)
                expected = _legacy_restore(counts, routes)
                actual, visits = _warp_scan_restore(counts, routes)
                self.assertEqual(actual, expected)
                self.assertEqual(len(visits), NUM_SOURCES * EXPERTS_PER_RANK)
                self.assertEqual(len(set(visits)), len(visits))

    def test_boundary_between_the_two_scans_is_exact(self):
        counts = [[0] * EXPERTS_PER_RANK for _ in range(NUM_SOURCES)]
        for source in range(NUM_SOURCES):
            counts[source][0] = 1
            counts[source][31] = 2
            counts[source][32] = 3
            counts[source][55] = 4
        routes = [
            [(source, slot) for slot in range(sum(counts[source]))]
            for source in range(NUM_SOURCES)
        ]
        actual, _ = _warp_scan_restore(counts, routes)
        self.assertEqual(actual[(0, 31)], ((0, 1), (0, 2)))
        self.assertEqual(
            actual[(0, 32)], ((0, 3), (0, 4), (0, 5))
        )
        self.assertEqual(actual[(0, 55)], tuple((0, i) for i in range(6, 10)))

    def test_zero_and_max_capacity_packets(self):
        cases = (
            [0] * EXPERTS_PER_RANK,
            [MAX_TOKENS] * 16 + [0] * (EXPERTS_PER_RANK - 16),
            [0] * 40 + [MAX_TOKENS] * 16,
        )
        for source_counts in cases:
            counts = [list(source_counts) for _ in range(NUM_SOURCES)]
            routes = [
                list(range(sum(source_counts)))
                for source_counts in counts
            ]
            expected = _legacy_restore(counts, routes)
            actual, _ = _warp_scan_restore(counts, routes)
            self.assertEqual(actual, expected)

    def test_count_load_reduction(self):
        legacy_prefix_loads = NUM_SOURCES * sum(range(EXPERTS_PER_RANK))
        legacy_count_loads = NUM_SOURCES * EXPERTS_PER_RANK
        scan_count_loads = NUM_SOURCES * EXPERTS_PER_RANK
        self.assertEqual(legacy_prefix_loads, 12320)
        self.assertEqual(legacy_count_loads, 448)
        self.assertEqual(scan_count_loads, 448)


class TestMegaMoeDispatchWarpScanSourceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel = (
            REPO_ROOT
            / "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh"
        ).read_text()
        cls.jit = (
            REPO_ROOT
            / "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp"
        ).read_text()
        cls.compile_tu = (
            REPO_ROOT / "tests/compile_sm100_megamoe_gin.cu"
        ).read_text()

    def test_feature_is_default_off_and_requires_direct_dispatch(self):
        self.assertIn(
            "#define DG_MEGAMOE_GIN_DISPATCH_WARP_SCAN 0", self.kernel
        )
        self.assertIn(
            "static constexpr bool kMegaMoeGinDispatchWarpScan", self.kernel
        )
        self.assertIn(
            "not kMegaMoeGinDispatchWarpScan or\n"
            "              kMegaMoeGinDirectDispatch",
            self.kernel,
        )
        self.assertIn(
            "#define DG_MEGAMOE_GIN_DISPATCH_WARP_SCAN {}", self.jit
        )
        self.assertIn("bool gin_dispatch_warp_scan;", self.jit)
        self.assertIn(
            'get_env<int>("DG_MEGAMOE_GIN_DISPATCH_WARP_SCAN", 0)',
            self.jit,
        )
        self.assertIn(
            "not gin_dispatch_warp_scan or gin_direct_dispatch", self.jit
        )
        self.assertIn(
            "#define DG_MEGAMOE_GIN_DISPATCH_WARP_SCAN 1",
            self.compile_tu,
        )

    def test_candidate_uses_four_warps_actual_peer_waves_and_two_scans(self):
        candidate = self.kernel.split(
            "if constexpr (kMegaMoeGinDispatchWarpScan) {", 1
        )[1].split("} else {", 1)[0]
        self.assertIn("kNumDispatchWarps == 4", candidate)
        self.assertIn("source_wave < kGinPeerCount / kNumDispatchWarps", candidate)
        self.assertIn(
            "warp_idx + source_wave *\n"
            "                                                   kNumDispatchWarps",
            candidate,
        )
        self.assertEqual(candidate.count("math::warp_inclusive_sum("), 2)
        self.assertIn("lane_idx + 32u", candidate)
        self.assertIn("first_total + second_inclusive", candidate)
        self.assertIn("0xffffffffu, first_inclusive, 31", candidate)

    def test_wire_protocol_and_outer_barriers_stay_outside_candidate(self):
        candidate = self.kernel.split(
            "if constexpr (kMegaMoeGinDispatchWarpScan) {", 1
        )[1].split("} else {", 1)[0]
        self.assertNotIn("mega_moe_gin_put", candidate)
        self.assertNotIn("mega_moe_gin_wait_direct_dispatch", candidate)
        self.assertNotIn("nvlink_lsa_barrier", candidate)
        self.assertNotIn("sync_aligned", candidate)
        restore = self.kernel.index(
            "if constexpr (kMegaMoeGinDispatchWarpScan) {"
        )
        wait = self.kernel.rindex(
            "mega_moe_gin_wait_direct_dispatch", 0, restore
        )
        post_restore_barrier = self.kernel.index(
            "ptx::sync_aligned(\n"
            "                        kNumDispatchThreads",
            restore,
        )
        self.assertLess(wait, restore)
        self.assertLess(restore, post_restore_barrier)

    def test_legacy_restore_remains_the_default_branch(self):
        fallback = self.kernel.split(
            "if constexpr (kMegaMoeGinDispatchWarpScan) {", 1
        )[1].split("} else {", 1)[1].split("__threadfence();", 1)[0]
        self.assertIn("kNumRemoteExpertPairs", fallback)
        self.assertIn("pair += kNumDispatchThreads", fallback)
        self.assertIn("for (uint32_t prior_expert = 0;", fallback)
        self.assertIn("prior_expert < local_expert", fallback)


if __name__ == "__main__":
    unittest.main()
