#!/usr/bin/env python3
"""Host-only exact-packet and source contracts for cooperative direct packing."""

from __future__ import annotations

import random
import unittest
from pathlib import Path


NUM_OWNERS = 8
WARP_SIZE = 32
NUM_HELPERS = 4
EXPERTS_PER_HELPER = 14
EXPERTS_PER_RANK = NUM_HELPERS * EXPERTS_PER_HELPER
MAX_TOKENS = 48
MAX_ROUTES = MAX_TOKENS * 16

REPO_ROOT = Path(__file__).resolve().parents[1]


def _serial_pack(counts, staged):
    packed_counts = [0] * EXPERTS_PER_RANK
    packed_routes = []
    for expert in range(EXPERTS_PER_RANK):
        count = counts[expert]
        if count != len(staged[expert]):
            raise AssertionError("count does not match staged route cells")
        packed_counts[expert] = count
        packed_routes.extend(staged[expert])
    return tuple(packed_counts), tuple(packed_routes)


def _cooperative_pack(counts, staged):
    """Model lanes owner+8*helper and the device kernel's two passes."""
    packed_counts = [None] * EXPERTS_PER_RANK
    packed_routes = [None] * sum(counts)
    segment_counts = [0] * NUM_HELPERS
    visits = []

    # First pass: each helper writes one consecutive 14-expert count segment.
    for helper in range(NUM_HELPERS):
        first_expert = helper * EXPERTS_PER_HELPER
        for offset in range(EXPERTS_PER_HELPER):
            expert = first_expert + offset
            packed_counts[expert] = counts[expert]
            segment_counts[helper] += counts[expert]
            visits.append(("count", helper, expert))

    # Full-warp shuffles give every helper all four segment totals.
    route_count = sum(segment_counts)
    for helper in range(NUM_HELPERS):
        prefix = sum(segment_counts[:helper])
        first_expert = helper * EXPERTS_PER_HELPER
        for offset in range(EXPERTS_PER_HELPER):
            expert = first_expert + offset
            count = counts[expert]
            if count != len(staged[expert]):
                raise AssertionError("count does not match staged route cells")
            packed_routes[prefix:prefix + count] = staged[expert]
            prefix += count
            visits.append(("route", helper, expert))
        if prefix != sum(segment_counts[:helper + 1]):
            raise AssertionError("helper did not cover its exact segment")

    if route_count != len(packed_routes) or any(
        route is None for route in packed_routes
    ):
        raise AssertionError("cooperative packet has a gap")
    return tuple(packed_counts), tuple(packed_routes), visits


def _seeded_owner(seed, owner):
    rng = random.Random((seed << 8) ^ owner)
    remaining = rng.randrange(MAX_ROUTES + 1)
    counts = []
    staged = []
    for expert in range(EXPERTS_PER_RANK):
        count = rng.randrange(min(MAX_TOKENS, remaining) + 1)
        remaining -= count
        counts.append(count)
        values = [
            (owner, expert, slot, rng.getrandbits(32))
            for slot in range(count)
        ]
        rng.shuffle(values)
        staged.append(values)
    return counts, staged


class TestMegaMoeCoopDirectPackModel(unittest.TestCase):
    def _assert_exact(self, counts, staged):
        expected = _serial_pack(counts, staged)
        actual_counts, actual_routes, visits = _cooperative_pack(counts, staged)
        self.assertEqual((actual_counts, actual_routes), expected)
        self.assertEqual(
            len(visits), 2 * EXPERTS_PER_RANK
        )
        self.assertEqual(len(set(visits)), len(visits))

    def test_seeded_packets_match_serial_byte_order_for_all_owners(self):
        for seed in range(256):
            for owner in range(NUM_OWNERS):
                with self.subTest(seed=seed, owner=owner):
                    counts, staged = _seeded_owner(seed, owner)
                    self._assert_exact(counts, staged)

    def test_adversarial_helper_boundaries_are_exact(self):
        boundary_counts = [0] * EXPERTS_PER_RANK
        for expert, count in (
            (0, 1), (13, 2), (14, 3), (27, 4),
            (28, 5), (41, 6), (42, 7), (55, 8),
        ):
            boundary_counts[expert] = count
        staged = [
            [(expert, slot) for slot in range(count)]
            for expert, count in enumerate(boundary_counts)
        ]
        self._assert_exact(boundary_counts, staged)

    def test_zero_and_max_capacity_packets(self):
        cases = (
            [0] * EXPERTS_PER_RANK,
            [MAX_TOKENS] * 16 + [0] * 40,
            [0] * 40 + [MAX_TOKENS] * 16,
            [MAX_TOKENS if expert % 4 == 0 else 0
             for expert in range(EXPERTS_PER_RANK)],
        )
        for counts in cases:
            self.assertLessEqual(sum(counts), MAX_ROUTES)
            staged = [
                [(expert, slot) for slot in range(count)]
                for expert, count in enumerate(counts)
            ]
            self._assert_exact(counts, staged)

    def test_lane_mapping_keeps_original_eight_gin_leaders(self):
        workers = {
            owner: [owner + 8 * helper for helper in range(NUM_HELPERS)]
            for owner in range(NUM_OWNERS)
        }
        self.assertEqual(sorted(lane for lanes in workers.values()
                                for lane in lanes), list(range(WARP_SIZE)))
        self.assertEqual([lanes[0] for lanes in workers.values()],
                         list(range(NUM_OWNERS)))


class TestMegaMoeCoopDirectPackSourceContract(unittest.TestCase):
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
        candidate_start = cls.kernel.index(
            "// Preserve lanes 0-7 as the sole GIN issuers"
        )
        candidate_end = cls.kernel.index(
            "                    } else {\n"
            "                        if (use_gin_direct_dispatch and "
            "lane_idx < lsa_size)",
            candidate_start,
        )
        cls.candidate = cls.kernel[candidate_start:candidate_end]

    def test_feature_is_default_off_jit_keyed_and_requires_direct(self):
        self.assertIn("#define DG_MEGAMOE_GIN_COOP_DIRECT_PACK 0",
                      self.kernel)
        self.assertIn("static constexpr bool kMegaMoeGinCoopDirectPack",
                      self.kernel)
        self.assertIn(
            "not kMegaMoeGinCoopDirectPack or\n"
            "              kMegaMoeGinDirectDispatch",
            self.kernel,
        )
        self.assertIn("#define DG_MEGAMOE_GIN_COOP_DIRECT_PACK {}", self.jit)
        self.assertIn("bool gin_coop_direct_pack;", self.jit)
        self.assertIn(
            'get_env<int>("DG_MEGAMOE_GIN_COOP_DIRECT_PACK", 0)', self.jit
        )
        self.assertIn("not gin_coop_direct_pack or gin_direct_dispatch",
                      self.jit)
        self.assertIn("#define DG_MEGAMOE_GIN_COOP_DIRECT_PACK 1",
                      self.compile_tu)

    def test_candidate_is_four_helpers_by_fourteen_and_two_passes(self):
        self.assertIn("constexpr uint32_t kExpertsPerHelper = 14;",
                      self.candidate)
        self.assertIn("lane_idx & 7u", self.candidate)
        self.assertIn("lane_idx >> 3", self.candidate)
        self.assertEqual(
            self.candidate.count("helper_expert < kExpertsPerHelper"), 2
        )
        for source_lane in ("peer_in_lsa", "peer_in_lsa + 8u",
                            "peer_in_lsa + 16u", "peer_in_lsa + 24u"):
            self.assertIn(source_lane, self.candidate)

    def test_all_writers_fence_before_original_leader_mask_issues_gin(self):
        fence = self.candidate.index("__threadfence_system();")
        sync = self.candidate.index("__syncwarp();", fence)
        leaders = self.candidate.index("if (helper_idx == 0)", sync)
        publish = self.candidate.index(
            "mega_moe_gin_publish_direct_dispatch", leaders
        )
        self.assertLess(fence, sync)
        self.assertLess(sync, leaders)
        self.assertLess(leaders, publish)
        self.assertIn("/*context_stripe=*/ peer_in_lsa", self.candidate)

    def test_protocol_layout_barriers_and_fallback_remain_outside_candidate(self):
        self.assertNotIn("mega_moe_gin_wait_direct_dispatch", self.candidate)
        self.assertNotIn("nvlink_lsa_barrier", self.candidate)
        self.assertNotIn("get_direct_dispatch_ready_ptr(\n"
                         "                                    /*send=*/ true",
                         self.candidate)
        fallback_start = self.kernel.index(
            "                    } else {\n"
            "                        if (use_gin_direct_dispatch and "
            "lane_idx < lsa_size)"
        )
        fallback = self.kernel[fallback_start:].split(
            "if (not use_gin_direct_dispatch", 1
        )[0]
        self.assertIn("for (uint32_t local_expert = 0;", fallback)
        self.assertIn("local_expert < kNumExpertsPerRank", fallback)
        self.assertIn("__threadfence_system();", fallback)
        self.assertIn("mega_moe_gin_publish_direct_dispatch", fallback)


if __name__ == "__main__":
    unittest.main()
