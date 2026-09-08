#!/usr/bin/env python3
"""Exact model and source contracts for pre-consensus direct packing."""

from __future__ import annotations

import random
import unittest
from pathlib import Path

from test_mega_moe_coop_direct_pack_model import (
    EXPERTS_PER_RANK,
    MAX_ROUTES,
    MAX_TOKENS,
    _cooperative_pack,
    _serial_pack,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _packet(seed: int, tokens: int):
    """Build a bounded packet with arbitrary expert boundaries."""
    rng = random.Random(seed)
    remaining = rng.randrange(tokens * 16 + 1)
    counts = []
    staged = []
    for expert in range(EXPERTS_PER_RANK):
        count = rng.randrange(min(tokens, remaining) + 1)
        remaining -= count
        counts.append(count)
        staged.append([(seed, expert, slot) for slot in range(count)])
    return counts, staged


def _transition(token_counts, packets, direct_decisions):
    """Model scratch lifetime and post-consensus issue selection."""
    scratch = 0xDEADBEEF
    issued = []
    for tokens, packet, use_direct in zip(
        token_counts, packets, direct_decisions, strict=True
    ):
        prepared = tokens <= MAX_TOKENS
        if prepared:
            counts, staged = packet
            packed_counts, packed_routes, _ = _cooperative_pack(
                counts, staged
            )
            scratch = len(packed_routes)
        if use_direct:
            if not prepared:
                raise AssertionError("world selected direct without local prep")
            issued.append((scratch, packed_counts, packed_routes))
        else:
            issued.append(None)
    return issued


class TestMegaMoePreconsensusPackModel(unittest.TestCase):
    def test_prepared_packets_remain_byte_identical(self):
        for tokens in (0, 1, 32, 40, 48):
            for seed in range(128):
                counts, staged = _packet(seed, tokens)
                expected = _serial_pack(counts, staged)
                actual_counts, actual_routes, _ = _cooperative_pack(
                    counts, staged
                )
                self.assertEqual((actual_counts, actual_routes), expected)
                self.assertLessEqual(len(actual_routes), tokens * 16)
                self.assertLessEqual(len(actual_routes), MAX_ROUTES)

    def test_48_64_48_transition_never_uses_stale_scratch(self):
        packets = (_packet(1, 48), _packet(2, 64), _packet(3, 48))
        issued = _transition((48, 64, 48), packets, (True, False, True))
        self.assertIsNotNone(issued[0])
        self.assertIsNone(issued[1])
        self.assertIsNotNone(issued[2])
        self.assertEqual(issued[0][0], sum(packets[0][0]))
        self.assertEqual(issued[2][0], sum(packets[2][0]))

    def test_locally_eligible_rank_can_prepare_then_follow_world_fallback(self):
        packets = (_packet(4, 48), _packet(5, 48))
        issued = _transition((48, 48), packets, (False, True))
        self.assertIsNone(issued[0])
        self.assertEqual(issued[1][0], sum(packets[1][0]))

    def test_zero_route_direct_packet_replaces_stale_route_count(self):
        zero = ([0] * EXPERTS_PER_RANK, [[] for _ in range(EXPERTS_PER_RANK)])
        issued = _transition((48, 0), (_packet(9, 48), zero), (True, True))
        self.assertGreaterEqual(issued[0][0], 0)
        self.assertEqual(issued[1][0], 0)


class TestMegaMoePreconsensusPackSourceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel = (
            REPO_ROOT
            / "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh"
        ).read_text(encoding="utf-8")
        cls.jit = (
            REPO_ROOT
            / "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp"
        ).read_text(encoding="utf-8")
        cls.compile_tu = (
            REPO_ROOT / "tests/compile_sm100_megamoe_gin.cu"
        ).read_text(encoding="utf-8")
        cls.prep_start = cls.kernel.index(
            "if constexpr (kUseGin and kMegaMoeGinPreconsensusPack)"
        )
        cls.consensus_start = cls.kernel.index(
            "if constexpr (kUseGin and kMegaMoeGinActiveFastPath)",
            cls.prep_start,
        )
        cls.prep = cls.kernel[cls.prep_start:cls.consensus_start]
        cls.issue_start = cls.kernel.index(
            "if constexpr (kMegaMoeGinPreconsensusPack)",
            cls.consensus_start,
        )
        cls.issue_end = cls.kernel.index("} else {", cls.issue_start)
        cls.issue = cls.kernel[cls.issue_start:cls.issue_end]

    def test_flag_is_default_off_jit_keyed_and_strictly_dependent(self):
        self.assertIn("#define DG_MEGAMOE_GIN_PRECONSENSUS_PACK 0",
                      self.kernel)
        self.assertIn("static constexpr bool kMegaMoeGinPreconsensusPack",
                      self.kernel)
        self.assertIn("kMegaMoeGinCoopDirectPack and\n"
                      "               kMegaMoeGinActiveFastPath",
                      self.kernel)
        self.assertIn("bool gin_preconsensus_pack;", self.jit)
        self.assertIn("#define DG_MEGAMOE_GIN_PRECONSENSUS_PACK {}",
                      self.jit)
        self.assertIn(
            'get_env<int>("DG_MEGAMOE_GIN_PRECONSENSUS_PACK", 0)',
            self.jit,
        )
        self.assertIn("gin_coop_direct_pack and gin_active_fast_path",
                      self.jit)
        self.assertIn("#define DG_MEGAMOE_GIN_PRECONSENSUS_PACK 1",
                      self.compile_tu)

    def test_pack_is_after_route_grid_sync_and_before_consensus(self):
        route_grid = self.kernel.rfind(
            "comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>",
            0,
            self.prep_start,
        )
        self.assertGreater(route_grid, 0)
        self.assertLess(route_grid, self.prep_start)
        self.assertLess(self.prep_start, self.consensus_start)
        self.assertIn("sm_idx == 1 and warp_idx == 0", self.prep)
        self.assertIn("num_tokens <= "
                      "layout::kMegaMoeGinDirectDispatchMaxTokens",
                      self.prep)

    def test_preparation_has_no_gin_issue_and_preserves_packet_order(self):
        self.assertNotIn("mega_moe_gin_publish_direct_dispatch", self.prep)
        self.assertNotIn("flushAsync", self.prep)
        self.assertNotIn("wait_direct_dispatch", self.prep)
        self.assertEqual(
            self.prep.count("helper_expert < kExpertsPerHelper"), 2
        )
        self.assertIn("packed_counts[local_expert] = count", self.prep)
        self.assertIn("packed_routes[route_prefix + slot]", self.prep)
        self.assertIn("__threadfence_system();", self.prep)
        self.assertIn("__syncwarp();", self.prep)

    def test_route_count_uses_only_untransmitted_send_ready_prefix(self):
        self.assertIn("get_direct_dispatch_ready_ptr(\n"
                      "                                /*send=*/ true",
                      self.prep)
        self.assertIn("get_direct_dispatch_ready_ptr(\n"
                      "                                                "
                      "/*send=*/ true",
                      self.issue)
        self.assertIn("get_direct_dispatch_count_ptr(\n"
                      "                                        "
                      "/*send=*/ true",
                      self.issue)
        self.assertIn("mega_moe_gin_publish_direct_dispatch", self.issue)
        self.assertNotIn("get_route_staging_ptr", self.issue)
        self.assertNotIn("helper_expert", self.issue)

    def test_issue_remains_post_consensus_and_world_selected(self):
        final_decision_grid = self.kernel.index(
            "comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>",
            self.consensus_start,
        )
        issue_role = self.kernel.rfind(
            "if (sm_idx == 1 and warp_idx == 0)",
            self.consensus_start,
            self.issue_start,
        )
        self.assertLess(final_decision_grid, self.issue_start)
        self.assertGreater(issue_role, self.consensus_start)
        self.assertIn("sm_idx == 1 and warp_idx == 0", self.prep)
        self.assertIn("if (use_gin_direct_dispatch and", self.issue)
        self.assertIn("lane_idx < lsa_size", self.issue)
        self.assertIn("peer_in_lsa = lane_idx", self.issue)
        self.assertIn("helper_idx == 0", self.prep)
        self.assertNotIn("__threadfence_system", self.issue)


if __name__ == "__main__":
    unittest.main()
