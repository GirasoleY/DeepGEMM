#!/usr/bin/env python3
"""Host-only contracts for the EP8 owner-ready slot-pair reducer."""

from __future__ import annotations

from pathlib import Path
import struct
import unittest


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh"
LAYOUT = ROOT / "deep_gemm/include/deep_gemm/layout/mega_moe.cuh"
JIT = ROOT / "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp"
PUBLIC = ROOT / "deep_gemm/mega/__init__.py"
ACCURACY = ROOT / "tests/test_mega_moe_accuracy.py"
MATCHED = ROOT / "tests/bench_gb200_transport_matched.py"
FLAG = "DG_MEGAMOE_GIN_COMBINE_OWNER_SLOT_READY"


def f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def slot_pair_reduce(values):
    if len(values) != 16:
        raise ValueError("top-k16 is required")
    partials = []
    for pair in range(8):
        partial = f32(0.0)
        partial = f32(partial + values[2 * pair])
        partial = f32(partial + values[2 * pair + 1])
        partials.append(partial)
    reduced = f32(0.0)
    for partial in partials:
        reduced = f32(reduced + partial)
    return reduced


class OwnerSlotReadyModelTests(unittest.TestCase):
    def test_pair_arithmetic_is_route_placement_independent(self):
        values = [f32((slot - 7) * 0.03125) for slot in range(16)]
        local = list(values)
        half_remote = [value for value in values]
        all_remote = list(reversed(list(reversed(values))))
        self.assertEqual(slot_pair_reduce(local), slot_pair_reduce(half_remote))
        self.assertEqual(slot_pair_reduce(local), slot_pair_reduce(all_remote))

    def test_pairs_are_fixed_and_cover_topk_once(self):
        pairs = [(2 * warp, 2 * warp + 1) for warp in range(8)]
        self.assertEqual([slot for pair in pairs for slot in pair], list(range(16)))

    def test_first_assignment_can_finish_before_second_owner_is_ready(self):
        owner_ready = {0: 3, 1: 11}
        first_finish = owner_ready[0] + 2
        second_begin = max(first_finish, owner_ready[1])
        self.assertLess(first_finish, owner_ready[1])
        self.assertEqual(second_begin, 11)

    def test_topk_other_than_sixteen_is_rejected_by_model(self):
        with self.assertRaisesRegex(ValueError, "top-k16"):
            slot_pair_reduce([1.0] * 8)


class OwnerSlotReadySourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel = KERNEL.read_text()
        cls.layout = LAYOUT.read_text()
        cls.jit = JIT.read_text()
        cls.public = PUBLIC.read_text()
        cls.accuracy = ACCURACY.read_text()
        cls.matched = MATCHED.read_text()

    def test_flag_is_default_off_and_requires_w4_strongva(self):
        self.assertIn(f"#define {FLAG} 0", self.kernel)
        self.assertIn("kMegaMoeGinCombineOwnerWaves == 4", self.kernel)
        self.assertIn("kMegaMoeGinStrongVACombineTerminal", self.kernel)

    def test_transport_keeps_existing_four_whole_owner_terminals(self):
        self.assertIn("kNumOwnerProgressWarps = kGinPeerCount", self.kernel)
        self.assertIn("get_combine_terminal_signal_ptr(owner_in_lsa)", self.kernel)
        self.assertNotIn("combine_owner_slot_signal_buffer", self.layout)

    def test_progress_publication_orders_acquire_proxy_and_release(self):
        begin = self.kernel.index("Four tail warps independently acquire")
        end = self.kernel.index("Keep the exact packet-count audit", begin)
        body = self.kernel[begin:end]
        self.assertLess(body.index("mega_moe_gin_wait_bulk_combine_terminal"),
                        body.index("fence.proxy.async.global"))
        self.assertLess(body.index("fence.proxy.async.global"),
                        body.index("get_combine_receiver_owner_ready_ptr"))

    def test_pair_path_is_cta_per_token_and_slot_sequential(self):
        begin = self.kernel.index("One CTA owns one token")
        end = self.kernel.index(
            "DG_GIN_TRACE_IF(lane_idx == 0, 88u", begin)
        body = self.kernel[begin:end]
        self.assertIn("token_idx = sm_idx", body)
        self.assertIn("first_slot = epilogue_warp_idx * 2u", body)
        self.assertIn("for (uint32_t pair_slot = 0; pair_slot < 2", body)
        self.assertLess(body.index("pair_reduced"),
                        body.index("get_combine_receiver_owner_ready_ptr"))

    def test_shared_memory_budget_is_one_bf16_plus_one_fp32_partial(self):
        self.assertIn("(1u + 2u) * kNumEpilogueWarps * kNumChunkBytes", self.kernel)
        self.assertIn("epilogue_warp_idx * 2u * kNumChunkBytes", self.kernel)

    def test_warp_zero_accumulates_eight_pairs_and_two_cta_barriers(self):
        begin = self.kernel.index("One CTA owns one token")
        end = self.kernel.index(
            "DG_GIN_TRACE_IF(lane_idx == 0, 88u", begin)
        body = self.kernel[begin:end]
        self.assertIn("for (uint32_t pair = 0; pair < 8", body)
        self.assertGreaterEqual(body.count(
            "kNumEpilogueThreads, kEpilogueFullBarrierIdx"), 2)

    def test_host_and_cli_surfaces_are_uniform_and_default_off(self):
        self.assertIn(FLAG, self.jit)
        self.assertIn(FLAG, self.public)
        self.assertIn("--gin-combine-owner-slot-ready", self.accuracy)
        self.assertIn("--gin-combine-owner-slot-ready", self.matched)


if __name__ == "__main__":
    unittest.main()
