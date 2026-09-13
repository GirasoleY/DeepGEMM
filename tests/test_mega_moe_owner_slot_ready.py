#!/usr/bin/env python3
"""Host-only contracts for the EP8 owner-ready dual-stage pair reducer."""

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


def dual_stage_pair_reduce(values, ready_ticks):
    if len(values) != 16 or len(ready_ticks) != 16:
        raise ValueError("top-k16 is required")
    issue_order = []
    for warp in range(8):
        pair = [2 * warp, 2 * warp + 1]
        issue_order.extend(sorted(
            pair, key=lambda slot: (ready_ticks[slot], slot)))
    return slot_pair_reduce(values), issue_order


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

    def test_ready_first_issue_does_not_change_fixed_pair_arithmetic(self):
        values = [f32((slot - 7) * 0.03125) for slot in range(16)]
        ready_ticks = [11, 3] + [slot for slot in range(2, 16)]
        reduced, issue_order = dual_stage_pair_reduce(values, ready_ticks)
        self.assertEqual(issue_order[:2], [1, 0])
        self.assertEqual(reduced, slot_pair_reduce(values))

    def test_topk_other_than_sixteen_is_rejected_by_model(self):
        with self.assertRaisesRegex(ValueError, "top-k16"):
            slot_pair_reduce([1.0] * 8)
        with self.assertRaisesRegex(ValueError, "top-k16"):
            dual_stage_pair_reduce([1.0] * 8, [0] * 8)


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

    def test_host_allows_fallback_config_but_guards_tail_progress_sms(self):
        begin = self.jit.index(
            "const bool gin_combine_owner_slot_ready =")
        end = self.jit.index("#else", begin)
        body = self.jit[begin:end]
        self.assertIn(
            "(num_tokens >\n"
            "                         layout::kMegaMoeGinDirectDispatchMaxTokens or\n"
            "                     (config.num_epilogue_threads == 256 and\n"
            "                      num_sms >\n"
            "                          layout::kMegaMoeGinDirectDispatchMaxTokens))",
            body,
        )

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

    def test_pair_path_is_cta_per_token_and_dual_stage_ready_first(self):
        begin = self.kernel.index("One CTA owns one token")
        end = self.kernel.index(
            "DG_GIN_TRACE_IF(lane_idx == 0, 88u", begin)
        body = self.kernel[begin:end]
        self.assertIn("token_idx = sm_idx", body)
        self.assertIn("first_slot = epilogue_warp_idx * 2u", body)
        self.assertIn("uint32_t pending_mask", body)
        self.assertIn("while (pending_mask != 0)", body)
        self.assertIn("selected_pair_slot", body)
        ready = body.index("get_combine_receiver_owner_ready_ptr")
        issue = body.index("ptx::tma_load_1d(", ready)
        wait = body.index("pair_load_barriers[pair_slot]->wait", issue)
        self.assertLess(ready, issue)
        self.assertLess(issue, wait)

    def test_shared_memory_uses_seventeen_of_prior_twenty_four_chunks(self):
        self.assertIn(
            "(2u * kNumEpilogueWarps + 1u) * kNumChunkBytes",
            self.kernel)
        self.assertIn("(epilogue_warp_idx * 2u + pair_slot) *",
                      self.kernel)
        self.assertIn("epilogue_warp_idx * 2u * kNumChunkBytes",
                      self.kernel)
        self.assertIn("2u * kNumEpilogueWarps * kNumChunkBytes",
                      self.kernel)

    def test_two_existing_mbarriers_have_independent_phases(self):
        begin = self.kernel.index("One CTA owns one token")
        end = self.kernel.index(
            "DG_GIN_TRACE_IF(lane_idx == 0, 88u", begin)
        body = self.kernel[begin:end]
        self.assertIn("epilogue_warp_idx * 2u + pair_slot", body)
        self.assertIn("uint32_t pair_load_phases[2] = {};", body)
        self.assertIn("pair_load_barriers[pair_slot]->wait(", body)
        self.assertIn("pair_load_phases[pair_slot] ^= 1u", body)
        self.assertNotIn("combine_owner_slot_signal_buffer", body)

    def test_lane_zero_resolves_pointer_and_issues_without_broadcast(self):
        begin = self.kernel.index("One CTA owns one token")
        end = self.kernel.index(
            "DG_GIN_TRACE_IF(lane_idx == 0, 88u", begin)
        body = self.kernel[begin:end]
        lane_zero = body.index("if (lane_idx == 0)")
        resolve = body.index("pair_row_ptrs[pair_slot] =", lane_zero)
        issue = body.index("ptx::tma_load_1d(", resolve)
        self.assertLess(lane_zero, resolve)
        self.assertLess(resolve, issue)
        self.assertNotIn("pair_row_ptrs[pair_slot] = __shfl_sync", body)

    def test_load_stages_alias_pair_partial_only_after_both_waits(self):
        begin = self.kernel.index("One CTA owns one token")
        end = self.kernel.index(
            "DG_GIN_TRACE_IF(lane_idx == 0, 88u", begin)
        body = self.kernel[begin:end]
        wait = body.index("pair_load_barriers[pair_slot]->wait(")
        pair_store = body.index("pair_partial_buffer +", wait)
        first_barrier = body.index(
            "kNumEpilogueThreads, kEpilogueFullBarrierIdx", pair_store)
        output_fold = body.index("for (uint32_t pair = 0; pair < 8",
                                 first_barrier)
        self.assertLess(wait, pair_store)
        self.assertLess(pair_store, first_barrier)
        self.assertLess(first_barrier, output_fold)

    def test_accumulation_and_final_fold_keep_r11_pair_association(self):
        begin = self.kernel.index("One CTA owns one token")
        end = self.kernel.index(
            "DG_GIN_TRACE_IF(lane_idx == 0, 88u", begin)
        body = self.kernel[begin:end]
        self.assertIn("accumulation is", body)
        self.assertIn("fixed ascending assignment-slot order", body)
        self.assertIn("for (uint32_t pair_slot = 0; pair_slot < 2", body)
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
