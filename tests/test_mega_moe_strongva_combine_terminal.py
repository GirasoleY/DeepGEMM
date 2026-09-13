#!/usr/bin/env python3
"""CPU/source contracts for the default-off StrongVA combine terminal.

This is a protocol model, not a CUDA or fabric simulation. It distinguishes
sender-local completion from receiver visibility and keeps signal generations
persistent across eligible/fallback/eligible replay transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh"
COMM = ROOT / "deep_gemm/include/deep_gemm/comm/mega_moe_gin.cuh"
LAYOUT = ROOT / "deep_gemm/include/deep_gemm/layout/mega_moe.cuh"
JIT = ROOT / "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp"
PUBLIC = ROOT / "deep_gemm/mega/__init__.py"


class ProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class Span:
    peer: int
    begin: int
    count: int


class StrongVATerminalModel:
    """Small executable specification of one owner's peer chains."""

    def __init__(self, peers=(0, 1)):
        self.peers = tuple(peers)
        self.epoch = 0
        self.signal_cells = {peer: 0 for peer in self.peers}
        self.packet_counts = {peer: 0 for peer in self.peers}
        self.active = False

    def begin(self, tokens, expected, *, world_eligible=True,
              uniform_eligibility=True):
        if not uniform_eligibility:
            raise ProtocolError("terminal eligibility must be world-uniform")
        if set(expected) != set(self.peers) or any(
                type(count) is not int or count < 0
                for count in expected.values()):
            raise ProtocolError("one exact nonnegative count per peer required")
        self.active = world_eligible and tokens <= 48
        self.expected = dict(expected)
        self.headers = {}
        self.records = {peer: set() for peer in self.peers}
        self.spans = []
        self.terminal = {}
        self.settled = set()
        self.flushed = set()
        self.acquired = False
        self.last_acquire_target = None
        self.fallback_world_put = False
        if self.active:
            self.epoch += 1

    def queue_headers(self):
        if not self.active or self.headers:
            raise ProtocolError("headers queue once only on the eligible path")
        # Count publication is exact for every pair, including empty pairs.
        self.headers = dict(self.expected)
        self.packet_counts.update(self.headers)

    def submit(self, peer, begin, count):
        if not self.active or self.headers.get(peer) != self.expected[peer]:
            raise ProtocolError("exact header must precede every record span")
        if peer in self.terminal or count <= 0 or begin < 0 or \
                begin + count > self.expected[peer]:
            raise ProtocolError("invalid or post-terminal span")
        records = set(range(begin, begin + count))
        if records & self.records[peer]:
            raise ProtocolError("duplicate record publication")
        final = len(self.records[peer]) + count == self.expected[peer]
        self.records[peer].update(records)
        span = Span(peer, begin, count)
        self.spans.append(span)
        if final:
            if self.records[peer] != set(range(self.expected[peer])):
                raise ProtocolError("terminal cannot cover missing records")
            self.terminal[peer] = ("span", span)
        return final

    def close_empty_pairs(self):
        if not self.active or self.headers != self.expected:
            raise ProtocolError("all exact headers precede empty terminals")
        for peer, count in self.expected.items():
            if count == 0:
                if peer in self.terminal:
                    raise ProtocolError("duplicate empty terminal")
                self.terminal[peer] = ("signal", None)

    def flush(self):
        if set(self.terminal) != set(self.peers):
            raise ProtocolError("each pair needs exactly one terminal")
        # This is sender-local source-storage completion. It does not publish
        # receiver visibility and may finish before or after remote settlement.
        self.flushed = set(self.peers)

    def settle(self):
        if set(self.terminal) != set(self.peers) or self.settled:
            raise ProtocolError("each posted terminal settles remotely once")
        for peer in self.peers:
            self.signal_cells[peer] += 1
        self.settled = set(self.peers)

    def acquire(self, expected_epoch=None):
        expected_epoch = self.epoch if expected_epoch is None else expected_epoch
        # NCCL's rolling wait is >=, not equality. Passing the current local
        # epoch is therefore an essential caller invariant.
        if any(value < expected_epoch for value in self.signal_cells.values()):
            raise ProtocolError("receiver has not acquired every terminal")
        self.last_acquire_target = expected_epoch
        self.acquired = True

    def fallback(self):
        if self.active:
            raise ProtocolError("eligible terminal launch cannot use fallback")
        self.fallback_world_put = True


class StrongVAProtocolModelTests(unittest.TestCase):
    def test_terminal_tracks_actual_last_submission_not_address_order(self):
        model = StrongVATerminalModel((0,))
        model.begin(48, {0: 2})
        model.queue_headers()
        self.assertFalse(model.submit(0, 1, 1))
        self.assertTrue(model.submit(0, 0, 1))
        self.assertEqual(model.terminal[0], ("span", Span(0, 0, 1)))
        model.close_empty_pairs()
        # Receiver visibility may arrive before sender-local completion.
        with self.assertRaises(ProtocolError):
            model.acquire()
        model.settle()
        model.acquire()
        self.assertFalse(model.flushed)
        model.flush()
        self.assertEqual(model.flushed, {0})

    def test_empty_pair_publishes_zero_count_and_one_signal_terminal(self):
        model = StrongVATerminalModel((0, 1))
        model.begin(32, {0: 1, 1: 0})
        model.queue_headers()
        self.assertEqual(model.headers, {0: 1, 1: 0})
        self.assertTrue(model.submit(0, 0, 1))
        model.close_empty_pairs()
        self.assertEqual(model.terminal[1], ("signal", None))
        self.assertEqual(sum(kind == "signal" for kind, _ in model.terminal.values()), 1)
        model.settle()
        model.acquire()
        model.flush()

    def test_early_duplicate_and_count_mismatched_terminals_are_rejected(self):
        model = StrongVATerminalModel((0,))
        model.begin(48, {0: 2})
        with self.assertRaises(ProtocolError):
            model.submit(0, 0, 1)
        model.queue_headers()
        self.assertFalse(model.submit(0, 0, 1))
        with self.assertRaises(ProtocolError):
            model.submit(0, 0, 1)
        with self.assertRaises(ProtocolError):
            model.flush()
        self.assertTrue(model.submit(0, 1, 1))
        with self.assertRaises(ProtocolError):
            model.submit(0, 1, 1)

    def test_t48_t64_t48_pauses_epoch_and_waits_on_current_generation(self):
        model = StrongVATerminalModel((0,))
        model.begin(48, {0: 1})
        model.queue_headers(); model.submit(0, 0, 1)
        model.close_empty_pairs(); model.settle(); model.acquire(); model.flush()
        self.assertEqual((model.epoch, model.signal_cells[0]), (1, 1))

        model.begin(64, {0: 64})
        model.fallback()
        self.assertTrue(model.fallback_world_put)
        self.assertEqual((model.epoch, model.signal_cells[0]), (1, 1))
        # T64 does not consume the compact packet. Its stale count is harmless
        # and, crucially, no local clear can race a future early NIC header.
        self.assertEqual(model.packet_counts[0], 1)

        model.begin(48, {0: 0})
        model.queue_headers(); model.close_empty_pairs(); model.settle(); model.flush()
        self.assertEqual((model.epoch, model.signal_cells[0]), (2, 2))
        self.assertEqual(model.packet_counts[0], 0)
        # A stale rolling target would also pass at generation2, so correctness
        # is that the implementation supplies its current epoch, not that GIN
        # rejects an old one.
        model.acquire()
        self.assertEqual(model.last_acquire_target, 2)

    def test_rank_local_eligibility_is_rejected(self):
        model = StrongVATerminalModel((0,))
        with self.assertRaises(ProtocolError):
            model.begin(48, {0: 1}, uniform_eligibility=False)


class StrongVASourceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel = KERNEL.read_text()
        cls.comm = COMM.read_text()
        cls.layout = LAYOUT.read_text()
        cls.jit = JIT.read_text()
        cls.public = PUBLIC.read_text()

    def test_default_off_jit_keyed_and_collectively_validated(self):
        flag = "DG_MEGAMOE_GIN_STRONGVA_COMBINE_TERMINAL"
        self.assertIn(f"#define {flag} 0", self.kernel)
        self.assertIn(f"#define {flag} {{}}", self.jit)
        self.assertIn(f'get_env<std::string>(\n            "{flag}", "0")', self.jit)
        self.assertIn("gin_strongva_combine_terminal", self.jit)
        self.assertIn("kMegaMoeGinStrongVACombineTerminal", self.kernel)
        self.assertIn("kMegaMoeGinCombineOverlap", self.kernel)
        self.assertIn("strongva_combine_terminal", self.public)
        self.assertIn("activity_gate_opt", self.public)
        self.assertIn("configuration mismatch across ranks", self.public)

    def test_strongva_sender_and_receiver_share_world_only_eligibility(self):
        begin = self.kernel.index(
            "const auto use_gin_combine_overlap_this_launch = [&]()")
        end = self.kernel.index(
            "const auto use_gin_strongva_combine_terminal_this_launch", begin)
        overlap = self.kernel[begin:end]
        self.assertIn("const bool world_eligible =", overlap)
        self.assertIn(
            "if constexpr (kMegaMoeGinStrongVACombineTerminal)", overlap)
        strongva = overlap.split(
            "if constexpr (kMegaMoeGinStrongVACombineTerminal)", 1)[1]
        strongva = strongva[:strongva.index("// Local fit fallback")]
        self.assertIn("return world_eligible;", strongva)
        self.assertNotIn("combine_overlap_alias_fits", strongva)
        self.assertIn(
            "return world_eligible and\n"
            "                   buffer.gin_workspace.combine_overlap_alias_fits();",
            overlap,
        )

    def test_dedicated_registered_signal_layout_and_epoch(self):
        self.assertIn("kMegaMoeGinCombineTerminalSignalStride = 128", self.layout)
        self.assertIn("Buffer combine_terminal_signal_buffer;", self.layout)
        return_construct = self.layout.index(
            "bulk_combine_return_index_buffer = Buffer(\n"
            "                bulk_return_index_layout")
        signal_construct = self.layout.index(
            "combine_terminal_signal_buffer = Buffer(\n"
            "                Data(kMegaMoeGinCombineTerminalSignalStride)",
            return_construct,
        )
        self.assertLess(return_construct, signal_construct)
        self.assertIn(
            "num_remote_peers, 1",
            self.layout[signal_construct:signal_construct + 400],
        )
        self.assertIn("get_combine_terminal_signal_ptr", self.layout)
        self.assertIn("owner_in_lsa < num_ranks / 2u", self.layout)
        self.assertIn("get_gin_combine_terminal_epoch_ptr", self.layout)
        epoch = self.layout.split("get_gin_combine_terminal_epoch_ptr", 1)[1]
        self.assertIn("math::advance_ptr<uint64_t>(base, 64u)", epoch[:180])
        self.assertIn("combine_terminal_signal_buffer.get_end_ptr()", self.layout)

    def test_header_terminal_and_empty_signal_helpers_are_completion_free(self):
        header_begin = self.comm.index(
            "void mega_moe_gin_put_bulk_combine_header_async(")
        terminal_begin = self.comm.index(
            "void mega_moe_gin_put_bulk_combine_terminal_span(")
        empty_begin = self.comm.index(
            "void mega_moe_gin_signal_bulk_combine_terminal(")
        wait_begin = self.comm.index(
            "void mega_moe_gin_wait_bulk_combine_terminal(")
        header = self.comm[header_begin:terminal_begin]
        terminal = self.comm[terminal_begin:empty_begin]
        empty = self.comm[empty_begin:wait_begin]
        self.assertIn("remote_packet, 16u", header)
        self.assertNotIn("flushAsync", header)
        self.assertNotIn("gin.wait(", header)
        self.assertIn("ncclGin_StrongVASignalInc", terminal)
        self.assertIn("cuda::thread_scope_system", terminal)
        self.assertNotIn("flushAsync", terminal)
        self.assertIn("gin.signal(", empty)
        self.assertIn("ncclGin_StrongVASignalInc", empty)
        self.assertNotIn("gin.put(", empty)

    def test_actual_final_span_and_empty_pair_do_not_wait_in_drainer(self):
        begin = self.kernel.index(
            "if (use_gin_combine_overlap and sm_idx == 0 and warp_idx == 0)")
        end = self.kernel.index(
            "if constexpr (kUseGin and kMegaMoeGinDispatchOverlap)", begin)
        drainer = self.kernel[begin:end]
        expected = drainer.index("uint32_t expected_records = 0;")
        header = drainer.index("mega_moe_gin_put_bulk_combine_header_async(")
        loop = drainer.index("while (__any_sync")
        final = drainer.index(
            "sent_records + batch_records == expected_records")
        terminal = drainer.index(
            "mega_moe_gin_put_bulk_combine_terminal_span(")
        sent_update = drainer.index("sent_records += batch_records;")
        pending_update = drainer.index("pending_first = remaining_first;")
        empty = drainer.index("if (expected_records == 0)")
        self.assertLess(expected, header)
        self.assertLess(header, loop)
        self.assertLess(final, terminal)
        self.assertLess(terminal, sent_update)
        self.assertLess(sent_update, pending_update)
        self.assertLess(terminal, empty)
        self.assertNotIn("mega_moe_gin_flush_data_peer_async(", drainer)
        self.assertNotIn("mega_moe_gin_wait_data_peer(", drainer)
        self.assertNotIn("DG_GIN_TRACE_IF(true, 80u + lane_idx)", drainer)
        self.assertIn("get_combine_terminal_signal_ptr(owner_lane)", drainer)
        self.assertIn("DG_DEVICE_ASSERT(sent_records == expected_records)", drainer)

    def test_sender_completion_precedes_fallback_or_deferred_rendezvous(self):
        cleanup_begin = self.kernel.index(
            "// Finish workspace cleanup.")
        cleanup_end = self.kernel.index(
            "} else if (warp_idx == kNumDispatchWarps)", cleanup_begin)
        cleanup = self.kernel[cleanup_begin:cleanup_end]
        grid = cleanup.index("comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(")
        strongva = cleanup.index(
            "if constexpr (kMegaMoeGinStrongVACombineTerminal)", grid)
        flush = cleanup.index("mega_moe_gin_flush_data_peer_async(", strongva)
        wait = cleanup.index("mega_moe_gin_wait_data_peer(", flush)
        marker = cleanup.index("DG_GIN_TRACE_IF(true, 80u + lane_idx)", wait)
        warp_join = cleanup.index("__syncwarp();", marker)
        fallback = cleanup.index(
            "if (not use_gin_strongva_combine_terminal)", warp_join)
        world = cleanup.index("comm::mega_moe_gin_world_barrier(", fallback)
        deferred = cleanup.index(
            "if (use_gin_strongva_combine_terminal)", world)
        lsa = cleanup.index("comm::nvlink_lsa_barrier<", deferred)
        self.assertLess(grid, strongva)
        self.assertLess(strongva, flush)
        self.assertLess(flush, wait)
        self.assertLess(wait, marker)
        self.assertLess(marker, warp_join)
        self.assertLess(warp_join, fallback)
        self.assertLess(fallback, world)
        self.assertLess(world, deferred)
        self.assertLess(deferred, lsa)
        self.assertIn("/*context_stripe=*/ 0u", cleanup[flush:wait])
        self.assertIn("if (use_gin_strongva_combine_terminal)",
                      cleanup[strongva:flush])
        self.assertEqual(
            self.kernel.count("DG_GIN_TRACE_IF(true, 80u + lane_idx)"), 1)

    def test_early_header_cannot_race_the_accepted_startup_count_clear(self):
        begin = self.kernel.index(
            "if constexpr (kUseGin and kMegaMoeGinBulkCombine)")
        end = self.kernel.index(
            "if constexpr (kUseGin and kMegaMoeGinLocalAblationStage < 1)",
            begin,
        )
        startup = self.kernel[begin:end]
        self.assertIn("if (use_gin_bulk_combine and sm_idx == 0 and", startup)
        self.assertIn(
            "not use_gin_strongva_combine_terminal and warp_idx == 0",
            startup,
        )
        self.assertIn("get_bulk_combine_packet_count_ptr(", startup)
        self.assertIn("/*send=*/ false, lane_idx) = 0", startup)

    def test_eligible_visibility_replaces_only_late_header_and_world_put(self):
        late = self.kernel[self.kernel.index("// Deallocate tensor memory"):]
        self.assertIn(
            "epilogue_warp_idx == 0 and\n                not use_gin_strongva_combine_terminal",
            late,
        )
        lsa = late.index("if constexpr (kMegaMoeGinStrongVACombineTerminal)")
        terminal_wait = late.index(
            "mega_moe_gin_wait_bulk_combine_terminal(", lsa)
        grid2 = late.index(
            "comm::grid_sync<kNumSMs, kEpilogueGridSyncIndex>", terminal_wait)
        proxy = late.index('asm volatile("fence.proxy.async.global;', grid2)
        self.assertLess(lsa, terminal_wait)
        self.assertLess(terminal_wait, grid2)
        self.assertLess(grid2, proxy)
        self.assertIn("mega_moe_gin_put_bulk_combine_header(", late[:lsa])
        self.assertIn("mega_moe_gin_put_bulk_combine_packet(", late[:lsa])
        self.assertIn("mega_moe_gin_world_barrier_all_contexts(", late[:lsa])
        self.assertIn("kGinCleanupBarrierIdx", self.kernel)
        self.assertIn("if (not use_gin_strongva_combine_terminal)", late)

    def test_signal_cells_are_persistent_and_never_reset_by_kernel(self):
        # The accessor occurs only as a remote terminal target or local wait;
        # the compile-time-exclusive owner-wave sender and owner-ready progress
        # warp each have their own target call.
        # No assignment through it appears in startup, fallback, or cleanup.
        occurrences = self.kernel.count("get_combine_terminal_signal_ptr(")
        self.assertEqual(occurrences, 5)
        self.assertNotIn("*buffer.gin_workspace\n"
                         "                     .get_combine_terminal_signal_ptr",
                         self.kernel)
        cleanup_begin = self.kernel.index(
            "// Finish workspace cleanup.")
        cleanup_end = self.kernel.index(
            "} else if (warp_idx == kNumDispatchWarps)", cleanup_begin)
        cleanup = self.kernel[cleanup_begin:cleanup_end]
        self.assertNotIn("get_combine_terminal_signal_ptr", cleanup)


if __name__ == "__main__":
    unittest.main()
