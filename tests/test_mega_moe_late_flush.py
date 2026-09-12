"""R7 CPU completion-order checks, not NIC scheduling or GPU validation."""

from dataclasses import replace
from pathlib import Path
import re
import unittest

from mega_moe_gin_late_flush_protocol import LateFlushGeneration, ProtocolError, SharedSQ


def complete_producers(model):
    for expert, blocks in enumerate(model.blocks):
        for block in range(blocks):
            for fragment in range(model.n_fragments):
                model.release(model.generation, expert, block, fragment)


def retire_inputs(model):
    for peer in model.peers:
        model.post_input(model.generation, peer, "empty_or_weight_terminal")
        queue = model.queue(peer, 1 + peer % 8)
        snapshot = queue.flush()
        queue.complete()
        model.retire_input(model.generation, peer, snapshot, inbound_terminal_acquired=True)


def headers_complete(model):
    model.join_dispatch_and_epilogue(model.generation)
    model.publish_headers(model.generation)
    snapshots = {}
    for peer in model.headers:
        queue = model.queue(peer)
        snapshots[peer] = queue.flush()
        queue.complete()
        model.finish_header_flush(model.generation, peer, snapshots[peer])
    return snapshots


def target_and_consumers_complete(model):
    for span in model.spans:
        model.settle(model.generation, span)
    model.headers_settled = True
    model.acquire_world_put(model.generation)
    for peer in model.peers:
        model.consume(model.generation, peer)
    model.retire(model.generation)


def braced_block(source, start):
    begin = source.index("{", start)
    end, depth = begin + 1, 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


class LateFlushModelTests(unittest.TestCase):
    def generation(self, counts=((2, 0, 3), (0, 1, 2)), **kwargs):
        model = LateFlushGeneration(range(len(counts)), n_fragments=2, **kwargs)
        model.begin(0, counts)
        retire_inputs(model)
        complete_producers(model)
        return model

    def test_late_header_covers_earlier_payloads_without_early_completion(self):
        model = self.generation(cap=1)
        model.drain_ready(0)
        self.assertFalse(model.flushed)
        self.assertTrue(any(q.completed < q.submitted for q in model.queues.values()))
        with self.assertRaises(ProtocolError):
            model.flush_payloads(0)
        with self.assertRaises(ProtocolError):
            model.check_send_source_reusable(0)
        model.join_dispatch_and_epilogue(0)  # Payload CQ completion is not a prerequisite.
        model.publish_headers(0)
        for peer in model.headers:
            queue = model.queue(peer)
            snapshot = queue.flush()
            with self.assertRaises(ProtocolError):
                model.finish_header_flush(0, peer, snapshot)
            queue.complete()
            model.finish_header_flush(0, peer, snapshot)
            self.assertTrue(all(ticket.index < model.header_tickets[peer].index
                                for span, ticket in model.span_tickets.items() if span.peer == peer))
        model.check_send_source_reusable(0)
        target_and_consumers_complete(model)

    def test_early_or_wrong_queue_snapshot_cannot_retire_payloads(self):
        model = self.generation()
        model.drain_ready(0)
        early = model.queue(0).flush()
        model.join_dispatch_and_epilogue(0)
        model.publish_headers(0)
        queue = model.queue(0)
        queue.complete()
        with self.assertRaises(ProtocolError):
            model.finish_header_flush(0, 0, early)
        snapshot = queue.flush()
        for wrong in (replace(snapshot, context=2), replace(snapshot, peer=1)):
            with self.assertRaises(ProtocolError):
                model.finish_header_flush(0, 0, wrong)
        model.finish_header_flush(0, 0, snapshot)

    def test_default_combine_can_submit_constructed_input_aggregate_prefix(self):
        # Backend stress case: physical peer8's input stripe0 shares context1
        # with combine. R5 startup itself does not create this interleaving.
        counts = ((0, 0),) * 8 + ((1, 1),)
        model = LateFlushGeneration((8,), n_fragments=1, cap=1)
        model.begin(0, counts)
        model.post_input(0, 8, "control")
        model.post_input(0, 8, "activation", aggregate=True)
        complete_producers(model)
        first = model.plan_wave(0)[8]
        model.issue_batch(0, first)
        queue = model.queue(8)
        self.assertEqual(queue.reserved[:3], ["input:control", "input:activation", "combine_payload"])
        self.assertEqual(queue.submitted, 3)
        model.post_input(0, 8, "scale", aggregate=True)
        model.post_input(0, 8, "weights_terminal")
        input_snapshot = queue.flush()
        queue.complete()
        model.retire_input(0, 8, input_snapshot, inbound_terminal_acquired=True)
        model.drain_ready(0)  # This later payload is NOT covered by the earlier input flush.
        self.assertLess(input_snapshot.upper, len(queue.reserved))
        headers_complete(model)
        self.assertGreater(model.header_tickets[8].index, input_snapshot.upper)
        target_and_consumers_complete(model)

    def test_default_does_not_ring_through_unconstructed_reservation_hole(self):
        queue = SharedSQ(1, 8)
        earlier = queue.reserve("input_activation")
        later = queue.reserve("combine_payload")
        queue.construct(later)
        with self.assertRaises(ProtocolError):
            queue.finish_post(later)
        self.assertEqual(queue.submitted, 0)
        queue.construct(earlier)
        queue.finish_post(earlier, aggregate=True)
        queue.finish_post(later)
        queue.complete()
        queue.wait(queue.flush())

    def test_flush_snapshot_does_not_submit_open_aggregate_chain(self):
        queue = SharedSQ(1, 8)
        queue.post("activation", aggregate=True)
        queue.post("scale", aggregate=True)
        snapshot = queue.flush()
        queue.complete()
        with self.assertRaises(ProtocolError):
            queue.wait(snapshot)
        queue.post("weights_terminal")
        queue.complete()
        queue.wait(snapshot)

    def test_empty_peers_still_require_all_input_terminals_and_retirement(self):
        model = LateFlushGeneration((0, 1), n_fragments=1)
        model.begin(0, ((0, 0), (0, 0)))
        with self.assertRaises(ProtocolError):
            model.join_dispatch_and_epilogue(0)
        model.post_input(0, 0, "empty_terminal")
        queue = model.queue(0)
        queue.complete()
        with self.assertRaises(ProtocolError):
            model.retire_input(0, 0, queue.flush(), inbound_terminal_acquired=False)
        model.retire_input(0, 0, queue.flush(), inbound_terminal_acquired=True)
        with self.assertRaises(ProtocolError):
            model.join_dispatch_and_epilogue(0)
        model.post_input(0, 1, "empty_terminal")
        queue = model.queue(1, 2)
        queue.complete()
        model.retire_input(0, 1, queue.flush(), inbound_terminal_acquired=True)
        headers_complete(model)
        self.assertFalse(model.spans)
        self.assertFalse(model.headers)
        target_and_consumers_complete(model)

    def test_handoff_requires_record_coverage_and_all_producers(self):
        model = LateFlushGeneration((1,), n_fragments=1)
        model.begin(0, ((33, 0), (0, 1)))  # Expert0 is local-only and still needs both producers.
        retire_inputs(model)
        model.release(0, 1, 0, 0)
        with self.assertRaises(ProtocolError):
            model.join_dispatch_and_epilogue(0)  # Remote record has not been queued.
        model.drain_ready(0)
        with self.assertRaises(ProtocolError):
            model.join_dispatch_and_epilogue(0)  # Local-only producer still running.
        model.release(0, 0, 0, 0)
        model.release(0, 0, 1, 0)
        headers_complete(model)
        target_and_consumers_complete(model)

    def test_local_header_flush_is_not_remote_visibility_or_receive_retirement(self):
        model = self.generation()
        model.drain_ready(0)
        headers_complete(model)
        model.check_send_source_reusable(0)
        with self.assertRaises(ProtocolError):
            model.acquire_world_put(0)
        with self.assertRaises(ProtocolError):
            model.consume(0, 0)
        with self.assertRaises(ProtocolError):
            model.begin(1, model.counts)
        for span in model.spans:
            model.settle(0, span)
        model.headers_settled = True
        model.acquire_world_put(0)
        model.consume(0, 0)
        with self.assertRaises(ProtocolError):
            model.retire(0)
        model.consume(0, 1)
        model.retire(0)

    def test_queue_indices_and_stale_snapshot_across_replay(self):
        model = self.generation()
        model.drain_ready(0)
        snapshots = headers_complete(model)
        last_index = model.header_tickets[0].index
        target_and_consumers_complete(model)
        model.begin(1, ((1, 3, 0), (2, 0, 1)))
        retire_inputs(model)
        complete_producers(model)
        model.drain_ready(1)
        model.join_dispatch_and_epilogue(1)
        model.publish_headers(1)
        self.assertGreater(model.header_tickets[0].index, last_index)
        model.queue(0).complete()
        with self.assertRaises(ProtocolError):
            model.finish_header_flush(1, 0, snapshots[0])
        with self.assertRaises(ProtocolError):
            model.finish_header_flush(0, 0, model.queue(0).flush())

    def test_off_fit_fallback_and_large_token_fallback_keep_old_completion(self):
        model = LateFlushGeneration((0,), n_fragments=1)
        sequence = ({"enabled": False}, {}, {"eligible": False},
                    {"scratch_fits": False}, {}, {"enabled": False})
        for generation, flags in enumerate(sequence):
            counts = ((1024 if flags.get("eligible") is False else 16,),)
            model.begin(generation, counts, **flags)
            complete_producers(model)
            if model.early:
                retire_inputs(model)
                model.drain_ready(generation)
                headers_complete(model)
            else:
                model.publish_fallback(generation)
                with self.assertRaises(ProtocolError):
                    model.publish_headers(generation)
                model.flush_payloads(generation)
                model.publish_headers(generation)
            target_and_consumers_complete(model)


class LateFlushSourceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.kernel = (root / "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh").read_text()
        cls.helper = (root / "deep_gemm/include/deep_gemm/comm/mega_moe_gin.cuh").read_text()

    def test_drainer_retains_count_audit_and_candidate_local_completion(self):
        source = self.kernel
        body = braced_block(source, source.index(
            "if (use_gin_combine_overlap and sm_idx == 0 and warp_idx == 0)",
            source.index("DG_GIN_TRACE_IF(lane_idx == 0, 48u + warp_idx);")))
        self.assertEqual(body.count("comm::mega_moe_gin_put_bulk_combine_span("), 2)
        self.assertIn("mega_moe_gin_put_bulk_combine_terminal_span", body)
        self.assertIn("/*context_stripe=*/ 0u", body)
        self.assertIn("DG_GIN_TRACE_IF(lane_idx == 0, 100)", body)
        self.assertIn("DG_DEVICE_ASSERT(sent_records == expected_records)", body)
        self.assertIn("flush_data_peer_async", body)
        self.assertIn("wait_data_peer", body)
        self.assertIn("ncclGinRequest_t", body)
        self.assertNotRegex(source, r"DG_GIN_TRACE(?:_IF)?\([^;]*\b101\b")
        self.assertLess(body.index("DG_DEVICE_ASSERT(sent_records == expected_records)"),
                        body.rindex("__syncwarp();"))

    def test_late_header_same_context_peer_flush_stays_after_handoff_grid(self):
        source = self.kernel
        epilogue = source[source.index("// Publish every same-LSA mapped store before handing phase 2") :]
        header = epilogue.index("comm::mega_moe_gin_put_bulk_combine_header(")
        prefix = epilogue[:header]
        handoff = prefix.index("ptx::sync_unaligned(")
        grid = prefix.index("comm::grid_sync<kNumSMs, kEpilogueGridSyncIndex>(")
        self.assertLess(handoff, grid)
        self.assertIn("if (route_count > 0)", prefix)
        call = epilogue[header:epilogue.index(");", header) + 2]
        self.assertIn("remote_source", call)
        self.assertIn("/*context_stripe=*/ 0u", call)
        helper = braced_block(self.helper, self.helper.index("void mega_moe_gin_put_bulk_combine_header("))
        self.assertIn("remote_packet, 16u", helper)
        put = helper.index("mega_moe_gin_put_data(")
        flush = helper.index("mega_moe_gin_flush_data_peer_async(")
        wait = helper.index("mega_moe_gin_wait_data_peer(")
        self.assertLess(put, flush)
        self.assertLess(flush, wait)
        self.assertIn("context_stripe", helper)
        self.assertIn("ncclGinFenceLevel::Put", epilogue[header:])

    def test_input_retirement_and_nonaggregate_system_publication_unchanged(self):
        source = self.kernel
        late = braced_block(source, source.index(
            "if (use_gin_direct_dispatch and sm_idx == 1 and warp_idx == 0 and",
            source.index("// All active peer chains were posted before the count rendezvous.")))
        self.assertIn("mega_moe_gin_wait_direct_dispatch", late)
        self.assertIn("get_direct_dispatch_payload_ready_ptr(lane_idx)", late)
        self.assertIn("mega_moe_gin_flush_data_peer_async", late)
        self.assertIn("mega_moe_gin_wait_data_peer", late)
        span = braced_block(self.helper, self.helper.index("void mega_moe_gin_put_bulk_combine_span("))
        self.assertIn("ncclGinOptFlagsDefault", span)
        self.assertIn("cuda::thread_scope_device", span)
        self.assertIn("cuda::thread_scope_system", span)
        self.assertNotIn("AggregateRequests", span)
        self.assertIn('fence.proxy.async.global;', source)
        self.assertIn("Release dispatch cleanup here only for the scatter/fallback", source)


if __name__ == "__main__":
    unittest.main()
