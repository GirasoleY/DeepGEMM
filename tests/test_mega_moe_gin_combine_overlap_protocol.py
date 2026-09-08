"""CPU logical-protocol acceptance; GPU helper and numerical gates separate."""

import random
import unittest

from mega_moe_gin_combine_overlap_protocol import CombineGeneration, ProtocolError, logical_blocks
from mega_moe_gin_overlap_protocol import fixed_assignment_counts


def complete(model, block, generation):
    for producer in range(model.n_producers):
        model.producer_release(block, producer, generation)


def finish(model, generation):
    for peer in model.peers:
        model.flush_peer(peer, generation)
    model.publish_headers(generation)
    for span in model.spans:
        model.settle_span(span)
    model.settle_headers(generation)
    model.acquire_final_put_barrier(generation)
    for peer in model.peers:
        model.consume_peer(peer, generation)
    model.retire()


class CompletedBlockProtocol(unittest.TestCase):
    def test_actual_round_robin_membership_not_source_major(self):
        blocks = logical_blocks(((32,), (32,)), 32, (0, 1))
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0], tuple((source, index) for index in range(16) for source in (0, 1)))
        self.assertEqual(blocks[1][0], (0, 16))

    def test_partial_and_local_rows_do_not_create_padding_records(self):
        blocks = logical_blocks(((2, 1), (3, 0)), 4, (1,))
        self.assertEqual(blocks, ((None, (1, 0), None, (1, 1)), ((1, 2),), (None,)))
        model = CombineGeneration((1, 2))
        model.begin(0, blocks)
        self.assertEqual(model.expected, {1: 3, 2: 0})

    def test_every_n_producer_and_every_issuing_lane_must_acquire(self):
        model = CombineGeneration((0,), n_producers=28)
        model.begin(0, (((0, 0),),))
        for producer in range(27):
            model.producer_release(0, producer, 0)
        self.assertEqual(model.scan(0), ())
        with self.assertRaises(ProtocolError):
            model.producer_release(0, 0, 0)
        model.producer_release(0, 27, 0)
        self.assertEqual(model.scan(0, acquiring_lanes=range(31)), ())
        self.assertEqual(len(model.scan(0)), 1)
        self.assertEqual(model.scan(0), ())
        finish(model, 0)

    def test_later_ready_blocks_bypass_earlier_unfinished_block_without_hole(self):
        model = CombineGeneration((0,))
        model.begin(0, (((0, 0),), ((0, 1),), ((0, 2),)))
        complete(model, 0, 0)
        complete(model, 2, 0)
        spans = model.scan(0)
        self.assertEqual([(s.begin, s.count) for s in spans], [(0, 1), (2, 1)])
        self.assertNotIn((0, 1), model.issued)
        complete(model, 1, 0)
        self.assertEqual([(s.begin, s.count) for s in model.scan(0)], [(1, 1)])
        finish(model, 0)

    def test_exact_adjacent_per_peer_spans_coalesce_and_flush_at_end_of_pass(self):
        blocks = (((0, 0), (1, 0)), ((0, 1),), ((0, 2), (1, 1)))
        model = CombineGeneration((0, 1))
        model.begin(0, blocks)
        for block in range(3):
            complete(model, block, 0)
        self.assertEqual([(s.peer, s.begin, s.count) for s in model.scan(0)], [(0, 0, 3), (1, 0, 2)])
        self.assertEqual(len(model.issued), 5)
        finish(model, 0)

    def test_random_completion_and_scan_order_sends_every_record_exactly_once(self):
        blocks = logical_blocks(((7, 1), (2, 6), (4, 3)), 4, (0, 1, 2))
        for seed in range(12):
            rng = random.Random(seed)
            model = CombineGeneration((0, 1, 2), n_producers=3)
            model.begin(0, blocks)
            events = [(b, p) for b in range(len(blocks)) for p in range(3)]
            rng.shuffle(events)
            for block, producer in events:
                model.producer_release(block, producer, 0)
                order = list(range(len(blocks)))
                rng.shuffle(order)
                model.scan(0, order=order)
            expected = {(p, i) for p, n in model.expected.items() for i in range(n)}
            self.assertEqual(model.issued, expected)
            self.assertEqual(sum(span.count for span in model.spans), len(expected))
            finish(model, 0)

    def test_late_headers_flush_is_not_remote_acquire_and_reuse_waits_for_consumers(self):
        model = CombineGeneration((0, 1))
        model.begin(0, (((0, 0),),))
        with self.assertRaises(ProtocolError):
            model.publish_headers(0)
        complete(model, 0, 0)
        model.scan(0)
        for peer in model.peers:
            model.flush_peer(peer, 0)
        model.publish_headers(0)
        self.assertEqual(model.headers[1], (0, 0))
        model.settle_headers(0)
        with self.assertRaises(ProtocolError):
            model.acquire_final_put_barrier(0)
        with self.assertRaises(ProtocolError):
            model.consume_peer(0, 0)
        with self.assertRaises(ProtocolError):
            model.begin(1, ())
        model.settle_span(model.spans[0])
        model.acquire_final_put_barrier(0)
        with self.assertRaises(ProtocolError):
            model.retire()
        for peer in model.peers:
            model.consume_peer(peer, 0)
        model.retire()

    def test_mode_zero_one_zero_fit_failure_and_t64_fallback_reset_generation(self):
        model = CombineGeneration((0,), n_producers=2)
        selections = ((False, True, True), (True, True, True), (True, False, True),
                      (True, True, False), (True, True, True), (False, True, True))
        old_span = None
        for generation, (enabled, eligible, fits) in enumerate(selections):
            model.begin(generation, (((0, 0),), ((0, 1),)), enabled=enabled,
                        eligible=eligible, storage_fits=fits)
            self.assertFalse(model.claimed or model.issued or model.headers)
            if old_span:
                with self.assertRaises(ProtocolError):
                    model.settle_span(old_span)
            complete(model, 1, generation)
            self.assertEqual(bool(model.scan(generation)), enabled and eligible and fits)
            complete(model, 0, generation)
            model.scan(generation)
            old_span = model.spans[0]
            finish(model, generation)

    def test_balanced_hot_owner_empty_asymmetric_balanced_fixed_shapes(self):
        for tokens in (32, 40, 48):
            model = CombineGeneration(tuple(range(8)), n_producers=2)
            for generation, mode in enumerate(("balanced", "single_hot_owner", "empty", "asymmetric", "balanced")):
                counts = fixed_assignment_counts(mode, tokens)[8]
                blocks = logical_blocks(counts, 16, model.peers)
                model.begin(generation, blocks)
                for block in reversed(range(len(blocks))):
                    complete(model, block, generation)
                model.scan(generation)
                self.assertEqual(model.expected, {p: sum(counts[p]) for p in model.peers})
                self.assertEqual(sum(s.count for s in model.spans), sum(model.expected.values()))
                finish(model, generation)

    def test_bad_metadata_and_unfair_scan_rejected(self):
        model = CombineGeneration((0,))
        for blocks in ((((0, 1),),), (((0, 0),), ((0, 0),)), ((),)):
            with self.assertRaises(ProtocolError):
                model.begin(0, blocks)
        model.begin(0, (((0, 0),), ((0, 1),)))
        for order in ((0,), (0, 0), (1, 2)):
            with self.assertRaises(ProtocolError):
                model.scan(0, order=order)


if __name__ == "__main__":
    unittest.main()
