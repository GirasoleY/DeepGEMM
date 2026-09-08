"""CPU ordering-contract tests. Passing does not validate a GIN device kernel."""

import random
import unittest

from mega_moe_gin_overlap_protocol import (
    DispatchGeneration, ProtocolError, TerminalEpochs, fixed_assignment_counts,
)


def controls(model, values):
    for source, counts in enumerate(values):
        model.publish_control(source, counts, model.generation)
        model.acquire_control(source, model.generation)


def payload(model, source):
    epoch = model.generation
    model.issue_payload(source, epoch)
    model.finish_local_send(source, epoch)
    model.settle_payload(source, epoch)
    model.signal_payload(source, epoch)
    model.acquire_payload(source, epoch)


def consume_ready(model):
    for expert in range(model.experts):
        for tile in range(len(model.tiles(expert))):
            if (expert, tile) not in model.issued and model.tile_ready(expert, tile):
                model.issue_tile(expert, tile, 0)
                model.complete_tile(expert, tile)
                model.release_scratch(0)


class OverlapProtocolContracts(unittest.TestCase):
    def test_control_must_be_issued_before_payload(self):
        model = DispatchGeneration(sources=1, experts=1)
        with self.assertRaisesRegex(ProtocolError, "before control"):
            model.issue_payload(0, 0)

    def test_control_acquire_never_implies_payload_ready(self):
        model = DispatchGeneration(sources=1, experts=1)
        controls(model, [(1,)])
        self.assertEqual(len(model.tiles(0)), 1)
        self.assertFalse(model.tile_ready(0, 0))
        with self.assertRaises(ProtocolError):
            model.issue_tile(0, 0, 0)

    def test_last_exact_count_required_even_when_later_source_will_be_empty(self):
        model = DispatchGeneration(sources=2, experts=1)
        model.publish_control(0, (32,), 0)
        model.acquire_control(0, 0)
        payload(model, 0)
        with self.assertRaisesRegex(ProtocolError, "every source"):
            model.tiles(0)
        model.publish_control(1, (0,), 0)
        model.acquire_control(1, 0)
        self.assertTrue(model.tile_ready(0, 0))
        self.assertEqual(len(model.tiles(0)[0]), 32)

    def test_complete_tile_must_include_every_contributing_source(self):
        model = DispatchGeneration(sources=3, experts=1)
        controls(model, [(16,), (17,), (0,)])
        payload(model, 0)
        self.assertEqual(tuple(map(len, model.tiles(0))), (32, 1))
        self.assertFalse(model.tile_ready(0, 0))
        self.assertFalse(model.tile_ready(0, 1))
        payload(model, 1)
        self.assertTrue(model.tile_ready(0, 0))
        self.assertTrue(model.tile_ready(0, 1))
        # Zero-count source2 does not block useful compute, only retirement.
        consume_ready(model)
        with self.assertRaises(ProtocolError):
            model.retire()
        payload(model, 2)
        model.retire()

    def test_exact_full_tile_can_start_while_other_tiles_wait(self):
        model = DispatchGeneration(sources=2, experts=2)
        controls(model, [(32, 0), (0, 32)])
        payload(model, 0)
        self.assertTrue(model.tile_ready(0, 0))
        self.assertFalse(model.tile_ready(1, 0))
        model.issue_tile(0, 0, 0)
        self.assertNotIn(1, model.payload_acquired)

    def test_actual_round_robin_tile_does_not_manufacture_source_major_readiness(self):
        model = DispatchGeneration(sources=2, experts=1)
        controls(model, [(32,), (32,)])
        payload(model, 0)
        self.assertEqual(model.tiles(0)[0][:4], ((0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1)))
        self.assertFalse(model.tile_ready(0, 0))
        self.assertFalse(model.tile_ready(0, 1))

    def test_flush_allows_sender_reuse_but_not_receiver_loads(self):
        model = DispatchGeneration(sources=1, experts=1)
        controls(model, [(1,)])
        model.issue_payload(0, 0)
        self.assertFalse(model.sender_can_reuse(0))
        model.finish_local_send(0, 0)
        self.assertTrue(model.sender_can_reuse(0))
        self.assertFalse(model.tile_ready(0, 0))
        with self.assertRaises(ProtocolError):
            model.signal_payload(0, 0)
        model.settle_payload(0, 0)
        model.signal_payload(0, 0)
        self.assertFalse(model.tile_ready(0, 0))
        model.acquire_payload(0, 0)
        self.assertTrue(model.tile_ready(0, 0))

    def test_payload_can_arrive_before_receiver_acquires_control(self):
        model = DispatchGeneration(sources=1, experts=1)
        model.publish_control(0, (1,), 0)
        payload(model, 0)
        with self.assertRaises(ProtocolError):
            model.tiles(0)
        model.acquire_control(0, 0)
        self.assertTrue(model.tile_ready(0, 0))

    def test_payload_terminal_covers_activation_scale_and_weight_not_just_data(self):
        for last in ("activation", "scale", "weight"):
            model = DispatchGeneration(sources=1, experts=1)
            controls(model, [(1,)])
            model.issue_payload(0, 0)
            first = {"activation", "scale", "weight"} - {last}
            model.settle_payload(0, 0, first)
            with self.assertRaises(ProtocolError):
                model.signal_payload(0, 0)
            model.settle_payload(0, 0, {last})
            model.signal_payload(0, 0)
            self.assertFalse(model.tile_ready(0, 0))
            model.acquire_payload(0, 0)
            self.assertTrue(model.tile_ready(0, 0))

    def test_no_scratch_reuse_at_issue_or_before_consumer_completion(self):
        model = DispatchGeneration(sources=1, experts=1, scratch_slots=1)
        controls(model, [(64,)])
        payload(model, 0)
        model.issue_tile(0, 0, 0)
        with self.assertRaises(ProtocolError):
            model.issue_tile(0, 1, 0)
        with self.assertRaises(ProtocolError):
            model.release_scratch(0)
        with self.assertRaises(ProtocolError):
            model.begin(1)
        model.complete_tile(0, 0)
        with self.assertRaises(ProtocolError):
            model.issue_tile(0, 1, 0)
        model.release_scratch(0)
        model.issue_tile(0, 1, 0)
        model.complete_tile(0, 1)
        model.release_scratch(0)
        model.retire()
        model.begin(1)
        self.assertFalse(model.sender_can_reuse(0))

    def test_equal_total_wrong_expert_or_old_payload_does_not_pass(self):
        model = DispatchGeneration(sources=1, experts=2)
        controls(model, [(1, 1)])
        for bad in (((0, 0, 0, 0), (0, 0, 0, 1)),
                    ((-1, 0, 0, 0), (-1, 0, 1, 0))):
            with self.assertRaises(ProtocolError):
                model.issue_payload(0, 0, bad)

    def test_empty_generation_still_drains_both_terminals(self):
        model = DispatchGeneration(sources=2, experts=1)
        controls(model, [(0,), (0,)])
        self.assertEqual(model.tiles(0), ())
        payload(model, 0)
        with self.assertRaises(ProtocolError):
            model.retire()
        payload(model, 1)
        model.retire()
        model.begin(1)
        with self.assertRaises(ProtocolError):
            model.acquire_payload(0, 0)
        with self.assertRaises(ProtocolError):
            model.acquire_payload(0, 1)

    def test_payload_counter_advances_only_on_split_invocations(self):
        counters = TerminalEpochs()
        self.assertEqual([counters.advance(mode) for mode in (False, True, False, True)],
                         [(1, None), (2, 1), (3, None), (4, 2)])
        # T64 fallback does not advance either direct-dispatch terminal.
        self.assertEqual(counters.advance(True, direct_eligible=False), (None, None))
        self.assertEqual(counters.advance(True), (5, 3))

    def test_fixed_shape_imbalance_totals_and_compact_peer_capacity(self):
        for tokens in (32, 40, 48):
            balanced = fixed_assignment_counts("balanced", tokens)
            self.assertEqual([sum(map(sum, owner)) for owner in balanced], [tokens * 16] * 16)
            for mode, total, active_owners in (
                    ("single_hot_owner", tokens * 16 * 16, 1),
                    ("paired_hot_owners", tokens * 16 * 16, 2),
                    ("asymmetric", tokens * 16, 1), ("empty", 0, 0)):
                fixture = fixed_assignment_counts(mode, tokens)
                totals = [sum(map(sum, owner)) for owner in fixture]
                self.assertEqual(sum(totals), total)
                self.assertEqual(sum(n > 0 for n in totals), active_owners)
                self.assertLessEqual(max(sum(source) for owner in fixture for source in owner), 768)
            hot = fixed_assignment_counts("single_hot_owner", tokens)
            expert_counts = [sum(source[e] for source in hot[8]) for e in range(56)]
            self.assertEqual(expert_counts, [16 * tokens] * 16 + [0] * 40)

    def test_balanced_hot_empty_asymmetric_generations_random_delivery(self):
        for tokens in (32, 40, 48):
            models = [DispatchGeneration(sources=16) for _ in range(16)]
            rng = random.Random(tokens)
            for generation, mode in enumerate(("balanced", "single_hot_owner", "balanced",
                                                "empty", "asymmetric", "paired_hot_owners", "balanced")):
                fixture = fixed_assignment_counts(mode, tokens)
                for owner, model in enumerate(models):
                    if generation:
                        model.begin(generation)
                    controls(model, fixture[owner])
                    sources = list(range(16))
                    rng.shuffle(sources)
                    for source in sources:
                        payload(model, source)
                        consume_ready(model)
                    self.assertEqual(sum(len(model.tiles(e)[tile]) for e, tile in model.completed),
                                     sum(map(sum, fixture[owner])))
                    model.retire()
                    with self.assertRaises(ProtocolError):
                        model.acquire_payload(0, generation)


if __name__ == "__main__":
    unittest.main()
