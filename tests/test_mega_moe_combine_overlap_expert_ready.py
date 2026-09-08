"""CPU numerical/protocol contracts; not generated-code or GPU evidence."""

import random
from pathlib import Path
import re
import unittest

from mega_moe_gin_expert_ready_protocol import (
    ExpertReadyGeneration, ProtocolError, saved_dispatch_prefixes,
)


def complete(model, expert):
    for block in range(model.blocks[expert]):
        for fragment in range(model.n_fragments):
            model.release(model.generation, expert, block, fragment)


def finish(model):
    generation = model.generation
    model.flush_payloads(generation)
    model.publish_headers(generation)
    for span in model.spans:
        model.settle(generation, span)
    model.headers_settled = True
    model.acquire_world_put(generation)
    for peer in model.peers:
        model.consume(generation, peer)
    model.retire(generation)


def braced_block(source, start):
    begin = source.index("{", start)
    end, depth = begin + 1, 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


class ExpertReadySourceContracts(unittest.TestCase):
    """Narrow integration guards; generated PTX and GPU tests remain required."""

    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.source = (root / "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh").read_text()

    def test_drainer_uses_exact_targets_parallel_masks_and_immediate_spans(self):
        source = self.source
        after_pulls = source.index("DG_GIN_TRACE_IF(lane_idx == 0, 48u + warp_idx);")
        start = source.index("if (use_gin_combine_overlap and sm_idx == 0 and warp_idx == 0)", after_pulls)
        body = braced_block(source, start)
        for forbidden in ("get_token_src_metadata_ptr", "get_bulk_combine_return_index_ptr",
                          "get_num_total_pool_blocks", "scheduler.get_num_tokens", "pending_count",
                          "pool_offset", "num_active_routes_hint"):
            self.assertNotIn(forbidden, body)
        self.assertIn("*workspace.get_expert_recv_count_sum_ptr(expert)", body)
        self.assertIn("math::ceil_div(tokens, BLOCK_M) * kNumL2Fragments", body)
        self.assertIn("math::constexpr_ceil_div(kNumExpertsPerRank, 32u)", body)
        self.assertIn("retired[group] = target == 0", body)
        self.assertIn("ready_masks[group] = __ballot_sync(0xffffffffu, ready)", body)
        self.assertIn("while (ready_mask != 0)", body)
        selected = body[body.index("const uint32_t selected_lane"):]
        acquire = selected.index("comm::mega_moe_gin_combine_ready_acquire(")
        prefix = selected.index("get_combine_overlap_prefix_ptr(lane_idx, expert)")
        issue = selected.index("comm::mega_moe_gin_put_bulk_combine_span(")
        retire = selected.index("retired[group] = true")
        self.assertLess(acquire, prefix)
        self.assertLess(prefix, issue)
        self.assertLess(issue, retire)
        self.assertIn("} while (completed != target)", selected[:prefix])
        self.assertIn("/*context_stripe=*/ 0u", selected[issue:retire])
        self.assertIn("__syncwarp();", selected[issue:retire])
        self.assertIn("ready_mask &= ready_mask - 1u", selected)
        self.assertLess(body.index("DG_GIN_TRACE_IF(lane_idx == 0, 100)"),
                        body.index("comm::mega_moe_gin_flush_data_peer_async("))
        self.assertLess(body.index("comm::mega_moe_gin_wait_data_peer("),
                        body.index("DG_GIN_TRACE_IF(lane_idx == 0, 101)"))

    def test_all_dispatch_unpack_variants_save_existing_prefix_before_publication(self):
        source = self.source
        begin = source.index("// Restore the legacy per-source/per-expert route and count")
        end = source.index("// GIN publishes per-source cells.", begin)
        unpack = source[begin:end]
        assignments = re.findall(
            r"get_combine_overlap_prefix_ptr\(\s*source_lane,\s*(\w+)\)\s*=\s*(\w+);",
            unpack)
        self.assertEqual(assignments, [("first_expert", "first_prefix"),
                                       ("second_expert", "second_prefix"),
                                       ("local_expert", "route_prefix")])
        self.assertEqual(unpack.count("if (use_gin_combine_overlap)"), 3)
        self.assertGreater(unpack.index("__threadfence();"),
                           unpack.rindex("get_combine_overlap_prefix_ptr("))
        self.assertGreater(source.index('asm volatile("st.release.gpu.global.u64', end), end)

    def test_producer_indexes_actual_expert_and_reset_precedes_initial_handoff(self):
        source = self.source
        initialization = source[source.index("// The tail of GET scale scratch"):
                                source.index("// VA readiness counters are cumulative")]
        self.assertIn("expert < layout::kMegaMoeGinCombineOverlapNumExperts", initialization)
        self.assertRegex(initialization, r"get_combine_overlap_ready_ptr\(\s*expert\) = 0")
        self.assertRegex(initialization, r"get_combine_overlap_sent_ptr\(\s*expert\) = 0")
        self.assertIn("__threadfence();", initialization)
        self.assertLess(source.index("// VA readiness counters are cumulative"),
                        source.index("// Ensure the epilogue barrier cannot run with the pull barrier"))
        start = source.index("if (use_gin_combine_overlap and not task_info.is_shared())")
        producer = braced_block(source, start)
        self.assertLess(producer.index("mega_moe_gin_combine_producer_barrier("),
                        producer.index("mega_moe_gin_combine_ready_release("))
        self.assertIn("if (epilogue_warp_idx == 0 and lane_idx == 0)", producer)
        self.assertRegex(producer, r"get_combine_overlap_ready_ptr\(\s*task_info.local_expert_idx\)")
        self.assertNotIn("pool_block_idx", producer)
        late = source[source.index("// Deallocate tensor memory"):]
        header = late.index("comm::mega_moe_gin_put_bulk_combine_header(")
        self.assertLess(late.index("kDispatchWithEpilogueBarrierIdx"), header)
        self.assertLess(header, late.index("mega_moe_gin_world_barrier_single_combine_context("))


class ExpertReadyModelTests(unittest.TestCase):
    def test_prefixes_are_exact_exclusive_and_snapshot_once(self):
        counts = [[2, 0, 3, 1], [0, 4, 0, 2]]
        model = ExpertReadyGeneration((0, 1))
        model.begin(0, counts)
        self.assertEqual([model.descriptors[0, e] for e in range(4)],
                         [(0, 2), (2, 0), (2, 3), (5, 1)])
        self.assertEqual([model.descriptors[1, e] for e in range(4)],
                         [(0, 0), (0, 4), (4, 0), (4, 2)])
        counts[0][0] = 100
        self.assertEqual(model.counts[0][0], 2)
        for expert in range(4):
            complete(model, expert)
        model.drain_ready(0, (3, 1, 0, 2))
        self.assertEqual(model.descriptor_constructions, 1)
        self.assertEqual(model.expected, {0: 6, 1: 6})
        finish(model)

    def test_exact_targets_include_partial_blocks_and_all_sources(self):
        model = ExpertReadyGeneration((1,), block_m=32)
        model.begin(0, [[32, 0, 1, 64], [1, 0, 32, 0]])
        self.assertEqual(model.totals, (33, 0, 33, 64))
        self.assertEqual(model.blocks, (2, 0, 2, 2))
        self.assertEqual(model.targets, (56, 0, 56, 56))
        for fragment in range(28):
            model.release(0, 0, 0, fragment)
        self.assertFalse(model.acquire(0, 0, 1))
        with self.assertRaises(ProtocolError):
            model.issue_expert(0, 0)
        for fragment in range(28):
            model.release(0, 0, 1, fragment)
        self.assertTrue(model.acquire(0, 0, 1))
        self.assertEqual(model.issue_expert(0, 0)[0].count, 1)

    def test_missing_last_fragment_duplicate_and_missing_acquire_rejected(self):
        model = ExpertReadyGeneration((0, 1))
        model.begin(0, [[1], [1]])
        for fragment in range(27):
            model.release(0, 0, 0, fragment)
        self.assertEqual(model.drain_ready(0), ())
        with self.assertRaises(ProtocolError):
            model.release(0, 0, 0, 0)
        model.release(0, 0, 0, 27)
        model.acquire(0, 0, 0)
        with self.assertRaises(ProtocolError):
            model.issue_expert(0, 0)
        self.assertEqual(model.spans, [])
        model.acquire(0, 0, 1)
        self.assertEqual(len(model.issue_expert(0, 0)), 2)
        with self.assertRaises(ProtocolError):
            model.issue_expert(0, 0)

    def test_ready_expert_is_issued_without_waiting_for_prior_experts(self):
        model = ExpertReadyGeneration((1,))
        model.begin(0, [[2, 0, 3], [1, 4, 2]])
        complete(model, 2)
        first = model.drain_ready(0)
        self.assertEqual([(s.expert, s.begin, s.count) for s in first], [(2, 5, 2)])
        self.assertNotIn(0, model.issued_experts)
        self.assertEqual(model.drain_ready(0), ())
        complete(model, 0)
        self.assertEqual([(s.begin, s.count) for s in model.drain_ready(0)], [(0, 1)])
        complete(model, 1)
        model.drain_ready(0)
        finish(model)

    def test_balanced_t32_has_same_granularity_as_full_m_block(self):
        counts = [[0] * 56 for _ in range(16)]
        for route in range(512):
            counts[route % 16][route % 56] += 1
        model = ExpertReadyGeneration(range(8, 16))
        model.begin(0, counts)
        self.assertEqual(sorted(model.totals), [9] * 48 + [10] * 8)
        self.assertEqual(model.targets, (28,) * 56)
        self.assertEqual(sum(model.targets), 1568)

    def test_hot_owner_uses_actual_counts_not_balanced_hint(self):
        for tokens, blocks, target, expected in ((32, 16, 448, 512), (48, 24, 672, 768)):
            with self.subTest(tokens=tokens):
                model = ExpertReadyGeneration(range(8, 16))
                model.begin(0, [[tokens] * 16 + [0] * 40 for _ in range(16)])
                self.assertEqual(model.blocks[:16], (blocks,) * 16)
                self.assertEqual(model.targets[:16], (target,) * 16)
                self.assertEqual(model.targets[16:], (0,) * 40)
                for expert in range(16):
                    complete(model, expert)
                model.drain_ready(0, tuple(reversed(range(56))))
                self.assertEqual(len(model.spans), 8 * 16)
                self.assertEqual(model.expected, dict.fromkeys(range(8, 16), expected))
                self.assertEqual(len(model.issued_records), 8 * expected)
                finish(model)

    def test_empty_experts_and_peers_emit_no_network_spans_or_headers(self):
        model = ExpertReadyGeneration((1, 2))
        model.begin(0, [[3, 0], [0, 0], [0, 0]])
        self.assertEqual(model.drain_ready(0), ())
        # No remote work does not excuse prematurely resetting local producers.
        model.flush_payloads(0)
        with self.assertRaises(ProtocolError):
            model.publish_headers(0)
        complete(model, 0)
        model.publish_headers(0)
        model.headers_settled = True
        model.acquire_world_put(0)
        model.consume(0, 1)
        model.consume(0, 2)
        model.retire(0)
        self.assertFalse(model.spans or model.flushed or model.headers)

    def test_all_masked_generation_retires_without_puts(self):
        model = ExpertReadyGeneration(range(8, 16))
        model.begin(0, [[0] * 56 for _ in range(16)])
        self.assertEqual(model.targets, (0,) * 56)
        self.assertEqual(model.drain_ready(0), ())
        finish(model)
        self.assertFalse(model.spans or model.flushed or model.headers)

    def test_corrupt_saved_prefix_cannot_duplicate_or_omit_records(self):
        for bad_prefix in (0, 3):
            model = ExpertReadyGeneration((0,))
            model.begin(0, [[2, 1]])
            complete(model, 0)
            complete(model, 1)
            model.acquire(0, 0, 0)
            model.issue_expert(0, 0)
            model.descriptors[0, 1] = (bad_prefix, 1)
            model.acquire(0, 1, 0)
            if bad_prefix == 0:
                with self.assertRaises(ProtocolError):
                    model.issue_expert(0, 1)
            else:
                model.issue_expert(0, 1)
                with self.assertRaises(ProtocolError):
                    model.flush_payloads(0)

    def test_random_counts_ready_orders_never_overlap_or_drop_records(self):
        rng = random.Random(20260908)
        for trial in range(80):
            counts = [[rng.randrange(5) for _ in range(56)] for _ in range(16)]
            model = ExpertReadyGeneration(range(8, 16), block_m=(16, 32, 64)[trial % 3])
            model.begin(0, counts)
            events = [(e, block, fragment) for e in range(56)
                      for block in range(model.blocks[e]) for fragment in range(28)]
            rng.shuffle(events)
            for event, (expert, block, fragment) in enumerate(events):
                model.release(0, expert, block, fragment)
                if event % 97 == 0:
                    order = list(range(56)); rng.shuffle(order)
                    model.drain_ready(0, order)
            order = list(range(56)); rng.shuffle(order)
            model.drain_ready(0, order)
            expected = {(peer, ordinal) for peer in model.peers
                        for ordinal in range(sum(counts[peer]))}
            self.assertEqual(model.issued_records, expected)
            self.assertEqual(sum(s.count for s in model.spans), len(expected))
            self.assertEqual(model.descriptor_constructions, 1)
            finish(model)

    def test_local_flush_does_not_prove_remote_visibility(self):
        model = ExpertReadyGeneration((0,))
        model.begin(0, [[1]])
        complete(model, 0)
        model.drain_ready(0)
        model.flush_payloads(0)
        model.publish_headers(0)
        model.headers_settled = True
        with self.assertRaises(ProtocolError):
            model.acquire_world_put(0)
        with self.assertRaises(ProtocolError):
            model.consume(0, 0)
        model.settle(0, model.spans[0])
        model.acquire_world_put(0)
        model.consume(0, 0)
        model.retire(0)

    def test_replay_mode_and_scratch_fit_fallback_preserve_all_records(self):
        model = ExpertReadyGeneration((1,))
        previous_span = None
        schedule = ((False, True, True, 48), (True, True, True, 48),
                    (True, False, True, 64), (True, True, False, 48),
                    (True, True, True, 48), (False, True, True, 48))
        for generation, (enabled, eligible, fits, tokens) in enumerate(schedule):
            model.begin(generation, [[0] * 56, [tokens] * 16 + [0] * 40],
                        enabled=enabled, eligible=eligible, scratch_fits=fits)
            self.assertEqual(model.early, enabled and eligible and fits)
            self.assertFalse(model.issued_records or model.spans or model.flushed)
            if previous_span is not None:
                with self.assertRaises(ProtocolError):
                    model.settle(generation, previous_span)
            for expert in range(56):
                complete(model, expert)
            if model.early:
                model.drain_ready(generation)
            else:
                self.assertEqual(model.drain_ready(generation), ())
                model.publish_fallback(generation)
            self.assertEqual(model.expected[1], tokens * 16)
            self.assertEqual(len(model.issued_records), tokens * 16)
            previous_span = model.spans[0]
            finish(model)

    def test_capacity_does_not_silently_truncate_and_live_generation_cannot_reset(self):
        model = ExpertReadyGeneration((0,))
        with self.assertRaises(ProtocolError):
            model.begin(0, [[64] * 16])
        model.begin(0, [[64] * 16], eligible=False)
        self.assertEqual(model.expected[0], 1024)
        with self.assertRaises(ProtocolError):
            model.begin(1, [[1] * 16])
        with self.assertRaises(ProtocolError):
            model.publish_fallback(0)
        with self.assertRaises(ProtocolError):
            model.release(-1, 0, 0, 0)

    def test_invalid_counts_and_configuration_rejected(self):
        for counts, peers in (([], ()), ([[1], []], (0,)), ([[-1]], (0,)),
                              ([[True]], (0,)), ([[1]], (1,)), ([[1]], (0, 0))):
            with self.subTest(counts=counts, peers=peers), self.assertRaises(ValueError):
                saved_dispatch_prefixes(counts, peers)
        for block_m in (0, -1, True, 1.5):
            with self.subTest(block_m=block_m), self.assertRaises(ValueError):
                ExpertReadyGeneration((0,), block_m=block_m)


if __name__ == "__main__":
    unittest.main()
