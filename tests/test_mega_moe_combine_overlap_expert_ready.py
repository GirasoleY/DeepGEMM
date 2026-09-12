"""CPU numerical/protocol contracts; not generated-code or GPU evidence."""

import ast
from dataclasses import replace
import math
import random
from pathlib import Path
import re
import unittest

from mega_moe_gin_expert_ready_protocol import (
    ExpertReadyGeneration, PeerReadyGeneration, ReadyCoalesceGeneration,
    ProtocolError, saved_dispatch_prefixes,
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

    def test_drainer_uses_exact_targets_parallel_masks_and_bounded_ready_batches(self):
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
        self.assertIn("discovered[group] = target == 0", body)
        self.assertIn("ready_masks[group] |= __ballot_sync(0xffffffffu, newly_ready)", body)
        self.assertIn("while (__any_sync(0xffffffffu, (pending_first | pending_second) != 0))", body)
        self.assertNotIn("while (ready_mask != 0)", body)
        self.assertIn("pending_first & ready_masks[0]", body)
        self.assertIn("pending_second & ready_masks[1]", body)
        self.assertNotRegex(re.sub(r"//[^\n]*", "", body), r"\bpending\[")
        self.assertIn("(nonempty >> (kNumExpertsPerRank - 32u)) == 0", body)
        self.assertRegex(body, r"nonempty = peer_lane\s*\?\s*\*buffer.gin_workspace."
                         r"get_combine_overlap_nonempty_mask_ptr\(\s*lane_idx, group\)\s*:\s*0u")
        self.assertRegex(body, r"first_target = __shfl_sync\(\s*0xffffffffu, "
                         r"expected_fragments\[0\], selected_lane\)")
        self.assertRegex(body, r"second_target = __shfl_sync\(\s*0xffffffffu, "
                         r"expected_fragments\[1\], selected_lane\)")
        self.assertNotIn("expected_fragments[selected_group]", re.sub(r"//[^\n]*", "", body))
        self.assertIn("constexpr uint32_t kMaxReadyExpertsPerPut = 8u;", body)
        self.assertRegex(body, r"stage < kMaxReadyExpertsPerPut and\s*__any_sync\(0xffffffffu, can_extend\)")
        planner = braced_block(body, body.index("for (uint32_t stage = 0;"))
        self.assertLess(planner.index("const uint32_t first_target"), planner.index("if (can_extend)"))
        self.assertLess(planner.index("const uint32_t second_target"), planner.index("if (can_extend)"))
        self.assertNotIn("ready_masks[", planner)
        self.assertNotIn("mega_moe_gin_put_bulk_combine_span", planner)
        self.assertNotIn("get_bulk_combine_record_ptr", planner)
        self.assertRegex(planner, r"accept = batch_records == 0 or\s*prefix == batch_prefix \+ batch_records")
        self.assertRegex(planner, r"if \(not accept\)\s*can_extend = false")
        accepted = braced_block(planner, planner.index("if (accept)"))
        acquire = accepted.index("comm::mega_moe_gin_combine_ready_acquire(")
        self.assertLess(acquire, accepted.index("batch_records += count"))
        self.assertIn("} while (completed != target)", accepted)
        self.assertIn("accepted_first |= bit", accepted)
        self.assertIn("accepted_second |= bit", accepted)
        self.assertIn("available_first &= ~bit", accepted)
        self.assertIn("available_second &= ~bit", accepted)
        self.assertNotIn("pending_", accepted)
        self.assertLess(body.index(planner) + len(planner), body.index("if (batch_records != 0)"))
        submission = braced_block(body, body.index("if (batch_records != 0)"))
        # One ordinary helper remains in each compile-time branch: non-final
        # StrongVA spans and the unchanged default-off path.
        self.assertEqual(body.count("comm::mega_moe_gin_put_bulk_combine_span("), 2)
        self.assertEqual(
            body.count("comm::mega_moe_gin_put_bulk_combine_terminal_span("), 1)
        issue = submission.index("comm::mega_moe_gin_put_bulk_combine_span(")
        self.assertLess(issue, submission.index("pending_first = remaining_first"))
        self.assertLess(issue, submission.index("pending_second = remaining_second"))
        self.assertIn("/*context_stripe=*/ 0u", submission)
        self.assertRegex(
            submission,
            r"batch_records\s*\*\s*buffer\.gin_workspace\.bulk_record_bytes",
        )
        self.assertRegex(body, r"pending_second = remaining_second;\s*}\s*__syncwarp\(\);")
        self.assertLess(body.index("DG_GIN_TRACE_IF(lane_idx == 0, 100)"),
                        body.index("get_combine_overlap_sent_ptr(expert) = 1"))
        self.assertLess(body.index("DG_GIN_TRACE_IF(lane_idx == 0, 100)"),
                        body.index("DG_DEVICE_ASSERT(sent_records == expected_records)"))
        # The default-off branch still defers completion; the StrongVA branch
        # completes once per peer after its exact final terminal.
        audit = body.index("DG_DEVICE_ASSERT(sent_records == expected_records)")
        self.assertLess(audit, body.index("mega_moe_gin_flush_data_peer_async(", audit))
        self.assertLess(audit, body.index("mega_moe_gin_wait_data_peer(", audit))
        self.assertNotRegex(source, r"DG_GIN_TRACE(?:_IF)?\([^;]*\b101\b")

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
        self.assertGreater(unpack.index("__threadfence();"),
                           unpack.rindex("get_combine_overlap_prefix_ptr("))
        self.assertGreater(unpack.index("__threadfence();"),
                           unpack.rindex("get_combine_overlap_nonempty_mask_ptr("))
        self.assertGreater(source.index('asm volatile("st.release.gpu.global.u64', end), end)

    def test_nonempty_mask_publication_covers_warp_and_scalar_paths(self):
        source = self.source
        begin = source.index("// Restore the legacy per-source/per-expert route and count")
        end = source.index("// GIN publishes per-source cells.", begin)
        unpack = source[begin:end]
        first = unpack.index("const uint32_t first_nonempty = __ballot_sync(")
        second = unpack.index("const uint32_t second_nonempty = __ballot_sync(")
        writer = unpack.index("if (lane_idx == 0)", second)
        self.assertLess(first, second)
        self.assertLess(second, writer)
        self.assertRegex(unpack, r"second_expert < kNumExpertsPerRank\s*\? "
                         r"static_cast<uint32_t>\(\s*packed_counts\[second_expert\]\)\s*:\s*0u")
        self.assertRegex(unpack, r"get_combine_overlap_nonempty_mask_ptr\(\s*source_lane, 0u\) = first_nonempty")
        self.assertRegex(unpack, r"get_combine_overlap_nonempty_mask_ptr\(\s*source_lane, 1u\) = second_nonempty")
        scalar = unpack[unpack.index("uint32_t nonempty_mask = 0;"):]
        self.assertIn("local_expert == 31u or", scalar)
        self.assertIn("local_expert + 1u == kNumExpertsPerRank", scalar)
        self.assertIn("prior_expert / 32u == expert_group", scalar)
        self.assertIn("nonempty_mask |= 1u << (prior_expert % 32u)", scalar)
        self.assertIn("nonempty_mask |= 1u << (local_expert % 32u)", scalar)
        self.assertRegex(scalar, r"get_combine_overlap_nonempty_mask_ptr\(\s*source_lane, expert_group\) = nonempty_mask")
        self.assertNotIn("atomic", re.sub(r"//[^\n]*", "", scalar))

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


def matched_owner_counts(mode, tokens=32, owner=0):
    """Execute the real CPU route-owner functions without importing torch."""
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "tests/test_mega_moe_accuracy.py").read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in ("_rank_candidates", "_owner_for")]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
    namespace = {"math": math, "GIN_ACTIVITY_ASYMMETRIC_MODE": "single_remote_source"}
    exec(compile(ast.fix_missing_locations(module), "<actual-route-owners>", "exec"), namespace)
    hostnames = ["host0"] * 8 + ["host1"] * 8
    counts = [[0] * 56 for _ in range(16)]
    for source in range(16):
        for token in range(tokens):
            for slot in range(16):
                destination = namespace["_owner_for"](
                    mode, source, token, slot, hostnames, 16, tokens)
                if destination == owner:
                    counts[source][(token * 16 + slot) % 56] += 1
    return counts


class PeerReadyModelTests(unittest.TestCase):
    def test_scalar_mask_writers_match_warp_ballots_including_empty_sources(self):
        rng = random.Random(20260908)
        for trial in range(60):
            counts = [[rng.randrange(49) if rng.random() < .5 else 0
                       for _ in range(56)] for _ in range(8)]
            counts[trial % 8] = [0] * 56
            writes = {}
            prefixes = {}
            # Same pair-to-thread mapping as scalar dispatch unpack. Only
            # experts31/55 consume their existing prefix counts into a mask.
            for thread in range(128):
                for pair in range(thread, 8 * 56, 128):
                    source, expert = divmod(pair, 56)
                    group = expert // 32
                    mask, prefix = 0, 0
                    writer = expert == 31 or expert + 1 == 56
                    for prior in range(expert):
                        count = counts[source][prior]
                        prefix += count
                        if writer and prior // 32 == group and count:
                            mask |= 1 << (prior % 32)
                    prefixes[source, expert] = prefix
                    if writer:
                        if counts[source][expert]:
                            mask |= 1 << (expert % 32)
                        self.assertNotIn((source, group), writes)
                        writes[source, group] = mask
            self.assertEqual(len(writes), 16)
            for source in range(8):
                for group in range(2):
                    ballot = sum(1 << lane for lane in range(32)
                                 if group * 32 + lane < 56 and counts[source][group * 32 + lane])
                    self.assertEqual(writes[source, group], ballot)
                self.assertEqual(writes[source, 1] >> 24, 0)
                for expert in range(56):
                    self.assertEqual(prefixes[source, expert], sum(counts[source][:expert]))

    def test_masks_include_only_nonempty_peer_experts_and_snapshot_once(self):
        counts = [[0] * 56 for _ in range(3)]
        for expert in (0, 31, 32, 55):
            counts[1][expert] = 1
        counts[2][32] = 3
        model = PeerReadyGeneration((1, 2))
        model.begin(0, counts)
        self.assertEqual(model.nonempty_masks[1], (0x80000001, 0x800001))
        self.assertEqual(model.nonempty_masks[2], (0, 1))
        self.assertEqual(model.pending_masks[1], list(model.nonempty_masks[1]))
        counts[1][1] = 9
        self.assertEqual(model.nonempty_masks[1], (0x80000001, 0x800001))
        self.assertEqual(model.descriptor_constructions, 1)
        for bad in (1 << 24, 1 << 31, -1):
            model.pending_masks[1][1] = bad
            with self.assertRaises(ProtocolError):
                model.poll_ready_masks(0)

    def test_crossing_groups_uses_two_uniform_target_gathers(self):
        counts = [[0] * 56 for _ in range(4)]
        counts[1][31], counts[2][32], counts[3][55] = 1, 33, 65
        model = PeerReadyGeneration((0, 1, 2, 3))
        model.begin(0, counts)
        for expert in (31, 32, 55):
            complete(model, expert)
        selections = model.select_wave(0)
        self.assertEqual(selections, {0: (None, None), 1: (31, 28),
                                      2: (32, 56), 3: (55, 84)})
        group0, group1 = model.target_gathers[-1]
        self.assertEqual(set(group0), set(model.peers))
        self.assertEqual(set(group1), set(model.peers))
        self.assertEqual((group0[1], group1[1]), (28, 0))
        self.assertEqual((group0[2], group1[2]), (0, 56))
        model.acquire(0, 32, 2)
        with self.assertRaises(ProtocolError):
            model.issue_peer(0, 2, 32, selected_target=group0[2])
        self.assertEqual(model.pending_masks[2], [0, 1])
        self.assertEqual(len(model.drain_ready(0)), 3)
        finish(model)

    def test_peer_selection_skips_earlier_unready_and_is_not_common_expert(self):
        counts = [[0] * 56 for _ in range(3)]
        counts[1][0], counts[1][40], counts[2][31] = 1, 2, 3
        model = PeerReadyGeneration((1, 2))
        model.begin(0, counts)
        complete(model, 31)
        complete(model, 40)
        self.assertEqual(model.select_wave(0), {1: (40, 28), 2: (31, 28)})
        self.assertEqual([(span.peer, span.expert) for span in model.drain_ready(0)],
                         [(1, 40), (2, 31)])
        self.assertEqual(model.pending_masks, {1: [1, 0], 2: [0, 0]})
        complete(model, 0)
        self.assertEqual([(span.peer, span.expert) for span in model.drain_ready(0)], [(1, 0)])
        finish(model)

    def test_peer_acquire_and_clear_are_independent_and_exactly_once(self):
        model = PeerReadyGeneration((0, 1))
        model.begin(0, [[1], [2]])
        for fragment in range(27):
            model.release(0, 0, 0, fragment)
        self.assertEqual(model.select_wave(0), {0: (None, None), 1: (None, None)})
        with self.assertRaises(ProtocolError):
            model.issue_peer(0, 0, 0)
        model.release(0, 0, 0, 27)
        model.acquire(0, 0, 0)
        model.issue_peer(0, 0, 0)
        self.assertEqual(model.pending_masks, {0: [0, 0], 1: [1, 0]})
        with self.assertRaises(ProtocolError):
            model.issue_peer(0, 1, 0)
        with self.assertRaises(ProtocolError):
            model.issue_peer(0, 0, 0)
        with self.assertRaises(ProtocolError):
            model.finish_queuing(0)
        model.acquire(0, 0, 1)
        model.issue_peer(0, 1, 0)
        finish(model)

    def test_actual_matched_t32_has_same_90_and_112_nonempty_spans(self):
        for mode, expected_puts, expected_records in (
                ("half_remote", 90, 256), ("all_remote", 112, 512)):
            with self.subTest(mode=mode):
                counts = matched_owner_counts(mode)
                model = PeerReadyGeneration(range(8, 16))
                model.begin(0, counts)
                self.assertEqual(sorted(model.totals), [9] * 48 + [10] * 8)
                for expert in reversed(range(56)):
                    complete(model, expert)
                    model.drain_ready(0, tuple(reversed(model.peers)))
                self.assertEqual(len(model.spans), expected_puts)
                self.assertEqual(sum(span.count for span in model.spans), expected_records)
                self.assertEqual(len({(span.peer, span.expert) for span in model.spans}),
                                 expected_puts)
                self.assertEqual(len(model.issued_records), expected_records)
                self.assertEqual(sum(count != 0 for count in model.expected.values()), 7)
                finish(model)

    def test_hot_owner_counts_and_partial_blocks_are_not_truncated(self):
        for tokens, blocks, target in ((32, 16, 448), (48, 24, 672)):
            for block_m in (16, 32, 64):
                model = PeerReadyGeneration(range(8, 16), block_m=block_m)
                model.begin(0, [[tokens] * 16 + [0] * 40 for _ in range(16)])
                self.assertEqual(model.totals[:16], (16 * tokens,) * 16)
                self.assertEqual(model.targets[0], ((16 * tokens + block_m - 1) // block_m) * 28)
                if block_m == 32:
                    self.assertEqual((model.blocks[0], model.targets[0]), (blocks, target))
                for expert in range(16):
                    complete(model, expert)
                model.drain_ready(0)
                self.assertEqual(len(model.spans), 128)
                self.assertEqual(len(model.issued_records), 8 * tokens * 16)
                finish(model)

    def test_all_masked_and_local_only_sent_bookkeeping_is_not_compute_completion(self):
        for local_count in (0, 65):
            model = PeerReadyGeneration((1,))
            model.begin(0, [[local_count] + [0] * 55, [0] * 56])
            self.assertEqual(model.drain_ready(0), ())
            self.assertEqual(model.sent_bookkeeping, set(range(56)))
            model.flush_payloads(0)
            if local_count:
                with self.assertRaises(ProtocolError):
                    model.publish_headers(0)
                complete(model, 0)
            finish(model)
            self.assertFalse(model.spans or model.headers or model.flushed)

    def test_asymmetric_ready_arrivals_and_peer_orders_preserve_all_records(self):
        rng = random.Random(20260909)
        for trial in range(40):
            counts = [[rng.randrange(4) if rng.random() < .4 else 0
                       for _ in range(56)] for _ in range(16)]
            if trial % 2 == 0:
                counts[8] = [0] * 56
            model = PeerReadyGeneration(range(8, 16), block_m=(16, 32, 64)[trial % 3])
            model.begin(0, counts)
            events = [(expert, block, fragment) for expert in range(56)
                      for block in range(model.blocks[expert]) for fragment in range(28)]
            rng.shuffle(events)
            previous_ready = (0, 0)
            for index, event in enumerate(events):
                model.release(0, *event)
                if index % 79 == 0:
                    peers = list(model.peers); rng.shuffle(peers)
                    model.drain_ready(0, peers)
                    ready = tuple(model.discovered_masks)
                    self.assertTrue(all(old & new == old for old, new in zip(previous_ready, ready)))
                    self.assertEqual(ready[1] & 0xff000000, 0)
                    previous_ready = ready
            model.drain_ready(0)
            expected = {(peer, ordinal) for peer in model.peers
                        for ordinal in range(sum(counts[peer]))}
            self.assertEqual(model.issued_records, expected)
            self.assertEqual(len(model.spans), sum(bool(counts[peer][expert])
                             for peer in model.peers for expert in range(56)))
            self.assertEqual(model.descriptor_constructions, 1)
            finish(model)

    def test_replay_masks_rebuild_through_mode_fit_and_t64_fallback(self):
        model = PeerReadyGeneration((1,))
        previous_span = None
        for generation, (enabled, eligible, fits, tokens) in enumerate((
                (False, True, True, 48), (True, True, True, 48),
                (True, False, True, 64), (True, True, False, 48),
                (True, True, True, 48), (False, True, True, 48))):
            expert = (31, 32, 55)[generation % 3]
            counts = [[0] * 56 for _ in range(2)]
            counts[1][expert] = tokens
            model.begin(generation, counts, enabled=enabled, eligible=eligible, scratch_fits=fits)
            self.assertEqual(model.pending_masks[1][expert // 32], 1 << (expert % 32))
            self.assertEqual(model.discovered_masks, [0, 0])
            self.assertFalse(model.sent_bookkeeping or model.spans or model.flushed)
            if previous_span:
                with self.assertRaises(ProtocolError):
                    model.settle(generation, previous_span)
            complete(model, expert)
            if model.early:
                model.drain_ready(generation)
            else:
                self.assertEqual(model.drain_ready(generation), ())
                model.publish_fallback(generation)
            previous_span = model.spans[0]
            self.assertEqual(len(model.issued_records), tokens)
            finish(model)

    def test_final_flush_and_bookkeeping_do_not_prove_remote_visibility(self):
        model = PeerReadyGeneration((0,))
        model.begin(0, [[1]])
        complete(model, 0)
        model.drain_ready(0)
        self.assertEqual(model.sent_bookkeeping, {0})
        model.flush_payloads(0)
        model.publish_headers(0)
        model.headers_settled = True
        with self.assertRaises(ProtocolError):
            model.acquire_world_put(0)
        with self.assertRaises(ProtocolError):
            model.retire(0)
        finish(model)


class ReadyCoalesceModelTests(unittest.TestCase):
    def test_all_ready_actual_route_totals_for_bounded_cpu_caps(self):
        expected = {"half_remote": (90, 48, 27, 14), "all_remote": (112, 56, 28, 14)}
        for mode, totals in expected.items():
            counts = matched_owner_counts(mode)
            for cap, puts in zip((1, 2, 4, 8), totals):
                with self.subTest(mode=mode, cap=cap):
                    model = ReadyCoalesceGeneration(range(8, 16), cap=cap)
                    model.begin(0, counts)
                    for expert in range(56):
                        complete(model, expert)
                    model.drain_ready(0)
                    self.assertEqual(len(model.spans), puts)
                    self.assertTrue(all(len(span.experts) <= cap for span in model.spans))
                    self.assertEqual(sum(span.count for span in model.spans),
                                     256 if mode == "half_remote" else 512)
                    self.assertEqual(sum(len(span.experts) for span in model.spans), totals[0])
                    self.assertEqual(model.descriptor_constructions, 1)
                    finish(model)

    def test_cap1_matches_r2_spans_and_exact_coverage(self):
        rng = random.Random(20260910)
        for trial in range(12):
            counts = [[rng.randrange(3) for _ in range(56)] for _ in range(16)]
            old = PeerReadyGeneration(range(8, 16))
            new = ReadyCoalesceGeneration(range(8, 16), cap=1)
            for model in (old, new):
                model.begin(0, counts)
            order = list(range(56)); rng.shuffle(order)
            for expert in order:
                for model in (old, new):
                    complete(model, expert)
                    if expert % 4 == 0:
                        model.drain_ready(0)
            old.drain_ready(0); new.drain_ready(0)
            self.assertEqual([(span.peer, span.expert, span.begin, span.count) for span in old.spans],
                             [(span.peer, span.experts[0], span.begin, span.count) for span in new.spans])
            self.assertEqual(old.issued_records, new.issued_records)
            finish(old); finish(new)

    def test_zero_count_expert_gaps_merge_across31_32_and_partial_high_group(self):
        counts = [[0] * 56 for _ in range(3)]
        for expert, count in ((0, 1), (31, 33), (32, 2), (55, 1)):
            counts[1][expert] = count
        counts[2][32] = 65
        model = ReadyCoalesceGeneration((0, 1, 2))
        model.begin(0, counts)
        for expert in (0, 31, 32, 55):
            complete(model, expert)
        plans = model.plan_wave(0)
        self.assertEqual(plans[1].experts, (0, 31, 32, 55))
        self.assertEqual(plans[1].targets, (28, 56, 84, 28))
        self.assertEqual(plans[2].experts, (32,))
        self.assertNotIn(0, plans)
        self.assertEqual(model.planning_rounds[-1][1], 4)
        for group0, group1 in model.target_gathers:
            self.assertEqual(set(group0), {0, 1, 2})
            self.assertEqual(set(group1), {0, 1, 2})
        for plan in plans.values():
            model.issue_batch(0, plan)
        self.assertEqual(len(model.spans), 2)
        finish(model)

    def test_unready_positive_gap_stops_batch_but_not_later_ready_send(self):
        model = ReadyCoalesceGeneration((0,))
        model.begin(0, [[1, 2, 3]])
        complete(model, 0); complete(model, 2)
        first = model.plan_wave(0)[0]
        self.assertEqual(first.experts, (0,))
        model.issue_batch(0, first)
        second = model.plan_wave(0)[0]
        self.assertEqual((second.experts, second.begin, second.count), ((2,), 3, 3))
        model.issue_batch(0, second)
        self.assertEqual(model.pending_masks[0], [2, 0])
        complete(model, 1)
        model.drain_ready(0)
        self.assertEqual(len(model.spans), 3)
        finish(model)

    def test_previously_sent_positive_gap_cannot_be_retransmitted_by_coalescing(self):
        model = ReadyCoalesceGeneration((0,))
        model.begin(0, [[1, 2, 3]])
        complete(model, 1)
        model.drain_ready(0)
        complete(model, 0); complete(model, 2)
        plan = model.plan_wave(0)[0]
        self.assertEqual(plan.experts, (0,))
        forged = replace(plan, experts=(0, 2), count=6, targets=(28, 28))
        model.acquire(0, 2, 0)
        with self.assertRaises(ProtocolError):
            model.issue_batch(0, forged)
        self.assertEqual(len(model.spans), 1)
        model.drain_ready(0)
        self.assertEqual([span.experts for span in model.spans], [(1,), (0,), (2,)])
        self.assertEqual(len(model.issued_records), 6)
        finish(model)

    def test_frozen_snapshot_does_not_add_arriving_expert_mid_plan(self):
        model = ReadyCoalesceGeneration((0,))
        model.begin(0, [[1, 1, 1]])
        complete(model, 0); complete(model, 2)
        def arrive(stage, current):
            if stage == 0:
                complete(current, 1)
        plan = model.plan_wave(0, stage_hook=arrive)[0]
        self.assertEqual(plan.ready_snapshot, (5, 0))
        self.assertEqual(plan.experts, (0,))
        self.assertNotIn(0, model.acquires[1])
        forged = replace(plan, experts=(0, 1, 2), count=3, targets=(28, 28, 28))
        model.acquire(0, 1, 0); model.acquire(0, 2, 0)
        with self.assertRaises(ProtocolError):
            model.issue_batch(0, forged)
        model.issue_batch(0, plan)
        self.assertEqual(model.plan_wave(0)[0].experts, (1, 2))
        model.drain_ready(0)
        finish(model)

    def test_every_contributor_acquired_and_bits_clear_only_after_put_return(self):
        model = ReadyCoalesceGeneration((0, 1))
        model.begin(0, [[1, 2, 3], [4, 0, 5]])
        for expert in range(3):
            complete(model, expert)
        before = {peer: masks[:] for peer, masks in model.pending_masks.items()}
        plans = model.plan_wave(0)
        self.assertEqual(model.pending_masks, before)
        model.acquires[1].remove(0)
        with self.assertRaises(ProtocolError):
            model.issue_batch(0, plans[0])
        self.assertEqual(model.pending_masks, before)
        model.acquire(0, 1, 0)
        calls = []
        def on_submit(current, batch):
            self.assertEqual(current.pending_masks[batch.peer], before[batch.peer])
            self.assertTrue(all(batch.peer in current.acquires[e] for e in batch.experts))
            self.assertFalse(current.flushed)
            calls.append(batch.peer)
        for plan in plans.values():
            model.issue_batch(0, plan, on_submit=on_submit)
        self.assertEqual(calls, [0, 1])
        self.assertEqual(model.pending_masks, {0: [0, 0], 1: [0, 0]})
        with self.assertRaises(ProtocolError):
            model.issue_batch(0, plans[0])
        finish(model)

    def test_arbitrary_peer_lengths_stop_independently_with_fixed_cap(self):
        lengths = (0, 1, 2, 7, 8, 9, 16, 28)
        counts = [[1] * length + [0] * (56 - length) for length in lengths]
        model = ReadyCoalesceGeneration(range(8))
        model.begin(0, counts)
        for expert in range(56):
            complete(model, expert)
        plans = model.plan_wave(0)
        self.assertEqual({peer: len(plan.experts) for peer, plan in plans.items()},
                         {peer: min(length, 8) for peer, length in enumerate(lengths) if length})
        self.assertEqual(model.planning_rounds[-1][1], 8)
        for plan in plans.values():
            model.issue_batch(0, plan)
        model.drain_ready(0)
        self.assertEqual(len(model.spans), sum((length + 7) // 8 for length in lengths))
        self.assertTrue(all(stages <= 8 for _, stages, _ in model.planning_rounds))
        finish(model)

    def test_hot_empty_and_local_only_paths_keep_actual_lifetimes(self):
        for tokens in (32, 48):
            model = ReadyCoalesceGeneration(range(8, 16))
            model.begin(0, [[tokens] * 16 + [0] * 40 for _ in range(16)])
            self.assertEqual(model.targets[0], (16 * tokens // 32) * 28)
            for expert in range(16):
                complete(model, expert)
            model.drain_ready(0)
            self.assertEqual(len(model.spans), 16)
            self.assertEqual(len(model.issued_records), 8 * 16 * tokens)
            finish(model)
        for local_count in (0, 65):
            model = ReadyCoalesceGeneration((1,))
            model.begin(0, [[local_count], [0]])
            self.assertEqual(model.drain_ready(0), ())
            model.flush_payloads(0)
            if local_count:
                with self.assertRaises(ProtocolError):
                    model.publish_headers(0)
                complete(model, 0)
            finish(model)

    def test_random_readiness_and_asymmetric_masks_are_disjoint_and_complete(self):
        rng = random.Random(20260911)
        for trial in range(36):
            counts = [[rng.randrange(4) if rng.random() < .5 else 0
                       for _ in range(56)] for _ in range(16)]
            counts[8 + trial % 8] = [0] * 56
            model = ReadyCoalesceGeneration(range(8, 16), cap=(1, 2, 4, 8)[trial % 4],
                                            block_m=(16, 32, 64)[trial % 3])
            model.begin(0, counts)
            events = [(expert, block, fragment) for expert in range(56)
                      for block in range(model.blocks[expert]) for fragment in range(28)]
            rng.shuffle(events)
            for index, event in enumerate(events):
                model.release(0, *event)
                if index % 83 == 0:
                    model.drain_ready(0)
            model.drain_ready(0)
            expected = {(peer, ordinal) for peer in model.peers
                        for ordinal in range(sum(counts[peer]))}
            self.assertEqual(model.issued_records, expected)
            self.assertEqual(sum(span.count for span in model.spans), len(expected))
            self.assertTrue(all(span.ready_snapshot[e // 32] & (1 << (e % 32))
                                for span in model.spans for e in span.experts))
            self.assertTrue(all(len(span.experts) <= model.cap for span in model.spans))
            finish(model)

    def test_replay_fallback_and_remote_completion_remain_distinct(self):
        model = ReadyCoalesceGeneration((1,))
        old_span = None
        for generation, (enabled, eligible, fits, tokens) in enumerate((
                (False, True, True, 48), (True, True, True, 48),
                (True, False, True, 64), (True, True, False, 48),
                (True, True, True, 48), (False, True, True, 48))):
            counts = [[0] * 56, [tokens] * 16 + [0] * 40]
            model.begin(generation, counts, enabled=enabled, eligible=eligible, scratch_fits=fits)
            self.assertFalse(model.planning_rounds or model.spans or model.flushed)
            if old_span is not None:
                with self.assertRaises(ProtocolError):
                    model.settle(generation, old_span)
            for expert in range(16):
                complete(model, expert)
            if model.early:
                model.drain_ready(generation)
            else:
                self.assertEqual(model.plan_wave(generation), {})
                model.publish_fallback(generation)
            self.assertEqual(len(model.issued_records), tokens * 16)
            model.flush_payloads(generation); model.publish_headers(generation)
            model.headers_settled = True
            with self.assertRaises(ProtocolError):
                model.acquire_world_put(generation)
            old_span = model.spans[0]
            finish(model)

    def test_invalid_cap_and_corrupt_batch_bounds_rejected(self):
        for cap in (0, 3, 16, True):
            with self.assertRaises(ValueError):
                ReadyCoalesceGeneration((0,), cap=cap)
        model = ReadyCoalesceGeneration((0,), cap=2)
        model.begin(0, [[1, 1]])
        complete(model, 0); complete(model, 1)
        plan = model.plan_wave(0)[0]
        for bad in (replace(plan, count=3), replace(plan, targets=(28, 56)),
                    replace(plan, experts=(0, 0)), replace(plan, generation=-1)):
            with self.assertRaises(ProtocolError):
                model.issue_batch(0, bad)
        self.assertFalse(model.spans or model.issued_records)


if __name__ == "__main__":
    unittest.main()
