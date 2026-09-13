#!/usr/bin/env python3
"""Host-only contracts for fixed contiguous MegaMoE COMBINE owner waves.

The executable model specifies sender coverage, readiness, and terminal order.
It is not a CUDA, GIN, or fabric timing simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh"
JIT = ROOT / "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp"
PUBLIC = ROOT / "deep_gemm/mega/__init__.py"
ACCURACY = ROOT / "tests/test_mega_moe_accuracy.py"
MATCHED = ROOT / "tests/bench_gb200_transport_matched.py"
COMPARATOR = ROOT / "tests/bench_deepep_trtllm_isolated.py"
COMPILE_TU = ROOT / "tests/compile_sm100_megamoe_gin_ep8.cu"
FLAG = "DG_MEGAMOE_GIN_COMBINE_OWNER_WAVES"
EXPERTS = 56
PEERS = 4


class ProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class Submission:
    peer_lane: int
    wave: int | None
    kind: str
    begin: int
    count: int


class CoarseOwnerWaveModel:
    """Exact EP8 sender-side packet scheduling model."""

    def __init__(self, peer_expert_counts, waves):
        if waves not in (2, 4, 8):
            raise ProtocolError("owner waves must be 2, 4, or 8")
        if len(peer_expert_counts) != PEERS or any(
                len(counts) != EXPERTS for counts in peer_expert_counts):
            raise ProtocolError("EP8 requires four peers and 56 owner experts")
        if any(type(count) is not int or count < 0
               for counts in peer_expert_counts for count in counts):
            raise ProtocolError(
                "expert-assignment counts must be nonnegative integers")
        self.counts = [list(counts) for counts in peer_expert_counts]
        self.waves = waves
        self.experts_per_wave = EXPERTS // waves
        self.prefixes = []
        for counts in self.counts:
            cursor = 0
            prefixes = []
            for count in counts:
                prefixes.append(cursor)
                cursor += count
            self.prefixes.append(prefixes)
        self.headers = [sum(counts) for counts in self.counts]

    def expert_range(self, wave):
        begin = wave * self.experts_per_wave
        return range(begin, begin + self.experts_per_wave)

    def schedule(self, expert_completion_order):
        if len(expert_completion_order) != len(set(expert_completion_order)):
            raise ProtocolError("expert completion order contains a duplicate")
        if any(type(expert) is not int or not 0 <= expert < EXPERTS
               for expert in expert_completion_order):
            raise ProtocolError("invalid owner expert in completion order")

        completed = set()
        pending = []
        for counts in self.counts:
            pending.append({
                wave for wave in range(self.waves)
                if any(counts[expert] for expert in self.expert_range(wave))
            })
        submissions = []
        acquisitions = []

        def issue_ready(peer_lane):
            counts = self.counts[peer_lane]
            while True:
                ready = [
                    wave for wave in sorted(pending[peer_lane])
                    if all(
                        counts[expert] == 0 or expert in completed
                        for expert in self.expert_range(wave)
                    )
                ]
                if not ready:
                    return
                wave = ready[0]
                records = 0
                prefix = None
                for expert in self.expert_range(wave):
                    count = counts[expert]
                    if count == 0:
                        continue
                    if expert not in completed:
                        raise ProtocolError(
                            "payload submitted before contributing expert")
                    saved_prefix = self.prefixes[peer_lane][expert]
                    if prefix is None:
                        prefix = saved_prefix
                    elif saved_prefix != prefix + records:
                        raise ProtocolError(
                            "owner-wave packet interval has a gap or duplicate")
                    records += count
                    acquisitions.append((peer_lane, wave, expert))
                if not records or prefix is None:
                    raise ProtocolError("empty owner wave was submitted")
                pending[peer_lane].remove(wave)
                submissions.append(Submission(
                    peer_lane, wave,
                    "strongva_terminal_span"
                    if not pending[peer_lane] else "payload_span",
                    prefix, records,
                ))

        for expert in expert_completion_order:
            completed.add(expert)
            for peer_lane in range(PEERS):
                issue_ready(peer_lane)

        if any(peer_pending for peer_pending in pending):
            raise ProtocolError("completion order omitted a contributing expert")
        for peer_lane, header in enumerate(self.headers):
            if header == 0:
                submissions.append(Submission(
                    peer_lane, None, "strongva_signal", 0, 0))
        return submissions, acquisitions


class CoarseOwnerWaveProtocolTests(unittest.TestCase):
    def assert_exact_peer_coverage(self, model, submissions, peer_lane):
        spans = [submission for submission in submissions
                 if submission.peer_lane == peer_lane and submission.count]
        covered = []
        for submission in spans:
            covered.extend(range(submission.begin,
                                 submission.begin + submission.count))
        self.assertEqual(len(covered), len(set(covered)))
        self.assertEqual(sorted(covered), list(range(model.headers[peer_lane])))
        if spans:
            self.assertEqual(
                sum(submission.kind == "strongva_terminal_span"
                    for submission in spans), 1)
            self.assertEqual(spans[-1].kind, "strongva_terminal_span")

    def test_all_partitions_cover_each_owner_expert_exactly_once(self):
        empty = [[0] * EXPERTS for _ in range(PEERS)]
        for waves in (2, 4, 8):
            with self.subTest(waves=waves):
                model = CoarseOwnerWaveModel(empty, waves)
                ranges = [list(model.expert_range(wave))
                          for wave in range(waves)]
                flattened = [expert for expert_range in ranges
                             for expert in expert_range]
                self.assertEqual(flattened, list(range(EXPERTS)))
                self.assertEqual({len(expert_range) for expert_range in ranges},
                                 {EXPERTS // waves})

    def test_every_single_expert_assignment_maps_to_one_terminal_range(self):
        for waves in (2, 4, 8):
            for expert in range(EXPERTS):
                with self.subTest(waves=waves, expert=expert):
                    counts = [[0] * EXPERTS for _ in range(PEERS)]
                    counts[0][expert] = expert % 3 + 1
                    model = CoarseOwnerWaveModel(counts, waves)
                    submissions, acquisitions = model.schedule([expert])
                    peer_zero = [submission for submission in submissions
                                 if submission.peer_lane == 0]
                    self.assertEqual(peer_zero, [Submission(
                        0, expert // (EXPERTS // waves),
                        "strongva_terminal_span", 0, expert % 3 + 1)])
                    self.assertEqual(acquisitions, [
                        (0, expert // (EXPERTS // waves), expert)])
                    self.assert_exact_peer_coverage(model, submissions, 0)
                    self.assertEqual(
                        sum(submission.kind == "strongva_signal"
                            for submission in submissions), 3)

    def test_sparse_ranges_issue_in_readiness_not_address_order_without_gaps(self):
        for waves in (2, 4, 8):
            counts = [
                [((expert + peer) % 5 == 0) * (expert % 3 + 1)
                 for expert in range(EXPERTS)]
                for peer in range(PEERS)
            ]
            model = CoarseOwnerWaveModel(counts, waves)
            completion_order = [
                expert
                for wave in reversed(range(waves))
                for expert in model.expert_range(wave)
            ]
            submissions, acquisitions = model.schedule(completion_order)
            for peer in range(PEERS):
                with self.subTest(waves=waves, peer=peer):
                    peer_submissions = [
                        submission for submission in submissions
                        if submission.peer_lane == peer and submission.count
                    ]
                    self.assertEqual(
                        [submission.wave for submission in peer_submissions],
                        list(reversed(range(waves))))
                    self.assertEqual(peer_submissions[-1].wave, 0)
                    self.assert_exact_peer_coverage(model, submissions, peer)
            expected_acquisitions = sum(
                count != 0 for peer_counts in counts for count in peer_counts)
            self.assertEqual(len(acquisitions), expected_acquisitions)

    def test_extreme_imbalance_and_empty_pairs_keep_exact_terminal_rules(self):
        counts = [[0] * EXPERTS for _ in range(PEERS)]
        counts[0][55] = 192
        counts[1][0] = 1
        counts[3][7], counts[3][48] = 3, 5
        for waves in (2, 4, 8):
            with self.subTest(waves=waves):
                model = CoarseOwnerWaveModel(counts, waves)
                submissions, _ = model.schedule(list(reversed(range(EXPERTS))))
                for peer in range(PEERS):
                    self.assert_exact_peer_coverage(model, submissions, peer)
                empty_peer = [submission for submission in submissions
                              if submission.peer_lane == 2]
                self.assertEqual(empty_peer, [Submission(
                    2, None, "strongva_signal", 0, 0)])

    def test_all_zero_expert_assignments_emit_only_empty_pair_signals(self):
        for waves in (2, 4, 8):
            model = CoarseOwnerWaveModel(
                [[0] * EXPERTS for _ in range(PEERS)], waves)
            submissions, acquisitions = model.schedule([])
            self.assertEqual(acquisitions, [])
            self.assertEqual(submissions, [
                Submission(peer, None, "strongva_signal", 0, 0)
                for peer in range(PEERS)
            ])

    def test_missing_readiness_and_corrupt_prefix_fail_closed(self):
        counts = [[0] * EXPERTS for _ in range(PEERS)]
        counts[0][1], counts[0][5] = 2, 3
        model = CoarseOwnerWaveModel(counts, 2)
        with self.assertRaisesRegex(ProtocolError, "omitted a contributing"):
            model.schedule([1])
        model.prefixes[0][5] += 1
        with self.assertRaisesRegex(ProtocolError, "gap or duplicate"):
            model.schedule([1, 5])


class CoarseOwnerWaveSourceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel = KERNEL.read_text()
        cls.jit = JIT.read_text()
        cls.public = PUBLIC.read_text()
        cls.accuracy = ACCURACY.read_text()
        cls.matched = MATCHED.read_text()
        cls.comparator = COMPARATOR.read_text()
        cls.compile_tu = COMPILE_TU.read_text()

    def test_flag_is_default_off_exact_and_jit_keyed(self):
        self.assertIn(f"#define {FLAG} 0", self.kernel)
        self.assertIn(f"#define {FLAG} {{}}", self.jit)
        self.assertIn(f'"{FLAG}", "0"', self.jit)
        self.assertIn("args.gin_combine_owner_waves", self.jit)
        for raw in ('"0"', '"2"', '"4"', '"8"'):
            self.assertIn(f"gin_combine_owner_waves_value == {raw}", self.jit)
        self.assertIn("kOwnerWaveDivisor", self.kernel)
        self.assertIn("kMegaMoeGinCombineOwnerWaves == 0\n"
                      "                            ? 1u", self.kernel)
        self.assertIn(f"{FLAG}=0/2/4/8", self.compile_tu)

    def test_candidate_fails_closed_to_ep8_e448_strongva_stack(self):
        candidate = self.kernel.split(
            "static constexpr uint32_t kMegaMoeGinCombineOwnerWaves", 1)[1]
        self.assertIn("kMegaMoeGinStrongVACombineTerminal", candidate[:1000])
        sender = self.kernel.split(
            "// Divide the fixed 56 owner experts", 1)[1]
        for requirement in (
                "kNumRanks == 8", "kNumExperts == 448",
                "kGinPeerCount == 4", "kNumExpertsPerRank == 56"):
            self.assertIn(requirement, sender[:1200])
        self.assertIn("num_ranks == 8 and num_experts == 448", self.jit)
        self.assertIn("canonical['world_size'] == 8", self.public)
        self.assertIn("canonical['num_experts'] == 448", self.public)

    def test_sender_uses_two_mask_intersections_and_actual_last_terminal(self):
        sender = self.kernel.split(
            "// Divide the fixed 56 owner experts", 1)[1].split(
                "DG_GIN_TRACE_IF(lane_idx == 0, 100)", 1)[0]
        self.assertIn("contributing_first & ~ready_masks[0]", sender)
        self.assertIn("contributing_second & ~ready_masks[1]", sender)
        readiness = sender.split("uint32_t ready_owner_waves", 1)[1].split(
            "if (has_ready_owner_wave)", 1)[0]
        self.assertNotIn("expert_offset", readiness)
        issue = sender.split("if (has_ready_owner_wave)", 1)[1]
        self.assertIn("for (uint32_t expert = expert_begin;", issue)
        self.assertIn("prefix == batch_prefix + batch_records", issue)
        self.assertIn("remaining_owner_waves == 0", issue)
        self.assertIn(
            "mega_moe_gin_put_bulk_combine_terminal_span(", issue)
        self.assertLess(
            self.kernel.index("mega_moe_gin_put_bulk_combine_header_async"),
            self.kernel.index("// Divide the fixed 56 owner experts"))

    def test_cli_collective_and_outer_metadata_are_explicit(self):
        self.assertIn('"--gin-combine-owner-waves"', self.accuracy)
        self.assertIn('"--gin-combine-owner-waves"', self.matched)
        self.assertIn(
            "DG_MEGAMOE_GIN_COMBINE_OWNER_WAVES must be exactly",
            self.public)
        self.assertIn("owner_wave_values", self.accuracy)
        self.assertIn("configuration mismatch across ranks", self.public)
        self.assertIn("actual_last_strongva_terminal", self.comparator)
        final_forward = self.matched.split(
            "record.update(matched.dispatch_candidate_metadata", 1)[1]
        self.assertIn(
            "options.mode, options.gin_combine_owner_waves",
            final_forward[:300])

    def test_default_reducer_fallback_and_gated_candidate_are_pinned(self):
        start = self.kernel.index(
            "        // Combine: reduce top-k results and write back")
        end = self.kernel.index("\n    }\n#else", start)
        reducer = self.kernel[start:end]
        self.assertIn("if (use_gin_owner_slot_pair_reduce)", reducer)
        fallback = reducer.split(
            "if (use_gin_owner_slot_pair_reduce)", 1)[1].split(
                "} else {", 1)[1]
        self.assertIn("uint32_t combine_phase = 0", fallback)
        self.assertIn("move_mask_and_load", fallback)
        self.assertEqual(
            hashlib.sha256(reducer.encode()).hexdigest(),
            "2a65d7f516feb9efc6ab51ea4418eedcdd387f9fe6e5da6a6931716ac359cfea",
        )


if __name__ == "__main__":
    unittest.main()
