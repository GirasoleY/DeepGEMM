"""CPU-only runner policy contracts; device gates remain mandatory."""

import argparse
import contextlib
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import bench_mega_moe_single_combine_context as sweep


class TestSingleCombineContextSweep(unittest.TestCase):
    def test_default_control_candidate_control_schedule(self):
        self.assertEqual(sweep.mode_schedule(sweep.DEFAULT_MODES, True), [0, 1, 0])
        self.assertEqual(sweep.mode_schedule([0, 1], False), [0, 1])
        for invalid in ([], [1], [1, 0], [0, 0], [0, 2], [0, 1, 1]):
            with self.assertRaises(ValueError):
                sweep.mode_schedule(invalid, True)

    def test_default_shape_timing_and_accuracy_coverage(self):
        with patch.object(sys, "argv", ["sweep", "--output", "/tmp/unused.json"]):
            options, args = sweep.parse_args()
        self.assertEqual(options.schedule, [0, 1, 0])
        self.assertEqual(options.benchmark_replays, 204)
        self.assertEqual(args.benchmark_replays, 0)
        self.assertEqual(args.benchmark_conditioning_replays, 60)
        self.assertEqual((args.num_tokens, args.num_topk, args.num_experts), (32, 16, 896))
        self.assertEqual(args.payload_epochs, 3)
        self.assertTrue(options.payload_per_mode and options.transitions_per_mode)
        self.assertFalse(args.heterogeneous_input_scales)
        self.assertEqual((args.gin_completion_batch, args.gin_combine_issue_wave,
                          args.gin_outbox_depth, args.gin_combine_chunk_bytes), (8, 8, 64, 7168))

    def test_explicit_shape_and_short_smoke_are_honored(self):
        argv = ["sweep", "--output", "/tmp/unused.json", "--decode-mns=10",
                "--modes", "0", "1", "--no-repeat-control", "--benchmark-replays=24",
                "--benchmark-conditioning-replays=6", "--no-payload-per-mode",
                "--no-transitions-per-mode", "--payload-epochs=1"]
        with patch.object(sys, "argv", argv):
            options, args = sweep.parse_args()
        self.assertEqual(options.schedule, [0, 1])
        self.assertEqual((args.num_tokens, options.benchmark_replays), (40, 24))
        self.assertFalse(options.payload_per_mode or options.transitions_per_mode)

    def test_t64_is_only_a_transition_not_a_timed_shape(self):
        with patch.object(sys, "argv", ["sweep", "--output", "/tmp/unused.json", "--num-tokens=64"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                sweep.parse_args()

    def test_protocol_default_is_pinned_only_after_retired_knobs_are_validated(self):
        for mode in ("0", "1"):
            for flag, value in ((sweep.EXPERT_WIDTH_ENV, "8"),
                                (sweep.BARRIER_WIDTH_ENV, "8"),
                                (sweep.EXPERT_WIDTH_ENV, "00"),
                                (sweep.BARRIER_WIDTH_ENV, "01")):
                with patch.dict(os.environ, {sweep.MODE_ENV: mode, flag: value}, clear=True):
                    before = dict(os.environ)
                    with self.assertRaisesRegex(ValueError, "retired"):
                        sweep.prepare_sweep_environment()
                    self.assertEqual(dict(os.environ), before)
        with patch.dict(os.environ, {sweep.MODE_ENV: "1", sweep.EXPERT_WIDTH_ENV: "0",
                                     sweep.BARRIER_WIDTH_ENV: "1"}, clear=True):
            sweep.prepare_sweep_environment()
            self.assertEqual(os.environ[sweep.MODE_ENV], "0")
            self.assertEqual(os.environ[sweep.EXPERT_WIDTH_ENV], "0")
            self.assertEqual(os.environ[sweep.BARRIER_WIDTH_ENV], "1")
            self.assertEqual(os.environ["DG_MEGAMOE_GIN_DIAGNOSTICS"], "0")
            self.assertNotIn(sweep.MODE_ENV, sweep.accuracy.GIN_EXPERIMENT_FLAG_ENVS)

    def test_context_evidence_keeps_nine_allocated_and_all_input_contexts(self):
        for mode in (0, 1):
            record = sweep.context_evidence(mode, 32)
            self.assertEqual(record["allocated_context_count"], 9)
            self.assertEqual(record["input_data_contexts_unchanged"], list(range(1, 9)))
            self.assertEqual(record["experts_per_wave"], 0)
            self.assertEqual(record["barrier_warps"], 1)
            expected = [1] if mode else list(range(9))
            self.assertEqual(record["effective_combine_barrier_contexts_by_route"]["all_same_host"], [])
            self.assertEqual(record["effective_combine_barrier_contexts_by_route"]["half_remote"], expected)
            self.assertEqual(record["context_peer_pairs_by_route"]["all_remote"], 16 if mode else 144)
            self.assertEqual(record["remote_active_t48_t64_t48"], [expected, list(range(9)), expected])

    def test_ineligible_modes_never_claim_a_single_context_barrier(self):
        for keywords in ({"bulk_eligible": False}, {"direct_dispatch": False}):
            self.assertEqual(sweep.effective_combine_contexts(1, 32, True, **keywords), list(range(9)))
        self.assertEqual(sweep.effective_combine_contexts(1, 64, True), list(range(9)))
        self.assertEqual(sweep.effective_combine_contexts(1, 64, False), [])
        with self.assertRaises(ValueError):
            sweep.effective_combine_contexts(2, 32, True)

    def test_explicit_graph_retirement_survives_one_reset_error(self):
        calls = []
        def fail():
            raise RuntimeError("reset failed")
        graphs = {"bad": argparse.Namespace(reset=fail),
                  "other": argparse.Namespace(reset=lambda: calls.append("other"))}
        with self.assertRaisesRegex(RuntimeError, "reset failed"):
            sweep._retire_graphs(graphs)
        self.assertEqual(graphs, {})
        self.assertEqual(calls, ["other"])

    def test_payload_control_restores_candidate_mode(self):
        seen = []
        tensor = argparse.Namespace(view=lambda dtype: "bits")
        def snapshot(harness, torch, dist):
            seen.append(int(os.environ[sweep.MODE_ENV]))
            return {"all_remote": tensor}, {"checked": True}
        def validate(harness, torch, dist):
            sweep.accuracy._snapshot_and_check_oracles(harness, torch, dist)
            return {"enabled": True}
        torch = argparse.Namespace(equal=lambda a, b: a == b, int16="int16")
        with patch.dict(os.environ, {sweep.MODE_ENV: "1"}), \
             patch.object(sweep.accuracy, "_snapshot_and_check_oracles", side_effect=snapshot), \
             patch.object(sweep.accuracy, "_run_payload_epoch_validation", side_effect=validate), \
             patch.object(sweep.accuracy, "_collective_transition_check"):
            result = sweep._payload_with_mode0_control(None, 1, torch, None)
            self.assertEqual(os.environ[sweep.MODE_ENV], "1")
        self.assertEqual(seen, [0, 1])
        self.assertTrue(result["same_context_mode0_per_epoch_bitwise"])

    def test_control_drift_is_reported_without_correction(self):
        def record(mode, value):
            return {"single_combine_context": mode, "timing": {"routes": {
                route: {"max_rank_per_replay_us": {"p50": value},
                        "isolated": {"max_rank_per_replay_us": {"p50": value - 10}}}
                for route in sweep.accuracy.MATCHED_BENCHMARK_ROUTES}}}
        self.assertFalse(sweep._drift_record([record(0, 100), record(1, 90)])["enabled"])
        drift = sweep._drift_record([record(0, 100), record(1, 90), record(0, 110)])
        self.assertTrue(drift["enabled"])
        self.assertFalse(drift["automatic_drift_correction"])
        self.assertEqual(drift["routes"]["half_remote"]["balanced"]["last_minus_first_us"], 10)

    def test_source_identity_and_success_gate_are_not_weakened(self):
        source = Path(sweep.__file__).read_text()
        self.assertIn("tests/bench_mega_moe_single_combine_context.py", sweep.SOURCE_FILES)
        self.assertIn("tests/test_mega_moe_accuracy.py", sweep.SOURCE_FILES)
        self.assertLess(source.index('accuracy._worker(int(os.environ["LOCAL_RANK"])'),
                        source.index('result.update(status="passed"'))
        self.assertIn('"context_mapping_and_barrier_participation_changed": True', source)
        self.assertIn('"dense_rank_distinct_gate_is_separate": True', source)
        self.assertIn('"same_context_mode0_bitwise": True', source)


if __name__ == "__main__":
    unittest.main()
