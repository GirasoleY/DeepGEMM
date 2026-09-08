"""Host-only checks for calibration ordering, timing bounds and JIT levels."""

import os
from types import SimpleNamespace
import unittest
from unittest import mock

import capture_mega_moe_phases as capture


class DiagnosticCalibrationTest(unittest.TestCase):
    def test_single_context_metadata_preserves_input_and_fallback_contexts(self):
        args = SimpleNamespace(gin_completion_batch=8, gin_combine_issue_wave=8,
                               gin_outbox_depth=64, gin_combine_chunk_bytes=7168)
        env = capture.accuracy.GIN_SINGLE_COMBINE_CONTEXT_ENV
        with mock.patch.dict(os.environ, {env: "1"}, clear=True):
            record = capture._kernel_configuration(args)
        self.assertTrue(record["single_combine_context_requested"])
        self.assertEqual(record["allocated_gin_context_count"], 9)
        self.assertEqual(record["input_data_contexts_unchanged"], list(range(1, 9)))
        policy = record["combine_barrier_contexts_effective_policy"]
        self.assertEqual(policy["bulk_and_direct_eligible_remote"], [1])
        self.assertEqual(policy["remote_ineligible"], list(range(9)))
        self.assertEqual(policy["all_local"], [])
        for name, value in ((env, "2"), ("DG_MEGAMOE_GIN_COMBINE_EXPERTS_PER_WAVE", "8"),
                            ("DG_MEGAMOE_GIN_COMBINE_BARRIER_WARPS", "4")):
            with mock.patch.dict(os.environ, {env: "1", name: value}, clear=True), self.assertRaises(ValueError):
                capture._kernel_configuration(args)

    def test_barrier_requested_width_does_not_mislabel_local_or_fallback(self):
        args = SimpleNamespace(gin_completion_batch=8, gin_combine_issue_wave=8,
                               gin_outbox_depth=64, gin_combine_chunk_bytes=7168)
        env = "DG_MEGAMOE_GIN_COMBINE_BARRIER_WARPS"
        for requested in (1,):
            with mock.patch.dict(os.environ, {env: str(requested)}):
                record = capture._kernel_configuration(args)
            self.assertEqual(record["combine_barrier_warps_requested"], requested)
            policy = record["combine_barrier_warps_effective_policy"]
            self.assertEqual(policy["bulk_and_direct_eligible_remote_with_sufficient_epilogue_warps"], requested)
            self.assertEqual(policy["remote_ineligible_or_insufficient_epilogue_warps"], 1)
            self.assertEqual(policy["all_local"], 0)
            self.assertFalse(record["combine_barrier_effective_policy_is_device_observation"])
            self.assertEqual(record["environment"][env], str(requested))
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(capture._kernel_configuration(args)["combine_barrier_warps_requested"], 1)
        for invalid in (0, 2, 3, 4, 8, 16, "01"):
            with mock.patch.dict(os.environ, {env: str(invalid)}), self.assertRaisesRegex(
                ValueError, "barrier warp count"
            ):
                capture._kernel_configuration(args)

    def test_retired_expert_waves_rejected_even_in_control_mode(self):
        args = SimpleNamespace(gin_completion_batch=8, gin_combine_issue_wave=8,
                               gin_outbox_depth=64, gin_combine_chunk_bytes=7168)
        for mode in ("0", "1"):
            for width in ("1", "2", "4", "8", "56", "00"):
                with mock.patch.dict(os.environ, {
                        capture.accuracy.GIN_SINGLE_COMBINE_CONTEXT_ENV: mode,
                        "DG_MEGAMOE_GIN_COMBINE_EXPERTS_PER_WAVE": width}, clear=True):
                    with self.assertRaisesRegex(ValueError, "retired"):
                        capture._kernel_configuration(args)
        with mock.patch.dict(os.environ, {"DG_MEGAMOE_GIN_COMBINE_EXPERTS_PER_WAVE": "0"}):
            record = capture._kernel_configuration(args)
        self.assertEqual(record["combine_schedule"], "post_compute_full_packet")
        self.assertEqual(record["combine_wave_count"], 0)
        self.assertFalse(record["retired_expert_wave_and_coop_width_experiments_supported"])

    def test_default_calibration_and_equals_shape_argument(self):
        with mock.patch("sys.argv", ["capture", "--output", "/tmp/unused.json", "--decode-mns=10"]):
            diagnostic, args = capture.parse_args()
        self.assertEqual(diagnostic.calibration_replays, 32)
        self.assertEqual(args.num_tokens, 40)

    def test_single_replay_is_rejected(self):
        with mock.patch("sys.argv", ["capture", "--output", "/tmp/unused.json", "--calibration-replays", "1"]), mock.patch(
            "sys.stderr"
        ), self.assertRaises(SystemExit):
            capture.parse_args()

    def test_disabled_calibration_has_no_side_effects(self):
        result = capture.calibrate_diagnostics(None, None, None, SimpleNamespace(calibration_replays=0), None, None)
        self.assertEqual(result, {"enabled": False, "reason": "explicitly_disabled"})

    def test_exact_levels_are_captured_and_only_launch_is_timed(self):
        state = SimpleNamespace(clock=0.0, marker_level=0, stats=0, route=None)
        captured_levels = []
        measured_checks = []

        class Markers:
            def data_ptr(self):
                return 1234

            def zero_(self):
                state.marker_level = 0
                state.clock += 70.0

            def __getitem__(self, key):
                column = key[1]
                present = state.marker_level > 0 if column == 0 else state.marker_level == 2
                return SimpleNamespace(any=lambda: SimpleNamespace(item=lambda: present))

        class Event:
            def __init__(self, enable_timing):
                self.tick = None

            def record(self):
                self.tick = state.clock

            def elapsed_time(self, end):
                return (end.tick - self.tick) / 1000.0

        markers = Markers()
        cases = {route: SimpleNamespace(expected_local_stats=3, route=route)
                 for route in capture.accuracy.MATCHED_BENCHMARK_ROUTES}
        harness = SimpleNamespace(
            cases=cases, output=SimpleNamespace(fill_=lambda unused: None),
            stats=SimpleNamespace(zero_=lambda: setattr(state, "stats", 0)),
            copy_inputs=lambda case: setattr(state, "route", case.route),
            assert_guards=lambda label: measured_checks.append(label),
        )
        harness.assert_stats = lambda expected, label: self.assertEqual(state.stats, expected, label)

        def capture_graph(*args):
            level = int(os.environ["DG_MEGAMOE_GIN_DIAGNOSTICS"])
            captured_levels.append(level)

            def replay():
                state.marker_level = level
                state.stats += 3
                state.clock += 100.0 + level * 10.0

            return SimpleNamespace(replay=replay)

        def timing(durations, *args):
            return {"local_rank_samples_us": [durations],
                    "max_rank_per_replay_us": capture.accuracy._timing_summary(durations)}

        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None, Event=Event))
        dist = SimpleNamespace(barrier=lambda: setattr(state, "clock", state.clock + 1000.0))
        diagnostic = SimpleNamespace(calibration_replays=2, diagnostic_warmups=1)
        with mock.patch.dict(os.environ, {"DG_MEGAMOE_GIN_DIAGNOSTICS": "1"}), mock.patch.object(
            capture.accuracy, "_capture_launch_only_graph", side_effect=capture_graph
        ), mock.patch.object(capture.accuracy, "_assert_bitwise_equal"), mock.patch.object(
            capture.accuracy, "_distributed_timing_record", side_effect=timing
        ):
            result = capture.calibrate_diagnostics(harness, {route: None for route in cases}, markers, diagnostic, torch, dist)
            self.assertEqual(os.environ["DG_MEGAMOE_GIN_DIAGNOSTICS"], "1")

        self.assertEqual(captured_levels, [0, 1, 2])
        self.assertEqual(len(measured_checks), 18)
        self.assertFalse(result["performance_claim"])
        self.assertTrue(result["same_marker_storage"])
        for route in result["routes"].values():
            for level in (0, 1, 2):
                self.assertEqual(route["levels"][str(level)]["local_rank_samples_us"], [[100.0 + level * 10.0] * 2])
            self.assertAlmostEqual(route["overhead_vs_level0"]["1"]["max_rank_p50_delta_vs_level0_us"], 10.0)
            self.assertAlmostEqual(route["overhead_vs_level0"]["2"]["max_rank_p50_ratio_vs_level0"], 1.2)

    def test_jit_environment_restored_after_capture_failure(self):
        diagnostic = SimpleNamespace(calibration_replays=2)
        with mock.patch.dict(os.environ, {"DG_MEGAMOE_GIN_DIAGNOSTICS": "2"}), mock.patch.object(
            capture.accuracy, "_capture_launch_only_graph", side_effect=RuntimeError("compile failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "compile failed"):
                capture.calibrate_diagnostics(None, None, SimpleNamespace(data_ptr=lambda: 1), diagnostic, None, None)
            self.assertEqual(os.environ["DG_MEGAMOE_GIN_DIAGNOSTICS"], "2")


if __name__ == "__main__":
    unittest.main()
