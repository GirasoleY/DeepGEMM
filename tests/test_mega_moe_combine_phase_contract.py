"""Host-only phase metadata/analysis checks; no physical-overlap assertion."""

import copy
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import capture_mega_moe_phases as capture
from analyze_mega_moe_phases import summarize_capture, summarize_sample


DISPATCH = "DG_MEGAMOE_GIN_DISPATCH_OVERLAP"
COMBINE = "DG_MEGAMOE_GIN_COMBINE_OVERLAP"
SINGLE = capture.accuracy.GIN_SINGLE_COMBINE_CONTEXT_ENV
PREPACK = "DG_MEGAMOE_GIN_PRECONSENSUS_PACK"
COOP = "DG_MEGAMOE_GIN_COOP_DIRECT_PACK"


def args():
    return SimpleNamespace(
        gin_completion_batch=8, gin_combine_issue_wave=8,
        gin_outbox_depth=64, gin_combine_chunk_bytes=7168,
        gin_bulk_combine=True, gin_direct_dispatch=True,
        gin_active_fast_path=True, num_tokens=32)


def candidate_env():
    return {DISPATCH: "1", COMBINE: "1", SINGLE: "1", PREPACK: "1", COOP: "1"}


def fake_dist(mutate=None):
    calls = []

    def gather(destination, local):
        calls.append(copy.deepcopy(local))
        destination[:] = [copy.deepcopy(local), copy.deepcopy(local)]
        if mutate is not None:
            mutate(destination[1])

    return SimpleNamespace(get_world_size=lambda: 2, all_gather_object=gather), calls


def marker_sample(issue=20000, exit_stamp=30000, *, offset=0, rank=0):
    rows = {"0": {"0": 1000, "64": issue, "72": 40000, "80": 50000},
            "1": {"0": 1100, "53": exit_stamp}}
    return {"rank": rank, "replay": 0, "sm_markers_ns": {
        sm: {slot: value + offset for slot, value in row.items() if value is not None}
        for sm, row in rows.items()
    }}


class CombinePhaseContract(unittest.TestCase):
    def test_default_and_candidate_policies_are_not_runtime_attestation(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            baseline = capture._kernel_configuration(args())
        self.assertEqual(baseline["combine_schedule"], "post_compute_full_packet")
        self.assertFalse(baseline["dispatch_overlap_requested"])
        self.assertFalse(baseline["combine_overlap_requested"])
        with mock.patch.dict(os.environ, candidate_env(), clear=True):
            result = capture._kernel_configuration(args())
        self.assertEqual(result["combine_schedule"], "peer_parallel_ready_expert_spans_then_late_header")
        self.assertTrue(result["combine_ready_expert_policy"]["no_token_metadata_rescan"])
        self.assertEqual(result["combine_ready_expert_policy"]["metadata_storage_bytes"], 2304)
        self.assertIn("peer_independent", result["combine_ready_expert_policy"]["selection"])
        self.assertTrue(result["combine_ready_expert_policy"]["nonempty_masks_saved_during_existing_dispatch_prefix_scan"])
        self.assertTrue(result["combine_ready_expert_policy"]["payload_span_granularity_unchanged_one_nonempty_peer_expert"])
        self.assertFalse(result["combine_ready_expert_policy"]["policy_is_device_observation"])
        self.assertTrue(result["combine_schedule_is_requested_policy_not_device_observation"])
        self.assertEqual(result["combine_overlap_effective_policy"][
            "bulk_direct_remote_but_scratch_alias_insufficient"], "post_compute_full_packet")
        self.assertEqual(result["combine_overlap_effective_policy"]["all_local"], "unchanged_local_path")
        self.assertEqual(result["phase_marker_semantics"]["32_39"],
                         "receiver_control_terminal_acquired_not_payload")
        self.assertIn("includes_late_header", result["phase_marker_semantics"]["72_79"])
        self.assertIn("not_last_payload_time", result["phase_marker_semantics"]["80_87"])
        self.assertEqual(result["environment"][DISPATCH], "1")
        self.assertEqual(result["environment"][COMBINE], "1")

    def test_invalid_raw_values_are_rejected_only_after_collective(self):
        for name in (DISPATCH, COMBINE):
            for invalid in ("", "01", " 1", "1x", "-1", "2"):
                with self.subTest(name=name, raw=invalid):
                    dist, calls = fake_dist()
                    environment = candidate_env() | {name: invalid}
                    with mock.patch.dict(os.environ, environment, clear=True):
                        with self.assertRaisesRegex(ValueError, "exactly 0 or 1"):
                            capture._collective_kernel_configuration(args(), SimpleNamespace(), dist)
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(calls[0]["environment"][name], invalid)

    def test_mismatched_mode_or_shape_fails_before_local_parsing(self):
        for mutate in (
            lambda record: record["environment"].update({COMBINE: "0"}),
            lambda record: record["environment"].update({DISPATCH: "0"}),
            lambda record: record["setup"].update({"num_tokens": 40}),
        ):
            dist, calls = fake_dist(mutate)
            with mock.patch.dict(os.environ, candidate_env(), clear=True), mock.patch.object(
                capture, "_kernel_configuration", side_effect=AssertionError("parsed too soon")
            ):
                with self.assertRaisesRegex(RuntimeError, "differs across ranks"):
                    capture._collective_kernel_configuration(args(), SimpleNamespace(), dist)
            self.assertEqual(len(calls), 1)

    def test_candidate_prerequisites_are_enforced_and_setup_retained(self):
        for name in (SINGLE, DISPATCH, PREPACK, COOP):
            with mock.patch.dict(os.environ, candidate_env() | {name: "0"}, clear=True):
                with self.assertRaises(ValueError):
                    capture._kernel_configuration(args())
        for field in ("gin_bulk_combine", "gin_direct_dispatch"):
            configuration = args()
            setattr(configuration, field, False)
            with mock.patch.dict(os.environ, candidate_env(), clear=True):
                with self.assertRaisesRegex(ValueError, "requires bulk/direct"):
                    capture._kernel_configuration(configuration)
        dist, calls = fake_dist()
        with mock.patch.dict(os.environ, candidate_env(), clear=True):
            result = capture._collective_kernel_configuration(
                args(), SimpleNamespace(diagnostic_level=2, diagnostic_replays=8), dist)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["collectively_validated_setup"]["num_tokens"], 32)
        self.assertEqual(result["collectively_validated_diagnostic_settings"]["diagnostic_level"], 2)

    def test_signed_interval_uses_first_issue_and_last_exit_only_on_same_gpu(self):
        sample = marker_sample()
        sample["sm_markers_ns"]["2"] = {"0": 1200, "65": 23000, "53": 35000}
        first = summarize_sample(sample)
        name = "first_combine_issue_to_last_epilogue_loop_exit_signed_us"
        self.assertEqual(first["metrics_us"][name], 15.0)
        self.assertTrue(first["first_combine_issue_before_last_epilogue_loop_exit"])
        shifted = copy.deepcopy(sample)
        shifted["rank"] = 15
        shifted["sm_markers_ns"] = {
            sm: {slot: stamp + 987654321000 for slot, stamp in row.items()}
            for sm, row in sample["sm_markers_ns"].items()}
        self.assertEqual(first["metrics_us"], summarize_sample(shifted)["metrics_us"])
        self.assertEqual(summarize_sample(marker_sample(issue=40000))["metrics_us"][name], -10.0)
        self.assertFalse(summarize_sample(marker_sample(issue=30000))[
            "first_combine_issue_before_last_epilogue_loop_exit"])
        # Moving late header/flush observations cannot move the new interval.
        sample["sm_markers_ns"]["0"].update({"72": 9000000, "80": 10000000})
        self.assertEqual(summarize_sample(sample)["metrics_us"][name], 15.0)

    def test_missing_markers_are_unknown_and_counts_do_not_cross_gpu_clocks(self):
        result = summarize_capture({
            "shape": {"tokens_per_rank": 32}, "diagnostic_level": 2,
            "routes": {"half_remote": [
                marker_sample(), marker_sample(issue=40000, offset=999999999, rank=1),
                marker_sample(issue=30000, rank=2), marker_sample(issue=None, rank=3),
                marker_sample(exit_stamp=None, rank=4),
            ]},
        })
        counts = result["routes"]["half_remote"]["combine_issue_epilogue_observation_counts"]
        self.assertEqual(counts, {
            "samples_with_both_marker_families": 3,
            "issue_before_last_epilogue_loop_exit": 1,
            "issue_at_or_after_last_epilogue_loop_exit": 2,
            "samples_missing_a_marker_family": 2,
        })
        self.assertFalse(result["performance_claim"])
        self.assertTrue(any("not physical IB or MMA overlap proof" in note for note in result["notes"]))
        self.assertTrue(any("Do not infer last-payload timing" in note for note in result["notes"]))

    def test_runtime_validation_precedes_allocation_and_success_follows_teardown(self):
        source = Path(capture.__file__).read_text()
        begin = source.index("def initialize_then_validate(")
        end = source.index("def allocate(", begin)
        body = source[begin:end]
        self.assertLess(body.index("original_init_distributed("),
                        body.index("_collective_kernel_configuration("))
        self.assertIn('patch.object(accuracy, "_init_distributed", initialize_then_validate)', source)
        self.assertIn("phase kernel configuration changed before capture", source)
        self.assertIn("phase kernel configuration changed during capture", source)
        self.assertLess(source.index("accuracy._worker("),
                        source.index('["accuracy_and_teardown_passed"] = True'))


if __name__ == "__main__":
    unittest.main()
