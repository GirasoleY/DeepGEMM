"""CPU checks for comparison scope and collective numerical failure gates."""

import contextlib
import io
import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import bench_deepep_trtllm_isolated as comparison


class FakeDist:
    def __init__(self, records):
        self.records = records

    def get_world_size(self):
        return len(self.records)

    def all_gather_object(self, output, local):
        output[:] = self.records


class ComparisonContractTest(unittest.TestCase):
    def test_dispatch_metadata_distinguishes_control_first_without_claiming_measured_overlap(self):
        for raw, family in (("0", "clean_single_combine_context_only"),
                            ("1", "direct_control_first_dispatch")):
            flags = {"DG_MEGAMOE_GIN_DISPATCH_OVERLAP": raw}
            metadata = comparison.dispatch_candidate_metadata(flags)
            self.assertEqual(metadata["candidate_family"], family)
            contract = metadata["dispatch_overlap_contract"]
            self.assertEqual(contract["requested_raw"], raw)
            self.assertEqual(contract["requested"], raw == "1")
            self.assertEqual(contract["separate_payload_terminal_acquire_before_payload_reads"], raw == "1")
            self.assertFalse(contract["compute_hints_tiling_sm_count_and_math_changed"])
            self.assertIn("not a measured-overlap observation", contract["basis"])
            self.assertEqual(flags, {"DG_MEGAMOE_GIN_DISPATCH_OVERLAP": raw})
        for invalid in ("01", "2", 1, True):
            with self.assertRaises(ValueError):
                comparison.dispatch_candidate_metadata({"DG_MEGAMOE_GIN_DISPATCH_OVERLAP": invalid})

    def test_dispatch_flag_is_gathered_and_retained_before_comparator_construction(self):
        import inspect
        source = inspect.getsource(comparison.main)
        self.assertLess(source.index('"DG_MEGAMOE_GIN_DISPATCH_OVERLAP"'),
                        source.index("dist.all_gather_object(rank_flags, flags)"))
        self.assertLess(source.index("dist.all_gather_object(rank_flags, flags)"),
                        source.index("comparator = DeepEPTRTLLM"))
        self.assertIn('"megamoe_flags": flags', source)
        self.assertIn("**dispatch_candidate_metadata(flags)", source)

    def test_clean_candidate_rejects_retired_knobs_even_when_context_mode_is_off(self):
        for name, invalid in (("DG_MEGAMOE_GIN_COMBINE_EXPERTS_PER_WAVE", "8"),
                              ("DG_MEGAMOE_GIN_COMBINE_BARRIER_WARPS", "8")):
            with patch.dict(os.environ, {name: invalid,
                                         "DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT": "0"}, clear=True):
                with self.assertRaisesRegex(ValueError, "retired"):
                    comparison.validate_clean_experiment_environment()
        with patch.dict(os.environ, {}, clear=True):
            comparison.validate_clean_experiment_environment()

    def parse(self, *options):
        with patch.object(sys, "argv", ["bench", "--output", "unused.json", *options]):
            return comparison.parse_args()

    def test_requested_token_count_is_never_replaced_by_default(self):
        for options, expected in ((["--decode-mns=10"], 40),
                                  (["--decode-mns", "12"], 48),
                                  (["--num-tokens=32"], 32)):
            controls, args = self.parse(*options)
            self.assertEqual(args.num_tokens, expected)
            self.assertEqual(controls.deepep_capacity, expected)
            self.assertEqual(args.benchmark_replays, 0)
            self.assertEqual(args.gin_outbox_depth, 64)
            self.assertEqual(args.gin_combine_chunk_bytes, 7168)

    def test_nonfinite_threshold_and_single_sample_are_rejected(self):
        for option in ("--comparison-rel-l2=nan", "--comparison-max-abs=inf",
                       "--comparison-replays=1", "--component-replays=1"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.parse(option)

    def test_capacity_change_is_explicit_and_cannot_drop_active_tokens(self):
        controls, _ = self.parse("--decode-mns=10", "--deepep-capacity=384")
        self.assertEqual(controls.deepep_capacity, 384)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.parse("--decode-mns=10", "--deepep-capacity=32")

    def test_default_dispatch_bucket_preserves_serving_rounding_and_capacity_clamp(self):
        for mns, tokens in ((8, 32), (10, 40), (12, 48)):
            for capacity in (tokens, 384):
                controls, _ = self.parse(f"--decode-mns={mns}", f"--deepep-capacity={capacity}")
                evidence = controls.dispatch_bucket_evidence
                self.assertIsNone(controls.deepep_dispatch_bucket)
                self.assertIsNone(evidence["dispatch_bucket_requested"])
                self.assertEqual(evidence["dispatch_bucket"], min(1 << (tokens - 1).bit_length(), capacity))
                self.assertEqual(evidence["dispatch_bucket_policy"], "serving_power_of_two_clamped_to_capacity")

    def test_explicit_exact_buckets_keep_capacity_and_numerical_gates_unchanged(self):
        for mns, tokens in ((10, 40), (12, 48)):
            controls, args = self.parse(f"--decode-mns={mns}", "--deepep-capacity=384",
                                       f"--deepep-dispatch-bucket={tokens}")
            self.assertEqual((args.num_tokens, controls.deepep_capacity), (tokens, 384))
            self.assertEqual(controls.dispatch_bucket_evidence, {
                "dispatch_bucket": tokens, "dispatch_bucket_requested": tokens,
                "serving_default_dispatch_bucket": 64,
                "dispatch_bucket_policy": "explicit_exact_active_tokens",
                "dispatch_padding_tokens": 0,
            })
            self.assertEqual((controls.comparison_rel_l2, controls.comparison_max_abs), (0.02, 0.02))
            self.assertIsNone(controls.deepep_num_sms)
            self.assertIsNone(controls.deepep_num_qps)

    def test_explicit_padded_bucket_is_labeled_and_out_of_range_rejected(self):
        controls, _ = self.parse("--decode-mns=10", "--deepep-capacity=384", "--deepep-dispatch-bucket=64")
        self.assertEqual(controls.dispatch_bucket_evidence["dispatch_bucket_policy"], "explicit_padded")
        self.assertEqual(controls.dispatch_bucket_evidence["dispatch_padding_tokens"], 24)
        for invalid in (0, 39, 385):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.parse("--decode-mns=10", "--deepep-capacity=384", f"--deepep-dispatch-bucket={invalid}")

    def test_recipe_profile_is_opt_in_and_does_not_change_timing_controls(self):
        defaults, _ = self.parse()
        profile, _ = self.parse("--profile-recipe")
        self.assertFalse(defaults.profile_recipe)
        self.assertTrue(profile.profile_recipe)
        self.assertEqual(profile.comparison_replays, defaults.comparison_replays)
        self.assertEqual(profile.comparison_max_abs, 0.02)

    def test_ablation_or_historical_perf_mode_cannot_be_mislabeled_comparison(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parse("--benchmark-replays=204")

    def test_remote_rank_error_rejects_whole_comparison(self):
        good = {"finite": True, "relative_l2": 0.001, "max_abs": 0.001}
        bad = {"finite": True, "relative_l2": 0.021, "max_abs": 0.001}
        controls = SimpleNamespace(comparison_rel_l2=0.02, comparison_max_abs=0.02)
        with patch.object(comparison, "_error_record", return_value=good):
            with self.assertRaisesRegex(AssertionError, "rank 1"):
                comparison._validate_output(None, None, "test", controls, None,
                                            FakeDist([good, bad]))

    def test_absolute_error_and_nonfinite_are_independent_failures(self):
        controls = SimpleNamespace(comparison_rel_l2=0.02, comparison_max_abs=0.02)
        for bad in ({"finite": True, "relative_l2": 0, "max_abs": 0.021},
                    {"finite": False, "relative_l2": 0, "max_abs": 0},
                    {"finite": True, "relative_l2": float("nan"), "max_abs": 0}):
            with patch.object(comparison, "_error_record", return_value=bad):
                with self.assertRaises(AssertionError):
                    comparison._validate_output(None, None, "test", controls, None,
                                                FakeDist([bad]))

    def test_pass_report_retains_worst_rank_and_raw_errors(self):
        good = {"finite": True, "relative_l2": 0.002, "max_abs": 0.003}
        controls = SimpleNamespace(comparison_rel_l2=0.02, comparison_max_abs=0.02)
        with patch.object(comparison, "_error_record", return_value=good):
            result = comparison._validate_output(None, None, "test", controls, None,
                                                  FakeDist([good]))
        self.assertEqual(result["max_relative_l2"], 0.002)
        self.assertEqual(result["per_rank"], [good])

    def test_native_recipe_drift_reports_failed_old_absolute_rule_without_widening_it(self):
        drift = {"finite": True, "relative_l2": 0.0063, "max_abs": 0.03125}
        controls = SimpleNamespace(comparison_rel_l2=0.02, comparison_max_abs=0.02)
        with patch.object(comparison, "_error_record", return_value=drift):
            result = comparison._validate_backend_drift(None, None, "test", controls, None,
                                                         FakeDist([drift]))
            with self.assertRaises(AssertionError):
                comparison._validate_output(None, None, "own-recipe", controls, None,
                                              FakeDist([drift]))
        self.assertEqual(result["max_abs"], 0.03125)
        self.assertFalse(result["previous_absolute_rule"]["passed"])
        self.assertEqual(result["previous_absolute_rule"]["limit"], 0.02)

    def test_native_recipe_drift_still_rejects_bad_relative_l2_or_nonfinite(self):
        controls = SimpleNamespace(comparison_rel_l2=0.02, comparison_max_abs=0.02)
        for record in ({"finite": True, "relative_l2": 0.021, "max_abs": 0.001},
                       {"finite": True, "relative_l2": 0.001, "max_abs": float("nan")},
                       {"finite": False, "relative_l2": 0.001, "max_abs": 0.001}):
            with patch.object(comparison, "_error_record", return_value=record):
                with self.assertRaises(AssertionError):
                    comparison._validate_backend_drift(None, None, "test", controls, None,
                                                         FakeDist([record]))

    def test_unexpanded_receive_rows_count_every_local_route(self):
        record = comparison._receipt_histogram(
            [[112, -1, 114], [-1, 112, 113], [-1, -1, -1]],
            valid_rows=2, owner_offset=112, local_experts=3, expected=[2, 1, 1])
        self.assertEqual(record["received_routes"], 4)
        self.assertEqual(record["valid_received_rows"], 2)

    def test_receive_padding_cannot_be_treated_as_real_routes(self):
        with self.assertRaisesRegex(AssertionError, "unmasked padding"):
            comparison._receipt_histogram([[112], [112]], 1, 112, 1, [1])

    def test_global_id_in_world_but_outside_owner_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "nonlocal expert"):
            comparison._receipt_histogram([[115]], 1, 112, 3, [0, 0, 1])

    def test_missing_route_or_bad_prefix_cannot_pass_numeric_equivalence(self):
        with self.assertRaisesRegex(AssertionError, "histogram mismatch"):
            comparison._receipt_histogram([[112, -1]], 1, 112, 2, [1, 1])
        with self.assertRaisesRegex(AssertionError, "exceeds allocation"):
            comparison._receipt_histogram([[112]], 2, 112, 1, [1])

    def test_all_masked_receipt_has_exact_zero_counts(self):
        result = comparison._receipt_histogram([[-1, -1]], 0, 112, 2, [0, 0])
        self.assertEqual(result["expert_histogram"], [0, 0])

    def test_graph_executables_retire_even_when_python_references_survive(self):
        from unittest.mock import Mock
        graph = Mock()
        graphs = {"backend": graph}
        comparison._retire_graphs(graphs)
        graph.reset.assert_called_once()
        self.assertEqual(graphs, {})

    def test_cleanup_failure_does_not_mask_original_numerical_failure(self):
        from unittest.mock import Mock
        graph = Mock()
        graph.reset.side_effect = RuntimeError("reset failed")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(AssertionError, "oracle failed"):
                try:
                    raise AssertionError("oracle failed")
                finally:
                    comparison._retire_graphs({"backend": graph})


if __name__ == "__main__":
    unittest.main()
