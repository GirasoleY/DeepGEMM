"""CPU-only contracts for dense numerical and zero-error transport gates.

These tests check acceptance policy, not GPU math or communication correctness.
The separate dense runner must still execute the full two-host device gates.
"""

import math
import inspect
import os
import unittest
from unittest.mock import patch

import test_mega_moe_dense_accuracy as dense


class TestDenseAccuracyPolicy(unittest.TestCase):
    def test_combine_mode_is_collective_canonical_boolean(self):
        for value in ("0", "1"):
            self.assertEqual(dense.validate_combine_modes([value] * 16), int(value))
        for values in ([], ["0"] * 8 + ["1"] * 8, ["01"] * 16, ["1 "] * 16,
                       ["2"] * 16, [True] * 16, [1] * 16):
            with self.assertRaises(ValueError):
                dense.validate_combine_modes(values)

    def test_combine_baseline_changes_only_combine_with_dispatch_and_sc_fixed(self):
        self.assertTrue(dense.baseline_control_required(0, 1, 1, 1, 1))
        candidate = {dense.EXPERT_WIDTH_ENV: "0", dense.BARRIER_WIDTH_ENV: "1",
                     dense.SINGLE_CONTEXT_ENV: "1", dense.DISPATCH_OVERLAP_ENV: "1",
                     dense.COMBINE_OVERLAP_ENV: "1"}
        baseline = dense.baseline_control_environment(1, 1)
        self.assertEqual(baseline, {**candidate, dense.COMBINE_OVERLAP_ENV: "0"})
        for single, dispatch in ((0, 0), (0, 1), (1, 0)):
            with self.assertRaises(ValueError):
                dense.validate_experiment_combination(0, 1, single, dispatch, 1)
        with patch.dict(os.environ, candidate, clear=True):
            with self.assertRaisesRegex(RuntimeError, "capture"):
                with patch.dict(os.environ, baseline):
                    self.assertEqual(dict(os.environ), baseline)
                    raise RuntimeError("capture")
            self.assertEqual(dict(os.environ), candidate)

    def test_disabled_combine_retains_dispatch_axis_and_explicit_zero(self):
        baseline = dense.baseline_control_environment(1, 0)
        self.assertEqual(baseline[dense.COMBINE_OVERLAP_ENV], "0")
        self.assertEqual(baseline[dense.DISPATCH_OVERLAP_ENV], "0")
        self.assertEqual(baseline[dense.SINGLE_CONTEXT_ENV], "1")
        self.assertEqual(dense.baseline_control_environment(0, 0)[dense.SINGLE_CONTEXT_ENV], "0")

    def test_combine_preflight_and_labels_precede_allocation_without_runtime_claim(self):
        source = inspect.getsource(dense.worker)
        self.assertLess(source.index("validate_combine_modes("), source.index("dg.get_symm_buffer_for_mega_moe("))
        self.assertIn('"combine_overlap_requested_raw": candidate_environment[COMBINE_OVERLAP_ENV]', source)
        self.assertIn('"combine0-dispatch1-sc1-control/"', source)
        self.assertIn('"combine_overlap_effective_policy_not_device_observation"', source)
        self.assertIn('"physical_overlap_measured": False', source)
        self.assertIn('"producer_target": "ceil(actual expert assignments / actual BM) * (H / BN)"', source)
        self.assertIn("saved dispatch source/expert prefixes", source)
        self.assertIn("dynamically select any ready expert", source)
        self.assertIn('"tail_bytes": 2240', source)
        self.assertIn('"required_scratch_bytes": 59584', source)
        self.assertIn('"byte_capacity_depends_on_bm": False', source)
        self.assertIn('"compute_hints_tiling_sm_count_and_math_changed": False', source)

    def test_source_manifest_includes_changed_expert_ready_layout(self):
        source = inspect.getsource(dense.source_fingerprint)
        self.assertIn('"deep_gemm/include/deep_gemm/layout/mega_moe.cuh"', source)
        self.assertIn("not loaded-binary attestation", source)

    def test_dispatch_mode_requires_collective_canonical_boolean(self):
        for value in ("0", "1"):
            self.assertEqual(dense.validate_dispatch_modes([value] * 16), int(value))
        for values in ([], ["0"] * 8 + ["1"] * 8, ["01"] * 16, ["2"] * 16,
                       [True] * 16, [1] * 16):
            with self.assertRaises(ValueError):
                dense.validate_dispatch_modes(values)

    def test_dispatch_baseline_changes_only_dispatch_and_requires_combine_one(self):
        self.assertTrue(dense.baseline_control_required(0, 1, 1, 1))
        baseline = dense.baseline_control_environment(1)
        self.assertEqual(baseline[dense.SINGLE_CONTEXT_ENV], "1")
        self.assertEqual(baseline[dense.DISPATCH_OVERLAP_ENV], "0")
        for single_context in (0, 2):
            with self.assertRaises(ValueError):
                dense.validate_experiment_combination(0, 1, single_context, 1)
        self.assertEqual(dense.baseline_control_environment(0)[dense.SINGLE_CONTEXT_ENV], "0")
        self.assertFalse(dense.baseline_control_required(0, 1, 0, 0))

    def test_dispatch_capture_scope_restores_candidate_flags_even_on_failure(self):
        candidate = {dense.EXPERT_WIDTH_ENV: "0", dense.BARRIER_WIDTH_ENV: "1",
                     dense.SINGLE_CONTEXT_ENV: "1", dense.DISPATCH_OVERLAP_ENV: "1"}
        with patch.dict(os.environ, candidate, clear=True):
            for fail in (False, True):
                try:
                    with patch.dict(os.environ, dense.baseline_control_environment(1)):
                        self.assertEqual(os.environ[dense.SINGLE_CONTEXT_ENV], "1")
                        self.assertEqual(os.environ[dense.DISPATCH_OVERLAP_ENV], "0")
                        if fail:
                            raise RuntimeError("capture failed")
                except RuntimeError:
                    self.assertTrue(fail)
                self.assertEqual(dict(os.environ), candidate)

    def test_dispatch_source_and_labels_are_explicit_without_numeric_gate_changes(self):
        source = inspect.getsource(dense.worker)
        self.assertLess(source.index("validate_dispatch_modes("), source.index("dg.get_symm_buffer_for_mega_moe("))
        self.assertIn('"dispatch_overlap_requested_raw": candidate_environment[DISPATCH_OVERLAP_ENV]', source)
        self.assertIn('"dispatch0-combine1-control/"', source)
        self.assertIn('"baseline_control_environment": baseline_environment', source)
        self.assertIn('"candidate_transport_environment": candidate_environment', source)
        self.assertIn('"dense-source-unchanged"', source)
        self.assertIn('"source": source', source)

    def test_relative_and_elementwise_gates_are_both_required(self):
        self.assertEqual(dense.numerical_gate_errors(True, 0.001, 0.0, 0.001), [])
        self.assertTrue(dense.numerical_gate_errors(True, 0.00101, 0.0, 0.001))
        self.assertTrue(dense.numerical_gate_errors(True, 0.00001, 0.00001, 0.001))

    def test_nonfinite_metrics_cannot_pass(self):
        self.assertTrue(dense.numerical_gate_errors(False, 0.0, 0.0, 0.001))
        for value in (math.inf, -math.inf, math.nan):
            self.assertTrue(dense.numerical_gate_errors(True, value, 0.0, 0.001))
            self.assertTrue(dense.numerical_gate_errors(True, 0.0, value, 0.001))

    def test_cancellation_uses_contribution_scale_not_only_final_output(self):
        # Exact BF16 contributions 64 + (-63.75) produce 0.25. Moving the
        # first contribution by one of its ULPs changes the final sum by 0.5,
        # far more than a final-output ULP. This is the cancellation case
        # exposed by the independent FP32 GEMM oracle on the device baseline.
        reference = 64.0 - 63.75
        candidate = 64.5 - 63.75
        route_budget = 0.5 + 0.25
        final_interval_spacing = 2.0 ** -7  # max |sum| <= 1.0
        budget = route_budget + final_interval_spacing
        error = abs(candidate - reference)
        self.assertGreater(error, 0.0625)
        self.assertGreater(error, 2.0 ** -9 + 0.0625)
        self.assertEqual(dense.numerical_gate_errors(
            True, 0.0001, max(0.0, error - budget), 0.001), [])
        self.assertTrue(dense.numerical_gate_errors(
            True, 0.0001, max(0.0, 1.0 - budget), 0.001))

    def test_numerical_budget_never_licenses_transport_error(self):
        self.assertEqual(dense.numerical_gate_errors(True, 0.0001, 0.0, 0.001), [])
        self.assertEqual(dense.transport_gate_errors(True, True), [])
        self.assertEqual(len(dense.transport_gate_errors(False, True)), 1)
        self.assertEqual(len(dense.transport_gate_errors(True, False)), 1)
        self.assertEqual(len(dense.transport_gate_errors(False, False)), 2)

    def test_width_configuration_cannot_diverge_across_ranks(self):
        for width in (0,):
            self.assertEqual(dense.validate_expert_widths([str(width)] * 16), width)
        for widths in (["0"] * 8 + ["8"] * 8, ["3"] * 16,
                       *([str(width)] * 16 for width in (1, 2, 4, 8, 56)),
                       ["00"] * 16, ["nonnumeric"] * 16, []):
            with self.assertRaises(ValueError):
                dense.validate_expert_widths(widths)

    def test_barrier_configuration_cannot_diverge_or_select_zero(self):
        for width in (1,):
            self.assertEqual(dense.validate_barrier_widths([str(width)] * 16), width)
        for widths in (["1"] * 8 + ["8"] * 8, ["0"] * 16,
                       ["2"] * 16, ["4"] * 16, ["8"] * 16,
                       ["01"] * 16, ["56"] * 16, ["nonnumeric"] * 16, []):
            with self.assertRaises(ValueError):
                dense.validate_barrier_widths(widths)

    def test_only_single_context_experiment_requires_strict_baseline(self):
        self.assertFalse(dense.baseline_control_required(0, 1))
        for expert_width in (0, 1, 2, 4, 8, 56):
            for barrier_width in (1, 2, 4, 8):
                if (expert_width, barrier_width) != (0, 1):
                    with self.assertRaises(ValueError):
                        dense.baseline_control_required(expert_width, barrier_width)

    def test_single_context_configuration_is_collective_boolean(self):
        for mode in (0, 1):
            self.assertEqual(dense.validate_single_context_modes([str(mode)] * 16), mode)
        for values in (["0"] * 8 + ["1"] * 8, ["2"] * 16,
                       ["-1"] * 16, ["01"] * 16, ["nonnumeric"] * 16, []):
            with self.assertRaises(ValueError):
                dense.validate_single_context_modes(values)

    def test_single_context_alone_requires_unchanged_all_context_control(self):
        self.assertTrue(dense.baseline_control_required(0, 1, 1))
        self.assertEqual(dense.baseline_control_environment()[dense.SINGLE_CONTEXT_ENV], "0")

    def test_single_context_first_prototype_rejects_mixed_experiments(self):
        dense.validate_experiment_combination(0, 1, 1)
        dense.validate_experiment_combination(0, 1, 0)
        for mode in (0, 1):
            for expert_width, barrier_width in ((1, 1), (0, 2), (8, 8)):
                with self.assertRaises(ValueError):
                    dense.validate_experiment_combination(expert_width, barrier_width, mode)

    def test_baseline_capture_restores_all_candidate_flags_including_failure(self):
        candidates = ({dense.EXPERT_WIDTH_ENV: "0", dense.BARRIER_WIDTH_ENV: "1", dense.SINGLE_CONTEXT_ENV: "0"},
                      {dense.EXPERT_WIDTH_ENV: "0", dense.BARRIER_WIDTH_ENV: "1", dense.SINGLE_CONTEXT_ENV: "1"})
        for candidate in candidates:
            with patch.dict(os.environ, candidate):
                with patch.dict(os.environ, dense.baseline_control_environment()):
                    self.assertEqual(os.environ[dense.EXPERT_WIDTH_ENV], "0")
                    self.assertEqual(os.environ[dense.BARRIER_WIDTH_ENV], "1")
                    self.assertEqual(os.environ[dense.SINGLE_CONTEXT_ENV], "0")
                self.assertEqual({name: os.environ[name] for name in candidate}, candidate)
                with self.assertRaisesRegex(RuntimeError, "capture failed"):
                    with patch.dict(os.environ, dense.baseline_control_environment()):
                        raise RuntimeError("capture failed")
                self.assertEqual({name: os.environ[name] for name in candidate}, candidate)


if __name__ == "__main__":
    unittest.main()
