"""CPU-only contracts for dense numerical and zero-error transport gates.

These tests check acceptance policy, not GPU math or communication correctness.
The separate dense runner must still execute the full two-host device gates.
"""

import math
import os
import unittest
from unittest.mock import patch

import test_mega_moe_dense_accuracy as dense


class TestDenseAccuracyPolicy(unittest.TestCase):
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
