"""CPU runner policy checks; no claim of GPU accuracy or performance."""

import os
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import bench_mega_moe_dispatch_overlap as sweep


class DispatchSweepContracts(unittest.TestCase):
    def test_defaults_keep_full_control_candidate_control_and_accuracy(self):
        with patch.object(sys, "argv", ["sweep", "--output", "/tmp/not-created.json"]):
            options, args = sweep.parse_args()
        self.assertEqual(options.schedule, [0, 1, 0])
        self.assertEqual(options.benchmark_replays, 204)
        self.assertEqual(args.benchmark_replays, 0)
        self.assertEqual((args.num_tokens, args.num_max_tokens_per_rank), (32, 384))
        self.assertEqual(args.payload_epochs, 3)
        self.assertTrue(options.payload_per_mode and options.transitions_per_mode)
        self.assertFalse(args.fast_math)

    def test_short_timing_does_not_disable_gates_or_change_shape(self):
        with patch.object(sys, "argv", ["sweep", "--output", "/tmp/not-created.json",
                                        "--decode-mns=12", "--benchmark-replays=8"]):
            options, args = sweep.parse_args()
        self.assertEqual((args.num_tokens, options.benchmark_replays), (48, 8))
        for flag in ("--no-payload-per-mode", "--no-transitions-per-mode", "--no-repeat-control"):
            with patch.object(sys, "argv", ["sweep", "--output", "/tmp/not-created.json", flag]):
                with self.assertRaises(ValueError):
                    sweep.parse_args()

    def test_combine_fixed_one_dispatch_starts_zero_and_prepack_preserved(self):
        with patch.dict(os.environ, {}, clear=True):
            sweep.prepare_sweep_environment()
            self.assertEqual(os.environ[sweep.MODE_ENV], "0")
            self.assertEqual(os.environ[sweep.COMBINE_ENV], "1")
            for name in sweep.accuracy.GIN_EXPERIMENT_FLAG_ENVS:
                self.assertEqual(os.environ[name], "1")
            self.assertEqual(os.environ[sweep.common.BARRIER_WIDTH_ENV], "1")
            self.assertEqual(os.environ[sweep.common.EXPERT_WIDTH_ENV], "0")
        with patch.dict(os.environ, {sweep.common.BARRIER_WIDTH_ENV: "8"}, clear=True):
            before = dict(os.environ)
            with self.assertRaises(ValueError):
                sweep.prepare_sweep_environment()
            self.assertEqual(dict(os.environ), before)

    def test_hot_owner_fixture_preserves_distinct_experts_and_fixed_counts(self):
        for tokens in (32, 40, 48):
            for owner in (0, 8):
                counts = [0] * 56
                for source in range(16):
                    for _ in range(tokens):
                        ids = [sweep.hot_assignment(source, slot, owner) for slot in range(16)]
                        self.assertEqual(len(set(ids)), 16)
                        for expert in ids:
                            self.assertEqual(expert // 56, owner)
                            counts[expert % 56] += 1
                self.assertEqual(counts, [16 * tokens] * 16 + [0] * 40)
                self.assertLessEqual(tokens * 16, 768)
            self.assertEqual(sweep.hot_assignment(1, 0, 8, only_source=0), -1)

    def test_fixed_graph_sequence_has_balanced_hot_return_and_masked_hot(self):
        seq = sweep.IMBALANCE_SEQUENCE
        self.assertEqual(seq[:4], ("all_remote", "hot_owner8", "hot_owner0", "all_remote"))
        self.assertIn(("all_masked", "hot_owner8"), list(zip(seq, seq[1:])))

    def test_payload_baseline_toggles_dispatch_only(self):
        seen = []
        tensor = SimpleNamespace(view=lambda dtype: "bits")
        def snapshot(*args):
            seen.append((os.environ[sweep.MODE_ENV], os.environ[sweep.COMBINE_ENV]))
            return {"all_remote": tensor}, {}
        def payload(*args):
            sweep.accuracy._snapshot_and_check_oracles(*args)
            return {"enabled": True}
        torch = SimpleNamespace(equal=lambda a, b: a == b, int16="int16")
        with patch.dict(os.environ, {sweep.MODE_ENV: "1", sweep.COMBINE_ENV: "1"}), \
             patch.object(sweep.accuracy, "_snapshot_and_check_oracles", side_effect=snapshot), \
             patch.object(sweep.accuracy, "_run_payload_epoch_validation", side_effect=payload), \
             patch.object(sweep.accuracy, "_collective_transition_check"):
            result = sweep._payload_with_control(None, 1, torch, None)
            self.assertEqual(os.environ[sweep.MODE_ENV], "1")
        self.assertEqual(seen, [("0", "1"), ("1", "1")])
        self.assertTrue(result["same_context_dispatch0_per_epoch_bitwise"])

    def test_contract_never_labels_fallback_or_local_path_as_overlap_observation(self):
        for mode in (0, 1):
            contract = sweep.dispatch_contract(mode)
            self.assertEqual(contract["direct_t48_t64_t48_split_policy"], [bool(mode), False, bool(mode)])
            self.assertTrue(contract["effective_policy_not_device_observation"])
            self.assertTrue(contract["compute_tiling_sm_count_and_math_unchanged"])
            self.assertEqual(contract["fixed_single_combine_context"], 1)

    def test_success_is_after_worker_and_sources_include_every_imported_harness(self):
        source = Path(sweep.__file__).read_text()
        self.assertLess(source.index('accuracy._worker(int(os.environ["LOCAL_RANK"])'),
                        source.index('result.update(status="passed"'))
        self.assertLess(source.index('axis.label + "/preflight-before-candidate"'),
                        source.index('for entry, mode in enumerate(options.schedule)'))
        for name in ("tests/bench_mega_moe_dispatch_overlap.py", "tests/test_mega_moe_accuracy.py",
                     "tests/bench_mega_moe_single_combine_context.py"):
            self.assertIn(name, sweep.SOURCE_FILES)
        self.assertNotIn("num_active_routes_hint", source)
        self.assertIn('"dense_rank_distinct_gate_is_separate": True', source)
        self.assertIn('"heterogeneous_input_scales_untimed_only": True', source)
        self.assertIn('"candidate_sequence_has_no_interleaved_control_replays": True', source)
        fixed_shape = source[source.index("def _fixed_shape_imbalance"):source.index("def _drift")]
        self.assertLess(fixed_shape.index("for epoch in dict.fromkeys(epochs)"),
                        fixed_shape.index("for epoch in epochs:"))
        self.assertNotIn("baseline.replay()", fixed_shape[fixed_shape.index("for epoch in epochs:"):])


if __name__ == "__main__":
    unittest.main()
