"""CPU driver contracts, not device accuracy/performance evidence."""

import inspect
import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import bench_mega_moe_combine_overlap as sweep


class CombineSweepContracts(unittest.TestCase):
    def test_full_schedule_gates_and_unchanged_timing_defaults(self):
        for mns, tokens in ((8, 32), (10, 40), (12, 48)):
            with patch.object(sys, "argv", ["sweep", "--decode-mns", str(mns),
                                            "--output", "/tmp/not-created.json"]):
                options, args = sweep.parse_args()
            self.assertEqual(options.schedule, [0, 1, 0])
            self.assertEqual(options.benchmark_replays, 204)
            self.assertEqual(args.benchmark_conditioning_replays, 60)
            self.assertEqual((args.num_tokens, args.num_max_tokens_per_rank), (tokens, 384))
            self.assertEqual(args.payload_epochs, 3)
            self.assertFalse(args.fast_math)
            self.assertTrue(options.payload_per_mode and options.transitions_per_mode)
        for flag in ("--no-payload-per-mode", "--no-transitions-per-mode", "--no-repeat-control"):
            with patch.object(sys, "argv", ["sweep", "--output", "/tmp/not-created.json", flag]):
                with self.assertRaises(ValueError):
                    sweep.parse_args()

    def test_only_combine_varies_dispatch_and_single_context_fixed(self):
        with patch.dict(os.environ, {}, clear=True):
            sweep.prepare_sweep_environment()
            fixed = dict(sweep.AXIS.fixed_environment)
            self.assertEqual(fixed, {sweep.shared.MODE_ENV: "1", sweep.shared.COMBINE_ENV: "1"})
            self.assertEqual(os.environ[sweep.MODE_ENV], "0")
            for mode in (0, 1, 0):
                with patch.dict(os.environ, {sweep.MODE_ENV: str(mode)}):
                    self.assertEqual({name: os.environ[name] for name in fixed}, fixed)
                    for name in sweep.shared.accuracy.GIN_EXPERIMENT_FLAG_ENVS:
                        self.assertEqual(os.environ[name], "1")
            self.assertNotIn(sweep.MODE_ENV, sweep.shared.accuracy.GIN_EXPERIMENT_FLAG_ENVS)
            self.assertIn(sweep.MODE_ENV, sweep.shared.accuracy.GIN_PROTOCOL_FLAG_ENVS)

    def test_payload_control_executes_combine0_without_resetting_dispatch(self):
        seen = []
        tensor = SimpleNamespace(view=lambda dtype: "bits")
        def snapshot(*args):
            seen.append(tuple(os.environ[key] for key in
                              (sweep.MODE_ENV, sweep.shared.MODE_ENV, sweep.shared.COMBINE_ENV)))
            return {"all_remote": tensor}, {}
        def payload(*args):
            sweep.shared.accuracy._snapshot_and_check_oracles(*args)
            return {"enabled": True}
        torch = SimpleNamespace(equal=lambda a, b: a == b, int16="int16")
        with patch.dict(os.environ, {sweep.MODE_ENV: "1", **dict(sweep.AXIS.fixed_environment)}), \
             patch.object(sweep.shared.accuracy, "_snapshot_and_check_oracles", side_effect=snapshot), \
             patch.object(sweep.shared.accuracy, "_run_payload_epoch_validation", side_effect=payload), \
             patch.object(sweep.shared.accuracy, "_collective_transition_check"):
            result = sweep.shared._payload_with_control(None, 1, torch, None, sweep.AXIS)
            self.assertEqual(os.environ[sweep.MODE_ENV], "1")
        self.assertEqual(seen, [("0", "1", "1"), ("1", "1", "1")])
        self.assertTrue(result["same_context_combine0_per_epoch_bitwise"])
        self.assertNotIn("same_context_dispatch0_per_epoch_bitwise", result)

    def test_fit_fallback_and_readiness_are_policy_not_observed_overlap(self):
        for mode in (0, 1):
            contract = sweep.combine_contract(mode)
            self.assertTrue(contract["effective_policy_not_device_observation"])
            self.assertTrue(contract["counter_storage_fit_is_per_rank"])
            self.assertEqual(contract["expert_metadata_storage_bytes"], 2304)
            self.assertIn("actual_expert_assignments", contract["expert_readiness_target_policy"])
            self.assertIn("dispatch_saved", contract["send_range_policy"])
            self.assertIn("immediate_issue", contract["ready_selection_policy"])
            self.assertIn("peer_independent", contract["ready_selection_policy"])
            self.assertTrue(contract["nonempty_masks_saved_during_existing_dispatch_prefix_scan"])
            self.assertTrue(contract["payload_span_granularity_unchanged_one_nonempty_peer_expert"])
            self.assertTrue(contract["warp_collectives_remain_uniform_outside_peer_specific_issue"])
            self.assertTrue(contract["hot_experts_wait_for_all_actual_m_blocks_without_count_truncation"])
            self.assertTrue(contract["fit_failure_keeps_existing_full_packet_sc1_protocol"])
            self.assertEqual(contract["t48_t64_t48_overlap_request_policy"], [bool(mode), False, bool(mode)])
            self.assertFalse(contract["physical_compute_communication_overlap_measured"])
        for invalid in (True, -1, 2, "1"):
            with self.assertRaises(ValueError):
                sweep.combine_contract(invalid)

    def test_reused_driver_keeps_all_gates_and_source_and_teardown_checks(self):
        source = inspect.getsource(sweep.shared.main)
        for call in ("_fixed_shape_imbalance", "_payload_with_control", "_run_gin_bulk_transition_validation",
                     "_check_graph_routes", "_check_identity", "source_fingerprint(axis)"):
            self.assertIn(call, source)
        self.assertLess(source.index("preflight-before-candidate"), source.index("for entry, mode"))
        per_mode = source[source.index("for entry, mode"):]
        self.assertLess(per_mode.index("_collect_gin_experiment_flags("),
                        per_mode.index("_capture_launch_only_graph("))
        self.assertLess(source.index("accuracy._worker("), source.index('result.update(status="passed"'))
        fingerprint = sweep.shared.source_fingerprint(sweep.AXIS)
        self.assertIn("tests/bench_mega_moe_combine_overlap.py", fingerprint["files_sha256"])
        self.assertIn("tests/bench_mega_moe_dispatch_overlap.py", fingerprint["files_sha256"])
        self.assertEqual(sweep.shared.IMBALANCE_SEQUENCE[:4],
                         ("all_remote", "hot_owner8", "hot_owner0", "all_remote"))
        self.assertIn("single_active_hot_owner8", sweep.shared.IMBALANCE_SEQUENCE)
        self.assertIn("all_masked", sweep.shared.IMBALANCE_SEQUENCE)


if __name__ == "__main__":
    unittest.main()
