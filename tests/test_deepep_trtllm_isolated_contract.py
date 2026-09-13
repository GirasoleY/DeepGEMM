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
    def test_late_header_completion_metadata_is_explicit_for_both_modes(self):
        for mode in ("0", "1"):
            metadata = comparison.dispatch_candidate_metadata({
                "DG_MEGAMOE_GIN_DISPATCH_OVERLAP": "1",
                "DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT": "1",
                "DG_MEGAMOE_GIN_COMBINE_OVERLAP": mode,
            })
            self.assertEqual(metadata["combine_overlap_contract"]["combine_payload_local_completion"], {
                "requested_by_combine_overlap": mode == "1",
                "eligibility": "early_record_combine_path_and_scratch_alias_fits",
                "completion": "late_header_same_context_peer",
                "payload_only_flush_before_handoff": False,
                "all_input_flushes_retained": True,
                "late_header_put_and_flush_retained": True,
                "original_handoff_and_grid_order_retained": True,
                "final_world_put_barrier_retained": True,
                "source_storage_retained_until_late_header_flush": True,
                "header_flush_does_not_prove_remote_visibility": True,
                "fallback": "unchanged_full_packet_local_flush",
                "slot101_writer_present": False,
                "policy_not_device_observation": True,
            })

    def test_combine_metadata_requires_dispatch_sc_and_labels_only_transport_policy(self):
        flags = {"DG_MEGAMOE_GIN_DISPATCH_OVERLAP": "1",
                 "DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT": "1",
                 "DG_MEGAMOE_GIN_COMBINE_OVERLAP": "1"}
        metadata = comparison.dispatch_candidate_metadata(flags)
        self.assertEqual(metadata["candidate_family"], "direct_control_first_dispatch_ready_coalesced_direct_reduce_preload_late_flush")
        contract = metadata["combine_overlap_contract"]
        self.assertEqual(contract["requested_raw"], "1")
        self.assertTrue(contract["requested"])
        reducer = contract["direct_reducer"]
        self.assertEqual(reducer["address_preparation"],
                         "lane_local_token_row_pointer_preload_before_chunk_loop")
        self.assertIs(reducer["full_warp_pointer_gather_before_elected_issuer"], True)
        self.assertIs(type(reducer["metadata_resolution_passes_per_active_remote_assignment_per_token"]), int)
        self.assertEqual(reducer["metadata_resolution_passes_per_active_remote_assignment_per_token"], 1)
        self.assertTrue(reducer["policy_not_device_observation"])
        self.assertTrue(reducer["requested"])
        self.assertTrue(reducer["source_inverse_written_during_actual_pack"])
        self.assertTrue(reducer["received_count_and_put_visibility_preserved"])
        self.assertIn("async_global_proxy_fence", reducer["target_visibility_to_tma_proxy"])
        self.assertEqual(reducer["additional_source_local_ordinal_bytes"], 3072)
        self.assertEqual(reducer["required_scratch_extent_bytes"], 62720)
        self.assertEqual(reducer["added_barriers"], 0)
        self.assertIn("original_scatter", reducer["fit_failure"])
        self.assertIn("deferred_until_all_local_packet_reads_complete", reducer["cleanup_handoff"])
        self.assertIn("expert-ready storage fit", contract["eligibility"])
        self.assertEqual(contract["producer_target"],
                         "ceil(actual expert assignments / actual BM) * (H / BN)")
        self.assertIn("saved dispatch source/expert prefixes", contract["span_descriptors"])
        self.assertIn("each peer independently selects", contract["early_payload_policy"])
        self.assertEqual(contract["combine_schedule"], "peer_parallel_ready_coalesced_spans_then_late_header")
        self.assertEqual(contract["ready_selection_policy"],
                         "warp_parallel_readiness_peer_independent_bounded_ready_coalescing")
        self.assertIn("fixed cap8", contract["submission_granularity"])
        self.assertEqual(contract["ready_batch_max_experts"], 8)
        self.assertTrue(contract["ready_batch_cap_is_compile_time_constant"])
        self.assertTrue(contract["frozen_ready_snapshot_no_fill_wait"])
        self.assertTrue(contract["issuer_acquires_every_batched_expert"])
        self.assertIn("not early readiness", contract["sent_entry_role"])
        layout = contract["scratch_layout"]
        self.assertEqual(layout["saved_source_nonempty_mask_uint32"], 16)
        self.assertEqual(layout["tail_bytes"], 4 * (56 + 56 + 8 * 56 + 8 * 2))
        self.assertEqual(layout["required_scratch_bytes"], layout["tail_bytes"] + layout["direct_control_bytes"])
        self.assertEqual(layout["required_scratch_bytes"], 59648)
        self.assertFalse(layout["byte_capacity_depends_on_bm"])
        self.assertFalse(contract["physical_overlap_measured"])
        self.assertIn("not device branch", contract["basis"])
        self.assertFalse(contract["compute_hints_tiling_sm_count_and_math_changed"])
        for invalid in ("01", "2", 1, True):
            with self.assertRaises(ValueError):
                comparison.dispatch_candidate_metadata({**flags, "DG_MEGAMOE_GIN_COMBINE_OVERLAP": invalid})
        for missing in ("DG_MEGAMOE_GIN_DISPATCH_OVERLAP", "DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT"):
            with self.assertRaises(ValueError):
                comparison.dispatch_candidate_metadata({**flags, missing: "0"})
        import inspect
        source = inspect.getsource(comparison.main)
        self.assertLess(source.index('"DG_MEGAMOE_GIN_COMBINE_OVERLAP"'),
                        source.index("dist.all_gather_object(rank_flags, flags)"))

    def test_strongva_terminal_metadata_is_explicit_and_default_off(self):
        flags = {
            "DG_MEGAMOE_GIN_DISPATCH_OVERLAP": "1",
            "DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT": "1",
            "DG_MEGAMOE_GIN_COMBINE_OVERLAP": "1",
            "DG_MEGAMOE_GIN_STRONGVA_COMBINE_TERMINAL": "1",
        }
        metadata = comparison.dispatch_candidate_metadata(flags)
        self.assertIn("strongva_terminal", metadata["candidate_family"])
        contract = metadata["combine_overlap_contract"]
        self.assertTrue(contract["strongva_terminal_requested"])
        self.assertIn("final_strongva_terminal", contract["combine_schedule"])
        completion = contract["combine_payload_local_completion"]
        self.assertEqual(
            completion["eligibility"],
            "world_uniform_bulk_direct_remote_after_host_scratch_preflight",
        )
        self.assertEqual(
            completion["completion"],
            "late_cleanup_post_terminal_same_context_peer_flush",
        )
        self.assertFalse(completion["late_header_put_and_flush_retained"])
        self.assertFalse(completion["final_world_put_barrier_retained"])
        self.assertFalse(
            completion["source_storage_retained_until_late_header_flush"])
        default = comparison.dispatch_candidate_metadata({
            key: value for key, value in flags.items()
            if key != "DG_MEGAMOE_GIN_STRONGVA_COMBINE_TERMINAL"
        })
        self.assertFalse(
            default["combine_overlap_contract"]["strongva_terminal_requested"])
        with self.assertRaises(ValueError):
            comparison.dispatch_candidate_metadata({
                **flags, "DG_MEGAMOE_GIN_COMBINE_OVERLAP": "0"})

    def test_coarse_owner_wave_metadata_is_exact(self):
        base = {
            "DG_MEGAMOE_GIN_DISPATCH_OVERLAP": "1",
            "DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT": "1",
            "DG_MEGAMOE_GIN_COMBINE_OVERLAP": "1",
            "DG_MEGAMOE_GIN_STRONGVA_COMBINE_TERMINAL": "1",
        }
        for waves in (2, 4, 8):
            with self.subTest(waves=waves):
                metadata = comparison.dispatch_candidate_metadata({
                    **base,
                    "DG_MEGAMOE_GIN_COMBINE_OWNER_WAVES": str(waves),
                })
                self.assertIn("coarse_owner_waves",
                              metadata["candidate_family"])
                contract = metadata["combine_overlap_contract"]
                self.assertEqual(
                    contract["combine_owner_waves_requested"], waves)
                self.assertEqual(contract["experts_per_owner_wave"],
                                 56 // waves)
                self.assertEqual(contract["owner_wave_ranges"][0],
                                 [0, 56 // waves])
                self.assertEqual(contract["owner_wave_ranges"][-1],
                                 [56 - 56 // waves, 56])
                self.assertTrue(contract["owner_wave_protocol"][
                    "terminal_attaches_to_actual_last_submitted_nonempty_range"])
        for invalid in ("1", "3", "6", "02", "8 ", 2):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                    ValueError, "exactly 0, 2, 4, or 8"):
                comparison.dispatch_candidate_metadata({
                    **base,
                    "DG_MEGAMOE_GIN_COMBINE_OWNER_WAVES": invalid,
                })
        with self.assertRaisesRegex(ValueError,
                                    "require StrongVA combine terminal1"):
            comparison.dispatch_candidate_metadata({
                **base,
                "DG_MEGAMOE_GIN_STRONGVA_COMBINE_TERMINAL": "0",
                "DG_MEGAMOE_GIN_COMBINE_OWNER_WAVES": "2",
            })

    def test_owner_slot_metadata_is_exact_and_default_off(self):
        base = {
            "DG_MEGAMOE_GIN_DISPATCH_OVERLAP": "1",
            "DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT": "1",
            "DG_MEGAMOE_GIN_COMBINE_OVERLAP": "1",
            "DG_MEGAMOE_GIN_STRONGVA_COMBINE_TERMINAL": "1",
            "DG_MEGAMOE_GIN_COMBINE_OWNER_WAVES": "4",
        }
        implicit_r9 = comparison.dispatch_candidate_metadata(base)
        explicit_r9 = comparison.dispatch_candidate_metadata({
            **base, "DG_MEGAMOE_GIN_COMBINE_OWNER_SLOT_READY": "0",
        })
        self.assertEqual(implicit_r9, explicit_r9)
        self.assertEqual(
            implicit_r9["candidate_family"],
            "direct_control_first_dispatch_coarse_owner_waves_direct_reduce_"
            "strongva_terminal",
        )
        self.assertNotIn(
            "combine_owner_slot_ready_requested",
            implicit_r9["combine_overlap_contract"],
        )

        metadata = comparison.dispatch_candidate_metadata({
            **base, "DG_MEGAMOE_GIN_COMBINE_OWNER_SLOT_READY": "1",
        })
        self.assertEqual(
            metadata["candidate_family"],
            "direct_control_first_dispatch_coarse_owner_waves_owner_slot_"
            "ready_dual_stage_pair_load_fixed_pair_reduce_strongva_terminal",
        )
        contract = metadata["combine_overlap_contract"]
        self.assertEqual(contract["combine_owner_slot_ready_requested_raw"],
                         "1")
        self.assertTrue(contract["combine_owner_slot_ready_requested"])
        self.assertIn("four_tail_progress_warps",
                      contract["receiver_schedule"])
        self.assertIn("one_cta_per_token", contract["receiver_schedule"])
        self.assertIn("slots_2w_and_2w_plus_1",
                      contract["receiver_schedule"])
        self.assertIn("independently_issues_each_ready_assignment",
                      contract["receiver_schedule"])

        delta = contract["dual_stage_pair_load_delta"]
        self.assertEqual(delta["baseline"],
                         "owner_slot_ready_fixed_pair_reduce")
        self.assertTrue(delta["receiver_load_schedule_changed"])
        for field in (
                "wire_bytes_changed", "registered_workspace_bytes_changed",
                "network_operations_changed", "gemm_tiling_changed",
                "launch_sm_count_changed", "arithmetic_association_changed"):
            self.assertFalse(delta[field])

        protocol = contract["owner_wave_protocol"]
        self.assertNotIn(
            "receiver_reducer_packet_layout_math_tiling_sm_changed", protocol)
        self.assertFalse(protocol["sender_wire_protocol_changed"])
        self.assertFalse(protocol["packet_layout_changed"])
        self.assertTrue(protocol["receiver_reducer_schedule_changed"])
        self.assertTrue(protocol["receiver_reducer_math_association_changed"])
        self.assertFalse(protocol["gemm_tiling_changed"])
        self.assertFalse(protocol["launch_sm_count_changed"])
        self.assertIn("StrongVA_acquire_then_async_proxy_fence_then_rank_local_release",
                      protocol["receiver_owner_ready_publication"])

        completion = contract["combine_payload_local_completion"]
        self.assertNotIn("original_handoff_and_grid_order_retained",
                         completion)
        self.assertTrue(completion["local_handoff_and_grid_count_retained"])
        self.assertTrue(
            completion["final_grid_moved_before_remote_owner_progress"])

        reducer = contract["direct_reducer"]
        self.assertNotIn("full_warp_pointer_gather_before_elected_issuer",
                         reducer)
        self.assertNotIn("added_barriers", reducer)
        self.assertEqual(
            reducer["address_preparation"],
            "lane0_dual_stage_fixed_pair_pointer_resolution_on_chunk0_"
            "before_assignment_TMA",
        )
        self.assertFalse(
            reducer["lane0_pointer_broadcast_before_lane0_tma_issuer"])
        self.assertTrue(reducer["lane0_pointer_resolution_and_tma_issuer"])
        self.assertEqual(reducer["assignment_pairs"],
                         [[2 * pair, 2 * pair + 1] for pair in range(8)])
        self.assertIn("first_observed_ready_owner",
                      reducer["within_pair_assignment_issue_order"])
        self.assertIn("ascending_slot_tie_break",
                      reducer["within_pair_assignment_issue_order"])
        self.assertEqual(
            reducer["within_pair_assignment_accumulation_order"],
            "ascending_slot_after_both_stage_waits")
        self.assertIn("ascending_pair_index", reducer["reduction_order"])
        self.assertNotIn("original_ascending_topk_slot",
                         reducer["reduction_order"])
        self.assertIn("consumer_owner_ready_acquire_then_async_proxy_fence",
                      reducer["target_visibility_to_tma_proxy"])
        self.assertIn("none_for_local_or_world_ineligible_fallback",
                      reducer["owner_ready_reads"])
        self.assertEqual(reducer["additional_barrier_objects"], 0)
        self.assertIn("two_existing_independent_mbarriers",
                      reducer["assignment_mbarriers"])
        self.assertIn(
            "alias_the_same_warp_FP32_pair_partial_only_after_both_load_waits",
            reducer["shared_staging"])
        self.assertEqual(reducer["shared_chunk_regions"], {
            "assignment_stage_or_fp32_pair_partial": 16,
            "output": 1,
            "total": 17,
            "existing_budget": 24,
        })
        self.assertIn("two_existing_full_epilogue_barrier_sync_points",
                      reducer["cta_barrier_participation"])

        scratch = contract["scratch_layout"]
        self.assertEqual(scratch["receiver_owner_ready_uint32"], 4)
        self.assertEqual(scratch["receiver_owner_ready_alias"],
                         "sent_uint32_0_through_3")
        self.assertEqual(scratch["additional_registered_bytes"], 0)
        self.assertIn("tokens_le_48_local_or_world_ineligible_fallback",
                      contract["local_and_t64_paths"])
        self.assertIn("T64_uses_original_r75",
                      contract["local_and_t64_paths"])
        self.assertNotIn("count_headers_and_final_put_barrier", contract)
        self.assertEqual(contract["count_headers"], "late_and_unchanged")
        self.assertIn("per_owner_StrongVA_terminals",
                      contract["final_world_put_barrier"])
        self.assertNotIn(
            "compute_hints_tiling_sm_count_and_math_changed", contract)
        self.assertFalse(contract["compute_hints_changed"])
        self.assertFalse(contract["gemm_tiling_changed"])
        self.assertFalse(contract["launch_sm_count_changed"])
        self.assertTrue(contract["combine_reducer_math_association_changed"])

        for invalid in ("01", "2", 1, True):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                    ValueError, "canonical0/1"):
                comparison.dispatch_candidate_metadata({
                    **base,
                    "DG_MEGAMOE_GIN_COMBINE_OWNER_SLOT_READY": invalid,
                })
        with self.assertRaisesRegex(ValueError, "requires W4 StrongVA"):
            comparison.dispatch_candidate_metadata({
                **base,
                "DG_MEGAMOE_GIN_COMBINE_OWNER_WAVES": "2",
                "DG_MEGAMOE_GIN_COMBINE_OWNER_SLOT_READY": "1",
            })

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
        self.assertLess(
            source.index('"DG_MEGAMOE_GIN_COMBINE_OWNER_SLOT_READY"'),
            source.index("dist.all_gather_object(rank_flags, flags)"),
        )
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
