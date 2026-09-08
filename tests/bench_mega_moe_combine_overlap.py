"""Accuracy-gated ready-span coalescing and direct packet reduction.

Two-node torchrun, eight ranks/node:
  python tests/bench_mega_moe_combine_overlap.py --decode-mns 8 --output RESULT.json

Only COMBINE_OVERLAP changes 0→1→0. DISPATCH_OVERLAP=1 and
SINGLE_COMBINE_CONTEXT=1 stay fixed. The shared driver preserves balanced
launch-only timings and untimed exact-count hot-owner, masked/asymmetric,
varying FP8/SF payload and same-allocation T48→64→48 gates. Device overlap and
the per-rank counter-storage fit branch are not inferred from environment flags.
"""

import bench_mega_moe_dispatch_overlap as shared


MODE_ENV = "DG_MEGAMOE_GIN_COMBINE_OVERLAP"
AXIS = shared.SweepAxis("combine", MODE_ENV, (
    (shared.MODE_ENV, "1"), (shared.COMBINE_ENV, "1")),
    ("tests/bench_mega_moe_combine_overlap.py", "tests/mega_moe_serialized_streams.py"))


def combine_contract(mode):
    if type(mode) is not int or mode not in (0, 1):
        raise ValueError("combine overlap mode must be 0 or 1")
    return {
        "requested_combine_overlap": bool(mode),
        "requested_combine_overlap_raw": str(mode),
        "fixed_dispatch_overlap": 1, "fixed_single_combine_context": 1,
        "compute_tiling_sm_count_and_math_unchanged": True,
        "effective_policy_not_device_observation": True,
        "eligible_policy": "remote direct+bulk, all actual expert M/N producers complete, per-rank expert metadata fits",
        "counter_storage_fit_is_per_rank": True,
        "counter_storage_capacity_policy": "56 expert ready counts + 56 sent entries + 8*56 saved source/expert prefixes + 8*2 nonempty mask words",
        "expert_metadata_storage_bytes": 2304,
        "direct_reducer": {
            "requested": bool(mode),
            "additional_source_local_ordinal_bytes": 3072,
            "required_scratch_extent_bytes": 62720,
            "fit_policy": "combine_overlap_eligible_and_source_local_inverse_map_fits",
            "source_inverse_written_during_actual_pack": True,
            "remote_inputs": "owner_packet_payload_via_original_assignment_inverse",
            "local_shared_inputs": "unchanged_combine_buffer",
            "reduction_order": "original_ascending_topk_slot_fp32_then_bf16",
            "received_count_and_put_visibility_preserved": True,
            "target_visibility_to_tma_proxy": "one_async_global_proxy_fence_per_epilogue_thread_before_reduction",
            "scatter_and_third_epilogue_grid_skipped_if_eligible": True,
            "cleanup_handoff": "existing_second_handoff_deferred_until_all_local_packet_reads_complete",
            "added_barriers": 0,
            "fit_failure": "retain_current_sender_policy_and_original_scatter",
            "stream_lifetime": "same_buffer_launches_event_or_stream_serialized",
            "policy_not_device_observation": True,
        },
        "expert_readiness_target_policy": "ceil(actual_expert_assignments / actual_block_m) * (hidden / actual_block_n)",
        "send_range_policy": "dispatch_saved_source_expert_prefix_and_existing_exact_count",
        "ready_selection_policy": "warp_parallel_readiness_peer_independent_bounded_ready_coalescing",
        "nonempty_masks_saved_during_existing_dispatch_prefix_scan": True,
        "payload_span_granularity_unchanged_one_nonempty_peer_expert": False,
        "ready_batch_max_experts": 8,
        "ready_batch_cap_is_compile_time_constant": True,
        "frozen_ready_snapshot_no_fill_wait": True,
        "issuer_acquires_every_batched_expert": True,
        "packet_payload_records_and_reduction_order_unchanged": True,
        "warp_collectives_remain_uniform_outside_peer_specific_issue": True,
        "fit_failure_keeps_existing_full_packet_sc1_protocol": True,
        "completed_expert_payload_may_issue_early": bool(mode),
        "hot_experts_wait_for_all_actual_m_blocks_without_count_truncation": True,
        "count_headers_and_final_sc1_put_barrier_remain_late": True,
        "t48_t64_t48_overlap_request_policy": [bool(mode), False, bool(mode)],
        "all_local_remote_combine_skipped": True,
        "physical_compute_communication_overlap_measured": False,
    }


def parse_args():
    return shared.parse_args(AXIS)


def prepare_sweep_environment():
    shared.prepare_sweep_environment(AXIS)


def main():
    shared.main(AXIS, combine_contract)


if __name__ == "__main__":
    main()
