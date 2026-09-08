"""Capture instrumented MegaMoE phases after the existing accuracy gate.

Run separately for --decode-mns 8/10/12 under two-node torchrun. Example:
  python tests/capture_mega_moe_phases.py --decode-mns 8 \
      --diagnostic-level 2 --diagnostic-replays 8 --output phases-mns8.json

Requires the diagnostic extension in this sibling checkout. The runner uses
the established allocator, inputs, routes, correctness checks, and teardown.
It records no MEGAMOE_PERF_JSON and does not time GPU-to-CPU readback.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import test_mega_moe_accuracy as accuracy
from analyze_mega_moe_phases import summarize_capture


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic-level", type=int, choices=(1, 2), default=1)
    parser.add_argument("--diagnostic-replays", type=int, default=8)
    parser.add_argument("--diagnostic-warmups", type=int, default=10)
    parser.add_argument(
        "--calibration-replays", type=int, default=32,
        help="Event-time each diagnostic level 0/1/2 per route (default 32; 0 disables calibration)",
    )
    diagnostic, remaining = parser.parse_known_args()
    if diagnostic.diagnostic_replays < 1 or diagnostic.diagnostic_warmups < 1:
        parser.error("diagnostic replay and warmup counts must be positive")
    if diagnostic.calibration_replays < 0 or diagnostic.calibration_replays == 1:
        parser.error("--calibration-replays must be 0 or at least 2")
    defaults = {
        "--decode-mns": "8", "--eager-iterations": "16", "--graph-replays": "32",
        "--gin-completion-batch": "8", "--gin-combine-issue-wave": "8",
        "--gin-outbox-depth": "64", "--gin-combine-chunk-bytes": "7168",
    }
    for flag, value in defaults.items():
        present = any(option == flag or option.startswith(flag + "=") for option in remaining)
        tokens_present = any(option == "--num-tokens" or option.startswith("--num-tokens=")
                             for option in remaining)
        if not present and not (flag == "--decode-mns" and tokens_present):
            remaining.extend((flag, value))
    remaining.extend(("--k3", "--require-gin", "--require-cross-host",
                      "--gin-active-fast-path", "--gin-bulk-combine",
                      "--gin-direct-dispatch"))
    with patch.object(sys, "argv", [sys.argv[0], *remaining]):
        args = accuracy._parse_args()
    if args.benchmark_replays or args.gin_local_ablation_stages:
        parser.error("phase capture must be separate from benchmarks and local ablations")
    if args.num_tokens not in (32, 40, 48):
        parser.error("this runner covers target token counts 32, 40, and 48")
    return diagnostic, args


def _calibration_deltas(level_records):
    """Compare equivalent same-rank samples without crossing GPU clocks."""
    baseline = level_records["0"]
    baseline_p50 = baseline["max_rank_per_replay_us"]["p50"]
    result = {}
    for level in (1, 2):
        current = level_records[str(level)]
        rank_deltas = [
            [measured - reference for measured, reference in zip(samples, baseline_samples)]
            for samples, baseline_samples in zip(
                current["local_rank_samples_us"], baseline["local_rank_samples_us"]
            )
        ]
        delta_p50 = current["max_rank_per_replay_us"]["p50"] - baseline_p50
        result[str(level)] = {
            "max_rank_p50_delta_vs_level0_us": delta_p50,
            "max_rank_p50_ratio_vs_level0": current["max_rank_per_replay_us"]["p50"] / baseline_p50,
            "same_rank_ordinal_sample_deltas_us": rank_deltas,
            "same_rank_delta_distribution_us": accuracy._timing_summary(
                [value for samples in rank_deltas for value in samples]
            ),
        }
    return result


def _raw_kernel_environment():
    names = (
        "DG_MEGAMOE_GIN_ACTIVITY_GATE_OPT", "DG_MEGAMOE_GIN_DISPATCH_WARP_SCAN",
        "DG_MEGAMOE_GIN_COOP_DIRECT_PACK", "DG_MEGAMOE_GIN_PRECONSENSUS_PACK",
        "DG_MEGAMOE_GIN_COMBINE_EXPERTS_PER_WAVE",
        accuracy.GIN_SINGLE_COMBINE_CONTEXT_ENV,
        "DG_MEGAMOE_GIN_DISPATCH_OVERLAP", "DG_MEGAMOE_GIN_COMBINE_OVERLAP",
    )
    result = {name: os.environ.get(name, "0") for name in names}
    result["DG_MEGAMOE_GIN_COMBINE_BARRIER_WARPS"] = os.environ.get(
        "DG_MEGAMOE_GIN_COMBINE_BARRIER_WARPS", "1")
    return result


def _collective_kernel_configuration(args, diagnostic, dist):
    # Collect raw strings BEFORE parsing them: a rank-local flag error must not
    # leave peers entering allocation/GIN setup or a mismatched graph capture.
    local = {
        "environment": _raw_kernel_environment(),
        "setup": {name: getattr(args, name, None) for name in (
            "num_tokens", "hidden", "intermediate_hidden", "num_experts", "num_topk",
            "num_max_tokens_per_rank", "fast_math", "activation_clamp",
            "gin_active_fast_path", "gin_bulk_combine", "gin_direct_dispatch",
            "gin_completion_batch", "gin_combine_issue_wave", "gin_outbox_depth",
            "gin_combine_chunk_bytes", "eager_iterations", "graph_replays",
        )},
        "diagnostic": {name: getattr(diagnostic, name, None) for name in (
            "diagnostic_level", "diagnostic_replays", "diagnostic_warmups",
            "calibration_replays",
        )},
    }
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    if any(record != gathered[0] for record in gathered):
        raise RuntimeError("phase capture raw configuration differs across ranks")
    return {
        **_kernel_configuration(args, environment=gathered[0]["environment"]),
        "collectively_validated_setup": gathered[0]["setup"],
        "collectively_validated_diagnostic_settings": gathered[0]["diagnostic"],
    }


def _kernel_configuration(args, *, environment=None):
    environment = _raw_kernel_environment() if environment is None else dict(environment)
    width_raw = environment["DG_MEGAMOE_GIN_COMBINE_EXPERTS_PER_WAVE"]
    if width_raw != "0":
        raise ValueError("retired GIN expert waves are unsupported; requires experts per wave 0")
    width = 0
    barrier_env = "DG_MEGAMOE_GIN_COMBINE_BARRIER_WARPS"
    if environment[barrier_env] != "1":
        raise ValueError("retired GIN barrier warp count is unsupported; requires 1")
    barrier_warps = 1
    single_env = accuracy.GIN_SINGLE_COMBINE_CONTEXT_ENV
    single_raw = environment[single_env]
    if single_raw not in ("0", "1"):
        raise ValueError("invalid single combine context mode")
    single_context = int(single_raw)
    dispatch_raw = environment["DG_MEGAMOE_GIN_DISPATCH_OVERLAP"]
    combine_raw = environment["DG_MEGAMOE_GIN_COMBINE_OVERLAP"]
    for name, raw in (("dispatch_overlap", dispatch_raw), ("combine_overlap", combine_raw)):
        if raw not in ("0", "1"):
            raise ValueError(f"{name} must be exactly 0 or 1")
    if dispatch_raw == "1" and not (
        getattr(args, "gin_bulk_combine", False)
        and getattr(args, "gin_direct_dispatch", False)
        and environment["DG_MEGAMOE_GIN_PRECONSENSUS_PACK"] == "1"
        and environment["DG_MEGAMOE_GIN_COOP_DIRECT_PACK"] == "1"
    ):
        raise ValueError("dispatch_overlap requires bulk/direct and preconsensus/cooperative packing")
    if combine_raw == "1" and not (single_raw == "1" and dispatch_raw == "1"):
        raise ValueError("combine_overlap requires single_combine_context=1 and dispatch_overlap=1")
    combine_schedule = (
        "dynamic_ready_expert_spans_then_late_header" if combine_raw == "1"
        else "post_compute_full_packet"
    )
    return {
        "combine_experts_per_wave": width,
        "combine_wave_count": 0,
        "combine_schedule": combine_schedule,
        "dispatch_overlap_requested": dispatch_raw == "1",
        "combine_overlap_requested": combine_raw == "1",
        "combine_schedule_is_requested_policy_not_device_observation": True,
        "combine_overlap_effective_policy": {
            "bulk_direct_remote_and_scratch_alias_fits_expert_metadata": combine_schedule,
            "bulk_direct_remote_but_scratch_alias_insufficient": "post_compute_full_packet",
            "remote_ineligible": "unchanged_fallback",
            "all_local": "unchanged_local_path",
        },
        "combine_ready_expert_policy": {
            "metadata_storage_bytes": 2240,
            "span_lookup": "dispatch_saved_source_expert_prefix_and_existing_count",
            "readiness_target": "ceil(actual_expert_assignments / actual_block_m) * (hidden / actual_block_n)",
            "selection": "warp_parallel_dynamic_ready_experts_immediate_issue",
            "no_token_metadata_rescan": True,
            "hot_expert_readiness_is_coarser_not_count_truncated": True,
            "policy_is_device_observation": False,
        },
        "phase_marker_semantics": {
            "32_39": ("receiver_control_terminal_acquired_not_payload" if dispatch_raw == "1"
                      else "receiver_combined_control_payload_terminal_acquired"),
            "24_31": "late_sender_local_flush_not_receiver_arrival_or_last_payload_time",
            "64_71": "first_combine_issue_observation_per_peer",
            "72_79": ("last_recorded_queue_observation_includes_late_header" if combine_raw == "1"
                      else "whole_packet_queue_observation"),
            "80_87": ("late_header_local_flush_not_last_payload_time" if combine_raw == "1"
                      else "whole_packet_local_flush_not_receiver_arrival"),
            "53": "epilogue_task_loop_exit_not_mma_completion_or_nic_visibility",
            "99": "first_ready_expert_selected_software_observation",
            "100": "all_combine_payload_puts_queued_not_remote_completion",
            "101": "combine_payload_local_flush_not_remote_visibility",
        },
        "retired_expert_wave_and_coop_width_experiments_supported": False,
        "combine_barrier_warps_requested": barrier_warps,
        # Source-level policy, not a new device observation. The clean
        # candidate contains only the original one-warp combine barrier.
        "combine_barrier_warps_effective_policy": {
            "bulk_and_direct_eligible_remote_with_sufficient_epilogue_warps": barrier_warps,
            "remote_ineligible_or_insufficient_epilogue_warps": 1,
            "all_local": 0,
        },
        "combine_barrier_effective_policy_is_device_observation": False,
        "single_combine_context_requested": bool(single_context),
        "allocated_gin_context_count": 9,
        "input_data_contexts_unchanged": list(range(1, 9)),
        "combine_barrier_contexts_effective_policy": {
            "bulk_and_direct_eligible_remote": [1] if single_context else list(range(9)),
            "remote_ineligible": list(range(9)),
            "all_local": [],
        },
        "combine_barrier_protocol": (
            "public_context1_world_put_fence_for_eligible_remote_else_all_context"
            if single_context else "unchanged_all_context_world_put_fence"
        ),
        "completion_batch": args.gin_completion_batch,
        "combine_issue_wave": args.gin_combine_issue_wave,
        "outbox_depth": args.gin_outbox_depth,
        "combine_chunk_bytes": args.gin_combine_chunk_bytes,
        "environment": environment,
    }


def calibrate_diagnostics(harness, snapshots, markers, diagnostic, torch, dist):
    """Measure instrumentation overhead with the same attached trace storage."""
    if diagnostic.calibration_replays == 0:
        return {"enabled": False, "reason": "explicitly_disabled"}
    env_name = "DG_MEGAMOE_GIN_DIAGNOSTICS"
    previous_level = os.environ.get(env_name)
    graphs = {}
    original_marker_pointer = int(markers.data_ptr())
    routes = {}
    try:
        # get_env is evaluated on each host launch and emitted into the JIT
        # source, so each graph permanently captures the requested level.
        for level in (0, 1, 2):
            os.environ[env_name] = str(level)
            graphs[level] = accuracy._capture_launch_only_graph(harness, torch, dist)

        for route_index, route in enumerate(accuracy.MATCHED_BENCHMARK_ROUTES):
            case = harness.cases[route]
            harness.copy_inputs(case)
            levels = {}
            level_order = [(route_index + offset) % 3 for offset in range(3)]
            for level in level_order:
                # Rewarm each CUfunction separately: the phase runner also
                # observes one specialization at a time, without I-cache
                # churn from swapping variants inside the measured sequence.
                for _ in range(diagnostic.diagnostic_warmups):
                    markers.zero_()
                    graphs[level].replay()
                torch.cuda.synchronize()
                harness.stats.zero_()
                durations = []
                for replay in range(diagnostic.calibration_replays):
                    markers.zero_()
                    harness.output.fill_(float("nan"))
                    torch.cuda.synchronize()
                    dist.barrier()
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[level].replay()
                    end.record()
                    torch.cuda.synchronize()
                    durations.append(float(start.elapsed_time(end) * 1000.0))
                    label = f"calibration/level{level}/{route}/{replay}"
                    harness.assert_stats(case.expected_local_stats * (replay + 1), label)
                    harness.assert_guards(label)
                    accuracy._assert_bitwise_equal(harness.output, snapshots[route], label, torch)
                if int(markers.data_ptr()) != original_marker_pointer:
                    raise AssertionError("calibration replaced attached marker storage")
                # Behavioral evidence that separate JIT levels were captured;
                # these checks and marker readback occur after the timed work.
                has_entry = bool(markers[:, 0].any().item())
                has_expert = bool(markers[:, 128:].any().item())
                if has_entry != (level > 0) or has_expert != (level == 2):
                    raise AssertionError(
                        f"diagnostic level {level} marker contract failed: "
                        f"entry={has_entry}, per_expert={has_expert}"
                    )
                levels[str(level)] = {
                    **accuracy._distributed_timing_record(durations, torch, dist),
                    "entry_markers_present": has_entry,
                    "per_expert_markers_present": has_expert,
                }
            routes[route] = {
                "level_order": level_order,
                "levels": levels,
                "overhead_vs_level0": _calibration_deltas(levels),
            }
    finally:
        graphs.clear()
        if previous_level is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = previous_level
    dist.barrier()
    return {
        "enabled": True,
        "performance_claim": False,
        "timing_scope": "diagnostic_launch_only_cuda_events_host_synchronized",
        "levels": [0, 1, 2],
        "replays_per_level_per_route": diagnostic.calibration_replays,
        "warmups_per_level_per_route": diagnostic.diagnostic_warmups,
        "same_marker_storage": True,
        "accuracy_verified_every_replay": True,
        "excluded_from_timing": ["input_preparation", "marker_zero", "output_fill", "host_barrier", "readback"],
        "notes": [
            "Each duration uses two events on the same GPU; maximum rank is computed per replay ordinal.",
            "Level-isolated batches are re-warmed; ordinal deltas are not simultaneously paired launches.",
            "This measures instrumentation overhead for the diagnostic capture regime, not selected benchmark performance.",
            "Overhead is observational; do not subtract a uniform correction from individual phase windows.",
        ],
        "routes": routes,
    }


def main() -> None:
    diagnostic, args = parse_args()
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("launch under torchrun with two hosts and eight ranks per host")
    for name in ("DG_MEGAMOE_GIN_ACTIVITY_GATE_OPT", "DG_MEGAMOE_GIN_DISPATCH_WARP_SCAN",
                 "DG_MEGAMOE_GIN_COOP_DIRECT_PACK", "DG_MEGAMOE_GIN_PRECONSENSUS_PACK"):
        os.environ[name] = "1"
    os.environ["DG_MEGAMOE_GIN_DIAGNOSTICS"] = str(diagnostic.diagnostic_level)
    torch, dist, deep_gemm = accuracy._load_runtime()
    original_allocate = deep_gemm.get_symm_buffer_for_mega_moe
    original_init_distributed = accuracy._init_distributed
    original_graph_stress = accuracy._run_graph_stress
    state = {}

    def initialize_then_validate(*positional, **kwargs):
        result = original_init_distributed(*positional, **kwargs)
        state["kernel_configuration"] = _collective_kernel_configuration(args, diagnostic, dist)
        return result

    def allocate(*positional, **kwargs):
        buffer = original_allocate(*positional, **kwargs)
        sm_count = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        markers = torch.zeros((sm_count, 240), dtype=torch.int64, device="cuda")
        buffer.gin_context.set_diagnostic_buffer(markers)
        state["markers"] = markers
        return buffer

    def graph_stress_then_capture(harness, snapshots, torch, dist):
        if _collective_kernel_configuration(args, diagnostic, dist) != state["kernel_configuration"]:
            raise RuntimeError("phase kernel configuration changed before capture")
        original_graph_stress(harness, snapshots, torch, dist)
        markers = state["markers"]
        calibration = calibrate_diagnostics(harness, snapshots, markers, diagnostic, torch, dist)
        graph = accuracy._capture_launch_only_graph(harness, torch, dist)
        routes = {}
        try:
            for route in accuracy.MATCHED_BENCHMARK_ROUTES:
                case = harness.cases[route]
                harness.copy_inputs(case)
                for _ in range(diagnostic.diagnostic_warmups):
                    graph.replay()
                torch.cuda.synchronize()
                harness.stats.zero_()
                samples = []
                for replay in range(diagnostic.diagnostic_replays):
                    # Diagnostic readback serializes launches by design. A fresh
                    # zero makes absence of a conditional marker unambiguous.
                    markers.zero_()
                    harness.output.fill_(float("nan"))
                    torch.cuda.synchronize()
                    dist.barrier()
                    torch.cuda.synchronize()
                    graph.replay()
                    torch.cuda.synchronize()
                    harness.assert_stats(case.expected_local_stats * (replay + 1),
                                         f"diagnostic/{route}/{replay}")
                    harness.assert_guards(f"diagnostic/{route}/{replay}")
                    accuracy._assert_bitwise_equal(harness.output, snapshots[route],
                                                  f"diagnostic/{route}/{replay}", torch)
                    dense = markers.cpu().tolist()
                    sparse = {
                        str(sm): {str(slot): value for slot, value in enumerate(row) if value}
                        for sm, row in enumerate(dense) if any(row)
                    }
                    samples.append({"rank": dist.get_rank(), "replay": replay,
                                    "sm_markers_ns": sparse})
                gathered = [None] * dist.get_world_size()
                dist.all_gather_object(gathered, samples)
                routes[route] = [sample for rank_samples in gathered for sample in rank_samples]
        finally:
            graph = None
        if _collective_kernel_configuration(args, diagnostic, dist) != state["kernel_configuration"]:
            raise RuntimeError("phase kernel configuration changed during capture")
        capture = {
            "schema": "megamoe-gin-phase-capture-v1", "performance_claim": False,
            "diagnostic_level": diagnostic.diagnostic_level,
            "kernel_configuration": state["kernel_configuration"],
            "kernel_configuration_collectively_validated_before_allocation": True,
            "kernel_configuration_stable_through_capture": True,
            "shape": {"tokens_per_rank": args.num_tokens, "hidden": args.hidden,
                      "intermediate_hidden": args.intermediate_hidden,
                      "num_experts": args.num_experts, "num_topk": args.num_topk,
                      "world_size": dist.get_world_size()},
            "routes": routes,
            "diagnostic_calibration": calibration,
        }
        # Validate and summarize on every rank before collective cleanup.
        state["capture"] = capture
        state["summary"] = summarize_capture(capture)
        state["summary"]["diagnostic_calibration"] = calibration

    rank = int(os.environ.get("RANK", "0"))
    with patch.object(deep_gemm, "get_symm_buffer_for_mega_moe", allocate), \
         patch.object(accuracy, "_init_distributed", initialize_then_validate), \
         patch.object(accuracy, "_run_graph_stress", graph_stress_then_capture):
        accuracy._worker(int(os.environ["LOCAL_RANK"]), 8, args)
    if rank == 0:
        state["capture"]["accuracy_and_teardown_passed"] = True
        with diagnostic.output.open("x") as output:
            json.dump(state["capture"], output, sort_keys=True)
            output.write("\n")
        summary_path = diagnostic.output.with_suffix(".summary.json")
        with summary_path.open("x") as output:
            json.dump(state["summary"], output, sort_keys=True, indent=2)
            output.write("\n")
        print("MEGAMOE_DIAGNOSTICS_JSON=" + json.dumps({
            "status": "passed", "capture": str(diagnostic.output),
            "summary": str(summary_path), "performance_claim": False,
        }), flush=True)


if __name__ == "__main__":
    main()
