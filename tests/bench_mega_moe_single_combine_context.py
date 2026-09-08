"""Accuracy-gated single-COMBINE-context sweep using one allocation.

Default modes 0,1,0 compare original all-context completion against an
eligible bulk/direct path whose eight peer lanes all use context 1 (one QP
per peer) and the public context-1 world Put barrier. Nine contexts remain
allocated; input publication is unchanged. T64 fallback uses the original
all-context barrier and all-local routes skip GIN combine synchronization.
Expert waves are pinned to 0 and barrier cooperation to one warp.

Run with two-node torchrun, eight ranks/node:
  python tests/bench_mega_moe_single_combine_context.py --decode-mns 8 \\
      --output /path/single-combine-context-mns8.json

Defaults are 204 measured and 60 conditioning replays per route. Every
mode must pass bitwise controls, payload epochs and 48→64→48 transitions.
Only complete accuracy and collective teardown writes a success artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import test_mega_moe_accuracy as accuracy

MODE_ENV = "DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT"
BARRIER_WIDTH_ENV = "DG_MEGAMOE_GIN_COMBINE_BARRIER_WARPS"
EXPERT_WIDTH_ENV = "DG_MEGAMOE_GIN_COMBINE_EXPERTS_PER_WAVE"
DEFAULT_MODES = (0, 1)
SOURCE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "tests/bench_mega_moe_single_combine_context.py",
    "tests/test_mega_moe_accuracy.py",
    "deep_gemm/mega/__init__.py",
    "csrc/apis/mega_gin.hpp",
    "csrc/apis/mega.hpp",
    "csrc/jit/compiler.hpp",
    "csrc/jit_kernels/heuristics/mega_moe.hpp",
    "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp",
    "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
    "deep_gemm/include/deep_gemm/comm/mega_moe_gin.cuh",
    "deep_gemm/include/deep_gemm/layout/mega_moe.cuh",
    "deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh",
)


def mode_schedule(modes, repeat_control):
    modes = list(modes)
    if not modes or modes[0] != 0:
        raise ValueError("the first mode must be the unchanged mode-0 control")
    if len(set(modes)) != len(modes):
        raise ValueError("modes must be unique; use --repeat-control for the final drift control")
    if any(mode not in DEFAULT_MODES for mode in modes):
        raise ValueError("modes must be selected from 0,1")
    return modes + ([0] if repeat_control else [])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", type=int, default=list(DEFAULT_MODES))
    parser.add_argument("--repeat-control", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--payload-per-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--transitions-per-mode", action=argparse.BooleanOptionalAction, default=True)
    options, remaining = parser.parse_known_args()
    try:
        options.schedule = mode_schedule(options.modes, options.repeat_control)
    except ValueError as exc:
        parser.error(str(exc))
    defaults = {
        "--decode-mns": "8", "--eager-iterations": "16", "--graph-replays": "32",
        "--benchmark-replays": "204", "--benchmark-conditioning-replays": "60",
        "--payload-epochs": "3", "--gin-completion-batch": "8",
        "--gin-combine-issue-wave": "8", "--gin-outbox-depth": "64",
        "--gin-combine-chunk-bytes": "7168",
    }
    def has(flag):
        return any(value == flag or value.startswith(flag + "=") for value in remaining)
    for flag, value in defaults.items():
        if not has(flag) and not (flag == "--decode-mns" and has("--num-tokens")):
            remaining.extend((flag, value))
    remaining.extend(("--k3", "--require-gin", "--require-cross-host",
                      "--gin-active-fast-path", "--gin-bulk-combine",
                      "--gin-direct-dispatch"))
    with patch.object(sys, "argv", [sys.argv[0], *remaining]):
        args = accuracy._parse_args()
    if args.benchmark_replays < 2 or args.gin_local_ablation_stages or args.fast_math:
        parser.error("sweep requires matched timing, no local ablation, and fast_math=False")
    if args.num_tokens not in (32, 40, 48):
        parser.error("target token counts are 32, 40 and 48; T64 is tested as a fallback transition")
    if options.payload_per_mode and args.payload_epochs < 2:
        parser.error("--payload-per-mode requires at least two payload epochs")
    options.benchmark_replays = args.benchmark_replays
    # The hook owns all timed work. The base worker still runs its full one-warp
    # accuracy/transition gates, but must not run a duplicate timed benchmark.
    args.benchmark_replays = 0
    return options, args


def source_fingerprint():
    files = {name: hashlib.sha256((SOURCE_ROOT / name).read_bytes()).hexdigest()
             for name in SOURCE_FILES}
    return {"combined_sha256": hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "files_sha256": files, "scope": "listed runner, harness, MegaMoE host/JIT/kernel sources"}


def _identity(harness):
    return {
        **accuracy._input_storage_pointers(harness.inputs, harness.buffer),
        "registered_buffer": int(harness.buffer.buffer.data_ptr()),
        "context_object": id(harness.buffer.gin_context),
        "weight_storage": tuple(int(tensor.data_ptr()) for matrix in
                                (harness.weights.l1, harness.weights.l2) for tensor in matrix),
    }


def _check_identity(harness, initial, label, dist):
    accuracy._collective_transition_check(label,
        [] if _identity(harness) == initial else ["allocation, input, context, or weight storage changed"], dist)


def _retire_graphs(graphs):
    # An exception traceback may retain a helper's graph argument even after
    # the caller clears its local reference. Retire the executable explicitly.
    active_error = sys.exc_info()[0] is not None
    first_error = None
    try:
        for graph in tuple(graphs.values()):
            try:
                graph.reset()
            except Exception as exc:
                if not active_error:
                    first_error = first_error or exc
                else:
                    print(f"CUDA graph retirement also failed: {exc}", file=sys.stderr, flush=True)
    finally:
        graphs.clear()
    if first_error is not None:
        raise first_error


def _check_graph_routes(harness, graph, snapshots, label, torch, dist):
    schedule = list(accuracy.ROUTE_MODES)
    if harness.args.gin_active_fast_path:
        schedule.extend(accuracy.GIN_ACTIVITY_TRANSITION_SEQUENCE)
    for step, name in enumerate(schedule):
        case = harness.cases[name]
        harness.copy_inputs(case)
        harness.stats.zero_()
        harness.output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        errors = []
        try:
            harness.assert_guards(label + "/" + name)
            harness.assert_stats(case.expected_local_stats, label + "/" + name)
            if not torch.equal(harness.output.view(torch.int16), snapshots[name].view(torch.int16)):
                errors.append("candidate graph differs bitwise from mode-0 snapshot")
        except AssertionError as exc:
            errors.append(str(exc))
        accuracy._collective_transition_check(f"{label}/{step}/{name}", errors, dist)
    return {"same_context_mode0_bitwise": True, "stats_and_canaries": True,
            "nan_output_prefill": True, "route_schedule": schedule}


def _payload_with_mode0_control(harness, mode, torch, dist):
    original_snapshot = accuracy._snapshot_and_check_oracles

    def snapshots_with_control(harness, torch, dist):
        with patch.dict(os.environ, {MODE_ENV: "0"}):
            baseline, _ = original_snapshot(harness, torch, dist)
        current, metrics = original_snapshot(harness, torch, dist)
        errors = [f"payload route {name} differs bitwise from mode-0 eager output"
                  for name, tensor in current.items()
                  if not torch.equal(tensor.view(torch.int16), baseline[name].view(torch.int16))]
        accuracy._collective_transition_check("single-combine-context/payload-mode0", errors, dist)
        return current, metrics

    if mode != 0:
        with patch.object(accuracy, "_snapshot_and_check_oracles", snapshots_with_control):
            result = accuracy._run_payload_epoch_validation(harness, torch, dist)
    else:
        result = accuracy._run_payload_epoch_validation(harness, torch, dist)
    result["same_context_mode0_per_epoch_bitwise"] = bool(mode != 0 and result["enabled"])
    return result


def _drift_record(records):
    if len(records) < 2 or records[-1]["single_combine_context"] != 0:
        return {"enabled": False}
    first, last = records[0]["timing"]["routes"], records[-1]["timing"]["routes"]
    routes = {}
    for name in accuracy.MATCHED_BENCHMARK_ROUTES:
        values = {}
        for key, nested in (("balanced", False), ("isolated", True)):
            earlier = first[name]["isolated"] if nested else first[name]
            later = last[name]["isolated"] if nested else last[name]
            before = earlier["max_rank_per_replay_us"]["p50"]
            after = later["max_rank_per_replay_us"]["p50"]
            values[key] = {"first_p50_us": before, "last_p50_us": after,
                           "last_minus_first_us": after - before, "last_over_first": after / before}
        routes[name] = values
    return {"enabled": True, "automatic_drift_correction": False, "routes": routes}


def effective_combine_contexts(requested, tokens, remote_active, *, bulk_eligible=True,
                               direct_dispatch=True):
    """Known K3 policy, not an observation of compiled CUDA participation."""
    if requested not in DEFAULT_MODES:
        raise ValueError("single combine context mode must be 0 or 1")
    if not remote_active:
        return []
    if requested and bulk_eligible and direct_dispatch and tokens <= 48:
        return [1]
    return list(range(9))


def context_evidence(requested, tokens):
    contexts = {
        route: effective_combine_contexts(requested, tokens, route != "all_same_host")
        for route in accuracy.MATCHED_BENCHMARK_ROUTES
    }
    return {
        "single_combine_context_requested": bool(requested),
        "allocated_context_count": 9,
        "input_data_contexts_unchanged": list(range(1, 9)),
        "eligible_combine_put_contexts": [1] if requested else list(range(1, 9)),
        "effective_combine_barrier_contexts_by_route": contexts,
        "context_peer_pairs_by_route": {route: len(value) * 16 for route, value in contexts.items()},
        "effective_context_basis": "known uniform K3 eligibility contract, not kernel introspection",
        "empty_context_list_means": "local-fast-path skips GIN combine barrier",
        "remote_active_t48_t64_t48": [effective_combine_contexts(requested, count, True)
                                      for count in (48, 64, 48)],
        "public_put_visibility_semantics_preserved": True,
        "experts_per_wave": 0,
        "barrier_warps": 1,
    }


def prepare_sweep_environment():
    """Reject retired experiments, then pin control before graph capture."""
    if (os.environ.get(EXPERT_WIDTH_ENV, "0") != "0"
            or os.environ.get(BARRIER_WIDTH_ENV, "1") != "1"):
        raise ValueError("retired experiments are unsupported; requires expert waves 0 and barrier warps 1")
    os.environ[MODE_ENV] = "0"
    os.environ[BARRIER_WIDTH_ENV] = "1"
    os.environ[EXPERT_WIDTH_ENV] = "0"
    os.environ["DG_MEGAMOE_GIN_DIAGNOSTICS"] = "0"
    os.environ[accuracy.GIN_LOCAL_ABLATION_ENV] = "0"
    for name in (accuracy.GIN_ACTIVITY_GATE_OPT_ENV, *accuracy.GIN_EXPERIMENT_FLAG_ENVS):
        os.environ[name] = "1"


def main():
    options, args = parse_args()
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("launch under two-node torchrun, eight ranks per node")
    prepare_sweep_environment()
    provenance = source_fingerprint()
    torch, dist, deep_gemm = accuracy._load_runtime()
    original_graph_stress = accuracy._run_graph_stress
    original_weight_fingerprint = accuracy._weight_fingerprint
    state = {}

    def retain_weight_fingerprint(*positional):
        result = original_weight_fingerprint(*positional)
        state["weight_fingerprint"] = result
        return result

    def graph_stress_then_sweep(harness, snapshots, torch, dist):
        configuration = {
            "schedule": options.schedule, "benchmark_replays": options.benchmark_replays,
            "benchmark_conditioning_replays": args.benchmark_conditioning_replays,
            "payload_per_mode": options.payload_per_mode,
            "transitions_per_mode": options.transitions_per_mode,
            "args": dict(vars(args)), "source_sha256": provenance["combined_sha256"],
            # Cache-directory paths may legitimately differ across hosts;
            # compare the compile-time MegaMoE behavior flags collectively.
            "environment": {name: value for name, value in sorted(os.environ.items())
                            if name.startswith("DG_MEGAMOE_")},
        }
        configurations = [None] * dist.get_world_size()
        dist.all_gather_object(configurations, configuration)
        errors = []
        if any(value != configuration for value in configurations):
            errors.append("sweep configuration/source/environment differs across ranks")
        if Path(deep_gemm.__file__).resolve().parent != SOURCE_ROOT / "deep_gemm":
            errors.append("loaded DeepGEMM package does not match the fingerprinted source tree")
        if dist.get_rank() == 0 and (options.output.exists() or not options.output.parent.is_dir()):
            errors.append("output must not exist and its parent directory must exist")
        accuracy._collective_transition_check("single-combine-context/preflight", errors, dist)
        # This is the original one-warp correctness stress, not a weakened sweep-only
        # substitute. Its graph is immediately retired before capturing others.
        original_graph_stress(harness, snapshots, torch, dist)
        initial_identity = _identity(harness)
        hostnames = accuracy._all_hostnames(dist)
        records = []
        for index, mode in enumerate(options.schedule):
            label = f"entry={index}/single_combine_context={mode}"
            graph = None
            timing_graphs = {}
            transition_graphs = {}
            with patch.dict(os.environ, {MODE_ENV: str(mode)}):
                try:
                    if dist.get_rank() == 0:
                        print("MEGAMOE_SINGLE_COMBINE_CONTEXT_STAGE=" + label + "/capture", flush=True)
                    graph = accuracy._capture_launch_only_graph(harness, torch, dist)
                    timing_graphs["launch_only"] = graph
                    checks = _check_graph_routes(harness, graph, snapshots, label, torch, dist)
                    # The helper only changes routes, so explicitly reestablish
                    # the fixed epoch-0 activation/SF payload before timing.
                    harness.copy_inputs(harness.cases[accuracy.MATCHED_BENCHMARK_ROUTES[0]])
                    if dist.get_rank() == 0:
                        print("MEGAMOE_SINGLE_COMBINE_CONTEXT_STAGE=" + label + "/benchmark", flush=True)
                    with patch.object(args, "benchmark_replays", options.benchmark_replays):
                        timing = accuracy._benchmark_graph_routes(harness, graph, snapshots, torch, dist)
                    _retire_graphs(timing_graphs)
                    graph = None
                    payload = (_payload_with_mode0_control(harness, mode, torch, dist)
                               if options.payload_per_mode else {"enabled": False, "reason": "disabled_per_mode"})
                    transition = (accuracy._run_gin_bulk_transition_validation(
                        harness, harness.rank, args, hostnames, transition_graphs,
                        torch, dist, harness.deep_gemm)
                        if options.transitions_per_mode else {"enabled": False, "reason": "disabled_per_mode"})
                    # The transition helper deliberately overwrites fixed-MNS
                    # input-tail canaries. Its own checks have completed; restore
                    # this harness's canaries for the next mode and base worker.
                    if transition["enabled"]:
                        harness._initialize_input_tail_canaries()
                    harness.copy_inputs(harness.cases["all_local"])
                    harness.stats.zero_()
                    torch.cuda.synchronize()
                    _check_identity(harness, initial_identity, label + "/identity", dist)
                    record = {"entry_index": index, "single_combine_context": mode,
                              "context_contract": context_evidence(mode, args.num_tokens),
                              "schedule_kind": "post_compute_full_packet_context_selection",
                              "environment": {**configuration["environment"], MODE_ENV: str(mode)},
                              "checks": checks, "payload_validation": payload,
                              "fast_path_transition": transition, "timing": timing}
                    records.append(record)
                    if dist.get_rank() == 0:
                        print("MEGAMOE_SINGLE_COMBINE_CONTEXT_PENDING_JSON=" + json.dumps({
                            "status": "pending_remaining_accuracy_and_teardown", "entry": index,
                            "single_combine_context": mode,
                            "isolated_p50_us": {name: item["isolated"]["max_rank_per_replay_us"]["p50"]
                                                for name, item in timing["routes"].items()}}), flush=True)
                finally:
                    _retire_graphs(timing_graphs)
                    graph = None
                    _retire_graphs(transition_graphs)
        state["result"] = {
            "schema": "megamoe-single-combine-context-sweep-v1", "configuration": configuration,
            "shape": {"tokens_per_rank": args.num_tokens, "hidden": args.hidden,
                      "intermediate_hidden": args.intermediate_hidden, "experts": args.num_experts,
                      "topk": args.num_topk, "world_size": dist.get_world_size()},
            "source": provenance, "source_root": str(SOURCE_ROOT),
            "loaded_package": str(Path(deep_gemm.__file__).resolve()),
            "loaded_extension": str(Path(deep_gemm._C.__file__).resolve()),
            "hostnames": hostnames, "weight_fingerprint": state["weight_fingerprint"],
            "same_allocation_context_weights_and_input_storage": True,
            "timing_scope": "launch_only_cuda_graph_device_time",
            "aggregation": "slowest_rank_for_each_replay",
            "variants_separately_conditioned": True,
            "fixed_experts_per_wave": 0,
            "same_input_and_payload_bytes": True,
            "context_mapping_and_barrier_participation_changed": True,
            "fixed_barrier_warps": 1,
            "dense_rank_distinct_gate_is_separate": True,
            "modes": records, "repeated_mode0_drift": _drift_record(records),
        }
        accuracy._collective_transition_check("single-combine-context/source-unchanged",
            [] if source_fingerprint() == provenance else ["fingerprinted sources changed during sweep"], dist)

    with patch.object(accuracy, "_run_graph_stress", graph_stress_then_sweep), \
         patch.object(accuracy, "_weight_fingerprint", retain_weight_fingerprint):
        accuracy._worker(int(os.environ["LOCAL_RANK"]), 8, args)
    # _worker has completed its remaining one-warp payload/fallback gates and has
    # returned from collective symmetric-memory/context teardown successfully.
    if int(os.environ["RANK"]) == 0:
        result = state["result"]
        result.update(status="passed", accuracy_and_teardown_passed=True)
        with options.output.open("x") as output:
            json.dump(result, output, sort_keys=True)
            output.write("\n")
        print("MEGAMOE_SINGLE_COMBINE_CONTEXT_SWEEP_JSON=" + json.dumps({
            "status": "passed", "output": str(options.output),
            "schedule": options.schedule, "accuracy_and_teardown_passed": True}), flush=True)


if __name__ == "__main__":
    main()
