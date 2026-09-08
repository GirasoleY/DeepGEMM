"""Accuracy-gated fused GIN dispatch-overlap experiment; no compute tuning.

Run under two-node torchrun (eight ranks/node), for example:
  python tests/bench_mega_moe_dispatch_overlap.py --decode-mns 8 --output RESULT.json

Dispatch modes default to0,1,0; single-context COMBINE stays1 in every mode.
Timing uses unchanged balanced/isolated fixtures. Untimed gates also exercise
fixed-T severe owner imbalance, changing FP8/SF payloads and T48→64→48 fallback.
Only a return from the complete base worker/collective teardown writes success.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import bench_mega_moe_single_combine_context as common
import test_mega_moe_accuracy as accuracy


MODE_ENV = "DG_MEGAMOE_GIN_DISPATCH_OVERLAP"
COMBINE_ENV = common.MODE_ENV
SOURCE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = tuple(dict.fromkeys(("tests/bench_mega_moe_dispatch_overlap.py", *common.SOURCE_FILES)))
IMBALANCE_SEQUENCE = ("all_remote", "hot_owner8", "hot_owner0", "all_remote",
                      "all_masked", "hot_owner8", "single_active_hot_owner8", "all_remote")


@dataclass(frozen=True)
class SweepAxis:
    """Explicit transport axis; shared gates and timing stay identical."""
    name: str
    mode_env: str
    fixed_environment: tuple
    extra_sources: tuple = ()

    @property
    def label(self):
        return self.name + "-overlap"

    @property
    def record_key(self):
        return self.name + "_overlap"

    @property
    def control(self):
        return self.name + "0"


DISPATCH_AXIS = SweepAxis("dispatch", MODE_ENV, (
    (COMBINE_ENV, "1"), (accuracy.GIN_COMBINE_OVERLAP_ENV, "0")))


def parse_args(axis=DISPATCH_AXIS):
    options, args = common.parse_args()
    if not options.payload_per_mode or not options.transitions_per_mode:
        raise ValueError(f"{axis.name} sweep requires payload and fallback validation in every mode")
    if options.schedule != [0, 1, 0]:
        raise ValueError(f"{axis.name} sweep requires the full0,1,0 control/candidate/control schedule")
    return options, args


def prepare_sweep_environment(axis=DISPATCH_AXIS):
    common.prepare_sweep_environment()
    os.environ.update(dict(axis.fixed_environment))
    os.environ[axis.mode_env] = "0"


def source_fingerprint(axis=DISPATCH_AXIS):
    files = {name: hashlib.sha256((SOURCE_ROOT / name).read_bytes()).hexdigest()
             for name in (*SOURCE_FILES, *axis.extra_sources)}
    return {"combined_sha256": hashlib.sha256(json.dumps(
        files, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "files_sha256": files,
        "scope": "listed runner, imported harnesses, MegaMoE host/JIT/kernel sources"}


def hot_assignment(source, slot, owner, *, only_source=None):
    if not 0 <= source < 16 or not 0 <= slot < 16 or owner not in (0, 8):
        raise ValueError("hot fixture requires EP16/top-k16 and owner0 or8")
    return -1 if only_source is not None and source != only_source else owner * 56 + slot


def dispatch_contract(mode):
    if mode not in (0, 1):
        raise ValueError("dispatch overlap mode must be0 or1")
    return {"requested_dispatch_overlap": bool(mode), "fixed_single_combine_context": 1,
            "compute_tiling_sm_count_and_math_unchanged": True,
            "effective_policy_not_device_observation": True,
            "eligible_control_before_payload": bool(mode),
            "payload_terminal_required_before_payload_load": bool(mode),
            "direct_t48_t64_t48_split_policy": [bool(mode), False, bool(mode)],
            "fallback_advances_neither_direct_terminal": True,
            "single_host_fast_path_skips_remote_dispatch": True}


def _payload_with_control(harness, mode, torch, dist, axis=DISPATCH_AXIS):
    original = accuracy._snapshot_and_check_oracles

    def compare(harness, torch, dist):
        with patch.dict(os.environ, {axis.mode_env: "0"}):
            baseline, _ = original(harness, torch, dist)
        current, metrics = original(harness, torch, dist)
        errors = [f"payload expert assignments {name} differ from {axis.control} output"
                  for name, value in current.items()
                  if not torch.equal(value.view(torch.int16), baseline[name].view(torch.int16))]
        accuracy._collective_transition_check(axis.label + "/payload-control", errors, dist)
        return current, metrics

    with patch.object(accuracy, "_snapshot_and_check_oracles", compare if mode else original):
        result = accuracy._run_payload_epoch_validation(harness, torch, dist)
    result[f"same_context_{axis.control}_per_epoch_bitwise"] = bool(mode and result["enabled"])
    return result


def _imbalance_cases(harness, torch, dist):
    cases = {name: harness.cases[name] for name in ("all_remote", "all_masked")}
    for name, owner, only_source in (("hot_owner8", 8, None), ("hot_owner0", 0, None),
                                    ("single_active_hot_owner8", 8, 0)):
        ids = [[hot_assignment(harness.rank, slot, owner, only_source=only_source)
                for slot in range(16)] for _ in range(harness.args.num_tokens)]
        weights = [[2.0 ** (-(slot % 3 + 1)) if expert >= 0 else 0.
                    for slot, expert in enumerate(row)] for row in ids]
        idx = torch.tensor(ids, dtype=torch.int64, device="cuda")
        weight = torch.tensor(weights, dtype=torch.float32, device="cuda")
        counts = accuracy._global_route_stats(idx, harness.rank, 56, torch, dist)
        cases[name] = accuracy.RouteCase(name, idx, weight, counts, "fixed_shape_hot_owner")
    return cases


def _check_imbalance_output(harness, case, reference, expected, label, torch, dist):
    relative, maximum = accuracy._symmetric_diff(harness.output, expected)
    limits = accuracy._oracle_thresholds(harness.args)
    errors = []
    if not all(map(math.isfinite, (relative, maximum))) or relative > limits[0] or maximum > limits[1]:
        errors.append(f"native sparse oracle mismatch: symmetric_diff={relative}, max_abs={maximum}")
    if reference is not None and not torch.equal(harness.output.view(torch.int16), reference.view(torch.int16)):
        errors.append("output differs bitwise from same-context mode0 graph")
    try:
        harness.assert_guards(label)
        harness.assert_stats(case.expected_local_stats, label)
        if case.name == "all_masked" and not torch.equal(harness.output, torch.zeros_like(harness.output)):
            errors.append("masked output is not exactly zero")
    except AssertionError as error:
        errors.append(str(error))
    accuracy._collective_transition_check(label, errors, dist)
    metrics = torch.tensor((relative, maximum), dtype=torch.float64, device="cuda")
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
    return dict(zip(("worst_symmetric_diff", "worst_max_abs"), metrics.cpu().tolist()))


def _fixed_shape_imbalance(harness, graph, mode, torch, dist, axis=DISPATCH_AXIS):
    """One candidate graph, fixed shape/addresses, exact baseline per payload."""
    initial = common._identity(harness)
    cases = _imbalance_cases(harness, torch, dist)
    graphs = {}
    records = []
    epochs = accuracy._payload_epoch_sequence(harness.args.payload_epochs)
    stress_args = SimpleNamespace(**{**vars(harness.args), "heterogeneous_input_scales": True})
    counts = {name: [None] * dist.get_world_size() for name in cases}
    for name, case in cases.items():
        dist.all_gather_object(counts[name], case.expected_local_stats.cpu().tolist())
    try:
        with patch.dict(os.environ, {axis.mode_env: "0"}):
            baseline = accuracy._capture_launch_only_graph(harness, torch, dist)
            graphs[axis.control] = baseline
        references = {}
        # Build all controls first. Interleaving a same-payload baseline before
        # every candidate could refresh stale receive slabs and conceal a bug.
        for epoch in dict.fromkeys(epochs):
            fresh = accuracy._make_inputs(harness.rank, stress_args, torch, harness.deep_gemm,
                                          input_epoch=epoch)
            accuracy._copy_input_storage(harness.inputs, fresh)
            for name, case in cases.items():
                expected = accuracy._analytic_oracle(harness.inputs.oracle_x, case,
                    harness.args, harness.experts_per_rank, torch)
                harness.copy_inputs(case)
                harness.stats.zero_()
                harness.output.fill_(float("nan"))
                baseline.replay()
                torch.cuda.synchronize()
                label = f"{axis.label}/mode{mode}/fixed-T/control/epoch{epoch}/{name}"
                _check_imbalance_output(harness, case, None, expected, label, torch, dist)
                references[(epoch, name)] = (harness.output.clone(), expected)
        for epoch in epochs:
            fresh = accuracy._make_inputs(harness.rank, stress_args, torch, harness.deep_gemm,
                                          input_epoch=epoch)
            accuracy._copy_input_storage(harness.inputs, fresh)
            for step, name in enumerate(IMBALANCE_SEQUENCE):
                case = cases[name]
                control, expected = references[(epoch, name)]
                harness.copy_inputs(case)
                harness.stats.zero_()
                harness.output.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                label = f"{axis.label}/mode{mode}/fixed-T/epoch{epoch}/{step}/{name}"
                metrics = _check_imbalance_output(harness, case, control, expected, label, torch, dist)
                records.append({"epoch": epoch, "step": step, "expert_assignment_case": name, **metrics})
        common._check_identity(harness, initial, axis.label + "/fixed-T-identity", dist)
    finally:
        common._retire_graphs(graphs)
        fresh = accuracy._make_inputs(harness.rank, harness.args, torch, harness.deep_gemm, input_epoch=0)
        accuracy._copy_input_storage(harness.inputs, fresh)
        harness.copy_inputs(harness.cases["all_local"])
        harness.stats.zero_()
        torch.cuda.synchronize()
    return {"enabled": True, "timed": False, "single_candidate_graph": True,
            "fixed_tokens_per_rank": harness.args.num_tokens, "sequence": list(IMBALANCE_SEQUENCE),
            "payload_epochs": epochs, f"same_context_{axis.control}_bitwise": True,
            "candidate_sequence_has_no_interleaved_control_replays": True,
            "heterogeneous_input_scales_untimed_only": True,
            "exact_per_rank_expert_assignment_counts": counts,
            "sparse_native_oracle": True, "dense_numerical_claim": False,
            "stats_canaries_nan_prefill": True, "same_storage_and_compute_configuration": True,
            "checks": records}


def _drift(records, axis=DISPATCH_AXIS):
    # Reuse only the arithmetic; the output has no combine-mode labels.
    adapted = [{"single_combine_context": record[axis.record_key], "timing": record["timing"]}
               for record in records]
    return common._drift_record(adapted)


def main(axis=DISPATCH_AXIS, contract=dispatch_contract):
    options, args = parse_args(axis)
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("launch under two-node torchrun, eight ranks per node")
    prepare_sweep_environment(axis)
    provenance = source_fingerprint(axis)
    log_prefix = "MEGAMOE_" + axis.name.upper() + "_OVERLAP_"
    torch, dist, deep_gemm = accuracy._load_runtime()
    original_graph_stress = accuracy._run_graph_stress
    original_weight_fingerprint = accuracy._weight_fingerprint
    state = {}

    def retain_weights(*positional):
        result = original_weight_fingerprint(*positional)
        state["weight_fingerprint"] = result
        return result

    def sweep(harness, snapshots, torch, dist):
        configuration = {
            "sweep_axis": axis.name, "fixed_environment": dict(axis.fixed_environment),
            "schedule": options.schedule, "benchmark_replays": options.benchmark_replays,
            "benchmark_conditioning_replays": args.benchmark_conditioning_replays,
            "args": dict(vars(args)), "source_sha256": provenance["combined_sha256"],
            "environment": {name: value for name, value in sorted(os.environ.items())
                            if name.startswith("DG_MEGAMOE_")},
        }
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, configuration)
        errors = []
        if any(value != configuration for value in gathered):
            errors.append(f"{axis.name} sweep configuration/source/environment differs across ranks")
        if Path(deep_gemm.__file__).resolve().parent != SOURCE_ROOT / "deep_gemm":
            errors.append("loaded DeepGEMM is not the fingerprinted source tree")
        if dist.get_rank() == 0 and (options.output.exists() or not options.output.parent.is_dir()):
            errors.append("output must be absent and parent directory must exist")
        accuracy._collective_transition_check(axis.label + "/preflight-before-candidate", errors, dist)
        original_graph, _, _ = original_graph_stress(harness, snapshots, torch, dist)
        common._retire_graphs({"base_stress": original_graph})
        initial = common._identity(harness)
        hostnames = accuracy._all_hostnames(dist)
        records = []
        for entry, mode in enumerate(options.schedule):
            graphs, transition_graphs = {}, {}
            with patch.dict(os.environ, {axis.mode_env: str(mode)}):
                try:
                    label = f"entry{entry}/{axis.record_key}{mode}"
                    # Runtime mode switches reuse one communicator. Confirm
                    # raw flags on all ranks before each candidate JIT/capture.
                    mode_flags = accuracy._collect_gin_experiment_flags(
                        args, harness.rank, dist.get_world_size(), dist)
                    if dist.get_rank() == 0:
                        print(log_prefix + "STAGE=" + label + "/capture", flush=True)
                    graph = accuracy._capture_launch_only_graph(harness, torch, dist)
                    graphs["candidate"] = graph
                    checks = common._check_graph_routes(harness, graph, snapshots, label, torch, dist)
                    checks[f"same_context_{axis.control}_bitwise"] = checks.pop("same_context_mode0_bitwise")
                    harness.copy_inputs(harness.cases[accuracy.MATCHED_BENCHMARK_ROUTES[0]])
                    with patch.object(args, "benchmark_replays", options.benchmark_replays):
                        timing = accuracy._benchmark_graph_routes(harness, graph, snapshots, torch, dist)
                    imbalance = _fixed_shape_imbalance(harness, graph, mode, torch, dist, axis)
                    common._retire_graphs(graphs)
                    graph = None
                    payload = _payload_with_control(harness, mode, torch, dist, axis)
                    transition = accuracy._run_gin_bulk_transition_validation(
                        harness, harness.rank, args, hostnames, transition_graphs,
                        torch, dist, harness.deep_gemm)
                    if transition["enabled"]:
                        harness._initialize_input_tail_canaries()
                    harness.copy_inputs(harness.cases["all_local"])
                    harness.stats.zero_()
                    torch.cuda.synchronize()
                    common._check_identity(harness, initial, label + "/identity", dist)
                    records.append({"entry_index": entry, axis.record_key: mode,
                        axis.name + "_contract": contract(mode),
                        "collectively_validated_mode_flags": mode_flags,
                        "environment": {**configuration["environment"], axis.mode_env: str(mode)},
                        "checks": checks, "fixed_shape_imbalance": imbalance,
                        "payload_validation": payload, "fast_path_transition": transition, "timing": timing})
                    if dist.get_rank() == 0:
                        print(log_prefix + "PENDING_JSON=" + json.dumps({
                            "status": "pending_remaining_accuracy_and_teardown", "entry": entry,
                            axis.record_key: mode,
                            "isolated_p50_us": {name: value["isolated"]["max_rank_per_replay_us"]["p50"]
                                                for name, value in timing["routes"].items()}}), flush=True)
                finally:
                    common._retire_graphs(graphs)
                    common._retire_graphs(transition_graphs)
        state["result"] = {
            "schema": f"megamoe-{axis.label}-sweep-v1", "configuration": configuration,
            "shape": {"tokens_per_rank": args.num_tokens, "hidden": args.hidden,
                "intermediate_hidden": args.intermediate_hidden, "experts": args.num_experts,
                "topk": args.num_topk, "world_size": dist.get_world_size()},
            "source": provenance, "source_root": str(SOURCE_ROOT),
            "loaded_package": str(Path(deep_gemm.__file__).resolve()),
            "loaded_extension": str(Path(deep_gemm._C.__file__).resolve()),
            "hostnames": hostnames, "weight_fingerprint": state["weight_fingerprint"],
            "same_allocation_context_weights_and_input_storage": True,
            "same_input_and_payload_bytes": True, "fixed_single_combine_context": 1,
            "fixed_transport_environment": dict(axis.fixed_environment),
            "compute_tiling_sm_count_and_math_unchanged": True,
            "timing_scope": "launch_only_cuda_graph_device_time",
            "aggregation": "slowest_rank_for_each_replay", "variants_separately_conditioned": True,
            "dense_rank_distinct_gate_is_separate": True,
            "modes": records, f"repeated_{axis.control}_drift": _drift(records, axis),
        }
        accuracy._collective_transition_check(axis.label + "/source-unchanged",
            [] if source_fingerprint(axis) == provenance else ["fingerprinted source changed during sweep"], dist)

    with patch.object(accuracy, "_run_graph_stress", sweep), \
         patch.object(accuracy, "_weight_fingerprint", retain_weights):
        accuracy._worker(int(os.environ["LOCAL_RANK"]), 8, args)
    if int(os.environ["RANK"]) == 0:
        result = state["result"]
        result.update(status="passed", accuracy_and_teardown_passed=True)
        with options.output.open("x") as output:
            json.dump(result, output, sort_keys=True)
            output.write("\n")
        print(log_prefix + "SWEEP_JSON=" + json.dumps({
            "status": "passed", "output": str(options.output), "schedule": options.schedule,
            "accuracy_and_teardown_passed": True}), flush=True)


if __name__ == "__main__":
    main()
