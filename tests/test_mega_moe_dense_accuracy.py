"""Dense, rank-distinct quantization-aware MegaMoE accuracy gate.

Run with two-node torchrun, eight ranks/node. Example:
  python tests/test_mega_moe_dense_accuracy.py --decode-mns 8 \
      --dense-routes all_same_host half_remote all_remote --output dense.json

This is an additional accuracy test, not a performance benchmark. Each rank
owns 56 distinct random dense experts at H3584/I3072. The independent reference
uses dequantized canonical MXFP4 weights and FP32 torch matmuls, BF16 gate/up
rounding, SwiGLU, top-k weighting before MXFP8 requantization, BF16 expert
outputs, and an ordered FP32 top-k sum. No fused MegaMoE output is the oracle.

The numerical gate combines relative L2 <= 0.001 with a reference-derived
per-element rounding budget: one BF16 ULP per expert contribution plus one
final BF16 rounding ULP. This accounts for cancellation when one expert's GEMM
accumulator lands on a different BF16 rounding boundary. It is a tolerance
budget, not a proof about arbitrary intermediate FP8 errors. Historical fixed
absolute-gate failures remain reported. Enabled combine overlap must match
combine0 BITWISE while dispatch overlap and single-context COMBINE stay1.
Otherwise enabled dispatch overlap must match
dispatch0 BITWISE while single-context COMBINE stays1. With overlap disabled,
enabled single-context mode retains its original all-context-COMBINE baseline.
Both comparisons use identical inputs, weights, context and compute settings.
Retired expert waves and cooperative barrier widths remain rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch

import test_mega_moe_accuracy as accuracy

EXPERT_WIDTH_ENV = "DG_MEGAMOE_GIN_COMBINE_EXPERTS_PER_WAVE"
BARRIER_WIDTH_ENV = "DG_MEGAMOE_GIN_COMBINE_BARRIER_WARPS"
SINGLE_CONTEXT_ENV = "DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT"
DISPATCH_OVERLAP_ENV = "DG_MEGAMOE_GIN_DISPATCH_OVERLAP"
COMBINE_OVERLAP_ENV = "DG_MEGAMOE_GIN_COMBINE_OVERLAP"


def source_fingerprint():
    root = Path(__file__).resolve().parents[1]
    names = ("tests/test_mega_moe_dense_accuracy.py", "tests/test_mega_moe_accuracy.py",
             "deep_gemm/mega/__init__.py", "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp",
             "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
             "deep_gemm/include/deep_gemm/comm/mega_moe_gin.cuh",
             "deep_gemm/include/deep_gemm/layout/mega_moe.cuh")
    files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    return {"files_sha256": files, "combined_sha256": hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "scope": "listed source files, not loaded-binary attestation"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dense-routes", nargs="+", choices=accuracy.MATCHED_BENCHMARK_ROUTES,
                        default=list(accuracy.MATCHED_BENCHMARK_ROUTES))
    parser.add_argument("--dense-epochs", type=int, default=3)
    parser.add_argument("--dense-seed", type=int, default=20260906)
    parser.add_argument("--dense-rel-l2", type=float, default=0.001)
    parser.add_argument("--dense-max-abs", type=float, default=0.0625,
                        help="Legacy absolute diagnostic floor, not the rounding-budget gate")
    options, remaining = parser.parse_known_args()
    if options.dense_epochs < 2:
        parser.error("at least two changing payload epochs are required")
    if len(set(options.dense_routes)) != len(options.dense_routes):
        parser.error("dense routes must be unique")
    if not (math.isfinite(options.dense_rel_l2) and options.dense_rel_l2 > 0 and
            math.isfinite(options.dense_max_abs) and options.dense_max_abs > 0):
        parser.error("accuracy tolerances must be finite and positive")
    def has(flag):
        return any(value == flag or value.startswith(flag + "=") for value in remaining)
    for flag, value in (("--decode-mns", "8"), ("--eager-iterations", "1"),
                        ("--graph-replays", "1"), ("--gin-completion-batch", "8"),
                        ("--gin-combine-issue-wave", "8"), ("--gin-outbox-depth", "64")):
        if not has(flag) and not (flag == "--decode-mns" and has("--num-tokens")):
            remaining.extend((flag, value))
    remaining.extend(("--k3", "--require-gin", "--require-cross-host",
                      "--gin-active-fast-path", "--gin-bulk-combine",
                      "--gin-direct-dispatch", "--heterogeneous-input-scales"))
    with patch.object(sys, "argv", [sys.argv[0], *remaining]):
        args = accuracy._parse_args()
    if (args.mma_type != "fp8xfp4" or args.fast_math or args.benchmark_replays or
            args.gin_local_ablation_stages):
        parser.error("dense oracle requires fp8xfp4, fast_math=False, and no benchmark/ablation")
    if args.num_tokens not in (32, 40, 48):
        parser.error("dense gate targets 32/40/48 tokens per rank")
    return options, args


def make_dense_weights(rank, experts, args, options, torch, dg):
    """Quantize one dense expert at a time; retain exact dequantized operands."""
    from deep_gemm.utils import per_token_cast_to_fp4
    from deep_gemm.utils.math import cast_back_from_fp4

    canonical, reference, summaries = [], [], []
    digest = hashlib.sha256()
    for matrix, (n, k) in enumerate(((2 * args.intermediate_hidden, args.hidden),
                                    (args.hidden, args.intermediate_hidden))):
        packed = torch.empty((experts, n, k // 2), device="cuda", dtype=torch.int8)
        scales = torch.empty((experts, n, k // 32), device="cuda", dtype=torch.float32)
        restored = []
        block = torch.arange(k // 32, device="cuda").view(1, -1)
        rows = torch.arange(n, device="cuda").view(-1, 1)
        metrics = []
        for expert in range(experts):
            global_expert = rank * experts + expert
            generator = torch.Generator(device="cuda")
            generator.manual_seed(options.dense_seed + global_expert * 1009 + matrix * 104729)
            dense = torch.randn((n, k), generator=generator, device="cuda", dtype=torch.float32)
            # Different powers of two across rows and K groups guarantee that
            # the SF path cannot pass by treating scales as one constant.
            exponent = ((rows // 8 + block * 3 + global_expert + matrix) % 5) - 2
            amplitude = torch.pow(2.0, exponent.float()) / math.sqrt(k)
            dense = (dense.view(n, k // 32, 32) * amplitude[..., None]).view(n, k)
            dense = dense.to(torch.bfloat16)
            packed[expert], scales[expert] = per_token_cast_to_fp4(
                dense, use_ue8m0=True, gran_k=32)
            restored.append(cast_back_from_fp4(packed[expert], scales[expert], gran_k=32))
            sample = packed[expert].flatten()[::max(1, packed[expert].numel() // 4096)]
            sf_sample = scales[expert].flatten()[::max(1, scales[expert].numel() // 4096)]
            nonzero = (((sample & 7) != 0).float().mean() +
                       (((sample >> 4) & 7) != 0).float().mean()) * 0.5
            metrics.append((float(nonzero.item()), float(scales[expert].min().item()),
                            float(scales[expert].max().item())))
            digest.update(sample.contiguous().cpu().numpy().tobytes())
            digest.update(sf_sample.contiguous().cpu().numpy().tobytes())
            del dense
        min_density = min(record[0] for record in metrics)
        if min_density < 0.5 or not all(lo < hi for _, lo, hi in metrics):
            raise AssertionError("dense weight generator did not exercise nonzero data and varying SF")
        summaries.append({"matrix": matrix, "sampled_min_nonzero_fraction": min_density,
                          "sf_min": min(record[1] for record in metrics),
                          "sf_max": max(record[2] for record in metrics)})
        canonical.append((packed, dg.transform_sf_into_required_layout(
            scales, n, k, (1, 32), experts)))
        reference.append(restored)
    l1, l2 = dg.transform_weights_for_mega_moe(*canonical)
    return accuracy.KernelWeights(l1, l2), reference, {
        "rank": rank, "sampled_canonical_sha256": digest.hexdigest(),
        "seed": options.dense_seed, "matrices": summaries,
    }


def dense_reference(harness, case, reference_weights):
    """Each owner calculates its distinct experts, then returns reference slots."""
    torch, dist, args = harness.torch, harness.dist, harness.args
    from deep_gemm.utils import per_token_cast_to_fp8
    from deep_gemm.utils.math import cast_back_from_fp8

    def gather(tensor):
        values = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(values, tensor.contiguous())
        return torch.stack(values)

    x = gather(harness.inputs.oracle_x)
    topk = gather(case.topk_idx)
    route_weights = gather(case.topk_weights)
    # One owner writes each (source,token,topk) slot. All-reduce sums exactly one
    # BF16-representable contribution plus zeros; top-k accumulation is later
    # performed in the same slot order as the fused reduction.
    slots = torch.zeros((*topk.shape, args.hidden), device="cuda", dtype=torch.float32)
    for expert in range(harness.experts_per_rank):
        positions = torch.nonzero(topk == harness.rank * harness.experts_per_rank + expert)
        if positions.shape[0] == 0:
            continue
        source, token, slot = positions.unbind(dim=1)
        first = x[source, token].float() @ reference_weights[0][expert].t()
        gate, up = first.to(torch.bfloat16).float().chunk(2, dim=1)
        if math.isfinite(args.activation_clamp):
            gate = gate.clamp_max(args.activation_clamp)
            up = up.clamp(-args.activation_clamp, args.activation_clamp)
        intermediate = (gate / (1.0 + torch.exp(-gate))) * up
        intermediate = intermediate * route_weights[source, token, slot, None]
        quantized, sf = per_token_cast_to_fp8(
            intermediate, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        intermediate = cast_back_from_fp8(quantized, sf, gran_k=32, use_packed_ue8m0=True)
        output = (intermediate @ reference_weights[1][expert].t()).to(torch.bfloat16).float()
        slots[source, token, slot] = output
    dist.all_reduce(slots, op=dist.ReduceOp.SUM)
    result = torch.zeros_like(harness.inputs.oracle_x)
    route_rounding_budget = torch.zeros_like(result)
    for slot in range(args.num_topk):
        contribution = slots[harness.rank, :, slot]
        result += contribution
        route_rounding_budget += bf16_spacing(contribution.to(torch.bfloat16), torch)
    expected = result.to(torch.bfloat16)
    # Bound final rounding using the widest spacing in the reference-derived
    # interval. This also handles crossing a power-of-two binade boundary.
    final_spacing = bf16_spacing((result.abs() + route_rounding_budget).to(torch.bfloat16), torch)
    return expected, route_rounding_budget + final_spacing


def bf16_spacing(value, torch):
    bits = value.contiguous().view(torch.int16).to(torch.int32) & 65535
    exponent = (bits >> 7) & 255
    return torch.pow(2.0, torch.where(exponent == 0, -133, exponent - 134).float())


def numerical_gate_errors(finite, relative_l2, max_budget_excess, relative_limit):
    errors = []
    if not finite or not math.isfinite(relative_l2) or not math.isfinite(max_budget_excess):
        errors.append("non-finite dense output or numerical error")
    if relative_l2 > relative_limit:
        errors.append("strict relative-L2 tolerance exceeded")
    if max_budget_excess > 0:
        errors.append("reference-derived per-element BF16 rounding budget exceeded")
    return errors


def transport_gate_errors(eager_equal, control_equal):
    errors = []
    if not eager_equal:
        errors.append("same-input CUDA graph output differs bitwise from eager")
    if not control_equal:
        errors.append("communication experiment differs bitwise from the selected same-context baseline graph")
    return errors


def validate_expert_widths(values):
    """Reject asymmetric graph-capture sequences before any capture starts."""
    if len(set(values)) != 1:
        raise ValueError("experts-per-wave flag differs across ranks")
    try:
        width = int(values[0])
    except (ValueError, TypeError, IndexError) as exc:
        raise ValueError("invalid experts-per-wave width") from exc
    if values[0] != "0":
        raise ValueError("retired experts-per-wave width is unsupported; requires 0")
    return width


def validate_barrier_widths(values):
    if len(set(values)) != 1:
        raise ValueError("combine-barrier warp count differs across ranks")
    try:
        width = int(values[0])
    except (ValueError, TypeError, IndexError) as exc:
        raise ValueError("invalid combine-barrier warp count") from exc
    if values[0] != "1":
        raise ValueError("retired combine-barrier warp count is unsupported; requires 1")
    return width


def baseline_control_environment(dispatch_overlap=0, combine_overlap=0):
    if type(dispatch_overlap) is not int or dispatch_overlap not in (0, 1):
        raise ValueError("dispatch overlap mode must be0 or1")
    if type(combine_overlap) is not int or combine_overlap not in (0, 1):
        raise ValueError("combine overlap mode must be0 or1")
    if combine_overlap and not dispatch_overlap:
        raise ValueError("combine overlap requires dispatch overlap1")
    return {EXPERT_WIDTH_ENV: "0", BARRIER_WIDTH_ENV: "1",
            SINGLE_CONTEXT_ENV: "1" if dispatch_overlap else "0",
            DISPATCH_OVERLAP_ENV: "1" if combine_overlap else "0", COMBINE_OVERLAP_ENV: "0"}


def validate_dispatch_modes(values):
    if not values or any(value not in ("0", "1") for value in values):
        raise ValueError("invalid dispatch overlap mode; requires canonical0/1")
    if len(set(values)) != 1:
        raise ValueError("dispatch overlap mode differs across ranks")
    return int(values[0])


def validate_combine_modes(values):
    if not values or any(value not in ("0", "1") for value in values):
        raise ValueError("invalid combine overlap mode; requires canonical0/1")
    if len(set(values)) != 1:
        raise ValueError("combine overlap mode differs across ranks")
    return int(values[0])


def validate_single_context_modes(values):
    if len(set(values)) != 1:
        raise ValueError("single-combine-context mode differs across ranks")
    try:
        mode = int(values[0])
    except (ValueError, TypeError, IndexError) as exc:
        raise ValueError("invalid single-combine-context mode") from exc
    if values[0] not in ("0", "1"):
        raise ValueError("invalid single-combine-context mode")
    return mode


def validate_experiment_combination(experts_per_wave, barrier_warps, single_context,
                                    dispatch_overlap=0, combine_overlap=0):
    if experts_per_wave != 0 or barrier_warps != 1:
        raise ValueError("retired experiments are unsupported; requires expert waves off and barrier width one")
    if dispatch_overlap not in (0, 1):
        raise ValueError("invalid dispatch overlap mode")
    if dispatch_overlap and single_context != 1:
        raise ValueError("dispatch overlap dense gate requires single-context COMBINE1")
    if type(combine_overlap) is not int or combine_overlap not in (0, 1):
        raise ValueError("invalid combine overlap mode")
    if combine_overlap and (single_context != 1 or dispatch_overlap != 1):
        raise ValueError("combine overlap dense gate requires dispatch overlap1 and single-context COMBINE1")


def baseline_control_required(experts_per_wave, barrier_warps, single_context=0,
                              dispatch_overlap=0, combine_overlap=0):
    validate_experiment_combination(experts_per_wave, barrier_warps, single_context, dispatch_overlap, combine_overlap)
    return bool(single_context or dispatch_overlap or combine_overlap)


def check_output(harness, case, expected, rounding_budget, options, label,
                 eager_snapshot=None, control_snapshot=None):
    torch, dist = harness.torch, harness.dist
    actual = harness.output
    problems = []
    try:
        harness.assert_guards(label)
        harness.assert_stats(case.expected_local_stats, label)
        if case.name == "all_masked" and bool(torch.count_nonzero(actual).item()):
            problems.append("masked output is not exactly zero")
        # Compare storage bits, including the sign of zero, not just numeric
        # equality. All three paths return the same BF16 output allocation.
        problems.extend(transport_gate_errors(
            eager_snapshot is None or torch.equal(actual.view(torch.int16), eager_snapshot.view(torch.int16)),
            control_snapshot is None or torch.equal(actual.view(torch.int16), control_snapshot.view(torch.int16))))
    except AssertionError as exc:
        problems.append(str(exc))
    delta = actual.float() - expected.float()
    relative_l2 = float((delta.norm() / expected.float().norm().clamp_min(1e-12)).item())
    max_abs = float(delta.abs().max().item())
    finite = bool(torch.isfinite(actual).all().item() and
                  torch.isfinite(expected).all().item() and
                  torch.isfinite(rounding_budget).all().item())
    # Preserve historical fixed-absolute and final-output-ULP diagnostics. The
    # active budget below is computed from the oracle's individual expert
    # contributions before examining the candidate output.
    def ordered_bf16(tensor):
        bits = tensor.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, 32768 - (bits & 32767), 32768 + bits)
    ulp_distance = (ordered_bf16(actual) - ordered_bf16(expected)).abs()
    spacing = bf16_spacing(expected, torch)
    excess = (delta.abs() - (spacing + options.dense_max_abs)).clamp_min(0)
    worst_flat = int(delta.abs().flatten().argmax().item())
    worst = (worst_flat // harness.args.hidden, worst_flat % harness.args.hidden)
    rounding_diagnostics = {
        "diagnostic_only_does_not_change_acceptance": True,
        "worst_absolute_error_index": list(worst),
        "worst_actual": float(actual[worst].item()),
        "worst_reference": float(expected[worst].item()),
        "worst_reference_bf16_spacing": float(spacing[worst].item()),
        "worst_bf16_ulp_distance": int(ulp_distance[worst].item()),
        "max_bf16_ulp_distance": int(ulp_distance.max().item()),
        "unequal_elements": int(torch.count_nonzero(delta).item()),
        "count_over_one_reference_ulp_plus_atol": int(torch.count_nonzero(excess).item()),
        "max_error_beyond_one_reference_ulp_plus_atol": float(excess.max().item()),
        "diagnostic_atol_floor": options.dense_max_abs,
    }
    budget_excess = (delta.abs() - rounding_budget).clamp_min(0)
    max_budget_excess = float(budget_excess.max().item())
    problems.extend(numerical_gate_errors(finite, relative_l2, max_budget_excess,
                                          options.dense_rel_l2))
    legacy_passed = finite and math.isfinite(relative_l2) and relative_l2 <= 0.02 and max_abs <= 0.0625
    local = {"rank": harness.rank, "relative_l2": relative_l2, "max_abs": max_abs,
             "reference_rms": float(expected.float().square().mean().sqrt().item()),
             "reference_max_abs": float(expected.float().abs().max().item()),
             "finite": finite, "problems": problems,
             "rounding_diagnostics": rounding_diagnostics,
             "legacy_fixed_gate": {"relative_l2_limit": 0.02, "max_abs_limit": 0.0625,
                                   "passed": legacy_passed},
             "active_rounding_budget": {
                 "max_budget": float(rounding_budget.max().item()),
                 "budget_at_worst_absolute_error": float(rounding_budget[worst].item()),
                 "count_exceeding_budget": int(torch.count_nonzero(budget_excess).item()),
                 "max_error_beyond_budget": max_budget_excess,
                 "max_error_budget_ratio": float((delta.abs() / rounding_budget.clamp_min(1e-30)).max().item()),
             }}
    records = [None] * dist.get_world_size()
    dist.all_gather_object(records, local)
    if any(record["problems"] for record in records):
        if harness.rank == 0:
            print("MEGAMOE_DENSE_FAILURE_JSON=" + json.dumps({"label": label, "ranks": records}), flush=True)
        raise AssertionError(f"{label}: dense reference or invariant failed; raw rank errors printed")
    return {"label": label, "ranks": records,
            "max_relative_l2": max(record["relative_l2"] for record in records),
            "max_abs": max(record["max_abs"] for record in records)}


def worker(options, args):
    torch, dist, dg = accuracy._load_runtime()
    buffer = registration = graph = control_graph = None
    success = False
    record = None
    try:
        accuracy._init_distributed(int(os.environ["LOCAL_RANK"]), 8, torch, dist)
        accuracy._check_runtime(torch, dg)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        backend, registration = accuracy._configure_symmetric_memory_backend(args, torch, dist)
        rank, world = dist.get_rank(), dist.get_world_size()
        accuracy._validate_args(args, world)
        hostnames = accuracy._all_hostnames(dist)
        accuracy._validate_gin_host_placement(args, world, hostnames)
        if world != 16 or len(set(hostnames)) != 2:
            raise RuntimeError("dense GIN gate requires two hosts and 16 ranks")
        configurations = [None] * world
        source = source_fingerprint()
        candidate_environment = {EXPERT_WIDTH_ENV: os.environ.get(EXPERT_WIDTH_ENV, "0"),
                                 BARRIER_WIDTH_ENV: os.environ.get(BARRIER_WIDTH_ENV, "1"),
                                 SINGLE_CONTEXT_ENV: os.environ.get(SINGLE_CONTEXT_ENV, "0"),
                                 DISPATCH_OVERLAP_ENV: os.environ.get(DISPATCH_OVERLAP_ENV, "0"),
                                 COMBINE_OVERLAP_ENV: os.environ.get(COMBINE_OVERLAP_ENV, "0")}
        configuration = {"candidate_environment": candidate_environment, "source": source}
        dist.all_gather_object(configurations, configuration)
        if any(item["source"] != source for item in configurations):
            raise ValueError("dense runner source differs across ranks")
        environments = [item["candidate_environment"] for item in configurations]
        expert_width = validate_expert_widths([item[EXPERT_WIDTH_ENV] for item in environments])
        barrier_width = validate_barrier_widths([item[BARRIER_WIDTH_ENV] for item in environments])
        single_context = validate_single_context_modes([item[SINGLE_CONTEXT_ENV] for item in environments])
        dispatch_overlap = validate_dispatch_modes([item[DISPATCH_OVERLAP_ENV] for item in environments])
        combine_overlap = validate_combine_modes([item[COMBINE_OVERLAP_ENV] for item in environments])
        validate_experiment_combination(expert_width, barrier_width, single_context, dispatch_overlap, combine_overlap)
        control_enabled = baseline_control_required(expert_width, barrier_width, single_context, dispatch_overlap, combine_overlap)
        baseline_environment = baseline_control_environment(dispatch_overlap, combine_overlap)
        comparison_axis = ("combine_overlap_with_dispatch1_sc1_fixed" if combine_overlap else
                           "dispatch_overlap_with_combine1_fixed" if dispatch_overlap else "single_combine_context")
        experts = args.num_experts // world
        if (experts, args.hidden, args.intermediate_hidden, args.num_topk) != (56, 3584, 3072, 16):
            raise RuntimeError("dense gate requires actual H3584/I3072/E896/topk16")
        cases = accuracy._make_route_cases(rank, args, experts, hostnames, torch, dist)
        buffer = dg.get_symm_buffer_for_mega_moe(
            dist.group.WORLD, args.num_experts, args.num_max_tokens_per_rank,
            args.num_topk, args.hidden, args.intermediate_hidden,
            num_shared_experts=0, mma_type=args.mma_type, enable_gin=True,
            gin_completion_batch=args.gin_completion_batch,
            gin_combine_chunk_bytes=args.gin_combine_chunk_bytes,
            gin_outbox_depth=args.gin_outbox_depth,
            gin_combine_issue_wave=args.gin_combine_issue_wave,
            gin_queue_depth=args.gin_queue_depth,
            gin_active_fast_path=True, gin_bulk_combine=True, gin_direct_dispatch=True)
        transport = accuracy._gin_transport_evidence(
            buffer, args, rank, world, hostnames, backend, registration, dist)
        if rank == 0:
            print("MEGAMOE_DENSE_STAGE=generate_dense_weights", flush=True)
        weights, oracle_weights, fingerprint = make_dense_weights(rank, experts, args, options, torch, dg)
        fingerprints = [None] * world
        dist.all_gather_object(fingerprints, fingerprint)
        if len({item["sampled_canonical_sha256"] for item in fingerprints}) != world:
            raise AssertionError("rank-distinct weight fingerprints are not all distinct")
        inputs = accuracy._make_inputs(rank, args, torch, dg)
        harness = accuracy.AccuracyHarness(rank, args, inputs, weights, cases, buffer, torch, dist, dg)
        pointers = accuracy._input_storage_pointers(inputs, buffer)
        if control_enabled:
            # Change only the selected transport axis, keeping the registered
            # buffer, context, weights and input storage. Restore all candidate
            # flags before capturing its graph;
            # subsequent environment changes cannot alter captured kernel nodes.
            with patch.dict(os.environ, baseline_environment):
                control_graph = accuracy._capture_launch_only_graph(
                    harness, torch, dist, options.dense_routes[0])
        graph = accuracy._capture_launch_only_graph(harness, torch, dist, options.dense_routes[0])
        results = []
        initial_outputs = {}
        epochs = accuracy._payload_epoch_sequence(options.dense_epochs)
        for epoch in epochs:
            accuracy._copy_input_storage(inputs, accuracy._make_inputs(rank, args, torch, dg, input_epoch=epoch))
            accuracy._collective_transition_check(
                f"dense-pointers-epoch={epoch}",
                (["changing payload replaced captured input storage"]
                 if pointers != accuracy._input_storage_pointers(inputs, buffer) else []), dist)
            for mode in (*options.dense_routes, "all_masked"):
                case = cases[mode]
                label = f"epoch={epoch}/route={mode}"
                if rank == 0:
                    print("MEGAMOE_DENSE_STAGE=" + label, flush=True)
                expected, rounding_budget = dense_reference(harness, case, oracle_weights)
                harness.stats.zero_()
                eager = harness.eager(case).clone()
                results.append(check_output(harness, case, expected, rounding_budget,
                                            options, "eager/" + label))
                payload_errors = []
                if epoch == 0:
                    if mode not in initial_outputs:
                        initial_outputs[mode] = eager.clone()
                    elif not torch.equal(eager, initial_outputs[mode]):
                        payload_errors.append("return to epoch zero differs from original eager output")
                elif mode != "all_masked" and torch.equal(eager, initial_outputs[mode]):
                    payload_errors.append("changed payload failed to change dense output")
                accuracy._collective_transition_check("dense-" + label, payload_errors, dist)
                control_output = None
                if control_graph is not None:
                    harness.copy_inputs(case)
                    harness.stats.zero_()
                    harness.output.fill_(float("nan"))
                    control_graph.replay()
                    torch.cuda.synchronize()
                    control_output = harness.output.clone()
                    results.append(check_output(harness, case, expected, rounding_budget,
                                                options, ("combine0-dispatch1-sc1-control/" if combine_overlap else
                                                          "dispatch0-combine1-control/" if dispatch_overlap
                                                          else "single-context0-control/") + label))
                harness.copy_inputs(case)
                harness.stats.zero_()
                harness.output.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                results.append(check_output(harness, case, expected, rounding_budget, options,
                                            "graph/" + label, eager_snapshot=eager,
                                            control_snapshot=control_output))
        graph = control_graph = None
        torch.cuda.synchronize()
        accuracy._collective_transition_check("dense-source-unchanged",
            [] if source_fingerprint() == source else ["listed sources changed during dense gate"], dist)
        record = {"schema": "megamoe-dense-accuracy-v2", "status": "passed",
                  "shape": {"tokens_per_rank": args.num_tokens, "hidden": args.hidden,
                            "intermediate_hidden": args.intermediate_hidden,
                            "experts_per_rank": experts, "world_size": world, "topk": args.num_topk},
                  "routes": options.dense_routes, "payload_epochs": epochs,
                  "limits": {"relative_l2": options.dense_rel_l2,
                             "absolute_budget": "one BF16 ULP per oracle contribution plus final rounding ULP",
                             "legacy_absolute_diagnostic_floor": options.dense_max_abs},
                  "rounding_rationale": "BF16 route-rounding boundaries followed by cancellation require contribution-scale tolerance; strict relative-L2 remains independent",
                  "combine_experts_per_wave": expert_width,
                  "retired_expert_wave_and_coop_width_experiments_supported": False,
                  "combine_barrier_warps_requested": barrier_width,
                  "single_combine_context_requested": single_context,
                  "dispatch_overlap_requested": dispatch_overlap,
                  "dispatch_overlap_requested_raw": candidate_environment[DISPATCH_OVERLAP_ENV],
                  "combine_overlap_requested": combine_overlap,
                  "combine_overlap_requested_raw": candidate_environment[COMBINE_OVERLAP_ENV],
                  "combine_overlap_effective_policy_not_device_observation": {
                      "requires": "remote direct+bulk and per-rank counter storage fit",
                      "fit_failure": "unchanged full-packet SC1 combine",
                      "physical_overlap_measured": False},
                  "transport_comparison_axis": comparison_axis,
                  "single_combine_context_effective_by_route": {
                      mode: bool(single_context and mode not in ("all_same_host", "all_masked"))
                      for mode in (*options.dense_routes, "all_masked")},
                  "combine_barrier_warps_effective_by_route": {
                      mode: 0 if mode in ("all_same_host", "all_masked") else barrier_width
                      for mode in (*options.dense_routes, "all_masked")},
                  "candidate_environment": candidate_environment,
                  "candidate_transport_environment": candidate_environment,
                  "configuration": configuration,
                  "source": source,
                  # Retain the legacy expert-width-0 evidence key: the new
                  # baseline also explicitly restores barrier width to one.
                  "bitwise_width0_control_enabled": control_enabled,
                  "bitwise_baseline_control_enabled": control_enabled,
                  "baseline_control_environment": baseline_environment,
                  "bitwise_dispatch0_combine1_control_enabled": bool(dispatch_overlap and not combine_overlap and control_enabled),
                  "bitwise_combine0_dispatch1_sc1_control_enabled": bool(combine_overlap and control_enabled),
                  "control_uses_identical_context_weights_and_storage": True,
                  "oracle": "canonical_dequantized_FP32_matmul_BF16_SwiGLU_weight_MXFP8_FP32_matmul_BF16_ordered_combine",
                  "tf32": False, "fast_math": False, "heterogeneous_input_sf": True,
                  "distinct_weights_per_rank": True, "changing_graph_payload": True,
                  "same_input_graph_eager_bitwise": True, "timing_claim": False,
                  "transport": transport, "weight_fingerprints": fingerprints, "checks": results}
        success = accuracy._synchronize_worker_success(dist)
    finally:
        graph = control_graph = None
        accuracy._teardown_worker(buffer, dist, success, registration)
    return record


def main():
    options, args = parse_args()
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("launch dense gate using two-node torchrun, eight ranks/node")
    os.environ[accuracy.GIN_LOCAL_ABLATION_ENV] = "0"
    os.environ["DG_MEGAMOE_GIN_DIAGNOSTICS"] = "0"
    for flag in (accuracy.GIN_ACTIVITY_GATE_OPT_ENV, *accuracy.GIN_EXPERIMENT_FLAG_ENVS):
        os.environ[flag] = "1"
    rank = int(os.environ["RANK"])
    record = worker(options, args)
    if rank == 0:
        with options.output.open("x") as output:
            json.dump(record, output, sort_keys=True)
            output.write("\n")
        print("MEGAMOE_DENSE_ACCURACY_JSON=" + json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
