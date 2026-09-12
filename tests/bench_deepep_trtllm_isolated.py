"""Shared-input SwiGLU MegaMoE versus direct DeepEP-v2 + TRTLLM comparison.

Two-node torchrun example (repeat --decode-mns 8/10/12 in fresh processes):
  python tests/bench_deepep_trtllm_isolated.py --decode-mns 8 \
      --comparison-replays 204 --output isolated-mns8.json

The timed scope starts with identical, prequantized MXFP8 activation bytes,
packed UE8M0 scales, top-k IDs/weights, and canonical MXFP4 expert weights.
DeepEP dispatch, receive-ID conversion, TRTLLM routing/compute/finalization,
DeepEP combine, and final output copy are all included. Input quantization
and source-to-symmetric-buffer copies are outside both intervals. This is a
SwiGLU operator comparison, not the historical production SITU trace.

Each process first passes the established MegaMoE correctness harness. Both
backends then pass a sparse quantization-aware oracle before any timing. A
backend mismatch fails the run; tolerances and raw errors are recorded.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch

import test_mega_moe_accuracy as accuracy

TRT_RECIPE_EVIDENCE = {
    "status": "exact_MNS8_r4_selected_FC1_cubin_independently_disassembled",
    "bounded_scope": "MNS8 r4 all-rank profiler binding; future tactics are not individually audited by this record",
    "fc1_kernel": "bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x192x256u2_s3_et128x32_m256x192x32_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_tma_ldgstsSf_rgTma_clmp_swiGlu_lbW8_lsfbW4_dynB_sm100f",
    "cubin_sha256": "511eef5c21b62623c6bff3ddfc13677a0cad7532d8805abd5ef2189746eb9a44",
    "sequence": "FP32 exponent mask; multiply 1/256; E8 RP; reciprocal; SATFINITE E4M3",
}


def _has_option(options, flag):
    return any(option == flag or option.startswith(flag + "=") for option in options)


def dispatch_bucket_evidence(tokens, capacity, requested=None):
    serving_default = min(1 << (tokens - 1).bit_length(), capacity)
    bucket = serving_default if requested is None else requested
    if not tokens <= bucket <= capacity:
        raise ValueError("DeepEP dispatch bucket must satisfy active tokens <= bucket <= capacity")
    return {
        "dispatch_bucket": bucket,
        "dispatch_bucket_requested": requested,
        "serving_default_dispatch_bucket": serving_default,
        "dispatch_bucket_policy": (
            "serving_power_of_two_clamped_to_capacity" if requested is None else
            "explicit_exact_active_tokens" if bucket == tokens else "explicit_padded"),
        "dispatch_padding_tokens": bucket - tokens,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison-replays", type=int, default=204)
    parser.add_argument("--comparison-warmups", type=int, default=10)
    parser.add_argument("--comparison-conditioning", type=int, default=60)
    parser.add_argument("--component-replays", type=int, default=8,
                        help="Separate instrumented component graph samples; 0 disables")
    parser.add_argument("--comparison-rel-l2", type=float, default=0.02)
    parser.add_argument("--comparison-max-abs", type=float, default=0.02)
    parser.add_argument("--deepep-num-sms", type=int, default=None)
    parser.add_argument("--deepep-num-qps", type=int, default=None)
    parser.add_argument("--deepep-capacity", type=int, default=None)
    parser.add_argument("--deepep-dispatch-bucket", type=int, default=None,
                        help="Explicit runtime dispatch bucket; default is min(next_power_of_two(T), capacity)")
    parser.add_argument("--profile-recipe", action="store_true", help=(
        "Profile one untimed TRTLLM graph replay and retain CUDA kernel names even if accuracy fails"))
    parser.add_argument("--compare-preparation", action="store_true", help=(
        "Separately time placement of already-quantized inputs/routes into each backend's "
        "source buffers; input quantization itself remains excluded"))
    comparison, remaining = parser.parse_known_args()
    for name in ("comparison_replays", "comparison_warmups", "comparison_conditioning"):
        if getattr(comparison, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if comparison.comparison_replays < 2:
        parser.error("--comparison-replays must be at least 2 for split-sample summaries")
    if any(not math.isfinite(value) or value <= 0 for value in
           (comparison.comparison_rel_l2, comparison.comparison_max_abs)):
        parser.error("accuracy thresholds must be finite and positive")
    if comparison.component_replays < 0 or comparison.component_replays == 1:
        parser.error("--component-replays must be 0 or at least 2")
    for flag, value in {"--decode-mns": "8", "--eager-iterations": "16",
                        "--graph-replays": "32", "--gin-completion-batch": "8",
                        "--gin-combine-issue-wave": "8", "--gin-outbox-depth": "64",
                        "--gin-combine-chunk-bytes": "7168"}.items():
        if not _has_option(remaining, flag) and not (
            flag == "--decode-mns" and _has_option(remaining, "--num-tokens")
        ):
            remaining.extend((flag, value))
    remaining.extend(("--k3", "--require-gin", "--require-cross-host",
                      "--gin-active-fast-path", "--gin-bulk-combine",
                      "--gin-direct-dispatch"))
    with patch.object(sys, "argv", [sys.argv[0], *remaining]):
        args = accuracy._parse_args()
    if args.benchmark_replays or args.gin_local_ablation_stages:
        parser.error("use --comparison-replays; separate historical benchmark/ablation modes")
    if args.num_tokens not in (32, 40, 48):
        parser.error("target token counts are 32, 40, and 48")
    if comparison.deepep_capacity is None:
        comparison.deepep_capacity = args.num_tokens
    if comparison.deepep_capacity < args.num_tokens:
        parser.error("DeepEP capacity cannot be smaller than active tokens")
    try:
        comparison.dispatch_bucket_evidence = dispatch_bucket_evidence(
            args.num_tokens, comparison.deepep_capacity, comparison.deepep_dispatch_bucket)
    except ValueError as exc:
        parser.error(str(exc))
    return comparison, args


def _canonical_weights(harness):
    """Create one canonical FP4 quantization, verifying MegaMoE's exact view."""
    torch, dg, args = harness.torch, harness.deep_gemm, harness.args
    from deep_gemm.utils import per_token_cast_to_fp4
    from deep_gemm.utils.math import pack_ue8m0_to_int
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend, convert_weight_to_mxfp4_moe_kernel_format,
    )

    bf16 = accuracy._make_bf16_weights(harness.experts_per_rank, args.hidden,
                                      args.intermediate_hidden, torch)
    packed, scales, dg_pairs = [], [], []
    for matrix in bf16:
        groups, n, k = matrix.shape
        quant = torch.empty((groups, n, k // 2), device="cuda", dtype=torch.int8)
        sf = torch.empty((groups, n, k // 32), device="cuda", dtype=torch.float32)
        for expert in range(groups):
            quant[expert], sf[expert] = per_token_cast_to_fp4(
                matrix[expert], use_ue8m0=True, gran_k=32)
        packed.append(quant)
        scales.append(pack_ue8m0_to_int(sf).view(torch.uint8))
        dg_pairs.append((quant, dg.transform_sf_into_required_layout(sf, n, k, (1, 32), groups)))
    expected = dg.transform_weights_for_mega_moe(*dg_pairs)
    if len(expected) != 2:
        raise AssertionError("canonical conversion must return exactly two layer weights")
    for label, generated, actual in zip(("l1", "l2"), expected,
                                         (harness.weights.l1, harness.weights.l2)):
        if len(generated) != 2 or len(actual) != 2:
            raise AssertionError(f"{label} must contain quantized weights and scales")
        for index, (lhs, rhs) in enumerate(zip(generated, actual)):
            if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
                raise AssertionError(f"shared canonical {label}/{index} shape or dtype mismatch")
            accuracy._assert_bitwise_equal(lhs, rhs, f"shared-canonical-{label}-{index}", torch)
    del expected, dg_pairs, bf16
    # Hash before backend layout changes, using the exact byte representation.
    digest = hashlib.sha256()
    for tensor in (*packed, *scales):
        digest.update(tensor.contiguous().cpu().numpy().tobytes())
    converted = convert_weight_to_mxfp4_moe_kernel_format(
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8, torch.nn.Module(),
        packed[0].view(torch.uint8), packed[1].view(torch.uint8),
        scales[0], scales[1], _cache_permute_indices={}, activation=MoEActivation.SILU,
    )
    return converted, digest.hexdigest()


def _quantize_activation(value, torch, recipe="deepgemm_ceil448"):
    """Return FP8 data, packed UE8M0 scales, and independently decoded values.

    The exponent-only recipe is supported by full-tensor GPU comparison and
    independently disassembled exact MNS8-r4 selected fused-SwiGLU cubin SASS.
    Future selected tactics remain separately auditable. It can saturate the largest elements,
    unlike DeepGemm's ceil(amax / 448) recipe; this is not a tolerance change.
    """
    from deep_gemm.utils import per_token_cast_to_fp8
    from deep_gemm.utils.math import cast_back_from_fp8
    if recipe == "deepgemm_ceil448":
        q, sf = per_token_cast_to_fp8(value, use_ue8m0=True, gran_k=32,
                                      use_packed_ue8m0=True)
    elif recipe == "ocp_exponent_only":
        blocks = value.float().reshape(value.shape[0], -1, 32)
        amax = blocks.abs().amax(dim=-1).clamp_min(1e-4)
        # Candidate OCP scale: largest input exponent minus E4M3 max exponent.
        exponent = (torch.floor(torch.log2(amax)).to(torch.int32) - 8).clamp(-126, 127)
        scale = torch.ldexp(torch.ones_like(amax), exponent)
        q = (blocks / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        q = q.reshape_as(value)
        scale_bytes = (exponent + 127).to(torch.uint8)
        padding = (-scale_bytes.shape[-1]) % 4
        if padding:
            scale_bytes = torch.nn.functional.pad(scale_bytes, (0, padding), value=127)
        sf = scale_bytes.contiguous().view(torch.int32)
    else:
        raise ValueError(f"unsupported intermediate quantizer recipe: {recipe}")
    return q, sf, cast_back_from_fp8(q, sf, gran_k=32, use_packed_ue8m0=True)


def _post_weight_oracle(harness, case, group_by_owner=False, activation_bf16=False,
                        quantizer_recipe="ocp_exponent_only"):
    """Sparse oracle with TRTLLM's weight-after-intermediate-quantization order.

    Backend GEMM/finalize accumulation can round differently. This independent
    reference catches wrong layout, wrong scale, wrong owner, and missing rows.
    """
    torch, args = harness.torch, harness.args
    result = torch.zeros((args.num_tokens, args.hidden), device="cuda", dtype=torch.float32)
    owner_partials = (torch.zeros((harness.dist.get_world_size(), args.num_tokens, args.hidden),
                                  device="cuda", dtype=torch.float32) if group_by_owner else None)
    j = torch.arange(args.intermediate_hidden, device="cuda").view(1, -1)
    o = torch.arange(args.hidden, device="cuda").view(1, -1)
    for slot in range(args.num_topk):
        valid = case.topk_idx[:, slot] >= 0
        expert = (case.topk_idx[:, slot].clamp_min(0) % harness.experts_per_rank).view(-1, 1)
        gate = torch.gather(harness.inputs.oracle_x, 1,
                            accuracy._gate_source(j, expert, args.hidden))
        up = torch.gather(harness.inputs.oracle_x, 1,
                          accuracy._up_source(j, expert, args.hidden))
        gate = (gate * torch.where(expert % 2 == 0, 0.5, 1.0)).to(torch.bfloat16).float()
        up = (up * torch.where(expert % 3 == 0, 1.0, 0.5)).to(torch.bfloat16).float()
        if math.isfinite(args.activation_clamp):
            gate = gate.clamp_max(args.activation_clamp)
            up = up.clamp(-args.activation_clamp, args.activation_clamp)
        activation = gate / (1.0 + torch.exp(-gate)) * up
        if activation_bf16:
            activation = activation.to(torch.bfloat16).float()
        _, _, activation = _quantize_activation(activation, torch, quantizer_recipe)
        out = torch.gather(activation, 1, accuracy._out_source(o, expert, args.intermediate_hidden))
        out = (out * torch.where(expert % 4 < 2, 0.5, 1.0)).to(torch.bfloat16).float()
        contribution = out * case.topk_weights[:, slot, None] * valid[:, None]
        result += contribution
        if owner_partials is not None:
            owner = (case.topk_idx[:, slot].clamp_min(0) // harness.experts_per_rank)
            owner_partials.scatter_add_(0, owner.view(1, -1, 1).expand(1, -1, args.hidden),
                                         contribution.unsqueeze(0))
    if owner_partials is not None:
        # Model TRTLLM's per-owner BF16 finalize, then DeepEP's combine.
        # This alternate reference is diagnostic; it does not change the gate.
        result = owner_partials.to(torch.bfloat16).float().sum(dim=0)
    return result.to(torch.bfloat16)


def _worst_slot_breakdown(harness, case, actual, reference):
    """Explain each sparse expert's contribution at the worst output element."""
    torch, args = harness.torch, harness.args
    flat = int((actual.float() - reference.float()).abs().flatten().argmax().item())
    token, column = divmod(flat, args.hidden)
    source = harness.inputs.oracle_x[token]
    slots = []
    for slot in range(args.num_topk):
        global_expert = int(case.topk_idx[token, slot].item())
        if global_expert < 0:
            slots.append({"slot": slot, "masked": True})
            continue
        expert = global_expert % harness.experts_per_rank
        intermediate = int(accuracy._out_source(column, expert, args.intermediate_hidden))
        group_start, within_group = intermediate // 32 * 32, intermediate % 32
        j = torch.arange(group_start, group_start + 32, device="cuda")
        gate_scale, up_scale, out_scale = accuracy._expert_scales(expert)
        gate = (source[accuracy._gate_source(j, expert, args.hidden)] * gate_scale).to(torch.bfloat16).float()
        up = (source[accuracy._up_source(j, expert, args.hidden)] * up_scale).to(torch.bfloat16).float()
        if math.isfinite(args.activation_clamp):
            gate = gate.clamp_max(args.activation_clamp)
            up = up.clamp(-args.activation_clamp, args.activation_clamp)
        activation = gate / (1.0 + torch.exp(-gate)) * up
        route_weight = float(case.topk_weights[token, slot].item())
        variants = {}
        for name, value, recipe in (
            ("fp32_activation", activation, "deepgemm_ceil448"),
            ("bf16_activation", activation.to(torch.bfloat16).float(), "deepgemm_ceil448"),
            ("ocp_exponent_only", activation, "ocp_exponent_only"),
            ("ocp_exponent_only_bf16_activation", activation.to(torch.bfloat16).float(),
             "ocp_exponent_only"),
        ):
            q, sf, decoded = _quantize_activation(value.view(1, 32), torch, recipe)
            contribution = (decoded[0, within_group] * out_scale).to(torch.bfloat16).float() * route_weight
            variants[name] = {
                "before_quantization": float(value[within_group].item()),
                "group_amax": float(value.abs().amax().item()),
                "quantizer_recipe": recipe,
                "fp8_value": float(q[0, within_group].float().item()),
                "fp8_byte": int(q.view(torch.uint8)[0, within_group].item()),
                "ue8m0_exponent_byte": int(sf.view(torch.uint8)[0, 0].item()),
                "dequantized": float(decoded[0, within_group].item()),
                "weighted_contribution": float(contribution.item()),
            }
        slots.append({"slot": slot, "global_expert": global_expert,
                      "owner": global_expert // harness.experts_per_rank,
                      "local_expert": expert, "intermediate_column": intermediate,
                      "route_weight": route_weight,
                      "gate": float(gate[within_group].item()),
                      "up": float(up[within_group].item()), "variants": variants})
    return {"index": [token, column], "actual": float(actual[token, column].item()),
            "reference": float(reference[token, column].item()), "slots": slots,
            "contribution_sums_before_final_bf16": {
                variant: sum(item["variants"][variant]["weighted_contribution"]
                             for item in slots if not item.get("masked", False))
                for variant in ("fp32_activation", "bf16_activation", "ocp_exponent_only",
                                "ocp_exponent_only_bf16_activation")},
            "quantization_group": "full 32-channel activation group containing the sparse W2 input"}


def _error_record(actual, reference, torch):
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(reference).all()):
        return {"finite": False, "relative_l2": float("inf"), "max_abs": float("inf")}
    delta = actual.float() - reference.float()
    denominator = reference.float().square().sum().sqrt().clamp_min(1e-12)
    worst = int(delta.abs().flatten().argmax().item())
    flat_actual, flat_reference = actual.flatten(), reference.flatten()
    coordinates = []
    remaining = worst
    for dimension in reversed(actual.shape):
        coordinates.append(remaining % dimension)
        remaining //= dimension
    coordinates.reverse()
    ulp = None
    if actual.dtype == torch.bfloat16 and reference.dtype == torch.bfloat16:
        def ordered(tensor):
            bits = tensor.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
            return torch.where((bits & 0x8000) != 0,
                               0x8000 - (bits & 0x7FFF), bits + 0x8000)
        distances = (ordered(actual) - ordered(reference)).abs()
        ulp = {"at_max_abs_error": int(distances.flatten()[worst].item()),
               "max": int(distances.max().item()),
               "fraction_within": {str(limit): float((distances <= limit).float().mean().item())
                                   for limit in (0, 1, 2, 4, 8)}}
    return {"finite": True,
            "relative_l2": float((delta.square().sum().sqrt() / denominator).item()),
            "max_abs": float(delta.abs().max().item()),
            "actual_max_abs": float(actual.float().abs().max().item()),
            "reference_max_abs": float(reference.float().abs().max().item()),
            "reference_rms": float(reference.float().square().mean().sqrt().item()),
            "error_rms": float(delta.square().mean().sqrt().item()),
            "worst_element": {"index": coordinates,
                              "actual": float(flat_actual[worst].item()),
                              "reference": float(flat_reference[worst].item()),
                              "signed_difference": float(delta.flatten()[worst].item())},
            "bf16_ulp_distance": ulp}


def _validate_output(actual, reference, label, comparison, torch, dist,
                     extra_references=None, reference_structure=None):
    metrics = _error_record(actual, reference, torch)
    if extra_references:
        metrics["diagnostic_reference_errors"] = {
            name: _error_record(actual, tensor, torch) for name, tensor in extra_references.items()
        }
    if reference_structure:
        metrics["reference_structure"] = reference_structure
    records = [None] * dist.get_world_size()
    dist.all_gather_object(records, metrics)
    for rank, record in enumerate(records):
        if (not record["finite"] or not math.isfinite(record["relative_l2"])
                or not math.isfinite(record["max_abs"])
                or record["relative_l2"] > comparison.comparison_rel_l2
                or record["max_abs"] > comparison.comparison_max_abs):
            raise AssertionError(f"{label}: rank {rank} failed shared-input reference: {record}")
    return {"per_rank": records, "max_relative_l2": max(r["relative_l2"] for r in records),
            "max_abs": max(r["max_abs"] for r in records)}


def _validate_backend_drift(actual, reference, label, comparison, torch, dist):
    """Bound native-recipe drift without treating distinct quantizers as equal.

    Each backend separately passes its strict recipe-aware absolute/L2 gate.
    Here the original absolute rule is reported, not silently widened.
    """
    records = [None] * dist.get_world_size()
    dist.all_gather_object(records, _error_record(actual, reference, torch))
    for rank, record in enumerate(records):
        if (not record["finite"] or not math.isfinite(record["relative_l2"])
                or not math.isfinite(record["max_abs"])
                or record["relative_l2"] > comparison.comparison_rel_l2):
            raise AssertionError(f"{label}: rank {rank} failed native-recipe drift ceiling: {record}")
    return {"per_rank": records, "max_relative_l2": max(r["relative_l2"] for r in records),
            "max_abs": max(r["max_abs"] for r in records),
            "semantics": "shared_inputs_different_native_intermediate_quantizers",
            "relative_l2_gate": comparison.comparison_rel_l2,
            "absolute_error_is_equality_gate": False,
            "previous_absolute_rule": {
                "limit": comparison.comparison_max_abs,
                "passed": all(r["max_abs"] <= comparison.comparison_max_abs for r in records),
                "per_rank_passed": [r["max_abs"] <= comparison.comparison_max_abs for r in records],
            }}


def _receipt_histogram(rows, valid_rows, owner_offset, local_experts, expected):
    """CPU-only check of a synchronized no-expand dispatch's globalized IDs."""
    if not 0 <= valid_rows <= len(rows):
        raise AssertionError(f"received row count {valid_rows} exceeds allocation {len(rows)}")
    histogram = [0] * local_experts
    for row_index, row in enumerate(rows):
        for expert_id in row:
            if row_index >= valid_rows:
                if expert_id != -1:
                    raise AssertionError(f"unmasked padding row {row_index}: expert {expert_id}")
                continue
            if expert_id == -1:
                continue
            local_expert = expert_id - owner_offset
            if not 0 <= local_expert < local_experts:
                raise AssertionError(f"received nonlocal expert {expert_id} at row {row_index}")
            histogram[local_expert] += 1
    if histogram != expected:
        raise AssertionError(f"receive histogram mismatch: actual={histogram}, expected={expected}")
    return {"valid_received_rows": valid_rows, "allocation_rows": len(rows),
            "received_routes": sum(histogram), "expert_histogram": histogram}


def _validate_receipt(receipt, case, harness, label):
    """Read captured receipt tensors outside every performance interval."""
    try:
        rows = receipt["globalized_idx"].cpu().tolist()
        prefix = receipt["recv_prefix"].cpu().tolist()
        if not prefix:
            raise AssertionError("empty receive prefix-sum tensor")
        record = _receipt_histogram(rows, int(prefix[-1]),
                                    harness.rank * harness.experts_per_rank,
                                    harness.experts_per_rank,
                                    case.expected_local_stats.cpu().tolist())
        record["status"] = "passed"
    except Exception as error:
        record = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    records = [None] * harness.dist.get_world_size()
    harness.dist.all_gather_object(records, record)
    failed = [(rank, result["error"]) for rank, result in enumerate(records)
              if result["status"] != "passed"]
    if failed:
        raise AssertionError(f"{label}: DeepEP exact receive validation failed: {failed}")
    return {"status": "passed", "method": "untimed_readback_of_captured_globalized_route_ids",
            "dispatch_stats_writes_enabled": False, "per_rank": records}


class DeepEPTRTLLM:
    def __init__(self, harness, comparison):
        import deep_ep
        import flashinfer
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
        from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_v2 import (
            _globalize_recv_topk_idx,
        )
        from vllm.model_executor.layers.fused_moe.utils import trtllm_moe_pack_topk_ids_weights
        from vllm.model_executor.layers.quantization.utils.flashinfer_utils import activation_to_flashinfer_int

        self.harness, self.comparison = harness, comparison
        self.torch, self.args = harness.torch, harness.args
        self.globalize, self.pack = _globalize_recv_topk_idx, trtllm_moe_pack_topk_ids_weights
        self.routed_moe = flashinfer.trtllm_fp4_block_scale_routed_moe
        self.recipe_profile = None
        self.activation_type = activation_to_flashinfer_int(MoEActivation.SILU)
        self.routing_method = RoutingMethodType.Renormalize
        self.weights, self.weight_sha256 = _canonical_weights(harness)
        self.x, self.sf = harness.inputs.x, harness.inputs.x_sf.contiguous()
        self.topk = self.torch.empty_like(harness.cases["all_same_host"].topk_idx)
        self.topk_weights = self.torch.empty_like(harness.cases["all_same_host"].topk_weights)
        self.output = self.torch.empty_like(harness.output)
        self.alpha = self.torch.ones(harness.experts_per_rank, device="cuda", dtype=self.torch.float32)
        self.beta = self.torch.zeros_like(self.alpha)
        self.clamp = (self.torch.full_like(self.alpha, self.args.activation_clamp)
                      if math.isfinite(self.args.activation_clamp) else None)
        self.comm_kwargs = {}
        if comparison.deepep_num_sms is not None:
            self.comm_kwargs["num_sms"] = comparison.deepep_num_sms
        if comparison.deepep_num_qps is not None:
            self.comm_kwargs["num_qps"] = comparison.deepep_num_qps
        self.dispatch_bucket = comparison.dispatch_bucket_evidence["dispatch_bucket"]
        self.buffer = deep_ep.ElasticBuffer(
            group=harness.dist.group.WORLD,
            num_max_tokens_per_rank=comparison.deepep_capacity,
            hidden=self.args.hidden, num_topk=self.args.num_topk,
            use_fp8_dispatch=True, deterministic=False, allow_hybrid_mode=False,
            allow_multiple_reduction=False, prefer_overlap_with_compute=False,
            num_cpu_timeout_secs=100, num_gpu_timeout_secs=100, explicitly_destroy=True,
        )
        self.api_evidence = {
            "flashinfer_path": flashinfer.__file__, "deep_ep_path": deep_ep.__file__,
            "routed_moe_signature": str(inspect.signature(self.routed_moe)),
            **comparison.dispatch_bucket_evidence,
            "capacity": comparison.deepep_capacity,
            "comm_overrides": self.comm_kwargs,
            "logical_domain_size": list(self.buffer.get_logical_domain_size()),
            "physical_domain_size": list(self.buffer.get_physical_domain_size()),
            "allow_hybrid_mode": False, "activation_type": self.activation_type,
            "tune_max_num_tokens": 8192,
        }

    def select_route(self, case):
        self.topk.copy_(case.topk_idx)
        self.topk_weights.copy_(case.topk_weights)

    def launch(self, markers=None):
        def mark(name):
            if markers is not None:
                markers[name].record()

        mark("dispatch_begin")
        recv_x, recv_idx, recv_weight, handle, event = self.buffer.dispatch(
            x=(self.x, self.sf), topk_idx=self.topk, topk_weights=self.topk_weights,
            num_experts=self.args.num_experts,
            num_max_tokens_per_rank=self.dispatch_bucket,
            do_expand=False, do_cpu_sync=False, async_with_compute_stream=False,
            **self.comm_kwargs,
        )
        if event.event is not None:
            event.current_stream_wait()
        mark("dispatch_complete")
        idx = self.globalize(recv_idx, handle.psum_num_recv_tokens_per_scaleup_rank,
                             self.harness.rank * self.harness.experts_per_rank,
                             self.args.num_experts)
        # Holding tensor references adds no GPU operation. Each graph's caller
        # freezes its own dictionary after capture; later captures may replace
        # this Python attribute but cannot replace the earlier graph's receipt.
        self.last_receipt = {"globalized_idx": idx,
                             "recv_prefix": handle.psum_num_recv_tokens_per_scaleup_rank}
        packed_topk = self.pack(idx.contiguous(), recv_weight.contiguous())
        mark("route_preparation_complete")
        x, sf = recv_x
        expert_output = self.torch.empty((x.shape[0], self.args.hidden),
                                         device=x.device, dtype=self.torch.bfloat16)
        w1, w2, s1, s2, b1, b2 = self.weights
        self.routed_moe(
            topk_ids=packed_topk, routing_bias=None,
            hidden_states=x, hidden_states_scale=sf.contiguous().view(self.torch.float8_e4m3fn),
            gemm1_weights=w1, gemm1_weights_scale=s1, gemm1_bias=b1,
            gemm1_alpha=self.alpha, gemm1_beta=self.beta, gemm1_clamp_limit=self.clamp,
            gemm2_weights=w2, gemm2_weights_scale=s2, gemm2_bias=b2,
            output1_scale_scalar=None, output1_scale_gate_scalar=None, output2_scale_scalar=None,
            num_experts=self.args.num_experts, top_k=self.args.num_topk,
            n_group=None, topk_group=None, intermediate_size=self.args.intermediate_hidden,
            local_expert_offset=self.harness.rank * self.harness.experts_per_rank,
            local_num_experts=self.harness.experts_per_rank,
            routed_scaling_factor=None, routing_method_type=self.routing_method,
            do_finalize=True, enable_pdl=True, activation_type=self.activation_type,
            output=expert_output, tune_max_num_tokens=8192,
        )
        mark("expert_finalize_complete")
        combined, _, event = self.buffer.combine(
            x=expert_output, handle=handle, topk_weights=None,
            async_with_compute_stream=False, **self.comm_kwargs,
        )
        if event.event is not None:
            event.current_stream_wait()
        mark("combine_complete")
        self.output.copy_(combined)
        mark("output_ready")
        self.last_recv_capacity = x.shape[0]
        return self.output


def _capture(call, torch, dist):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    torch.cuda.synchronize()
    return graph


def _retire_graphs(graphs):
    # An exception traceback can retain a graph's Python object after its dict
    # is cleared. Reset the CUDA executable explicitly before DeepEP destroy.
    active_error = sys.exc_info()[0] is not None
    for graph in tuple(graphs.values()):
        try:
            graph.reset()
        except Exception as reset_error:
            if not active_error:
                raise
            print(f"CUDA graph retirement also failed: {reset_error}", file=sys.stderr, flush=True)
    graphs.clear()


def _profile_recipe(graph, harness, comparison):
    """Capture names only; this replay is never a latency measurement."""
    torch, dist = harness.torch, harness.dist
    dist.barrier()
    torch.cuda.synchronize()
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, profile_memory=False, with_stack=False,
        ) as profiler:
            graph.replay()
            torch.cuda.synchronize()
        names = Counter(event.name for event in profiler.events()
                        if str(event.device_type).split(".")[-1] == "CUDA")
        local = {"rank": harness.rank, "status": "captured",
                 "cuda_kernel_names": dict(sorted(names.items()))}
        if not names:
            raise RuntimeError("profiler returned no CUDA kernel events")
    except Exception as error:
        local = {"rank": harness.rank, "status": "failed",
                 "error": f"{type(error).__name__}: {error}"}
    records = [None] * dist.get_world_size()
    dist.all_gather_object(records, local)
    evidence = {"schema": "untimed-deepep-trtllm-recipe-kernel-names-v1",
                "status": "diagnostic_only_not_accuracy_or_performance",
                "replays_per_rank": 1, "route": "all_same_host", "per_rank": records}
    persistence = [None]
    if harness.rank == 0:
        path = comparison.output.with_name(comparison.output.stem + ".recipe-kernels.json")
        try:
            with path.open("x") as output:
                json.dump(evidence, output, sort_keys=True)
                output.write("\n")
            print("DEEPEP_TRTLLM_RECIPE_KERNELS_JSON=" + json.dumps({
                "path": str(path), "rank0": records[0], "diagnostic_only": True,
            }, sort_keys=True), flush=True)
        except Exception as error:
            persistence[0] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(persistence, src=0)
    if persistence[0] is not None:
        raise RuntimeError(f"failed to persist recipe kernel names: {persistence[0]}")
    if any(record["status"] != "captured" for record in records):
        raise RuntimeError(f"recipe kernel profile failed: {records}")
    return evidence


def _benchmark_pair(harness, comparator, comparison, snapshots):
    torch, dist = harness.torch, harness.dist
    case = harness.cases["all_same_host"]
    harness.copy_inputs(case)
    comparator.select_route(case)
    graphs = {"megamoe": _capture(harness.launch, torch, dist),
              "deepep_trtllm": _capture(comparator.launch, torch, dist)}
    receipt = dict(comparator.last_receipt)
    routes = {}
    try:
        if comparison.profile_recipe:
            comparator.recipe_profile = _profile_recipe(graphs["deepep_trtllm"], harness, comparison)
        epoch_evidence = _validate_payload_refresh(harness, comparator, graphs, comparison, receipt)
        for route in accuracy.MATCHED_BENCHMARK_ROUTES:
            case = harness.cases[route]
            harness.copy_inputs(case)
            comparator.select_route(case)
            for _ in range(comparison.comparison_warmups):
                for graph in graphs.values():
                    graph.replay()
            torch.cuda.synchronize()
            expected_trt = _post_weight_oracle(harness, case)
            comparator.output.fill_(float("nan"))
            graphs["deepep_trtllm"].replay()
            torch.cuda.synchronize()
            receipt_evidence = _validate_receipt(receipt, case, harness, f"comparison/{route}")
            oracle = _validate_output(comparator.output, expected_trt, f"{route}/trt-oracle",
                                      comparison, torch, dist)
            backend_diff = _validate_backend_drift(comparator.output, snapshots[route],
                                                    f"{route}/native-recipe-drift", comparison, torch, dist)
            for _ in range(comparison.comparison_conditioning):
                for graph in graphs.values():
                    graph.replay()
            torch.cuda.synchronize()
            harness.stats.zero_()
            dist.barrier()
            torch.cuda.synchronize()
            events = {name: [] for name in graphs}
            for replay in range(comparison.comparison_replays):
                order = tuple(graphs) if replay % 2 == 0 else tuple(reversed(graphs))
                for name in order:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[name].replay()
                    end.record()
                    events[name].append((start, end))
            torch.cuda.synchronize()
            harness.assert_stats(case.expected_local_stats * comparison.comparison_replays,
                                 f"comparison/{route}")
            harness.assert_guards(f"comparison/{route}")
            accuracy._assert_bitwise_equal(harness.output, snapshots[route],
                                          f"comparison/{route}/megamoe", torch)
            _validate_output(comparator.output, expected_trt, f"{route}/trt-final",
                             comparison, torch, dist)
            route_record = {"accuracy": {"trt_oracle": oracle, "backend_quantization_drift": backend_diff,
                                          "exact_receive_counts": receipt_evidence},
                            "received_allocation_rows": comparator.last_recv_capacity}
            for name, pairs in events.items():
                route_record[name] = accuracy._distributed_timing_record(
                    [float(start.elapsed_time(end) * 1000.0) for start, end in pairs], torch, dist)
            # Keep the alternating A/B evidence and independently measure each
            # backend after rewarming its own instruction/cache working set.
            for name, graph in graphs.items():
                for _ in range(comparison.comparison_conditioning):
                    graph.replay()
                torch.cuda.synchronize()
                harness.stats.zero_()
                dist.barrier()
                torch.cuda.synchronize()
                pairs = []
                for _ in range(comparison.comparison_replays):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    graph.replay()
                    end.record()
                    pairs.append((start, end))
                torch.cuda.synchronize()
                if name == "megamoe":
                    harness.assert_stats(case.expected_local_stats * comparison.comparison_replays,
                                         f"comparison-isolated/{route}")
                    harness.assert_guards(f"comparison-isolated/{route}")
                    accuracy._assert_bitwise_equal(harness.output, snapshots[route],
                                                  f"comparison-isolated/{route}", torch)
                else:
                    _validate_output(comparator.output, expected_trt,
                                     f"{route}/trt-isolated-final", comparison, torch, dist)
                    _validate_receipt(receipt, case, harness, f"comparison-isolated/{route}")
                route_record[name]["isolated"] = accuracy._distributed_timing_record(
                    [float(start.elapsed_time(end) * 1000.0) for start, end in pairs], torch, dist)
            routes[route] = route_record
        return routes, epoch_evidence
    finally:
        _retire_graphs(graphs)
        receipt.clear()


def _validate_payload_refresh(harness, comparator, graphs, comparison, receipt):
    """Prove same graph/state consumes changing payloads and returns to epoch 0."""
    torch, dist, args = harness.torch, harness.dist, harness.args
    route = harness.cases["half_remote"]
    originals = (harness.inputs.x.clone(), harness.inputs.x_sf.clone(),
                 harness.inputs.oracle_x.clone(), comparator.sf.clone())
    metrics = []
    prior_reference = None
    harness.stats.zero_()
    try:
        for step, epoch in enumerate((0, 1, 0)):
            inputs = accuracy._make_inputs(harness.rank, args, torch, harness.deep_gemm,
                                          input_epoch=epoch)
            harness.inputs.x.copy_(inputs.x)
            harness.inputs.x_sf.copy_(inputs.x_sf)
            harness.inputs.oracle_x.copy_(inputs.oracle_x)
            comparator.sf.copy_(inputs.x_sf)
            harness.copy_inputs(route)
            comparator.select_route(route)
            expected_mega = accuracy._analytic_oracle(inputs.oracle_x, route, args,
                                                     harness.experts_per_rank, torch)
            expected_trt = _post_weight_oracle(harness, route)
            owner_grouped_trt = _post_weight_oracle(harness, route, group_by_owner=True)
            activation_bf16_trt = _post_weight_oracle(harness, route, activation_bf16=True)
            activation_bf16_grouped_trt = _post_weight_oracle(harness, route,
                                                             group_by_owner=True, activation_bf16=True)
            original_trt = _post_weight_oracle(harness, route,
                                               quantizer_recipe="deepgemm_ceil448")
            original_grouped_trt = _post_weight_oracle(harness, route, group_by_owner=True,
                                                       quantizer_recipe="deepgemm_ceil448")
            if prior_reference is not None and torch.equal(prior_reference, expected_trt):
                raise AssertionError("payload epochs failed to change expected output")
            prior_reference = expected_trt.clone()
            harness.output.fill_(float("nan"))
            comparator.output.fill_(float("nan"))
            for graph in graphs.values():
                graph.replay()
            torch.cuda.synchronize()
            harness.assert_stats(route.expected_local_stats * (step + 1),
                                 f"comparison-payload/{step}/{epoch}")
            harness.assert_guards(f"comparison-payload/{step}/{epoch}")
            metrics.append({
                "epoch": epoch,
                "exact_receive_counts": _validate_receipt(receipt, route, harness,
                                                            f"payload/{step}/{epoch}"),
                "megamoe": _validate_output(harness.output, expected_mega,
                                             f"payload/{step}/megamoe", comparison, torch, dist),
                "deepep_trtllm": _validate_output(comparator.output, expected_trt,
                                                  f"payload/{step}/trt", comparison, torch, dist,
                                                  extra_references={
                                                      "owner_grouped_bf16_finalize": owner_grouped_trt,
                                                      "activation_bf16_before_fp8": activation_bf16_trt,
                                                      "activation_bf16_before_fp8_and_owner_grouped": activation_bf16_grouped_trt,
                                                      "original_deepgemm_ceil448": original_trt,
                                                      "original_deepgemm_ceil448_owner_grouped": original_grouped_trt,
                                                  },
                                                  reference_structure={
                                                      "gate": ["FP4xFP8_GEMM1_to_BF16", "SwiGLU_FP32",
                                                               "intermediate_MXFP8_gran32_exponent_only_SF_saturating_RNE",
                                                               "GEMM2_to_BF16", "route_weight_FP32",
                                                               "all_routes_sum_FP32", "final_BF16"],
                                                      "alternate": "sum_per_owner_FP32_then_BF16_then_sum_owners_FP32_then_final_BF16",
                                                      "alternate_changes_gate": False,
                                                      "native_recipe_evidence": TRT_RECIPE_EVIDENCE,
                                                      "payload_epoch": epoch,
                                                      "bf16_activation_reference_evidence": "vllm tests/kernels/moe/test_ocp_mx_moe.py reference_moe passes SwiGLU result through t.to(torch.bfloat16) before mxfp8_quantize",
                                                      "worst_element_per_slot": _worst_slot_breakdown(harness, route, comparator.output, expected_trt),
                                                  }),
                "backend_quantization_drift": _validate_backend_drift(
                    comparator.output, harness.output, f"payload/{step}/native-recipe-drift",
                    comparison, torch, dist),
            })
        return {"sequence": [0, 1, 0], "same_graphs": True,
                "output_nan_poison_each_replay": True, "checks": metrics}
    finally:
        harness.inputs.x.copy_(originals[0])
        harness.inputs.x_sf.copy_(originals[1])
        harness.inputs.oracle_x.copy_(originals[2])
        comparator.sf.copy_(originals[3])
        harness.copy_inputs(route)


def _diagnose_components(harness, comparator, comparison):
    """Separate event-instrumented graph; never used as full-path perf samples."""
    torch, dist = harness.torch, harness.dist
    names = ("dispatch_begin", "dispatch_complete", "route_preparation_complete",
             "expert_finalize_complete", "combine_complete", "output_ready")
    markers = {name: torch.cuda.Event(enable_timing=True, external=True) for name in names}
    graph = _capture(lambda: comparator.launch(markers), torch, dist)
    receipt = dict(comparator.last_receipt)
    intervals = {
        "dispatch": (names[0], names[1]),
        "receive_id_and_route_pack": (names[1], names[2]),
        "trtllm_experts_and_finalize": (names[2], names[3]),
        "combine": (names[3], names[4]),
        "output_copy": (names[4], names[5]),
        "instrumented_total": (names[0], names[5]),
    }
    result = {
        "scope": "separate_event_instrumented_cuda_graph_component_diagnostics",
        "performance_claim": False,
        "notes": ["Same-GPU CUDA-event differences only.",
                  "External event nodes perturb this graph; use uninstrumented isolated totals for performance.",
                  "TRTLLM experts includes internal routing, both GEMMs, activation and finalization.",
                  "Component maxima can belong to different ranks; do not add max-rank component medians."],
        "routes": {},
    }
    try:
        for route in accuracy.MATCHED_BENCHMARK_ROUTES:
            case = harness.cases[route]
            comparator.select_route(case)
            for _ in range(comparison.comparison_warmups):
                graph.replay()
            torch.cuda.synchronize()
            samples = {name: [] for name in intervals}
            for _ in range(comparison.component_replays):
                dist.barrier()
                torch.cuda.synchronize()
                graph.replay()
                torch.cuda.synchronize()
                for name, (start, end) in intervals.items():
                    samples[name].append(float(markers[start].elapsed_time(markers[end]) * 1000.0))
            expected = _post_weight_oracle(harness, case)
            _validate_output(comparator.output, expected, f"component/{route}", comparison, torch, dist)
            _validate_receipt(receipt, case, harness, f"component/{route}")
            result["routes"][route] = {
                name: accuracy._distributed_timing_record(values, torch, dist)
                for name, values in samples.items()
            }
        return result
    finally:
        _retire_graphs({"component": graph})
        graph = None
        markers.clear()
        receipt.clear()


def _benchmark_source_placement(harness, comparator, comparison):
    """Time only backend-required copies from the shared prequantized input."""
    torch, dist = harness.torch, harness.dist
    case = harness.cases["half_remote"]
    graphs = {
        "megamoe": _capture(lambda: harness.copy_inputs(case), torch, dist),
        # DeepEP consumes canonical activation/scales directly. Its static
        # graph in this runner stages only the route IDs and weights.
        "deepep_trtllm": _capture(lambda: comparator.select_route(case), torch, dist),
    }
    result = {
        "scope": "backend_source_placement_from_prequantized_inputs",
        "input_quantization_included": False,
        "megamoe_operations": ["activation_copy", "scale_copy", "route_copy", "weight_copy"],
        "deepep_operations": ["route_copy", "weight_copy"],
        "deepep_activation_and_scales": "read_shared_canonical_tensors_directly",
        "samples": {},
    }
    try:
        for name, graph in graphs.items():
            for _ in range(comparison.comparison_warmups):
                graph.replay()
            torch.cuda.synchronize()
            pairs = []
            for _ in range(comparison.comparison_replays):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                pairs.append((start, end))
            torch.cuda.synchronize()
            result["samples"][name] = accuracy._distributed_timing_record(
                [float(start.elapsed_time(end) * 1000.0) for start, end in pairs], torch, dist)
        return result
    finally:
        _retire_graphs(graphs)


def validate_clean_experiment_environment():
    for name, required in (("DG_MEGAMOE_GIN_COMBINE_EXPERTS_PER_WAVE", "0"),
                           ("DG_MEGAMOE_GIN_COMBINE_BARRIER_WARPS", "1")):
        if os.getenv(name, required) != required:
            raise ValueError(f"clean ctx1 candidate requires retired {name}={required}")


def dispatch_candidate_metadata(flags):
    """Describe the selected transport path; do not infer measured overlap."""
    raw = flags["DG_MEGAMOE_GIN_DISPATCH_OVERLAP"]
    if raw not in ("0", "1"):
        raise ValueError("dispatch overlap metadata requires a canonical0/1 flag")
    enabled = raw == "1"
    combine_raw = flags.get("DG_MEGAMOE_GIN_COMBINE_OVERLAP", "0")
    if combine_raw not in ("0", "1"):
        raise ValueError("combine overlap metadata requires a canonical0/1 flag")
    combine_enabled = combine_raw == "1"
    if combine_enabled and (not enabled or flags.get("DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT") != "1"):
        raise ValueError("combine overlap metadata requires dispatch overlap1 and single context1")
    strongva_raw = flags.get(
        "DG_MEGAMOE_GIN_STRONGVA_COMBINE_TERMINAL", "0")
    if strongva_raw not in ("0", "1"):
        raise ValueError("StrongVA combine-terminal metadata requires a canonical0/1 flag")
    strongva_enabled = strongva_raw == "1"
    if strongva_enabled and not combine_enabled:
        raise ValueError("StrongVA combine terminal requires combine overlap1")
    return {
        "candidate_family": ("direct_control_first_dispatch_ready_coalesced_direct_reduce_strongva_terminal" if strongva_enabled else
                             "direct_control_first_dispatch_ready_coalesced_direct_reduce_preload_late_flush" if combine_enabled else
                             "direct_control_first_dispatch" if enabled else
                             "clean_single_combine_context_only"),
        "combine_overlap_contract": {
            "requested_raw": combine_raw, "requested": combine_enabled,
            "strongva_terminal_requested_raw": strongva_raw,
            "strongva_terminal_requested": strongva_enabled,
            "readiness_unit": "complete expert output, using actual expert assignment counts",
            "producer_target": "ceil(actual expert assignments / actual BM) * (H / BN)",
            "combine_schedule": (
                "peer_parallel_ready_coalesced_spans_with_final_strongva_terminal"
                if strongva_enabled else
                "peer_parallel_ready_coalesced_spans_then_late_header"
            ),
            "ready_selection_policy": "warp_parallel_readiness_peer_independent_bounded_ready_coalescing",
            "early_payload_policy": "each peer independently selects at most eight already-ready adjacent expert spans; never waits for future readiness",
            "readiness_tracking": "common monotonic discovered-ready experts; independent per-peer pending-ready selection",
            "submission_granularity": "one nonaggregate PUT per bounded contiguous ready batch; fixed cap8",
            "combine_payload_local_completion": {
                "requested_by_combine_overlap": combine_enabled,
                "eligibility": (
                    "world_uniform_bulk_direct_remote_after_host_scratch_preflight"
                    if strongva_enabled else
                    "early_record_combine_path_and_scratch_alias_fits"
                ),
                "completion": ("late_cleanup_post_terminal_same_context_peer_flush"
                               if strongva_enabled else
                               "late_header_same_context_peer"),
                "payload_only_flush_before_handoff": False,
                "all_input_flushes_retained": True,
                "late_header_put_and_flush_retained": not strongva_enabled,
                "original_handoff_and_grid_order_retained": True,
                "final_world_put_barrier_retained": not strongva_enabled,
                "source_storage_retained_until_late_header_flush": not strongva_enabled,
                "header_flush_does_not_prove_remote_visibility": not strongva_enabled,
                "fallback": "unchanged_full_packet_local_flush",
                "slot101_writer_present": False,
                "policy_not_device_observation": True,
            },
            "direct_reducer": {
                "requested": combine_enabled,
                "additional_source_local_ordinal_bytes": 3072,
                "required_scratch_extent_bytes": 62720,
                "fit_policy": "combine_overlap_eligible_and_source_local_inverse_map_fits",
                "address_preparation": "lane_local_token_row_pointer_preload_before_chunk_loop",
                "full_warp_pointer_gather_before_elected_issuer": True,
                "metadata_resolution_passes_per_active_remote_assignment_per_token": 1,
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
            "ready_batch_max_experts": 8,
            "ready_batch_cap_is_compile_time_constant": True,
            "frozen_ready_snapshot_no_fill_wait": True,
            "issuer_acquires_every_batched_expert": True,
            "span_descriptors": "saved dispatch source/expert prefixes and existing exact assignment counts; no return-record metadata walk",
            "nonempty_masks": "two uint32 expert masks per peer; saved at dispatch and immutable until launch completion",
            "sent_entry_role": "final all-peer submission-complete bookkeeping, not early readiness",
            "eligibility": "existing remote direct+bulk eligibility and per-rank expert-ready storage fit",
            "scratch_layout": {
                "ready_uint32": 56, "sent_uint32": 56,
                "saved_source_expert_prefix_uint32": 8 * 56,
                "saved_source_nonempty_mask_uint32": 8 * 2,
                "tail_bytes": 2304, "direct_control_bytes": 57344,
                "required_scratch_bytes": 59648,
                "byte_capacity_depends_on_bm": False,
            },
            "fit_failure": "unchanged full-packet SC1 combine",
            "local_and_t64_paths": "unchanged local path or non-direct fallback",
            "count_headers_and_final_put_barrier": "late, unchanged completion scope",
            "physical_overlap_measured": False,
            "compute_hints_tiling_sm_count_and_math_changed": False,
            "basis": "selected flag and source policy, not device branch or physical-overlap measurement",
        },
        "dispatch_overlap_contract": {
            "requested_raw": raw,
            "requested": enabled,
            "eligible_control_before_activation_scale_weight_payload": enabled,
            "separate_payload_terminal_acquire_before_payload_reads": enabled,
            "ineligible_and_local_only_paths": "unchanged fallback or skipped remote dispatch",
            "compute_hints_tiling_sm_count_and_math_changed": False,
            "basis": "selected transport flag and source contract, not a measured-overlap observation",
        },
    }


def main():
    validate_clean_experiment_environment()
    comparison, args = parse_args()
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("use torchrun with two hosts and eight ranks per host")
    os.environ["DG_MEGAMOE_GIN_DIAGNOSTICS"] = "0"
    for name in ("DG_MEGAMOE_GIN_ACTIVITY_GATE_OPT", "DG_MEGAMOE_GIN_DISPATCH_WARP_SCAN",
                 "DG_MEGAMOE_GIN_COOP_DIRECT_PACK", "DG_MEGAMOE_GIN_PRECONSENSUS_PACK"):
        os.environ[name] = "1"
    original_graph_stress = accuracy._run_graph_stress
    state = {}

    def stress_then_compare(harness, snapshots, torch, dist):
        original_graph_stress(harness, snapshots, torch, dist)
        flags = {name: os.getenv(name, "0") for name in (
            "DG_MEGAMOE_GIN_DIAGNOSTICS", "DG_MEGAMOE_GIN_ACTIVITY_GATE_OPT",
            "DG_MEGAMOE_GIN_DISPATCH_WARP_SCAN", "DG_MEGAMOE_GIN_COOP_DIRECT_PACK",
            "DG_MEGAMOE_GIN_PRECONSENSUS_PACK",
            "DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT",
            "DG_MEGAMOE_GIN_DISPATCH_OVERLAP",
            "DG_MEGAMOE_GIN_COMBINE_OVERLAP",
            "DG_MEGAMOE_GIN_STRONGVA_COMBINE_TERMINAL",
        )}
        rank_flags = [None] * dist.get_world_size()
        dist.all_gather_object(rank_flags, flags)
        if any(record != flags for record in rank_flags):
            raise AssertionError(f"comparator MegaMoE flags differ across ranks: {rank_flags}")
        comparator = DeepEPTRTLLM(harness, comparison)
        try:
            routes, payload_refresh = _benchmark_pair(harness, comparator, comparison, snapshots)
            components = (_diagnose_components(harness, comparator, comparison)
                          if comparison.component_replays else None)
            preparation = (_benchmark_source_placement(harness, comparator, comparison)
                           if comparison.compare_preparation else None)
        finally:
            # Numerical gates gather every rank before raising, so all peers
            # reach destruction even on a reference mismatch. Graphs have
            # already been retired by the capture helpers' finally blocks.
            active_error = sys.exc_info()[0] is not None
            try:
                torch.cuda.synchronize()
                comparator.last_receipt = None
                comparator.buffer.destroy()
            except Exception as cleanup_error:
                if not active_error:
                    raise
                print(f"DeepEP cleanup also failed: {cleanup_error}", file=sys.stderr, flush=True)
        allocated_context_count = int(harness.buffer.gin_context.launch_descriptor_snapshot()["context_count"])
        state["record"] = {
            "schema": "shared-input-megamoe-deepep-trtllm-v2", "operator": "SwiGLU",
            "historical_production_SITU_comparison": False,
            "timing_scope": "prequantized_activation_and_routes_to_combined_bf16_output",
            "excluded": ["input_quantization", "route_generation", "source_buffer_copies"],
            "aggregation": "p50_of_max_rank_cuda_event_duration_per_replay",
            "schedule": "route_isolated_alternating_backend_order_plus_backend_isolated_rewarm",
            "warmups": comparison.comparison_warmups,
            "conditioning": comparison.comparison_conditioning,
            "replays": comparison.comparison_replays,
            "shape": {"tokens_per_rank": args.num_tokens, "hidden": args.hidden,
                      "intermediate_hidden": args.intermediate_hidden,
                      "num_experts": args.num_experts, "num_topk": args.num_topk,
                      "world_size": dist.get_world_size()},
            "math": {"megamoe_fast_math": args.fast_math,
                     "activation_clamp": args.activation_clamp,
                     "topk_weights": "shared_exact_powers_of_two",
                     "megamoe_weight_position": "before_intermediate_mxfp8_quantization",
                     "trtllm_weight_position": "expert_finalization",
                     "intermediate_quantizers_identical": False,
                     "megamoe_intermediate_scale": "2**ceil(log2(amax/448)), per32 UE8M0",
                     "trtllm_intermediate_scale": "2**(floor(log2(amax))-8), per32 UE8M0 with saturating FP8 RNE",
                     "trtllm_activation_before_quantization": "FP32; no BF16 activation cast",
                     "trtllm_native_recipe_evidence": TRT_RECIPE_EVIDENCE,
                     "per_backend_correctness_gates": "own_recipe_reference_relative_l2_and_max_abs",
                     "cross_backend_gate": "relative_l2_drift_ceiling_only; original absolute rule reported separately",
                     "backend_bitwise_equivalence_required": False,
                     "max_relative_l2": comparison.comparison_rel_l2,
                     "max_abs": comparison.comparison_max_abs},
            "input_pattern": "heterogeneous_sf" if getattr(args, "heterogeneous_input_scales", False) else "selected_periodic",
            "canonical_weights_sha256": comparator.weight_sha256,
            "megamoe_weights_verified_byte_identical": True,
            "runtime": comparator.api_evidence, "routes": routes,
            "recipe_kernel_profile": comparator.recipe_profile,
            "megamoe_flags": flags,
            **dispatch_candidate_metadata(flags),
            "combine_barrier_warps": 1,
            "expert_wave_implementation_present": False,
            "cooperative_barrier_implementation_present": False,
            "single_combine_context_contract": {
                "requested": int(flags["DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT"]),
                "allocated_contexts_unchanged": True,
                "allocated_context_count": allocated_context_count,
                "effective_matched_route_contexts": {
                    name: [] if name == "all_same_host" else (
                        [1] if flags["DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT"] == "1"
                        else list(range(allocated_context_count)))
                    for name in accuracy.MATCHED_BENCHMARK_ROUTES},
                "input_context_mapping_unchanged": True,
                "remote_t64_fallback": "original all-context Put barrier",
                "basis": "known K3 uniform eligibility, not runtime kernel introspection",
            },
            "source_placement": preparation,
            "component_diagnostics": components,
            "same_graph_payload_refresh": payload_refresh,
        }

    rank = int(os.environ.get("RANK", "0"))
    with patch.object(accuracy, "_run_graph_stress", stress_then_compare):
        accuracy._worker(int(os.environ["LOCAL_RANK"]), 8, args)
    if rank == 0:
        state["record"]["status"] = "passed"
        state["record"]["accuracy_and_teardown_passed"] = True
        with comparison.output.open("x") as output:
            json.dump(state["record"], output, sort_keys=True)
            output.write("\n")
        print("DEEPEP_TRTLLM_ISOLATED_JSON=" + json.dumps(state["record"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
