"""Deterministic, dependency-free MegaMoE transport accuracy stress test.

This is intentionally a standalone executable instead of a pytest unit test.  It
requires two or more CUDA ranks and a built DeepGEMM extension; missing CUDA,
DeepGEMM, or the requested topology is a hard failure, never a skip.  DeepEP and
TileLang are not imported.

Small, same-host smoke test::

    python tests/test_mega_moe_accuracy.py --num-processes 2 \
        --eager-iterations 16 --graph-replays 32

Production-shape, two-host launch (example)::

    torchrun --nnodes 2 --nproc-per-node 8 \
        tests/test_mega_moe_accuracy.py --k3 --decode-mns 8 \
        --require-cross-host --require-gin

The expert matrices are sparse by construction, with one nonzero per output
row.  This makes it possible to calculate an independent analytic oracle while
still exercising both GEMMs, SwiGLU, top-k weighting, owner dispatch, combine,
masking, and cumulative receive statistics.  Each local expert slot has exactly
the same weights on every rank.  Therefore changing only the owner rank must
leave the result bitwise unchanged, which is the transport-equivalence check
used for both NVLink today and a GIN-backed symmetric buffer later.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import socket
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


ROUTE_MODES: Tuple[str, ...] = (
    "all_local",
    "all_same_host",
    "all_peer",
    "all_remote",
    "different_peers",
    "all_masked",
    "mixed",
)


@dataclass
class RouteCase:
    name: str
    topk_idx: Any
    topk_weights: Any
    expected_local_stats: Any
    owner_kind: str


@dataclass
class KernelInputs:
    x: Any
    x_sf: Optional[Any]
    oracle_x: Any


@dataclass
class KernelWeights:
    l1: Any
    l2: Any


def _load_runtime() -> Tuple[Any, Any, Any]:
    try:
        import torch
        import torch.distributed as dist
    except Exception as exc:
        raise RuntimeError(
            "MegaMoE accuracy requires a CUDA-enabled PyTorch installation; "
            "the test does not skip missing runtime dependencies"
        ) from exc

    try:
        import deep_gemm
    except Exception as exc:
        raise RuntimeError(
            "MegaMoE accuracy requires an importable, built DeepGEMM extension; "
            "the test does not fall back to DeepEP/TileLang or skip correctness"
        ) from exc
    return torch, dist, deep_gemm


def _check_runtime(torch: Any, deep_gemm: Any) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("MegaMoE accuracy requires CUDA; refusing to skip")
    major, minor = torch.cuda.get_device_capability()
    if major != 10:
        raise RuntimeError(
            f"MegaMoE accuracy requires an SM100-family GPU, got capability {major}.{minor}"
        )
    required = (
        "get_symm_buffer_for_mega_moe",
        "transform_weights_for_mega_moe",
        "bf16_mega_moe",
        "fp8_fp4_mega_moe",
    )
    missing = [name for name in required if not hasattr(deep_gemm, name)]
    if missing:
        raise RuntimeError(f"built DeepGEMM is missing MegaMoE APIs: {missing}")
    if int(os.getenv("DG_COMM_KERNEL_DEBUG", "0")) != 0:
        raise RuntimeError(
            "DG_COMM_KERNEL_DEBUG must be 0: debug zeroing defeats persistent-buffer canaries"
        )


def _configure_symmetric_memory_backend(
    args: argparse.Namespace, torch: Any, dist: Any
) -> Tuple[Optional[str], Optional[Any]]:
    """Select the cross-host allocator required by the GIN accuracy path.

    Production callers may configure symmetric memory before constructing a
    MegaMoE buffer.  This standalone harness owns that setup so a successful
    ``--require-gin`` result proves the registered allocation itself spans the
    two hosts instead of accidentally using the CUDA/IPC-only default.
    """
    if not args.require_gin:
        return None, None
    try:
        import torch.distributed._symmetric_memory as symm_mem
    except Exception as exc:
        raise RuntimeError(
            "--require-gin needs torch.distributed._symmetric_memory"
        ) from exc

    set_backend = getattr(symm_mem, "set_backend", None)
    get_backend = getattr(symm_mem, "get_backend", None)
    if not callable(set_backend) or not callable(get_backend):
        raise RuntimeError(
            "--require-gin needs symmetric-memory set_backend/get_backend support"
        )

    # Some PyTorch builds still require the compatibility registration to
    # attach the process-group store/ranks used by the NCCL host communicator.
    # It is a no-op when the group was already registered.
    enable_group = getattr(symm_mem, "enable_symm_mem_for_group", None)
    if callable(enable_group):
        enable_group(dist.group.WORLD.group_name)
    set_backend("NCCL")

    # Publish ProcessGroupNCCL's device/host communicator pair before the
    # symmetric-memory rendezvous looks it up.  Object collectives are not a
    # sufficient warmup for this registry in all PyTorch builds.
    communicator_probe = torch.ones(1, dtype=torch.int32, device="cuda")
    dist.all_reduce(communicator_probe, group=dist.group.WORLD)
    expected = dist.get_world_size()
    actual = int(communicator_probe.item())
    if actual != expected:
        raise RuntimeError(
            "NCCL communicator publication probe returned "
            f"{actual}, expected {expected}"
        )
    device = torch.device("cuda", torch.cuda.current_device())
    backend = str(get_backend(device))
    if backend.upper() != "NCCL":
        raise RuntimeError(
            f"--require-gin selected NCCL symmetric memory, but got {backend!r}"
        )

    # PyTorch 2.13 wheels built before upstream fix 1b6b0f687149 can have a
    # duplicated, DSO-local NCCL symmetric-memory registry.  Publish the
    # native ProcessGroupNCCL communicator through the supported external
    # registration bridge so rendezvous and the process group see one entry.
    try:
        from torch.distributed._symmetric_memory._nccl import (
            register_external_nccl_comm,
        )
        backend_impl = dist.group.WORLD._get_backend(device)
        comm_ptr = getattr(backend_impl, "_comm_ptr", None)
        if not callable(comm_ptr):
            raise RuntimeError("ProcessGroupNCCL does not expose _comm_ptr()")
        comm_ptr_value = int(comm_ptr())
        if comm_ptr_value == 0:
            raise RuntimeError("ProcessGroupNCCL returned a null communicator pointer")
        registration = register_external_nccl_comm(
            dist.group.WORLD.group_name,
            comm_ptr_value,
            device,
            backend_impl,
        )
    except Exception as exc:
        raise RuntimeError(
            "--require-gin could not bridge ProcessGroupNCCL into the "
            "symmetric-memory communicator registry"
        ) from exc
    return backend.upper(), registration


def _init_distributed(local_rank: int, local_world_size: int, torch: Any, dist: Any) -> None:
    torch.cuda.set_device(local_rank)
    if "LOCAL_RANK" in os.environ:
        kwargs: Dict[str, Any] = {"backend": "nccl", "init_method": "env://"}
    else:
        master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
        master_port = int(os.getenv("MASTER_PORT", "29617"))
        kwargs = {
            "backend": "nccl",
            "init_method": f"tcp://{master_addr}:{master_port}",
            "world_size": local_world_size,
            "rank": local_rank,
        }
    if "device_id" in inspect.signature(dist.init_process_group).parameters:
        kwargs["device_id"] = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(**kwargs)


def _all_hostnames(dist: Any) -> List[str]:
    host = socket.gethostname()
    result: List[Optional[str]] = [None] * dist.get_world_size()
    dist.all_gather_object(result, host)
    if any(item is None for item in result):
        raise AssertionError(f"failed to gather rank hostnames: {result}")
    return [str(item) for item in result]


def _rank_candidates(rank: int, hostnames: Sequence[str]) -> Tuple[List[int], List[int], List[int]]:
    peers = [candidate for candidate in range(len(hostnames)) if candidate != rank]
    same_host = [candidate for candidate in peers if hostnames[candidate] == hostnames[rank]]
    remote_host = [candidate for candidate in peers if hostnames[candidate] != hostnames[rank]]
    return peers, same_host, remote_host


def _owner_for(
    mode: str,
    rank: int,
    token: int,
    slot: int,
    hostnames: Sequence[str],
) -> Optional[int]:
    peers, same_host, remote_host = _rank_candidates(rank, hostnames)
    if mode == "all_local":
        return rank
    if mode == "all_same_host":
        if same_host:
            # Select a cyclic lane relative to the source rank, rather than an
            # index into a source-dependent list with the source removed.  For
            # every (token, top-k slot), this is a permutation of the ranks in
            # one host.  Consequently every destination receives exactly the
            # same local-expert histogram as all_local, while every payload
            # still crosses an NVLink peer (the nonzero offset excludes self).
            local_group = [
                candidate
                for candidate, hostname in enumerate(hostnames)
                if hostname == hostnames[rank]
            ]
            source_lane = local_group.index(rank)
            offset = 1 + (token + slot) % (len(local_group) - 1)
            return local_group[(source_lane + offset) % len(local_group)]
        return peers[(token + slot) % len(peers)]
    if mode == "all_masked":
        return None
    if mode == "all_peer":
        return peers[(token + slot) % len(peers)]
    if mode == "all_remote":
        local_group = [
            candidate
            for candidate, hostname in enumerate(hostnames)
            if hostname == hostnames[rank]
        ]
        remote_hostnames = {hostnames[candidate] for candidate in remote_host}
        if (
            len(local_group) > 1
            and len(remote_host) == len(local_group)
            and len(remote_hostnames) == 1
        ):
            # Mirror the same cyclic lane into the other host.  The paired
            # lane is deliberately excluded so this has the same seven-peer
            # fanout as all_same_host on the 2x8 target.  Across all sources,
            # each destination still receives every (token, slot) exactly
            # once, matching both expert occupancy and GEMM block count.
            source_lane = local_group.index(rank)
            offset = 1 + (token + slot) % (len(local_group) - 1)
            return remote_host[(source_lane + offset) % len(remote_host)]
        candidates = remote_host if remote_host else peers
        return candidates[(token + slot) % len(candidates)]
    if mode == "different_peers":
        return peers[(token * 3 + slot) % len(peers)]
    if mode == "mixed":
        selector = (token + slot) % 4
        if selector == 0:
            return rank
        if selector == 2:
            return None
        candidates = remote_host if remote_host else peers
        return candidates[(token + slot) % len(candidates)]
    raise ValueError(f"unknown route mode: {mode}")


def _build_route_tensors(
    mode: str,
    rank: int,
    num_tokens: int,
    num_topk: int,
    experts_per_rank: int,
    hostnames: Sequence[str],
    torch: Any,
) -> Tuple[Any, Any]:
    idx_cpu = torch.full((num_tokens, num_topk), -1, dtype=torch.int64, device="cpu")
    weight_cpu = torch.zeros((num_tokens, num_topk), dtype=torch.float32, device="cpu")
    for token in range(num_tokens):
        for slot in range(num_topk):
            owner = _owner_for(mode, rank, token, slot, hostnames)
            if owner is None:
                continue
            local_expert = (token * num_topk + slot) % experts_per_rank
            idx_cpu[token, slot] = owner * experts_per_rank + local_expert
            # Exact powers of two avoid adding an unrelated weight-rounding variable.
            weight_cpu[token, slot] = 2.0 ** (-(slot % 3 + 1))
    return idx_cpu.cuda(), weight_cpu.cuda()


def _global_route_stats(
    topk_idx: Any,
    rank: int,
    experts_per_rank: int,
    torch: Any,
    dist: Any,
) -> Any:
    gathered = [torch.empty_like(topk_idx) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, topk_idx)
    flat = torch.stack(gathered).flatten()
    begin, end = rank * experts_per_rank, (rank + 1) * experts_per_rank
    owned = flat[(flat >= begin) & (flat < end)] - begin
    return torch.bincount(owned, minlength=experts_per_rank).to(torch.int32)


def _make_route_cases(
    rank: int,
    args: argparse.Namespace,
    experts_per_rank: int,
    hostnames: Sequence[str],
    torch: Any,
    dist: Any,
) -> Dict[str, RouteCase]:
    cases: Dict[str, RouteCase] = {}
    _peers, same_host, remote_host = _rank_candidates(rank, hostnames)
    has_same_host = bool(same_host)
    has_remote = bool(remote_host)
    for mode in ROUTE_MODES:
        topk_idx, topk_weights = _build_route_tensors(
            mode,
            rank,
            args.num_tokens,
            args.num_topk,
            experts_per_rank,
            hostnames,
            torch,
        )
        expected_stats = _global_route_stats(topk_idx, rank, experts_per_rank, torch, dist)
        if mode == "all_remote":
            owner_kind = "cross_host" if has_remote else "same_host_peer_fallback"
        elif mode == "all_same_host":
            owner_kind = "same_host_peer" if has_same_host else "peer_fallback"
        elif mode == "all_local":
            owner_kind = "local"
        elif mode == "all_masked":
            owner_kind = "masked"
        else:
            owner_kind = "mixed_or_peer"
        cases[mode] = RouteCase(mode, topk_idx, topk_weights, expected_stats, owner_kind)
    return cases


def _input_values(rank: int, num_tokens: int, hidden: int, torch: Any) -> Any:
    token = torch.arange(num_tokens, device="cuda", dtype=torch.int64).view(-1, 1)
    column = torch.arange(hidden, device="cuda", dtype=torch.int64).view(1, -1)
    code = (rank * 131 + token * 17 + column * 7) % 30
    magnitude = (code % 15 + 1).to(torch.float32) / 16.0
    sign = torch.where((code & 1) == 0, 1.0, -1.0)
    return (magnitude * sign).to(torch.bfloat16)


def _make_inputs(rank: int, args: argparse.Namespace, torch: Any, deep_gemm: Any) -> KernelInputs:
    from deep_gemm.utils import per_token_cast_to_fp8
    from deep_gemm.utils.math import cast_back_from_fp8

    source = _input_values(rank, args.num_tokens, args.hidden, torch)
    if args.mma_type == "bf16xbf16":
        return KernelInputs(source, None, source.float())
    x_fp8, x_sf = per_token_cast_to_fp8(
        source, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    oracle_x = cast_back_from_fp8(x_fp8, x_sf, gran_k=32, use_packed_ue8m0=True)
    return KernelInputs(x_fp8, x_sf, oracle_x)


def _expert_scales(expert: int) -> Tuple[float, float, float]:
    # All are exact powers of two and exactly representable by BF16/FP4.
    gate = 0.5 if expert % 2 == 0 else 1.0
    up = 1.0 if expert % 3 == 0 else 0.5
    out = 0.5 if expert % 4 < 2 else 1.0
    return gate, up, out


def _gate_source(j: Any, expert: Any, hidden: int) -> Any:
    return (j * 17 + expert * 13 + 1) % hidden


def _up_source(j: Any, expert: Any, hidden: int) -> Any:
    return (j * 29 + expert * 7 + 3) % hidden


def _out_source(o: Any, expert: Any, intermediate: int) -> Any:
    return (o * 31 + expert * 11 + 5) % intermediate


def _make_bf16_weights(
    experts_per_rank: int,
    hidden: int,
    intermediate: int,
    torch: Any,
) -> Tuple[Any, Any]:
    l1 = torch.zeros(
        (experts_per_rank, intermediate * 2, hidden),
        device="cuda",
        dtype=torch.bfloat16,
    )
    l2 = torch.zeros(
        (experts_per_rank, hidden, intermediate),
        device="cuda",
        dtype=torch.bfloat16,
    )
    j = torch.arange(intermediate, device="cuda", dtype=torch.int64)
    o = torch.arange(hidden, device="cuda", dtype=torch.int64)
    for expert in range(experts_per_rank):
        gate_scale, up_scale, out_scale = _expert_scales(expert)
        l1[expert, j, _gate_source(j, expert, hidden)] = gate_scale
        l1[expert, intermediate + j, _up_source(j, expert, hidden)] = up_scale
        l2[expert, o, _out_source(o, expert, intermediate)] = out_scale
    return l1, l2


def _cast_grouped_fp4(weights: Any, torch: Any, deep_gemm: Any) -> Tuple[Any, Any]:
    from deep_gemm.utils import per_token_cast_to_fp4

    groups, n, k = weights.shape
    packed = torch.empty((groups, n, k // 2), device="cuda", dtype=torch.int8)
    sf = torch.empty((groups, n, k // 32), device="cuda", dtype=torch.float32)
    for group in range(groups):
        packed[group], sf[group] = per_token_cast_to_fp4(
            weights[group], use_ue8m0=True, gran_k=32
        )
    transformed_sf = deep_gemm.transform_sf_into_required_layout(
        sf, n, k, (1, 32), groups
    )
    return packed, transformed_sf


def _make_kernel_weights(
    experts_per_rank: int,
    args: argparse.Namespace,
    torch: Any,
    deep_gemm: Any,
) -> KernelWeights:
    l1_bf16, l2_bf16 = _make_bf16_weights(
        experts_per_rank, args.hidden, args.intermediate_hidden, torch
    )
    if args.mma_type == "bf16xbf16":
        l1, l2 = deep_gemm.transform_weights_for_mega_moe(l1_bf16, l2_bf16)
    else:
        l1_fp4 = _cast_grouped_fp4(l1_bf16, torch, deep_gemm)
        l2_fp4 = _cast_grouped_fp4(l2_bf16, torch, deep_gemm)
        l1, l2 = deep_gemm.transform_weights_for_mega_moe(l1_fp4, l2_fp4)
    del l1_bf16, l2_bf16
    return KernelWeights(l1, l2)


def _weight_fingerprint(weights: KernelWeights, torch: Any, dist: Any) -> List[float]:
    tensors: List[Any] = []
    for value in (weights.l1, weights.l2):
        tensors.extend(value if isinstance(value, tuple) else (value,))
    parts = []
    for tensor in tensors:
        flat = tensor.contiguous().view(torch.uint8).flatten()
        stride = max(1, flat.numel() // 4096)
        sample = flat[::stride][:4096].to(torch.float64)
        positions = torch.arange(1, sample.numel() + 1, device="cuda", dtype=torch.float64)
        parts.extend((sample.sum(), (sample * positions).sum()))
    fingerprint = torch.stack(parts)
    gathered = [torch.empty_like(fingerprint) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, fingerprint)
    for peer, peer_fingerprint in enumerate(gathered):
        if not torch.equal(fingerprint, peer_fingerprint):
            raise AssertionError(
                f"rank-local expert weights differ from rank {peer}; "
                f"local={fingerprint.tolist()} peer={peer_fingerprint.tolist()}"
            )
    return [float(value) for value in fingerprint.cpu().tolist()]


def _analytic_oracle(
    oracle_x: Any,
    case: RouteCase,
    args: argparse.Namespace,
    experts_per_rank: int,
    torch: Any,
) -> Any:
    from deep_gemm.utils import per_token_cast_to_fp8
    from deep_gemm.utils.math import cast_back_from_fp8

    result = torch.zeros(
        (args.num_tokens, args.hidden), device="cuda", dtype=torch.float32
    )
    j = torch.arange(args.intermediate_hidden, device="cuda", dtype=torch.int64).view(1, -1)
    o = torch.arange(args.hidden, device="cuda", dtype=torch.int64).view(1, -1)
    clamp = args.activation_clamp
    for slot in range(args.num_topk):
        valid = case.topk_idx[:, slot] >= 0
        expert = torch.remainder(case.topk_idx[:, slot].clamp_min(0), experts_per_rank)
        expert_2d = expert.view(-1, 1)

        gate_scale = torch.where((expert_2d % 2) == 0, 0.5, 1.0)
        up_scale = torch.where((expert_2d % 3) == 0, 1.0, 0.5)
        out_scale = torch.where((expert_2d % 4) < 2, 0.5, 1.0)

        gate_idx = _gate_source(j, expert_2d, args.hidden)
        up_idx = _up_source(j, expert_2d, args.hidden)
        gate = (torch.gather(oracle_x, 1, gate_idx) * gate_scale).to(torch.bfloat16).float()
        up = (torch.gather(oracle_x, 1, up_idx) * up_scale).to(torch.bfloat16).float()
        if math.isfinite(clamp):
            gate = torch.minimum(gate, torch.full_like(gate, clamp))
            up = torch.clamp(up, min=-clamp, max=clamp)

        gate = gate / (1.0 + torch.exp(-gate))
        activation = gate * up
        activation = activation * case.topk_weights[:, slot].view(-1, 1)
        if args.mma_type == "fp8xfp4":
            activation_fp8, activation_sf = per_token_cast_to_fp8(
                activation, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
            )
            activation = cast_back_from_fp8(
                activation_fp8,
                activation_sf,
                gran_k=32,
                use_packed_ue8m0=True,
            )
        else:
            activation = activation.to(torch.bfloat16).float()

        out_idx = _out_source(o, expert_2d, args.intermediate_hidden)
        contribution = torch.gather(activation, 1, out_idx) * out_scale
        contribution = contribution.to(torch.bfloat16).float()
        contribution = torch.where(valid.view(-1, 1), contribution, 0.0)
        result = result + contribution
    return result.to(torch.bfloat16)


def _symmetric_diff(actual: Any, expected: Any) -> Tuple[float, float]:
    actual64, expected64 = actual.double(), expected.double()
    denominator = (actual64.square() + expected64.square()).sum()
    if denominator.item() == 0:
        relative = 0.0
    else:
        relative = float((1.0 - 2.0 * (actual64 * expected64).sum() / denominator).item())
    max_abs = float((actual64 - expected64).abs().max().item())
    return max(0.0, relative), max_abs


def _assert_bitwise_equal(actual: Any, expected: Any, label: str, torch: Any) -> None:
    if torch.equal(actual, expected):
        return
    mismatch = actual.view(torch.uint8) != expected.view(torch.uint8)
    first_byte = int(mismatch.flatten().nonzero()[0].item())
    element = first_byte // actual.element_size()
    raise AssertionError(
        f"{label}: bitwise mismatch, bytes={int(mismatch.sum().item())}, "
        f"first_element={element}, actual={actual.flatten()[element].item()}, "
        f"expected={expected.flatten()[element].item()}"
    )


class AccuracyHarness:
    def __init__(
        self,
        rank: int,
        args: argparse.Namespace,
        inputs: KernelInputs,
        weights: KernelWeights,
        cases: Dict[str, RouteCase],
        buffer: Any,
        torch: Any,
        dist: Any,
        deep_gemm: Any,
    ) -> None:
        self.rank = rank
        self.args = args
        self.inputs = inputs
        self.weights = weights
        self.cases = cases
        self.buffer = buffer
        self.torch = torch
        self.dist = dist
        self.deep_gemm = deep_gemm
        self.experts_per_rank = args.num_experts // dist.get_world_size()
        self.stats = torch.zeros(self.experts_per_rank, device="cuda", dtype=torch.int32)

        guard_elements = max(256, args.hidden)
        self.output_storage = torch.full(
            (guard_elements + args.num_tokens * args.hidden + guard_elements,),
            -91.5,
            device="cuda",
            dtype=torch.bfloat16,
        )
        self.output = self.output_storage[
            guard_elements : guard_elements + args.num_tokens * args.hidden
        ].view(args.num_tokens, args.hidden)
        self.prefix = self.output_storage[:guard_elements]
        self.suffix = self.output_storage[-guard_elements:]
        self.prefix_expected = self.prefix.clone()
        self.suffix_expected = self.suffix.clone()

        self._initialize_input_tail_canaries()

    def _initialize_input_tail_canaries(self) -> None:
        start = self.args.num_tokens
        self.tail_expected: Dict[str, Any] = {}
        if start >= self.buffer.x.shape[0]:
            raise AssertionError(
                "symmetric buffer has no inactive input row for overrun canaries; "
                "increase --num-max-tokens-per-rank"
            )
        self.buffer.x[start:].fill_(-3.25)
        self.tail_expected["x"] = self.buffer.x[start:].clone()
        if self.inputs.x_sf is not None:
            self.buffer.x_sf[start:].fill_(0x53535353)
            self.tail_expected["x_sf"] = self.buffer.x_sf[start:].clone()
        self.buffer.topk_idx[start:].fill_(-777)
        self.buffer.topk_weights[start:].fill_(-17.0)
        self.tail_expected["topk_idx"] = self.buffer.topk_idx[start:].clone()
        self.tail_expected["topk_weights"] = self.buffer.topk_weights[start:].clone()

    def copy_inputs(self, case: RouteCase) -> None:
        count = self.args.num_tokens
        self.buffer.x[:count].copy_(self.inputs.x)
        if self.inputs.x_sf is not None:
            self.buffer.x_sf[:count].copy_(self.inputs.x_sf)
        self.buffer.topk_idx[:count].copy_(case.topk_idx)
        self.buffer.topk_weights[:count].copy_(case.topk_weights)

    def launch(self) -> None:
        kwargs = {
            "y": self.output,
            "l1_weights": self.weights.l1,
            "l2_weights": self.weights.l2,
            "sym_buffer": self.buffer,
            "cumulative_local_expert_recv_stats": self.stats,
            "activation_clamp": self.args.activation_clamp,
            "fast_math": self.args.fast_math,
        }
        if self.args.mma_type == "bf16xbf16":
            self.deep_gemm.bf16_mega_moe(**kwargs)
        else:
            self.deep_gemm.fp8_fp4_mega_moe(**kwargs)

    def eager(self, case: RouteCase) -> Any:
        self.output.fill_(float("nan"))
        self.copy_inputs(case)
        self.launch()
        self.torch.cuda.synchronize()
        return self.output

    def assert_guards(self, label: str) -> None:
        if not self.torch.equal(self.prefix, self.prefix_expected):
            raise AssertionError(f"{label}: output prefix canary was overwritten")
        if not self.torch.equal(self.suffix, self.suffix_expected):
            raise AssertionError(f"{label}: output suffix canary was overwritten")
        if not bool(self.torch.isfinite(self.output).all().item()):
            count = int((~self.torch.isfinite(self.output)).sum().item())
            raise AssertionError(f"{label}: output contains {count} unwritten/non-finite elements")

        start = self.args.num_tokens
        current = {
            "x": self.buffer.x[start:],
            "topk_idx": self.buffer.topk_idx[start:],
            "topk_weights": self.buffer.topk_weights[start:],
        }
        if self.inputs.x_sf is not None:
            current["x_sf"] = self.buffer.x_sf[start:]
        for name, value in current.items():
            if not self.torch.equal(value, self.tail_expected[name]):
                raise AssertionError(f"{label}: inactive symmetric-buffer {name} canary changed")

    def assert_stats(self, expected: Any, label: str) -> None:
        if self.torch.equal(self.stats, expected):
            return
        raise AssertionError(
            f"{label}: cumulative stats mismatch; "
            f"actual={self.stats.cpu().tolist()} expected={expected.cpu().tolist()}"
        )


def _oracle_thresholds(args: argparse.Namespace) -> Tuple[float, float]:
    if args.oracle_rel_diff is None:
        rel_diff = 3.0e-5 if args.mma_type == "fp8xfp4" else 2.0e-2
    else:
        rel_diff = args.oracle_rel_diff
    if args.oracle_max_abs is None:
        max_abs = 0.25 if args.mma_type == "fp8xfp4" else 0.125
    else:
        max_abs = args.oracle_max_abs
    return rel_diff, max_abs


def _snapshot_and_check_oracles(
    harness: AccuracyHarness,
    torch: Any,
    dist: Any,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    snapshots: Dict[str, Any] = {}
    metrics: Dict[str, Dict[str, float]] = {}
    rel_limit, abs_limit = _oracle_thresholds(harness.args)
    for name in ROUTE_MODES:
        case = harness.cases[name]
        harness.stats.zero_()
        actual = harness.eager(case)
        harness.assert_guards(f"snapshot/{name}")
        harness.assert_stats(case.expected_local_stats, f"snapshot/{name}")
        if name == "all_masked" and not torch.equal(actual, torch.zeros_like(actual)):
            raise AssertionError("all_masked must produce bitwise-zero output")

        oracle = _analytic_oracle(
            harness.inputs.oracle_x,
            case,
            harness.args,
            harness.experts_per_rank,
            torch,
        )
        relative, max_abs = _symmetric_diff(actual, oracle)
        global_metrics = torch.tensor([relative, max_abs], device="cuda", dtype=torch.float64)
        dist.all_reduce(global_metrics, op=dist.ReduceOp.MAX)
        worst_relative, worst_max_abs = [float(value) for value in global_metrics.cpu().tolist()]
        if worst_relative > rel_limit or worst_max_abs > abs_limit:
            raise AssertionError(
                f"snapshot/{name}: analytic oracle failed across ranks: "
                f"symmetric_diff={worst_relative:.8g} (limit {rel_limit:.8g}), "
                f"max_abs={worst_max_abs:.8g} (limit {abs_limit:.8g})"
            )
        metrics[name] = {
            "worst_symmetric_diff": worst_relative,
            "worst_max_abs": worst_max_abs,
        }
        snapshots[name] = actual.clone()

    # The local expert slot, input, top-k slot, and top-k weight are identical;
    # only the owner rank changes.  Any byte difference is a transport bug.
    for name in ("all_same_host", "all_peer", "all_remote", "different_peers"):
        _assert_bitwise_equal(
            snapshots[name], snapshots["all_local"], f"transport/{name}", torch
        )
    return snapshots, metrics


def _run_eager_stress(
    harness: AccuracyHarness,
    snapshots: Dict[str, Any],
    torch: Any,
) -> None:
    initial = torch.arange(
        harness.experts_per_rank, device="cuda", dtype=torch.int32
    ) + 17
    harness.stats.copy_(initial)
    expected = initial.clone()
    for epoch in range(harness.args.eager_iterations):
        name = ROUTE_MODES[epoch % len(ROUTE_MODES)]
        case = harness.cases[name]
        output = harness.eager(case)
        expected.add_(case.expected_local_stats)
        harness.assert_stats(expected, f"eager/{epoch}/{name}")
        harness.assert_guards(f"eager/{epoch}/{name}")
        _assert_bitwise_equal(output, snapshots[name], f"eager/{epoch}/{name}", torch)


def _capture_graph(harness: AccuracyHarness, torch: Any, dist: Any) -> Tuple[Any, Any, Any]:
    static_idx = torch.empty_like(harness.cases["all_local"].topk_idx)
    static_weights = torch.empty_like(harness.cases["all_local"].topk_weights)

    def copy_static_and_launch() -> None:
        harness.output.fill_(float("nan"))
        count = harness.args.num_tokens
        harness.buffer.x[:count].copy_(harness.inputs.x)
        if harness.inputs.x_sf is not None:
            harness.buffer.x_sf[:count].copy_(harness.inputs.x_sf)
        harness.buffer.topk_idx[:count].copy_(static_idx)
        harness.buffer.topk_weights[:count].copy_(static_weights)
        harness.launch()

    static_idx.copy_(harness.cases["all_local"].topk_idx)
    static_weights.copy_(harness.cases["all_local"].topk_weights)
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            copy_static_and_launch()
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    harness.stats.zero_()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        copy_static_and_launch()
    torch.cuda.synchronize()
    # Capture executes the body once.  Reset only the user-visible cumulative
    # counter; the kernel's internal workspace has already completed/cleaned.
    harness.stats.zero_()
    dist.barrier()
    return graph, static_idx, static_weights


def _run_graph_stress(
    harness: AccuracyHarness,
    snapshots: Dict[str, Any],
    torch: Any,
    dist: Any,
) -> Tuple[Any, Any, Any]:
    graph, static_idx, static_weights = _capture_graph(harness, torch, dist)
    expected = torch.zeros_like(harness.stats)
    for epoch in range(harness.args.graph_replays):
        # Offset from eager order so adjacent phases do not repeat one scenario.
        name = ROUTE_MODES[(epoch + 1) % len(ROUTE_MODES)]
        case = harness.cases[name]
        static_idx.copy_(case.topk_idx)
        static_weights.copy_(case.topk_weights)
        graph.replay()
        torch.cuda.synchronize()
        expected.add_(case.expected_local_stats)
        harness.assert_stats(expected, f"graph/{epoch}/{name}")
        harness.assert_guards(f"graph/{epoch}/{name}")
        _assert_bitwise_equal(harness.output, snapshots[name], f"graph/{epoch}/{name}", torch)
    return graph, static_idx, static_weights


def _timing_summary(values: Sequence[float]) -> Dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("timing summary requires at least one value")

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "min": ordered[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def _capture_launch_only_graph(
    harness: AccuracyHarness,
    torch: Any,
    dist: Any,
) -> Any:
    """Capture only MegaMoE, leaving route/input preparation outside timing."""
    harness.copy_inputs(harness.cases["all_local"])
    harness.stats.zero_()
    for _ in range(3):
        harness.launch()
    torch.cuda.synchronize()

    harness.stats.zero_()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        harness.launch()
    torch.cuda.synchronize()
    # Capture executes once.  It is setup, not a measured replay.
    harness.stats.zero_()
    dist.barrier()
    return graph


def _benchmark_graph_routes(
    harness: AccuracyHarness,
    graph: Any,
    snapshots: Dict[str, Any],
    torch: Any,
    dist: Any,
) -> Dict[str, Any]:
    """Time full captured MegaMoE steps and retain the slowest rank per replay.

    `all_same_host` and `all_remote` keep routed expert occupancy exactly
    matched while changing owner payload transport from mapped NVLink to GIN.
    Their delta is therefore the communication latency still exposed after the
    fused kernel's compute/communication overlap, not raw one-sided-operation
    latency or an overlap percentage.  Route copies are ordered before the
    start event, and the measured graph contains only the MegaMoE launch.

    Routes are interleaved with a rotating order on every cycle to limit clock,
    thermal, and fixed-order bias.  CUDA events retain one paired sample per
    route and cycle; CPU launch and synchronize overhead are outside timing.
    """
    route_names = list(harness.args.benchmark_routes)
    count = harness.args.num_tokens

    def select_route(name: str) -> None:
        case = harness.cases[name]
        harness.buffer.topk_idx[:count].copy_(case.topk_idx)
        harness.buffer.topk_weights[:count].copy_(case.topk_weights)

    def rotated(cycle: int) -> List[str]:
        offset = cycle % len(route_names)
        return route_names[offset:] + route_names[:offset]

    # Warm each path in the same rotating order used by measurement.
    harness.stats.zero_()
    torch.cuda.synchronize()
    dist.barrier()
    for cycle in range(harness.args.benchmark_warmups):
        for name in rotated(cycle):
            select_route(name)
            graph.replay()
    torch.cuda.synchronize()

    # Reset the only user-visible cumulative state so measured replays have an
    # exact postcondition independent of capture and warmup execution.
    harness.stats.zero_()
    harness.output.fill_(float("nan"))
    torch.cuda.synchronize()
    dist.barrier()

    starts: Dict[str, List[Any]] = {name: [] for name in route_names}
    ends: Dict[str, List[Any]] = {name: [] for name in route_names}
    for cycle in range(harness.args.benchmark_replays):
        for name in rotated(cycle):
            select_route(name)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            starts[name].append(start)
            ends[name].append(end)
            start.record()
            graph.replay()
            end.record()
    torch.cuda.synchronize()

    expected_stats = torch.zeros_like(harness.stats)
    for name in route_names:
        expected_stats.add_(
            harness.cases[name].expected_local_stats,
            alpha=harness.args.benchmark_replays,
        )
    harness.assert_stats(expected_stats, "benchmark/measured-batch")
    harness.assert_guards("benchmark/measured-batch")
    last_name = rotated(harness.args.benchmark_replays - 1)[-1]
    _assert_bitwise_equal(
        harness.output,
        snapshots[last_name],
        f"benchmark/final/{last_name}",
        torch,
    )

    results: Dict[str, Any] = {}
    max_rank_samples: Dict[str, List[float]] = {}
    for name in route_names:
        case = harness.cases[name]
        local_us_values = [
            float(start.elapsed_time(end) * 1.0e3)
            for start, end in zip(starts[name], ends[name])
        ]
        local_us = torch.tensor(local_us_values, device="cuda", dtype=torch.float64)
        max_rank_us = local_us.clone()
        dist.all_reduce(max_rank_us, op=dist.ReduceOp.MAX)

        local_summary = _timing_summary(local_us_values)
        summary_vector = torch.tensor(
            [
                local_summary["min"],
                local_summary["p50"],
                local_summary["p90"],
                local_summary["p99"],
                local_summary["max"],
                local_summary["mean"],
            ],
            device="cuda",
            dtype=torch.float64,
        )
        gathered = [torch.empty_like(summary_vector) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, summary_vector)
        rank_p50_values = [float(item[1].item()) for item in gathered]
        rank_mean_values = [float(item[5].item()) for item in gathered]

        global_us_values = [float(value) for value in max_rank_us.cpu().tolist()]
        max_rank_samples[name] = global_us_values
        results[name] = {
            "owner_kind": case.owner_kind,
            "max_rank_per_replay_us": _timing_summary(global_us_values),
            "local_rank_p50_us": _timing_summary(rank_p50_values),
            "local_rank_mean_us": _timing_summary(rank_mean_values),
        }

    baseline = results.get("all_same_host")
    if baseline is not None:
        baseline_samples = max_rank_samples["all_same_host"]
        baseline_p50 = baseline["max_rank_per_replay_us"]["p50"]
        for name, value in results.items():
            p50 = value["max_rank_per_replay_us"]["p50"]
            paired_delta = [
                sample - reference
                for sample, reference in zip(
                    max_rank_samples[name], baseline_samples
                )
            ]
            paired_summary = _timing_summary(paired_delta)
            value["paired_delta_vs_same_host_us"] = paired_summary
            value["p50_delta_vs_same_host_us"] = paired_summary["p50"]
            value["p50_ratio_vs_same_host"] = p50 / baseline_p50
    dist.barrier()
    return results


def _validate_args(args: argparse.Namespace, world_size: int) -> None:
    if world_size < 2:
        raise ValueError("MegaMoE owner-transport accuracy requires at least two ranks")
    if args.num_experts % world_size != 0:
        raise ValueError(
            f"--num-experts={args.num_experts} must be divisible by world size {world_size}"
        )
    experts_per_rank = args.num_experts // world_size
    if args.num_topk > experts_per_rank:
        raise ValueError(
            f"--num-topk={args.num_topk} must be <= experts per rank {experts_per_rank} "
            "for unique paired-owner routes"
        )
    if args.hidden % 256 != 0:
        raise ValueError("--hidden must be divisible by 256 for MegaMoE combine chunks")
    if args.intermediate_hidden % 128 != 0:
        raise ValueError("--intermediate-hidden must be divisible by 128")
    if args.mma_type == "fp8xfp4" and (
        args.hidden % 512 != 0 or args.intermediate_hidden % 512 != 0
    ):
        raise ValueError(
            "FP8xFP4 hidden and intermediate dimensions must both be divisible by 512 "
            "for packed scale-buffer TMA alignment"
        )
    if args.num_tokens <= 0:
        raise ValueError("--num-tokens must be positive")
    if args.num_max_tokens_per_rank <= args.num_tokens:
        raise ValueError(
            "--num-max-tokens-per-rank must exceed --num-tokens to exercise input canaries"
        )
    if args.num_topk + args.num_shared_experts > 32:
        raise ValueError("top-k plus shared experts must fit in one warp")
    if args.num_shared_experts != 0:
        raise ValueError(
            "this transport-isolation harness requires --num-shared-experts=0; "
            "shared-expert numerical coverage belongs in a separate test"
        )
    if args.require_gin:
        if args.mma_type != "fp8xfp4":
            raise ValueError(
                "--require-gin currently requires --mma-type=fp8xfp4; the BF16 "
                "MegaMoE launch has not been wired to the GIN transport"
            )
        if world_size != 16:
            raise ValueError(
                f"--require-gin currently requires exactly 16 ranks (2x8), got {world_size}"
            )
        if args.gin_completion_batch != 1:
            raise ValueError(
                "the correctness-first fused GIN path currently requires "
                "--gin-completion-batch=1; use the standalone probe for the "
                "B=2/4/8 design sweep"
            )
        row_bytes = args.hidden * 2
        if row_bytes % args.gin_combine_chunk_bytes != 0:
            raise ValueError(
                f"--gin-combine-chunk-bytes={args.gin_combine_chunk_bytes} must "
                f"divide one BF16 output row ({row_bytes} bytes)"
            )


def _validate_gin_host_placement(
    args: argparse.Namespace,
    world_size: int,
    hostnames: Sequence[str],
) -> None:
    if not args.require_gin:
        return
    if world_size != 16 or len(hostnames) != 16:
        raise RuntimeError(
            "--require-gin transport evidence requires exactly 16 gathered rank hostnames"
        )
    if len(set(hostnames)) < 2:
        raise RuntimeError(
            "--require-gin was requested, but all ranks are on one host; no "
            "activation or combine payload could exercise the IB/GIN path"
        )
    first_host, second_host = hostnames[0], hostnames[8]
    if (
        first_host == second_host
        or any(host != first_host for host in hostnames[:8])
        or any(host != second_host for host in hostnames[8:])
    ):
        raise RuntimeError(
            "--require-gin requires contiguous 2x8 rank placement: ranks 0..7 "
            "on one host and ranks 8..15 on a different host"
        )


def _gin_transport_evidence(
    buffer: Any,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    hostnames: Sequence[str],
    symmetric_memory_backend: Optional[str],
    symmetric_memory_registration: Optional[Any],
) -> Dict[str, Any]:
    """Validate the facts needed to call a run GIN transport coverage.

    A truthy ``gin_enabled`` property alone is insufficient: the accuracy run
    must route data across hosts through the FP8xFP4 GIN kernel and the live
    context descriptor must carry the exact tuning requested by the CLI.
    """
    gin_enabled_value = getattr(buffer, "gin_enabled", None)
    gin_enabled = bool(gin_enabled_value) if gin_enabled_value is not None else False
    if not args.require_gin:
        return {
            "requested": "auto",
            "gin_enabled": gin_enabled if gin_enabled_value is not None else None,
            "gin_api_present": gin_enabled_value is not None,
        }

    _validate_gin_host_placement(args, world_size, hostnames)
    if not gin_enabled:
        detail = (
            "the buffer API does not expose gin_enabled"
            if gin_enabled_value is None
            else "buffer.gin_enabled is false"
        )
        raise RuntimeError(f"--require-gin was requested, but {detail}")
    if symmetric_memory_backend != "NCCL":
        raise RuntimeError(
            "--require-gin requires the NCCL symmetric-memory backend, got "
            f"{symmetric_memory_backend!r}"
        )
    if symmetric_memory_registration is None:
        raise RuntimeError(
            "--require-gin requires a live symmetric-memory NCCL communicator "
            "registration"
        )

    context = getattr(buffer, "gin_context", None)
    snapshot_fn = getattr(context, "launch_descriptor_snapshot", None)
    if context is None or not callable(snapshot_fn):
        raise RuntimeError(
            "--require-gin requires a live context with launch_descriptor_snapshot()"
        )
    snapshot = dict(snapshot_fn())
    expected = {
        "enabled": True,
        "rank": rank,
        "world_size": world_size,
        "lsa_rank": rank % 8,
        "lsa_size": 8,
        "context_count": 9,
        "requested_context_count": 9,
        "queue_depth": args.gin_queue_depth,
        "world_barrier_count": 3,
        "completion_batch": args.gin_completion_batch,
        "combine_chunk_bytes": args.gin_combine_chunk_bytes,
        "outbox_depth": args.gin_outbox_depth,
    }
    mismatches = []
    for name, expected_value in expected.items():
        actual_value = snapshot.get(name)
        if name == "enabled":
            actual_value = bool(actual_value)
        if actual_value != expected_value:
            mismatches.append(f"{name}={actual_value!r}, expected {expected_value!r}")

    gin_type = getattr(context, "gin_type_string", None)
    if gin_type != "gdaki":
        mismatches.append(f"gin_type_string={gin_type!r}, expected 'gdaki'")
    if int(snapshot.get("connection_count", 0)) <= 0:
        mismatches.append("connection_count must be positive")
    if int(snapshot.get("window", 0)) == 0:
        mismatches.append("registered window handle is null")
    if int(snapshot.get("dev_comm_bytes", 0)) <= 0:
        mismatches.append("device communicator descriptor is empty")

    context_buffer_bytes = int(getattr(context, "buffer_bytes", -1))
    tensor_buffer_bytes = int(buffer.buffer.numel() * buffer.buffer.element_size())
    if context_buffer_bytes != tensor_buffer_bytes:
        mismatches.append(
            f"buffer_bytes={context_buffer_bytes}, expected {tensor_buffer_bytes}"
        )
    raw_buffer_ptrs = [int(ptr) for ptr in buffer.handle.buffer_ptrs]
    buffer_offset = int(buffer.handle.offset)
    buffer_ptrs = [int(ptr) for ptr in buffer.buffer_ptrs]
    expected_lsa_ranks = [
        peer for peer, hostname in enumerate(hostnames)
        if hostname == hostnames[rank]
    ]
    nonzero_ranks = [peer for peer, ptr in enumerate(buffer_ptrs) if ptr != 0]
    missing_lsa_ranks = [
        peer for peer in expected_lsa_ranks if buffer_ptrs[peer] == 0
    ]
    if len(buffer_ptrs) != world_size:
        mismatches.append(
            f"symmetric buffer exposes {len(buffer_ptrs)} pointers, expected {world_size}"
        )
    if missing_lsa_ranks:
        mismatches.append(
            f"missing symmetric-memory aliases for local LSA ranks="
            f"{missing_lsa_ranks}; nonzero ranks={nonzero_ranks}"
        )
    if mismatches:
        raise RuntimeError(
            "--require-gin context/launch descriptor validation failed: "
            + "; ".join(mismatches)
        )

    return {
        "requested": "gin",
        "gin_enabled": True,
        "gin_api_present": True,
        "gin_type": gin_type,
        "symmetric_memory_backend": symmetric_memory_backend,
        "symmetric_memory_communicator_registration": "external_bridge",
        "cross_host_payload_routes": True,
        "launch_descriptor": {
            name: (bool(snapshot[name]) if name == "enabled" else snapshot[name])
            for name in expected
        },
        "connection_count": int(snapshot["connection_count"]),
        "registered_buffer_bytes": context_buffer_bytes,
        "symmetric_memory_buffer_offset": buffer_offset,
        "rank_local_raw_buffer_pointer": raw_buffer_ptrs[rank],
        "rank_local_adjusted_buffer_pointer": buffer_ptrs[rank],
        "rank_local_tensor_pointer": int(buffer.buffer.data_ptr()),
        "rank_local_pointer_alias_delta": (
            buffer_ptrs[rank] - int(buffer.buffer.data_ptr())
        ),
        "nonzero_symmetric_memory_ranks": nonzero_ranks,
    }


def _abort_buffer_rank_local(buffer: Optional[Any]) -> None:
    """Best-effort, non-collective cleanup for a failing worker.

    A rank may reach this path while another rank is still inside a kernel or
    collective.  Calling ``SymmBuffer.destroy()`` here would enter its
    collective teardown and can hide the original failure behind a deadlock.
    ``abort()`` is deliberately rank-local; cleanup errors are suppressed so
    they cannot replace the exception that caused the worker to fail.
    """
    if buffer is None:
        return
    abort = getattr(buffer, "abort", None)
    if not callable(abort):
        return
    try:
        abort()
    except BaseException:
        pass


def _synchronize_worker_success(dist: Any) -> bool:
    """Return success only after every rank reaches the final barrier."""
    dist.barrier()
    return True


def _unregister_symmetric_memory_comm(
    registration: Optional[Any], *, suppress_errors: bool
) -> None:
    if registration is None:
        return
    try:
        registration.unregister()
    except BaseException:
        if not suppress_errors:
            raise


def _teardown_worker(
    buffer: Optional[Any],
    dist: Any,
    synchronized_success: bool,
    symmetric_memory_registration: Optional[Any],
) -> None:
    """Tear down collectives only after every rank crossed the success gate."""
    if not synchronized_success:
        _abort_buffer_rank_local(buffer)
        _unregister_symmetric_memory_comm(
            symmetric_memory_registration, suppress_errors=True
        )
        return

    try:
        if buffer is not None:
            buffer.destroy()
    except BaseException:
        # A destroy failure must not be followed by process-group teardown;
        # doing so could strand a peer in SymmBuffer's collective destroy.
        _abort_buffer_rank_local(buffer)
        _unregister_symmetric_memory_comm(
            symmetric_memory_registration, suppress_errors=True
        )
        raise

    _unregister_symmetric_memory_comm(
        symmetric_memory_registration, suppress_errors=False
    )
    if dist.is_initialized():
        dist.destroy_process_group()


def _worker(local_rank: int, local_world_size: int, args: argparse.Namespace) -> None:
    torch, dist, deep_gemm = _load_runtime()
    buffer = None
    symmetric_memory_registration = None
    synchronized_success = False
    passed_record = None
    perf_record = None
    try:
        _init_distributed(local_rank, local_world_size, torch, dist)
        _check_runtime(torch, deep_gemm)
        (
            symmetric_memory_backend,
            symmetric_memory_registration,
        ) = _configure_symmetric_memory_backend(
            args, torch, dist
        )
        rank, world_size = dist.get_rank(), dist.get_world_size()
        _validate_args(args, world_size)
        hostnames = _all_hostnames(dist)
        if args.require_cross_host and len(set(hostnames)) < 2:
            raise RuntimeError(
                "--require-cross-host was requested, but every rank is on "
                f"{hostnames[0]}; refusing to relabel same-host NVLink as GIN coverage"
            )
        _validate_gin_host_placement(args, world_size, hostnames)

        experts_per_rank = args.num_experts // world_size
        cases = _make_route_cases(
            rank, args, experts_per_rank, hostnames, torch, dist
        )
        buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            dist.group.WORLD,
            args.num_experts,
            args.num_max_tokens_per_rank,
            args.num_topk,
            args.hidden,
            args.intermediate_hidden,
            num_shared_experts=0,
            mma_type=args.mma_type,
            enable_gin=args.require_gin,
            gin_completion_batch=args.gin_completion_batch,
            gin_combine_chunk_bytes=args.gin_combine_chunk_bytes,
            gin_outbox_depth=args.gin_outbox_depth,
            gin_queue_depth=args.gin_queue_depth,
        )
        transport_evidence = _gin_transport_evidence(
            buffer,
            args,
            rank,
            world_size,
            hostnames,
            symmetric_memory_backend,
            symmetric_memory_registration,
        )
        inputs = _make_inputs(rank, args, torch, deep_gemm)
        weights = _make_kernel_weights(experts_per_rank, args, torch, deep_gemm)
        fingerprint = _weight_fingerprint(weights, torch, dist)
        harness = AccuracyHarness(
            rank, args, inputs, weights, cases, buffer, torch, dist, deep_gemm
        )

        snapshots, oracle_metrics = _snapshot_and_check_oracles(harness, torch, dist)
        _run_eager_stress(harness, snapshots, torch)
        _run_graph_stress(
            harness, snapshots, torch, dist
        )
        if args.benchmark_replays > 0:
            benchmark_graph = _capture_launch_only_graph(harness, torch, dist)
            benchmark_metrics = _benchmark_graph_routes(
                harness,
                benchmark_graph,
                snapshots,
                torch,
                dist,
            )
        else:
            benchmark_metrics = None

        if rank == 0:
            rel_limit, abs_limit = _oracle_thresholds(args)
            result = {
                "status": "passed",
                "world_size": world_size,
                "hostnames": hostnames,
                "cross_host_exercised": len(set(hostnames)) > 1,
                "all_remote_realization": cases["all_remote"].owner_kind,
                "transport": transport_evidence,
                "mma_type": args.mma_type,
                "shape": {
                    "tokens_per_rank": args.num_tokens,
                    "max_tokens_per_rank": buffer.num_max_tokens_per_rank,
                    "hidden": args.hidden,
                    "intermediate_hidden": args.intermediate_hidden,
                    "num_experts": args.num_experts,
                    "num_topk": args.num_topk,
                },
                "iterations": {
                    "eager": args.eager_iterations,
                    "cuda_graph_replays": args.graph_replays,
                },
                "checks": {
                    "analytic_oracle": True,
                    "bitwise_owner_transport_equivalence": True,
                    "exact_cumulative_stats": True,
                    "output_and_input_canaries": True,
                    "nan_stale_output_detection": True,
                    "all_masked_bitwise_zero": True,
                },
                "oracle_limits": {
                    "symmetric_diff": rel_limit,
                    "max_abs": abs_limit,
                },
                "oracle_metrics": oracle_metrics,
                "weight_fingerprint": fingerprint,
            }
            passed_record = "MEGAMOE_ACCURACY_JSON=" + json.dumps(
                result, sort_keys=True)
            if benchmark_metrics is not None:
                perf_result = {
                    "status": "passed",
                    "world_size": world_size,
                    "cross_host_exercised": len(set(hostnames)) > 1,
                    "timing_scope": "launch_only_cuda_graph_device_time",
                    "aggregation": "slowest_rank_for_each_replay",
                    "route_schedule": "interleaved_rotating_paired_cycles",
                    "same_host_control": (
                        "GIN-enabled fused kernel with same-LSA payload routes; "
                        "not the legacy GIN-disabled kernel"
                    ),
                    "warmups": args.benchmark_warmups,
                    "replays": args.benchmark_replays,
                    "shape": result["shape"],
                    "gin": {
                        "type": transport_evidence["gin_type"],
                        "completion_batch": args.gin_completion_batch,
                        "combine_chunk_bytes": args.gin_combine_chunk_bytes,
                        "outbox_depth": args.gin_outbox_depth,
                        "queue_depth": args.gin_queue_depth,
                    },
                    "routes": benchmark_metrics,
                }
                perf_record = "MEGAMOE_PERF_JSON=" + json.dumps(
                    perf_result, sort_keys=True
                )

        # This must be the final operation that can diverge by rank before
        # collective teardown.  Only ranks that all cross this barrier may
        # call SymmBuffer.destroy() and destroy the process group.
        synchronized_success = _synchronize_worker_success(dist)
        if passed_record is not None:
            print(passed_record, flush=True)
        if perf_record is not None:
            print(perf_record, flush=True)
    finally:
        _teardown_worker(
            buffer,
            dist,
            synchronized_success,
            symmetric_memory_registration,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic MegaMoE owner transport accuracy/replay test"
    )
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--mma-type", choices=("fp8xfp4", "bf16xbf16"), default="fp8xfp4")
    parser.add_argument("--num-tokens", type=int, default=None)
    parser.add_argument("--num-max-tokens-per-rank", type=int, default=None)
    parser.add_argument("--decode-mns", type=int, default=None)
    parser.add_argument("--draft-width", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--intermediate-hidden", type=int, default=None)
    parser.add_argument("--num-experts", type=int, default=None)
    parser.add_argument("--num-topk", type=int, default=None)
    parser.add_argument("--num-shared-experts", type=int, default=0)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--fast-math", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eager-iterations", type=int, default=256)
    parser.add_argument("--graph-replays", type=int, default=1000)
    parser.add_argument(
        "--benchmark-replays",
        type=int,
        default=0,
        help=(
            "After correctness, time this many back-to-back CUDA graph replays "
            "per benchmark route (0 disables timing)"
        ),
    )
    parser.add_argument("--benchmark-warmups", type=int, default=10)
    parser.add_argument(
        "--benchmark-routes",
        nargs="+",
        choices=ROUTE_MODES,
        default=("all_local", "all_same_host", "all_remote"),
    )
    parser.add_argument("--oracle-rel-diff", type=float, default=None)
    parser.add_argument("--oracle-max-abs", type=float, default=None)
    parser.add_argument("--require-cross-host", action="store_true")
    parser.add_argument(
        "--require-gin",
        action="store_true",
        help=(
            "Require cross-host FP8xFP4 GIN payload routes and validate the live "
            "context/launch descriptor before running accuracy checks"
        ),
    )
    parser.add_argument(
        "--gin-completion-batch", type=int, choices=(1, 2, 4, 8), default=1
    )
    parser.add_argument(
        "--gin-combine-chunk-bytes",
        type=int,
        choices=(256, 1792, 3584, 7168),
        default=7168,
    )
    parser.add_argument(
        "--gin-outbox-depth", type=int, choices=(4, 8, 16), default=8
    )
    parser.add_argument("--gin-queue-depth", type=int, default=64)
    parser.add_argument(
        "--k3",
        action="store_true",
        help=(
            "Use matched K3 defaults: hidden 3584, intermediate 3072, "
            "896 experts, top-k 16, max tokens 384, draft width 3"
        ),
    )
    args = parser.parse_args()

    defaults = {
        "hidden": 3584 if args.k3 else 512,
        "intermediate_hidden": 3072 if args.k3 else 512,
        "num_experts": 896 if args.k3 else 8,
        "num_topk": 16 if args.k3 else 2,
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.draft_width is None:
        args.draft_width = 3 if args.k3 else 1
    if args.decode_mns is not None and args.num_tokens is not None:
        parser.error("use either --num-tokens or --decode-mns/--draft-width, not both")
    if args.decode_mns is not None:
        args.num_tokens = args.decode_mns * args.draft_width
    elif args.num_tokens is None:
        args.num_tokens = 8
    if args.num_max_tokens_per_rank is None:
        args.num_max_tokens_per_rank = 384 if args.k3 else args.num_tokens + 16
    if args.num_processes < 1:
        parser.error("--num-processes must be positive")
    if args.eager_iterations < 1 or args.graph_replays < 1:
        parser.error("eager and graph iteration counts must both be positive")
    if args.benchmark_replays < 0:
        parser.error("--benchmark-replays must be non-negative")
    if args.benchmark_replays > 0 and not args.require_gin:
        parser.error("--benchmark-replays currently requires --require-gin")
    if args.benchmark_warmups < 1:
        parser.error("--benchmark-warmups must be positive")
    if len(set(args.benchmark_routes)) != len(args.benchmark_routes):
        parser.error("--benchmark-routes must not contain duplicates")
    return args


def main() -> None:
    args = _parse_args()
    torch, _dist, _deep_gemm = _load_runtime()
    if "LOCAL_RANK" in os.environ:
        _worker(int(os.environ["LOCAL_RANK"]), int(os.getenv("LOCAL_WORLD_SIZE", "1")), args)
    else:
        torch.multiprocessing.spawn(
            _worker,
            args=(args.num_processes, args),
            nprocs=args.num_processes,
            join=True,
        )


if __name__ == "__main__":
    main()
