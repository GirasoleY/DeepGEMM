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

``--decode-mns`` counts decode sequences.  Each sequence verifies one target
token plus ``--draft-width`` draft tokens, so MNS8 with draft width 3 launches
32 tokens per rank.

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
    "half_remote",
    "all_peer",
    "all_remote",
    "different_peers",
    "all_masked",
    "mixed",
)
GIN_ACTIVITY_ASYMMETRIC_MODE = "single_remote_source"
GIN_ACTIVITY_TRANSITION_SEQUENCE: Tuple[str, ...] = (
    "all_same_host",
    GIN_ACTIVITY_ASYMMETRIC_MODE,
    "all_same_host",
    "half_remote",
    "all_same_host",
    "all_remote",
    "all_same_host",
)
GIN_ACTIVITY_TRANSITION_CYCLES = 3
GIN_BULK_TRANSITION_SEQUENCE: Tuple[str, ...] = ("48a", "64", "48b")
GIN_BULK_TRANSITION_COUNTS: Dict[str, int] = {
    "48a": 48,
    "64": 64,
    "48b": 48,
}
GIN_BULK_TRANSITION_INPUT_EPOCHS: Dict[str, int] = {
    "48a": 0,
    "64": 1,
    "48b": 2,
}
GIN_BULK_COMBINE_CAPACITY = 48 * 16
GIN_FAST_PATH_MAX_TOKENS = 48
GIN_FALLBACK_PROBE_TOKENS = 64

MATCHED_BENCHMARK_ROUTES: Tuple[str, ...] = (
    "all_same_host",
    "half_remote",
    "all_remote",
)
MATCHED_BENCHMARK_TOKEN_COUNTS: Tuple[int, ...] = (32, 40, 48)
MATCHED_BENCHMARK_BLOCK_M = 32
BALANCED_BENCHMARK_PERIOD: Tuple[str, ...] = (
    "all_same_host",
    "half_remote",
    "all_remote",
    "all_same_host",
    "all_remote",
    "half_remote",
)

GIN_LOCAL_ABLATION_ENV = "DG_MEGAMOE_GIN_LOCAL_ABLATION_STAGE"
GIN_ACTIVITY_GATE_OPT_ENV = "DG_MEGAMOE_GIN_ACTIVITY_GATE_OPT"
GIN_DISPATCH_WARP_SCAN_ENV = "DG_MEGAMOE_GIN_DISPATCH_WARP_SCAN"
GIN_COOP_DIRECT_PACK_ENV = "DG_MEGAMOE_GIN_COOP_DIRECT_PACK"
GIN_PRECONSENSUS_PACK_ENV = "DG_MEGAMOE_GIN_PRECONSENSUS_PACK"
GIN_SINGLE_COMBINE_CONTEXT_ENV = "DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT"
GIN_DISPATCH_OVERLAP_ENV = "DG_MEGAMOE_GIN_DISPATCH_OVERLAP"
GIN_COMBINE_OVERLAP_ENV = "DG_MEGAMOE_GIN_COMBINE_OVERLAP"
GIN_STRONGVA_COMBINE_TERMINAL_ENV = (
    "DG_MEGAMOE_GIN_STRONGVA_COMBINE_TERMINAL"
)
GIN_COMBINE_OWNER_WAVES_ENV = "DG_MEGAMOE_GIN_COMBINE_OWNER_WAVES"
GIN_EXPERIMENT_FLAG_ENVS: Tuple[str, ...] = (
    GIN_DISPATCH_WARP_SCAN_ENV,
    GIN_COOP_DIRECT_PACK_ENV,
    GIN_PRECONSENSUS_PACK_ENV,
)
# Legacy launchers explicitly enable every optimization in the tuple above.
# Protocol-changing, default-off experiments must not silently join that set.
GIN_PROTOCOL_FLAG_ENVS: Tuple[str, ...] = (
    GIN_SINGLE_COMBINE_CONTEXT_ENV, GIN_DISPATCH_OVERLAP_ENV, GIN_COMBINE_OVERLAP_ENV,
    GIN_STRONGVA_COMBINE_TERMINAL_ENV,
)
GIN_VALIDATED_FLAG_ENVS = GIN_EXPERIMENT_FLAG_ENVS + GIN_PROTOCOL_FLAG_ENVS
GIN_LOCAL_ABLATION_STAGES: Dict[int, str] = {
    0: "current_four_phase_gin_common_path",
    1: "cumulative_skip_paired_input_publish",
    2: "cumulative_skip_empty_remote_control",
    3: "cumulative_lsa_barriers_and_native_local_counts",
    4: "cumulative_bypass_outbox_and_drainer",
}


def _parse_gin_combine_owner_waves(raw: str) -> int:
    if raw not in ("0", "2", "4", "8"):
        raise argparse.ArgumentTypeError(
            "combine owner waves must be exactly 0, 2, 4, or 8")
    return int(raw)


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


@dataclass
class GinBulkTransitionPhase:
    name: str
    args: argparse.Namespace
    inputs: KernelInputs
    case: RouteCase
    oracle: Any


def _gin_fast_path_transition_requested(args: argparse.Namespace) -> bool:
    """Return whether this run needs the shared fast-path/fallback probe."""
    return bool(args.gin_bulk_combine or args.gin_direct_dispatch)


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
        "get_block_m_for_mega_moe",
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
    args: argparse.Namespace, torch: Any, dist: Any, *, force_nccl: bool = False
) -> Tuple[Optional[str], Optional[Any]]:
    """Select the cross-host allocator required by the GIN accuracy path.

    Production callers may configure symmetric memory before constructing a
    MegaMoE buffer.  This standalone harness owns that setup so a successful
    ``--require-gin`` result proves the registered allocation itself spans the
    two hosts instead of accidentally using the CUDA/IPC-only default.
    """
    # Explicit allocator-only opt-in for the separate GB200 native-NVLink
    # entrypoint. It does not enable a GIN context or change legacy defaults.
    if not args.require_gin and not force_nccl:
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
    num_topk: int = 16,
    num_tokens: int = 32,
) -> Optional[int]:
    if mode == GIN_ACTIVITY_ASYMMETRIC_MODE:
        mode = "all_remote" if rank == 0 else "all_same_host"
    if mode == "half_remote":
        # Permute route positions before splitting them between the two
        # occupancy-preserving controls.  A coprime stride keeps the split
        # exact while avoiding route/expert parity aliasing at the K3 shape.
        # Every exact-32 source sends 256/512 routes across the host boundary,
        # and every expert block contains both local and remote rows.
        route = token * num_topk + slot
        total_routes = num_tokens * num_topk
        stride = 5
        while math.gcd(stride, total_routes) != 1:
            stride += 2
        permuted_route = (route * stride) % total_routes
        mode = (
            "all_remote"
            if permuted_route < total_routes // 2
            else "all_same_host"
        )
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
            owner = _owner_for(
                mode,
                rank,
                token,
                slot,
                hostnames,
                num_topk=num_topk,
                num_tokens=num_tokens,
            )
            if owner is None:
                continue
            local_expert = (token * num_topk + slot) % experts_per_rank
            idx_cpu[token, slot] = owner * experts_per_rank + local_expert
            # Exact powers of two avoid adding an unrelated weight-rounding variable.
            weight_cpu[token, slot] = 2.0 ** (-(slot % 3 + 1))
    return idx_cpu.cuda(), weight_cpu.cuda()


def _build_bulk_transition_route_tensors(
    rank: int,
    num_tokens: int,
    num_topk: int,
    experts_per_rank: int,
    hostnames: Sequence[str],
    torch: Any,
) -> Tuple[Any, Any]:
    """Concentrate one source's routes on its paired remote owner.

    At top-k 16 this creates exactly 768 owner/source records for 48 tokens
    and 1,024 for 64 tokens.  The latter cannot fit either compact fast path,
    turning the transition check into a behavioral proof of uniform exact-r75
    fallback rather than relying only on the launch's requested token count.
    """
    world_size = len(hostnames)
    if world_size not in (8, 16) or num_topk != 16 or experts_per_rank < num_topk:
        raise ValueError(
            "fast-path transition routes require EP8/EP16, top-k 16, and at least "
            "16 experts per rank"
        )
    owner = (rank + world_size // 2) % world_size
    if hostnames[owner] == hostnames[rank]:
        raise ValueError(
            "fast-path transition paired owner must be on the other host"
        )

    idx_cpu = torch.empty(
        (num_tokens, num_topk), dtype=torch.int64, device="cpu"
    )
    weight_cpu = torch.empty(
        (num_tokens, num_topk), dtype=torch.float32, device="cpu"
    )
    for token in range(num_tokens):
        for slot in range(num_topk):
            local_expert = (token * num_topk + slot) % experts_per_rank
            idx_cpu[token, slot] = owner * experts_per_rank + local_expert
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
    route_modes = ROUTE_MODES + (
        (GIN_ACTIVITY_ASYMMETRIC_MODE,) if args.require_gin else ()
    )
    for mode in route_modes:
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
        elif mode == "half_remote":
            owner_kind = (
                "balanced_half_cross_host"
                if has_same_host and has_remote
                else "peer_fallback"
            )
        elif mode == "all_local":
            owner_kind = "local"
        elif mode == "all_masked":
            owner_kind = "masked"
        else:
            owner_kind = "mixed_or_peer"
        cases[mode] = RouteCase(mode, topk_idx, topk_weights, expected_stats, owner_kind)
    return cases


def _validate_single_remote_source_case(
    cases: Dict[str, RouteCase],
    rank: int,
    args: argparse.Namespace,
    experts_per_rank: int,
    hostnames: Sequence[str],
    dist: Any,
) -> List[int]:
    """Validate the actual launch tensor used for asymmetric gate coverage."""
    case = cases[GIN_ACTIVITY_ASYMMETRIC_MODE]
    valid = case.topk_idx >= 0
    owners = case.topk_idx.clamp_min(0) // experts_per_rank
    remote_owner_flags = owners.new_tensor(
        [hostname != hostnames[rank] for hostname in hostnames],
        dtype=owners.dtype,
    )
    local_cross = int((remote_owner_flags[owners] * valid).sum().item())
    cross_routes: List[Optional[int]] = [None] * dist.get_world_size()
    dist.all_gather_object(cross_routes, local_cross)
    actual = [int(value) for value in cross_routes]
    expected = [args.num_tokens * args.num_topk] + [0] * (
        dist.get_world_size() - 1
    )
    if actual != expected:
        raise AssertionError(
            f"{GIN_ACTIVITY_ASYMMETRIC_MODE} must have exactly one active "
            f"cross-LSA source; actual={actual}, expected={expected}"
        )
    return actual


def _finalize_matched_route_evidence(
    records: Sequence[Dict[str, Any]],
    num_tokens: int,
    num_topk: int,
    experts_per_rank: int,
    hostnames: Sequence[str],
) -> Dict[str, Any]:
    """Validate collectively gathered records from the actual route tensors.

    Every source reports the full owner/expert histogram encoded in its launch
    tensor.  Directly aggregating those reports proves the 0/50/100 controls
    have identical owner, expert, and GEMM block occupancy rather than merely
    having the intended route-mode labels.  ``expected_local_stats`` is also
    cross-checked as consistency evidence, not treated as an independent input.
    """
    world_size = len(hostnames)
    if world_size not in (8, 16) or len(records) != world_size:
        raise AssertionError(
            "matched route evidence requires one record from every EP8/EP16 rank"
        )
    lsa_size = world_size // 2
    if (hostnames[0] == hostnames[lsa_size]
            or any(value != hostnames[0] for value in hostnames[:lsa_size])
            or any(value != hostnames[lsa_size] for value in hostnames[lsa_size:])):
        raise AssertionError("matched route evidence requires two equal contiguous route domains")
    runtime_block_ms = [int(record["runtime_block_m"]) for record in records]
    if len(set(runtime_block_ms)) != 1:
        raise AssertionError(
            f"runtime BLOCK_M differs across ranks: {runtime_block_ms}"
        )
    block_m = runtime_block_ms[0]
    if (
        num_tokens not in MATCHED_BENCHMARK_TOKEN_COUNTS
        or num_topk != 16
        or experts_per_rank != 56
        or block_m != MATCHED_BENCHMARK_BLOCK_M
    ):
        raise AssertionError(
            "matched route evidence requires T32/40/48, top-k 16, "
            "56 experts/rank, and BLOCK_M=32"
        )

    routes_per_source = num_tokens * num_topk
    expected_cross = {
        "all_same_host": 0,
        "half_remote": routes_per_source // 2,
        "all_remote": routes_per_source,
    }
    expected_fanout = {
        "all_same_host": (lsa_size - 1, 0),
        "half_remote": (lsa_size - 1, lsa_size - 1),
        "all_remote": (0, lsa_size - 1),
    }
    ordered_records = sorted(records, key=lambda record: int(record["rank"]))
    if [int(record["rank"]) for record in ordered_records] != list(range(world_size)):
        raise AssertionError("matched route evidence has missing/duplicate ranks")

    route_evidence: Dict[str, Any] = {}
    received_by_mode: Dict[str, List[List[int]]] = {}
    blocks_by_mode: Dict[str, List[List[int]]] = {}
    for mode in MATCHED_BENCHMARK_ROUTES:
        received = [[0] * experts_per_rank for _ in range(world_size)]
        per_source_valid: List[int] = []
        per_source_cross: List[int] = []
        per_source_same_host_fanout: List[int] = []
        per_source_cross_host_fanout: List[int] = []
        for record in ordered_records:
            source = int(record["rank"])
            route_record = record["routes"][mode]
            sent = route_record["sent_owner_expert_histograms"]
            if len(sent) != world_size or any(
                len(owner_histogram) != experts_per_rank
                for owner_histogram in sent
            ):
                raise AssertionError(
                    f"{mode} rank {source} has malformed owner/expert histogram"
                )

            valid_routes = sum(
                int(count)
                for owner_histogram in sent
                for count in owner_histogram
            )
            cross_routes = sum(
                int(count)
                for owner, owner_histogram in enumerate(sent)
                if hostnames[owner] != hostnames[source]
                for count in owner_histogram
            )
            same_host_fanout = sum(
                any(int(count) != 0 for count in owner_histogram)
                and hostnames[owner] == hostnames[source]
                for owner, owner_histogram in enumerate(sent)
            )
            cross_host_fanout = sum(
                any(int(count) != 0 for count in owner_histogram)
                and hostnames[owner] != hostnames[source]
                for owner, owner_histogram in enumerate(sent)
            )
            if valid_routes != routes_per_source:
                raise AssertionError(
                    f"{mode} rank {source} has {valid_routes} valid routes, "
                    f"expected {routes_per_source}"
                )
            if int(route_record["valid_routes"]) != valid_routes:
                raise AssertionError(
                    f"{mode} rank {source} valid-route record disagrees with tensor"
                )
            if (
                cross_routes != expected_cross[mode]
                or int(route_record["cross_host_routes"]) != cross_routes
            ):
                raise AssertionError(
                    f"{mode} rank {source} has {cross_routes} cross-host routes, "
                    f"expected {expected_cross[mode]}"
                )
            if (same_host_fanout, cross_host_fanout) != expected_fanout[mode]:
                raise AssertionError(
                    f"{mode} rank {source} has same/cross-host fanout "
                    f"{same_host_fanout}/{cross_host_fanout}, "
                    f"expected {expected_fanout[mode]}"
                )

            per_source_valid.append(valid_routes)
            per_source_cross.append(cross_routes)
            per_source_same_host_fanout.append(same_host_fanout)
            per_source_cross_host_fanout.append(cross_host_fanout)
            for owner, owner_histogram in enumerate(sent):
                for expert, count in enumerate(owner_histogram):
                    received[owner][expert] += int(count)

        for owner, record in enumerate(ordered_records):
            collective_histogram = [
                int(value)
                for value in record["routes"][mode][
                    "expected_received_expert_histogram"
                ]
            ]
            if collective_histogram != received[owner]:
                raise AssertionError(
                    f"{mode} owner {owner} route aggregation disagrees with "
                    "expected_local_stats"
                )

        block_histograms = [
            [(count + block_m - 1) // block_m for count in histogram]
            for histogram in received
        ]
        received_by_mode[mode] = received
        blocks_by_mode[mode] = block_histograms
        route_evidence[mode] = {
            "cross_host_fraction": expected_cross[mode] / routes_per_source,
            "per_source_valid_routes": per_source_valid,
            "per_source_cross_host_routes": per_source_cross,
            "per_source_same_host_owner_fanout": (
                per_source_same_host_fanout
            ),
            "per_source_cross_host_owner_fanout": (
                per_source_cross_host_fanout
            ),
            "per_owner_received_routes": [sum(histogram) for histogram in received],
            "per_owner_received_expert_histograms": received,
            "per_owner_m_block_histograms": block_histograms,
            "per_owner_m_blocks": [sum(histogram) for histogram in block_histograms],
        }

    baseline = received_by_mode["all_same_host"]
    baseline_blocks = blocks_by_mode["all_same_host"]
    for mode in MATCHED_BENCHMARK_ROUTES[1:]:
        if received_by_mode[mode] != baseline:
            raise AssertionError(
                f"{mode} does not match all_same_host owner/expert occupancy"
            )
        if blocks_by_mode[mode] != baseline_blocks:
            raise AssertionError(
                f"{mode} does not match all_same_host GEMM block occupancy"
            )

    expected_total = routes_per_source * world_size
    if any(sum(histogram) != routes_per_source for histogram in baseline):
        raise AssertionError("matched routes do not balance total rows per owner")
    if sum(sum(histogram) for histogram in baseline) != expected_total:
        raise AssertionError("matched routes do not preserve the global route total")

    return {
        "enabled": True,
        "source": "collectively_gathered_actual_topk_idx_tensors",
        "tokens_per_rank": num_tokens,
        "num_topk": num_topk,
        "experts_per_rank": experts_per_rank,
        "block_m": block_m,
        "block_m_source": "deep_gemm.get_block_m_for_mega_moe",
        "routes_per_source": routes_per_source,
        "global_routes": expected_total,
        "verified_cross_host_fractions": [0.0, 0.5, 1.0],
        "owner_occupancy_matched": True,
        "expert_occupancy_matched": True,
        "m_block_occupancy_matched": True,
        "expected_local_stats_consistent": True,
        "routes": route_evidence,
    }


def _collect_matched_route_evidence(
    cases: Dict[str, RouteCase],
    rank: int,
    args: argparse.Namespace,
    experts_per_rank: int,
    hostnames: Sequence[str],
    block_m: int,
    dist: Any,
) -> Dict[str, Any]:
    """Gather and validate the actual T32/T40/T48 matched benchmark routes."""
    local_record: Dict[str, Any] = {
        "rank": rank,
        "runtime_block_m": block_m,
        "routes": {},
    }
    for mode in MATCHED_BENCHMARK_ROUTES:
        case = cases[mode]
        indices = [
            int(value)
            for value in case.topk_idx.reshape(-1).cpu().tolist()
        ]
        sent = [
            [0] * experts_per_rank for _ in range(dist.get_world_size())
        ]
        valid_routes = 0
        cross_host_routes = 0
        for expert_idx in indices:
            if expert_idx < 0:
                continue
            owner, local_expert = divmod(expert_idx, experts_per_rank)
            sent[owner][local_expert] += 1
            valid_routes += 1
            cross_host_routes += hostnames[owner] != hostnames[rank]
        local_record["routes"][mode] = {
            "valid_routes": valid_routes,
            "cross_host_routes": cross_host_routes,
            "sent_owner_expert_histograms": sent,
            "expected_received_expert_histogram": [
                int(value) for value in case.expected_local_stats.cpu().tolist()
            ],
        }

    gathered: List[Optional[Dict[str, Any]]] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_record)
    if any(record is None for record in gathered):
        raise AssertionError("failed to gather matched route evidence")
    return _finalize_matched_route_evidence(
        [record for record in gathered if record is not None],
        args.num_tokens,
        args.num_topk,
        experts_per_rank,
        hostnames,
    )


def _input_values(
    rank: int,
    num_tokens: int,
    hidden: int,
    torch: Any,
    input_epoch: int = 0,
    heterogeneous_input_scales: bool = False,
) -> Any:
    token = torch.arange(num_tokens, device="cuda", dtype=torch.int64).view(-1, 1)
    column = torch.arange(hidden, device="cuda", dtype=torch.int64).view(1, -1)
    code = (rank * 131 + token * 17 + column * 7 + input_epoch * 11) % 30
    magnitude = (code % 15 + 1).to(torch.float32) / 16.0
    sign = torch.where((code & 1) == 0, 1.0, -1.0)
    if heterogeneous_input_scales:
        # Exact BF16 powers of two vary every 32-column SF group, source,
        # token, and replay epoch. Constant scale bytes would hide a corrupted
        # packed-SF transfer or the owner-side SF transpose.
        exponent = _input_scale_exponent(rank, token, column // 32, input_epoch)
        magnitude = torch.ldexp(magnitude, exponent.to(torch.int32))
    return (magnitude * sign).to(torch.bfloat16)


def _input_scale_exponent(rank: Any, token: Any, group: Any, epoch: Any) -> Any:
    """Integer-only expression shared by the CUDA generator and host tests."""
    return (rank * 3 + token * 5 + group * 2 + epoch) % 7 - 3


def _make_inputs(
    rank: int,
    args: argparse.Namespace,
    torch: Any,
    deep_gemm: Any,
    input_epoch: int = 0,
) -> KernelInputs:
    from deep_gemm.utils import per_token_cast_to_fp8
    from deep_gemm.utils.math import cast_back_from_fp8

    source = _input_values(
        rank, args.num_tokens, args.hidden, torch, input_epoch=input_epoch,
        heterogeneous_input_scales=getattr(args, "heterogeneous_input_scales", False),
    )
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

    def launch(self, output: Optional[Any] = None) -> None:
        kwargs = {
            "y": self.output if output is None else output,
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
    snapshot_modes = ROUTE_MODES + (
        (GIN_ACTIVITY_ASYMMETRIC_MODE,)
        if GIN_ACTIVITY_ASYMMETRIC_MODE in harness.cases
        else ()
    )
    for name in snapshot_modes:
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
    for name in (
        "all_same_host",
        "half_remote",
        "all_peer",
        "all_remote",
        "different_peers",
        *(
            (GIN_ACTIVITY_ASYMMETRIC_MODE,)
            if GIN_ACTIVITY_ASYMMETRIC_MODE in snapshots
            else ()
        ),
    ):
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

    if harness.args.gin_active_fast_path:
        sequence = (
            GIN_ACTIVITY_TRANSITION_SEQUENCE * GIN_ACTIVITY_TRANSITION_CYCLES
        )
        for step, name in enumerate(sequence):
            case = harness.cases[name]
            output = harness.eager(case)
            expected.add_(case.expected_local_stats)
            label = f"eager/activity-transition/{step}/{name}"
            harness.assert_stats(expected, label)
            harness.assert_guards(label)
            _assert_bitwise_equal(output, snapshots[name], label, torch)


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

    # Cover every established mode under the captured graph even when a quick
    # benchmark requests only one additional stress replay.
    schedule = list(ROUTE_MODES)
    schedule.extend(
        ROUTE_MODES[(epoch + 1) % len(ROUTE_MODES)]
        for epoch in range(harness.args.graph_replays)
    )
    if harness.args.gin_active_fast_path:
        schedule.extend(
            GIN_ACTIVITY_TRANSITION_SEQUENCE * GIN_ACTIVITY_TRANSITION_CYCLES
        )

    for epoch, name in enumerate(schedule):
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


def _payload_epoch_sequence(num_epochs: int) -> List[int]:
    if num_epochs < 1:
        raise ValueError("payload epoch count must be positive")
    # Returning to epoch zero catches retained per-launch state as well as
    # stale payloads. The original input allocation is never replaced.
    return list(range(num_epochs)) + ([0] if num_epochs > 1 else [])


def _copy_input_storage(destination: KernelInputs, source: KernelInputs) -> None:
    destination.x.copy_(source.x)
    if destination.x_sf is not None:
        if source.x_sf is None:
            raise AssertionError("payload replay changed input scale format")
        destination.x_sf.copy_(source.x_sf)
    elif source.x_sf is not None:
        raise AssertionError("payload replay changed input scale format")
    destination.oracle_x.copy_(source.oracle_x)


def _input_storage_pointers(inputs: KernelInputs, buffer: Any) -> Dict[str, int]:
    pointers = {
        "captured_x": int(inputs.x.data_ptr()),
        "oracle_x": int(inputs.oracle_x.data_ptr()),
        "symmetric_x": int(buffer.x.data_ptr()),
        "symmetric_topk_idx": int(buffer.topk_idx.data_ptr()),
        "symmetric_topk_weights": int(buffer.topk_weights.data_ptr()),
    }
    if inputs.x_sf is not None:
        pointers["captured_x_sf"] = int(inputs.x_sf.data_ptr())
        pointers["symmetric_x_sf"] = int(buffer.x_sf.data_ptr())
    return pointers


def _run_payload_epoch_validation(
    harness: AccuracyHarness, torch: Any, dist: Any,
) -> Dict[str, Any]:
    """Replay changing payload/SF bytes through one captured graph/storage."""
    num_epochs = getattr(harness.args, "payload_epochs", 1)
    if num_epochs == 1:
        return {"enabled": False, "epochs": 1}

    sequence = _payload_epoch_sequence(num_epochs)
    original_pointers = _input_storage_pointers(harness.inputs, harness.buffer)
    modes = list(ROUTE_MODES)
    if GIN_ACTIVITY_ASYMMETRIC_MODE in harness.cases:
        modes.append(GIN_ACTIVITY_ASYMMETRIC_MODE)
    graph, static_idx, static_weights = _capture_graph(harness, torch, dist)
    records = []
    previous_x = harness.inputs.x.clone()
    previous_sf = (
        harness.inputs.x_sf.clone() if harness.inputs.x_sf is not None else None
    )
    try:
        for step, epoch in enumerate(sequence):
            fresh = _make_inputs(
                harness.rank, harness.args, torch, harness.deep_gemm,
                input_epoch=epoch,
            )
            x_changed = not torch.equal(fresh.x.view(torch.uint8), previous_x.view(torch.uint8))
            sf_changed = (
                not torch.equal(fresh.x_sf, previous_sf)
                if fresh.x_sf is not None else False
            )
            if step and not x_changed:
                raise AssertionError("payload epochs did not change FP8/BF16 input bytes")
            heterogeneous = getattr(harness.args, "heterogeneous_input_scales", False)
            unique_sf_bytes = (
                int(torch.unique(fresh.x_sf.contiguous().view(torch.uint8)).numel())
                if fresh.x_sf is not None else 0
            )
            if heterogeneous and fresh.x_sf is not None:
                if unique_sf_bytes < 7:
                    raise AssertionError("heterogeneous input did not exercise seven SF exponents")
                if step and not sf_changed:
                    raise AssertionError("payload epochs did not change packed SF bytes")
            _copy_input_storage(harness.inputs, fresh)
            if _input_storage_pointers(harness.inputs, harness.buffer) != original_pointers:
                raise AssertionError("payload replay replaced captured or symmetric input storage")

            # An independently recalculated quantization-aware oracle and an
            # all-local transport baseline are required for every payload.
            snapshots, oracle_metrics = _snapshot_and_check_oracles(harness, torch, dist)
            harness.stats.zero_()
            expected = torch.zeros_like(harness.stats)
            for name in modes:
                case = harness.cases[name]
                static_idx.copy_(case.topk_idx)
                static_weights.copy_(case.topk_weights)
                graph.replay()
                torch.cuda.synchronize()
                label = f"payload-epoch/{step}/{epoch}/{name}"
                expected.add_(case.expected_local_stats)
                harness.assert_stats(expected, label)
                harness.assert_guards(label)
                _assert_bitwise_equal(harness.output, snapshots[name], label, torch)
            records.append({
                "epoch": epoch,
                "input_bytes_changed": x_changed,
                "packed_sf_bytes_changed": sf_changed,
                "unique_packed_sf_exponents": unique_sf_bytes,
                "oracle_metrics": oracle_metrics,
            })
            previous_x.copy_(fresh.x)
            if previous_sf is not None:
                previous_sf.copy_(fresh.x_sf)
    finally:
        # The captured graph owns pointers into this allocation; retire it
        # before the caller performs the 48->64->48 lifetime test or teardown.
        graph = None
    dist.barrier()
    return {
        "enabled": True,
        "epochs": num_epochs,
        "sequence": sequence,
        "modes": modes,
        "single_graph": True,
        "same_input_storage": True,
        "same_symmetric_storage": True,
        "sparse_quantization_aware_oracle_per_epoch": True,
        "rank_0_payload_records": records,
    }


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


def _balanced_six_launch_schedule(samples_per_route: int) -> List[str]:
    """Return an exact-count triad schedule with balanced directed edges."""
    if samples_per_route <= 0 or samples_per_route % 2 != 0:
        raise ValueError(
            "balanced six-launch schedule requires a positive even sample "
            "count per route"
        )
    return list(BALANCED_BENCHMARK_PERIOD) * (samples_per_route // 2)


def _directed_transition_counts(schedule: Sequence[str]) -> Dict[str, int]:
    """Count measured predecessors, including the identically conditioned edge."""
    if not schedule:
        raise ValueError("directed transition accounting requires a schedule")
    counts: Dict[str, int] = {}
    previous = schedule[-1]
    for current in schedule:
        key = f"{previous}->{current}"
        counts[key] = counts.get(key, 0) + 1
        previous = current
    return counts


def _timing_halves(values: Sequence[float]) -> Dict[str, Dict[str, float]]:
    if len(values) < 2:
        raise ValueError("first/second-half timing requires at least two samples")
    midpoint = len(values) // 2
    return {
        "first": _timing_summary(values[:midpoint]),
        "second": _timing_summary(values[midpoint:]),
    }


def _six_launch_block_means(values: Sequence[float]) -> List[float]:
    """Collapse the two occurrences of one route in each balanced period."""
    if not values or len(values) % 2 != 0:
        raise ValueError("six-launch block pairing requires an even sample count")
    return [
        (float(values[index]) + float(values[index + 1])) / 2.0
        for index in range(0, len(values), 2)
    ]


def _distributed_timing_record(
    local_us_values: Sequence[float],
    torch: Any,
    dist: Any,
) -> Dict[str, Any]:
    """Retain auditable raw rank samples and slowest-rank replay samples."""
    local_us = torch.tensor(
        list(local_us_values), device="cuda", dtype=torch.float64
    )
    max_rank_us = local_us.clone()
    dist.all_reduce(max_rank_us, op=dist.ReduceOp.MAX)
    gathered = [
        torch.empty_like(local_us) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(gathered, local_us)

    local_rank_samples = [
        [float(value) for value in rank_values.cpu().tolist()]
        for rank_values in gathered
    ]
    max_rank_samples = [
        float(value) for value in max_rank_us.cpu().tolist()
    ]
    local_summaries = [
        _timing_summary(rank_values) for rank_values in local_rank_samples
    ]
    local_halves = [
        _timing_halves(rank_values) for rank_values in local_rank_samples
    ]
    return {
        "sample_count_per_route": len(max_rank_samples),
        "local_rank_sample_order": "outer_index_is_world_rank",
        "local_rank_samples_us": local_rank_samples,
        "max_rank_samples_us": max_rank_samples,
        "max_rank_per_replay_us": _timing_summary(max_rank_samples),
        "max_rank_per_replay_halves_us": _timing_halves(max_rank_samples),
        "local_rank_p50_us": _timing_summary(
            [summary["p50"] for summary in local_summaries]
        ),
        "local_rank_mean_us": _timing_summary(
            [summary["mean"] for summary in local_summaries]
        ),
        "local_rank_first_half_p50_us": _timing_summary(
            [halves["first"]["p50"] for halves in local_halves]
        ),
        "local_rank_second_half_p50_us": _timing_summary(
            [halves["second"]["p50"] for halves in local_halves]
        ),
    }


def _capture_launch_only_graph(
    harness: AccuracyHarness,
    torch: Any,
    dist: Any,
    route_name: str = "all_local",
) -> Any:
    """Capture only MegaMoE, leaving route/input preparation outside timing."""
    harness.copy_inputs(harness.cases[route_name])
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


def _collective_transition_check(
    label: str,
    local_errors: Sequence[str],
    dist: Any,
) -> None:
    local_error = "; ".join(local_errors) if local_errors else None
    gathered: List[Optional[str]] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_error)
    failures = [
        f"rank {rank}: {error}"
        for rank, error in enumerate(gathered)
        if error is not None
    ]
    if failures:
        raise AssertionError(
            f"GIN fast-path transition {label} failed collectively: "
            + "; ".join(failures)
        )


def _prepare_bulk_transition_phase(
    harness: AccuracyHarness,
    phase: GinBulkTransitionPhase,
    output_core: Any,
    torch: Any,
    dist: Any,
) -> Dict[str, Any]:
    """Arm phase-specific public-buffer canaries and copy active inputs."""
    count = phase.args.num_tokens
    output_core.fill_(float("nan"))

    tail_expected: Dict[str, Any] = {}
    harness.buffer.x[count:].fill_(-3.25)
    tail_expected["x"] = harness.buffer.x[count:].clone()
    if phase.inputs.x_sf is not None:
        harness.buffer.x_sf[count:].fill_(0x53535353)
        tail_expected["x_sf"] = harness.buffer.x_sf[count:].clone()
    harness.buffer.topk_idx[count:].fill_(-777)
    harness.buffer.topk_weights[count:].fill_(-17.0)
    tail_expected["topk_idx"] = harness.buffer.topk_idx[count:].clone()
    tail_expected["topk_weights"] = (
        harness.buffer.topk_weights[count:].clone()
    )

    harness.buffer.x[:count].copy_(phase.inputs.x)
    if phase.inputs.x_sf is not None:
        harness.buffer.x_sf[:count].copy_(phase.inputs.x_sf)
    harness.buffer.topk_idx[:count].copy_(phase.case.topk_idx)
    harness.buffer.topk_weights[:count].copy_(phase.case.topk_weights)
    torch.cuda.synchronize()
    dist.barrier()
    return tail_expected


def _validate_bulk_transition_phase(
    harness: AccuracyHarness,
    phase: GinBulkTransitionPhase,
    output_core: Any,
    output_prefix: Any,
    output_prefix_expected: Any,
    output_suffix: Any,
    output_suffix_expected: Any,
    tail_expected: Dict[str, Any],
    expected_stats: Any,
    reference: Optional[Any],
    must_differ_from: Optional[Any],
    label: str,
    torch: Any,
    dist: Any,
) -> Dict[str, float]:
    """Validate locally, then make every rank take one shared failure path."""
    count = phase.args.num_tokens
    output = output_core[:count]
    local_errors: List[str] = []

    try:
        relative, max_abs = _symmetric_diff(output, phase.oracle)
    except BaseException as exc:
        relative, max_abs = math.inf, math.inf
        local_errors.append(
            f"oracle calculation raised {type(exc).__name__}: {exc}"
        )
    global_metrics = torch.tensor(
        [relative, max_abs], device="cuda", dtype=torch.float64
    )
    dist.all_reduce(global_metrics, op=dist.ReduceOp.MAX)
    worst_relative, worst_max_abs = [
        float(value) for value in global_metrics.cpu().tolist()
    ]
    rel_limit, abs_limit = _oracle_thresholds(phase.args)
    if worst_relative > rel_limit or worst_max_abs > abs_limit:
        local_errors.append(
            "analytic oracle mismatch: "
            f"symmetric_diff={worst_relative:.8g}/{rel_limit:.8g}, "
            f"max_abs={worst_max_abs:.8g}/{abs_limit:.8g}"
        )

    if not torch.equal(output_prefix, output_prefix_expected):
        local_errors.append("output prefix canary was overwritten")
    if not torch.equal(output_suffix, output_suffix_expected):
        local_errors.append("output suffix canary was overwritten")
    if not bool(torch.isfinite(output).all().item()):
        local_errors.append("active output contains unwritten/non-finite values")
    if count < output_core.shape[0] and not bool(
        torch.isnan(output_core[count:]).all().item()
    ):
        local_errors.append(
            f"inactive output rows {count}:{output_core.shape[0]} were overwritten"
        )

    current_tails = {
        "x": harness.buffer.x[count:],
        "topk_idx": harness.buffer.topk_idx[count:],
        "topk_weights": harness.buffer.topk_weights[count:],
    }
    if phase.inputs.x_sf is not None:
        current_tails["x_sf"] = harness.buffer.x_sf[count:]
    for name, value in current_tails.items():
        if not torch.equal(value, tail_expected[name]):
            local_errors.append(f"inactive input tail {name} was overwritten")

    if not torch.equal(harness.stats, expected_stats):
        local_errors.append(
            "cumulative stats mismatch: "
            f"actual={harness.stats.cpu().tolist()} "
            f"expected={expected_stats.cpu().tolist()}"
        )
    if reference is not None and not torch.equal(output, reference):
        local_errors.append("output is not bitwise-equal to its eager snapshot")
    if must_differ_from is not None and torch.equal(output, must_differ_from):
        local_errors.append(
            "distinct input epoch reproduced the stale prior output bitwise"
        )

    _collective_transition_check(label, local_errors, dist)
    return {
        "worst_symmetric_diff": worst_relative,
        "worst_max_abs": worst_max_abs,
    }


def _run_gin_bulk_transition_validation(
    harness: AccuracyHarness,
    rank: int,
    args: argparse.Namespace,
    hostnames: Sequence[str],
    graph_holder: Dict[int, Any],
    torch: Any,
    dist: Any,
    deep_gemm: Any,
) -> Dict[str, Any]:
    """Exercise specialized -> exact-r75 -> specialized on one context."""
    enabled_features = {
        "bulk_combine": bool(args.gin_bulk_combine),
        "direct_dispatch": bool(args.gin_direct_dispatch),
    }
    if not _gin_fast_path_transition_requested(args):
        return {"enabled": False, "features": enabled_features}
    if harness.buffer.x.shape[0] <= GIN_FALLBACK_PROBE_TOKENS:
        raise AssertionError(
            "GIN fast-path transition requires at least one inactive row above 64"
        )

    experts_per_rank = args.num_experts // dist.get_world_size()
    phase_args: Dict[int, argparse.Namespace] = {}
    phase_cases: Dict[int, RouteCase] = {}
    for count in (GIN_FAST_PATH_MAX_TOKENS, GIN_FALLBACK_PROBE_TOKENS):
        current_args = argparse.Namespace(**vars(args))
        current_args.num_tokens = count
        phase_args[count] = current_args
        topk_idx, topk_weights = _build_bulk_transition_route_tensors(
            rank,
            count,
            args.num_topk,
            experts_per_rank,
            hostnames,
            torch,
        )
        phase_cases[count] = RouteCase(
            f"paired_remote_capacity_{count}",
            topk_idx,
            topk_weights,
            _global_route_stats(
                topk_idx, rank, experts_per_rank, torch, dist
            ),
            "paired_cross_host_capacity_probe",
        )

    phases: Dict[str, GinBulkTransitionPhase] = {}
    for name in GIN_BULK_TRANSITION_SEQUENCE:
        count = GIN_BULK_TRANSITION_COUNTS[name]
        inputs = _make_inputs(
            rank,
            phase_args[count],
            torch,
            deep_gemm,
            input_epoch=GIN_BULK_TRANSITION_INPUT_EPOCHS[name],
        )
        phases[name] = GinBulkTransitionPhase(
            name,
            phase_args[count],
            inputs,
            phase_cases[count],
            _analytic_oracle(
                inputs.oracle_x,
                phase_cases[count],
                phase_args[count],
                experts_per_rank,
                torch,
            ),
        )

    _collective_transition_check(
        "input-epoch-preflight",
        (
            ["48a and 48b analytic outputs are unexpectedly bitwise equal"]
            if torch.equal(phases["48a"].oracle, phases["48b"].oracle)
            else []
        ),
        dist,
    )

    initial_buffer = harness.buffer
    initial_tensor_ptr = int(initial_buffer.buffer.data_ptr())
    initial_context = initial_buffer.gin_context
    initial_descriptor = initial_context.launch_descriptor_snapshot()

    guard_elements = max(256, args.hidden)
    output_storage = torch.full(
        (
            guard_elements
            + GIN_FALLBACK_PROBE_TOKENS * args.hidden
            + guard_elements,
        ),
        -91.5,
        device="cuda",
        dtype=torch.bfloat16,
    )
    output_core_end = (
        guard_elements + GIN_FALLBACK_PROBE_TOKENS * args.hidden
    )
    output_core = output_storage[guard_elements:output_core_end].view(
        GIN_FALLBACK_PROBE_TOKENS, args.hidden
    )
    output_prefix = output_storage[:guard_elements]
    output_suffix = output_storage[-guard_elements:]
    output_prefix_expected = output_prefix.clone()
    output_suffix_expected = output_suffix.clone()

    eager_snapshots: Dict[str, Any] = {}
    eager_metrics: Dict[str, Dict[str, float]] = {}
    harness.stats.zero_()
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()
    eager_expected_stats = torch.zeros_like(harness.stats)
    for step, name in enumerate(GIN_BULK_TRANSITION_SEQUENCE):
        phase = phases[name]
        tail_expected = _prepare_bulk_transition_phase(
            harness, phase, output_core, torch, dist
        )
        harness.launch(output_core[: phase.args.num_tokens])
        torch.cuda.synchronize()
        eager_expected_stats.add_(phase.case.expected_local_stats)
        eager_metrics[name] = _validate_bulk_transition_phase(
            harness,
            phase,
            output_core,
            output_prefix,
            output_prefix_expected,
            output_suffix,
            output_suffix_expected,
            tail_expected,
            eager_expected_stats,
            reference=None,
            must_differ_from=(
                eager_snapshots.get("48a") if name == "48b" else None
            ),
            label=f"eager/{step}/{name}",
            torch=torch,
            dist=dist,
        )
        eager_snapshots[name] = output_core[: phase.args.num_tokens].clone()

    # Warm and capture one launch-only graph per token-count specialization.
    # The graph is stored directly in caller-owned state so failure teardown
    # can retire it even while this frame remains reachable from a traceback.
    for name in ("48a", "64"):
        phase = phases[name]
        count = phase.args.num_tokens
        harness.stats.zero_()
        tail_expected = _prepare_bulk_transition_phase(
            harness, phase, output_core, torch, dist
        )
        for _ in range(3):
            harness.launch(output_core[:count])
        torch.cuda.synchronize()

        harness.stats.zero_()
        tail_expected = _prepare_bulk_transition_phase(
            harness, phase, output_core, torch, dist
        )
        # `torch.cuda.graph` may capture on a stream distinct from the one
        # used by preparation. Complete the stats reset, input copies, and
        # NaN/canary stores before any captured kernel can race them, then
        # align every rank at the same generation boundary.
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        graph_holder[count] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_holder[count]):
            harness.launch(output_core[:count])
        # Capture records the launch but does not execute it on this runtime.
        # Replay once so the following check proves the captured path rather
        # than inspecting the pre-capture NaN/reset state.
        graph_holder[count].replay()
        torch.cuda.synchronize()
        _validate_bulk_transition_phase(
            harness,
            phase,
            output_core,
            output_prefix,
            output_prefix_expected,
            output_suffix,
            output_suffix_expected,
            tail_expected,
            phase.case.expected_local_stats,
            reference=eager_snapshots[name],
            must_differ_from=None,
            label=f"capture/{name}",
            torch=torch,
            dist=dist,
        )
        harness.stats.zero_()
        torch.cuda.synchronize()
        dist.barrier()

    graph_metrics: Dict[str, Dict[str, float]] = {}
    graph_expected_stats = torch.zeros_like(harness.stats)
    for step, name in enumerate(GIN_BULK_TRANSITION_SEQUENCE):
        phase = phases[name]
        count = phase.args.num_tokens
        tail_expected = _prepare_bulk_transition_phase(
            harness, phase, output_core, torch, dist
        )
        graph_holder[count].replay()
        torch.cuda.synchronize()
        graph_expected_stats.add_(phase.case.expected_local_stats)
        graph_metrics[name] = _validate_bulk_transition_phase(
            harness,
            phase,
            output_core,
            output_prefix,
            output_prefix_expected,
            output_suffix,
            output_suffix_expected,
            tail_expected,
            graph_expected_stats,
            reference=eager_snapshots[name],
            must_differ_from=(
                eager_snapshots["48a"] if name == "48b" else None
            ),
            label=f"graph/{step}/{name}",
            torch=torch,
            dist=dist,
        )

    identity_errors = []
    if harness.buffer is not initial_buffer:
        identity_errors.append("SymmBuffer object changed")
    if int(harness.buffer.buffer.data_ptr()) != initial_tensor_ptr:
        identity_errors.append("registered symmetric tensor pointer changed")
    if harness.buffer.gin_context is not initial_context:
        identity_errors.append("GIN context object changed")
    if initial_context.launch_descriptor_snapshot() != initial_descriptor:
        identity_errors.append("GIN launch descriptor changed")
    _collective_transition_check("context-identity", identity_errors, dist)

    harness.stats.zero_()
    torch.cuda.synchronize()
    graph_count = len(graph_holder)
    graph_holder.clear()
    dist.barrier()
    return {
        "enabled": True,
        "features": enabled_features,
        "route": "paired_remote_owner_capacity_probe",
        "eager_sequence": list(GIN_BULK_TRANSITION_SEQUENCE),
        "graph_sequence": list(GIN_BULK_TRANSITION_SEQUENCE),
        "token_counts": [
            GIN_BULK_TRANSITION_COUNTS[name]
            for name in GIN_BULK_TRANSITION_SEQUENCE
        ],
        "input_epochs": [
            GIN_BULK_TRANSITION_INPUT_EPOCHS[name]
            for name in GIN_BULK_TRANSITION_SEQUENCE
        ],
        "owner_source_records": {"48": 48 * 16, "64": 64 * 16},
        "bulk_capacity": GIN_BULK_COMBINE_CAPACITY,
        "fast_path_max_tokens": GIN_FAST_PATH_MAX_TOKENS,
        "fallback_probe_tokens": GIN_FALLBACK_PROBE_TOKENS,
        "fallback_protocol": "exact_r75",
        "phase_protocols": {
            name: (
                ["exact_r75"]
                if GIN_BULK_TRANSITION_COUNTS[name]
                > GIN_FAST_PATH_MAX_TOKENS
                else [
                    feature
                    for feature, enabled in enabled_features.items()
                    if enabled
                ]
            )
            for name in GIN_BULK_TRANSITION_SEQUENCE
        },
        "captured_graph_count": graph_count,
        "reused_48_graph": True,
        "same_buffer": True,
        "same_registered_pointer": True,
        "same_gin_context": True,
        "same_launch_descriptor": True,
        "eager_oracle_metrics": eager_metrics,
        "graph_oracle_metrics": graph_metrics,
    }


def _benchmark_graph_routes(
    harness: AccuracyHarness,
    graph: Any,
    snapshots: Dict[str, Any],
    torch: Any,
    dist: Any,
) -> Dict[str, Any]:
    """Time matched routes with balanced transitions and isolated controls.

    Route copies remain outside the CUDA-event interval.  The interleaved
    six-launch period gives each of the six directed transitions between the
    0/50/100 routes exactly the same frequency.  Each route is then re-warmed
    and timed alone to expose any instruction/cache cost caused by switching
    route behavior in the interleaved measurement.
    """
    route_names = list(MATCHED_BENCHMARK_ROUTES)
    count = harness.args.num_tokens
    conditioning_schedule = _balanced_six_launch_schedule(
        harness.args.benchmark_conditioning_replays
    )
    measurement_schedule = _balanced_six_launch_schedule(
        harness.args.benchmark_replays
    )

    def select_route(name: str) -> None:
        case = harness.cases[name]
        harness.buffer.topk_idx[:count].copy_(case.topk_idx)
        harness.buffer.topk_weights[:count].copy_(case.topk_weights)

    # Every candidate receives the exact same CLI-sized conditioning trace.
    harness.stats.zero_()
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()
    for name in conditioning_schedule:
        select_route(name)
        graph.replay()
    torch.cuda.synchronize()

    # Reset the only user-visible cumulative state so measured replays have an
    # exact postcondition independent of capture and warmup execution.
    harness.stats.zero_()
    harness.output.fill_(float("nan"))
    torch.cuda.synchronize()
    dist.barrier()
    # A CPU barrier does not imply completion of CUDA work enqueued by the
    # backend; make the beginning of the event train explicit on every rank.
    torch.cuda.synchronize()

    starts: Dict[str, List[Any]] = {name: [] for name in route_names}
    ends: Dict[str, List[Any]] = {name: [] for name in route_names}
    launch_indices: Dict[str, List[int]] = {name: [] for name in route_names}
    for launch_idx, name in enumerate(measurement_schedule):
        select_route(name)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        starts[name].append(start)
        ends[name].append(end)
        launch_indices[name].append(launch_idx)
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
    last_name = measurement_schedule[-1]
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
        if len(local_us_values) != harness.args.benchmark_replays:
            raise AssertionError(
                f"{name} has {len(local_us_values)} samples, expected "
                f"{harness.args.benchmark_replays}"
            )
        timing = _distributed_timing_record(local_us_values, torch, dist)
        max_rank_samples[name] = timing["max_rank_samples_us"]
        results[name] = {
            "owner_kind": case.owner_kind,
            "interleaved_launch_indices": launch_indices[name],
            **timing,
        }

    baseline = results.get("all_same_host")
    if baseline is not None:
        baseline_samples = max_rank_samples["all_same_host"]
        baseline_block_means = _six_launch_block_means(baseline_samples)
        baseline_p50 = baseline["max_rank_per_replay_us"]["p50"]
        for name, value in results.items():
            p50 = value["max_rank_per_replay_us"]["p50"]
            route_block_means = _six_launch_block_means(
                max_rank_samples[name]
            )
            paired_delta = [
                sample - reference
                for sample, reference in zip(
                    route_block_means, baseline_block_means
                )
            ]
            paired_summary = _timing_summary(paired_delta)
            value["six_launch_block_mean_samples_us"] = route_block_means
            value["paired_six_launch_block_delta_vs_same_host_us"] = (
                paired_summary
            )
            value["paired_six_launch_block_delta_samples_us"] = paired_delta
            value["p50_delta_vs_same_host_us"] = paired_summary["p50"]
            value["p50_ratio_vs_same_host"] = p50 / baseline_p50

    # A separately re-warmed route-isolated run distinguishes transition/cache
    # effects from the steady-state floor of each transport behavior.
    for name in route_names:
        case = harness.cases[name]
        harness.stats.zero_()
        harness.output.fill_(float("nan"))
        select_route(name)
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        for _ in range(harness.args.benchmark_conditioning_replays):
            graph.replay()
        torch.cuda.synchronize()

        harness.stats.zero_()
        harness.output.fill_(float("nan"))
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        isolated_starts: List[Any] = []
        isolated_ends: List[Any] = []
        for _ in range(harness.args.benchmark_replays):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            isolated_starts.append(start)
            isolated_ends.append(end)
            start.record()
            graph.replay()
            end.record()
        torch.cuda.synchronize()

        harness.assert_stats(
            case.expected_local_stats * harness.args.benchmark_replays,
            f"benchmark/isolated/{name}",
        )
        harness.assert_guards(f"benchmark/isolated/{name}")
        _assert_bitwise_equal(
            harness.output,
            snapshots[name],
            f"benchmark/isolated/{name}",
            torch,
        )
        isolated_local_us = [
            float(start.elapsed_time(end) * 1.0e3)
            for start, end in zip(isolated_starts, isolated_ends)
        ]
        results[name]["isolated"] = _distributed_timing_record(
            isolated_local_us, torch, dist
        )

    isolated_baseline_p50 = results["all_same_host"]["isolated"][
        "max_rank_per_replay_us"
    ]["p50"]
    for name in route_names:
        isolated = results[name]["isolated"]
        isolated_p50 = isolated["max_rank_per_replay_us"]["p50"]
        isolated["p50_delta_vs_same_host_us"] = (
            isolated_p50 - isolated_baseline_p50
        )
        isolated["p50_ratio_vs_same_host"] = (
            isolated_p50 / isolated_baseline_p50
        )

    dist.barrier()
    return {
        "schedule": {
            "kind": "balanced_six_launch_directed_transitions",
            "period": list(BALANCED_BENCHMARK_PERIOD),
            "measurement_samples_per_route": harness.args.benchmark_replays,
            "measurement_period_repetitions": (
                harness.args.benchmark_replays // 2
            ),
            "measurement_total_launches": len(measurement_schedule),
            "measurement_directed_transition_counts": (
                _directed_transition_counts(measurement_schedule)
            ),
            "conditioning_samples_per_route": (
                harness.args.benchmark_conditioning_replays
            ),
            "conditioning_period_repetitions": (
                harness.args.benchmark_conditioning_replays // 2
            ),
            "conditioning_total_launches": len(conditioning_schedule),
            "conditioning_directed_transition_counts": (
                _directed_transition_counts(conditioning_schedule)
            ),
            "route_isolated_order": route_names,
            "route_isolated_warmups_per_route": (
                harness.args.benchmark_conditioning_replays
            ),
        },
        "routes": results,
    }


def _validate_local_ablation_owner_values(
    rank: int,
    expected_slots: int,
    owner_values: Sequence[int],
) -> None:
    if len(owner_values) != expected_slots:
        raise AssertionError(
            "GIN local ablation requires every top-k slot to be populated"
        )
    invalid = [
        owner
        for owner in owner_values
        if owner == rank or owner // 8 != rank // 8
    ]
    if invalid:
        raise AssertionError(
            "GIN local ablation requires non-self owners in the same contiguous "
            f"eight-rank LSA; invalid owners={sorted(set(invalid))}"
        )


def _validate_local_ablation_route(harness: AccuracyHarness) -> None:
    case = harness.cases["all_same_host"]
    owners = case.topk_idx.cpu() // harness.experts_per_rank
    valid_owners = owners[owners >= 0]
    _validate_local_ablation_owner_values(
        harness.rank,
        harness.args.num_tokens * harness.args.num_topk,
        [int(value) for value in valid_owners.tolist()],
    )


def _collectively_validate_local_ablation_setup(
    harness: AccuracyHarness,
    dist: Any,
) -> None:
    """Fail every rank together before entering stage-specific device barriers."""
    local_error = None
    try:
        _validate_local_ablation_route(harness)
    except (AssertionError, KeyError, ValueError) as exc:
        local_error = str(exc)

    local_record = {
        "rank": harness.rank,
        "stages": tuple(harness.args.gin_local_ablation_stages),
        "route_error": local_error,
    }
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_record)
    stage_orders = {tuple(record["stages"]) for record in gathered}
    route_errors = [
        f"rank {record['rank']}: {record['route_error']}"
        for record in gathered
        if record["route_error"] is not None
    ]
    if len(stage_orders) != 1 or route_errors:
        details = []
        if len(stage_orders) != 1:
            details.append(f"stage orders differ across ranks: {sorted(stage_orders)}")
        details.extend(route_errors)
        raise RuntimeError(
            "GIN local ablation collective preflight failed: " + "; ".join(details)
        )


def _capture_local_ablation_graphs(
    harness: AccuracyHarness,
    snapshots: Dict[str, Any],
    torch: Any,
    dist: Any,
) -> Dict[int, Any]:
    """Capture and validate one compile-time kernel per cumulative stage."""
    _collectively_validate_local_ablation_setup(harness, dist)
    case = harness.cases["all_same_host"]
    graphs: Dict[int, Any] = {}
    try:
        for stage in harness.args.gin_local_ablation_stages:
            os.environ[GIN_LOCAL_ABLATION_ENV] = str(stage)
            graph = _capture_launch_only_graph(
                harness, torch, dist, route_name="all_same_host"
            )

            harness.stats.zero_()
            harness.output.fill_(float("nan"))
            harness.copy_inputs(case)
            graph.replay()
            torch.cuda.synchronize()
            harness.assert_stats(
                case.expected_local_stats,
                f"local-ablation/capture-check/stage-{stage}",
            )
            harness.assert_guards(
                f"local-ablation/capture-check/stage-{stage}"
            )
            _assert_bitwise_equal(
                harness.output,
                snapshots["all_same_host"],
                f"local-ablation/capture-check/stage-{stage}",
                torch,
            )
            harness.stats.zero_()
            graphs[stage] = graph
            dist.barrier()
    finally:
        os.environ[GIN_LOCAL_ABLATION_ENV] = "0"
    return graphs


def _benchmark_local_ablation_graphs(
    harness: AccuracyHarness,
    graphs: Dict[int, Any],
    snapshots: Dict[str, Any],
    torch: Any,
    dist: Any,
) -> Dict[str, Any]:
    """Interleave identical all-same-host work across kernel ablations."""
    stages = list(harness.args.gin_local_ablation_stages)
    case = harness.cases["all_same_host"]
    harness.copy_inputs(case)

    def rotated(cycle: int) -> List[int]:
        offset = cycle % len(stages)
        return stages[offset:] + stages[:offset]

    harness.stats.zero_()
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()
    for cycle in range(harness.args.benchmark_warmups):
        for stage in rotated(cycle):
            graphs[stage].replay()
    torch.cuda.synchronize()

    harness.stats.zero_()
    harness.output.fill_(float("nan"))
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()

    starts: Dict[int, List[Any]] = {stage: [] for stage in stages}
    ends: Dict[int, List[Any]] = {stage: [] for stage in stages}
    for cycle in range(harness.args.benchmark_replays):
        for stage in rotated(cycle):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            starts[stage].append(start)
            ends[stage].append(end)
            start.record()
            graphs[stage].replay()
            end.record()
    torch.cuda.synchronize()

    expected_stats = case.expected_local_stats * (
        harness.args.benchmark_replays * len(stages)
    )
    harness.assert_stats(expected_stats, "local-ablation/measured-batch")
    harness.assert_guards("local-ablation/measured-batch")
    _assert_bitwise_equal(
        harness.output,
        snapshots["all_same_host"],
        "local-ablation/final",
        torch,
    )

    results: Dict[str, Any] = {}
    max_rank_samples: Dict[int, List[float]] = {}
    for stage in stages:
        local_us_values = [
            float(start.elapsed_time(end) * 1.0e3)
            for start, end in zip(starts[stage], ends[stage])
        ]
        timing = _distributed_timing_record(local_us_values, torch, dist)
        max_rank_samples[stage] = timing["max_rank_samples_us"]
        results[str(stage)] = {
            "description": GIN_LOCAL_ABLATION_STAGES[stage],
            **timing,
        }

    baseline_samples = max_rank_samples[0]
    baseline_p50 = results["0"]["max_rank_per_replay_us"]["p50"]
    for stage in stages:
        stage_result = results[str(stage)]
        paired_delta = [
            sample - baseline
            for sample, baseline in zip(
                max_rank_samples[stage], baseline_samples
            )
        ]
        stage_result["paired_delta_vs_stage_0_us"] = _timing_summary(
            paired_delta
        )
        stage_result["paired_delta_vs_stage_0_samples_us"] = paired_delta
        stage_result["p50_ratio_vs_stage_0"] = (
            stage_result["max_rank_per_replay_us"]["p50"] / baseline_p50
        )

    # Switching among five large CUfunctions can perturb the instruction cache.
    # Re-warm and measure each specialization alone as the absolute-floor
    # control; retain the rotating run above as the paired common-mode A/B.
    isolated_starts: Dict[int, List[Any]] = {stage: [] for stage in stages}
    isolated_ends: Dict[int, List[Any]] = {stage: [] for stage in stages}
    for stage in stages:
        harness.stats.zero_()
        harness.output.fill_(float("nan"))
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        for _ in range(harness.args.benchmark_warmups):
            graphs[stage].replay()
        torch.cuda.synchronize()

        harness.stats.zero_()
        harness.output.fill_(float("nan"))
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        for _ in range(harness.args.benchmark_replays):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            isolated_starts[stage].append(start)
            isolated_ends[stage].append(end)
            start.record()
            graphs[stage].replay()
            end.record()
        torch.cuda.synchronize()
        harness.assert_stats(
            case.expected_local_stats * harness.args.benchmark_replays,
            f"local-ablation/isolated/stage-{stage}",
        )
        harness.assert_guards(f"local-ablation/isolated/stage-{stage}")
        _assert_bitwise_equal(
            harness.output,
            snapshots["all_same_host"],
            f"local-ablation/isolated/stage-{stage}",
            torch,
        )

    for stage in stages:
        local_us_values = [
            float(start.elapsed_time(end) * 1.0e3)
            for start, end in zip(
                isolated_starts[stage], isolated_ends[stage]
            )
        ]
        isolated_timing = _distributed_timing_record(
            local_us_values, torch, dist
        )
        stage_result = results[str(stage)]
        stage_result["isolated"] = isolated_timing
        stage_result["isolated_max_rank_per_replay_us"] = isolated_timing[
            "max_rank_per_replay_us"
        ]
        stage_result["isolated_local_rank_p50_us"] = isolated_timing[
            "local_rank_p50_us"
        ]
        stage_result["isolated_local_rank_mean_us"] = isolated_timing[
            "local_rank_mean_us"
        ]

    isolated_baseline_p50 = results["0"][
        "isolated_max_rank_per_replay_us"
    ]["p50"]
    for stage in stages:
        stage_result = results[str(stage)]
        isolated_p50 = stage_result[
            "isolated_max_rank_per_replay_us"
        ]["p50"]
        stage_result["isolated_p50_delta_vs_stage_0_us"] = (
            isolated_p50 - isolated_baseline_p50
        )
        stage_result["isolated_p50_ratio_vs_stage_0"] = (
            isolated_p50 / isolated_baseline_p50
        )

    dist.barrier()
    return {
        "route": "all_same_host",
        "timed_cross_host_payload": False,
        "compile_time_specialization": True,
        "cumulative": True,
        "stage_0_protocol": "current_four_phase_gin_kernel_not_r70",
        "paired_schedule": "interleaved_rotating_paired_cycles",
        "absolute_floor_schedule": "stage_isolated_after_per_stage_warmup",
        "stages": results,
    }


def _validate_args(args: argparse.Namespace, world_size: int, *,
                   gb200_world_size: Optional[int] = None) -> None:
    # Only the explicit, collectively prepared GB200 entry supplies this.
    # Normal accuracy/Novita invocations retain the original EP16 GIN gate.
    if gb200_world_size is not None and (
        type(gb200_world_size) is not int or gb200_world_size not in (8, 16)
        or world_size != gb200_world_size
    ):
        raise ValueError("GB200 profile world size must match actual EP8 or EP16")
    gin_world_size = 16 if gb200_world_size is None else gb200_world_size
    gin_num_experts = gin_world_size * 56
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
    if args.benchmark_replays > 0:
        if args.benchmark_replays < 2:
            raise ValueError(
                "--benchmark-replays must be at least 2 for half-run summaries"
            )
        if not args.require_gin:
            raise ValueError("--benchmark-replays currently requires --require-gin")
        if args.benchmark_warmups < 1:
            raise ValueError("--benchmark-warmups must be positive")
        if (
            args.benchmark_conditioning_replays < 1
            or args.benchmark_conditioning_replays % 2 != 0
        ):
            raise ValueError(
                "--benchmark-conditioning-replays must be a positive even number"
            )
        if not args.gin_local_ablation_stages:
            if args.benchmark_replays % 2 != 0:
                raise ValueError(
                    "matched --benchmark-replays must be even for the "
                    "balanced six-launch schedule"
                )
            if tuple(args.benchmark_routes) != MATCHED_BENCHMARK_ROUTES:
                raise ValueError(
                    "matched benchmarking requires exactly --benchmark-routes "
                    + " ".join(MATCHED_BENCHMARK_ROUTES)
                )
            if (
                args.num_experts,
                args.num_topk,
                args.hidden,
                args.intermediate_hidden,
                args.num_shared_experts,
            ) != (gin_num_experts, 16, 3584, 3072, 0):
                raise ValueError(
                    f"matched benchmarking requires E{gin_num_experts}/topk16/H3584/I3072/"
                    "no-shared"
                )
            if args.num_tokens not in MATCHED_BENCHMARK_TOKEN_COUNTS:
                raise ValueError(
                    "matched benchmarking requires 32, 40, or 48 tokens per rank"
                )
    if args.require_gin:
        if args.mma_type != "fp8xfp4":
            raise ValueError(
                "--require-gin currently requires --mma-type=fp8xfp4; the BF16 "
                "MegaMoE launch has not been wired to the GIN transport"
            )
        if world_size != gin_world_size:
            raise ValueError(
                f"--require-gin currently requires exactly {gin_world_size} ranks "
                f"(2x{gin_world_size // 2}), got {world_size}"
            )
        if args.gin_completion_batch not in (1, 2, 4, 8):
            raise ValueError(
                "--gin-completion-batch must be one of 1, 2, 4, or 8"
            )
        if args.gin_combine_issue_wave not in (1, 2, 4, 8):
            raise ValueError(
                "--gin-combine-issue-wave must be one of 1, 2, 4, or 8"
            )
        if args.gin_outbox_depth not in (4, 8, 16, 64):
            raise ValueError(
                "--gin-outbox-depth must be one of 4, 8, 16, or 64"
            )
        if args.gin_queue_depth < 64:
            raise ValueError("--gin-queue-depth must be at least 64")
        row_bytes = args.hidden * 2
        if row_bytes % args.gin_combine_chunk_bytes != 0:
            raise ValueError(
                f"--gin-combine-chunk-bytes={args.gin_combine_chunk_bytes} must "
                f"divide one BF16 output row ({row_bytes} bytes)"
            )
    if args.gin_local_ablation_stages:
        if not args.require_gin:
            raise ValueError(
                "--gin-local-ablation-stages requires --require-gin"
            )
        if 0 not in args.gin_local_ablation_stages:
            raise ValueError(
                "--gin-local-ablation-stages must include stage 0 as the "
                "paired baseline"
            )
        if tuple(args.benchmark_routes) != ("all_same_host",):
            raise ValueError(
                "local ablations require exactly "
                "--benchmark-routes all_same_host"
            )
    if args.gin_active_fast_path and not args.require_gin:
        raise ValueError("--gin-active-fast-path requires --require-gin")
    if args.gin_active_fast_path and args.gin_local_ablation_stages:
        raise ValueError(
            "--gin-active-fast-path cannot be combined with "
            "--gin-local-ablation-stages"
        )
    if (
        _gin_fast_path_transition_requested(args)
        and args.num_max_tokens_per_rank <= GIN_FALLBACK_PROBE_TOKENS
    ):
        raise ValueError(
            "--gin-bulk-combine/--gin-direct-dispatch transition validation "
            "requires --num-max-tokens-per-rank greater than 64"
        )
    if args.gin_bulk_combine:
        if not args.require_gin or not args.gin_active_fast_path:
            raise ValueError(
                "--gin-bulk-combine requires --require-gin and "
                "--gin-active-fast-path")
        if args.gin_local_ablation_stages:
            raise ValueError(
                "--gin-bulk-combine cannot be combined with "
                "--gin-local-ablation-stages")
        if (
            args.num_experts,
            args.num_topk,
            args.hidden,
            args.intermediate_hidden,
            args.num_shared_experts,
            args.gin_outbox_depth,
        ) != (gin_num_experts, 16, 3584, 3072, 0, 64):
            raise ValueError(
                "--gin-bulk-combine requires "
                f"E{gin_num_experts}/topk16/H3584/I3072/no-shared/outbox64")
    if args.gin_direct_dispatch:
        if not args.require_gin or not args.gin_active_fast_path:
            raise ValueError(
                "--gin-direct-dispatch requires --require-gin and "
                "--gin-active-fast-path")
        if args.gin_local_ablation_stages:
            raise ValueError(
                "--gin-direct-dispatch cannot be combined with "
                "--gin-local-ablation-stages")
        if (
            world_size,
            args.num_experts,
            args.num_topk,
            args.hidden,
            args.intermediate_hidden,
            args.num_shared_experts,
        ) != (gin_world_size, gin_num_experts, 16, 3584, 3072, 0):
            raise ValueError(
                "--gin-direct-dispatch requires "
                f"EP{gin_world_size}/E{gin_num_experts}/topk16/H3584/I3072/no-shared")
        if args.num_max_tokens_per_rank < 384:
            raise ValueError(
                "--gin-direct-dispatch requires "
                "--num-max-tokens-per-rank >= 384")
    combine_owner_waves = getattr(args, "gin_combine_owner_waves", 0)
    if combine_owner_waves:
        if not (world_size == 8 and args.num_experts == 448 and
                args.require_gin and args.gin_bulk_combine and
                args.gin_direct_dispatch and experts_per_rank == 56):
            raise ValueError(
                "--gin-combine-owner-waves requires EP8/E448 GIN "
                "bulk/direct mode")


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


def _collect_gin_experiment_flags(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    dist: Any,
) -> Dict[str, Any]:
    """Validate JIT flags collectively before the first kernel can launch."""
    local_record = {
        "rank": rank,
        "direct_dispatch": bool(args.gin_direct_dispatch),
        "flags": {
            name: os.getenv(name, "0") for name in GIN_VALIDATED_FLAG_ENVS
        },
        "expert_width": os.getenv("DG_MEGAMOE_GIN_COMBINE_EXPERTS_PER_WAVE", "0"),
        "barrier_warps": os.getenv("DG_MEGAMOE_GIN_COMBINE_BARRIER_WARPS", "1"),
        "owner_waves": os.getenv(GIN_COMBINE_OWNER_WAVES_ENV, "0"),
    }
    gathered: List[Optional[Dict[str, Any]]] = [None] * world_size
    dist.all_gather_object(gathered, local_record)

    expected_record_keys = {
        "rank", "direct_dispatch", "flags", "expert_width",
        "barrier_warps", "owner_waves",
    }
    expected_flag_keys = set(GIN_VALIDATED_FLAG_ENVS)
    for expected_rank, record in enumerate(gathered):
        if not isinstance(record, dict) or set(record) != expected_record_keys:
            raise RuntimeError(
                "GIN experiment flag evidence has a malformed collective record "
                f"at rank {expected_rank}: {record!r}"
            )
        if record["rank"] != expected_rank:
            raise RuntimeError(
                "GIN experiment flag evidence has an unexpected rank record: "
                f"slot {expected_rank} reported rank {record['rank']!r}"
            )
        flags = record["flags"]
        if not isinstance(flags, dict) or set(flags) != expected_flag_keys:
            raise RuntimeError(
                "GIN experiment flag evidence has malformed flags at rank "
                f"{expected_rank}: {flags!r}"
            )
        if not isinstance(record["direct_dispatch"], bool):
            raise RuntimeError(
                "GIN experiment flag evidence has a non-boolean direct-dispatch "
                f"setting at rank {expected_rank}: {record['direct_dispatch']!r}"
            )

    direct_dispatch_values = [
        bool(record["direct_dispatch"]) for record in gathered
    ]
    if len(set(direct_dispatch_values)) != 1:
        raise RuntimeError(
            "--gin-direct-dispatch is not uniform across ranks: "
            + ", ".join(
                f"rank {rank_idx}={value}"
                for rank_idx, value in enumerate(direct_dispatch_values)
            )
        )
    direct_dispatch = direct_dispatch_values[0]

    result: Dict[str, Any] = {}
    for name in GIN_VALIDATED_FLAG_ENVS:
        raw_values = [record["flags"][name] for record in gathered]
        invalid = [
            (rank_idx, value)
            for rank_idx, value in enumerate(raw_values)
            if value not in ("0", "1")
        ]
        if invalid:
            raise RuntimeError(
                f"{name} must be 0 or 1 on every rank, got "
                + ", ".join(
                    f"rank {rank_idx}={value!r}" for rank_idx, value in invalid
                )
            )
        if len(set(raw_values)) != 1:
            raise RuntimeError(
                f"{name} is not uniform across ranks: "
                + ", ".join(
                    f"rank {rank_idx}={value!r}"
                    for rank_idx, value in enumerate(raw_values)
                )
            )
        enabled = raw_values[0] == "1"
        if enabled and not direct_dispatch:
            raise RuntimeError(
                f"{name}=1 requires --gin-direct-dispatch on every rank"
            )
        result[name] = enabled
    if (result[GIN_COMBINE_OVERLAP_ENV] and not (
            result[GIN_SINGLE_COMBINE_CONTEXT_ENV] and result[GIN_DISPATCH_OVERLAP_ENV])):
        raise RuntimeError(
            f"{GIN_COMBINE_OVERLAP_ENV}=1 requires SINGLE_COMBINE_CONTEXT=1 "
            "and DISPATCH_OVERLAP=1 on every rank"
        )
    if (result[GIN_STRONGVA_COMBINE_TERMINAL_ENV] and
            not result[GIN_COMBINE_OVERLAP_ENV]):
        raise RuntimeError(
            f"{GIN_STRONGVA_COMBINE_TERMINAL_ENV}=1 requires "
            "COMBINE_OVERLAP=1 on every rank"
        )
    if (result[GIN_PRECONSENSUS_PACK_ENV] and
            not result[GIN_COOP_DIRECT_PACK_ENV]):
        raise RuntimeError(
            f"{GIN_PRECONSENSUS_PACK_ENV}=1 requires "
            f"{GIN_COOP_DIRECT_PACK_ENV}=1 on every rank"
        )
    owner_wave_values = [record["owner_waves"] for record in gathered]
    invalid_owner_waves = [
        (rank_idx, value)
        for rank_idx, value in enumerate(owner_wave_values)
        if value not in ("0", "2", "4", "8")
    ]
    if invalid_owner_waves:
        raise RuntimeError(
            f"{GIN_COMBINE_OWNER_WAVES_ENV} must be exactly 0, 2, 4, or 8 "
            "on every rank, got "
            + ", ".join(
                f"rank {rank_idx}={value!r}"
                for rank_idx, value in invalid_owner_waves
            )
        )
    if len(set(owner_wave_values)) != 1:
        raise RuntimeError(
            f"{GIN_COMBINE_OWNER_WAVES_ENV} is not uniform across ranks: "
            + ", ".join(
                f"rank {rank_idx}={value!r}"
                for rank_idx, value in enumerate(owner_wave_values)
            )
        )
    combine_owner_waves = int(owner_wave_values[0])
    if combine_owner_waves and not (
            direct_dispatch and world_size == 8 and
            result[GIN_STRONGVA_COMBINE_TERMINAL_ENV] and
            result[GIN_COMBINE_OVERLAP_ENV] and
            result[GIN_SINGLE_COMBINE_CONTEXT_ENV] and
            result[GIN_DISPATCH_OVERLAP_ENV]):
        raise RuntimeError(
            f"{GIN_COMBINE_OWNER_WAVES_ENV}={combine_owner_waves} requires "
            "EP8 direct dispatch, STRONGVA_COMBINE_TERMINAL=1, "
            "COMBINE_OVERLAP=1, SINGLE_COMBINE_CONTEXT=1 and "
            "DISPATCH_OVERLAP=1 on every rank"
        )
    result[GIN_COMBINE_OWNER_WAVES_ENV] = combine_owner_waves
    incompatible = [
        record["rank"] for record in gathered
        if record["expert_width"] != "0" or record["barrier_warps"] != "1"
    ]
    if incompatible:
        raise RuntimeError(
            "retired experiments are unsupported in the clean single-context build; "
            "requires experts_per_wave=0 and barrier_warps=1 on every rank; "
            f"incompatible ranks: {incompatible}"
        )
    return result


def _gin_transport_evidence(
    buffer: Any,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    hostnames: Sequence[str],
    symmetric_memory_backend: Optional[str],
    symmetric_memory_registration: Optional[Any],
    dist: Any,
    *,
    topology: Optional[Any] = None,
) -> Dict[str, Any]:
    """Validate the facts needed to call a run GIN transport coverage.

    A truthy ``gin_enabled`` property alone is insufficient: the accuracy run
    must route data across hosts through the FP8xFP4 GIN kernel and the live
    context descriptor must carry the exact tuning requested by the CLI.
    """
    gin_enabled_value = getattr(buffer, "gin_enabled", None)
    gin_enabled = bool(gin_enabled_value) if gin_enabled_value is not None else False
    experiment_flags = _collect_gin_experiment_flags(
        args, rank, world_size, dist
    )
    if topology is not None:
        topology.require_validated_buffer(buffer, hostnames)
    if not args.require_gin:
        return {
            "requested": "auto",
            "gin_enabled": gin_enabled if gin_enabled_value is not None else None,
            "gin_api_present": gin_enabled_value is not None,
            "jit_experiment_flags": experiment_flags,
        }

    if topology is None:
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

    activity_gate_opt_raw = os.getenv(GIN_ACTIVITY_GATE_OPT_ENV, "0")
    if activity_gate_opt_raw not in ("0", "1"):
        raise RuntimeError(
            f"{GIN_ACTIVITY_GATE_OPT_ENV} must be 0 or 1, got "
            f"{activity_gate_opt_raw!r}"
        )
    activity_gate_opt = bool(int(activity_gate_opt_raw))

    context = getattr(buffer, "gin_context", None)
    snapshot_fn = getattr(context, "launch_descriptor_snapshot", None)
    if context is None or not callable(snapshot_fn):
        raise RuntimeError(
            "--require-gin requires a live context with launch_descriptor_snapshot()"
        )
    snapshot = dict(snapshot_fn())
    lsa_size = topology.lsa_size if topology is not None else 8
    expected = {
        "enabled": True,
        "rank": rank,
        "world_size": world_size,
        "lsa_rank": rank % lsa_size,
        "lsa_size": lsa_size,
        "context_count": 9,
        "requested_context_count": 9,
        "requested_signal_count": 2,
        "queue_depth": args.gin_queue_depth,
        "world_barrier_count": 4,
        "completion_batch": args.gin_completion_batch,
        "combine_chunk_bytes": args.gin_combine_chunk_bytes,
        "outbox_depth": args.gin_outbox_depth,
        "combine_issue_wave": args.gin_combine_issue_wave,
        "active_fast_path": args.gin_active_fast_path,
        "bulk_combine": args.gin_bulk_combine,
        "direct_dispatch": args.gin_direct_dispatch,
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
    if int(snapshot.get("signal_count", 0)) < 2:
        mismatches.append("signal_count must be at least 2")
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
    expected_lsa_ranks = (
        topology.expected_lsa_ranks(rank) if topology is not None else [
            peer for peer, hostname in enumerate(hostnames)
            if hostname == hostnames[rank]
        ]
    )
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
        "activity_gate_opt": activity_gate_opt,
        "jit_experiment_flags": experiment_flags,
        "symmetric_memory_backend": symmetric_memory_backend,
        "symmetric_memory_communicator_registration": "external_bridge",
        "cross_host_payload_routes": True,
        "launch_descriptor": {
            **{
                name: (
                    bool(snapshot[name])
                    if name == "enabled"
                    else snapshot[name]
                )
                for name in expected
            },
            "signal_count": int(snapshot["signal_count"]),
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


def _worker(local_rank: int, local_world_size: int, args: argparse.Namespace,
            *, topology: Optional[Any] = None) -> None:
    # The private compile-time knob must never affect the normal correctness
    # suite, even if a parent shell happens to export it.
    os.environ[GIN_LOCAL_ABLATION_ENV] = "0"
    os.environ[GIN_COMBINE_OWNER_WAVES_ENV] = str(
        getattr(args, "gin_combine_owner_waves", 0))
    torch, dist, deep_gemm = _load_runtime()
    buffer = None
    symmetric_memory_registration = None
    synchronized_success = False
    passed_record = None
    perf_record = None
    benchmark_graph = None
    ablation_graphs = None
    transition_graphs: Dict[int, Any] = {}
    try:
        _init_distributed(local_rank, local_world_size, torch, dist)
        _check_runtime(torch, deep_gemm)
        physical_hostnames = None
        if topology is not None:
            # Only the separate GB200 entrypoint opts in. Never replace real
            # hostnames with logical route groups or relax the Novita guard.
            from mega_moe_gb200_topology import GB200Topology
            if not isinstance(topology, GB200Topology):
                raise TypeError("topology must be an explicit GB200Topology")
            physical_hostnames = _all_hostnames(dist)
            topology.prepare(args, physical_hostnames, local_rank,
                             local_world_size, torch, dist)
        (
            symmetric_memory_backend,
            symmetric_memory_registration,
        ) = (
            _configure_symmetric_memory_backend(args, torch, dist)
            if topology is None else
            _configure_symmetric_memory_backend(args, torch, dist, force_nccl=True)
        )
        rank, world_size = dist.get_rank(), dist.get_world_size()
        _validate_args(args, world_size,
                       **({"gb200_world_size": topology.world_size} if topology is not None else {}))
        hostnames = _all_hostnames(dist) if physical_hostnames is None else physical_hostnames
        if args.require_cross_host and len(set(hostnames)) < 2:
            raise RuntimeError(
                "--require-cross-host was requested, but every rank is on "
                f"{hostnames[0]}; refusing to relabel same-host NVLink as GIN coverage"
            )
        if topology is None:
            _validate_gin_host_placement(args, world_size, hostnames)
        else:
            topology.require_prepared(hostnames)
        route_domains = hostnames if topology is None else topology.logical_route_domains

        experts_per_rank = args.num_experts // world_size
        cases = _make_route_cases(
            rank, args, experts_per_rank, route_domains, torch, dist
        )
        asymmetric_cross_routes = (
            _validate_single_remote_source_case(
                cases,
                rank,
                args,
                experts_per_rank,
                route_domains,
                dist,
            )
            if args.require_gin
            else None
        )
        is_matched_benchmark_shape = (
            (args.require_gin or topology is not None)
            and (world_size == 16 or (topology is not None and world_size == 8))
            and args.num_experts == world_size * 56
            and args.num_topk == 16
            and args.hidden == 3584
            and args.intermediate_hidden == 3072
            and args.num_tokens in MATCHED_BENCHMARK_TOKEN_COUNTS
        )
        matched_route_evidence = (
            _collect_matched_route_evidence(
                cases,
                rank,
                args,
                experts_per_rank,
                route_domains,
                int(
                    deep_gemm.get_block_m_for_mega_moe(
                        world_size,
                        args.num_experts,
                        args.num_max_tokens_per_rank,
                        args.num_tokens,
                        args.num_topk,
                        args.mma_type,
                    )
                ),
                dist,
            )
            if is_matched_benchmark_shape
            else {
                "enabled": False,
                "reason": "shape_is_not_matched_EP16_K3_T32_T40_or_T48",
            }
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
            gin_combine_issue_wave=args.gin_combine_issue_wave,
            gin_queue_depth=args.gin_queue_depth,
            gin_active_fast_path=args.gin_active_fast_path,
            gin_bulk_combine=args.gin_bulk_combine,
            gin_direct_dispatch=args.gin_direct_dispatch,
        )
        topology_evidence = (
            topology.validate_buffer(buffer, args, symmetric_memory_backend,
                                     symmetric_memory_registration, torch, dist)
            if topology is not None else None
        )
        transport_evidence = _gin_transport_evidence(
            buffer,
            args,
            rank,
            world_size,
            hostnames,
            symmetric_memory_backend,
            symmetric_memory_registration,
            dist,
            **({"topology": topology} if topology is not None else {}),
        )
        if topology is not None:
            transport_evidence.update(requested=topology.mode, gb200_topology=topology_evidence)
        inputs = _make_inputs(rank, args, torch, deep_gemm)
        weights = _make_kernel_weights(experts_per_rank, args, torch, deep_gemm)
        fingerprint = _weight_fingerprint(weights, torch, dist)
        harness = AccuracyHarness(
            rank, args, inputs, weights, cases, buffer, torch, dist, deep_gemm
        )
        if topology is not None:
            harness.physical_hostnames = list(hostnames)
            harness.route_domains = list(route_domains)
            harness.topology_evidence = topology_evidence

        snapshots, oracle_metrics = _snapshot_and_check_oracles(harness, torch, dist)
        _run_eager_stress(harness, snapshots, torch)
        _run_graph_stress(
            harness, snapshots, torch, dist
        )
        if args.benchmark_replays > 0:
            if args.gin_local_ablation_stages:
                ablation_graphs = _capture_local_ablation_graphs(
                    harness, snapshots, torch, dist
                )
                local_ablation_metrics = _benchmark_local_ablation_graphs(
                    harness,
                    ablation_graphs,
                    snapshots,
                    torch,
                    dist,
                )
                # Graph nodes retain the kernel arguments, including the
                # symmetric allocation. Retire them before buffer teardown.
                ablation_graphs = None
                benchmark_metrics = None
            else:
                benchmark_graph = _capture_launch_only_graph(
                    harness, torch, dist
                )
                benchmark_metrics = _benchmark_graph_routes(
                    harness,
                    benchmark_graph,
                    snapshots,
                    torch,
                    dist,
                )
                benchmark_graph = None
                local_ablation_metrics = None
        else:
            benchmark_metrics = None
            local_ablation_metrics = None

        # Payload stress is intentionally outside timing and restores epoch 0.
        payload_epoch_evidence = _run_payload_epoch_validation(harness, torch, dist)

        # Run after all timed work: the variable-token canaries intentionally
        # overwrite the primary harness's fixed inactive input tail.
        gin_transition_evidence = _run_gin_bulk_transition_validation(
            harness,
            rank,
            args,
            route_domains,
            transition_graphs,
            torch,
            dist,
            deep_gemm,
        )

        if rank == 0:
            rel_limit, abs_limit = _oracle_thresholds(args)
            result = {
                "status": "passed",
                "world_size": world_size,
                "hostnames": hostnames,
                "cross_host_exercised": len(set(hostnames)) > 1,
                "all_remote_realization": cases["all_remote"].owner_kind,
                "single_remote_source_cross_routes": asymmetric_cross_routes,
                "matched_route_evidence": matched_route_evidence,
                "transport": transport_evidence,
                "mma_type": args.mma_type,
                "input_pattern": {
                    "heterogeneous_input_scales": args.heterogeneous_input_scales,
                    "scale_granularity": 32,
                    "input_scale_exponent_range": [-3, 3] if args.heterogeneous_input_scales else None,
                },
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
                    "all_established_modes_under_cuda_graph": True,
                    "changing_payload_same_graph": payload_epoch_evidence["enabled"],
                    "gin_activity_transition_cycles": (
                        GIN_ACTIVITY_TRANSITION_CYCLES
                        if args.gin_active_fast_path
                        else 0
                    ),
                    "gin_fast_path_token_transition": bool(
                        gin_transition_evidence["enabled"]
                    ),
                    "gin_bulk_token_transition": bool(
                        gin_transition_evidence["enabled"]
                        and args.gin_bulk_combine
                    ),
                },
                "oracle_limits": {
                    "symmetric_diff": rel_limit,
                    "max_abs": abs_limit,
                },
                "oracle_metrics": oracle_metrics,
                "payload_epoch_validation": payload_epoch_evidence,
                "weight_fingerprint": fingerprint,
                "gin_fast_path_transition": gin_transition_evidence,
                # Retain the validation-r2 evidence key for bulk runbooks.
                "gin_bulk_transition": gin_transition_evidence,
            }
            if topology is not None:
                result["gb200_topology"] = topology_evidence
                topology.accuracy_result = result
            passed_record = "MEGAMOE_ACCURACY_JSON=" + json.dumps(
                result, sort_keys=True)
            if benchmark_metrics is not None or local_ablation_metrics is not None:
                perf_result = {
                    "status": "passed",
                    "world_size": world_size,
                    "cross_host_exercised": len(set(hostnames)) > 1,
                    "timing_scope": "launch_only_cuda_graph_device_time",
                    "aggregation": "slowest_rank_for_each_replay",
                    "route_schedule": (
                        "ablation_paired_plus_stage_isolated"
                        if local_ablation_metrics is not None
                        else "balanced_six_launch_plus_route_isolated"
                    ),
                    "same_host_control": (
                        "compile-time cumulative local-only specializations; "
                        "stage 0 is the current four-phase GIN kernel and "
                        "stages 3-4 replace GIN collectives with LSA barriers"
                        if local_ablation_metrics is not None
                        else (
                            "GIN-enabled fused kernel with the collectively "
                            "configured global activity gate and same-LSA "
                            "payload routes"
                            if args.gin_active_fast_path
                            else "GIN-enabled fused kernel with the always-on "
                            "r75 remote protocol and same-LSA payload routes"
                        )
                    ),
                    "route_isolated_warmups": (
                        args.benchmark_warmups
                        if local_ablation_metrics is not None
                        else args.benchmark_conditioning_replays
                    ),
                    "conditioning_replays_per_route": (
                        args.benchmark_conditioning_replays
                        if benchmark_metrics is not None
                        else None
                    ),
                    "replays": args.benchmark_replays,
                    "shape": result["shape"],
                    "input_pattern": result["input_pattern"],
                    "matched_route_evidence": matched_route_evidence,
                    "gin": {
                        "type": transport_evidence["gin_type"],
                        "activity_gate_opt": transport_evidence[
                            "activity_gate_opt"
                        ],
                        "jit_experiment_flags": transport_evidence[
                            "jit_experiment_flags"
                        ],
                        "completion_batch": args.gin_completion_batch,
                        "combine_chunk_bytes": args.gin_combine_chunk_bytes,
                        "outbox_depth": args.gin_outbox_depth,
                        "combine_issue_wave": args.gin_combine_issue_wave,
                        "combine_owner_waves": args.gin_combine_owner_waves,
                        "queue_depth": args.gin_queue_depth,
                        "active_fast_path": args.gin_active_fast_path,
                        "bulk_combine": args.gin_bulk_combine,
                        "direct_dispatch": args.gin_direct_dispatch,
                        "direct_dispatch_eligible_this_launch": (
                            args.gin_direct_dispatch
                            and args.num_tokens <= GIN_FAST_PATH_MAX_TOKENS
                        ),
                        "bulk_combine_eligible_this_launch": (
                            args.gin_bulk_combine
                            and args.num_tokens <= GIN_FAST_PATH_MAX_TOKENS
                        ),
                    },
                    "benchmark_schedule": (
                        benchmark_metrics["schedule"]
                        if benchmark_metrics is not None
                        else None
                    ),
                    "routes": (
                        benchmark_metrics["routes"]
                        if benchmark_metrics is not None
                        else None
                    ),
                    "local_ablation": local_ablation_metrics,
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
        os.environ[GIN_LOCAL_ABLATION_ENV] = "0"
        os.environ[GIN_COMBINE_OWNER_WAVES_ENV] = "0"
        # Also retire graphs on a failed validation/benchmark before the
        # symmetric allocation and its GIN registration are destroyed.
        benchmark_graph = None
        ablation_graphs = None
        transition_graphs.clear()
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
    parser.add_argument(
        "--decode-mns",
        type=int,
        default=None,
        help=(
            "Decode sequences per rank; runtime tokens equal MNS times "
            "one target token plus --draft-width draft tokens"
        ),
    )
    parser.add_argument(
        "--draft-width",
        type=int,
        default=None,
        help="Number of draft tokens in addition to each target token",
    )
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
        "--heterogeneous-input-scales", action="store_true",
        help="Vary exact input magnitudes and packed SF exponents by source/token/32-column group/epoch",
    )
    parser.add_argument(
        "--payload-epochs", type=int, default=1,
        help="After timing, validate N changing payloads and a return to epoch 0 through one graph and unchanged storage (1 disables)",
    )
    parser.add_argument(
        "--benchmark-replays",
        type=int,
        default=0,
        help=(
            "After correctness, time this many back-to-back CUDA graph replays "
            "per benchmark route (0 disables timing)"
        ),
    )
    parser.add_argument(
        "--benchmark-warmups",
        type=int,
        default=10,
        help=(
            "Per-specialization warmups for local-ablation timing; matched "
            "route timing uses --benchmark-conditioning-replays for both its "
            "balanced conditioning and route-isolated rewarm"
        ),
    )
    parser.add_argument(
        "--benchmark-conditioning-replays",
        type=int,
        default=60,
        help=(
            "Exact per-route replay count for the fixed balanced six-launch "
            "conditioning schedule; use the same value for every candidate"
        ),
    )
    parser.add_argument(
        "--benchmark-routes",
        nargs="+",
        choices=ROUTE_MODES,
        default=MATCHED_BENCHMARK_ROUTES,
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
        "--gin-outbox-depth", type=int, choices=(4, 8, 16, 64), default=8
    )
    parser.add_argument(
        "--gin-combine-issue-wave", type=int, choices=(1, 2, 4, 8), default=8
    )
    parser.add_argument(
        "--gin-combine-owner-waves",
        type=_parse_gin_combine_owner_waves,
        choices=(0, 2, 4, 8),
        default=0,
        help=(
            "Compile the StrongVA combine sender with 2, 4, or 8 fixed "
            "contiguous owner-expert readiness ranges; 0 preserves r4"
        ),
    )
    parser.add_argument(
        "--gin-active-fast-path",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Collectively select the global route-activity gate; when no rank "
            "has a cross-LSA route, bypass the remote GIN protocol"
        ),
    )
    parser.add_argument(
        "--gin-bulk-combine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use one compact owner-to-source return packet per active remote "
            "source for eligible <=48-token launches, and validate a "
            "same-context 48->64 fallback->48 eager/graph transition; "
            "--num-max-tokens-per-rank must exceed 64"
        ),
    )
    parser.add_argument(
        "--gin-direct-dispatch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Send activation, scale, weight, and compact route/control data "
            "directly from each source to each owner for eligible <=48-token "
            "launches, and validate a same-context 48->64 exact-r75 "
            "fallback->48 eager/graph transition"
        ),
    )
    parser.add_argument(
        "--gin-local-ablation-stages",
        type=int,
        nargs="+",
        choices=tuple(GIN_LOCAL_ABLATION_STAGES),
        default=(),
        help=(
            "Capture compile-time cumulative all-same-host ablations and "
            "benchmark them in paired rotating order; stage 0 is required"
        ),
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
    if args.draft_width < 0:
        parser.error("--draft-width must be nonnegative")
    if args.decode_mns is not None and args.decode_mns <= 0:
        parser.error("--decode-mns must be positive")
    if args.decode_mns is not None and args.num_tokens is not None:
        parser.error("use either --num-tokens or --decode-mns/--draft-width, not both")
    if args.decode_mns is not None:
        args.num_tokens = args.decode_mns * (1 + args.draft_width)
    elif args.num_tokens is None:
        args.num_tokens = 8
    if args.num_max_tokens_per_rank is None:
        args.num_max_tokens_per_rank = 384 if args.k3 else args.num_tokens + 16
    if args.num_processes < 1:
        parser.error("--num-processes must be positive")
    if args.eager_iterations < 1 or args.graph_replays < 1:
        parser.error("eager and graph iteration counts must both be positive")
    if args.payload_epochs < 1:
        parser.error("--payload-epochs must be positive")
    if args.benchmark_replays < 0:
        parser.error("--benchmark-replays must be non-negative")
    if args.benchmark_replays == 1:
        parser.error(
            "--benchmark-replays must be at least 2 for half-run summaries"
        )
    if args.benchmark_replays > 0 and not args.require_gin:
        parser.error("--benchmark-replays currently requires --require-gin")
    if args.benchmark_warmups < 1:
        parser.error("--benchmark-warmups must be positive")
    if (
        args.benchmark_conditioning_replays < 1
        or args.benchmark_conditioning_replays % 2 != 0
    ):
        parser.error(
            "--benchmark-conditioning-replays must be a positive even number"
        )
    if (
        args.benchmark_replays > 0
        and not args.gin_local_ablation_stages
        and args.benchmark_replays % 2 != 0
    ):
        parser.error(
            "matched --benchmark-replays must be even for the balanced "
            "six-launch schedule"
        )
    if len(set(args.benchmark_routes)) != len(args.benchmark_routes):
        parser.error("--benchmark-routes must not contain duplicates")
    if len(set(args.gin_local_ablation_stages)) != len(
        args.gin_local_ablation_stages
    ):
        parser.error("--gin-local-ablation-stages must not contain duplicates")
    if args.gin_local_ablation_stages and args.benchmark_replays == 0:
        parser.error("--gin-local-ablation-stages requires --benchmark-replays")
    if args.benchmark_replays > 0 and not args.gin_local_ablation_stages:
        if tuple(args.benchmark_routes) != MATCHED_BENCHMARK_ROUTES:
            parser.error(
                "matched benchmarking requires exactly --benchmark-routes "
                + " ".join(MATCHED_BENCHMARK_ROUTES)
            )
        if args.num_tokens not in MATCHED_BENCHMARK_TOKEN_COUNTS:
            parser.error(
                "matched benchmarking requires 32, 40, or 48 tokens per rank"
            )
        if (
            args.num_experts,
            args.num_topk,
            args.hidden,
            args.intermediate_hidden,
            args.num_shared_experts,
        ) != (896, 16, 3584, 3072, 0):
            parser.error(
                "matched benchmarking requires E896/topk16/H3584/I3072/"
                "no-shared"
            )
    if args.gin_active_fast_path and not args.require_gin:
        parser.error("--gin-active-fast-path requires --require-gin")
    if args.gin_active_fast_path and args.gin_local_ablation_stages:
        parser.error(
            "--gin-active-fast-path cannot be combined with "
            "--gin-local-ablation-stages"
        )
    if args.gin_bulk_combine and (
        not args.require_gin or not args.gin_active_fast_path
    ):
        parser.error(
            "--gin-bulk-combine requires --require-gin and "
            "--gin-active-fast-path"
        )
    if args.gin_bulk_combine and args.gin_local_ablation_stages:
        parser.error(
            "--gin-bulk-combine cannot be combined with "
            "--gin-local-ablation-stages"
        )
    if args.gin_direct_dispatch and (
        not args.require_gin or not args.gin_active_fast_path
    ):
        parser.error(
            "--gin-direct-dispatch requires --require-gin and "
            "--gin-active-fast-path"
        )
    if args.gin_direct_dispatch and args.gin_local_ablation_stages:
        parser.error(
            "--gin-direct-dispatch cannot be combined with "
            "--gin-local-ablation-stages"
        )
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
