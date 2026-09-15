"""EP8 direct-GIN MegaMoE accuracy and launch-only latency gate.

Run with one four-GPU host per node, for example::

  torchrun --nnodes=2 --nproc-per-node=4 tests/run_mega_moe_gin_ep8.py

The test-only extension owns NCCL resources and exports the same named capsule
that an upper runtime would. The independent EP1 MegaMoE launch is a valid
numerical reference because every rank constructs identical weights for each
local expert slot: remapping global expert IDs modulo 56 removes transport while
preserving the exact expert arithmetic.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import socket
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TOKENS = (32, 40, 48)
MODES = ("balanced", "half_remote", "all_remote", "hot_owner0")
E, K, H, I, LOCAL_E = 448, 16, 3584, 3072, 56


class _Ep1Group:
    def size(self): return 1
    def rank(self): return 0
    def barrier(self): pass


def _load_owner(torch, deep_gemm, verbose):
    from scripts.gin_build_config import resolve_gin_nccl_config
    from torch.utils.cpp_extension import load

    config = resolve_gin_nccl_config(True)
    info = deep_gemm._C.megamoe_gin_build_info()
    if not info["enabled"]:
        raise RuntimeError("DeepGEMM was built without DG_MEGAMOE_GIN=1")
    if int(info["nccl_headers_fingerprint"]) != int(config.header_fingerprint, 16):
        raise RuntimeError("provider and DeepGEMM NCCL header fingerprints differ")
    candidates = [Path(value) for value in [os.getenv("DG_NCCL_LIBRARY", "")]
                  if value]
    candidates += [config.root / name / "libnccl.so.2" for name in ("lib", "lib64")]
    library = next((path.resolve() for path in candidates if path.is_file()), None)
    if library is None:
        raise RuntimeError("DG_NCCL_ROOT (or DG_NCCL_LIBRARY) must provide libnccl.so.2")
    definitions = [f"-D{value}" for value in config.compile_definitions]
    return load(
        name=f"dg_megamoe_gin_test_owner_{config.header_fingerprint}",
        sources=[str(ROOT / "tests" / "mega_moe_gin_test_owner.cu")],
        extra_include_paths=[str(ROOT / "deep_gemm" / "include"),
                             str(config.include_dir)],
        extra_cflags=["-std=c++20", "-O2", *definitions],
        extra_cuda_cflags=["-std=c++20", "-O2", *definitions],
        extra_ldflags=["-Wl,--no-as-needed", str(library), "-Wl,--as-needed",
                       f"-Wl,-rpath,{library.parent}"],
        with_cuda=True, verbose=verbose)


def _configure_symm(torch, dist):
    import torch.distributed._symmetric_memory as symm
    enable = getattr(symm, "enable_symm_mem_for_group", None)
    if callable(enable):
        enable(dist.group.WORLD.group_name)
    symm.set_backend("NCCL")
    probe = torch.ones(1, dtype=torch.int32, device="cuda")
    dist.all_reduce(probe)
    if int(probe.item()) != 8:
        raise RuntimeError("ProcessGroupNCCL publication failed")
    registration = None
    try:
        from torch.distributed._symmetric_memory._nccl import register_external_nccl_comm
        device = torch.device("cuda", torch.cuda.current_device())
        backend = dist.group.WORLD._get_backend(device)
        registration = register_external_nccl_comm(
            dist.group.WORLD.group_name, int(backend._comm_ptr()), device, backend)
    except ImportError:
        pass
    return registration


def _make_weights(torch, deep_gemm):
    from deep_gemm.utils import per_token_cast_to_fp4

    def layer(n, k, first):
        packed = torch.empty((LOCAL_E, n, k // 2), dtype=torch.int8, device="cuda")
        scales = torch.empty((LOCAL_E, n, k // 32), dtype=torch.float32, device="cuda")
        rows = torch.arange(n, dtype=torch.int64, device="cuda")
        for expert in range(LOCAL_E):
            dense = torch.zeros((n, k), dtype=torch.bfloat16, device="cuda")
            if first:
                j, gate = rows % I, rows < I
                columns = torch.where(gate, (j * 17 + expert * 13 + 1) % H,
                                      (j * 29 + expert * 7 + 3) % H)
                values = torch.where(gate, 0.5 if expert % 2 == 0 else 1.0,
                                     1.0 if expert % 3 == 0 else 0.5)
            else:
                columns = (rows * 31 + expert * 11 + 5) % I
                values = torch.full((n,), 0.5 if expert % 4 < 2 else 1.0,
                                    dtype=torch.float32, device="cuda")
            dense[rows, columns] = values.to(torch.bfloat16)
            packed[expert], scales[expert] = per_token_cast_to_fp4(
                dense, use_ue8m0=True, gran_k=32)
        scales = deep_gemm.transform_sf_into_required_layout(
            scales, n, k, (1, 32), LOCAL_E)
        return packed, scales

    return deep_gemm.transform_weights_for_mega_moe(
        layer(2 * I, H, True), layer(H, I, False))


def _inputs(torch, count, rank):
    from deep_gemm.utils import per_token_cast_to_fp8
    token = torch.arange(count, device="cuda", dtype=torch.int64)[:, None]
    column = torch.arange(H, device="cuda", dtype=torch.int64)[None, :]
    code = (rank * 131 + token * 17 + column * 7) % 30
    source = torch.where((code & 1) == 0, 1.0, -1.0)
    source *= (code % 15 + 1).float() / 16
    return per_token_cast_to_fp8(
        source.to(torch.bfloat16), use_ue8m0=True, gran_k=32,
        use_packed_ue8m0=True)


def _routes(torch, mode, count, rank):
    route = torch.arange(count * K, dtype=torch.int64).view(count, K)
    lane, lsa = rank % 4, rank // 4
    local = lsa * 4 + (lane + route) % 4
    remote = (1 - lsa) * 4 + (lane + route) % 4
    if mode == "balanced": owner = local
    elif mode == "half_remote": owner = torch.where((route % 2) == 0, remote, local)
    elif mode == "all_remote": owner = remote
    elif mode == "hot_owner0": owner = torch.zeros_like(route)
    else: raise ValueError(mode)
    expert = route % LOCAL_E
    weights = torch.pow(2.0, -((route % 3) + 1).float())
    return (owner * LOCAL_E + expert).cuda(), weights.cuda()


def _expected_stats(torch, dist, indices, rank):
    gathered = [torch.empty_like(indices) for _ in range(8)]
    dist.all_gather(gathered, indices)
    values = torch.stack(gathered).flatten()
    owned = values[(values >= rank * LOCAL_E) &
                   (values < (rank + 1) * LOCAL_E)] - rank * LOCAL_E
    return torch.bincount(owned, minlength=LOCAL_E).to(torch.int32)


def _copy(buffer, values):
    x, sf, indices, weights = values
    count = x.shape[0]
    buffer.x[:count].copy_(x); buffer.x_sf[:count].copy_(sf)
    buffer.topk_idx[:count].copy_(indices)
    buffer.topk_weights[:count].copy_(weights)


def _check(torch, dist, actual, reference, stats, expected, label):
    a, b = actual.double(), reference.double()
    denominator = (a.square() + b.square()).sum()
    relative = 0.0 if denominator.item() == 0 else max(
        0.0, float((1 - 2 * (a * b).sum() / denominator).item()))
    absolute = float((a - b).abs().max().item())
    errors = []
    if not bool(torch.isfinite(actual).all().item()):
        errors.append("output contains non-finite elements")
    if relative > 3e-5 or absolute > 0.25:
        errors.append(f"output rel={relative:.3g} abs={absolute:.3g}")
    if not torch.equal(stats, expected):
        errors.append(f"stats actual={stats.cpu().tolist()} expected={expected.cpu().tolist()}")
    gathered = [None] * 8
    dist.all_gather_object(gathered, errors)
    failures = [f"rank {rank}: {error}" for rank, items in enumerate(gathered)
                for error in items]
    if failures:
        raise AssertionError(f"{label}: " + "; ".join(failures))


def _run(args):
    import torch
    import torch.distributed as dist
    import deep_gemm

    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", init_method="env://",
                            device_id=torch.device(f"cuda:{local_rank}"))
    if dist.get_world_size() != 8 or torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("this gate requires exactly eight SM100 ranks")
    hostnames = [None] * 8
    dist.all_gather_object(hostnames, socket.gethostname())
    if len(set(hostnames[:4])) != 1 or len(set(hostnames[4:])) != 1 or hostnames[0] == hostnames[4]:
        raise RuntimeError("rank order must be contiguous across two four-GPU hosts")

    registration = _configure_symm(torch, dist)
    provider = _load_owner(torch, deep_gemm, args.verbose_build)
    gin = deep_gemm.SymmBuffer.for_ep8_gin(
        dist.group.WORLD, max_active_tokens=48)
    ids = [provider.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(ids, src=0)
    window_offset = int(gin.handle.offset)
    owner = provider.TestGinOwner(
        gin.buffer, ids[0], rank, gin.buffer.data_ptr() - window_offset,
        gin.buffer.nbytes + window_offset)
    facts = [None] * 8; dist.all_gather_object(facts, owner.facts())
    for peer, fact in enumerate(facts):
        if ((fact["rank"], fact["world_size"], fact["lsa_rank"],
             fact["lsa_size"]) != (peer, 8, peer % 4, 4) or
                fact["contexts"] < 6 or fact["connections"] <= 0):
            raise RuntimeError(f"collective transport validation failed: {facts}")
    if len({fact["contexts"] for fact in facts}) != 1:
        raise RuntimeError("GIN ranks received different context counts")
    if (len({fact["window_bytes"] for fact in facts}) != 1 or
            len({fact["window_offset"] for fact in facts}) != 1):
        raise RuntimeError("GIN windows have different extents or buffer offsets")
    gin.bind_gin_transport(owner.capsule(), owner)
    reference = deep_gemm.SymmBuffer(_Ep1Group(), LOCAL_E, 48, K, H, I)
    weights = _make_weights(torch, deep_gemm)
    stats = torch.zeros(LOCAL_E, dtype=torch.int32, device="cuda")
    cases, snapshots = {}, {}

    for count in TOKENS:
        x, sf = _inputs(torch, count, rank)
        for mode in MODES:
            indices, topk_weights = _routes(torch, mode, count, rank)
            cases[count, mode] = ((x, sf, indices, topk_weights),
                                  _expected_stats(torch, dist, indices, rank))
        local_indices = cases[count, "balanced"][0][2] % LOCAL_E
        _copy(reference, (x, sf, local_indices,
                          cases[count, "balanced"][0][3]))
        output = torch.empty((count, H), dtype=torch.bfloat16, device="cuda")
        deep_gemm.fp8_fp4_mega_moe(output, *weights, reference)
        torch.cuda.synchronize(); snapshots[count] = output.clone()

    expected = torch.zeros_like(stats)
    for generation in range(args.eager_generations):
        for count in TOKENS:
            for mode in MODES:
                values, increment = cases[count, mode]; _copy(gin, values)
                output = torch.empty((count, H), dtype=torch.bfloat16, device="cuda")
                deep_gemm.fp8_fp4_mega_moe(
                    output, *weights, gin,
                    cumulative_local_expert_recv_stats=stats)
                torch.cuda.synchronize(); expected.add_(increment)
                _check(torch, dist, output, snapshots[count], stats, expected,
                       f"eager/{generation}/{count}/{mode}")

    graphs, outputs = {}, {}
    for count in TOKENS:
        _copy(gin, cases[count, "balanced"][0])
        outputs[count] = torch.empty((count, H), dtype=torch.bfloat16, device="cuda")
        dist.barrier(); graphs[count] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[count]):
            deep_gemm.fp8_fp4_mega_moe(
                outputs[count], *weights, gin,
                cumulative_local_expert_recv_stats=stats)
        torch.cuda.synchronize()
    stats.zero_(); expected.zero_()
    for replay in range(args.graph_replays):
        for count in TOKENS:
            for mode in MODES:
                values, increment = cases[count, mode]; _copy(gin, values)
                graphs[count].replay(); torch.cuda.synchronize(); expected.add_(increment)
                _check(torch, dist, outputs[count], snapshots[count], stats, expected,
                       f"graph/{replay}/{count}/{mode}")

    latency = {}
    for count in TOKENS:
        for mode in MODES:
            values, increment = cases[count, mode]; _copy(gin, values)
            for _ in range(args.warmup): graphs[count].replay()
            torch.cuda.synchronize(); stats.zero_(); dist.barrier()
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for _ in range(args.benchmark_replays): graphs[count].replay()
            end.record(); end.synchronize()
            elapsed = torch.tensor(
                [start.elapsed_time(end) / args.benchmark_replays], device="cuda")
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            _check(torch, dist, outputs[count], snapshots[count], stats,
                   increment * args.benchmark_replays, f"timed/{count}/{mode}")
            latency[f"T{count}/{mode}"] = float(elapsed.item())
    if rank == 0:
        print(json.dumps({"shape": "EP8/E448/top16/H3584/I3072",
                          "accuracy": "passed", "max_rank_kernel_ms": latency},
                         indent=2, sort_keys=True), flush=True)

    for graph in graphs.values(): graph.reset()
    graphs.clear(); del graph
    torch.cuda.synchronize(); dist.barrier()
    owner.close(); dist.barrier()
    gin.destroy(); reference.destroy(); del gin, reference; gc.collect()
    if registration is not None: registration.unregister()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eager-generations", type=int, default=2)
    parser.add_argument("--graph-replays", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--benchmark-replays", type=int, default=50)
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()
    if min(args.eager_generations, args.graph_replays,
           args.warmup, args.benchmark_replays) <= 0:
        parser.error("all iteration counts must be positive")
    _run(args)


if __name__ == "__main__":
    main()
