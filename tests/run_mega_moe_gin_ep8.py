"""EP8 direct-GIN MegaMoE accuracy and launch-only latency gate.

Run with one four-GPU host per node, for example::

  torchrun --nnodes=2 --nproc-per-node=4 tests/run_mega_moe_gin_ep8.py

Add ``--capability-matrix`` to run correctness cases covering dynamic
model geometry, both sides of the 32/64-expert packet-unpack boundaries, an
active-token capacity above 48, routed FP8 weights, BF16, shared experts, and
cross-geometry buffer reuse through eager launches and captured graphs after
the default latency matrix. Add --dense-accuracy for dense reference weights.
--lsa-size selects the uniform contiguous LSA partition; the world size follows
torchrun, so the same gate can exercise EP8/LSA4, EP12/LSA4 and EP16/LSA8.

The test-only extension owns NCCL resources and exports the same named capsule
that an upper runtime would. The independent EP1 MegaMoE launch is a valid
numerical reference because every rank constructs identical weights for each
local expert slot: remapping global expert IDs modulo the local expert count
removes transport while preserving the exact expert arithmetic.
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
WORLD_SIZE = 8
LSA_SIZE = 4


class _Config:
    __slots__ = (
        "name", "experts_per_rank", "topk", "hidden",
        "intermediate_hidden", "max_active_tokens", "mma_type",
        "num_shared_experts", "activation_clamp", "fast_math")

    def __init__(self, name, experts_per_rank, topk, hidden,
                 intermediate_hidden, max_active_tokens,
                 mma_type="fp8xfp4", num_shared_experts=0,
                 activation_clamp=None, fast_math=True):
        self.name = name
        self.experts_per_rank = experts_per_rank
        self.topk = topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.max_active_tokens = max_active_tokens
        self.mma_type = mma_type
        self.num_shared_experts = num_shared_experts
        self.activation_clamp = activation_clamp
        self.fast_math = fast_math

    @property
    def num_experts(self):
        return WORLD_SIZE * self.experts_per_rank

    @property
    def shape(self):
        return (
            f"EP{WORLD_SIZE}/LSA{LSA_SIZE}/E{self.num_experts}/top{self.topk}/H{self.hidden}/"
            f"I{self.intermediate_hidden}/{self.mma_type}/"
            f"shared{self.num_shared_experts}/cap{self.max_active_tokens}"
        )


DEFAULT = _Config("default", 56, 16, 3584, 3072, 48)
# Retain the established target-shape names for lightweight source-contract
# tests and for scripts importing the default benchmark geometry.
E, K, H, I, LOCAL_E = (
    DEFAULT.num_experts, DEFAULT.topk, DEFAULT.hidden,
    DEFAULT.intermediate_hidden, DEFAULT.experts_per_rank)

# This matrix is deliberately small: each entry creates a distinct MegaMoE JIT
# specialization on every rank. The first entry jointly covers a partial final
# four-owner wave, a third 32-expert unpack round, non-default top-k/hidden
# sizes, a capacity above 48 active tokens, and routed FP8 weights. The second
# exercises an experts-per-rank count below 32 plus shared-expert scheduling
# and accumulation without duplicating the default benchmark matrix.
CAPABILITY_CASES = (
    (_Config("dynamic_fp8", 65, 6, 2048, 1024, 64,
             mma_type="fp8xfp8"), 64, "all_remote"),
    (_Config("shared_fp4", 17, 6, 2048, 1024, 24,
             num_shared_experts=2, activation_clamp=1.0,
             fast_math=False), 24, "masked_remote"),
    (_Config("shared_bf16", 17, 6, 2048, 1024, 24,
             mma_type="bf16xbf16", num_shared_experts=2,
             activation_clamp=1.0, fast_math=False), 17, "masked_remote"),
)


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
    if int(probe.item()) != WORLD_SIZE:
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


def _make_weights(torch, deep_gemm, config, dense=False):
    from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8

    def dense_layer(n, k, expert, first):
        if dense:
            generator = torch.Generator(device="cuda")
            generator.manual_seed(1709 + expert * 7919 + int(first) * 131)
            return (torch.randn((n, k), generator=generator, device="cuda") /
                    k ** 0.5).to(torch.bfloat16)
        rows = torch.arange(n, dtype=torch.int64, device="cuda")
        dense = torch.zeros((n, k), dtype=torch.bfloat16, device="cuda")
        if first:
            intermediate = n // 2
            j, gate = rows % intermediate, rows < intermediate
            columns = torch.where(gate, (j * 17 + expert * 13 + 1) % k,
                                  (j * 29 + expert * 7 + 3) % k)
            values = torch.where(gate, 0.5 if expert % 2 == 0 else 1.0,
                                 1.0 if expert % 3 == 0 else 0.5)
        else:
            columns = (rows * 31 + expert * 11 + 5) % k
            values = torch.full((n,), 0.5 if expert % 4 < 2 else 1.0,
                                dtype=torch.float32, device="cuda")
        dense[rows, columns] = values.to(torch.bfloat16)
        return dense

    def routed_layer(n, k, first):
        if config.mma_type == "bf16xbf16":
            return torch.stack([dense_layer(n, k, expert, first)
                                for expert in range(config.experts_per_rank)])
        if config.mma_type == "fp8xfp8":
            packed = torch.empty(
                (config.experts_per_rank, n, k),
                dtype=torch.float8_e4m3fn, device="cuda")
            cast = per_token_cast_to_fp8
        else:
            packed = torch.empty(
                (config.experts_per_rank, n, k // 2),
                dtype=torch.int8, device="cuda")
            cast = per_token_cast_to_fp4
        scales = torch.empty(
            (config.experts_per_rank, n, k // 32),
            dtype=torch.float32, device="cuda")
        for expert in range(config.experts_per_rank):
            packed[expert], scales[expert] = cast(
                dense_layer(n, k, expert, first),
                use_ue8m0=True, gran_k=32)
        scales = deep_gemm.transform_sf_into_required_layout(
            scales, n, k, (1, 32), config.experts_per_rank)
        return packed, scales

    routed = deep_gemm.transform_weights_for_mega_moe(
        routed_layer(2 * config.intermediate_hidden, config.hidden, True),
        routed_layer(config.hidden, config.intermediate_hidden, False))
    if config.num_shared_experts == 0:
        return routed, (None, None)

    def shared_layer(n, k, first):
        if config.mma_type == "bf16xbf16":
            return dense_layer(n, k, config.experts_per_rank, first)
        weight, scales = per_token_cast_to_fp8(
            dense_layer(n, k, config.experts_per_rank, first),
            use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        aligned_n = (n + 3) // 4 * 4
        tma_scales = torch.empty_strided(
            scales.shape, (1, aligned_n), dtype=scales.dtype,
            device=scales.device)
        tma_scales.copy_(scales)
        return weight, tma_scales

    shared_intermediate = (
        config.intermediate_hidden * config.num_shared_experts)
    shared = deep_gemm.transform_weights_for_mega_moe(
        shared_layer(2 * shared_intermediate, config.hidden, True),
        shared_layer(config.hidden, shared_intermediate, False))
    return routed, shared


def _inputs(torch, count, rank, config):
    from deep_gemm.utils import per_token_cast_to_fp8
    token = torch.arange(count, device="cuda", dtype=torch.int64)[:, None]
    column = torch.arange(
        config.hidden, device="cuda", dtype=torch.int64)[None, :]
    code = (rank * 131 + token * 17 + column * 7) % 30
    source = torch.where((code & 1) == 0, 1.0, -1.0)
    source *= (code % 15 + 1).float() / 16
    if config.mma_type == "bf16xbf16":
        return source.to(torch.bfloat16), None
    return per_token_cast_to_fp8(
        source.to(torch.bfloat16), use_ue8m0=True, gran_k=32,
        use_packed_ue8m0=True)


def _routes(torch, mode, count, rank, config):
    route = torch.arange(
        count * config.topk, dtype=torch.int64).view(count, config.topk)
    lane, lsa_base = rank % LSA_SIZE, rank // LSA_SIZE * LSA_SIZE
    local = lsa_base + (lane + route) % LSA_SIZE
    remote_slot = (rank + route) % (WORLD_SIZE - LSA_SIZE)
    remote = torch.where(remote_slot < lsa_base,
                         remote_slot, remote_slot + LSA_SIZE)
    if mode == "balanced": owner = local
    elif mode == "half_remote": owner = torch.where((route % 2) == 0, remote, local)
    elif mode in ("all_remote", "masked_remote"): owner = remote
    elif mode == "hot_owner0": owner = torch.zeros_like(route)
    else: raise ValueError(mode)
    expert = route % config.experts_per_rank
    weights = torch.pow(2.0, -((route % 3) + 1).float())
    indices = owner * config.experts_per_rank + expert
    if mode == "masked_remote":
        indices = torch.where((route % 5) == 0, -1, indices)
    return indices.cuda(), weights.cuda()


def _expected_stats(torch, dist, indices, rank, config):
    gathered = [torch.empty_like(indices) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered, indices)
    values = torch.stack(gathered).flatten()
    owned = values[(values >= rank * config.experts_per_rank) &
                   (values < (rank + 1) * config.experts_per_rank)]
    owned -= rank * config.experts_per_rank
    return torch.bincount(
        owned, minlength=config.experts_per_rank).to(torch.int32)


def _shared_input_sf(torch, deep_gemm, buffer, sf, count, config,
                     num_ranks, num_experts):
    if config.num_shared_experts == 0 or config.mma_type == "bf16xbf16":
        return None
    block_m = deep_gemm.get_block_m_for_mega_moe(
        num_ranks, num_experts, buffer.num_max_tokens_per_rank, count,
        config.topk, config.mma_type)
    aligned_block_m = (block_m + 127) // 128 * 128
    num_m_blocks = (count + block_m - 1) // block_m
    num_max_sf_tokens = buffer.shared_l1_acts_sf.shape[0]
    result = torch.empty_strided(
        (num_max_sf_tokens, sf.shape[1]),
        (1, num_max_sf_tokens), dtype=sf.dtype, device=sf.device)
    result.zero_()
    for block_idx in range(num_m_blocks):
        num_block_tokens = min(block_m, count - block_idx * block_m)
        for token_in_block in range(num_block_tokens):
            transposed_token = (
                (token_in_block // 128) * 128
                + (token_in_block % 32) * 4
                + (token_in_block % 128) // 32
            )
            result[block_idx * aligned_block_m + transposed_token].copy_(
                sf[block_idx * block_m + token_in_block])
    return result


def _copy(buffer, values, shared_input_sf=None):
    x, sf, indices, weights = values
    count = x.shape[0]
    buffer.x[:count].copy_(x)
    if sf is not None:
        buffer.x_sf[:count].copy_(sf)
    buffer.topk_idx[:count].copy_(indices)
    buffer.topk_weights[:count].copy_(weights)
    if shared_input_sf is not None:
        buffer.shared_l1_acts_sf.copy_(shared_input_sf)


def _check(torch, dist, actual, reference, stats, expected, label):
    a, b = actual.double(), reference.double()
    denominator = (a.square() + b.square()).sum()
    relative = 0.0 if denominator.item() == 0 else max(
        0.0, float((1 - 2 * (a * b).sum() / denominator).item()))
    absolute = float((a - b).abs().max().item())
    errors = []
    if not bool(torch.isfinite(actual).all().item()):
        errors.append("output contains non-finite elements")
    if not bool(torch.isfinite(reference).all().item()):
        errors.append("reference contains non-finite elements")
    if relative > 3e-5 or absolute > 0.25:
        errors.append(f"output rel={relative:.3g} abs={absolute:.3g}")
    if not torch.equal(stats, expected):
        errors.append(f"stats actual={stats.cpu().tolist()} expected={expected.cpu().tolist()}")
    gathered = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, errors)
    failures = [f"rank {rank}: {error}" for rank, items in enumerate(gathered)
                for error in items]
    if failures:
        raise AssertionError(f"{label}: " + "; ".join(failures))


def _launch(deep_gemm, output, routed_weights, shared_weights, buffer,
            stats=None, activation_clamp=None, fast_math=True):
    kwargs = {}
    if shared_weights[0] is not None:
        kwargs.update(
            shared_l1_weights=shared_weights[0],
            shared_l2_weights=shared_weights[1])
    if stats is not None:
        kwargs["cumulative_local_expert_recv_stats"] = stats
    operation = (deep_gemm.bf16_mega_moe if buffer.mma_type == "bf16xbf16"
                 else deep_gemm.fp8_fp4_mega_moe)
    operation(
        output, *routed_weights, buffer,
        activation_clamp=activation_clamp, fast_math=fast_math, **kwargs)


def _open_gin(torch, dist, deep_gemm, provider, rank, config):
    gin = deep_gemm.SymmBuffer.for_gin(
        dist.group.WORLD, lsa_size=LSA_SIZE,
        max_active_tokens=config.max_active_tokens,
        num_experts=config.num_experts,
        num_topk=config.topk,
        hidden=config.hidden,
        intermediate_hidden=config.intermediate_hidden,
        num_shared_experts=config.num_shared_experts,
        mma_type=config.mma_type)
    ids = [provider.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(ids, src=0)
    window_offset = int(gin.handle.offset)
    owner = provider.TestGinOwner(
        gin.buffer, ids[0], rank, gin.buffer.data_ptr() - window_offset,
        gin.buffer.nbytes + window_offset, WORLD_SIZE, LSA_SIZE)
    facts = [None] * WORLD_SIZE
    dist.all_gather_object(facts, owner.facts())
    for peer, fact in enumerate(facts):
        if ((fact["rank"], fact["world_size"], fact["lsa_rank"],
             fact["lsa_size"]) != (peer, WORLD_SIZE, peer % LSA_SIZE, LSA_SIZE) or
                fact["contexts"] < 6 or fact["connections"] <= 0):
            raise RuntimeError(f"collective transport validation failed: {facts}")
    if len({fact["contexts"] for fact in facts}) != 1:
        raise RuntimeError("GIN ranks received different context counts")
    if (len({fact["window_bytes"] for fact in facts}) != 1 or
            len({fact["window_offset"] for fact in facts}) != 1):
        raise RuntimeError("GIN windows have different extents or buffer offsets")
    gin.bind_gin_transport(owner.capsule(), owner)
    return gin, owner


def _close_case(torch, dist, gin, reference, owner):
    torch.cuda.synchronize()
    dist.barrier()
    owner.close()
    dist.barrier()
    gin.destroy()
    reference.destroy()
    torch.cuda.empty_cache()


def _accuracy_fixture(torch, dist, deep_gemm, rank, config, count,
                      reference, routed_weights, shared_weights,
                      mode, generation):
    x, sf = _inputs(torch, count, rank + generation * WORLD_SIZE, config)
    indices, topk_weights = _routes(torch, mode, count, rank, config)
    values = x, sf, indices, topk_weights
    expected = _expected_stats(torch, dist, indices, rank, config)
    local_indices = torch.where(
        indices >= 0, indices % config.experts_per_rank, indices)
    local_sf = _shared_input_sf(
        torch, deep_gemm, reference, sf, count, config,
        num_ranks=1, num_experts=config.experts_per_rank)
    _copy(reference, (x, sf, local_indices, topk_weights), local_sf)
    reference_output = torch.empty(
        (count, config.hidden), dtype=torch.bfloat16, device="cuda")
    _launch(deep_gemm, reference_output, routed_weights, shared_weights,
            reference, activation_clamp=config.activation_clamp,
            fast_math=config.fast_math)
    return values, expected, reference_output


def _run_capability_case(torch, dist, deep_gemm, provider, rank,
                         config, count, mode, dense=False):
    gin, owner = _open_gin(
        torch, dist, deep_gemm, provider, rank, config)
    reference = deep_gemm.SymmBuffer(
        _Ep1Group(), config.experts_per_rank,
        config.max_active_tokens, config.topk,
        config.hidden, config.intermediate_hidden,
        num_shared_experts=config.num_shared_experts,
        mma_type=config.mma_type)
    routed, shared = _make_weights(torch, deep_gemm, config, dense=dense)
    stats = torch.zeros(config.experts_per_rank, dtype=torch.int32, device="cuda")
    output = torch.empty(
        (count, config.hidden), dtype=torch.bfloat16, device="cuda")
    graph = None
    for generation in range(2):
        for placement in dict.fromkeys((mode, *MODES, "masked_remote")):
            values, expected, oracle = _accuracy_fixture(
                torch, dist, deep_gemm, rank, config, count,
                reference, routed, shared, placement, generation)
            gin_sf = _shared_input_sf(
                torch, deep_gemm, gin, values[1], count, config,
                num_ranks=WORLD_SIZE, num_experts=config.num_experts)
            _copy(gin, values, gin_sf)
            stats.zero_()
            _launch(deep_gemm, output, routed, shared, gin, stats,
                    activation_clamp=config.activation_clamp,
                    fast_math=config.fast_math)
            torch.cuda.synchronize()
            label = f"capability/{config.name}/{placement}/{generation}"
            _check(torch, dist, output, oracle, stats, expected, label + "/eager")
            if graph is None:
                dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    _launch(deep_gemm, output, routed, shared, gin, stats,
                            activation_clamp=config.activation_clamp,
                            fast_math=config.fast_math)
            stats.zero_()
            graph.replay()
            torch.cuda.synchronize()
            _check(torch, dist, output, oracle, stats, expected, label + "/graph")

    graph.reset()
    result = {"shape": config.shape, "tokens": count,
              "dense_weights": dense, "accuracy": "passed"}
    del (graph, routed, shared, stats, output, values,
         expected, oracle, gin_sf)
    _close_case(torch, dist, gin, reference, owner)
    return result


def _run_reuse_case(torch, dist, deep_gemm, provider, rank):
    # Deliberately alternate both geometry and activation dtype. Each graph
    # captures its alias once; replay must retain one allocation-wide generation.
    configs = (CAPABILITY_CASES[0][0], CAPABILITY_CASES[2][0])
    required = []
    alignment = deep_gemm._C.get_token_alignment_for_mega_moe()
    for config in configs:
        native_capacity = (
            (config.max_active_tokens + alignment - 1) // alignment * alignment)
        size, _ = deep_gemm._C.get_symm_buffer_size_for_mega_moe_gin(
            WORLD_SIZE,
            (config.num_experts, native_capacity, config.topk,
             config.hidden, config.intermediate_hidden, config.num_shared_experts,
             config.mma_type, "swiglu", config.max_active_tokens),
            LSA_SIZE)
        required.append(size)
    base_index = max(range(len(configs)), key=required.__getitem__)
    base, owner = _open_gin(
        torch, dist, deep_gemm, provider, rank, configs[base_index])
    entries = []
    for index, config in enumerate(configs):
        gin = base if index == base_index else deep_gemm.SymmBuffer(
            dist.group.WORLD, config.num_experts, config.max_active_tokens,
            config.topk, config.hidden, config.intermediate_hidden,
            num_shared_experts=config.num_shared_experts,
            mma_type=config.mma_type, base=base)
        assert gin.buffer.data_ptr() == base.buffer.data_ptr()
        reference = deep_gemm.SymmBuffer(
            _Ep1Group(), config.experts_per_rank, config.max_active_tokens,
            config.topk, config.hidden, config.intermediate_hidden,
            num_shared_experts=config.num_shared_experts,
            mma_type=config.mma_type)
        routed, shared = _make_weights(torch, deep_gemm, config)
        count = min(17, config.max_active_tokens)
        output = torch.empty(
            (count, config.hidden), dtype=torch.bfloat16, device="cuda")
        stats = torch.zeros(
            config.experts_per_rank, dtype=torch.int32, device="cuda")
        entries.append([config, gin, reference, routed, shared,
                        count, output, stats, None])

    for generation in range(3):
        for index in (0, 1, 0):
            config, gin, reference, routed, shared, count, output, stats, graph = entries[index]
            values, expected, oracle = _accuracy_fixture(
                torch, dist, deep_gemm, rank, config, count, reference,
                routed, shared, "all_remote", generation)
            gin_sf = _shared_input_sf(
                torch, deep_gemm, gin, values[1], count, config,
                num_ranks=WORLD_SIZE, num_experts=config.num_experts)
            _copy(gin, values, gin_sf)
            stats.zero_()
            if graph is None:
                _launch(deep_gemm, output, routed, shared, gin, stats,
                        activation_clamp=config.activation_clamp,
                        fast_math=config.fast_math)
                torch.cuda.synchronize()
                _check(torch, dist, output, oracle, stats, expected,
                       f"reuse/{config.name}/eager")
                graph = torch.cuda.CUDAGraph()
                dist.barrier()
                with torch.cuda.graph(graph):
                    _launch(deep_gemm, output, routed, shared, gin, stats,
                            activation_clamp=config.activation_clamp,
                            fast_math=config.fast_math)
                entries[index][-1] = graph
                stats.zero_()
            graph.replay()
            torch.cuda.synchronize()
            _check(torch, dist, output, oracle, stats, expected,
                   f"reuse/{config.name}/graph/{generation}")

    for entry in entries:
        entry[-1].reset()
    torch.cuda.synchronize()
    dist.barrier()
    owner.close()
    dist.barrier()
    for entry in entries:
        entry[1].destroy()
        entry[2].destroy()
    del (entries, entry, routed, shared, output, stats, graph,
         values, expected, oracle, gin_sf)
    torch.cuda.empty_cache()
    return {"sequence": "FP8/BF16/FP8", "accuracy": "passed"}


def _run(args):
    global WORLD_SIZE, LSA_SIZE
    import torch
    import torch.distributed as dist
    import deep_gemm

    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", init_method="env://",
                            device_id=torch.device(f"cuda:{local_rank}"))
    WORLD_SIZE, LSA_SIZE = dist.get_world_size(), args.lsa_size
    if (not (1 <= LSA_SIZE < WORLD_SIZE <= 72) or
            WORLD_SIZE % LSA_SIZE != 0 or
            torch.cuda.get_device_capability()[0] != 10):
        raise RuntimeError("this gate requires SM100 ranks in equal contiguous LSA teams")

    registration = _configure_symm(torch, dist)
    provider = _load_owner(torch, deep_gemm, args.verbose_build)
    gin, owner = _open_gin(
        torch, dist, deep_gemm, provider, rank, DEFAULT)
    reference = deep_gemm.SymmBuffer(
        _Ep1Group(), DEFAULT.experts_per_rank,
        DEFAULT.max_active_tokens, DEFAULT.topk,
        DEFAULT.hidden, DEFAULT.intermediate_hidden)
    routed_weights, shared_weights = _make_weights(torch, deep_gemm, DEFAULT)
    stats = torch.zeros(
        DEFAULT.experts_per_rank, dtype=torch.int32, device="cuda")
    cases, snapshots = {}, {}

    for count in TOKENS:
        x, sf = _inputs(torch, count, rank, DEFAULT)
        for mode in MODES:
            indices, topk_weights = _routes(
                torch, mode, count, rank, DEFAULT)
            cases[count, mode] = ((x, sf, indices, topk_weights),
                                  _expected_stats(
                                      torch, dist, indices, rank, DEFAULT))
        local_indices = (
            cases[count, "balanced"][0][2] % DEFAULT.experts_per_rank)
        _copy(reference, (x, sf, local_indices,
                          cases[count, "balanced"][0][3]))
        output = torch.empty(
            (count, DEFAULT.hidden), dtype=torch.bfloat16, device="cuda")
        _launch(deep_gemm, output, routed_weights, shared_weights, reference)
        torch.cuda.synchronize(); snapshots[count] = output.clone()

    expected = torch.zeros_like(stats)
    for generation in range(args.eager_generations):
        for count in TOKENS:
            for mode in MODES:
                values, increment = cases[count, mode]; _copy(gin, values)
                output = torch.empty(
                    (count, DEFAULT.hidden),
                    dtype=torch.bfloat16, device="cuda")
                _launch(deep_gemm, output, routed_weights, shared_weights,
                        gin, stats)
                torch.cuda.synchronize(); expected.add_(increment)
                _check(torch, dist, output, snapshots[count], stats, expected,
                       f"eager/{generation}/{count}/{mode}")

    graphs, outputs = {}, {}
    for count in TOKENS:
        _copy(gin, cases[count, "balanced"][0])
        outputs[count] = torch.empty(
            (count, DEFAULT.hidden), dtype=torch.bfloat16, device="cuda")
        dist.barrier(); graphs[count] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[count]):
            _launch(deep_gemm, outputs[count], routed_weights,
                    shared_weights, gin, stats)
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
        print(json.dumps({"shape": DEFAULT.shape,
                          "accuracy": "passed", "max_rank_kernel_ms": latency},
                         indent=2, sort_keys=True), flush=True)

    for graph in graphs.values():
        graph.reset()
    graphs.clear()
    del graph
    del (outputs, cases, snapshots, stats, expected, routed_weights,
         shared_weights, output, values, increment)
    _close_case(torch, dist, gin, reference, owner)
    del gin, reference, owner
    gc.collect()

    if args.capability_matrix:
        capability_results = {}
        for config, count, mode in CAPABILITY_CASES:
            capability_results[config.name] = _run_capability_case(
                torch, dist, deep_gemm, provider, rank,
                config, count, mode, dense=args.dense_accuracy)
        capability_results["buffer_reuse"] = _run_reuse_case(
            torch, dist, deep_gemm, provider, rank)
        if rank == 0:
            print(json.dumps(
                {"capability_matrix": capability_results},
                indent=2, sort_keys=True), flush=True)

    if registration is not None: registration.unregister()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eager-generations", type=int, default=2)
    parser.add_argument("--graph-replays", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--benchmark-replays", type=int, default=50)
    parser.add_argument(
        "--capability-matrix", action="store_true",
        help="run the bounded non-default correctness matrix after benchmarking")
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--lsa-size", type=int, default=4)
    parser.add_argument("--dense-accuracy", action="store_true")
    args = parser.parse_args()
    if min(args.eager_generations, args.graph_replays,
           args.warmup, args.benchmark_replays) <= 0:
        parser.error("all iteration counts must be positive")
    _run(args)


if __name__ == "__main__":
    main()
