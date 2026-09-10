"""Matched EP16 GB200 transport experiment; run each mode in a fresh job.

Four physical hosts, four torchrun workers/host; repeat MNS8/10/12:
  python tests/bench_gb200_transport_matched.py --mode native_nvl \
      --decode-mns 8 --output native-mns8.json
  python tests/bench_gb200_transport_matched.py --mode gin_ib \
      --decode-mns 8 --profile-recipe --output gin-mns8.json
  python tests/bench_gb200_transport_matched.py --mode gin_roce \
      --decode-mns 8 --profile-recipe --output roce-mns8.json

Logical route domains remain ranks0..7/ranks8..15, exactly as in the Novita
fixture. They are NOT physical hostnames. Native mode requires all sixteen
actual mapped peers, including cross-OS-host read/write sentinels, before
MegaMoE compute. GIN modes require two actual eight-rank LSA teams and GDAKI.
gin_ib requests InfiniBand; gin_roce requests RoCE/Ethernet. Neither the mode
label nor GDAKI alone proves the physical link layer or selected payload path.

GIN uses the existing full MegaMoE/DeepEP+TRT benchmark pair unchanged. Native
times only MegaMoE, with the same backend-isolated CUDA-event convention.
Inputs/weights are sparse-valued full-shape fixtures, not random-dense timing.
This is not a production trace or a compute-only floor measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import test_mega_moe_accuracy as accuracy
import bench_deepep_trtllm_isolated as matched
from gb200_mxfp4_compat import prepare_converter_adapter


ROOT = Path(__file__).resolve().parents[1]
GIN_MODES = ("gin_ib", "gin_roce")
MODES = ("native_nvl", *GIN_MODES)
SOURCE_FILES = (
    "tests/bench_gb200_transport_matched.py",
    "tests/gb200_mxfp4_compat.py",
    "tests/mega_moe_gb200_topology.py", "tests/probe_gb200_topology.py",
    "tests/test_mega_moe_accuracy.py", "tests/bench_deepep_trtllm_isolated.py",
    "deep_gemm/__init__.py", "deep_gemm/mega/__init__.py", "deep_gemm/utils/math.py",
    "csrc/apis/mega.hpp", "csrc/apis/mega_gin.hpp", "csrc/jit/compiler.hpp",
    "csrc/jit_kernels/heuristics/mega_moe.hpp",
    "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp",
    "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
    "deep_gemm/include/deep_gemm/comm/mega_moe_gin.cuh",
    "deep_gemm/include/deep_gemm/layout/mega_moe.cuh",
    "deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--decode-mns", type=int, choices=(8, 10, 12), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison-replays", type=int, default=204)
    parser.add_argument("--comparison-conditioning", type=int, default=60)
    parser.add_argument("--comparison-warmups", type=int, default=10)
    parser.add_argument("--eager-iterations", type=int, default=16)
    parser.add_argument("--graph-replays", type=int, default=32)
    parser.add_argument("--payload-epochs", type=int, default=3)
    parser.add_argument("--profile-recipe", action="store_true")
    options = parser.parse_args(argv)
    if options.comparison_replays < 2:
        parser.error("at least two timed replays are required for split summaries")
    if min(options.comparison_conditioning, options.comparison_warmups,
           options.eager_iterations, options.graph_replays) < 1:
        parser.error("conditioning, warmups and correctness iterations must be positive")
    if options.payload_epochs < 2:
        parser.error("changing-payload correctness must remain enabled")
    if options.profile_recipe and options.mode not in GIN_MODES:
        parser.error("recipe profiling applies only to the GIN + DeepEP/TRT job")
    common = ["--k3", "--decode-mns", str(options.decode_mns),
              "--eager-iterations", str(options.eager_iterations),
              "--graph-replays", str(options.graph_replays),
              "--payload-epochs", str(options.payload_epochs), "--require-cross-host",
              "--no-fast-math", "--activation-clamp", "10",
              "--gin-completion-batch", "8", "--gin-combine-issue-wave", "8",
              "--gin-outbox-depth", "64", "--gin-combine-chunk-bytes", "7168"]
    if options.mode in GIN_MODES:
        common += ["--require-gin", "--gin-active-fast-path", "--gin-bulk-combine",
                   "--gin-direct-dispatch"]
    with patch.object(sys, "argv", [sys.argv[0], *common]):
        args = accuracy._parse_args()
    comparison = SimpleNamespace(
        output=options.output, comparison_replays=options.comparison_replays,
        comparison_conditioning=options.comparison_conditioning,
        comparison_warmups=options.comparison_warmups, comparison_rel_l2=0.02,
        comparison_max_abs=0.02, component_replays=0, compare_preparation=False,
        deepep_capacity=384, deepep_dispatch_bucket=args.num_tokens,
        deepep_num_sms=16, deepep_num_qps=9, profile_recipe=options.profile_recipe,
        dispatch_bucket_evidence=matched.dispatch_bucket_evidence(
            args.num_tokens, 384, args.num_tokens),
    )
    return options, args, comparison


def fixed_environment(mode):
    if mode not in MODES:
        raise ValueError("mode must be native_nvl, gin_ib or gin_roce")
    enabled = "1" if mode in GIN_MODES else "0"
    return {
        **{name: enabled for name in accuracy.GIN_VALIDATED_FLAG_ENVS},
        accuracy.GIN_ACTIVITY_GATE_OPT_ENV: enabled,
        "DG_MEGAMOE_GIN_DIAGNOSTICS": "0",
        accuracy.GIN_LOCAL_ABLATION_ENV: "0",
        "DG_MEGAMOE_GIN_COMBINE_EXPERTS_PER_WAVE": "0",
        "DG_MEGAMOE_GIN_COMBINE_BARRIER_WARPS": "1",
    }


def network_transport_metadata(mode):
    """Requested physical transport; NIC/link/payload evidence stays separate."""
    if mode not in MODES:
        raise ValueError("unknown GB200 transport mode")
    label, link_layer = {
        "native_nvl": ("NVLink", None),
        "gin_ib": ("InfiniBand", "InfiniBand"),
        "gin_roce": ("RoCE", "Ethernet"),
    }[mode]
    return {
        "requested_transport": label, "requested_rdma_link_layer": link_layer,
        "observed_rdma_link_layer": None,
        "physical_network_payload_verified": False,
        "basis": "explicit mode request, not measured NIC/link-layer/payload evidence",
        "gin_protocol_and_topology_checks_shared_between_ib_and_roce": mode in GIN_MODES,
    }


def source_manifest():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


def _load_runtime_for_gb200(local_rank):
    """Select the local device before importing the expected in-tree runtime.

    Do not evict an already loaded foreign package: reject it. This check is
    before process-group initialization, so launcher-wide failure handling is
    required for a missing runtime, just as with the original runtime loader.
    """
    torch = importlib.import_module("torch")
    torch.cuda.set_device(local_rank)
    dist = importlib.import_module("torch.distributed")
    with patch.object(sys, "path", [str(ROOT), *sys.path]):
        deep_gemm = importlib.import_module("deep_gemm")
    checkout = ROOT.resolve()
    expected = checkout / "deep_gemm"
    if Path(deep_gemm.__file__).resolve() != expected / "__init__.py":
        raise RuntimeError("GB200 benchmark imported a foreign deep_gemm Python package")
    extension = Path(deep_gemm._C.__file__).resolve()
    # develop.sh links deep_gemm/_C*.so to build/lib*/deep_gemm/_C*.so.
    # Resolve the target, not the allowed build directory: a build symlink
    # into another checkout must not silently broaden this allowlist.
    if extension.parent != expected and not extension.is_relative_to(checkout / "build"):
        raise RuntimeError("GB200 benchmark requires an extension built in the selected checkout")
    return torch, dist, deep_gemm


def configuration(options, args, comparison, sources):
    """CPU-only, all-rank equality checked by topology.prepare before compute."""
    return {
        "mode": options.mode, "world_size": 16, "physical_hosts": 4,
        "network_transport": network_transport_metadata(options.mode),
        "ranks_per_physical_host": 4, "logical_route_domains": [0] * 8 + [1] * 8,
        "shape": {"tokens_per_rank": args.num_tokens, "capacity": 384,
                  "hidden": args.hidden, "intermediate_hidden": args.intermediate_hidden,
                  "num_experts": args.num_experts, "num_topk": args.num_topk},
        "iterations": {"replays": comparison.comparison_replays,
                       "conditioning": comparison.comparison_conditioning,
                       "warmups": comparison.comparison_warmups,
                       "eager": args.eager_iterations, "graph": args.graph_replays,
                       "payload_epochs": args.payload_epochs},
        "profile_recipe": comparison.profile_recipe,
        "deepep": {"num_sms": 16, "num_qps": 9, "capacity": 384,
                   **comparison.dispatch_bucket_evidence},
        "megamoe_environment": fixed_environment(options.mode),
        "math": {"fast_math": False, "activation_clamp": 10.0,
                 "compute_hint": None, "compute_configuration_override": None},
        "source_sha256": sources,
    }


def _gather(dist, value):
    records = [None] * dist.get_world_size()
    dist.all_gather_object(records, value)
    return records


def _collective_call(dist, label, action):
    value, error = None, None
    try:
        value = action()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = _gather(dist, error)
    if any(item is not None for item in errors):
        raise RuntimeError(f"{label} failed collectively: {errors}")
    return value


def _tensor_sha256(tensor, torch):
    flat = tensor.contiguous().view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    for offset in range(0, flat.numel(), 16 * 1024 * 1024):
        digest.update(flat[offset:offset + 16 * 1024 * 1024].cpu().numpy().tobytes())
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "sha256": digest.hexdigest()}


def _fixture_identity(harness, snapshots):
    """Full bytes, not sampled sums; compare ranks by rank across fresh jobs."""
    tensors = {"x": harness.inputs.x, "x_sf": harness.inputs.x_sf}
    for name in accuracy.MATCHED_BENCHMARK_ROUTES:
        tensors[name + "/topk_idx"] = harness.cases[name].topk_idx
        tensors[name + "/topk_weights"] = harness.cases[name].topk_weights
        tensors[name + "/native_reference_snapshot"] = snapshots[name]
    for layer in ("l1", "l2"):
        for index, tensor in enumerate(getattr(harness.weights, layer)):
            tensors[f"{layer}/{index}"] = tensor
    return {"rank": harness.rank,
            "tensors": {name: _tensor_sha256(value, harness.torch)
                        for name, value in tensors.items()}}


def _benchmark_native(harness, snapshots, comparison):
    """Same isolated event train as _benchmark_pair; no transport inside wrapper."""
    torch, dist = harness.torch, harness.dist
    harness.copy_inputs(harness.cases["all_same_host"])
    graphs = {"megamoe": matched._capture(harness.launch, torch, dist)}
    routes = {}
    try:
        graph = graphs["megamoe"]
        for name in accuracy.MATCHED_BENCHMARK_ROUTES:
            case = harness.cases[name]
            harness.copy_inputs(case)
            for _ in range(comparison.comparison_warmups):
                graph.replay()
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
            def validate():
                harness.assert_stats(case.expected_local_stats * comparison.comparison_replays,
                                     f"native-isolated/{name}")
                harness.assert_guards(f"native-isolated/{name}")
                accuracy._assert_bitwise_equal(harness.output, snapshots[name],
                                              f"native-isolated/{name}", torch)
            _collective_call(dist, f"native-isolated/{name}", validate)
            routes[name] = {"megamoe": {"isolated": accuracy._distributed_timing_record(
                [float(start.elapsed_time(end) * 1000.0) for start, end in pairs], torch, dist)}}
        return routes
    finally:
        matched._retire_graphs(graphs)


def _deepep_domain_record(comparator, rank):
    """Read the separate DeepEP runtime, not MegaMoE's topology descriptor.

    Public tuple order is (RDMA, NVLink) and (scaleout, scaleup). Installed
    getter text and source identity are retained for the required version
    review; historical Novita or current upstream docs are not that proof.
    """
    buffer = comparator.buffer
    physical = tuple(buffer.get_physical_domain_size())
    logical = tuple(buffer.get_logical_domain_size())
    if (any(type(value) is not int for value in (*physical, *logical))
            or physical != (2, 8) or logical != (1, 16)):
        raise AssertionError(f"DeepEP requires physical(2,8), logical(1,16); got {physical}, {logical}")
    if buffer.allow_hybrid_mode is not False:
        raise AssertionError("DeepEP must use the actual non-hybrid/direct mode")
    named = {name: getattr(buffer, name) for name in (
        "num_rdma_ranks", "num_nvlink_ranks", "num_scaleout_ranks", "num_scaleup_ranks")}
    if tuple(named.values()) != (*physical, *logical):
        raise AssertionError("DeepEP named domain counts disagree with live getter tuples")
    if (comparator.api_evidence["physical_domain_size"] != list(physical)
            or comparator.api_evidence["logical_domain_size"] != list(logical)
            or comparator.api_evidence["allow_hybrid_mode"] is not False):
        raise AssertionError("DeepEP recorded API evidence disagrees with live runtime")
    source = Path(inspect.getsourcefile(type(buffer))).resolve()
    return {
        "rank": rank, "physical_domain_size": list(physical),
        "logical_domain_size": list(logical), "allow_hybrid_mode": False,
        "named_domain_counts": named,
        "loaded_python_source": str(source),
        "loaded_python_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "loaded_domain_getter_source": {
            name: inspect.getsource(getattr(type(buffer), name))
            for name in ("get_physical_domain_size", "get_logical_domain_size")},
    }


def _validate_deepep_domains(harness, comparator):
    record = _collective_call(harness.dist, "DeepEP actual domain preflight",
                              lambda: _deepep_domain_record(comparator, harness.rank))
    records = _gather(harness.dist, record)
    if len({item["loaded_python_source_sha256"] for item in records}) != 1:
        raise RuntimeError("DeepEP loaded Python source differs across ranks")
    return {
        "status": "passed", "before_graph_capture": True,
        "expected_physical_domain_size_rdma_nvlink": [2, 8],
        "expected_logical_domain_size_scaleout_scaleup": [1, 16],
        "direct_nonhierarchical": True, "per_rank": records,
        "installed_tuple_semantics_source_review_required_before_acceptance": True,
        "physical_ib_payload_probe": False,
        "scope": "live DeepEP domain/mode values and Python source, not NIC traffic measurement",
    }


def _compare_gin(harness, snapshots, comparison):
    adapter = _collective_call(harness.dist, "GB200 MXFP4 converter compatibility", prepare_converter_adapter)
    converter_records = _gather(harness.dist, adapter.evidence)
    if any(record != adapter.evidence for record in converter_records):
        raise RuntimeError("GB200 MXFP4 converter selection/source differs across ranks")
    # The imported converter is restored before graph capture and timing.
    with adapter.install():
        comparator = matched.DeepEPTRTLLM(harness, comparison)
    try:
        domains = _validate_deepep_domains(harness, comparator)
        routes, payload = matched._benchmark_pair(harness, comparator, comparison, snapshots)
        return {"routes": routes, "same_graph_pair_payload_refresh": payload,
                "gb200_mxfp4_converter_compatibility": {"per_rank": converter_records,
                                                        "installed_only_during_construction": True},
                "deepep_actual_domain_preflight": domains,
                "runtime": comparator.api_evidence,
                "canonical_weights_sha256": comparator.weight_sha256,
                "megamoe_weights_verified_byte_identical": True,
                "recipe_kernel_profile": comparator.recipe_profile}
    finally:
        active_error = sys.exc_info()[0] is not None
        try:
            harness.torch.cuda.synchronize()
            comparator.last_receipt = None
            comparator.buffer.destroy()
        except Exception as cleanup_error:
            if not active_error:
                raise
            print(f"DeepEP cleanup also failed: {cleanup_error}", file=sys.stderr, flush=True)


def main():
    options, args, comparison = parse_args()
    if ("LOCAL_RANK" not in os.environ or os.getenv("LOCAL_WORLD_SIZE") != "4"
            or os.getenv("WORLD_SIZE") != "16"):
        raise RuntimeError("use a fresh four-host torchrun job with four workers/host (EP16)")
    matched.validate_clean_experiment_environment()
    os.environ.update(fixed_environment(options.mode))
    from mega_moe_gb200_topology import GB200Topology
    sources = source_manifest()
    config = configuration(options, args, comparison, sources)
    topology = GB200Topology(mode=options.mode, run_configuration=config)
    original_stress = accuracy._run_graph_stress
    original_success = accuracy._synchronize_worker_success
    state = {}

    def verify_final_sources(dist):
        def verify():
            if source_manifest() != sources:
                raise AssertionError("listed source changed before final worker success")
        _collective_call(dist, "final worker source stability", verify)
        return original_success(dist)

    def stress_then_measure(harness, snapshots, torch, dist):
        original_stress(harness, snapshots, torch, dist)
        before = _collective_call(dist, "fixture identity before", lambda: _fixture_identity(harness, snapshots))
        if options.mode in GIN_MODES:
            result = _compare_gin(harness, snapshots, comparison)
        else:
            result = {"routes": _benchmark_native(harness, snapshots, comparison)}
        after = _collective_call(dist, "fixture identity after", lambda: _fixture_identity(harness, snapshots))
        def check_stable():
            if before != after:
                raise AssertionError("timing changed fixture inputs, routes, weights or snapshots")
            if source_manifest() != sources:
                raise AssertionError("a listed source file changed during correctness/timing")
        _collective_call(dist, "fixture/source stability", check_stable)
        extension = Path(harness.deep_gemm._C.__file__).resolve()
        runtime = _collective_call(dist, "extension fingerprint", lambda: {
            "rank": harness.rank, "extension_path": str(extension),
            "extension_module_path": str(harness.deep_gemm._C.__file__),
            "deep_gemm_python_path": str(Path(harness.deep_gemm.__file__).resolve()),
            "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
            "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "num_sms": torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count,
        })
        state.update(result, fixture_identity=_gather(dist, before),
                     runtime_per_rank=_gather(dist, runtime),
                     physical_hostnames=list(harness.physical_hostnames),
                     logical_route_domains=list(harness.route_domains),
                     topology=harness.topology_evidence)

    rank = int(os.environ["RANK"])
    with patch.object(accuracy, "_run_graph_stress", stress_then_measure), patch.object(
            accuracy, "_synchronize_worker_success", verify_final_sources), patch.object(
            accuracy, "_load_runtime", lambda: _load_runtime_for_gb200(int(os.environ["LOCAL_RANK"]))):
        accuracy._worker(int(os.environ["LOCAL_RANK"]), 4, args, topology=topology)
    if rank != 0:
        return
    if topology.accuracy_result is None:
        raise AssertionError("worker returned without its complete accuracy evidence")
    record = {
        "schema": "gb200-ep16-matched-transport-v1", "status": "passed",
        "accuracy_and_teardown_passed": True, "configuration": config, **state,
        "accuracy": topology.accuracy_result,
        "operator": "SwiGLU", "input_pattern": "selected_periodic_sparse_valued_full_shape",
        "timing_scope": "prequantized_activation_and_routes_to_combined_bf16_output",
        "excluded": ["input_quantization", "route_generation", "source_buffer_copies"],
        "aggregation": "max_of_16_rank_cuda_event_durations_per_replay_then_quantiles",
        "schedule": ("route_isolated_alternating_backend_order_plus_backend_isolated_rewarm"
                     if options.mode in GIN_MODES else "native_backend_isolated_rewarm"),
        "historical_production_SITU_comparison": False,
        "compute_only_floor_claim": False, "random_dense_timing_claim": False,
        "physical_ib_payload_proven_by_environment": False,
        "network_transport": network_transport_metadata(options.mode),
        "megamoe_math": {"fast_math": False, "activation_clamp": 10.0,
                         "weight_position": "before_intermediate_mxfp8_quantization",
                         "intermediate_scale": "2**ceil(log2(amax/448)), per32 UE8M0"},
        "cross_job_fixture_equality": "requires comparing per-rank full-byte fixture_identity; not asserted by one job",
        "source_stability_checked": True,
        "source_attestation_scope": "listed source files plus per-rank extension hash; external headers/JIT binary not attested",
    }
    if options.mode in GIN_MODES:
        record.update(matched.dispatch_candidate_metadata(fixed_environment(options.mode)))
        record["quantization_boundary"] = {
            "intermediate_quantizers_identical": False,
            "trtllm_intermediate_scale": "2**(floor(log2(amax))-8), per32 UE8M0 with saturating FP8 RNE",
            "trtllm_activation_before_quantization": "FP32; no BF16 activation cast",
            "trtllm_weight_position": "expert_finalization",
            "recipe_evidence_scope": matched.TRT_RECIPE_EVIDENCE,
            "per_backend_correctness_gate": "own_recipe_relative_l2_and_max_abs",
            "cross_backend_gate": "relative_l2_drift_ceiling_only; old absolute outcome retained",
            "relative_l2_limit": 0.02, "max_abs_limit": 0.02,
        }
    with options.output.open("x") as output:
        json.dump(record, output, sort_keys=True)
        output.write("\n")
    print("GB200_TRANSPORT_MATCHED_JSON=" + json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
