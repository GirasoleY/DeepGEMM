"""Untimed GB200 topology/peer-memory preflight; no new CUDA kernel.

Run with four torchrun hosts, four workers each. ``--mode gin`` expects two
real eight-rank NCCL LSA teams; ``--mode native`` expects one sixteen-rank
team. Environment settings are recorded, never accepted as topology evidence.
Explicit ``--world-size 8`` uses two four-worker hosts, GIN LSA2x4/native LSA8.
The EP8 GIN clique is one physical host; its mapped-peer test makes no
cross-OS same-clique claim. Native EP8 still exercises cross-OS mapped peers.
The tiny GIN context proves backend/resource creation, NOT device PUT traffic.
An optional reusable callback can supply that separate payload test; otherwise
the following full MegaMoE accuracy smoke must establish payload correctness.
"""

import argparse
import ctypes
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
NCCL_VERSION = 23007
WORLD = 16
WINDOW_ALIGNMENT = 4096
SOURCE_FILES = (
    "tests/probe_gb200_topology.py",
    "tests/mega_moe_gb200_topology.py",
    "tests/test_mega_moe_accuracy.py",
    "deep_gemm/mega/__init__.py",
    "csrc/apis/mega_gin.hpp",
    "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
)
ENV_FIELDS = (
    "NCCL_MNNVL_ENABLE", "NCCL_CUMEM_ENABLE", "NCCL_MNNVL_CLIQUE_ID",
    "NCCL_MNNVL_UUID", "NCCL_MNNVL_CROSS_CLIQUE", "NCCL_LSA_TEAM_SIZE",
    "NCCL_GIN_TYPE", "NCCL_GIN_ENABLE", "NCCL_IB_HCA",
)


class NcclProperties23007(ctypes.Structure):
    """Exact nccl_device/core.h ncclCommProperties ABI in NCCL 2.30.7.

    Never call this structure against a different NCCL version. In particular,
    an environment-selected libnccl must not be mixed with a foreign PG DSO.
    """
    _fields_ = [
        ("size", ctypes.c_size_t), ("magic", ctypes.c_uint),
        ("version", ctypes.c_uint), ("rank", ctypes.c_int),
        ("nRanks", ctypes.c_int), ("cudaDev", ctypes.c_int),
        ("nvmlDev", ctypes.c_int), ("deviceApiSupport", ctypes.c_bool),
        ("multimemSupport", ctypes.c_bool), ("ginType", ctypes.c_int),
        ("nLsaTeams", ctypes.c_int), ("hostRmaSupport", ctypes.c_bool),
        ("railedGinType", ctypes.c_int),
    ]


def source_manifest():
    result = {}
    for name in SOURCE_FILES:
        path = ROOT / name
        if not path.is_file():
            raise RuntimeError(f"missing preflight source: {name}")
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def validate_world_size(world_size):
    if type(world_size) is not int or world_size not in (8, 16):
        raise ValueError("GB200 world size must be exactly 8 or 16")
    return world_size


def expected_peers(rank, mode, world_size=WORLD):
    validate_world_size(world_size)
    if mode not in ("gin", "native") or type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError("mode must be gin/native and rank must be in the selected world")
    width = world_size // 2 if mode == "gin" else world_size
    start = rank // width * width
    return list(range(start, start + width))


def validate_records(records, mode, world_size=WORLD):
    """Validate real physical placement, NCCL properties and observed aliases."""
    validate_world_size(world_size)
    expected_peers(0, mode, world_size)
    if len(records) != world_size or [r.get("rank") for r in records] != list(range(world_size)):
        raise AssertionError("probe requires exactly one ordered record per selected world rank")
    hosts = [r["hostname"] for r in records]
    if len(set(hosts)) != world_size // 4:
        raise AssertionError("probe requires four distinct physical hostnames for EP16, two for EP8")
    for node in range(world_size // 4):
        members = records[4 * node:4 * node + 4]
        if len({r["hostname"] for r in members}) != 1:
            raise AssertionError("physical hosts must occupy contiguous four-rank blocks")
        if [r["local_rank"] for r in members] != [0, 1, 2, 3]:
            raise AssertionError("each physical host needs local ranks 0,1,2,3")
        if any(r["local_world_size"] != 4 for r in members):
            raise AssertionError("LOCAL_WORLD_SIZE must be four on every rank")
    for rank, record in enumerate(records):
        props = record["nccl"]
        if record["world_size"] != world_size or props["version"] != NCCL_VERSION:
            raise AssertionError("world size or NCCL version differs from the fixed target")
        if props["rank"] != rank or props["n_ranks"] != world_size:
            raise AssertionError("queried NCCL communicator rank mapping is wrong")
        if props["cuda_device"] != record["cuda_device"]:
            raise AssertionError("queried NCCL communicator uses another CUDA device")
        if props["device_api_support"] is not True:
            raise AssertionError("actual NCCL communicator lacks device API support")
        if props["n_lsa_teams"] != (2 if mode == "gin" else 1):
            raise AssertionError("actual NCCL LSA team count does not match requested mode")
        if record["peer_alias_ranks"] != expected_peers(rank, mode, world_size):
            raise AssertionError(f"actual peer aliases disagree with LSA membership at rank{rank}")
        if mode == "gin" and props["gin_type"] != 3:
            raise AssertionError("actual NCCL backend is not GDAKI")
        peers = expected_peers(rank, mode, world_size)
        if (mode == "native" or world_size == 16) and not any(
                hosts[peer] != hosts[rank] for peer in peers):
            raise AssertionError("no same-clique cross-OS-host peer exists")
    return {
        "physical_host_count": world_size // 4, "ranks_per_physical_host": 4,
        "world_size": world_size, "lsa_team_count": 2 if mode == "gin" else 1,
        "lsa_team_size": world_size // 2 if mode == "gin" else world_size,
        "same_clique_cross_os_peers_required": mode == "native" or world_size == 16,
        "actual_properties_and_aliases_checked": True,
        "environment_is_not_topology_proof": True,
    }


def validate_context_records(records, world_size=WORLD):
    validate_world_size(world_size)
    if len(records) != world_size or [r.get("rank") for r in records] != list(range(world_size)):
        raise AssertionError("GIN context evidence must include all selected world ranks")
    for rank, record in enumerate(records):
        if (record["rank"], record["world_size"], record["lsa_rank"], record["lsa_size"]) != (
                rank, world_size, rank % (world_size // 2), world_size // 2):
            raise AssertionError("actual GIN device communicator topology differs")
        if record["gin_type_string"] != "gdaki" or record["context_count"] != 9:
            raise AssertionError("actual GDAKI/context count differs")
        if record["queue_depth"] != 64 or record["signal_count"] < 2:
            raise AssertionError("actual GIN queue/signals are insufficient")
        if record["connection_count"] < 1:
            raise AssertionError("GIN device communicator has no connections")


def _phase(dist, name, action):
    """Gather rank-local errors before any subsequent collective phase."""
    value, error = None, None
    try:
        value = action()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    if any(item is not None for item in errors):
        raise RuntimeError(f"topology probe phase {name} failed: {errors}")
    return value


def _gather(dist, value):
    records = [None] * dist.get_world_size()
    dist.all_gather_object(records, value)
    return records


def _query_nccl(torch, dist):
    # Resolve through the Torch CUDA DSO that owns ProcessGroupNCCL rather than
    # loading an arbitrary second NCCL library from an environment path.
    dso_path = Path(torch.__file__).resolve().parent / "lib/libtorch_cuda.so"
    if not dso_path.is_file():
        raise RuntimeError(f"Torch CUDA DSO unavailable: {dso_path}")
    dso = ctypes.CDLL(str(dso_path))
    get_version = dso.ncclGetVersion
    get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_version.restype = ctypes.c_int
    version = ctypes.c_int()
    if get_version(ctypes.byref(version)) != 0 or version.value != NCCL_VERSION:
        raise RuntimeError(f"exact NCCL2.30.7 required, queried {version.value}")
    device = torch.device("cuda", torch.cuda.current_device())
    comm = int(dist.group.WORLD._get_backend(device)._comm_ptr())
    if not comm:
        raise RuntimeError("ProcessGroupNCCL returned a null communicator")
    props = NcclProperties23007()
    props.size, props.magic, props.version = ctypes.sizeof(props), 0xCAFEBEEF, NCCL_VERSION
    query = dso.ncclCommQueryProperties
    query.argtypes = [ctypes.c_void_p, ctypes.POINTER(NcclProperties23007)]
    query.restype = ctypes.c_int
    result = query(ctypes.c_void_p(comm), ctypes.byref(props))
    if result:
        raise RuntimeError(f"ncclCommQueryProperties returned {result}")
    return {
        "version": version.value, "rank": props.rank, "n_ranks": props.nRanks,
        "cuda_device": props.cudaDev, "device_api_support": bool(props.deviceApiSupport),
        "n_lsa_teams": props.nLsaTeams, "gin_type": props.ginType,
        "multimem_support": bool(props.multimemSupport),
        "query_dso": str(dso_path), "properties_abi_bytes": ctypes.sizeof(props),
    }


def sentinel(epoch, source, destination, slot):
    return (epoch + 1) * 1_000_000 + (source + 1) * 10_000 + (destination + 1) * 100 + slot


def _check_values(actual, expected, label):
    if list(actual) != list(expected):
        raise AssertionError(f"{label} sentinel mismatch: actual={actual}, expected={expected}")


def _retire_context(context, success):
    """Try registration release even when collective destroy fails."""
    errors = []
    if context is None:
        return errors
    try:
        context.destroy() if success else context.abort()
    except Exception as exc:
        errors.append(f"context retirement: {type(exc).__name__}: {exc}")
        try:
            context.abort()
        except Exception as abort_error:
            errors.append(f"context abort: {abort_error}")
    try:
        context._release_buffer_registration()
    except Exception as exc:
        errors.append(f"context registration release: {exc}")
    return errors


def _peer_memory_test(mode, torch, dist, allocation, handle, records):
    rank = dist.get_rank()
    world_size = validate_world_size(dist.get_world_size())
    peers = expected_peers(rank, mode, world_size)
    own = allocation[:world_size * 8].view(torch.int64)
    offset = int(handle.offset)
    views = {}
    def prepare():
        if offset < 0 or offset % own.element_size() or own.element_size() != 8:
            raise AssertionError("byte allocation offset must align to int64 sentinel elements")
        for peer in peers:
            # get_buffer's storage_offset is in requested dtype elements;
            # handle.offset is bytes. Check the resulting address before use.
            view = handle.get_buffer(peer, (world_size,), dtype=torch.int64,
                                     storage_offset=offset // own.element_size())
            raw = int(handle.buffer_ptrs[peer])
            expected = raw + offset
            if raw <= 0 or expected + world_size * 8 > 1 << 64:
                raise AssertionError("mapped sentinel address range must fit positive uint64 storage")
            if view.data_ptr() != expected:
                raise AssertionError("get_buffer/offset semantics disagree with MegaMoE's pointer contract")
            views[peer] = view
        # A self mapping may have another VA. The following all-peer writes
        # observed through ORIGINAL own, and original writes observed through
        # every mapped view (including self), prove shared bytes, not VA equality.
        own.fill_(-1)
        torch.cuda.synchronize()
    _phase(dist, "prepare_peer_views", prepare)
    def write():
        for peer, view in views.items():
            view[rank:rank + 1].fill_(sentinel(0, rank, peer, rank))
        torch.cuda.synchronize()
    _phase(dist, "same_lsa_peer_writes", write)
    _phase(dist, "verify_peer_writes", lambda: _check_values(
        own.cpu().tolist(),
        [sentinel(0, peer, rank, peer) if peer in peers else -1 for peer in range(world_size)],
        "peer-write"))
    def prepare_reads():
        own.copy_(torch.tensor([sentinel(1, rank, rank, slot) for slot in range(world_size)],
                               dtype=torch.int64, device=own.device))
        torch.cuda.synchronize()
    _phase(dist, "publish_read_sentinels", prepare_reads)
    def read():
        for peer, view in views.items():
            _check_values(view.clone().cpu().tolist(),
                          [sentinel(1, peer, peer, slot) for slot in range(world_size)], "peer-read")
    _phase(dist, "same_lsa_peer_reads", read)
    cross_os = [peer for peer in peers if records[peer]["hostname"] != records[rank]["hostname"]]
    return {"status": "passed", "read_peers": peers, "write_peers": peers,
            "cross_os_peers_tested": cross_os, "words_per_read": world_size,
            "original_self_pointer": int(own.data_ptr()),
            "mapped_self_pointer": int(views[rank].data_ptr()),
            "self_virtual_addresses_equal": views[rank].data_ptr() == own.data_ptr(),
            "get_buffer_storage_offset_elements": offset // own.element_size(),
            "symmetric_allocation_offset_bytes": offset,
            "original_to_mapped_and_mapped_to_original_checked": True,
            "intra_kernel_alias_proxy_ordering_claim": False,
            "independent_writer_slots": True, "path": "mapped_peer_memory_torch_cuda_ops",
            "gin_payload_test": False}


def run_topology_probe(mode, *, torch, dist, deep_gemm=None,
                       validate_gin_context=True, gin_payload_probe=None, world_size=WORLD):
    """Reusable preflight on initialized WORLD, before benchmark allocations.

    Owns/releases a temporary registration and allocation, never destroys WORLD.
    Do not call while another symmetric-memory registration for WORLD is live.
    A supplied payload callback runs collectively after context creation and
    must return ``status=passed, backend=gdaki, actual_payload_checked=True``.
    The default does not claim to exercise GIN payloads.
    """
    import test_mega_moe_accuracy as accuracy
    import torch.distributed._symmetric_memory as symm_mem
    rank = dist.get_rank()
    configuration = {"mode": mode, "world_size": world_size,
                     "validate_gin_context": bool(validate_gin_context),
                     "has_payload_callback": gin_payload_probe is not None}
    configurations = _gather(dist, configuration)
    if any(item != configuration for item in configurations):
        raise RuntimeError("topology probe configuration differs across ranks")
    validate_world_size(world_size)
    if mode not in ("gin", "native") or dist.get_world_size() != world_size:
        raise RuntimeError("topology probe requires mode gin/native and the selected WORLD")
    if gin_payload_probe is not None and (mode != "gin" or not validate_gin_context):
        raise ValueError("payload callback requires GIN mode with a validated context")
    manifest = _phase(dist, "source_manifest", source_manifest)
    manifests = _gather(dist, manifest)
    if any(item != manifest for item in manifests):
        raise RuntimeError("preflight source hashes differ across ranks")
    registration = allocation = handle = context = gin_buffer = None
    success = False
    try:
        # Force the existing tested NCCL allocator/registry bridge in BOTH
        # modes; require_gin here selects allocation, not the MegaMoE kernel.
        _, registration = accuracy._configure_symmetric_memory_backend(
            SimpleNamespace(require_gin=True), torch, dist)
        nccl = _phase(dist, "query_nccl_properties", lambda: _query_nccl(torch, dist))
        device = torch.device("cuda", torch.cuda.current_device())
        # Reserve room for a 4096-aligned GIN window even if Torch prepends a
        # signal pad. The aligned slice remains owned until context retirement.
        allocation = _phase(dist, "allocate", lambda: symm_mem.empty(
            3 * WINDOW_ALIGNMENT, dtype=torch.uint8, device=device))
        handle = symm_mem.rendezvous(allocation, group=dist.group.WORLD)
        props = torch.cuda.get_device_properties(device)
        record = {
            "rank": rank, "world_size": dist.get_world_size(), "hostname": socket.gethostname(),
            "local_rank": int(os.environ["LOCAL_RANK"]),
            "local_world_size": int(os.environ.get("LOCAL_WORLD_SIZE", "0")),
            "cuda_device": torch.cuda.current_device(), "device_name": props.name,
            "capability": list(torch.cuda.get_device_capability()), "num_sms": props.multi_processor_count,
            "nccl": nccl, "peer_alias_ranks": [i for i, ptr in enumerate(handle.buffer_ptrs) if int(ptr)],
            "symmetric_offset": int(handle.offset), "allocation_bytes": allocation.numel(),
            "environment_not_proof": {key: os.getenv(key) for key in ENV_FIELDS},
        }
        records = _gather(dist, record)
        topology = validate_records(records, mode, world_size)
        memory = _peer_memory_test(mode, torch, dist, allocation, handle, records)
        memory_records = _gather(dist, memory)
        contexts = None
        if mode == "gin" and validate_gin_context:
            def build_check():
                if deep_gemm is None:
                    raise RuntimeError("GIN context check needs the built DeepGEMM extension")
                info = deep_gemm._C.megamoe_gin_build_info()
                if not info["enabled"] or info["compiled_nccl_version"] != NCCL_VERSION:
                    raise RuntimeError("GIN context requires a matching NCCL2.30.7 extension")
                return info
            _phase(dist, "gin_build", build_check)
            uid = _phase(dist, "gin_uid", lambda: deep_gemm._C.get_megamoe_gin_unique_id()
                         if rank == 0 else bytes(128))
            uid_list = [uid]
            dist.broadcast_object_list(uid_list, src=0)
            skip = (-allocation.data_ptr()) % WINDOW_ALIGNMENT
            gin_buffer = allocation[skip:skip + WINDOW_ALIGNMENT]
            context = deep_gemm._C.create_megamoe_gin_context(
                gin_buffer, uid_list[0], rank, world_size, context_count=9,
                queue_depth=64, world_barrier_count=4, expected_lsa_size=world_size // 2,
                required_gin_type="gdaki")
            context_record = {name: getattr(context, name) for name in (
                "rank", "world_size", "lsa_rank", "lsa_size", "gin_type_string",
                "context_count", "queue_depth", "signal_count", "connection_count")}
            contexts = _gather(dist, context_record)
            validate_context_records(contexts, world_size)
        payload = {"status": "not_run", "actual_payload_checked": False,
                   "reason": "no standalone device PUT export; full MegaMoE accuracy smoke is required"}
        if gin_payload_probe is not None:
            payload = gin_payload_probe(context, gin_buffer, torch, dist)
            def payload_check():
                if (payload.get("status") != "passed" or payload.get("backend") != "gdaki"
                        or payload.get("actual_payload_checked") is not True):
                    raise AssertionError("GIN callback did not attest checked GDAKI payloads")
            _phase(dist, "gin_payload_callback", payload_check)
        payloads = _gather(dist, payload)
        _phase(dist, "source_stability", lambda: _check_values(
            sorted(source_manifest().items()), sorted(manifest.items()), "source"))
        torch.cuda.synchronize()
        dist.barrier()
        success = True
        result = {"status": "passed", "scope": "topology_and_mapped_peer_memory",
                  "mode": mode, "source_sha256": manifest, "ranks": records,
                  "topology": topology, "peer_memory": memory_records,
                  "gin_contexts": contexts, "gin_payload": payloads,
                  "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                  "timing_claim": False, "megamoe_accuracy_smoke_required": True}
    finally:
        cleanup_errors = _retire_context(context, success)
        # Release the NCCL window handle while WORLD and its registry still
        # exist. On a broken distributed job launcher timeout remains necessary.
        gin_buffer = context = handle = allocation = None
        try:
            accuracy._unregister_symmetric_memory_comm(registration, suppress_errors=not success)
        except Exception as exc:
            cleanup_errors.append(f"allocator registration release: {exc}")
        if cleanup_errors and success:
            raise RuntimeError(f"topology probe cleanup failed: {cleanup_errors}")
    result["temporary_resources_retired"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("gin", "native"), required=True)
    parser.add_argument("--world-size", type=int, choices=(8, 16), default=WORLD)
    parser.add_argument("--skip-gin-context", action="store_true",
                        help="explicit topology-only mode; no GIN context/payload claim")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()
    import torch
    import torch.distributed as dist
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("launch using selected EP8/EP16 world, four ranks per host")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", init_method="env://",
                            timeout=timedelta(seconds=args.timeout_seconds))
    rank = dist.get_rank()
    try:
        dg = None
        if args.mode == "gin" and not args.skip_gin_context:
            dg = _phase(dist, "import_deep_gemm", lambda: __import__("deep_gemm"))
        result = run_topology_probe(args.mode, torch=torch, dist=dist, deep_gemm=dg,
                                    validate_gin_context=not args.skip_gin_context,
                                    world_size=args.world_size)
    except Exception as exc:
        # Do not enter fresh teardown collectives after a failed distributed
        # probe. torchrun's bounded launcher owns timeout/peer termination.
        if rank == 0:
            print(json.dumps({"GB200_TOPOLOGY_PREFLIGHT": "failed", "mode": args.mode,
                              "error": f"{type(exc).__name__}: {exc}"}), flush=True)
        raise
    dist.barrier()
    dist.destroy_process_group()
    result["process_group_retired"] = True
    if rank == 0:
        if args.output:
            with args.output.open("x") as stream:
                json.dump(result, stream, sort_keys=True, indent=2)
                stream.write("\n")
        print(json.dumps({"GB200_TOPOLOGY_PREFLIGHT": result["status"],
                          "mode": args.mode, "peer_memory": "passed",
                          "gin_payload": "not_run", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
