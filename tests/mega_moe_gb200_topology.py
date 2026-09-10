"""Explicit GB200 harness policy: actual 4x4 hosts, logical 2x8 routes.

No device/kernel change. Native mode is admitted only after actual NCCL LSA16
and adjusted peer aliases plus a GPU sentinel on the REAL work allocation.
Both gin_ib and gin_roce require actual LSA2x8/GDAKI, not an environment label.
The mode does not prove InfiniBand versus RoCE NIC payloads. Legacy Novita
host placement checks are untouched. Imports are CPU-only until hook execution.
"""

from copy import deepcopy


WORLD = 16
MODES = {"gin_ib": "gin", "gin_roce": "gin", "native_nvl": "native"}


def validate_physical_hostnames(hostnames):
    hosts = tuple(hostnames)
    if len(hosts) != WORLD or any(type(host) is not str or not host for host in hosts):
        raise ValueError("GB200 requires sixteen nonempty actual hostname strings")
    if len(set(hosts)) != 4 or any(len(set(hosts[start:start + 4])) != 1
                                 for start in range(0, WORLD, 4)):
        raise ValueError("GB200 requires four real contiguous four-rank hosts")
    return hosts


def logical_route_domains(hostnames):
    """Rank group IDs, deliberately not fabricated hostnames."""
    validate_physical_hostnames(hostnames)
    return tuple(rank // 8 for rank in range(WORLD))


def validate_adjusted_aliases(raw, adjusted, offset, tensor_pointer, rank, mode, *, num_bytes=1):
    """Validate mapped addresses; physical self aliasing requires the GPU sentinel."""
    if mode not in MODES or type(rank) is not int or not 0 <= rank < WORLD:
        raise ValueError("unsupported mode or rank")
    if len(raw) != WORLD or len(adjusted) != WORLD:
        raise AssertionError("sixteen raw and adjusted buffer aliases required")
    if any(type(value) is not int or not 0 <= value < 1 << 64
           for value in (*raw, *adjusted, offset)):
        raise AssertionError("buffer pointers/offset must be nonnegative integers")
    if (type(tensor_pointer) is not int or tensor_pointer <= 0 or
            type(num_bytes) is not int or num_bytes <= 0 or
            any(value + num_bytes > 1 << 64
                for value in (*adjusted, tensor_pointer) if value)):
        raise AssertionError("work-buffer address range must fit positive uint64 storage")
    width = 8 if MODES[mode] == "gin" else WORLD
    expected = list(range(rank // width * width, (rank // width + 1) * width))
    observed = [peer for peer, value in enumerate(adjusted) if value]
    if observed != expected or [peer for peer, value in enumerate(raw) if value] != expected:
        raise AssertionError("actual raw/adjusted peer aliases do not match required LSA membership")
    if list(adjusted) != [value + offset if value else 0 for value in raw]:
        raise AssertionError("adjusted peer aliases must apply the symmetric allocation offset exactly once")
    return observed


def _buffer_alias_identity(buffer):
    """Process-local launch guard; not serialized source or binary attestation."""
    return (id(buffer.buffer), id(buffer.handle), int(buffer.buffer.data_ptr()),
            int(buffer.buffer.numel()), int(buffer.handle.offset),
            tuple(int(value) for value in buffer.handle.buffer_ptrs),
            tuple(int(value) for value in buffer.buffer_ptrs))


class GB200Topology:
    def __init__(self, mode, run_configuration=None):
        if mode not in MODES:
            raise ValueError("GB200 mode must be gin_ib, gin_roce or native_nvl")
        self.mode = mode
        self.run_configuration = deepcopy(run_configuration)
        self.physical_hostnames = None
        self.logical_route_domains = None
        self.evidence = None
        self.accuracy_result = None
        self._validated_buffer_id = None
        self._validated_alias_identity = None

    def prepare(self, args, hostnames, local_rank, local_world_size, torch, dist):
        from probe_gb200_topology import _phase, _gather

        def local_record():
            hosts = validate_physical_hostnames(hostnames)
            rank = dist.get_rank()
            if dist.get_world_size() != WORLD or local_world_size != 4 or local_rank != rank % 4:
                raise ValueError("GB200 baseline requires WORLD16 / four torchrun workers per real host")
            if torch.cuda.current_device() != local_rank:
                raise ValueError("CUDA device does not match local rank")
            if bool(args.require_gin) != (MODES[self.mode] == "gin"):
                raise ValueError("explicit GB200 mode and require_gin disagree")
            shape = (args.num_experts, args.num_topk, args.hidden, args.intermediate_hidden,
                     args.num_max_tokens_per_rank, args.num_shared_experts, args.mma_type)
            if shape != (896, 16, 3584, 3072, 384, 0, "fp8xfp4") or args.num_tokens not in (32, 40, 48):
                raise ValueError("GB200 baseline is scoped to exact matched K3 decode shapes/capacity")
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            if "GB200" not in props.name or torch.cuda.get_device_capability()[0] != 10:
                raise ValueError("actual device is not an SM100-family GB200")
            return {"mode": self.mode, "physical_hostnames": list(hosts),
                    "tokens": args.num_tokens, "shape": shape,
                    "run_configuration": self.run_configuration}

        configuration = _phase(dist, "gb200_configuration", local_record)
        configurations = _gather(dist, configuration)
        if any(item != configuration for item in configurations):
            raise RuntimeError("GB200 mode, shape, source or run configuration differs across ranks")
        self.physical_hostnames = validate_physical_hostnames(hostnames)
        self.logical_route_domains = logical_route_domains(hostnames)
        self._local_rank, self._local_world_size = local_rank, local_world_size

    def require_prepared(self, hostnames):
        if self.physical_hostnames is None or tuple(hostnames) != self.physical_hostnames:
            raise RuntimeError("GB200 actual placement has not been collectively validated")

    def expected_lsa_ranks(self, rank):
        from probe_gb200_topology import expected_peers
        if self.evidence is None:
            raise RuntimeError("actual GB200 LSA membership is not validated")
        return expected_peers(rank, MODES[self.mode])

    def validate_buffer(self, buffer, args, backend, registration, torch, dist):
        from probe_gb200_topology import (
            _phase, _gather, _query_nccl, _peer_memory_test,
            validate_records, validate_context_records,
        )
        self.require_prepared(self.physical_hostnames)
        if self._validated_buffer_id is not None:
            raise RuntimeError("work-buffer alias validation must precede its first compute only")
        rank = dist.get_rank()
        def record():
            if backend != "NCCL" or registration is None:
                raise AssertionError("both GB200 modes require the live NCCL symmetric allocator bridge")
            if bool(buffer.gin_enabled) != (MODES[self.mode] == "gin"):
                raise AssertionError("actual buffer GIN state differs from requested transport")
            if self.mode == "native_nvl" and getattr(buffer, "gin_context", None) is not None:
                raise AssertionError("native mode must not create a GIN context")
            raw = [int(value) for value in buffer.handle.buffer_ptrs]
            adjusted = [int(value) for value in buffer.buffer_ptrs]
            offset = int(buffer.handle.offset)
            aliases = validate_adjusted_aliases(raw, adjusted, offset,
                                               int(buffer.buffer.data_ptr()), rank, self.mode,
                                               num_bytes=int(buffer.buffer.numel()))
            if buffer.buffer.dtype not in (torch.int8, torch.uint8) or buffer.buffer.ndim != 1 or buffer.buffer.numel() < 128:
                raise AssertionError("sentinel requires a flat byte work allocation of at least128 bytes")
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            return {"rank": rank, "world_size": dist.get_world_size(),
                    "hostname": self.physical_hostnames[rank],
                    "local_rank": self._local_rank, "local_world_size": self._local_world_size,
                    "cuda_device": torch.cuda.current_device(), "device_name": props.name,
                    "capability": list(torch.cuda.get_device_capability()),
                    "num_sms": props.multi_processor_count,
                    "nccl": _query_nccl(torch, dist), "peer_alias_ranks": aliases,
                    "raw_buffer_ptrs": raw, "adjusted_buffer_ptrs": adjusted,
                    "mapped_adjusted_buffer_ptrs": adjusted,
                    "buffer_offset": offset, "buffer_pointer": int(buffer.buffer.data_ptr()),
                    "buffer_bytes": int(buffer.buffer.numel())}
        records = _gather(dist, _phase(dist, "gb200_real_buffer_properties", record))
        mapped_identity = _buffer_alias_identity(buffer)
        actual_topology = validate_records(records, MODES[self.mode])
        contexts = None
        if MODES[self.mode] == "gin":
            contexts = _gather(dist, _phase(dist, "gb200_real_gin_context", lambda: {
                name: getattr(buffer.gin_context, name) for name in (
                    "rank", "world_size", "lsa_rank", "lsa_size", "gin_type_string",
                    "context_count", "queue_depth", "signal_count", "connection_count")}))
            validate_context_records(contexts)

        def save():
            original = buffer.buffer[:128].clone()
            torch.cuda.synchronize()
            return original
        saved = _phase(dist, "save_real_work_buffer_sentinel_bytes", save)
        memory = None
        try:
            memory = _peer_memory_test(MODES[self.mode], torch, dist,
                                       buffer.buffer, buffer.handle, records)
        finally:
            # The probe touches only these first128 bytes, including peer
            # writes. Its collective phases finish all accesses before this
            # restoration; no MegaMoE task has yet consumed the workspace.
            def restore():
                buffer.buffer[:128].copy_(saved)
                torch.cuda.synchronize()
                if not torch.equal(buffer.buffer[:128], saved):
                    raise AssertionError("real work allocation was not restored after sentinel")
            _phase(dist, "restore_real_work_buffer_after_sentinel", restore)
        memory_records = _gather(dist, memory)
        def admit_canonicalization():
            if any(item.get("status") != "passed" for item in memory_records):
                raise AssertionError("all-rank mapped-memory proof must pass before canonicalization")
            if _buffer_alias_identity(buffer) != mapped_identity:
                raise AssertionError("work-buffer aliases changed during sentinel validation")
        _phase(dist, "admit_original_self_kernel_pointer", admit_canonicalization)
        # The original tensor owns storage, supplies TMA views and is registered
        # with GIN. Keep local generic accesses at that same VA. Only the copied
        # launch list changes; NCCL's mappings and all remote/null peers remain.
        kernel_pointers = list(buffer.buffer_ptrs)
        kernel_pointers[rank] = int(buffer.buffer.data_ptr())
        buffer.buffer_ptrs = kernel_pointers
        canonical_identity = _buffer_alias_identity(buffer)
        kernel_records = _gather(dist, {
            "rank": rank, "mapped_adjusted_buffer_ptrs": list(mapped_identity[-1]),
            "kernel_buffer_ptrs": list(kernel_pointers),
            "original_buffer_pointer": int(buffer.buffer.data_ptr()),
            "mapped_self_pointer": int(mapped_identity[-1][rank]),
            "only_self_pointer_canonicalized": True,
        })
        self.evidence = {
            "status": "passed", "mode": self.mode,
            "physical_hostnames": list(self.physical_hostnames),
            "logical_route_domains": list(self.logical_route_domains),
            "route_domain_meaning": "explicit rank0..7/rank8..15 groups; not physical OS hostnames",
            "legacy_cross_host_route_fields_mean": "logical route-domain predicates in this entrypoint",
            "actual_topology": actual_topology, "ranks": records,
            "actual_gin_contexts": contexts, "peer_memory_sentinels": memory_records,
            "sentinel_allocation": "actual MegaMoE symmetric work allocation",
            "sentinel_workspace_bytes_restored": 128,
            "kernel_pointer_records": kernel_records,
            "kernel_self_pointer_policy": "original_allocation_after_mapped_sentinels_and_restore",
            "original_allocation_tma_and_gin_registration_unchanged": True,
            "sentinel_proves_shared_bytes_not_intra_kernel_alias_proxy_ordering": True,
            "before_first_megamoe_compute": True,
            "native_gin_disabled": self.mode == "native_nvl",
            "gin_payload_test": False,
            "gin_payload_proof_requires_following_accuracy": MODES[self.mode] == "gin",
        }
        self._validated_buffer_id = id(buffer)
        self._validated_alias_identity = canonical_identity
        buffer._gb200_validated_alias_identity = canonical_identity
        return self.evidence

    def require_validated_buffer(self, buffer, hostnames):
        self.require_prepared(hostnames)
        if self.evidence is None or self._validated_buffer_id != id(buffer):
            raise RuntimeError("actual work-buffer LSA/alias/sentinel validation is required before compute")
        if (self._validated_alias_identity != _buffer_alias_identity(buffer) or
                getattr(buffer, "_gb200_validated_alias_identity", None) != self._validated_alias_identity):
            raise RuntimeError("validated work-buffer allocation or aliases changed before compute")
