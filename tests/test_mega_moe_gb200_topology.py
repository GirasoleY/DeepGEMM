"""CPU GB200 harness/alias contracts; no GPU topology or performance claim."""

from copy import deepcopy
import inspect
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import mega_moe_gb200_topology as topology
import probe_gb200_topology as probe
import test_mega_moe_accuracy as accuracy


HOSTS = tuple(f"actual-host-{rank // 4}" for rank in range(16))


def args(mode="gin_ib", tokens=32):
    return SimpleNamespace(require_gin=mode in ("gin_ib", "gin_roce"), num_experts=896,
                           num_topk=16, hidden=3584, intermediate_hidden=3072,
                           num_max_tokens_per_rank=384, num_shared_experts=0,
                           mma_type="fp8xfp4", num_tokens=tokens)


class FakeDist:
    def __init__(self):
        self.records = []
        self.mismatch = False

    def get_rank(self):
        return 0

    def get_world_size(self):
        return 16

    def all_gather_object(self, output, item):
        self.records.append(deepcopy(item))
        output[:] = [deepcopy(item) for _ in range(16)]
        if self.mismatch and isinstance(item, dict) and "mode" in item:
            output[15]["mode"] = "native_nvl"


class FakeBytes:
    def __init__(self, values=None, start=0, end=None):
        self.values = list(range(256)) if values is None else values
        self.start = start
        self.end = len(self.values) if end is None else end
        self.dtype, self.ndim = "int8", 1

    def __getitem__(self, selection):
        assert isinstance(selection, slice) and selection.step is None
        start = 0 if selection.start is None else selection.start
        end = self.numel() if selection.stop is None else selection.stop
        return FakeBytes(self.values, self.start + start, self.start + end)

    def numel(self):
        return self.end - self.start

    def data_ptr(self):
        return 0x1100 + self.start

    def clone(self):
        return FakeBytes(self.values[self.start:self.end])

    def copy_(self, other):
        self.values[self.start:self.end] = other.values[other.start:other.end]


def fake_torch():
    props = SimpleNamespace(name="NVIDIA GB200", multi_processor_count=148)
    return SimpleNamespace(int8="int8", uint8="uint8", equal=lambda a,b:
                           a.values[a.start:a.end] == b.values[b.start:b.end],
                           cuda=SimpleNamespace(current_device=lambda: 0,
                                                get_device_properties=lambda device: props,
                                                get_device_capability=lambda: (10, 0),
                                                synchronize=mock.Mock()))


def native_buffer():
    raw = [0x1000 * (rank + 1) for rank in range(16)]
    return SimpleNamespace(buffer=FakeBytes(), gin_enabled=False, gin_context=None,
                           buffer_ptrs=[pointer + 256 for pointer in raw],
                           handle=SimpleNamespace(buffer_ptrs=raw, offset=256))


class GB200TopologyTests(unittest.TestCase):
    def test_physical_hosts_and_logical_domains_are_distinct(self):
        self.assertEqual(topology.validate_physical_hostnames(HOSTS), HOSTS)
        self.assertEqual(topology.logical_route_domains(HOSTS), (0,) * 8 + (1,) * 8)
        self.assertEqual(len(set(HOSTS)), 4)
        for bad in (("logical0",) * 8 + ("logical1",) * 8,
                    ("one",) * 16, HOSTS[:-1], (None,) + HOSTS[1:],
                    HOSTS[:3] + (HOSTS[4],) + HOSTS[4:]):
            with self.assertRaises(ValueError):
                topology.logical_route_domains(bad)

    def test_every_matched_owner_assignment_identical_to_novita_fixture(self):
        logical = topology.logical_route_domains(HOSTS)
        novita = ["novita0"] * 8 + ["novita1"] * 8
        for tokens in (32, 40, 48):
            for rank in range(16):
                for mode in ("all_same_host", "half_remote", "all_remote"):
                    for token in range(tokens):
                        for slot in range(16):
                            self.assertEqual(
                                accuracy._owner_for(mode, rank, token, slot, logical,
                                                    num_tokens=tokens, num_topk=16),
                                accuracy._owner_for(mode, rank, token, slot, novita,
                                                    num_tokens=tokens, num_topk=16))

    def test_legacy_gin_placement_still_rejects_actual_four_hosts(self):
        with self.assertRaisesRegex(RuntimeError, "contiguous 2x8"):
            accuracy._validate_gin_host_placement(args(), 16, HOSTS)
        accuracy._validate_gin_host_placement(args(), 16, ["a"] * 8 + ["b"] * 8)

    def test_all_sixteen_native_adjusted_aliases_required(self):
        buffer = native_buffer()
        self.assertEqual(topology.validate_adjusted_aliases(
            buffer.handle.buffer_ptrs, buffer.buffer_ptrs, 256,
            buffer.buffer.data_ptr(), 0, "native_nvl"), list(range(16)))
        for changed in (0, 3, 8, 15):
            aliases = buffer.buffer_ptrs.copy(); aliases[changed] = 0
            with self.assertRaises(AssertionError):
                topology.validate_adjusted_aliases(buffer.handle.buffer_ptrs, aliases,
                                                   256, buffer.buffer.data_ptr(), 0, "native_nvl")

    def test_gin_exact_lsa_membership_and_offset_not_environment(self):
        for mode in ("gin_ib", "gin_roce"):
            for rank in (0, 7, 8, 15):
                peers = list(range(rank // 8 * 8, rank // 8 * 8 + 8))
                raw = [0x1000 * (peer + 1) if peer in peers else 0 for peer in range(16)]
                adjusted = [pointer + 256 if pointer else 0 for pointer in raw]
                self.assertEqual(topology.validate_adjusted_aliases(raw, adjusted, 256,
                                                                  adjusted[rank], rank, mode), peers)
                bad = adjusted.copy(); bad[(rank + 8) % 16] = 256
                with self.assertRaises(AssertionError):
                    topology.validate_adjusted_aliases(raw, bad, 256, adjusted[rank], rank, mode)
                with self.assertRaises(AssertionError):
                    topology.validate_adjusted_aliases(raw, adjusted, 0, adjusted[rank], rank, mode)
                self.assertEqual(topology.validate_adjusted_aliases(
                    raw, adjusted, 256, raw[rank], rank, mode), peers)

    def test_distinct_self_va_is_allowed_but_invalid_ranges_are_not(self):
        buffer = native_buffer()
        raw, adjusted = buffer.handle.buffer_ptrs, buffer.buffer_ptrs
        self.assertEqual(topology.validate_adjusted_aliases(
            raw, adjusted, 256, 0x900000, 0, "native_nvl", num_bytes=256), list(range(16)))
        for pointer in (0, -1, True, (1 << 64) - 127, 1 << 64):
            with self.assertRaises(AssertionError):
                topology.validate_adjusted_aliases(raw, adjusted, 256, pointer, 0,
                                                   "native_nvl", num_bytes=128)
        for pointer in (-1, True, 1 << 64, (1 << 64) - 127):
            bad_raw, bad_adjusted = raw.copy(), adjusted.copy()
            bad_raw[3], bad_adjusted[3] = pointer, pointer
            with self.assertRaises(AssertionError):
                topology.validate_adjusted_aliases(bad_raw, bad_adjusted, 0,
                                                   0x1100, 0, "native_nvl", num_bytes=128)

    def test_roce_retains_gdaki_context_and_exact_same_clique_probe(self):
        policy = topology.GB200Topology("gin_roce")
        torch, dist = fake_torch(), FakeDist()
        policy.prepare(args("gin_roce"), HOSTS, 0, 4, torch, dist)
        self.assertEqual(dist.records[-1]["mode"], "gin_roce")
        buffer = native_buffer()
        buffer.gin_enabled = True
        buffer.handle.buffer_ptrs[8:] = [0] * 8
        buffer.buffer_ptrs[8:] = [0] * 8
        buffer.gin_context = SimpleNamespace(rank=0, world_size=16, lsa_rank=0, lsa_size=8,
            gin_type_string="gdaki", context_count=9, queue_depth=64, signal_count=2, connection_count=1)
        with mock.patch.object(probe, "_query_nccl", return_value={}), mock.patch.object(
                probe, "validate_records", return_value={"mock_only": True}) as records, mock.patch.object(
                probe, "validate_context_records") as contexts, mock.patch.object(
                probe, "_peer_memory_test", return_value={"status": "passed"}) as sentinel:
            evidence = policy.validate_buffer(buffer, args("gin_roce"), "NCCL", object(), torch, dist)
        self.assertEqual(records.call_args.args[1], "gin")
        contexts.assert_called_once()
        self.assertEqual(sentinel.call_args.args[0], "gin")
        self.assertEqual(evidence["mode"], "gin_roce")
        self.assertFalse(evidence["native_gin_disabled"])
        self.assertTrue(evidence["gin_payload_proof_requires_following_accuracy"])
        self.assertFalse(evidence["gin_payload_test"])
        self.assertEqual(policy.expected_lsa_ranks(8), list(range(8, 16)))
        self.assertEqual(buffer.buffer_ptrs[8:], [0] * 8)

    def test_roce_and_ib_mode_labels_must_be_collectively_uniform(self):
        class OtherGinMode(FakeDist):
            def all_gather_object(self, output, item):
                super().all_gather_object(output, item)
                if isinstance(item, dict) and item.get("mode") == "gin_roce":
                    output[15]["mode"] = "gin_ib"
        with self.assertRaisesRegex(RuntimeError, "configuration differs"):
            topology.GB200Topology("gin_roce").prepare(
                args("gin_roce"), HOSTS, 0, 4, fake_torch(), OtherGinMode())

    def test_prepare_configuration_is_collective_before_admission(self):
        policy = topology.GB200Topology("gin_ib", run_configuration={"replays": 204, "source": "abc"})
        dist = FakeDist(); torch = fake_torch()
        policy.prepare(args(), HOSTS, 0, 4, torch, dist)
        self.assertEqual(policy.physical_hostnames, HOSTS)
        self.assertEqual(policy.logical_route_domains, (0,) * 8 + (1,) * 8)
        self.assertIsNone(policy.evidence)
        self.assertEqual(dist.records[-1]["run_configuration"], {"replays": 204, "source": "abc"})
        other = topology.GB200Topology("gin_ib"); dist.mismatch = True
        with self.assertRaisesRegex(RuntimeError, "configuration differs"):
            other.prepare(args(), HOSTS, 0, 4, torch, dist)
        self.assertIsNone(other.physical_hostnames)

    def test_wrong_mode_placement_shape_and_hardware_fail_collectively(self):
        for updates in ({"require_gin": False}, {"num_tokens": 64}, {"num_topk": 8}):
            value = args(); value.__dict__.update(updates)
            with self.assertRaisesRegex(RuntimeError, "gb200_configuration"):
                topology.GB200Topology("gin_ib").prepare(value, HOSTS, 0, 4, fake_torch(), FakeDist())
        for local_rank, local_world in ((1, 4), (0, 8)):
            with self.assertRaises(RuntimeError):
                topology.GB200Topology("gin_ib").prepare(args(), HOSTS, local_rank, local_world,
                                                         fake_torch(), FakeDist())
        torch = fake_torch()
        torch.cuda.get_device_properties = lambda device: SimpleNamespace(name="NVIDIA B300")
        with self.assertRaisesRegex(RuntimeError, "actual device"):
            topology.GB200Topology("gin_ib").prepare(args(), HOSTS, 0, 4, torch, FakeDist())

    def native_validated_policy(self, *, sentinel_error=False, restore_error=False, buffer_override=None):
        policy = topology.GB200Topology("native_nvl")
        torch, dist = fake_torch(), FakeDist()
        policy.prepare(args("native_nvl"), HOSTS, 0, 4, torch, dist)
        buffer = native_buffer() if buffer_override is None else buffer_override
        original = buffer.buffer.values.copy()
        mapped_before = tuple(buffer.buffer_ptrs)
        def gather(_dist, value):
            if isinstance(value, dict) and "nccl" in value:
                records = []
                for rank in range(16):
                    record = deepcopy(value)
                    record.update(rank=rank, hostname=HOSTS[rank], local_rank=rank % 4,
                                  cuda_device=rank % 4)
                    record["nccl"].update(rank=rank, cuda_device=rank % 4)
                    records.append(record)
                return records
            return [deepcopy(value) for _ in range(16)]
        def sentinel(mode, _torch, _dist, allocation, handle, records):
            self.assertIs(allocation, buffer.buffer)
            self.assertIs(handle, buffer.handle)
            self.assertEqual(mode, "native")
            self.assertEqual(len(records), 16)
            self.assertEqual(tuple(buffer.buffer_ptrs), mapped_before)
            self.assertFalse(hasattr(buffer, "_gb200_validated_alias_identity"))
            allocation.values[:128] = [-1] * 128
            if sentinel_error:
                raise AssertionError("injected peer-read mismatch")
            return {"status": "passed", "read_peers": list(range(16))}
        props = {"version": 23007, "rank": 0, "n_ranks": 16, "cuda_device": 0,
                 "device_api_support": True, "n_lsa_teams": 1, "gin_type": 0}
        with mock.patch.object(probe, "_query_nccl", return_value=props), \
                mock.patch.object(probe, "_gather", side_effect=gather), \
                mock.patch.object(probe, "_peer_memory_test", side_effect=sentinel) as peer_test:
            if restore_error:
                torch.equal = lambda a, b: False
                with self.assertRaisesRegex(RuntimeError, "restore_real_work_buffer"):
                    policy.validate_buffer(buffer, args("native_nvl"), "NCCL", object(), torch, dist)
            elif sentinel_error:
                with self.assertRaisesRegex(AssertionError, "injected peer-read"):
                    policy.validate_buffer(buffer, args("native_nvl"), "NCCL", object(), torch, dist)
            else:
                policy.validate_buffer(buffer, args("native_nvl"), "NCCL", object(), torch, dist)
            self.assertEqual(peer_test.call_count, 1)
        self.assertEqual(buffer.buffer.values, original)
        if sentinel_error or restore_error:
            self.assertEqual(tuple(buffer.buffer_ptrs), mapped_before)
            self.assertFalse(hasattr(buffer, "_gb200_validated_alias_identity"))
        return policy, buffer

    def test_actual_buffer_sentinel_restoration_and_identity_gate(self):
        policy, buffer = self.native_validated_policy()
        policy.require_validated_buffer(buffer, HOSTS)
        self.assertEqual(policy.expected_lsa_ranks(0), list(range(16)))
        self.assertTrue(policy.evidence["before_first_megamoe_compute"])
        self.assertEqual(policy.evidence["sentinel_workspace_bytes_restored"], 128)
        with self.assertRaises(RuntimeError):
            policy.require_validated_buffer(native_buffer(), HOSTS)

    def test_failed_sentinel_restores_bytes_but_never_admits_compute(self):
        for failure in ("sentinel_error", "restore_error"):
            buffer = native_buffer()
            buffer.handle.buffer_ptrs[0] += 0x100000
            buffer.buffer_ptrs[0] += 0x100000
            policy, buffer = self.native_validated_policy(buffer_override=buffer, **{failure: True})
            self.assertIsNone(policy.evidence)
            with self.assertRaises(RuntimeError):
                policy.require_validated_buffer(buffer, HOSTS)

    def test_only_copied_self_pointer_changes_after_alias_proof_and_restore(self):
        buffer = native_buffer()
        buffer.handle.buffer_ptrs[0] += 0x100000
        buffer.buffer_ptrs[0] += 0x100000
        original_tensor, original_handle = buffer.buffer, buffer.handle
        mapped_list = buffer.buffer_ptrs
        raw = tuple(buffer.handle.buffer_ptrs)
        mapped = tuple(mapped_list)
        policy, buffer = self.native_validated_policy(buffer_override=buffer)
        self.assertIs(buffer.buffer, original_tensor)
        self.assertIs(buffer.handle, original_handle)
        self.assertEqual(tuple(buffer.handle.buffer_ptrs), raw)
        self.assertEqual(tuple(mapped_list), mapped)
        self.assertIsNot(buffer.buffer_ptrs, mapped_list)
        self.assertEqual(buffer.buffer_ptrs, [buffer.buffer.data_ptr(), *mapped[1:]])
        self.assertEqual(policy.evidence["ranks"][0]["mapped_adjusted_buffer_ptrs"], list(mapped))
        self.assertEqual(policy.evidence["kernel_pointer_records"][0]["kernel_buffer_ptrs"],
                         buffer.buffer_ptrs)
        self.assertEqual(buffer._gb200_validated_alias_identity, topology._buffer_alias_identity(buffer))
        policy.require_validated_buffer(buffer, HOSTS)
        with self.assertRaisesRegex(RuntimeError, "first compute only"):
            policy.validate_buffer(buffer, args("native_nvl"), "NCCL", object(), fake_torch(), FakeDist())

    def test_validated_allocation_raw_and_canonical_pointer_state_is_immutable(self):
        mutations = (
            lambda b: b.buffer_ptrs.__setitem__(0, b.buffer_ptrs[0] + 8),
            lambda b: b.buffer_ptrs.__setitem__(15, b.buffer_ptrs[15] + 8),
            lambda b: b.handle.buffer_ptrs.__setitem__(2, b.handle.buffer_ptrs[2] + 8),
            lambda b: setattr(b.handle, "offset", 512),
            lambda b: setattr(b, "handle", deepcopy(b.handle)),
            lambda b: setattr(b, "buffer", FakeBytes(b.buffer.values.copy())),
            lambda b: setattr(b, "_gb200_validated_alias_identity", None),
        )
        for mutate in mutations:
            policy, buffer = self.native_validated_policy()
            mutate(buffer)
            with self.assertRaisesRegex(RuntimeError, "aliases changed"):
                policy.require_validated_buffer(buffer, HOSTS)

    def test_unprepared_buffer_and_legacy_native_allocator_remain_fail_closed(self):
        with self.assertRaises(RuntimeError):
            topology.GB200Topology("native_nvl").require_validated_buffer(native_buffer(), HOSTS)
        self.assertEqual(accuracy._configure_symmetric_memory_backend(
            SimpleNamespace(require_gin=False), None, None), (None, None))
        signature = inspect.signature(accuracy._configure_symmetric_memory_backend)
        self.assertIs(signature.parameters["force_nccl"].default, False)

    def test_native_force_nccl_uses_same_registry_bridge_without_enabling_gin(self):
        events, registration = [], object()
        symm = ModuleType("torch.distributed._symmetric_memory")
        symm.set_backend = lambda value: events.append(("backend", value))
        symm.get_backend = lambda device: "NCCL"
        symm.enable_symm_mem_for_group = lambda group: events.append(("group", group))
        nccl = ModuleType("torch.distributed._symmetric_memory._nccl")
        def register(group, pointer, device, owner):
            events.append(("register", group, pointer, device))
            return registration
        nccl.register_external_nccl_comm = register
        symm._nccl = nccl
        dist_module = ModuleType("torch.distributed")
        dist_module._symmetric_memory = symm
        torch_module = ModuleType("torch")
        torch_module.distributed = dist_module
        runtime_torch = SimpleNamespace(int32="int32", ones=lambda *a, **kw:
                                        SimpleNamespace(item=lambda: 16),
                                        device=lambda kind, ordinal: (kind, ordinal),
                                        cuda=SimpleNamespace(current_device=lambda: 0))
        world = SimpleNamespace(group_name="world", _get_backend=lambda device:
                                SimpleNamespace(_comm_ptr=lambda: 123))
        runtime_dist = SimpleNamespace(group=SimpleNamespace(WORLD=world),
                                       get_world_size=lambda: 16,
                                       all_reduce=lambda *a, **kw: events.append(("all_reduce",)))
        options = SimpleNamespace(require_gin=False)
        with mock.patch.dict(sys.modules, {"torch": torch_module,
                                           "torch.distributed": dist_module,
                                           symm.__name__: symm, nccl.__name__: nccl}):
            result = accuracy._configure_symmetric_memory_backend(
                options, runtime_torch, runtime_dist, force_nccl=True)
        self.assertEqual(result, ("NCCL", registration))
        self.assertFalse(options.require_gin)
        self.assertEqual([item[0] for item in events], ["group", "backend", "all_reduce", "register"])

    def test_native_gin_context_and_missing_registration_fail_before_sentinel(self):
        for gin_enabled, context, backend, registration in (
                (True, object(), "NCCL", object()),
                (False, object(), "NCCL", object()),
                (False, None, "CUDA", object()),
                (False, None, "NCCL", None)):
            policy = topology.GB200Topology("native_nvl")
            runtime_torch, dist = fake_torch(), FakeDist()
            policy.prepare(args("native_nvl"), HOSTS, 0, 4, runtime_torch, dist)
            buffer = native_buffer()
            buffer.gin_enabled, buffer.gin_context = gin_enabled, context
            with mock.patch.object(probe, "_peer_memory_test") as sentinel:
                with self.assertRaisesRegex(RuntimeError, "gb200_real_buffer_properties"):
                    policy.validate_buffer(buffer, args("native_nvl"), backend,
                                           registration, runtime_torch, dist)
                sentinel.assert_not_called()
            self.assertIsNone(policy.evidence)

    def test_worker_hooks_precede_first_compute_and_keep_physical_metadata(self):
        source = inspect.getsource(accuracy._worker)
        self.assertLess(source.index("topology.prepare("), source.index("_configure_symmetric_memory_backend("))
        self.assertLess(source.index("topology.validate_buffer("), source.index("inputs = _make_inputs"))
        self.assertLess(source.index("topology.validate_buffer("), source.index("_snapshot_and_check_oracles"))
        self.assertIn('"hostnames": hostnames', source)
        self.assertIn("harness.physical_hostnames = list(hostnames)", source)
        self.assertIn("harness.route_domains = list(route_domains)", source)
        self.assertIn("topology.accuracy_result = result", source)
        self.assertIn("enable_gin=args.require_gin", source)
        self.assertIn("force_nccl=True", source)
        self.assertIs(inspect.signature(accuracy._worker).parameters["topology"].default, None)
        self.assertIn("if topology is None:\n            _validate_gin_host_placement", source)


if __name__ == "__main__":
    unittest.main()
