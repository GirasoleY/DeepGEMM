"""CPU contracts for the topology probe; no CUDA/NIC execution claim."""

import copy
import ctypes
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import probe_gb200_topology as probe


def records(mode="gin"):
    return [{
        "rank": rank, "world_size": 16, "hostname": f"host{rank // 4}",
        "local_rank": rank % 4, "local_world_size": 4, "cuda_device": rank % 4,
        "peer_alias_ranks": probe.expected_peers(rank, mode),
        "nccl": {"version": 23007, "rank": rank, "n_ranks": 16,
                 "cuda_device": rank % 4, "device_api_support": True,
                 "n_lsa_teams": 2 if mode == "gin" else 1, "gin_type": 3},
    } for rank in range(16)]


class ProbeContracts(unittest.TestCase):
    def peer_fixture(self, *, offset=0, bad_pointer=False, bad_bytes=False,
                     world_size=16, mode="native"):
        """CPU call-path mock, not a device alias/coherence simulation."""
        own = mock.MagicMock()
        own.element_size.return_value = 8
        own.data_ptr.return_value = 0x900000
        own.device = "cuda:0"
        peers = probe.expected_peers(0, mode, world_size)
        own.cpu().tolist.return_value = [probe.sentinel(0, peer, 0, peer) if peer in peers else -1
                                         for peer in range(world_size)]
        if bad_bytes:
            own.cpu().tolist.return_value = [-1] * world_size
        allocation = mock.MagicMock()
        allocation.__getitem__.return_value.view.return_value = own
        raw = [0x1000 * (peer + 1) if peer in peers else 0 for peer in range(world_size)]
        def get_buffer(peer, shape, *, dtype, storage_offset):
            self.assertEqual(shape, (world_size,))
            self.assertEqual(dtype, "int64")
            self.assertEqual(storage_offset, offset // 8)
            view = mock.MagicMock()
            view.data_ptr.return_value = raw[peer] + offset + (8 if bad_pointer else 0)
            view.clone().cpu().tolist.return_value = [probe.sentinel(1, peer, peer, slot)
                                                      for slot in range(world_size)]
            return view
        handle = SimpleNamespace(buffer_ptrs=raw, offset=offset,
                                 get_buffer=mock.Mock(side_effect=get_buffer))
        def gather(output, item):
            output[:] = [item] * world_size
        dist = SimpleNamespace(get_rank=lambda: 0, get_world_size=lambda: world_size,
                               all_gather_object=mock.Mock(side_effect=gather))
        torch = SimpleNamespace(int64="int64", tensor=mock.Mock(),
                                cuda=SimpleNamespace(synchronize=mock.Mock()))
        return own, allocation, handle, dist, torch

    def test_ep8_sentinel_uses_eight_slots_and_only_actual_lsa_views(self):
        records8 = [{"hostname": "host" + str(rank // 4)} for rank in range(8)]
        for mode in ("gin", "native"):
            for offset in (0, 256):
                own, allocation, handle, dist, torch = self.peer_fixture(
                    world_size=8, mode=mode, offset=offset)
                value = probe._peer_memory_test(mode, torch, dist, allocation, handle, records8)
                self.assertEqual(value["words_per_read"], 8)
                self.assertEqual(value["write_peers"], list(range(4 if mode == "gin" else 8)))
                self.assertEqual(value["cross_os_peers_tested"], [] if mode == "gin" else list(range(4, 8)))
                self.assertTrue(value["original_to_mapped_and_mapped_to_original_checked"])
                self.assertEqual(value["get_buffer_storage_offset_elements"], offset // 8)
                allocation.__getitem__.assert_called_once_with(slice(None, 64, None))

    def test_distinct_self_mapping_keeps_original_as_bidirectional_sentinel_target(self):
        for offset in (0, 256):
            own, allocation, handle, dist, torch = self.peer_fixture(offset=offset)
            result = probe._peer_memory_test("native", torch, dist, allocation, handle, records("native"))
            self.assertEqual(result["status"], "passed")
            self.assertFalse(result["self_virtual_addresses_equal"])
            self.assertTrue(result["original_to_mapped_and_mapped_to_original_checked"])
            self.assertFalse(result["intra_kernel_alias_proxy_ordering_claim"])
            self.assertEqual(result["get_buffer_storage_offset_elements"], offset // 8)
            self.assertEqual(result["read_peers"], list(range(16)))
            self.assertEqual(result["write_peers"], list(range(16)))
            own.fill_.assert_called_once_with(-1)
            own.cpu.assert_called()
            own.copy_.assert_called_once()
            self.assertEqual(handle.get_buffer.call_count, 16)

    def test_offset_alignment_and_pointer_equation_fail_before_any_memory_access(self):
        for options in ({"offset": 7}, {"offset": -8}, {"bad_pointer": True}):
            own, allocation, handle, dist, torch = self.peer_fixture(**options)
            with self.assertRaisesRegex(RuntimeError, "prepare_peer_views"):
                probe._peer_memory_test("native", torch, dist, allocation, handle, records("native"))
            own.fill_.assert_not_called()
            own.copy_.assert_not_called()

    def test_distinct_mapping_without_matching_original_bytes_is_rejected(self):
        own, allocation, handle, dist, torch = self.peer_fixture(bad_bytes=True)
        with self.assertRaisesRegex(RuntimeError, "peer-write sentinel mismatch"):
            probe._peer_memory_test("native", torch, dist, allocation, handle, records("native"))
        own.copy_.assert_not_called()

    def test_exact_nccl_23007_abi_on_64bit_host(self):
        self.assertEqual(ctypes.sizeof(ctypes.c_void_p), 8)
        self.assertEqual(ctypes.sizeof(probe.NcclProperties23007), 56)
        for field, offset in {"size": 0, "magic": 8, "version": 12,
                              "rank": 16, "nRanks": 20, "cudaDev": 24,
                              "nvmlDev": 28, "deviceApiSupport": 32,
                              "multimemSupport": 33, "ginType": 36,
                              "nLsaTeams": 40, "hostRmaSupport": 44,
                              "railedGinType": 48}.items():
            self.assertEqual(getattr(probe.NcclProperties23007, field).offset, offset)

    def test_two8_and_native16_actual_records(self):
        for mode in ("gin", "native"):
            result = probe.validate_records(records(mode), mode)
            self.assertTrue(result["actual_properties_and_aliases_checked"])
            self.assertEqual(result["physical_host_count"], 4)
            self.assertEqual(result["lsa_team_size"], 8 if mode == "gin" else 16)

    def test_environment_cannot_override_actual_bad_topology(self):
        value = records()
        for record in value:
            record["environment_not_proof"] = {"NCCL_LSA_TEAM_SIZE": "8"}
        value[0]["nccl"]["n_lsa_teams"] = 4
        with self.assertRaisesRegex(AssertionError, "LSA team count"):
            probe.validate_records(value, "gin")

    def test_missing_or_extra_alias_rejected_even_if_properties_match(self):
        for peer_list in ([0, 1, 2, 3], list(range(16)), [0, 1, 2, 3, 4, 5, 6]):
            value = records()
            value[0]["peer_alias_ranks"] = peer_list
            with self.assertRaisesRegex(AssertionError, "peer aliases"):
                probe.validate_records(value, "gin")

    def test_physical_layout_and_rank_metadata_are_not_logical_host_spoofs(self):
        for key, replacement in (("hostname", "host1"), ("local_rank", 3),
                                 ("local_world_size", 8)):
            value = records()
            value[0][key] = replacement
            with self.assertRaises(AssertionError):
                probe.validate_records(value, "gin")
        value = records()
        for item in value:
            item["hostname"] = "logical" + str(item["rank"] // 8)
        with self.assertRaisesRegex(AssertionError, "four distinct"):
            probe.validate_records(value, "gin")

    def test_version_device_backend_and_rank_mismatch_fail(self):
        for key, replacement in (("version", 23003), ("rank", 1), ("n_ranks", 8),
                                 ("cuda_device", 2), ("device_api_support", False),
                                 ("gin_type", 2)):
            value = records()
            value[0]["nccl"][key] = replacement
            with self.assertRaises(AssertionError):
                probe.validate_records(value, "gin")
        value = records("native")
        for item in value:
            item["nccl"]["gin_type"] = 0
        probe.validate_records(value, "native")  # Native does not claim GIN.

    def test_live_context_resources_not_only_host_properties(self):
        good = [{"rank": rank, "world_size": 16, "lsa_rank": rank % 8,
                 "lsa_size": 8, "gin_type_string": "gdaki", "context_count": 9,
                 "queue_depth": 64, "signal_count": 2, "connection_count": 1}
                for rank in range(16)]
        probe.validate_context_records(good)
        for key, replacement in (("lsa_rank", 3), ("lsa_size", 4),
                                 ("gin_type_string", "proxy"), ("context_count", 8),
                                 ("queue_depth", 32), ("signal_count", 1),
                                 ("connection_count", 0)):
            value = copy.deepcopy(good)
            value[0][key] = replacement
            with self.assertRaises(AssertionError):
                probe.validate_context_records(value)

    def test_sentinel_identity_and_negative_controls(self):
        values = {probe.sentinel(epoch, writer, peer, slot)
                  for epoch in range(2) for writer in range(16)
                  for peer in range(16) for slot in range(16)}
        self.assertEqual(len(values), 2 * 16**3)
        expected = [probe.sentinel(0, writer, 7, writer) for writer in range(8)]
        probe._check_values(expected, expected, "write")
        for wrong in (expected[::-1], expected[:-1], [-1] * 8):
            with self.assertRaisesRegex(AssertionError, "sentinel mismatch"):
                probe._check_values(wrong, expected, "write")

    def test_local_phase_error_is_gathered_before_raising(self):
        class Dist:
            gathered = False
            @staticmethod
            def get_world_size():
                return 16
            def all_gather_object(self, out, error):
                self.gathered = True
                out[:] = [error] * 16
        dist = Dist()
        def fail():
            raise ValueError("rank failure")
        with self.assertRaisesRegex(RuntimeError, "probe phase mock failed"):
            probe._phase(dist, "mock", fail)
        self.assertTrue(dist.gathered)

    def test_context_cleanup_tries_abort_and_release_after_destroy_failure(self):
        class Context:
            calls = []
            def destroy(self):
                self.calls.append("destroy")
                raise ValueError("failure")
            def abort(self):
                self.calls.append("abort")
            def _release_buffer_registration(self):
                self.calls.append("release")
        context = Context()
        self.assertEqual(len(probe._retire_context(context, True)), 1)
        self.assertEqual(context.calls, ["destroy", "abort", "release"])

    def test_source_preserves_separate_memory_and_gin_claims_and_lifetime(self):
        source = Path(probe.__file__).read_text()
        self.assertIn('"status": "not_run", "actual_payload_checked": False', source)
        self.assertIn('"gin_payload_test": False', source)
        self.assertIn('"environment_not_proof"', source)
        self.assertIn('SimpleNamespace(require_gin=True)', source)
        self.assertIn('handle.get_buffer(peer, (world_size,), dtype=torch.int64,', source)
        self.assertIn('world_size = validate_world_size(dist.get_world_size())', source)
        self.assertIn('storage_offset=offset // own.element_size()', source)
        self.assertIn('view[rank:rank + 1].fill_', source)
        self.assertIn('view.clone().cpu().tolist()', source)
        self.assertLess(source.index('version.value != NCCL_VERSION'),
                        source.index('query = dso.ncclCommQueryProperties'))
        self.assertLess(source.index('cleanup_errors = _retire_context(context, success)'),
                        source.index('gin_buffer = context = handle = allocation = None'))
        self.assertLess(source.index('gin_buffer = context = handle = allocation = None'),
                        source.index('accuracy._unregister_symmetric_memory_comm(registration'))
        self.assertNotIn('ctypes.CDLL(os.getenv', source)
        self.assertNotIn('cudaLaunchKernel', source)


if __name__ == "__main__":
    unittest.main()
