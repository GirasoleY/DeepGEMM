"""CPU-only EP8 topology/routing/gate tests; not GPU conformance evidence."""

from copy import deepcopy
import inspect
import os
from types import SimpleNamespace
import unittest
from unittest import mock

import bench_gb200_transport_matched as runner
import mega_moe_gb200_topology as topology
import probe_gb200_topology as probe
import test_mega_moe_accuracy as accuracy
from test_gb200_transport_matched_contract import FakeDeepEPBuffer
from test_mega_moe_accuracy_contract import _FakeBuffer, _FakeContext, _FakeEvidenceDist
from test_mega_moe_gb200_topology import FakeDist, fake_torch
from test_probe_gb200_topology import records as ep16_records


HOSTS = ["real-host0"] * 4 + ["real-host1"] * 4
DOMAINS = [0] * 4 + [1] * 4


def parse(mode="gin_roce", tokens=32):
    return runner.parse_args(["--mode", mode, "--world-size", "8",
                              "--decode-mns", str(tokens // 4), "--output", "not-written.json"])


def records(mode):
    result = deepcopy(ep16_records(mode)[:8])
    for rank, record in enumerate(result):
        record["world_size"] = record["nccl"]["n_ranks"] = 8
        record["peer_alias_ranks"] = probe.expected_peers(rank, mode, 8)
    return result


class Dist8(FakeDist):
    def get_world_size(self):
        return 8

    def all_gather_object(self, output, item):
        self.records.append(deepcopy(item))
        output[:] = [deepcopy(item) for _ in range(8)]
        if self.mismatch and isinstance(item, dict) and "mode" in item:
            output[7]["world_size"] = 16


def route_records(tokens, domains=DOMAINS):
    result = [{"rank": rank, "runtime_block_m": 32, "routes": {}} for rank in range(8)]
    for name in accuracy.MATCHED_BENCHMARK_ROUTES:
        received = [[0] * 56 for _ in range(8)]
        for rank in range(8):
            sent = [[0] * 56 for _ in range(8)]
            for token in range(tokens):
                for slot in range(16):
                    owner = accuracy._owner_for(name, rank, token, slot, domains,
                                                num_tokens=tokens, num_topk=16)
                    expert = (token * 16 + slot) % 56
                    sent[owner][expert] += 1
                    received[owner][expert] += 1
            result[rank]["routes"][name] = {
                "sent_owner_expert_histograms": sent, "valid_routes": tokens * 16,
                "cross_host_routes": sum(sum(row) for owner, row in enumerate(sent)
                                         if domains[owner] != domains[rank])}
        for rank in range(8):
            result[rank]["routes"][name]["expected_received_expert_histogram"] = received[rank]
    return result


class ProfileContracts(unittest.TestCase):
    def test_explicit_profile_preserves_shape_math_and_timing(self):
        for mode in runner.MODES:
            for tokens in (32, 40, 48):
                options, args, comparison = parse(mode, tokens)
                config = runner.configuration(options, args, comparison, {})
                self.assertEqual((options.world_size, args.num_experts, args.num_topk,
                                  args.hidden, args.intermediate_hidden, args.num_max_tokens_per_rank),
                                 (8, 448, 16, 3584, 3072, 384))
                self.assertEqual((config["physical_hosts"], config["ranks_per_physical_host"]), (2, 4))
                self.assertEqual(config["logical_route_domains"], DOMAINS)
                self.assertEqual((comparison.comparison_replays, comparison.comparison_conditioning,
                                  comparison.comparison_warmups), (204, 60, 10))
                self.assertFalse(args.fast_math)
                self.assertEqual(args.activation_clamp, 10.)
                self.assertEqual((comparison.deepep_num_sms, comparison.deepep_num_qps), (16, 9))
                accuracy._validate_args(args, 8, gb200_world_size=8)

    def test_ep8_requires_explicit_profile_and_normal_host_guard_is_retained(self):
        _, args, _ = parse()
        with self.assertRaisesRegex(ValueError, "exactly 16 ranks"):
            accuracy._validate_args(args, 8)
        with self.assertRaisesRegex(RuntimeError, "exactly 16"):
            accuracy._validate_gin_host_placement(args, 8, HOSTS)
        for world in (True, 4, 12, "8", 16):
            with self.assertRaises(ValueError):
                accuracy._validate_args(args, 8, gb200_world_size=world)

    def test_profile_rejects_shape_capacity_topology_mismatch_collectively(self):
        _, args, _ = parse()
        for key, bad in (("num_experts", 896), ("num_topk", 8), ("num_tokens", 64),
                         ("num_max_tokens_per_rank", 768), ("hidden", 7168)):
            wrong = deepcopy(args); setattr(wrong, key, bad)
            dist = Dist8()
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "gb200_configuration"):
                topology.GB200Topology("gin_roce", world_size=8).prepare(
                    wrong, HOSTS, 0, 4, fake_torch(), dist)
            self.assertTrue(dist.records)
        dist = Dist8(); dist.mismatch = True
        with self.assertRaisesRegex(RuntimeError, "differs across ranks"):
            topology.GB200Topology("gin_roce", world_size=8).prepare(args, HOSTS, 0, 4, fake_torch(), dist)

    def test_actual_ep8_topology_aliases_and_native_cross_os_requirement(self):
        for mode in ("gin", "native"):
            observed = probe.validate_records(records(mode), mode, 8)
            self.assertEqual(observed["lsa_team_size"], 4 if mode == "gin" else 8)
            self.assertEqual(observed["same_clique_cross_os_peers_required"], mode == "native")
            for rank in range(8):
                peers = probe.expected_peers(rank, mode, 8)
                raw = [4096 * (peer + 1) if peer in peers else 0 for peer in range(8)]
                adjusted = [value + 256 if value else 0 for value in raw]
                self.assertEqual(topology.validate_adjusted_aliases(
                    raw, adjusted, 256, 0x900000, rank,
                    "gin_roce" if mode == "gin" else "native_nvl", world_size=8), peers)
                for peer in peers:
                    bad = adjusted.copy(); bad[peer] = 0
                    with self.assertRaises(AssertionError):
                        topology.validate_adjusted_aliases(raw, bad, 256, 0x900000, rank,
                            "gin_roce" if mode == "gin" else "native_nvl", world_size=8)
        wrong = records("gin"); wrong[0]["peer_alias_ranks"] = list(range(8))
        with self.assertRaises(AssertionError):
            probe.validate_records(wrong, "gin", 8)

    def test_live_context_width_backend_and_resources_are_not_environment(self):
        good = [{"rank": rank, "world_size": 8, "lsa_rank": rank % 4, "lsa_size": 4,
                 "gin_type_string": "gdaki", "context_count": 9, "queue_depth": 64,
                 "signal_count": 2, "connection_count": 1} for rank in range(8)]
        probe.validate_context_records(good, 8)
        for key, value in (("lsa_size", 8), ("world_size", 16), ("gin_type_string", "proxy"),
                           ("context_count", 5), ("queue_depth", 32), ("connection_count", 0)):
            wrong = deepcopy(good); wrong[0][key] = value
            with self.assertRaises(AssertionError):
                probe.validate_context_records(wrong, 8)

    def test_descriptor_evidence_uses_actual_ep8_width_without_relaxing_other_fields(self):
        _, args, _ = parse()
        policy = SimpleNamespace(lsa_size=4, require_validated_buffer=mock.Mock(),
                                 expected_lsa_ranks=lambda rank: list(range(4)))
        context = _FakeContext(world_size=8, lsa_size=4, completion_batch=8, outbox_depth=64,
                               active_fast_path=True, bulk_combine=True, direct_dispatch=True)
        buffer = _FakeBuffer(context)
        buffer.buffer_ptrs = buffer.buffer_ptrs[:4] + [0] * 4
        buffer.handle.buffer_ptrs = list(buffer.buffer_ptrs)
        with mock.patch.dict(os.environ, runner.fixed_environment("gin_roce")):
            value = accuracy._gin_transport_evidence(buffer, args, 0, 8, HOSTS, "NCCL", object(),
                                                      _FakeEvidenceDist(), topology=policy)
        self.assertTrue(value["gin_enabled"])
        policy.require_validated_buffer.assert_called_once()
        context.snapshot["lsa_size"] = 8
        with self.assertRaisesRegex(RuntimeError, "lsa_size"):
            accuracy._gin_transport_evidence(buffer, args, 0, 8, HOSTS, "NCCL", object(),
                                             _FakeEvidenceDist(), topology=policy)

    def test_matched_actual_route_histograms_keep_0_50_100_and_all_owner_work(self):
        for tokens in (32, 40, 48):
            actual = route_records(tokens)
            result = accuracy._finalize_matched_route_evidence(actual, tokens, 16, 56, DOMAINS)
            self.assertEqual(result["global_routes"], 8 * tokens * 16)
            self.assertTrue(result["expert_occupancy_matched"])
            self.assertTrue(result["m_block_occupancy_matched"])
            for mode, fanout in (("all_same_host", (3, 0)), ("half_remote", (3, 3)),
                                 ("all_remote", (0, 3))):
                evidence = result["routes"][mode]
                self.assertEqual(evidence["per_owner_received_routes"], [tokens * 16] * 8)
                self.assertEqual(evidence["per_source_same_host_owner_fanout"], [fanout[0]] * 8)
                self.assertEqual(evidence["per_source_cross_host_owner_fanout"], [fanout[1]] * 8)
            wrong = deepcopy(actual); wrong[0]["routes"]["all_remote"]["sent_owner_expert_histograms"][4][0] += 1
            with self.assertRaises(AssertionError):
                accuracy._finalize_matched_route_evidence(wrong, tokens, 16, 56, DOMAINS)

    def test_single_remote_source_activity_is_still_exact_at_ep8(self):
        for tokens in (32, 40, 48):
            cross = []
            for rank in range(8):
                cross.append(sum(DOMAINS[accuracy._owner_for(accuracy.GIN_ACTIVITY_ASYMMETRIC_MODE,
                    rank, token, slot, DOMAINS, num_tokens=tokens)] != DOMAINS[rank]
                    for token in range(tokens) for slot in range(16)))
            self.assertEqual(cross, [tokens * 16] + [0] * 7)

    def test_paired_transition_keeps_768_to1024_capacity_boundary_and_actual_owner(self):
        class Tensor:
            def __init__(self, shape): self.rows = [[None] * shape[1] for _ in range(shape[0])]
            def __setitem__(self, key, value): self.rows[key[0]][key[1]] = value
            def cuda(self): return self
        torch = SimpleNamespace(int64="i", float32="f", empty=lambda shape, **kwargs: Tensor(shape))
        for rank in range(8):
            for tokens in (48, 64, 48):
                ids, _ = accuracy._build_bulk_transition_route_tensors(rank, tokens, 16, 56, DOMAINS, torch)
                values = [item for row in ids.rows for item in row]
                self.assertEqual(len(values), tokens * 16)
                self.assertEqual({item // 56 for item in values}, {(rank + 4) % 8})
        source = inspect.getsource(accuracy._worker)
        self.assertIn("_run_gin_bulk_transition_validation(", source)
        self.assertIn("_run_payload_epoch_validation(harness, torch, dist)", source)


class Buffer8(FakeDeepEPBuffer):
    num_nvlink_ranks, num_scaleup_ranks = 4, 8
    def get_physical_domain_size(self): return 2, 4
    def get_logical_domain_size(self): return 1, 8


class DeepEP8Contracts(unittest.TestCase):
    def test_live_domains_are_ep8_and_wrong_ep16_rejected_before_capture(self):
        comparator = SimpleNamespace(buffer=Buffer8(), api_evidence={
            "physical_domain_size": [2, 4], "logical_domain_size": [1, 8], "allow_hybrid_mode": False})
        record = runner._deepep_domain_record(comparator, 0, 8)
        self.assertEqual(record["logical_domain_size"], [1, 8])
        harness = SimpleNamespace(rank=0, dist=Dist8(), torch=SimpleNamespace(cuda=mock.Mock()))
        adapter = SimpleNamespace(evidence={}, install=lambda: mock.patch.dict(os.environ, {}))
        comparator.buffer = FakeDeepEPBuffer()
        with mock.patch.object(runner, "prepare_converter_adapter", return_value=adapter), mock.patch.object(
                runner.matched, "DeepEPTRTLLM", return_value=comparator), mock.patch.object(
                runner.matched, "_benchmark_pair") as capture, self.assertRaisesRegex(RuntimeError, "physical"):
            runner._compare_gin(harness, {}, None)
        capture.assert_not_called()
        comparator.buffer.destroy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
