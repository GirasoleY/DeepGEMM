import inspect
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import test_mega_moe_accuracy as accuracy


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _args(**overrides):
    values = {
        "num_experts": 896,
        "num_topk": 16,
        "hidden": 3584,
        "intermediate_hidden": 3072,
        "mma_type": "fp8xfp4",
        "num_tokens": 24,
        "num_max_tokens_per_rank": 384,
        "num_shared_experts": 0,
        "require_gin": True,
        "gin_completion_batch": 1,
        "gin_combine_chunk_bytes": 7168,
        "gin_outbox_depth": 8,
        "gin_combine_issue_wave": 8,
        "gin_queue_depth": 64,
        "gin_active_fast_path": False,
        "gin_bulk_combine": False,
        "gin_direct_dispatch": False,
        "gin_local_ablation_stages": (),
        "benchmark_replays": 0,
        "benchmark_warmups": 10,
        "benchmark_conditioning_replays": 60,
        "benchmark_routes": ("all_same_host",),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeTensor:
    def numel(self):
        return 4096

    def element_size(self):
        return 1

    def data_ptr(self):
        return 0x1000


class _FakeContext:
    active = True
    gin_type_string = "gdaki"
    buffer_bytes = 4096

    def __init__(self, **overrides):
        self.snapshot = {
            "enabled": 1,
            "rank": 0,
            "world_size": 16,
            "lsa_rank": 0,
            "lsa_size": 8,
            "context_count": 9,
            "requested_context_count": 9,
            "connection_count": 8,
            "requested_signal_count": 2,
            "signal_count": 66,
            "queue_depth": 64,
            "world_barrier_count": 4,
            "completion_batch": 1,
            "combine_chunk_bytes": 7168,
            "outbox_depth": 8,
            "combine_issue_wave": 8,
            "active_fast_path": False,
            "bulk_combine": False,
            "direct_dispatch": False,
            "window": 1234,
            "dev_comm_bytes": 256,
        }
        self.snapshot.update(overrides)

    def launch_descriptor_snapshot(self):
        return dict(self.snapshot)


class _FakeBuffer:
    def __init__(
        self, context=None, events=None, destroy_error=None, abort_error=None
    ):
        self.buffer = _FakeTensor()
        self.buffer_ptrs = [0x1000 + peer * 0x1000 for peer in range(8)] + [0] * 8
        self.handle = SimpleNamespace(
            buffer_ptrs=list(self.buffer_ptrs),
            offset=0,
        )
        self.gin_context = context
        self.events = events if events is not None else []
        self.destroy_error = destroy_error
        self.abort_error = abort_error

    @property
    def gin_enabled(self):
        return self.gin_context is not None and self.gin_context.active

    def destroy(self):
        self.events.append("buffer.destroy")
        if self.destroy_error is not None:
            raise self.destroy_error

    def abort(self):
        self.events.append("buffer.abort")
        if self.abort_error is not None:
            raise self.abort_error


class _FakeDist:
    def __init__(self, events, initialized=True):
        self.events = events
        self.initialized = initialized

    def is_initialized(self):
        self.events.append("dist.is_initialized")
        return self.initialized

    def barrier(self):
        self.events.append("dist.barrier")

    def destroy_process_group(self):
        self.events.append("dist.destroy_process_group")


class _FakeEvidenceDist:
    def all_gather_object(self, output, local_record):
        local_record = {
            **local_record,
            "flags": {
                name: "0" for name in accuracy.GIN_VALIDATED_FLAG_ENVS
            },
        }
        output[:] = [
            {**local_record, "rank": rank} for rank in range(len(output))
        ]


class TestMegaMoeAccuracyGinContract(unittest.TestCase):
    def test_decode_mns_counts_target_plus_draft_tokens(self):
        with mock.patch(
            "sys.argv",
            [
                "test_mega_moe_accuracy.py",
                "--k3",
                "--decode-mns",
                "8",
                "--draft-width",
                "3",
            ],
        ):
            args = accuracy._parse_args()

        self.assertEqual(args.num_tokens, 32)
        self.assertEqual(args.num_max_tokens_per_rank, 384)

    def test_decode_mns_rejects_nonpositive_sequences(self):
        with mock.patch(
            "sys.argv",
            ["test_mega_moe_accuracy.py", "--decode-mns", "0"],
        ), mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            accuracy._parse_args()

    def test_decode_mns_rejects_negative_draft_width(self):
        with mock.patch(
            "sys.argv",
            [
                "test_mega_moe_accuracy.py",
                "--decode-mns",
                "8",
                "--draft-width",
                "-1",
            ],
        ), mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            accuracy._parse_args()

    def test_matched_routes_have_identical_expert_work_on_2x8(self):
        hostnames = ["host-a"] * 8 + ["host-b"] * 8
        experts_per_rank = 56

        def received_histograms(mode, num_tokens):
            received = [Counter() for _ in range(16)]
            received_remote = [Counter() for _ in range(16)]
            sent = [[Counter() for _ in range(16)] for _ in range(16)]
            fanout = [set() for _ in range(16)]
            cross_host_routes = [0] * 16
            for source in range(16):
                for token in range(num_tokens):
                    for slot in range(16):
                        owner = accuracy._owner_for(
                            mode,
                            source,
                            token,
                            slot,
                            hostnames,
                            num_topk=16,
                            num_tokens=num_tokens,
                        )
                        self.assertIsNotNone(owner)
                        if mode == "all_same_host":
                            self.assertNotEqual(owner, source)
                            self.assertEqual(hostnames[owner], hostnames[source])
                        elif mode == "all_remote":
                            self.assertNotEqual(hostnames[owner], hostnames[source])
                        local_expert = (token * 16 + slot) % experts_per_rank
                        received[owner][local_expert] += 1
                        sent[source][owner][local_expert] += 1
                        if hostnames[owner] != hostnames[source]:
                            received_remote[owner][local_expert] += 1
                        fanout[source].add(owner)
                        cross_host_routes[source] += (
                            hostnames[owner] != hostnames[source]
                        )
            return received, received_remote, sent, fanout, cross_host_routes

        for num_tokens in (32, 40, 48):
            (
                local,
                _local_remote,
                _local_sent,
                local_fanout,
                local_cross,
            ) = received_histograms(
                "all_local", num_tokens
            )
            (
                same_host,
                _same_remote,
                same_sent,
                same_host_fanout,
                same_cross,
            ) = received_histograms("all_same_host", num_tokens)
            (
                half_remote,
                half_remote_rows,
                half_sent,
                half_remote_fanout,
                half_cross,
            ) = received_histograms("half_remote", num_tokens)
            (
                remote,
                _remote_rows,
                remote_sent,
                remote_fanout,
                remote_cross,
            ) = received_histograms("all_remote", num_tokens)
            self.assertEqual(same_host, local)
            self.assertEqual(half_remote, local)
            self.assertEqual(remote, local)
            self.assertTrue(all(len(peers) == 1 for peers in local_fanout))
            self.assertTrue(all(len(peers) == 7 for peers in same_host_fanout))
            self.assertTrue(all(len(peers) == 14 for peers in half_remote_fanout))
            self.assertTrue(all(len(peers) == 7 for peers in remote_fanout))
            routes_per_source = num_tokens * 16
            self.assertEqual(local_cross, [0] * 16)
            self.assertEqual(same_cross, [0] * 16)
            self.assertEqual(half_cross, [routes_per_source // 2] * 16)
            self.assertEqual(remote_cross, [routes_per_source] * 16)

            rows_per_expert, long_experts = divmod(
                routes_per_source, experts_per_rank
            )
            expected_histogram = (
                [rows_per_expert] * (experts_per_rank - long_experts)
                + [rows_per_expert + 1] * long_experts
            )
            for histogram, remote_histogram in zip(
                half_remote, half_remote_rows
            ):
                self.assertEqual(len(histogram), experts_per_rank)
                self.assertEqual(
                    sum((count + 31) // 32 for count in histogram.values()),
                    experts_per_rank,
                )
                self.assertEqual(sorted(histogram.values()), expected_histogram)
                for expert, count in histogram.items():
                    self.assertGreater(remote_histogram[expert], 0)
                    self.assertLess(remote_histogram[expert], count)

            mode_data = {
                "all_same_host": (same_host, same_sent, same_cross),
                "half_remote": (half_remote, half_sent, half_cross),
                "all_remote": (remote, remote_sent, remote_cross),
            }
            records = []
            for source in range(16):
                routes = {}
                for mode, (received_mode, sent_mode, cross_mode) in mode_data.items():
                    sent_histograms = [
                        [sent_mode[source][owner].get(expert, 0)
                         for expert in range(experts_per_rank)]
                        for owner in range(16)
                    ]
                    routes[mode] = {
                        "valid_routes": routes_per_source,
                        "cross_host_routes": cross_mode[source],
                        "sent_owner_expert_histograms": sent_histograms,
                        "expected_received_expert_histogram": [
                            received_mode[source].get(expert, 0)
                            for expert in range(experts_per_rank)
                        ],
                    }
                records.append({
                    "rank": source,
                    "runtime_block_m": 32,
                    "routes": routes,
                })

            evidence = accuracy._finalize_matched_route_evidence(
                records,
                num_tokens=num_tokens,
                num_topk=16,
                experts_per_rank=experts_per_rank,
                hostnames=hostnames,
            )
            self.assertTrue(evidence["owner_occupancy_matched"])
            self.assertTrue(evidence["expert_occupancy_matched"])
            self.assertTrue(evidence["m_block_occupancy_matched"])
            self.assertEqual(evidence["block_m_source"],
                             "deep_gemm.get_block_m_for_mega_moe")
            self.assertEqual(
                evidence["routes"]["all_same_host"]
                ["per_source_same_host_owner_fanout"],
                [7] * 16,
            )
            self.assertEqual(
                evidence["routes"]["half_remote"]
                ["per_source_cross_host_owner_fanout"],
                [7] * 16,
            )
            self.assertEqual(
                evidence["routes"]["all_remote"]
                ["per_source_same_host_owner_fanout"],
                [0] * 16,
            )
            if num_tokens == 32:
                records[0]["runtime_block_m"] = 16
                with self.assertRaisesRegex(
                    AssertionError, "BLOCK_M differs across ranks"
                ):
                    accuracy._finalize_matched_route_evidence(
                        records,
                        num_tokens=num_tokens,
                        num_topk=16,
                        experts_per_rank=experts_per_rank,
                        hostnames=hostnames,
                    )

    def test_required_gin_accepts_only_integrated_fp8_kernel(self):
        with self.assertRaisesRegex(ValueError, "BF16 MegaMoE launch"):
            accuracy._validate_args(_args(mma_type="bf16xbf16"), 16)

    def test_required_gin_accepts_only_initial_2x8_target(self):
        with self.assertRaisesRegex(ValueError, "exactly 16 ranks"):
            accuracy._validate_args(_args(), 8)

    def test_active_fast_path_requires_gin(self):
        with self.assertRaisesRegex(ValueError, "requires --require-gin"):
            accuracy._validate_args(
                _args(require_gin=False, gin_active_fast_path=True), 16
            )

    def test_active_fast_path_rejects_local_ablation(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            accuracy._validate_args(
                _args(
                    gin_active_fast_path=True,
                    gin_local_ablation_stages=(0,),
                ),
                16,
            )

    def test_bulk_combine_accepts_only_target_decode_contract(self):
        accuracy._validate_args(
            _args(
                gin_active_fast_path=True,
                gin_bulk_combine=True,
                gin_outbox_depth=64,
                num_tokens=48,
            ),
            16,
        )

    def test_bulk_combine_requires_activity_consensus(self):
        with self.assertRaisesRegex(ValueError, "active-fast-path"):
            accuracy._validate_args(
                _args(gin_bulk_combine=True, gin_outbox_depth=64), 16
            )

    def test_bulk_combine_accepts_more_than_48_tokens_for_uniform_fallback(self):
        accuracy._validate_args(
            _args(
                gin_active_fast_path=True,
                gin_bulk_combine=True,
                gin_outbox_depth=64,
                num_tokens=64,
            ),
            16,
        )

    def test_fast_path_transition_trigger_covers_every_feature_composition(self):
        expected = {
            (False, False): False,
            (True, False): True,
            (False, True): True,
            (True, True): True,
        }
        for (bulk_combine, direct_dispatch), enabled in expected.items():
            with self.subTest(
                bulk_combine=bulk_combine,
                direct_dispatch=direct_dispatch,
            ):
                args = _args(
                    gin_bulk_combine=bulk_combine,
                    gin_direct_dispatch=direct_dispatch,
                )
                self.assertEqual(
                    accuracy._gin_fast_path_transition_requested(args),
                    enabled,
                )

    def test_fast_path_transition_requires_an_inactive_canary_row_above_64(self):
        for bulk_combine, direct_dispatch in (
            (True, False),
            (False, True),
            (True, True),
        ):
            with self.subTest(
                bulk_combine=bulk_combine,
                direct_dispatch=direct_dispatch,
            ), self.assertRaisesRegex(ValueError, "greater than 64"):
                accuracy._validate_args(
                    _args(
                        gin_active_fast_path=True,
                        gin_bulk_combine=bulk_combine,
                        gin_direct_dispatch=direct_dispatch,
                        gin_outbox_depth=64,
                        num_tokens=48,
                        num_max_tokens_per_rank=64,
                    ),
                    16,
                )

    def test_fast_path_transition_contract_reuses_48_graph_across_exact_r75(self):
        self.assertEqual(
            accuracy.GIN_BULK_TRANSITION_SEQUENCE,
            ("48a", "64", "48b"),
        )
        self.assertEqual(
            [
                accuracy.GIN_BULK_TRANSITION_COUNTS[name]
                for name in accuracy.GIN_BULK_TRANSITION_SEQUENCE
            ],
            [48, 64, 48],
        )
        self.assertEqual(
            [
                accuracy.GIN_BULK_TRANSITION_INPUT_EPOCHS[name]
                for name in accuracy.GIN_BULK_TRANSITION_SEQUENCE
            ],
            [0, 1, 2],
        )
        self.assertEqual(accuracy.GIN_BULK_COMBINE_CAPACITY, 48 * 16)
        self.assertEqual(accuracy.GIN_FAST_PATH_MAX_TOKENS, 48)
        self.assertEqual(accuracy.GIN_FALLBACK_PROBE_TOKENS, 64)
        transition_source = inspect.getsource(
            accuracy._run_gin_bulk_transition_validation
        )
        self.assertIn(
            "if not _gin_fast_path_transition_requested(args)",
            transition_source,
        )
        self.assertIn('"fallback_protocol": "exact_r75"', transition_source)
        capture = transition_source.index(
            "graph_holder[count] = torch.cuda.CUDAGraph()"
        )
        preparation = transition_source.rindex(
            "tail_expected = _prepare_bulk_transition_phase(", 0, capture
        )
        capture_setup = transition_source[preparation:capture]
        self.assertIn("torch.cuda.synchronize()", capture_setup)
        self.assertIn("dist.barrier()", capture_setup)
        self.assertLess(
            capture_setup.index("torch.cuda.synchronize()"),
            capture_setup.index("dist.barrier()"),
        )
        replay = transition_source.index(
            "graph_holder[count].replay()", capture
        )
        validation = transition_source.index(
            "_validate_bulk_transition_phase(", replay
        )
        self.assertLess(capture, replay)
        self.assertLess(replay, validation)

    def test_bulk_combine_rejects_non_target_shape_or_outbox(self):
        for override in (
            {"intermediate_hidden": 4096},
            {"gin_outbox_depth": 8},
        ):
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, "outbox64"):
                    accuracy._validate_args(
                        _args(
                            gin_active_fast_path=True,
                            gin_bulk_combine=True,
                            **override,
                        ),
                        16,
                    )

    def test_direct_dispatch_is_independent_of_bulk_combine(self):
        accuracy._validate_args(
            _args(
                gin_active_fast_path=True,
                gin_direct_dispatch=True,
                gin_bulk_combine=False,
                gin_outbox_depth=8,
                num_tokens=48,
            ),
            16,
        )

    def test_direct_and_bulk_composition_accepts_target_contract(self):
        accuracy._validate_args(
            _args(
                gin_active_fast_path=True,
                gin_direct_dispatch=True,
                gin_bulk_combine=True,
                gin_outbox_depth=64,
                num_tokens=48,
            ),
            16,
        )

    def test_direct_dispatch_requires_activity_consensus(self):
        with self.assertRaisesRegex(ValueError, "active-fast-path"):
            accuracy._validate_args(
                _args(gin_direct_dispatch=True), 16
            )

    def test_direct_dispatch_accepts_more_than_48_tokens_for_exact_r75_fallback(self):
        accuracy._validate_args(
            _args(
                gin_active_fast_path=True,
                gin_direct_dispatch=True,
                num_tokens=64,
            ),
            16,
        )

    def test_direct_dispatch_rejects_non_target_shape_or_allocation(self):
        for override, message in (
            ({"intermediate_hidden": 4096}, "EP16/E896"),
            ({"num_max_tokens_per_rank": 383}, "at least 384|>= 384"),
        ):
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, message):
                    accuracy._validate_args(
                        _args(
                            gin_active_fast_path=True,
                            gin_direct_dispatch=True,
                            **override,
                        ),
                        16,
                    )

    def test_single_remote_source_route_has_exact_asymmetric_activity(self):
        hostnames = ["host-a"] * 8 + ["host-b"] * 8
        routes_per_source = 32 * 16
        cross_routes = []
        for source in range(16):
            count = 0
            for token in range(32):
                for slot in range(16):
                    owner = accuracy._owner_for(
                        accuracy.GIN_ACTIVITY_ASYMMETRIC_MODE,
                        source,
                        token,
                        slot,
                        hostnames,
                        num_topk=16,
                        num_tokens=32,
                    )
                    count += hostnames[owner] != hostnames[source]
            cross_routes.append(count)
        self.assertEqual(cross_routes, [routes_per_source] + [0] * 15)

    def test_activity_transition_schedule_reuses_both_signals(self):
        self.assertEqual(
            accuracy.GIN_ACTIVITY_TRANSITION_SEQUENCE,
            (
                "all_same_host",
                "single_remote_source",
                "all_same_host",
                "half_remote",
                "all_same_host",
                "all_remote",
                "all_same_host",
            ),
        )
        self.assertGreaterEqual(accuracy.GIN_ACTIVITY_TRANSITION_CYCLES, 3)

    def test_balanced_six_launch_schedule_has_exact_204_samples_and_edges(self):
        schedule = accuracy._balanced_six_launch_schedule(204)
        self.assertEqual(len(schedule), 3 * 204)
        self.assertEqual(
            Counter(schedule),
            Counter({route: 204 for route in accuracy.MATCHED_BENCHMARK_ROUTES}),
        )
        transitions = accuracy._directed_transition_counts(schedule)
        self.assertEqual(
            transitions,
            {
                "all_same_host->half_remote": 102,
                "half_remote->all_remote": 102,
                "all_remote->all_same_host": 102,
                "all_same_host->all_remote": 102,
                "all_remote->half_remote": 102,
                "half_remote->all_same_host": 102,
            },
        )

    def test_six_launch_block_pairing_uses_two_occurrences_per_route(self):
        self.assertEqual(
            accuracy._six_launch_block_means([1.0, 3.0, 5.0, 9.0]),
            [2.0, 7.0],
        )
        with self.assertRaisesRegex(ValueError, "even sample count"):
            accuracy._six_launch_block_means([1.0, 2.0, 3.0])

    def test_timing_halves_retain_separate_drift_summaries(self):
        halves = accuracy._timing_halves([1.0, 2.0, 5.0, 7.0])
        self.assertEqual(halves["first"]["mean"], 1.5)
        self.assertEqual(halves["second"]["mean"], 6.0)

    def test_matched_benchmark_contract_accepts_runbook_schedule(self):
        accuracy._validate_args(
            _args(
                num_tokens=32,
                benchmark_replays=204,
                benchmark_conditioning_replays=60,
                benchmark_routes=accuracy.MATCHED_BENCHMARK_ROUTES,
            ),
            16,
        )

    def test_matched_benchmark_contract_rejects_odd_or_partial_schedule(self):
        with self.assertRaisesRegex(ValueError, "must be even"):
            accuracy._validate_args(
                _args(
                    num_tokens=32,
                    benchmark_replays=203,
                    benchmark_routes=accuracy.MATCHED_BENCHMARK_ROUTES,
                ),
                16,
            )
        with self.assertRaisesRegex(ValueError, "requires exactly"):
            accuracy._validate_args(
                _args(
                    num_tokens=32,
                    benchmark_replays=204,
                    benchmark_routes=("all_same_host", "all_remote"),
                ),
                16,
            )

    def test_cli_exposes_fixed_conditioning_count(self):
        with mock.patch(
            "sys.argv",
            [
                "test_mega_moe_accuracy.py",
                "--k3",
                "--decode-mns",
                "8",
                "--draft-width",
                "3",
                "--require-gin",
                "--benchmark-replays",
                "204",
                "--benchmark-conditioning-replays",
                "60",
            ],
        ):
            args = accuracy._parse_args()
        self.assertEqual(args.benchmark_routes, accuracy.MATCHED_BENCHMARK_ROUTES)
        self.assertEqual(args.benchmark_replays, 204)
        self.assertEqual(args.benchmark_conditioning_replays, 60)

    def test_cli_rejects_partial_matched_route_set(self):
        with mock.patch(
            "sys.argv",
            [
                "test_mega_moe_accuracy.py",
                "--k3",
                "--num-tokens",
                "32",
                "--require-gin",
                "--benchmark-replays",
                "204",
                "--benchmark-routes",
                "all_same_host",
                "all_remote",
            ],
        ), mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            accuracy._parse_args()

    def test_fused_accuracy_accepts_completion_batch_sweep(self):
        for completion_batch in (1, 2, 4, 8):
            with self.subTest(completion_batch=completion_batch):
                accuracy._validate_args(
                    _args(gin_completion_batch=completion_batch), 16
                )

    def test_fused_accuracy_accepts_combine_issue_wave_sweep(self):
        for combine_issue_wave in (1, 2, 4, 8):
            with self.subTest(combine_issue_wave=combine_issue_wave):
                accuracy._validate_args(
                    _args(gin_combine_issue_wave=combine_issue_wave), 16
                )

    def test_fused_accuracy_rejects_unsupported_combine_issue_wave(self):
        with self.assertRaisesRegex(ValueError, "one of 1, 2, 4, or 8"):
            accuracy._validate_args(_args(gin_combine_issue_wave=3), 16)

    def test_fused_accuracy_accepts_full_shape_outbox_depth(self):
        accuracy._validate_args(_args(gin_outbox_depth=64), 16)

    def test_fused_accuracy_rejects_unsupported_outbox_depth(self):
        with self.assertRaisesRegex(ValueError, "one of 4, 8, 16, or 64"):
            accuracy._validate_args(_args(gin_outbox_depth=32), 16)

    def test_fused_accuracy_rejects_undersized_queue(self):
        with self.assertRaisesRegex(ValueError, "at least 64"):
            accuracy._validate_args(_args(gin_queue_depth=32), 16)

    def test_local_ablation_requires_stage_zero_baseline(self):
        with self.assertRaisesRegex(ValueError, "include stage 0"):
            accuracy._validate_args(
                _args(gin_local_ablation_stages=(1, 2, 3, 4)), 16
            )

    def test_local_ablation_requires_only_same_host_benchmark(self):
        with self.assertRaisesRegex(ValueError, "exactly.*all_same_host"):
            accuracy._validate_args(
                _args(
                    gin_local_ablation_stages=(0, 1),
                    benchmark_routes=("all_same_host", "all_remote"),
                ),
                16,
            )

    def test_local_ablation_accepts_complete_ladder(self):
        accuracy._validate_args(
            _args(gin_local_ablation_stages=(0, 1, 2, 3, 4)), 16
        )

    def test_local_ablation_owner_validation_accepts_same_lsa_nonself(self):
        accuracy._validate_local_ablation_owner_values(
            rank=3,
            expected_slots=4,
            owner_values=(0, 1, 4, 7),
        )

    def test_local_ablation_owner_validation_rejects_missing_slot(self):
        with self.assertRaisesRegex(AssertionError, "every top-k slot"):
            accuracy._validate_local_ablation_owner_values(
                rank=3,
                expected_slots=4,
                owner_values=(0, 1, 4),
            )

    def test_local_ablation_owner_validation_rejects_self_or_remote(self):
        with self.assertRaisesRegex(AssertionError, "invalid owners=\\[3, 8\\]"):
            accuracy._validate_local_ablation_owner_values(
                rank=3,
                expected_slots=4,
                owner_values=(0, 3, 7, 8),
            )

    def test_combine_chunk_must_tile_a_full_output_row(self):
        with self.assertRaisesRegex(ValueError, "must divide one BF16 output row"):
            accuracy._validate_args(
                _args(hidden=512, intermediate_hidden=512,
                      gin_combine_chunk_bytes=1792),
                16,
            )

    def test_transport_evidence_checks_and_reports_exact_tuning(self):
        evidence = accuracy._gin_transport_evidence(
            _FakeBuffer(_FakeContext()),
            _args(),
            rank=0,
            world_size=16,
            hostnames=["host-a"] * 8 + ["host-b"] * 8,
            symmetric_memory_backend="NCCL",
            symmetric_memory_registration=object(),
            dist=_FakeEvidenceDist(),
        )

        self.assertEqual(evidence["requested"], "gin")
        self.assertTrue(evidence["cross_host_payload_routes"])
        self.assertEqual(evidence["gin_type"], "gdaki")
        self.assertEqual(
            evidence["launch_descriptor"]["combine_chunk_bytes"], 7168
        )
        self.assertEqual(evidence["launch_descriptor"]["combine_issue_wave"], 8)
        self.assertEqual(
            evidence["launch_descriptor"]["requested_signal_count"], 2
        )
        self.assertGreaterEqual(
            evidence["launch_descriptor"]["signal_count"], 2
        )
        self.assertFalse(evidence["launch_descriptor"]["bulk_combine"])
        self.assertFalse(evidence["launch_descriptor"]["direct_dispatch"])
        self.assertEqual(
            evidence["jit_experiment_flags"],
            {
                accuracy.GIN_DISPATCH_WARP_SCAN_ENV: False,
                accuracy.GIN_COOP_DIRECT_PACK_ENV: False,
                accuracy.GIN_PRECONSENSUS_PACK_ENV: False,
                accuracy.GIN_SINGLE_COMBINE_CONTEXT_ENV: False,
            },
        )
        self.assertEqual(evidence["registered_buffer_bytes"], 4096)

    def test_transport_evidence_rejects_tuning_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "completion_batch=4"):
            accuracy._gin_transport_evidence(
                _FakeBuffer(_FakeContext(completion_batch=4)),
                _args(),
                rank=0,
                world_size=16,
                hostnames=["host-a"] * 8 + ["host-b"] * 8,
                symmetric_memory_backend="NCCL",
                symmetric_memory_registration=object(),
                dist=_FakeEvidenceDist(),
            )

    def test_transport_evidence_rejects_combine_issue_wave_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "combine_issue_wave=4"):
            accuracy._gin_transport_evidence(
                _FakeBuffer(_FakeContext(combine_issue_wave=4)),
                _args(),
                rank=0,
                world_size=16,
                hostnames=["host-a"] * 8 + ["host-b"] * 8,
                symmetric_memory_backend="NCCL",
                symmetric_memory_registration=object(),
                dist=_FakeEvidenceDist(),
            )

    def test_transport_evidence_rejects_direct_dispatch_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "direct_dispatch=True"):
            accuracy._gin_transport_evidence(
                _FakeBuffer(_FakeContext(direct_dispatch=True)),
                _args(),
                rank=0,
                world_size=16,
                hostnames=["host-a"] * 8 + ["host-b"] * 8,
                symmetric_memory_backend="NCCL",
                symmetric_memory_registration=object(),
                dist=_FakeEvidenceDist(),
            )

    def test_transport_evidence_rejects_three_barrier_slots(self):
        with self.assertRaisesRegex(RuntimeError, "world_barrier_count=3"):
            accuracy._gin_transport_evidence(
                _FakeBuffer(_FakeContext(world_barrier_count=3)),
                _args(),
                rank=0,
                world_size=16,
                hostnames=["host-a"] * 8 + ["host-b"] * 8,
                symmetric_memory_backend="NCCL",
                symmetric_memory_registration=object(),
                dist=_FakeEvidenceDist(),
            )

    def test_transport_evidence_requires_cross_host_payload(self):
        with self.assertRaisesRegex(RuntimeError, "all ranks are on one host"):
            accuracy._gin_transport_evidence(
                _FakeBuffer(_FakeContext()),
                _args(),
                rank=0,
                world_size=16,
                hostnames=["host-a"] * 16,
                symmetric_memory_backend="NCCL",
                symmetric_memory_registration=object(),
                dist=_FakeEvidenceDist(),
            )


class TestMegaMoeActiveConsensusSourceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel = (_REPO_ROOT / "deep_gemm/include/deep_gemm/impls/"
                      "sm100_fp8_fp4_mega_moe.cuh").read_text()
        cls.comm = (_REPO_ROOT / "deep_gemm/include/deep_gemm/comm/"
                    "mega_moe_gin.cuh").read_text()
        cls.layout = (_REPO_ROOT / "deep_gemm/include/deep_gemm/layout/"
                      "mega_moe.cuh").read_text()
        cls.jit = (_REPO_ROOT / "csrc/jit_kernels/impls/"
                   "sm100_fp8_fp4_mega_moe.hpp").read_text()
        cls.compile_tu = (
            _REPO_ROOT / "tests/compile_sm100_megamoe_gin.cu"
        ).read_text()

    def test_pair_flags_use_epoch_tagged_inline_mailbox(self):
        start = self.comm.index(
            "NCCL_DEVICE_INLINE uint32_t mega_moe_gin_exchange_pair_flags"
        )
        end = self.comm.index(
            "NCCL_DEVICE_INLINE bool mega_moe_gin_exchange_pair_activity",
            start,
        )
        exchange = self.comm[start:end]
        self.assertIn("gin.putValue<uint64_t>", exchange)
        self.assertIn("(generation << 2) | local_flags |", exchange)
        self.assertIn("static_cast<uint64_t>(local_flags) << 62", exchange)
        self.assertIn("ptx::ld_acq_sys(local_mailbox_ptr)", exchange)
        self.assertIn("received_generation == generation", exchange)
        self.assertIn("received_flags == received_flags_copy", exchange)
        self.assertIn("received & 3ull", exchange)
        self.assertIn("ncclGinOptFlagsDefault", exchange)
        self.assertNotIn("ncclGin_StrongSignalAdd", exchange)
        self.assertNotIn("waitSignalFollowShadow", exchange)
        self.assertNotIn("gin.signal(", exchange)
        self.assertNotIn("ncclGin_WeakSignalAdd", exchange)

        consensus = self.kernel.split(
            "A rank-local route test cannot safely select", 1
        )[1].split("comm::nvlink_lsa_barrier", 1)[0]
        self.assertIn("get_gin_pair_mailbox_ptr", consensus)
        self.assertIn(
            "sym_buffer.get_base_ptr(), pair_mailbox_ptr", consensus
        )
        self.assertIn("const uint64_t generation = epoch + 1", consensus)
        self.assertIn("*pair_active_ptr = local_flags | paired_flags", consensus)

        mailbox = self.layout.split("get_gin_pair_mailbox_ptr", 1)[1]
        self.assertIn("72u + parity * sizeof(uint64_t)", mailbox)
        epoch = self.layout.split("get_gin_active_launch_epoch_ptr", 1)[1]
        self.assertIn("math::advance_ptr<uint64_t>(base, 96u)", epoch)

    def test_activity_gate_opt_is_off_by_default_and_jit_specialized(self):
        self.assertIn(
            "#define DG_MEGAMOE_GIN_ACTIVITY_GATE_OPT 0", self.kernel
        )
        self.assertIn(
            "static constexpr bool kMegaMoeGinActivityGateOpt", self.kernel
        )
        self.assertIn(
            "#define DG_MEGAMOE_GIN_ACTIVITY_GATE_OPT {}", self.jit
        )
        self.assertIn("bool gin_activity_gate_opt;", self.jit)
        self.assertIn(
            'get_env<int>("DG_MEGAMOE_GIN_ACTIVITY_GATE_OPT", 0)',
            self.jit,
        )
        self.assertIn(
            "args.gin_activity_gate_opt ? \"1\" : \"0\"", self.jit
        )
        self.assertIn(
            "#define DG_MEGAMOE_GIN_ACTIVITY_GATE_OPT 1", self.compile_tu
        )

    def test_optimized_activity_gate_removes_only_inner_grid_syncs(self):
        optimized = self.kernel.split(
            "Route construction already completed a grid rendezvous", 1
        )[1].split("} else {", 1)[0]
        self.assertIn("comm::nvlink_lsa_barrier<", optimized)
        self.assertIn(
            "/* Pair publication is warp-synchronized above */ false",
            optimized,
        )
        self.assertIn(
            "/* Final decision has its own grid rendezvous */ false",
            optimized,
        )
        self.assertNotIn("comm::grid_sync<", optimized)

        # The original fully synchronized path remains the default branch.
        fallback = self.kernel.split(
            "/* Final decision has its own grid rendezvous */ false);", 1
        )[1].split("if (sm_idx == 0 and warp_idx == 0)", 1)[0]
        self.assertIn("} else {", fallback)
        self.assertIn("comm::nvlink_lsa_barrier<", fallback)

    def test_parallel_pair_loads_reconverge_reduce_and_publish(self):
        pair_store = self.kernel.index(
            "*pair_active_ptr = local_flags | paired_flags"
        )
        pair_fence = self.kernel.index("__threadfence_system();", pair_store)
        pair_sync = self.kernel.index("__syncwarp();", pair_fence)
        lsa_barrier = self.kernel.index(
            "Route construction already completed a grid rendezvous",
            pair_sync,
        )
        post_barrier_sync = self.kernel.index(
            "__syncwarp();",
            self.kernel.index("Lanes 1..31 can return", lsa_barrier),
        )
        lane_load = self.kernel.index(
            "if (lane_idx < lsa_size)", post_barrier_sync
        )
        reduction = self.kernel.index("__reduce_or_sync(", lane_load)
        publication = self.kernel.index(
            "workspace.get_gin_world_active_ptr()", reduction
        )
        final_grid = self.kernel.index(
            "comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>", publication
        )

        self.assertLess(pair_store, pair_fence)
        self.assertLess(pair_fence, pair_sync)
        self.assertLess(pair_sync, lsa_barrier)
        self.assertLess(lsa_barrier, post_barrier_sync)
        self.assertLess(post_barrier_sync, lane_load)
        self.assertLess(lane_load, reduction)
        self.assertLess(reduction, publication)
        self.assertLess(publication, final_grid)
        parallel = self.kernel[
            post_barrier_sync:
            self.kernel.index("} else if (lane_idx == 0)", post_barrier_sync)
        ]
        self.assertIn("const uint32_t peer = lsa_base + lane_idx", parallel)
        self.assertIn("0xffffffffu, lane_pair_flags", parallel)
        self.assertNotIn("local_rank < lsa_size", parallel)


class TestMegaMoeBulkCombineSourceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel = (_REPO_ROOT / "deep_gemm/include/deep_gemm/impls/"
                      "sm100_fp8_fp4_mega_moe.cuh").read_text()
        cls.layout = (_REPO_ROOT / "deep_gemm/include/deep_gemm/layout/"
                      "mega_moe.cuh").read_text()
        cls.api = (_REPO_ROOT / "csrc/apis/mega.hpp").read_text()

    def test_bulk_packets_alias_existing_row_outbox(self):
        self.assertIn("combine_outbox_buffer.base", self.layout)
        self.assertIn("bulk_combine_packet_tail_buffer", self.layout)
        self.assertNotIn("bulk_combine_send_buffer", self.layout)
        self.assertNotIn("bulk_combine_recv_buffer", self.layout)
        row_outbox_bytes = 64 * 192 * 7168
        record_bytes = 16 + 7168
        packet_bytes = 16 + 48 * 16 * record_bytes
        packet_storage_bytes = 2 * 8 * packet_bytes
        max_pool_tokens = 109_056
        return_index_bytes = max_pool_tokens * 4
        self.assertEqual(row_outbox_bytes, 88_080_384)
        self.assertEqual(packet_storage_bytes - row_outbox_bytes, 196_864)
        self.assertEqual(return_index_bytes, 436_224)
        self.assertEqual(
            packet_storage_bytes - row_outbox_bytes + return_index_bytes,
            633_088,
        )

    def test_bulk_selection_is_world_uniform_and_falls_back(self):
        self.assertIn("mega_moe_gin_exchange_pair_flags", self.kernel)
        self.assertIn("get_gin_world_bulk_ineligible_ptr", self.kernel)
        self.assertIn(
            "not use_gin_bulk_combine and",
            self.kernel,
        )
        self.assertEqual(
            self.kernel.count("use_gin_bulk_combine_this_launch();"), 2
        )

    def test_exposed_size_api_rejects_non_target_bulk_layout(self):
        self.assertIn("not gin_bulk_combine or", self.api)
        self.assertIn("num_experts == 896", self.api)
        self.assertIn("gin_outbox_depth == 64", self.api)

    def test_return_index_precedes_l1_ready_release(self):
        return_store = self.kernel.index(
            ".get_bulk_combine_return_index_ptr(pool_token_idx)"
        )
        l1_release = self.kernel.index(
            "workspace.get_l1_full_count_ptr", return_store
        )
        self.assertLess(return_store, l1_release)

    def test_recv_preclear_and_three_phase_combine_order(self):
        recv_clear = self.kernel.index(
            "get_bulk_combine_packet_count_ptr(\n"
            "                    /*send=*/ false, lane_idx)"
        )
        input_barrier = self.kernel.index("kGinInputPutBarrierIdx", recv_clear)
        slab_put = self.kernel.index("mega_moe_gin_put_bulk_combine_packet")
        visibility_barrier = self.kernel.index(
            "mega_moe_gin_world_barrier_all_contexts", slab_put
        )
        scatter = self.kernel.index("Scatter compact owner packets", visibility_barrier)
        third_grid = self.kernel.index(
            "third epilogue grid", scatter
        )
        cleanup_release = self.kernel.index(
            "Release dispatch cleanup", third_grid
        )
        self.assertLess(recv_clear, input_barrier)
        self.assertLess(slab_put, visibility_barrier)
        self.assertLess(visibility_barrier, scatter)
        self.assertLess(scatter, third_grid)
        self.assertLess(third_grid, cleanup_release)

    def test_transport_evidence_requires_contiguous_2x8_host_ranks(self):
        with self.assertRaisesRegex(RuntimeError, "contiguous 2x8 rank placement"):
            accuracy._gin_transport_evidence(
                _FakeBuffer(_FakeContext()),
                _args(),
                rank=0,
                world_size=16,
                hostnames=["host-a", "host-b"] * 8,
                symmetric_memory_backend="NCCL",
                symmetric_memory_registration=object(),
                dist=_FakeEvidenceDist(),
            )


class TestMegaMoeDirectDispatchHostSourceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = (_REPO_ROOT / "csrc/apis/mega.hpp").read_text()
        cls.gin_api = (_REPO_ROOT / "csrc/apis/mega_gin.hpp").read_text()
        cls.jit = (_REPO_ROOT / "csrc/jit_kernels/impls/"
                   "sm100_fp8_fp4_mega_moe.hpp").read_text()
        cls.kernel = (_REPO_ROOT / "deep_gemm/include/deep_gemm/impls/"
                      "sm100_fp8_fp4_mega_moe.cuh").read_text()
        cls.direct_compile_tu = (
            _REPO_ROOT / "tests/compile_sm100_megamoe_gin.cu"
        ).read_text()

    def test_size_api_enforces_zero_growth_alias_contract(self):
        self.assertIn("not gin_direct_dispatch or", self.api)
        self.assertIn("num_max_tokens_per_rank >= 384", self.api)
        self.assertIn("kMegaMoeGinDirectDispatchStorageBytes", self.api)
        self.assertIn("gin_active_fast_path", self.api)
        # Admission must prove the minimum B=1 scale-scratch extent; a larger
        # GET completion batch must not rescue an otherwise invalid device.
        alias_guard = self.api.split(
            "not gin_direct_dispatch or", 1
        )[1].split("kMegaMoeGinDirectDispatchStorageBytes", 1)[0]
        self.assertNotIn("gin_completion_batch", alias_guard)

    def test_live_gin_rejects_whole_buffer_debug_zero_before_launch(self):
        context_lookup = self.api.index("find_megamoe_gin_context")
        debug_guard = self.api.index(
            'get_env<int>("DG_COMM_KERNEL_DEBUG") != 0', context_lookup
        )
        kernel_launch = self.api.index("sm100_fp8_fp4_mega_moe(", debug_guard)
        trailing_zero = self.api.index("sym_buffer.zero_();", kernel_launch)
        self.assertLess(context_lookup, debug_guard)
        self.assertLess(debug_guard, kernel_launch)
        self.assertLess(kernel_launch, trailing_zero)
        self.assertIn("whole-buffer debug zeroing is not GIN-safe", self.api)

    def test_raw_bf16_binding_rejects_gin_debug_zero_before_launch(self):
        bf16_entry = self.api.index("static void bf16_mega_moe(")
        context_lookup = self.api.index("find_megamoe_gin_context", bf16_entry)
        debug_guard = self.api.index(
            'get_env<int>("DG_COMM_KERNEL_DEBUG") != 0', context_lookup
        )
        kernel_launch = self.api.index("sm100_bf16_mega_moe(", debug_guard)
        trailing_zero = self.api.index("sym_buffer.zero_();", kernel_launch)
        self.assertLess(bf16_entry, context_lookup)
        self.assertLess(context_lookup, debug_guard)
        self.assertLess(debug_guard, kernel_launch)
        self.assertLess(kernel_launch, trailing_zero)

    def test_jit_key_and_launch_guard_include_direct_dispatch(self):
        self.assertIn("DG_MEGAMOE_GIN_DIRECT_DISPATCH", self.jit)
        self.assertIn("args.gin_direct_dispatch ? \"1\" : \"0\"", self.jit)
        self.assertIn("gin_transport_opt->direct_dispatch", self.jit)
        self.assertIn("kMegaMoeGinDirectDispatchStorageBytes", self.jit)
        alias_guard = self.jit.split(
            "const bool gin_direct_dispatch", 1
        )[1].split("kMegaMoeGinDirectDispatchStorageBytes", 1)[0]
        self.assertNotIn("completion_batch", alias_guard)

    def test_direct_selection_uses_shared_limit_and_retains_exact_r75(self):
        consensus = self.kernel.split(
            "a small-decode ineligibility bit", 1
        )[1].split("comm::nvlink_lsa_barrier", 1)[0]
        self.assertIn("kMegaMoeGinBulkCombine or", consensus)
        self.assertIn("kMegaMoeGinDirectDispatch", consensus)
        self.assertIn("num_tokens > layout::", consensus)
        self.assertIn("kMegaMoeGinBulkCombineMaxTokens", consensus)

        direct_gate = self.kernel.split(
            "const auto use_gin_direct_dispatch_this_launch", 1
        )[1].split("// SF and its buffer configs", 1)[0]
        self.assertIn("get_gin_world_active_ptr", direct_gate)
        self.assertIn("get_gin_world_bulk_ineligible_ptr", direct_gate)
        self.assertIn(
            "if (not use_gin_direct_dispatch and lane_idx == 0",
            self.kernel,
        )
        self.assertIn(
            "if (not use_gin_direct_dispatch and warp_idx == 0)",
            self.kernel,
        )
        self.assertIn("The exact r75 fallback", self.kernel)

    def test_context_validates_direct_flag_before_and_after_device_init(self):
        self.assertIn("static constexpr int kCapabilityFields = 16", self.gin_api)
        self.assertIn("direct_dispatch_mismatch", self.gin_api)
        self.assertIn("direct_dispatch_resource_mismatch", self.gin_api)
        self.assertIn('result["direct_dispatch"]', self.gin_api)

    def test_compile_tu_instantiates_feature_on_direct_branches(self):
        self.assertIn(
            "#define DG_MEGAMOE_GIN_ACTIVE_FAST_PATH 1",
            self.direct_compile_tu,
        )
        self.assertIn(
            "#define DG_MEGAMOE_GIN_DIRECT_DISPATCH 1",
            self.direct_compile_tu,
        )
        self.assertIn(
            "#define DG_MEGAMOE_GIN_BULK_COMBINE 1",
            self.direct_compile_tu,
        )
        self.assertIn(
            "instantiate_k3_decode_gin<16, 256, 8, 5>()",
            self.direct_compile_tu,
        )
        self.assertIn(
            "instantiate_k3_decode_gin<32, 128, 16, 10>()",
            self.direct_compile_tu,
        )

    def test_compile_tu_matches_accuracy_runtime_math_defaults(self):
        with mock.patch(
            "sys.argv", ["test_mega_moe_accuracy.py", "--k3"]
        ):
            args = accuracy._parse_args()
        self.assertEqual(args.activation_clamp, 10.0)
        self.assertFalse(args.fast_math)
        self.assertIn("10.0f,", self.direct_compile_tu)
        self.assertIn("false, true>", self.direct_compile_tu)


class TestMegaMoeAccuracyFailureSafeTeardown(unittest.TestCase):
    def test_synchronized_success_collectively_destroys_in_order(self):
        events = []
        buffer = _FakeBuffer(events=events)
        dist = _FakeDist(events)

        synchronized_success = accuracy._synchronize_worker_success(dist)
        accuracy._teardown_worker(buffer, dist, synchronized_success, None)

        self.assertEqual(
            events,
            [
                "dist.barrier",
                "buffer.destroy",
                "dist.is_initialized",
                "dist.destroy_process_group",
            ],
        )

    def test_failure_aborts_rank_locally_and_skips_distributed_teardown(self):
        events = []
        buffer = _FakeBuffer(events=events)
        dist = _FakeDist(events)

        accuracy._teardown_worker(
            buffer, dist, synchronized_success=False,
            symmetric_memory_registration=None,
        )

        self.assertEqual(events, ["buffer.abort"])

    def test_failure_before_buffer_creation_skips_distributed_teardown(self):
        events = []
        dist = _FakeDist(events)

        accuracy._teardown_worker(
            None, dist, synchronized_success=False,
            symmetric_memory_registration=None,
        )

        self.assertEqual(events, [])

    def test_rank_local_abort_failure_does_not_mask_worker_failure(self):
        events = []
        buffer = _FakeBuffer(events=events, abort_error=RuntimeError("abort failed"))
        dist = _FakeDist(events)

        accuracy._teardown_worker(
            buffer, dist, synchronized_success=False,
            symmetric_memory_registration=None,
        )

        self.assertEqual(events, ["buffer.abort"])

    def test_collective_buffer_destroy_failure_aborts_and_skips_process_group(self):
        events = []
        buffer = _FakeBuffer(events=events, destroy_error=RuntimeError("destroy failed"))
        dist = _FakeDist(events)

        with self.assertRaisesRegex(RuntimeError, "destroy failed"):
            accuracy._teardown_worker(
                buffer, dist, synchronized_success=True,
                symmetric_memory_registration=None,
            )

        self.assertEqual(events, ["buffer.destroy", "buffer.abort"])

    def test_success_with_uninitialized_process_group_does_not_destroy_it(self):
        events = []
        buffer = _FakeBuffer(events=events)
        dist = _FakeDist(events, initialized=False)

        accuracy._teardown_worker(
            buffer, dist, synchronized_success=True,
            symmetric_memory_registration=None,
        )

        self.assertEqual(events, ["buffer.destroy", "dist.is_initialized"])


if __name__ == "__main__":
    unittest.main()
