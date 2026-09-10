"""CPU contracts only; no GB200/topology/latency claim is produced here."""

import ast
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import bench_gb200_transport_matched as runner


class ParsingContracts(unittest.TestCase):
    def parse(self, mode="gin_ib", mns=8, *extra):
        return runner.parse_args(["--mode", mode, "--decode-mns", str(mns),
                                  "--output", "result.json", *extra])

    def test_all_target_shapes_and_all_modes_keep_exact_math(self):
        for mode in runner.MODES:
            for mns in (8, 10, 12):
                with self.subTest(mode=mode, mns=mns):
                    options, args, comparison = self.parse(mode, mns)
                    self.assertEqual((args.num_tokens, args.num_max_tokens_per_rank,
                                      args.hidden, args.intermediate_hidden,
                                      args.num_experts, args.num_topk),
                                     (4 * mns, 384, 3584, 3072, 896, 16))
                    self.assertFalse(args.fast_math)
                    self.assertEqual(args.activation_clamp, 10.0)
                    self.assertEqual(args.benchmark_replays, 0)
                    self.assertEqual(args.require_gin, mode in runner.GIN_MODES)
                    self.assertEqual(args.gin_direct_dispatch, mode in runner.GIN_MODES)
                    self.assertEqual((comparison.deepep_capacity,
                                      comparison.deepep_dispatch_bucket,
                                      comparison.deepep_num_sms, comparison.deepep_num_qps),
                                     (384, 4 * mns, 16, 9))
                    self.assertEqual((comparison.comparison_replays,
                                      comparison.comparison_conditioning,
                                      comparison.comparison_warmups), (204, 60, 10))
                    self.assertEqual((comparison.comparison_rel_l2,
                                      comparison.comparison_max_abs), (0.02, 0.02))
                    self.assertFalse(hasattr(args, "num_active_routes_hint"))

    def test_quick_screen_is_explicit_not_relabeled_full(self):
        _, args, comparison = self.parse("gin_ib", 8, "--comparison-replays", "32",
                                        "--comparison-conditioning", "10",
                                        "--comparison-warmups", "4")
        self.assertEqual(comparison.comparison_replays, 32)
        self.assertEqual(args.payload_epochs, 3)

    def test_rejects_missing_gates_or_unmatched_configuration(self):
        for extra in (("--comparison-replays", "1"), ("--payload-epochs", "1"),
                      ("--eager-iterations", "0"), ("--graph-replays", "0"),
                      ("--comparison-conditioning", "0"), ("--comparison-warmups", "0"),
                      ("--fast-math",), ("--num-tokens", "16"),
                      ("--comparison-max-abs", "1"), ("--deepep-num-sms", "8")):
            with self.subTest(extra=extra), patch("sys.stderr"), self.assertRaises(SystemExit):
                self.parse("gin_ib", 8, *extra)

    def test_native_cannot_claim_trt_profile(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            self.parse("native_nvl", 8, "--profile-recipe")

    def test_configuration_fixes_logical_not_physical_placement(self):
        options, args, comparison = self.parse()
        config = runner.configuration(options, args, comparison, {"source": "digest"})
        self.assertEqual(config["logical_route_domains"], [0] * 8 + [1] * 8)
        self.assertEqual((config["physical_hosts"], config["ranks_per_physical_host"],
                          config["world_size"]), (4, 4, 16))
        self.assertIsNone(config["math"]["compute_hint"])
        self.assertEqual(config["source_sha256"], {"source": "digest"})

    def test_environment_selects_transport_only(self):
        for mode in runner.MODES:
            env = runner.fixed_environment(mode)
            for key in runner.accuracy.GIN_VALIDATED_FLAG_ENVS:
                self.assertEqual(env[key], "1" if mode in runner.GIN_MODES else "0")
            self.assertEqual(env[runner.accuracy.GIN_ACTIVITY_GATE_OPT_ENV],
                             "1" if mode in runner.GIN_MODES else "0")
            self.assertEqual(env[runner.accuracy.GIN_LOCAL_ABLATION_ENV], "0")
            self.assertEqual(env["DG_MEGAMOE_GIN_DIAGNOSTICS"], "0")
            self.assertFalse(any("NUM_SMS" in key or "BLOCK_M" in key for key in env))
        with self.assertRaises(ValueError):
            runner.fixed_environment("ep8")

    def test_roce_changes_only_transport_label_not_math_timing_or_flags(self):
        ib_options, ib_args, ib_comparison = self.parse("gin_ib", 10, "--profile-recipe")
        roce_options, roce_args, roce_comparison = self.parse("gin_roce", 10, "--profile-recipe")
        self.assertEqual(vars(ib_args), vars(roce_args))
        self.assertEqual(vars(ib_comparison), vars(roce_comparison))
        self.assertEqual(runner.fixed_environment("gin_ib"), runner.fixed_environment("gin_roce"))
        ib = runner.configuration(ib_options, ib_args, ib_comparison, {})
        roce = runner.configuration(roce_options, roce_args, roce_comparison, {})
        self.assertEqual(roce.pop("mode"), "gin_roce")
        self.assertEqual(ib.pop("mode"), "gin_ib")
        network = roce.pop("network_transport")
        self.assertEqual(network["requested_transport"], "RoCE")
        self.assertEqual(network["requested_rdma_link_layer"], "Ethernet")
        self.assertIsNone(network["observed_rdma_link_layer"])
        self.assertFalse(network["physical_network_payload_verified"])
        self.assertEqual(ib.pop("network_transport")["requested_transport"], "InfiniBand")
        self.assertEqual(ib, roce)


class FakeDist:
    def __init__(self, trace):
        self.trace = trace

    def get_world_size(self):
        return 16

    def all_gather_object(self, output, value):
        self.trace.append("collective")
        output[:] = [value] * 16

    def barrier(self):
        self.trace.append("barrier")


class FakeDeepEPBuffer:
    """CPU stand-in, not evidence about any installed DeepEP version."""
    allow_hybrid_mode = False
    num_rdma_ranks, num_nvlink_ranks = 2, 8
    num_scaleout_ranks, num_scaleup_ranks = 1, 16

    def __init__(self):
        self.destroy = Mock()

    def get_physical_domain_size(self):
        return 2, 8

    def get_logical_domain_size(self):
        return 1, 16


def fake_comparator():
    return SimpleNamespace(buffer=FakeDeepEPBuffer(), last_receipt={},
        api_evidence={"physical_domain_size": [2, 8], "logical_domain_size": [1, 16],
                      "allow_hybrid_mode": False},
        weight_sha256="sha", recipe_profile={"profile": True})


class TimingContracts(unittest.TestCase):
    def make_harness(self):
        trace = []
        event_count = 0

        class Event:
            def __init__(self, enable_timing):
                nonlocal event_count
                self.kind = "start" if event_count % 2 == 0 else "end"
                event_count += 1
                assert enable_timing

            def record(self):
                trace.append(self.kind)

            def elapsed_time(self, other):
                return 0.25

        graph = Mock(replay=lambda: trace.append("replay"), reset=lambda: trace.append("reset"))
        torch = SimpleNamespace(cuda=SimpleNamespace(
            Event=Event, synchronize=lambda: trace.append("synchronize")))
        dist = FakeDist(trace)
        cases = {name: SimpleNamespace(expected_local_stats=3) for name in runner.accuracy.MATCHED_BENCHMARK_ROUTES}
        harness = SimpleNamespace(
            torch=torch, dist=dist, cases=cases, output="actual", launch=Mock(),
            copy_inputs=lambda case: trace.append("copy"),
            stats=SimpleNamespace(zero_=lambda: trace.append("zero")),
            assert_stats=Mock(), assert_guards=Mock())
        comparison = SimpleNamespace(comparison_warmups=2, comparison_conditioning=6,
                                     comparison_replays=4)
        snapshots = {name: "snapshot" for name in cases}
        return harness, graph, comparison, snapshots, trace

    def test_native_event_intervals_contain_only_replay_and_raw_units(self):
        harness, graph, comparison, snapshots, trace = self.make_harness()
        with patch.object(runner.matched, "_capture", return_value=graph) as capture, patch.object(
                runner.accuracy, "_assert_bitwise_equal") as equal, patch.object(
                runner.accuracy, "_distributed_timing_record", side_effect=lambda x, *args: {"raw": x}):
            result = runner._benchmark_native(harness, snapshots, comparison)
        self.assertEqual(capture.call_count, 1)
        self.assertEqual(trace.count("replay"), 3 * (2 + 6 + 4))
        self.assertEqual(equal.call_count, 3)
        for name in harness.cases:
            self.assertEqual(result[name]["megamoe"]["isolated"]["raw"], [250.0] * 4)
        for i, entry in enumerate(trace):
            if entry == "start":
                self.assertEqual(trace[i:i + 3], ["start", "replay", "end"])
        self.assertEqual(trace[-1], "reset")
        self.assertEqual([call.args[0] for call in harness.assert_stats.call_args_list], [12] * 3)

    def test_native_failure_retires_graph(self):
        harness, graph, comparison, snapshots, trace = self.make_harness()
        harness.assert_guards.side_effect = AssertionError("corrupted")
        with patch.object(runner.matched, "_capture", return_value=graph), self.assertRaises(RuntimeError):
            runner._benchmark_native(harness, snapshots, comparison)
        self.assertEqual(trace[-1], "reset")

    def test_gin_calls_existing_pair_and_destroys_buffer(self):
        comparator = fake_comparator()
        harness = SimpleNamespace(rank=0, dist=FakeDist([]), torch=SimpleNamespace(cuda=Mock()))
        with patch.object(runner.matched, "DeepEPTRTLLM", return_value=comparator), patch.object(
                runner.matched, "_benchmark_pair", return_value=({"routes": True}, {"epochs": True})) as pair:
            result = runner._compare_gin(harness, "snapshots", "comparison")
        pair.assert_called_once_with(harness, comparator, "comparison", "snapshots")
        self.assertEqual(result["canonical_weights_sha256"], "sha")
        self.assertTrue(result["deepep_actual_domain_preflight"]["before_graph_capture"])
        comparator.buffer.destroy.assert_called_once()
        self.assertIsNone(comparator.last_receipt)

    def test_gin_cleanup_does_not_replace_original_failure(self):
        comparator = fake_comparator()
        comparator.buffer.destroy.side_effect = ValueError("cleanup")
        harness = SimpleNamespace(rank=0, dist=FakeDist([]), torch=SimpleNamespace(cuda=Mock()))
        with patch.object(runner.matched, "DeepEPTRTLLM", return_value=comparator), patch.object(
                runner.matched, "_benchmark_pair", side_effect=RuntimeError("numerical")), patch(
                "sys.stderr"), self.assertRaisesRegex(RuntimeError, "numerical"):
            runner._compare_gin(harness, {}, None)
        comparator.buffer.destroy.assert_called_once()


class DeepEPDomainContracts(unittest.TestCase):
    def test_actual_live_domains_and_source_are_recorded(self):
        value = runner._deepep_domain_record(fake_comparator(), 7)
        self.assertEqual(value["rank"], 7)
        self.assertEqual(value["physical_domain_size"], [2, 8])
        self.assertEqual(value["logical_domain_size"], [1, 16])
        self.assertEqual(value["loaded_python_source"], str(Path(__file__).resolve()))
        self.assertEqual(len(value["loaded_python_source_sha256"]), 64)
        self.assertIn("return 2, 8", value["loaded_domain_getter_source"]["get_physical_domain_size"])

    def test_wrong_geometry_hybrid_and_stale_evidence_fail_collectively_before_capture(self):
        changes = (
            ("get_physical_domain_size", lambda: (4, 4)),
            ("get_physical_domain_size", lambda: (1, 16)),
            ("get_logical_domain_size", lambda: (2, 8)),
            ("get_logical_domain_size", lambda: (True, 16)),
            ("allow_hybrid_mode", True), ("num_nvlink_ranks", 4),
        )
        for attribute, value in changes:
            with self.subTest(attribute=attribute, value=value):
                comparator = fake_comparator()
                setattr(comparator.buffer, attribute, value)
                trace = []
                harness = SimpleNamespace(rank=0, dist=FakeDist(trace), torch=SimpleNamespace(cuda=Mock()))
                with patch.object(runner.matched, "DeepEPTRTLLM", return_value=comparator), patch.object(
                        runner.matched, "_benchmark_pair") as pair, self.assertRaisesRegex(
                        RuntimeError, "failed collectively"):
                    runner._compare_gin(harness, {}, None)
                self.assertIn("collective", trace)
                pair.assert_not_called()
                comparator.buffer.destroy.assert_called_once()
        comparator = fake_comparator()
        comparator.api_evidence["physical_domain_size"] = [4, 4]
        with self.assertRaisesRegex(AssertionError, "evidence"):
            runner._deepep_domain_record(comparator, 0)

    def test_remote_rank_failure_is_seen_before_any_second_collective(self):
        class PeerFailure(FakeDist):
            def all_gather_object(self, output, value):
                super().all_gather_object(output, value)
                output[9] = "AssertionError: wrong domains"
        trace = []
        harness = SimpleNamespace(rank=0, dist=PeerFailure(trace))
        with self.assertRaisesRegex(RuntimeError, "wrong domains"):
            runner._validate_deepep_domains(harness, fake_comparator())
        self.assertEqual(trace, ["collective"])

    def test_getter_exception_is_gathered(self):
        comparator = fake_comparator()
        comparator.buffer.get_logical_domain_size = Mock(side_effect=ValueError("getter failed"))
        trace = []
        with self.assertRaisesRegex(RuntimeError, "getter failed"):
            runner._validate_deepep_domains(SimpleNamespace(rank=0, dist=FakeDist(trace)), comparator)
        self.assertEqual(trace, ["collective"])

    def test_installed_source_review_is_not_faked_by_value_gate(self):
        value = runner._validate_deepep_domains(SimpleNamespace(rank=0, dist=FakeDist([])), fake_comparator())
        self.assertTrue(value["installed_tuple_semantics_source_review_required_before_acceptance"])
        self.assertFalse(value["physical_ib_payload_probe"])


class RuntimeImportContracts(unittest.TestCase):
    def runtime(self, package_path=None, extension_path=None):
        trace = []
        torch = SimpleNamespace(cuda=SimpleNamespace(set_device=lambda rank: trace.append(("device", rank))))
        dist = SimpleNamespace()
        dg = SimpleNamespace(__file__=str(package_path or runner.ROOT / "deep_gemm/__init__.py"),
                             _C=SimpleNamespace(__file__=str(extension_path or runner.ROOT / "deep_gemm/_C.test.so")))
        def load(name):
            trace.append(("import", name))
            if name == "deep_gemm":
                self.assertEqual(sys.path[0], str(runner.ROOT))
            return {"torch": torch, "torch.distributed": dist, "deep_gemm": dg}[name]
        return trace, load, (torch, dist, dg)

    def test_device_selected_before_expected_package_import_and_path_restored(self):
        trace, load, runtime = self.runtime()
        original_path = list(sys.path)
        with patch.object(runner.importlib, "import_module", side_effect=load):
            self.assertEqual(runner._load_runtime_for_gb200(3), runtime)
        self.assertEqual(trace, [("import", "torch"), ("device", 3),
                                 ("import", "torch.distributed"), ("import", "deep_gemm")])
        self.assertEqual(sys.path, original_path)

    def test_foreign_python_package_and_extension_are_rejected(self):
        for package, extension in (("/foreign/deep_gemm/__init__.py", None),
                                   (None, "/foreign/deep_gemm/_C.so")):
            with self.subTest(package=package, extension=extension):
                _, load, _ = self.runtime(package, extension)
                with patch.object(runner.importlib, "import_module", side_effect=load), self.assertRaises(RuntimeError):
                    runner._load_runtime_for_gb200(0)

    def test_develop_style_extension_symlink_and_resolved_build_path_are_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory) / "selected"
            package = checkout / "deep_gemm"
            package.mkdir(parents=True)
            target = checkout / "build/lib.test/deep_gemm/_C.test.so"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"CPU fixture, not a loadable extension")
            link = package / target.name
            link.symlink_to(Path("../build/lib.test/deep_gemm") / target.name)
            with patch.object(runner, "ROOT", checkout):
                for module_path in (link, target):
                    with self.subTest(module_path=module_path):
                        _, load, runtime = self.runtime(extension_path=module_path)
                        with patch.object(runner.importlib, "import_module", side_effect=load):
                            self.assertEqual(runner._load_runtime_for_gb200(0), runtime)

    def test_foreign_extension_symlink_targets_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory) / "selected"
            package = checkout / "deep_gemm"
            package.mkdir(parents=True)
            foreign = Path(directory) / "foreign-build"
            foreign.mkdir()
            target = foreign / "_C.test.so"
            target.write_bytes(b"CPU foreign fixture")
            link = package / target.name
            link.symlink_to(target)
            (checkout / "build").symlink_to(foreign, target_is_directory=True)
            with patch.object(runner, "ROOT", checkout):
                for module_path in (link, checkout / "build" / target.name):
                    with self.subTest(module_path=module_path):
                        _, load, _ = self.runtime(extension_path=module_path)
                        with patch.object(runner.importlib, "import_module", side_effect=load), self.assertRaisesRegex(
                                RuntimeError, "selected checkout"):
                            runner._load_runtime_for_gb200(0)


class SourceAndLifecycleContracts(unittest.TestCase):
    def test_no_cuda_or_deepep_import_at_module_scope(self):
        tree = ast.parse(Path(runner.__file__).read_text())
        modules = [alias.name for node in tree.body if isinstance(node, ast.Import)
                   for alias in node.names]
        self.assertFalse(set(modules) & {"torch", "deep_ep", "flashinfer", "deep_gemm"})

    def test_sources_include_kernel_scheduler_layout_and_probe(self):
        for expected in ("tests/mega_moe_gb200_topology.py", "tests/probe_gb200_topology.py",
                         "deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh",
                         "deep_gemm/include/deep_gemm/layout/mega_moe.cuh"):
            self.assertIn(expected, runner.SOURCE_FILES)
        self.assertEqual(len(runner.SOURCE_FILES), len(set(runner.SOURCE_FILES)))

    def test_failed_worker_never_persists_success(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            argv = ["run", "--mode", "native_nvl", "--decode-mns", "8", "--output", str(output)]
            topology_module = SimpleNamespace(GB200Topology=Mock())
            env = {"RANK": "0", "LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "4", "WORLD_SIZE": "16"}
            with patch.object(sys, "argv", argv), patch.dict(os.environ, env), patch.dict(
                    sys.modules, {"mega_moe_gb200_topology": topology_module}), patch.object(
                    runner, "source_manifest", return_value={"source": "sha"}), patch.object(
                    runner.accuracy, "_worker", side_effect=RuntimeError("teardown failed")), self.assertRaisesRegex(
                    RuntimeError, "teardown failed"):
                runner.main()
            self.assertFalse(output.exists())

    def test_success_persists_only_after_final_check_and_worker_teardown(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            argv = ["run", "--mode", "native_nvl", "--decode-mns", "8", "--output", str(output)]
            topology = SimpleNamespace(accuracy_result=None)
            topology_module = SimpleNamespace(GB200Topology=Mock(return_value=topology))
            env = {"RANK": "0", "LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "4", "WORLD_SIZE": "16"}
            trace = []
            dist = FakeDist(trace)
            torch = SimpleNamespace(__version__="mock", version=SimpleNamespace(cuda="mock"),
                                    cuda=SimpleNamespace(current_device=lambda: 0,
                                        get_device_properties=lambda _: SimpleNamespace(multi_processor_count=148)))
            harness = SimpleNamespace(rank=0, torch=torch, dist=dist,
                physical_hostnames=[f"real-host-{rank // 4}" for rank in range(16)],
                route_domains=[0] * 8 + [1] * 8, topology_evidence={"mock_only": True},
                deep_gemm=SimpleNamespace(__file__=runner.__file__,
                                          _C=SimpleNamespace(__file__=runner.__file__)))

            def worker(*args, **kwargs):
                self.assertFalse(output.exists())
                self.assertIs(kwargs["topology"], topology)
                runner.accuracy._run_graph_stress(harness, {}, torch, dist)
                topology.accuracy_result = {"checks": {"mock_only": True}}
                runner.accuracy._synchronize_worker_success(dist)
                self.assertFalse(output.exists())
                trace.append("worker_teardown")

            with patch.object(sys, "argv", argv), patch.dict(os.environ, env), patch.dict(
                    sys.modules, {"mega_moe_gb200_topology": topology_module}), patch.object(
                    runner, "source_manifest", return_value={"source": "sha"}) as manifests, patch.object(
                    runner, "_fixture_identity", return_value={"rank": 0, "tensors": {}}), patch.object(
                    runner, "_benchmark_native", return_value={"mock_timing_not_runtime_evidence": True}), patch.object(
                    runner.accuracy, "_run_graph_stress") as original_stress, patch.object(
                    runner.accuracy, "_synchronize_worker_success", side_effect=lambda _: trace.append("worker_success")), patch.object(
                    runner.accuracy, "_worker", side_effect=worker), patch("builtins.print"):
                runner.main()
            self.assertGreaterEqual(manifests.call_count, 3)
            original_stress.assert_called_once()
            self.assertLess(trace.index("worker_success"), trace.index("worker_teardown"))
            record = json.loads(output.read_text())
            self.assertTrue(record["accuracy_and_teardown_passed"])
            self.assertEqual(record["physical_hostnames"], harness.physical_hostnames)
            self.assertEqual(record["logical_route_domains"], harness.route_domains)
            self.assertEqual(record["schedule"], "native_backend_isolated_rewarm")
            self.assertFalse(record["compute_only_floor_claim"])
            self.assertNotIn("canonical_weights_sha256", record)
            self.assertEqual(record["accuracy"], topology.accuracy_result)

    def test_requires_ep16_four_workers_per_physical_host(self):
        for world, local in (("8", "4"), ("16", "8")):
            with patch.object(sys, "argv", ["run", "--mode", "native_nvl", "--decode-mns", "8",
                                            "--output", "never-written.json"]), patch.dict(
                    os.environ, {"LOCAL_RANK": "0", "WORLD_SIZE": world, "LOCAL_WORLD_SIZE": local}), self.assertRaisesRegex(
                    RuntimeError, "four-host"):
                runner.main()


if __name__ == "__main__":
    unittest.main()
