"""CPU protocol/call-path tests; they do not execute CUDA or model GEMMs."""

from copy import deepcopy
import inspect
from types import SimpleNamespace
import unittest
from unittest import mock

import mega_moe_gb200_imbalance as gate
import bench_gb200_transport_matched as runner


class Tensor:
    def __init__(self, value): self.value = deepcopy(value)
    def clone(self): return Tensor(self.value)
    def copy_(self, other): self.value = deepcopy(other.value)
    def zero_(self): self.value = 0
    def fill_(self, value): self.value = value
    def cpu(self): return self
    def tolist(self): return self.value
    def contiguous(self): return self
    def view(self, dtype): return self


class Dist:
    def get_world_size(self): return 8
    def all_gather_object(self, output, value): output[:] = [deepcopy(value) for _ in range(8)]


class ImbalanceContracts(unittest.TestCase):
    def test_actual_hot_routes_include_all_sources_slots_and_no_truncation(self):
        for tokens in (32, 40, 48):
            for owner in (0, 4):
                counts = [0] * 448
                for source in range(8):
                    ids, weights = gate.hot_routes(source, tokens, owner)
                    for row, row_weights in zip(ids, weights):
                        self.assertEqual(len(set(row)), 16)
                        for slot, expert in enumerate(row):
                            self.assertEqual(expert, owner * 56 + slot)
                            self.assertEqual(row_weights[slot], 2.0 ** (-(slot % 3 + 1)))
                            counts[expert] += 1
                self.assertEqual(sum(counts), 8 * tokens * 16)
                self.assertEqual(counts[owner * 56:owner * 56 + 16], [8 * tokens] * 16)
                self.assertEqual(sum(counts[owner * 56 + 16:(owner + 1) * 56]), 0)

    def test_single_active_source_masks_all_other_ranks(self):
        for source in range(8):
            ids, weights = gate.hot_routes(source, 32, 4, only_source=0)
            self.assertEqual(sum(item >= 0 for row in ids for item in row), 512 if source == 0 else 0)
            if source:
                self.assertEqual(set(item for row in weights for item in row), {0.})
        for args in ((8, 32, 4), (0, 64, 4), (0, 32, 8)):
            with self.assertRaises(ValueError): gate.hot_routes(*args)

    def fixture(self):
        trace = []
        output, stats, input_tensor = Tensor("prior-output"), Tensor(77), Tensor("prior-input")
        inputs = SimpleNamespace(epoch=-1, oracle_x=input_tensor)
        cases = {name: SimpleNamespace(name=name, expected_local_stats=Tensor([0] * 56))
                 for name in ("all_remote", "all_masked", "all_local")}
        state = {"case": None}
        def copy(case): state["case"] = case.name
        def launch():
            trace.append(("eager", inputs.epoch, state["case"]))
            output.value = (inputs.epoch, state["case"])
            return output
        def replay():
            trace.append(("replay", inputs.epoch, state["case"]))
            output.value = (inputs.epoch, state["case"])
        graph = SimpleNamespace(replay=replay, reset=mock.Mock(side_effect=lambda: trace.append(("reset",))))
        torch = SimpleNamespace(int64="i64", float32="f32", uint8="u8",
            tensor=lambda values, **kwargs: Tensor(values), equal=lambda a, b: a.value == b.value,
            cuda=SimpleNamespace(synchronize=mock.Mock()))
        harness = SimpleNamespace(torch=torch, dist=Dist(), rank=0, experts_per_rank=56,
            args=SimpleNamespace(num_tokens=32, payload_epochs=3), inputs=inputs,
            deep_gemm=object(), cases=cases, output=output, stats=stats,
            launch=launch, copy_inputs=copy)
        mutable = {"output": output, "stats": stats, "input": input_tensor}
        def copy_input(destination, fresh):
            destination.epoch = fresh
            input_tensor.value = fresh
        def validate(harness, case, expected, reference, label):
            self.assertEqual(output.value, (inputs.epoch, case.name))
            if reference is not None: self.assertEqual(output.value, reference.value)
            return {"worst_symmetric_diff": 0., "worst_max_abs": 0.}
        patches = (
            mock.patch.object(gate, "_identity", return_value="stable"),
            mock.patch.object(gate, "_mutable_tensors", return_value=mutable),
            mock.patch.object(gate.accuracy, "_global_route_stats", return_value=Tensor([0] * 56)),
            mock.patch.object(gate.accuracy, "_make_inputs", side_effect=lambda *a, input_epoch: input_epoch),
            mock.patch.object(gate.accuracy, "_copy_input_storage", side_effect=copy_input),
            mock.patch.object(gate.accuracy, "_analytic_oracle", return_value="oracle"),
            mock.patch.object(gate.accuracy, "_capture_launch_only_graph", return_value=graph),
            mock.patch.object(gate, "_check", side_effect=validate),
        )
        for patch in patches:
            patch.start(); self.addCleanup(patch.stop)
        return harness, graph, trace, mutable, validate

    def test_one_graph_no_interleaved_controls_all_epochs_and_restore(self):
        harness, graph, trace, mutable, _ = self.fixture()
        before = {name: value.clone() for name, value in mutable.items()}
        result = gate.validate_ep8_imbalance(harness)
        self.assertEqual(result["payload_epochs"], [0, 1, 2, 0])
        self.assertEqual(len(result["checks"]), 4 * len(gate.SEQUENCE))
        first_replay = next(i for i, row in enumerate(trace) if row[0] == "replay")
        self.assertEqual(sum(row[0] == "eager" for row in trace[:first_replay]), 15)
        self.assertFalse(any(row[0] == "eager" for row in trace[first_replay:]))
        self.assertEqual([row[2] for row in trace if row[0] == "replay"], list(gate.SEQUENCE) * 4)
        self.assertEqual(trace[-1], ("reset",))
        graph.reset.assert_called_once()
        self.assertEqual({name: value.value for name, value in mutable.items()},
                         {name: value.value for name, value in before.items()})
        self.assertFalse(result["timed"])

    def test_failure_retires_graph_and_restores_not_false_success(self):
        harness, graph, trace, mutable, validate = self.fixture()
        before = {name: value.clone() for name, value in mutable.items()}
        def fail(*args):
            if "/graph/" in args[-1]: raise AssertionError("injected byte corruption")
            return validate(*args)
        with mock.patch.object(gate, "_check", side_effect=fail), self.assertRaisesRegex(AssertionError, "corruption"):
            gate.validate_ep8_imbalance(harness)
        graph.reset.assert_called_once()
        self.assertEqual({name: value.value for name, value in mutable.items()},
                         {name: value.value for name, value in before.items()})

    def test_cleanup_failure_cannot_report_pass(self):
        harness, graph, trace, mutable, _ = self.fixture()
        graph.reset.side_effect = RuntimeError("reset failed")
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            gate.validate_ep8_imbalance(harness)
        self.assertEqual(mutable["input"].value, "prior-input")

    def test_local_preparation_error_is_gathered_before_launch_and_restored(self):
        harness, graph, trace, mutable, _ = self.fixture()
        harness.copy_inputs = mock.Mock(side_effect=ValueError("copy failed"))
        with self.assertRaisesRegex(RuntimeError, "imbalance/eager_prepare"):
            gate.validate_ep8_imbalance(harness)
        self.assertEqual(trace, [])
        graph.reset.assert_not_called()
        self.assertEqual(mutable["input"].value, "prior-input")

    def test_real_check_uses_original_limits_and_collective_failure(self):
        harness = SimpleNamespace(torch=SimpleNamespace(), dist=Dist(), output="out",
            args=object(), assert_stats=mock.Mock(), assert_guards=mock.Mock())
        case = SimpleNamespace(name="hot", expected_local_stats="counts")
        with mock.patch.object(gate.accuracy, "_oracle_thresholds", return_value=(3e-5, .25)), mock.patch.object(
                gate.accuracy, "_symmetric_diff", return_value=(3.01e-5, 0.)):
            with self.assertRaisesRegex(RuntimeError, "unchanged sparse oracle"):
                gate._check(harness, case, "expected", None, "test")
        with mock.patch.object(gate.accuracy, "_oracle_thresholds", return_value=(3e-5, .25)), mock.patch.object(
                gate.accuracy, "_symmetric_diff", return_value=(0., 0.)), mock.patch.object(
                gate.accuracy, "_assert_bitwise_equal", side_effect=AssertionError("changed bytes")):
            with self.assertRaisesRegex(RuntimeError, "changed bytes"):
                gate._check(harness, case, "expected", "eager", "test")
        harness.assert_stats.assert_called_once_with("counts", "test")
        harness.assert_guards.assert_called_once_with("test")

    def test_only_ep8_hook_runs_before_timing_and_source_is_pinned(self):
        source = inspect.getsource(runner.main)
        self.assertIn("if options.world_size == 8:", source)
        self.assertLess(source.index("validate_ep8_imbalance(harness)"), source.index("_compare_gin(harness"))
        self.assertLess(source.index("validate_ep8_imbalance(harness)"), source.index("_benchmark_native(harness"))
        self.assertIn("tests/mega_moe_gb200_imbalance.py", runner.SOURCE_FILES)
        self.assertNotIn("patch.dict", inspect.getsource(gate))


if __name__ == "__main__":
    unittest.main()
