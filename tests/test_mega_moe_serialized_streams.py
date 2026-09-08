"""Laptop mocks of enqueue order; not a CUDA stream-lifetime proof."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from mega_moe_serialized_streams import SERIALIZED_SEQUENCE, enqueue_event_serialized


class FakeCuda:
    def __init__(self):
        self.log, self.current, self.streams, self.events = [], None, [], []
        self.default = SimpleNamespace(wait_event=lambda event: self.log.append(("join", event)))

    def Stream(self):
        stream = SimpleNamespace(index=len(self.streams))
        stream.wait_event = lambda event: self.log.append(("wait", stream.index, event))
        self.streams.append(stream)
        return stream

    def Event(self):
        event = SimpleNamespace(index=len(self.events))
        event.record = lambda stream: self.log.append(("record", stream, event))
        self.events.append(event)
        return event

    def current_stream(self):
        return self.default

    def stream(self, stream):
        cuda = self

        class Scope:
            def __enter__(self):
                cuda.current = stream

            def __exit__(self, *args):
                cuda.current = None
        return Scope()


class SerializedStreamsContracts(unittest.TestCase):
    def test_every_callback_after_previous_event_before_own_snapshot_event(self):
        cuda = FakeCuda()
        callbacks = []
        for step in range(6):
            def callback(step=step):
                cuda.log.extend((operation, cuda.current.index, step)
                                for operation in ("copy", "reset", "launch", "snapshot"))
                return step
            callbacks.append(callback)
        results, (streams, events) = enqueue_event_serialized(SimpleNamespace(cuda=cuda), callbacks)
        self.assertEqual(results, list(range(6)))
        self.assertEqual(len(streams), 2)
        self.assertEqual(len(events), 7)
        self.assertEqual(cuda.log[0][0], "record")
        for step in range(6):
            sequence = cuda.log[1 + step * 6:1 + (step + 1) * 6]
            self.assertEqual([item[0] for item in sequence],
                             ["wait", "copy", "reset", "launch", "snapshot", "record"])
            self.assertIs(sequence[0][2], events[step])
            self.assertIs(sequence[-1][2], events[step + 1])
            self.assertEqual(sequence[2][1], step % 2)
        self.assertEqual(cuda.log[-1], ("join", events[-1]))

    def test_no_host_wait_in_enqueue_primitive(self):
        source = Path(__file__).with_name("mega_moe_serialized_streams.py").read_text()
        tree = ast.parse(source)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "enqueue_event_serialized")
        calls = [ast.unparse(node.func) for node in ast.walk(function) if isinstance(node, ast.Call)]
        self.assertFalse(any("synchronize" in value or value.startswith("dist.")
                             for value in calls))
        self.assertNotIn(".item(", ast.unparse(function))

    def test_payload_returns_to_zero_and_routes_include_remote_masked_local(self):
        self.assertEqual(SERIALIZED_SEQUENCE[0], SERIALIZED_SEQUENCE[-1])
        self.assertEqual({epoch for epoch, _ in SERIALIZED_SEQUENCE}, {0, 1})
        self.assertTrue({"all_remote", "half_remote", "all_masked", "all_local"} <=
                        {name for _, name in SERIALIZED_SEQUENCE})

    def test_device_canaries_compare_bytes_even_for_strided_sf_and_fp8(self):
        source = Path(__file__).with_name("mega_moe_serialized_streams.py").read_text()
        start = source.index("def _device_guard_snapshot")
        end = source.index("def validate_serialized_streams", start)
        body = source[start:end]
        self.assertIn("[start:].contiguous().view(torch.uint8)", body)
        self.assertIn("expected.contiguous().view(torch.uint8)", body)
        self.assertIn("actual_bytes == expected_bytes", body)
        self.assertNotIn(".item()", body.split('"""')[-1])

    def test_same_graph_borrowed_and_mutations_are_inside_queued_callback(self):
        source = Path(__file__).with_name("mega_moe_serialized_streams.py").read_text()
        tree = ast.parse(source)
        function = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and node.name == "step")
        body = ast.unparse(function)
        for text in ("_copy_input_storage", "harness.copy_inputs", "harness.output.fill_",
                     "harness.stats.zero_", "harness.launch()", "graph.replay()",
                     "harness.output.clone()", "harness.stats.clone()", "_device_guard_snapshot"):
            self.assertIn(text, body)
        for forbidden in ("synchronize", "dist.", ".item(", "capture", "patch.dict"):
            self.assertNotIn(forbidden, body)
        self.assertIn('graphs["serialized-control"] = baseline', source)
        self.assertNotIn('graphs["candidate"]', source)
        self.assertIn('for kind in ("eager", "retained_graph")', source)

    def test_integration_combine_only_before_graph_retirement(self):
        source = Path(__file__).with_name("bench_mega_moe_dispatch_overlap.py").read_text()
        start = source.index('if axis.name == "combine":', source.index("imbalance = _fixed_shape_imbalance"))
        call = source.index("serialized = validate_serialized_streams(", start)
        retirement = source.index("common._retire_graphs(graphs)", call)
        self.assertLess(call, retirement)
        self.assertIn('"serialized_stream_validation": serialized', source)

    def test_cleanup_joins_work_before_retiring_control_and_restoring_inputs(self):
        source = Path(__file__).with_name("mega_moe_serialized_streams.py").read_text()
        tree = ast.parse(source)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "validate_serialized_streams")
        statement = next(node for node in function.body if isinstance(node, ast.Try))
        cleanup = ast.unparse(ast.Module(body=statement.finalbody, type_ignores=[]))
        self.assertLess(cleanup.index("torch.cuda.synchronize()"),
                        cleanup.index("common._retire_graphs(graphs)"))
        self.assertLess(cleanup.index("common._retire_graphs(graphs)"),
                        cleanup.index("accuracy._copy_input_storage"))
        for value in ("initial_inputs", "initial_buffer.items()", "initial_output", "initial_stats"):
            self.assertIn(value, cleanup)
        self.assertGreater(cleanup.rindex("torch.cuda.synchronize()"),
                           cleanup.index("harness.stats.copy_(initial_stats)"))

    def test_snapshots_validated_before_any_future_launch_or_control(self):
        source = Path(__file__).with_name("mega_moe_serialized_streams.py").read_text()
        begin = source.index("snapshots, keepalive = enqueue_event_serialized")
        end = source.index("    finally:", begin)
        body = source[begin:end]
        for value in ("output.view(torch.int16)", "expected_local_stats", "guards.all().item()",
                      "accuracy._collective_transition_check"):
            self.assertIn(value, body)
        for forbidden in ("graph.replay()", "harness.launch()", "baseline.replay()", "patch.dict"):
            self.assertNotIn(forbidden, body)


if __name__ == "__main__":
    unittest.main()
