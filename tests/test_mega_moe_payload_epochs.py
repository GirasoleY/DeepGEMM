"""CPU contract tests for varied payload/SF graph replay stress."""

import unittest
from types import SimpleNamespace
from unittest import mock

import test_mega_moe_accuracy as accuracy


class Tensor:
    def __init__(self, values):
        self.values = list(values)

    def copy_(self, other):
        self.values[:] = other.values
        return self

    def clone(self):
        return Tensor(self.values)

    def zero_(self):
        self.values[:] = [0] * len(self.values)

    def add_(self, other):
        self.values[:] = [a + b for a, b in zip(self.values, other.values)]

    def view(self, dtype):
        return self

    def contiguous(self):
        return self

    def numel(self):
        return len(self.values)

    def data_ptr(self):
        return id(self)


def inputs(epoch):
    return accuracy.KernelInputs(
        Tensor([epoch + 1]),
        Tensor([120 + (group + epoch) % 7 for group in range(7)]),
        Tensor([float(epoch + 1)]),
    )


class TestPayloadEpochs(unittest.TestCase):
    def test_flags_leave_existing_runs_unchanged(self):
        with mock.patch("sys.argv", ["accuracy"]):
            args = accuracy._parse_args()
        self.assertFalse(args.heterogeneous_input_scales)
        self.assertEqual(args.payload_epochs, 1)

    def test_epoch_count_rejects_zero(self):
        with mock.patch("sys.argv", ["accuracy", "--payload-epochs", "0"]), mock.patch(
            "sys.stderr"
        ), self.assertRaises(SystemExit):
            accuracy._parse_args()

    def test_sequence_returns_to_original_payload(self):
        self.assertEqual(accuracy._payload_epoch_sequence(1), [0])
        self.assertEqual(accuracy._payload_epoch_sequence(3), [0, 1, 2, 0])
        with self.assertRaises(ValueError):
            accuracy._payload_epoch_sequence(0)

    def test_every_coordinate_changes_scale_and_full_range_is_covered(self):
        exponent = accuracy._input_scale_exponent
        for rank in range(16):
            for token in range(48):
                for epoch in range(3):
                    values = [exponent(rank, token, group, epoch) for group in range(112)]
                    self.assertEqual(set(values), set(range(-3, 4)))
                    self.assertTrue(all(a != b for a, b in zip(values, values[1:])))
                    for group in range(7):
                        current = exponent(rank, token, group, epoch)
                        self.assertNotEqual(current, exponent(rank + 1, token, group, epoch))
                        self.assertNotEqual(current, exponent(rank, token + 1, group, epoch))
                        self.assertNotEqual(current, exponent(rank, token, group, epoch + 1))

    def test_wrong_scale_transpose_is_observable_at_target_shapes(self):
        for tokens in (32, 40, 48):
            groups = 3584 // 32
            row_major = [
                accuracy._input_scale_exponent(5, token, group, 1)
                for token in range(tokens) for group in range(groups)
            ]
            wrong = [
                row_major[group * tokens + token]
                for token in range(tokens) for group in range(groups)
            ]
            mismatches = sum(a != b for a, b in zip(row_major, wrong))
            self.assertGreater(mismatches, len(row_major) // 2)

    def test_copy_preserves_captured_storage_and_updates_all_fields(self):
        old = inputs(0)
        pointers = tuple(value.data_ptr() for value in (old.x, old.x_sf, old.oracle_x))
        fresh = inputs(1)
        accuracy._copy_input_storage(old, fresh)
        self.assertEqual(pointers, tuple(value.data_ptr() for value in (old.x, old.x_sf, old.oracle_x)))
        self.assertEqual(old.x.values, fresh.x.values)
        self.assertEqual(old.x_sf.values, fresh.x_sf.values)
        self.assertEqual(old.oracle_x.values, fresh.oracle_x.values)
        self.assertIsNot(old.x, fresh.x)

    def test_copy_rejects_scale_format_change(self):
        with self.assertRaisesRegex(AssertionError, "scale format"):
            accuracy._copy_input_storage(inputs(0), accuracy.KernelInputs(Tensor([1]), None, Tensor([1])))

    def test_single_graph_observes_each_payload_and_all_routes(self):
        torch = SimpleNamespace(
            uint8="uint8",
            equal=lambda a, b: a.values == b.values,
            unique=lambda a: Tensor(set(a.values)),
            zeros_like=lambda a: Tensor([0] * len(a.values)),
            cuda=SimpleNamespace(synchronize=lambda: None),
        )
        cases = {
            mode: SimpleNamespace(
                topk_idx=Tensor([index]), topk_weights=Tensor([1]),
                expected_local_stats=Tensor([index + 1]),
            )
            for index, mode in enumerate(accuracy.ROUTE_MODES)
        }
        harness = SimpleNamespace(
            rank=0, args=SimpleNamespace(payload_epochs=3, heterogeneous_input_scales=True),
            inputs=inputs(0), cases=cases, stats=Tensor([0]), output=Tensor([0]),
            buffer=SimpleNamespace(x=Tensor([0]), x_sf=Tensor([0]), topk_idx=Tensor([0]), topk_weights=Tensor([0])),
            deep_gemm=None, assert_guards=mock.Mock(),
        )
        harness.assert_stats = lambda expected, label: self.assertEqual(harness.stats.values, expected.values, label)
        static_idx, static_weights = Tensor([0]), Tensor([1])
        observed = []
        captured_x = harness.inputs.x

        def replay():
            route = static_idx.values[0]
            observed.append((captured_x.values[0] - 1, route))
            harness.output.values[:] = [captured_x.values[0] * 100 + route]
            harness.stats.add_(Tensor([route + 1]))

        graph = SimpleNamespace(replay=replay)

        def snapshots(h, *unused):
            return ({mode: Tensor([h.inputs.x.values[0] * 100 + index])
                     for index, mode in enumerate(cases)}, {})

        with mock.patch.object(accuracy, "_capture_graph", return_value=(graph, static_idx, static_weights)) as capture, mock.patch.object(
            accuracy, "_make_inputs", side_effect=lambda *a, input_epoch: inputs(input_epoch)
        ), mock.patch.object(accuracy, "_snapshot_and_check_oracles", side_effect=snapshots) as oracle:
            report = accuracy._run_payload_epoch_validation(harness, torch, SimpleNamespace(barrier=lambda: None))

        self.assertEqual(capture.call_count, 1)
        self.assertEqual(oracle.call_count, 4)
        self.assertEqual(observed, [(epoch, route) for epoch in (0, 1, 2, 0) for route in range(8)])
        self.assertEqual(harness.assert_guards.call_count, 32)
        self.assertIs(captured_x, harness.inputs.x)
        self.assertEqual(harness.inputs.x.values, [1])
        self.assertTrue(report["same_input_storage"])
        self.assertTrue(all(record["packed_sf_bytes_changed"] for record in report["rank_0_payload_records"][1:]))


if __name__ == "__main__":
    unittest.main()
