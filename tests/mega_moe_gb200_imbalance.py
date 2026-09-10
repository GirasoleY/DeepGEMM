"""Untimed EP8 fixed-shape imbalance gate; no tuning or transport-mode switch.

One retained graph sees hot, empty, asymmetric and balanced routes. Eager
references are prepared before, never interleaved with, its sequence. CPU
models/tests do not constitute GPU evidence.
"""

import math
import sys

import test_mega_moe_accuracy as accuracy
from probe_gb200_topology import _gather, _phase


SEQUENCE = ("all_remote", "hot_owner4", "hot_owner0", "all_remote",
            "all_masked", "hot_owner4", "single_active_hot_owner4", "all_remote")


def hot_routes(source, tokens, owner, only_source=None):
    if source not in range(8) or owner not in range(8) or tokens not in (32, 40, 48):
        raise ValueError("hot routes require EP8 and T32/40/48")
    if only_source is not None and only_source not in range(8):
        raise ValueError("invalid active source")
    ids = [[owner * 56 + slot if only_source is None or source == only_source else -1
            for slot in range(16)] for _ in range(tokens)]
    weights = [[2.0 ** (-(slot % 3 + 1)) if expert >= 0 else 0.0
                for slot, expert in enumerate(row)] for row in ids]
    return ids, weights


def _identity(harness):
    return (accuracy._input_storage_pointers(harness.inputs, harness.buffer),
            id(harness.buffer), id(getattr(harness.buffer, "gin_context", None)),
            int(harness.buffer.buffer.data_ptr()), int(harness.output.data_ptr()),
            tuple(int(tensor.data_ptr()) for matrix in (harness.weights.l1, harness.weights.l2)
                  for tensor in matrix))


def _mutable_tensors(harness):
    count = harness.args.num_tokens
    values = {"input_x": harness.inputs.x, "oracle_x": harness.inputs.oracle_x,
              "buffer_x": harness.buffer.x[:count],
              "buffer_ids": harness.buffer.topk_idx[:count],
              "buffer_weights": harness.buffer.topk_weights[:count],
              "output": harness.output, "stats": harness.stats}
    if harness.inputs.x_sf is not None:
        values.update(input_sf=harness.inputs.x_sf, buffer_sf=harness.buffer.x_sf[:count])
    return values


def _restore(values, saved, torch):
    for name, tensor in values.items():
        tensor.copy_(saved[name])
    torch.cuda.synchronize()
    for name, tensor in values.items():
        if not torch.equal(tensor.contiguous().view(torch.uint8),
                           saved[name].contiguous().view(torch.uint8)):
            raise AssertionError(f"imbalance restoration changed {name} bytes")


def _check(harness, case, expected, reference, label):
    torch, dist = harness.torch, harness.dist
    def local():
        relative, maximum = accuracy._symmetric_diff(harness.output, expected)
        rel_limit, abs_limit = accuracy._oracle_thresholds(harness.args)
        if (not all(map(math.isfinite, (relative, maximum)))
                or relative > rel_limit or maximum > abs_limit):
            raise AssertionError(f"unchanged sparse oracle failed: {relative}, {maximum}")
        harness.assert_stats(case.expected_local_stats, label)
        harness.assert_guards(label)
        if case.name == "all_masked" and not torch.equal(harness.output, torch.zeros_like(harness.output)):
            raise AssertionError("masked output must be exactly zero")
        if reference is not None:
            accuracy._assert_bitwise_equal(harness.output, reference, label, torch)
        return {"relative": relative, "max_abs": maximum}
    records = _gather(dist, _phase(dist, label, local))
    return {"worst_symmetric_diff": max(item["relative"] for item in records),
            "worst_max_abs": max(item["max_abs"] for item in records)}


def validate_ep8_imbalance(harness):
    torch, dist = harness.torch, harness.dist
    if dist.get_world_size() != 8 or harness.experts_per_rank != 56:
        raise ValueError("explicit EP8 / 56 experts per rank required")
    initial = _phase(dist, "imbalance/identity", lambda: _identity(harness))
    values = _phase(dist, "imbalance/mutable_views", lambda: _mutable_tensors(harness))
    saved = _phase(dist, "imbalance/save", lambda: {key: value.clone() for key, value in values.items()})
    graph = None
    checks, counts = [], {}
    epochs = accuracy._payload_epoch_sequence(harness.args.payload_epochs)
    def prepare(case):
        harness.copy_inputs(case)
        harness.stats.zero_()
        harness.output.fill_(float("nan"))
    def execute(call):
        call()
        torch.cuda.synchronize()
    try:
        cases = {name: harness.cases[name] for name in ("all_remote", "all_masked")}
        for name, owner, only_source in (("hot_owner4", 4, None), ("hot_owner0", 0, None),
                                        ("single_active_hot_owner4", 4, 0)):
            def make_routes():
                ids, weights = hot_routes(harness.rank, harness.args.num_tokens, owner, only_source)
                return (torch.tensor(ids, dtype=torch.int64, device="cuda"),
                        torch.tensor(weights, dtype=torch.float32, device="cuda"))
            ids, weights = _phase(dist, "imbalance/routes/" + name, make_routes)
            stats = accuracy._global_route_stats(ids, harness.rank, 56, torch, dist)
            cases[name] = accuracy.RouteCase(name, ids, weights, stats, "fixed_shape_hot_owner")
        for name, case in cases.items():
            counts[name] = _gather(dist, _phase(dist, "imbalance/counts/" + name,
                                              lambda: case.expected_local_stats.cpu().tolist()))
        references, inputs = {}, {}
        for epoch in dict.fromkeys(epochs):
            inputs[epoch] = _phase(dist, "imbalance/input", lambda: accuracy._make_inputs(
                harness.rank, harness.args, torch, harness.deep_gemm, input_epoch=epoch))
            _phase(dist, "imbalance/copy_input", lambda: accuracy._copy_input_storage(
                harness.inputs, inputs[epoch]))
            for name, case in cases.items():
                expected = _phase(dist, "imbalance/oracle", lambda: accuracy._analytic_oracle(
                    harness.inputs.oracle_x, case, harness.args, harness.experts_per_rank, torch))
                _phase(dist, "imbalance/eager_prepare", lambda: prepare(case))
                _phase(dist, "imbalance/eager_execute", lambda: execute(harness.launch))
                _check(harness, case, expected, None, f"imbalance/eager/{epoch}/{name}")
                references[(epoch, name)] = (_phase(dist, "imbalance/reference_clone",
                                                    harness.output.clone), expected)
        graph = accuracy._capture_launch_only_graph(harness, torch, dist)
        for epoch in epochs:
            _phase(dist, "imbalance/copy_input", lambda: accuracy._copy_input_storage(
                harness.inputs, inputs[epoch]))
            for step, name in enumerate(SEQUENCE):
                case = cases[name]
                reference, expected = references[(epoch, name)]
                _phase(dist, "imbalance/graph_prepare", lambda: prepare(case))
                _phase(dist, "imbalance/graph_execute", lambda: execute(graph.replay))
                metrics = _check(harness, case, expected, reference,
                                 f"imbalance/graph/{epoch}/{step}/{name}")
                checks.append({"epoch": epoch, "step": step, "case": name, **metrics})
        def check_identity():
            if _identity(harness) != initial:
                raise AssertionError("imbalance changed storage/context/weights")
        _phase(dist, "imbalance/stable_identity", check_identity)
    finally:
        active_error = sys.exc_info()[0] is not None
        cleanup_errors = []
        for action in (torch.cuda.synchronize,
                       lambda: graph.reset() if graph is not None else None,
                       lambda: _phase(dist, "imbalance/restore", lambda: _restore(values, saved, torch))):
            try:
                action()
            except Exception as error:
                cleanup_errors.append(str(error))
        if cleanup_errors:
            if not active_error:
                raise RuntimeError(f"imbalance cleanup failed: {cleanup_errors}")
            print(f"imbalance cleanup also failed: {cleanup_errors}", file=sys.stderr, flush=True)
    return {"enabled": True, "timed": False, "world_size": 8,
            "fixed_tokens_per_rank": harness.args.num_tokens,
            "sequence": list(SEQUENCE), "payload_epochs": epochs,
            "same_graph_and_storage": True, "no_interleaved_eager_controls": True,
            "eager_and_graph_sparse_oracle": True, "graph_bitwise_eager": True,
            "stats_canaries_nan_prefill": True, "no_configuration_or_hint_change": True,
            "mutable_tensor_bytes_restored": True, "graph_retired": True,
            "exact_per_rank_expert_assignment_counts": counts,
            "dense_numerical_claim": False, "checks": checks}
