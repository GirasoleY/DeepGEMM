"""Untimed, event-serialized same-buffer GPU validation, never concurrent use.

Imported by the combine sweep only. The small enqueue primitive is separately
mock-tested on the laptop; CUDA execution is evidence only when the sweep runs.
"""

from types import SimpleNamespace
from unittest.mock import patch


SERIALIZED_SEQUENCE = (
    (0, "all_remote"), (1, "half_remote"), (1, "all_masked"),
    (0, "all_local"), (1, "all_remote"), (0, "all_remote"),
)


def enqueue_event_serialized(torch, callbacks):
    """Queue all work without a host wait; retain every dependency until done.

Each callback includes input copies, counter reset, launch, and result snapshots.
The caller synchronizes and checks results only AFTER this function returns.
"""
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    initial = torch.cuda.Event()
    initial.record(torch.cuda.current_stream())
    events, results = [initial], []
    for index, callback in enumerate(callbacks):
        stream = streams[index % 2]
        with torch.cuda.stream(stream):
            stream.wait_event(events[-1])
            results.append(callback())
            event = torch.cuda.Event()
            event.record(stream)
            events.append(event)
    torch.cuda.current_stream().wait_event(events[-1])
    return results, (streams, events)


def _device_guard_snapshot(harness, torch):
    """GPU-only Boolean work, with no item()/CPU transfer in the queued chain."""
    checks = [
        (harness.prefix == harness.prefix_expected).all(),
        (harness.suffix == harness.suffix_expected).all(),
        torch.isfinite(harness.output).all(),
    ]
    start = harness.args.num_tokens
    for name, expected in harness.tail_expected.items():
        # FP8 elementwise equality is not available in every torch build.
        # SF can be column-major, so materialize a contiguous byte view.
        actual_bytes = getattr(harness.buffer, name)[start:].contiguous().view(torch.uint8)
        expected_bytes = expected.contiguous().view(torch.uint8)
        checks.append((actual_bytes == expected_bytes).all())
    return torch.stack(checks).clone()


def validate_serialized_streams(harness, graph, mode, torch, dist, *,
                                accuracy, common, axis, validate_reference):
    """All controls first, then two six-launch candidate-only event chains.

The borrowed candidate graph is not recaptured or retired here. The temporary
control graph is retired in finally, before the caller can destroy its context.
No timing is measured and no correctness tolerance is introduced by this test.
"""
    if axis.name != "combine":
        raise ValueError("serialized-stream validation is a combine-axis gate")
    identity = common._identity(harness)
    initial_inputs = type(harness.inputs)(
        harness.inputs.x.clone(),
        None if harness.inputs.x_sf is None else harness.inputs.x_sf.clone(),
        harness.inputs.oracle_x.clone())
    count = harness.args.num_tokens
    initial_buffer = {name: getattr(harness.buffer, name)[:count].clone()
                      for name in ("x", "topk_idx", "topk_weights")}
    if harness.inputs.x_sf is not None:
        initial_buffer["x_sf"] = harness.buffer.x_sf[:count].clone()
    initial_output, initial_stats = harness.output.clone(), harness.stats.clone()
    stress_args = SimpleNamespace(**{**vars(harness.args), "heterogeneous_input_scales": True})
    payloads = {epoch: accuracy._make_inputs(
        harness.rank, stress_args, torch, harness.deep_gemm, input_epoch=epoch)
        for epoch in (0, 1)}
    graphs, references, records = {}, {}, []
    keepalive = None
    try:
        with patch.dict("os.environ", {axis.mode_env: "0"}):
            baseline = accuracy._capture_launch_only_graph(harness, torch, dist)
            graphs["serialized-control"] = baseline
        for epoch, name in dict.fromkeys(SERIALIZED_SEQUENCE):
            case = harness.cases[name]
            accuracy._copy_input_storage(harness.inputs, payloads[epoch])
            harness.copy_inputs(case)
            harness.output.fill_(float("nan"))
            harness.stats.zero_()
            baseline.replay()
            torch.cuda.synchronize()
            expected = accuracy._analytic_oracle(harness.inputs.oracle_x, case,
                harness.args, harness.experts_per_rank, torch)
            label = f"combine/mode{mode}/serialized/control/epoch{epoch}/{name}"
            validate_reference(harness, case, None, expected, label, torch, dist)
            references[epoch, name] = harness.output.clone()
        changed = not torch.equal(payloads[0].x.view(torch.uint8), payloads[1].x.view(torch.uint8))
        scales_changed = payloads[0].x_sf is not None and not torch.equal(
            payloads[0].x_sf, payloads[1].x_sf)
        accuracy._collective_transition_check("serialized/payload-preflight",
            ([] if changed and scales_changed else ["expected changed FP8 and SF payload bytes"]), dist)
        torch.cuda.synchronize()
        dist.barrier()
        for kind in ("eager", "retained_graph"):
            callbacks = []
            for epoch, name in SERIALIZED_SEQUENCE:
                def step(epoch=epoch, name=name):
                    # EVERY mutation of the shared allocation follows the
                    # preceding completion event, including copies and NaNs.
                    accuracy._copy_input_storage(harness.inputs, payloads[epoch])
                    harness.copy_inputs(harness.cases[name])
                    harness.output.fill_(float("nan"))
                    harness.stats.zero_()
                    if kind == "eager":
                        harness.launch()
                    else:
                        graph.replay()
                    return (harness.output.clone(), harness.stats.clone(),
                            _device_guard_snapshot(harness, torch))
                callbacks.append(step)
            snapshots, keepalive = enqueue_event_serialized(torch, callbacks)
            # There is deliberately no synchronize or distributed operation
            # between the six launches above. Join only after the last event.
            torch.cuda.synchronize()
            errors = []
            for step_index, ((epoch, name), (output, stats, guards)) in enumerate(
                    zip(SERIALIZED_SEQUENCE, snapshots)):
                label = f"serialized/{kind}/{step_index}/epoch{epoch}/{name}"
                if not torch.equal(output.view(torch.int16), references[epoch, name].view(torch.int16)):
                    errors.append(label + ": output differs bitwise from prebuilt combine0 reference")
                if not torch.equal(stats, harness.cases[name].expected_local_stats):
                    errors.append(label + ": expert stats mismatch")
                if not bool(guards.all().item()):
                    errors.append(label + ": finite output or allocation canary failure")
                records.append({"launch_kind": kind, "step": step_index, "stream": step_index % 2,
                                "payload_epoch": epoch, "route": name})
            accuracy._collective_transition_check("serialized/" + kind, errors, dist)
            keepalive = None
    finally:
        # Even on a validation exception, graph-owned work and borrowed input
        # storage remain alive until the queued chain has completed.
        torch.cuda.synchronize()
        keepalive = None
        common._retire_graphs(graphs)
        accuracy._copy_input_storage(harness.inputs, initial_inputs)
        for name, value in initial_buffer.items():
            getattr(harness.buffer, name)[:count].copy_(value)
        harness.output.copy_(initial_output)
        harness.stats.copy_(initial_stats)
        torch.cuda.synchronize()
    common._check_identity(harness, identity, "serialized/restored-identity", dist)
    return {"enabled": True, "timed": False, "concurrent_same_buffer_use": False,
            "streams": 2, "dependency": "previous completion event before all copies/reset/launch",
            "explicit_host_sync_or_collective_between_tested_launches": False,
            "allocator_internal_synchronization_not_instrumented": True,
            "candidate_sequence_has_no_interleaved_control_replays": True,
            "same_retained_graph_and_storage": True, "per_launch_output_and_stats_snapshot": True,
            "same_context_combine0_bitwise": True, "heterogeneous_input_scales": True,
            "all_rank_validation": True, "checks": records}
