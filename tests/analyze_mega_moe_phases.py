"""Summarize diagnostic globaltimer samples using only same-GPU differences.

Usage: python tests/analyze_mega_moe_phases.py capture.json --output summary.json
These instrumented, host-synchronized observations are not benchmark latency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any


PHASES = {
    0: "entry", 1: "route_grid_done", 2: "prepack_done",
    3: "consensus_done", 4: "before_pull_barrier", 5: "after_pull_barrier",
    6: "dispatch_counts_cached", 7: "reconstructed_counts_published",
    52: "first_epilogue_task_begin",
    53: "epilogue_task_loop_exit", 54: "tmem_free_done",
    55: "combine_grid1_done", 56: "combine_barrier_begin",
    57: "combine_barrier_done", 58: "combine_grid2_done",
    59: "scatter_done_warp0_alias", 60: "scatter_grid_done", 61: "reduce_begin_warp0_alias",
    62: "dispatch_cleanup_region_observation", 63: "cleanup_rendezvous_exit",
    96: "first_mma_task_acquired", 97: "first_mma_operands_ready",
    98: "mma_scheduling_loop_exit",
}
for base, label in ((8, "dispatch_issue_begin"), (16, "dispatch_puts_queued"),
                    (24, "dispatch_flush_done"), (32, "recv_ready_observed"),
                    (40, "unpack_done"), (64, "combine_issue_begin"),
                    (72, "combine_put_queued"), (80, "combine_flush_done")):
    PHASES.update({base + peer: f"{label}_peer{peer}" for peer in range(8)})
PHASES.update({48 + warp: f"pull_done_warp{warp}" for warp in range(4)})
PHASES.update({88 + warp: f"reduction_loop_exit_warp{warp}" for warp in range(8)})
PHASES.update({104 + peer: f"receiver_wait_begin_peer{peer}" for peer in range(8)})
PHASES.update({112 + warp: f"scatter_done_warp{warp}" for warp in range(8)})
PHASES.update({120 + warp: f"reduce_begin_warp{warp}" for warp in range(8)})


def describe(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {"min": ordered[0], "p50": median(ordered), "max": ordered[-1]}


def summarize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Reduce SM rows from one rank and one replay, never across GPU clocks."""
    rows = sample["sm_markers_ns"]
    stamps: dict[int, list[int]] = {}
    for row in rows.values():
        for slot_text, value in row.items():
            slot = int(slot_text)
            if not isinstance(value, int) or value <= 0:
                raise ValueError("sparse markers must contain positive integer ns")
            stamps.setdefault(slot, []).append(value)
    if not stamps.get(0):
        raise ValueError("missing entry marker: diagnostic kernel may not be enabled")
    origin = min(stamps[0])

    def values(slots: range | tuple[int, ...]) -> list[int]:
        return [value for slot in slots for value in stamps.get(slot, [])]

    def span(begin: range | tuple[int, ...], end: range | tuple[int, ...],
             begin_policy: str = "min", end_policy: str = "max") -> float | None:
        lhs, rhs = values(begin), values(end)
        if not lhs or not rhs:
            return None
        start = min(lhs) if begin_policy == "min" else max(lhs)
        finish = min(rhs) if end_policy == "min" else max(rhs)
        # A negative value means the ranges overlapped in a different order;
        # retain it rather than inventing a serial critical path.
        return (finish - start) / 1000.0

    phases = {
        PHASES.get(slot, f"slot_{slot}"): {
            "first_us": (min(v) - origin) / 1000.0,
            "last_us": (max(v) - origin) / 1000.0,
            "writer_count": len(v),
        }
        for slot, v in sorted(stamps.items()) if slot < 128
    }
    experts = []
    for expert in range(56):
        first, last = stamps.get(128 + expert * 2), stamps.get(129 + expert * 2)
        if bool(first) != bool(last):
            raise ValueError(f"expert {expert} has an incomplete readiness pair")
        if first:
            experts.append({
                "expert": expert,
                "first_l2_epilogue_fragment_done_us": (min(first) - origin) / 1000.0,
                "last_l2_epilogue_fragment_done_us": (max(last) - origin) / 1000.0,
                "participating_sms": len(last),
            })
    metrics = {
        "route_to_consensus_us": span((1,), (3,)),
        "route_to_prepack_us": span((1,), (2,)),
        "dispatch_issue_to_last_flush_us": span(range(8, 16), range(24, 32)),
        "receiver_observed_ready_spread_us": span(range(32, 40), range(32, 40)),
        "last_observed_ready_to_last_unpack_us": span(range(32, 40), range(40, 48), "max"),
        "pull_barrier_span_us": span((4,), (5,)),
        "first_mma_operands_to_epilogue_loop_exit_us": span((97,), (53,)),
        "epilogue_work_span_us": span((52,), (53,)),
        "tmem_free_to_combine_issue_us": span((54,), range(64, 72), "max", "min"),
        "combine_issue_to_last_flush_us": span(range(64, 72), range(80, 88)),
        "combine_barrier_span_us": span((56,), (57,)),
        "combine_grid2_to_scatter_done_us": span((58,), range(112, 120)),
        "reduction_through_last_tma_issue_us": span(range(120, 128), range(88, 96)),
        "dispatch_cleanup_region_span_us": span((62,), (63,)),
        "last_scatter_grid_exit_to_cleanup_rendezvous_exit_us": span((60,), (63,), "max"),
        "entry_to_last_observed_marker_us": (max(max(v) for v in stamps.values()) - origin) / 1000.0,
    }
    if experts:
        complete = [expert["last_l2_epilogue_fragment_done_us"] for expert in experts]
        issue = values(range(64, 72))
        metrics["expert_last_l2_fragment_spread_us"] = max(complete) - min(complete)
        metrics["earliest_expert_last_fragment_to_first_combine_issue_us"] = (
            (min(issue) - origin) / 1000.0 - min(complete) if issue else None
        )
    return {"rank": sample["rank"], "replay": sample["replay"],
            "phase_windows_from_local_entry_us": phases,
            "expert_epilogue_observations": experts, "metrics_us": metrics}


def summarize_capture(capture: dict[str, Any]) -> dict[str, Any]:
    routes = {}
    for route, rank_samples in capture["routes"].items():
        samples = [summarize_sample(sample) for sample in rank_samples]
        metric_names = sorted({name for sample in samples for name in sample["metrics_us"]})
        routes[route] = {
            "samples": samples,
            "same_gpu_interval_distribution_us": {
                name: describe([sample["metrics_us"][name] for sample in samples
                                if sample["metrics_us"].get(name) is not None])
                for name in metric_names
            },
        }
    return {
        "schema": "megamoe-gin-phase-summary-v1", "shape": capture["shape"],
        "diagnostic_level": capture["diagnostic_level"],
        "kernel_configuration": capture.get("kernel_configuration"),
        "timing_scope": "instrumented_host_synchronized_same_gpu_globaltimer_differences",
        "performance_claim": False,
        "notes": [
            "GPU clocks are never subtracted across ranks.",
            "Intervals can overlap and must not be added as a serial breakdown.",
            "Slot62-to63 is an observed dispatch cleanup-region span, not isolated cleanup-barrier cost: recorded slot62 can precede combine issue/fencing on the same SM.",
            "Ready observations include polling order; they are not NIC arrival timestamps.",
            "Expert production completion uses max(last L2 epilogue fragment marker) over participating SMs; it is not a NIC visibility proof.",
            "Reduction end marks loop exit/last TMA store issue, not asynchronous store completion.",
            "Level 1 omits per-expert readiness; level 2 includes its instrumentation cost.",
        ],
        "routes": routes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize_capture(json.loads(args.capture.read_text()))
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        with args.output.open("x") as output:
            output.write(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
