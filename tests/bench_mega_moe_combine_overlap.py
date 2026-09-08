"""Accuracy-gated completed-block COMBINE experiment; no compute tuning.

Two-node torchrun, eight ranks/node:
  python tests/bench_mega_moe_combine_overlap.py --decode-mns 8 --output RESULT.json

Only COMBINE_OVERLAP changes 0→1→0. DISPATCH_OVERLAP=1 and
SINGLE_COMBINE_CONTEXT=1 stay fixed. The shared driver preserves balanced
launch-only timings and untimed exact-count hot-owner, masked/asymmetric,
varying FP8/SF payload and same-allocation T48→64→48 gates. Device overlap and
the per-rank counter-storage fit branch are not inferred from environment flags.
"""

import bench_mega_moe_dispatch_overlap as shared


MODE_ENV = "DG_MEGAMOE_GIN_COMBINE_OVERLAP"
AXIS = shared.SweepAxis("combine", MODE_ENV, (
    (shared.MODE_ENV, "1"), (shared.COMBINE_ENV, "1")),
    ("tests/bench_mega_moe_combine_overlap.py",))


def combine_contract(mode):
    if type(mode) is not int or mode not in (0, 1):
        raise ValueError("combine overlap mode must be 0 or 1")
    return {
        "requested_combine_overlap": bool(mode),
        "requested_combine_overlap_raw": str(mode),
        "fixed_dispatch_overlap": 1, "fixed_single_combine_context": 1,
        "compute_tiling_sm_count_and_math_unchanged": True,
        "effective_policy_not_device_observation": True,
        "eligible_policy": "remote direct+bulk, all N producers complete, per-rank counter storage fits",
        "counter_storage_fit_is_per_rank": True,
        "counter_storage_capacity_policy": "ceil(16*48*16 / actual_block_m) + 55",
        "fit_failure_keeps_existing_full_packet_sc1_protocol": True,
        "completed_block_payload_may_issue_early": bool(mode),
        "count_headers_and_final_sc1_put_barrier_remain_late": True,
        "t48_t64_t48_overlap_request_policy": [bool(mode), False, bool(mode)],
        "all_local_remote_combine_skipped": True,
        "physical_compute_communication_overlap_measured": False,
    }


def parse_args():
    return shared.parse_args(AXIS)


def prepare_sweep_environment():
    shared.prepare_sweep_environment(AXIS)


def main():
    shared.main(AXIS, combine_contract)


if __name__ == "__main__":
    main()
