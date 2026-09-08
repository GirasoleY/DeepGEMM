# Completed-block combine experiment

Status on 2026-09-08: source prepared and independently reviewed; 259 local
CPU tests passed. CUDA compilation, device accuracy/liveness, and timings are
pending. This is not a newly GPU-validated checkpoint or a measured speedup.

This is a separate opt-in experiment on dispatch checkpoint `ee1b644`, not a
compute-only EP1 replacement. Compute math, tile selection, scheduler, SM count
and warp roles remain unchanged. `FUSED_OVERLAP.md` describes the validated
dispatch baseline; its reported results do not validate this new combine mode.

`DG_MEGAMOE_GIN_COMBINE_OVERLAP=1` requires the selected dispatch-overlap and
single-context-combine modes. Default0 preserves the prior path. Runtime local
or T64/ineligible launches retain their old behavior. An actual-BM scratch fit
guard also falls back without changing registered allocation or offsets.

The L2 epilogue's existing full-thread barrier publishes each completed N
fragment through a device-release counter. A logical M-block becomes ready
only after all its N fragments finish. SM0 dispatch warp0, after its own pulls,
acquires those counters and sends ready compact record spans. It scans around
unfinished blocks and merges only adjacent, entirely ready per-peer records.
No producer waits for sends or reuses the immutable output slab.

Readiness uses full logical pool blocks, never reusable ring indices. Both
counter and claimed-state arrays alias only the unused tail after the compact
direct-control storage. Initialize them on every eligible invocation, including
after a fallback. Every issuing lane acquires readiness; GIN's explicit system
release then publishes producer data before NIC access. Default non-aggregate
PUTs retain queue-credit checks and doorbell submission.

Flush early sends before the original dispatch/epilogue handoff. Publish only
the packet count header at the original late point, retaining the final world
Put barrier, scatter and reduction. This prevents duplicate payload sends and
keeps the receiver's startup count-clear ordering unchanged.

Validation entry point: `tests/bench_mega_moe_combine_overlap.py`, which reuses
the dispatch runner's native sparse oracle, exact counts, bitwise comparisons,
graph payload changes, fixed-T extreme imbalance and T48/64/48 lifetime gates.
It fixes dispatch1/SC1 and varies only combine0/1/0. Dense native-reference
accuracy is separate, with unchanged numerical thresholds and identical
context/weights/storage for the combine0 control.

Current execution status and raw evidence belong in
`dev/novita-b300-vllm/artifacts/20260908/megamoe-combine-overlap/` in the shared
workspace. Local source/helper tests do not establish CUDA compilation, device
correctness, queue-wrap liveness, or a performance improvement. Diagnostics of
early PUT issue versus remaining epilogue work are not physical IB/MMA traces.
