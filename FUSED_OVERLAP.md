# Fused GIN overlap experiment

Starting point: validated clean fused kernel, commit `9dd41c1`. The staged
DeepEP/EP1 path is a comparator, not this implementation. Keep compute math,
tiling, scheduler, ring layout, and SM configuration unchanged.

## First A/B: control-first dispatch

`DG_MEGAMOE_GIN_DISPATCH_OVERLAP=0` preserves baseline publication ordering.
Mode 1 requires the existing direct/bulk/preconsensus-pack path:

1. Each source posts exact expert counts and assignment metadata, with a
   StrongVA control terminal, directly to every remote owner.
2. It posts the chained activation/scale/router-weight SoA payload with a
   separate StrongVA terminal, without blocking on a sender flush.
3. Owners reconstruct exact counts after control arrival. Existing pull warps
   acquire each needed source's payload terminal before any mirror access,
   caching that acquire per source/warp. The original complete-M-tile L1
   readiness still controls compute; partial tiles are never released early.
4. After pulls, the existing issuing warp retires all eight inbound payload
   terminals, including empty peers, and flushes outgoing chains once per
   peer/context before the original dispatch/epilogue handoff and cleanup.

No per-assignment request, new communicator, allocation, or transposed wire
format. The payload terminal uses the spare second 64-bit receive-header word;
its expected epoch uses workspace bytes112..119. Separate control/payload
epochs support baseline/candidate graph interleaving; ineligible T64 and local
fast-path launches advance neither. Public setup validates rank-uniform raw
flags. Subsequent graph captures require caller-enforced matching modes/order.

This removes a dependency, not all head-of-line waiting: original round-robin
tile membership can still require several remote sources. Whole-source
readiness deliberately isolates the protocol change before considering finer
granularity. A late flush timestamp is an observation, not proof that the NIC
was busy until that timestamp. Do not claim physical overlap from it alone.

## Validation and next step

Run `tests/bench_mega_moe_dispatch_overlap.py` with the prepared two-node
launcher. It holds combine-context selection at1, sweeps dispatch0/1/0, retains
matched timing fixtures, and separately tests fixed-shape extreme imbalance,
masked/asymmetric activity, payload epochs, and T48/T64/T48 fallback. Sparse
native-oracle checks and baseline bitwise comparisons are mandatory; dense
numerics are a separate gate. CPU model/contract tests are not GPU validation.

Combine is intentionally unchanged in this A/B. The next separate experiment
is completion-driven publication of full output blocks while other compute
continues. Do not reuse the rejected fixed-order expert-wave implementation
without addressing its measured producer/readiness costs.

Execution artifacts and authoritative current status:
`dev/novita-b300-vllm/artifacts/20260908/megamoe-fused-overlap/` in the shared
workspace. The September8 two-node MNS8/10/12 dispatch0/1/0 GPU sweeps and
separate dense accuracy gates passed. Remote-route isolated medians improved
about5–7% against the unchanged-protocol controls. This is a dispatch latency
result, not proof of physical IB/compute overlap; combine is still unchanged.
See the artifact README for matched comparator results and numerical scope.
