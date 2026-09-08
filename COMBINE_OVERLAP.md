# Ready-span coalesced combine experiment

Status on2026-09-08: R3 source prepared; CUDA and GPU validation pending.
R2 peer-parallel source remains at48e349a and its separate deployment.
R2 passed its short GPU correctness screen but regressed to475/501us for
half/all remote versus390/409us controls; it is not a selected strategy.
R1 expert-ready source remains at cd1aaf9 and its separate deployment.
R1 passed its short GPU correctness screen but regressed to784/795us for
half/all remote versus390/411us controls; it is not a selected strategy.

This remains one fused MegaMoE kernel, not an EP1 compute replacement.
The existing dispatch-overlap and single-combine-context paths stay fixed.
Compute math, tiling, scheduler, SM count and warp allocation are unchanged.
DG_MEGAMOE_GIN_COMBINE_OVERLAP=1 is opt-in; default0 retains the baseline.

Dispatch saves exact source/expert prefixes and two nonempty masks per peer
during its existing count scan. Scratch is2304B after57344B direct control:
56 ready counts,56 final-bookkeeping entries,8x56 prefixes and8x2 mask words.
Existing offsets and registered allocation are unchanged. Exact per-rank fit
requires59648B of available tail storage; ineligible/fallback launches retain
their prior behavior, and the next eligible dispatch overwrites every mask.

Each expert becomes ready after all actual M-block/N-fragment producers
publish through the existing epilogue barrier and device-release counter.
Hot experts are not truncated or sized from a balanced-load hint.

SM0 dispatch warp0 starts draining only after its own pulls. All32 lanes
discover ready experts with uniform ballots. Each peer lane selects its own
pending ready expert, so different peers may issue different expert spans.
R3 freezes the current readiness snapshot and joins at most eight already-
ready, physically adjacent expert spans per peer into one unchanged required-
system, nonaggregate GIN PUT. Eight is a fixed compile-time cap, not a tuning
hint. It never waits for future readiness to fill a batch, never bridges a
positive unready/already-sent gap, and safely skips zero-length expert gaps.
Two uniform fixed-group target shuffles precede each peer-specific selection;
the issuing lane acquires every contributing expert's exact producer target.
Scalar pending bits clear only after PUT submission. Uniform early stop avoids
empty planning stages. There is no extra metadata, context, warp or SM.
The bounded lookahead is not free: marker99 precedes first acquire/lookahead.

Readiness discovery is monotonic and shared, while pending state is peer-local.
A send to one peer cannot retire another peer's pending expert. The preserved
sent array is final bookkeeping only, not compute or remote-visibility proof.
Late flushes, count headers, world Put completion, scatter, ordered reduction
and cleanup retain their existing protocol and lifetime boundaries.

Validation: tests/bench_mega_moe_combine_overlap.py compares combine0/1/0 in
one allocation, with native sparse, bitwise, hot/masked/asymmetric, changed
payload and T48/64/48 graph-reuse gates. Separate random-dense native accuracy
and fresh matched DeepEP/TRT pipeline measurements are required to accept a
performance winner. No numerical threshold is relaxed.

R1 diagnostics observed earliest issue about93us before last epilogue, but
all payloads were queued about404/415us AFTER last epilogue for half/all remote.
R2 reduced firstissue-to-allqueued to179/202us, with firstissue still about
93us before last epilogue. That residual interval motivates bounded ready-span
coalescing. These same-GPU software intervals
do not prove physical NIC/MMA overlap or isolate pure posting cost.

Execution records: /Users/girasoley/dev/projects/novita-b300-vllm/artifacts/20260908/megamoe-ready-coalesce/
R1/R2 evidence: ../megamoe-expert-ready/ and ../megamoe-peer-ready/.
