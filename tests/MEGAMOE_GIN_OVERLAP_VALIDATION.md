# Split-control dispatch validation

This is the validation plan for the September8 fused-GIN overlap experiment.
It does not change compute hints, tile selection, launch SM count, or the
existing mathematical operation. CPU models passing is not evidence that the
CUDA kernel implements these rules or that IB overlap improves performance.

## CPU specification implemented

Run from the repository root:

```sh
python3 -m unittest discover -s tests -p 'test_mega_moe_gin_overlap_protocol.py' -v
```

`mega_moe_gin_overlap_protocol.py` describes one owner's receive state. It uses
round-robin assignment identities `(source, expert, ordinal)` and generation
tags. It does not simulate warp scheduling, caches, NIC progress, or actual
atomic packing order. Round-robin membership is equivalent to the kernel's
iterative min-peeling selection, not a source-major concatenation that could
incorrectly make the first tile appear ready after only one source arrives.

Required ordering:

- Source issues count/control before activation, scale, and weight payload.
- Receiver acquires exact counts for every contributing source, including zero
  counts, before constructing expert tile membership. A partial final tile is
  ready only when all its actual assignments are ready; padding is not work.
- The control terminal permits scheduling decisions, not payload loads. Every
  source contributing to a tile must have its payload terminal acquired. That
  terminal covers all three SoA fields. A complete tile can proceed while
  another tile waits on a different source.
- Local flush completion permits sender-buffer reuse only; it does not prove
  receiver visibility. Ring scratch cannot be reused at tile issue or before
  the consuming computation has completed its reads.
- All expected empty/nonempty peer terminals and consumers drain before the
  next generation reuses receive state. Inactive sources must not leave stale
  readiness or counts behind.
- The existing control epoch advances on every eligible baseline/split invocation; the
  new payload-terminal epoch advances only when split dispatch actually posts
  that terminal. Ineligible T64 fallback advances neither direct-dispatch
  terminal. The model covers off/on/off/on plus an ineligible fallback.
  This is an expected protocol contract, not a test of actual device counters.

Negative tests reject stale payload tags, equal-total/wrong-expert membership,
incomplete contributing sources, a missing SoA component, premature signals,
scratch overwrite, and premature next-generation reuse.

## Existing GPU coverage and its gap

`test_mega_moe_accuracy.py` already provides eager/graph snapshots, exact
per-expert assignment counts, input/output canaries, NaN poisoning, masked
outputs, a single-remote-source case, and changing FP8/SF payload epochs with
stable allocations. `_run_payload_epoch_validation` recomputes the analytic
quantization-aware oracle on each epoch. The dense runner independently uses
distinct owner weights and its own numerical budget, with exact candidate vs
baseline transport checks.

The concentrated `_build_bulk_transition_route_tensors` fixture pairs every
source with one owner, so owner totals remain balanced. Its T48→64→48 sequence
also changes eligibility/configuration. It is useful for fallback lifecycle,
but does not cover fixed-configuration severe owner imbalance.

## GPU runner and completed device gates

The September8 Novita two-node execution passed the full dispatch0/1/0
sweeps at MNS8/10/12, including the fixed-T stress below. Separate dense
dispatch0-vs1 gates also passed at all three sizes with combine context1
fixed. Raw logs, per-rank samples, fingerprints and independent audits are
under `dev/novita-b300-vllm/artifacts/20260908/megamoe-fused-overlap/` in the
shared workspace. The remaining overlap-observation gate is not satisfied
by those correctness and latency results.

`bench_mega_moe_dispatch_overlap.py` reuses the existing accuracy workflow and
single-context runner utilities. It sweeps dispatch0/1/0 with COMBINE context1
fixed, preserves the matched timing fixtures, and requires per-mode payload,
fallback, and fixed-T imbalance checks. Its eight CPU policy contracts pass;
the archived GPU runs, rather than those CPU contracts, support the device
result above. A short device smoke for a new allocation is:

```sh
# Invoke with the existing two-node torchrun setup (eight ranks per node).
python tests/bench_mega_moe_dispatch_overlap.py --decode-mns 8 \
  --benchmark-replays 8 --benchmark-conditioning-replays 2 \
  --eager-iterations 2 --graph-replays 2 --payload-epochs 2 --output RESULT.json
```

Fixed-T stress keeps EP16/H3584/I3072/E896/K16, capacity384 and every compute
flag identical. The following fixtures are modeled; the GPU runner currently
uses hot owner8 and hot owner0 separately, not the paired-hot-owner option:

| Fixture | Active sources | Active owners | Maximum owner assignment count |
|---|---:|---:|---:|
| Balanced |16|16|16T|
| Single hot owner8 |16|1|256T|
| Paired hot owners0/8, all cross-host |16|2|128T|
| Only source0→owner8 |1|1|16T|
| Empty |0|0|0|

For hot cases, local expert ID equals top-k slot0..15: distinct experts within
each token, with all activity concentrated on16 of56 experts. The single hot
owner receives512/640/768 assignments per active expert at T32/40/48. Per
source-owner packet counts remain512/640/768, so this stress does not itself
exceed the existing768 compact-packet limit. Validate actual count histograms
and launch evidence; do not infer runtime fast-path use solely from this bound.

The GPU runner uses one candidate graph with balanced→hot8→hot0→balanced→empty
→hot8→asymmetric→balanced and changing input/SF epochs0→1→0 (default also tests
epoch2) in the same storage. Heterogeneous SF is enabled only for this untimed
stress; original timing input bytes are restored afterward. It retains exact
per-rank, per-expert counts: totals alone cannot catch wrong-expert placement.
All control snapshots are collected before consecutive candidate replays, so
an intervening baseline cannot refresh a stale payload between candidate steps.

For every case/epoch, obtain the unchanged-protocol baseline on exactly the
same assignments and values, then require candidate bitwise equality plus the
existing independent sparse oracle, finite output, guards, and exact stats.
Do not compare hot-case output with an unrelated balanced-case snapshot: the
local expert IDs differ. In the dense test, use its existing distinct-owner
reference and acceptance policy without widening tolerances. Candidate and
baseline may differ only in the communication experiment flag.

Capture baseline/split/baseline graphs collectively on the same allocation to
check the mode epoch sequence, and keep the separate T48→64→48 fallback test.
Require rank-uniform experiment settings before first launch. Preserve all
graphs and tensor addresses through replay and retire graphs before teardown.

Use an externally bounded multi-rank timeout and retain both-node status files
and raw failures. No new GPU run, performance estimate, or liveness proof is
provided by these CPU tests. A later profile must demonstrate actual early
compute progress; correctness alone does not establish overlap.
