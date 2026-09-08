# Optional single-context bulk combine

`DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT=1` is the sole new transport
optimization in this clean, default-off candidate. It does not change the registration/allocation ABI, packet format,
activation publication, or source-side combine reduction. It is not yet a
performance result.

The host requires bulk combine and direct dispatch, expert waves off, and
combine-barrier width 1. The accuracy harness collectively checks this
protocol-changing flag before its first kernel launch; every rank must agree.
Changing the flag after that check also requires a collective runner check.

## Scope

For a world-uniform remote-active, bulk+direct-eligible launch, all eight
outgoing bulk combine peer packets use data stripe 0, which is context 1.
Each peer still has a distinct QP. Each packet's local flush and completion
wait use that same context. Peer-lane timestamp slots remain independent of
the chosen transport stripe.

The postcompute world barrier uses the public NCCL context-1 `Put` fence,
with the existing combine barrier index. Thus its nominal work becomes 16
context/peer pairs rather than 9 × 16 = 144. This is a work-count model, not a
latency prediction. All nine contexts remain allocated.

All-local launches still skip the remote combine protocol. Non-bulk or
non-direct launches, including the same-allocation T64 exact-r75 fallback,
retain their original striping and all-context `Put` barrier. The earlier
direct-input paths still use all eight data stripes. The final context-zero
cleanup rendezvous and all existing system fences, local grid joins, handoff,
scatter, and reduction remain unchanged.

## Ordering and replay rationale

NCCL 2.30.7 `gin.h` specifies that a StrongVA terminal settles preceding PUTs
on the same context to the same peer. A flush alone only permits local source
reuse; it does not prove remote visibility. This experiment therefore retains
a matching `Put` barrier, rather than substituting a context-zero milestone.

Before compute, the direct-input sender has flushed each original context's
publication, and the receiver has acquire-observed all eight source terminals.
The eligible owner pull path consumes local published mirrors, not unfinished
GIN GETs. Consequently those input contexts have no outstanding visibility or
reuse obligation that requires the later combine barrier to include them.
Local producer fences and joins are still necessary and are preserved.

NCCL barrier signal/shadow counters are independently indexed per context.
Context 1 advances on both the single-context and all-context combine paths.
The other contexts pause together on world-uniform eligible launches and
advance together on fallback launches. Serialized launch completion and the
unchanged cleanup preserve slab reuse. This permits off/on/off and
T48/T64/T48 transitions without resetting or borrowing another signal slot.

Independent review used the actual installed NCCL 2.30.7 headers; its
`impl/gin_barrier__funcs.h` SHA256 is
`0f91193cddd83d976820475877ced786f39b077238736c5298aebc7da39fd893`.
This argument does not claim that a fence on one context can settle PUTs on
another context. All combine PUTs must move with the selected barrier scope.

## Required evidence

CPU contracts cover scope selection, per-context counter transitions, distinct
peer QPs/diagnostic slots, default-off JIT specialization, and unchanged host
allocation requirements. They are not a device-ordering proof.

Before retaining the experiment, run same-allocation 0/1/0 controls with
unchanged timing input bytes; eager and captured-graph correctness; exact
receive counts, guards and NaN poisoning; changing payloads and scales;
all routing/active-rank cases; T48/T64/T48 transitions; and an independent
dense/random expert-weight test. Report uninstrumented kernel timings
separately from level-1/2 phase diagnostics and calibration.

The separate clean artifact build script supports a bounded compile-only matrix:

```text
bash build_single_context_clean_node.sh /data/k3-prod/megamoe-gin-single-context-clean-exp-20260906 full
```

This builds baseline and context-1 modes at diagnostics levels 0/1/2. Every TU
also instantiates the four-epilogue-warp T64 fallback. It launches no GPU test.
For a shorter first build, the `smoke` matrix compiles only the
context-1 mode at diagnostics levels 0/1 (two TUs, both still including T64).
It does not claim compile-only coverage of baseline or level 2; actual runner
JIT executions can supply those separate coverage records.
