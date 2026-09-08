# Clean context-1-only candidate

This is a separate, unvalidated candidate, not the selected production source.
No other experimental tree or archived measurement was changed to create it.
The only new transport optimization is
`DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT=0/1`, default **0**. The pre-existing
selected direct-dispatch/preconsensus optimizations and optional diagnostics
remain intact.

## Exact construction

The complete starting tree was
`megamoe-gin-preconsensus-pack-exp-20260904`, whose kernel SHA256 is
`dff445b04d657fa9ccb46e14d48b755de4c8e3592f3415cf4edfcd91343c4125`.
Four core files were replaced by the exact pre-wave diagnostic CONTROL from
`megamoe-overlap-diagnostic-r2.tgz`, archive SHA256
`10d8015e5d88283cbefc70447e36635bb0f044be65ca735fef39fb97446fbec6`:

| CONTROL file | SHA256 before ctx1 patch |
|---|---|
| `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` | `19caadaf85e6457e46d15c13533723be36eeffc491adf241be768ba7ee7049d8` |
| `deep_gemm/include/deep_gemm/comm/mega_moe_gin.cuh` | `4ab47c2f86aaadf1276b76db3541c7750df6635692655da64ff0a6b377e41fc0` |
| `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp` | `3b8cb43b6d1ec1641812e209dc378ad5f98af7fae039e9f066319e5eb0d84b06` |
| `csrc/apis/mega_gin.hpp` | `6872886e41f8a8fb7aad2bf27cc2a530b0f0ad01933ed3ee1c15b748c570d971` |

Only the first three files then received the context-1 transport patch. The
registration/API descriptor file is unchanged from CONTROL. No expert-wave
readiness counters, dispatch drainer, early receive-header clear, per-tile
release/fence additions, cooperative warp span, or extra named barrier were
ported. The layout, scheduler, and compiler remain those of the selected tree.

The current public-constructor preflight and an explicit test-file allowlist
were copied from the experimental sibling: base accuracy/contracts and
experiment-evidence contracts, payload
epochs, lifecycle, dense accuracy/contracts, phase capture/analyzer/contracts,
isolated DeepEP/TRTLLM comparator/contracts, scalar quantization recipe tests,
single-context sweep/model/contracts, and compile-only TU. Rejected experiment
runners and their implementation-specific tests were not copied.

## Supported behavior and guards

Eligible remote-active bulk+direct combine uses context 1 for every outgoing
peer packet, its local flush, and its public world `Put` barrier. Each peer
still owns a separate QP. Input publication keeps its original context mapping.
Nine contexts remain registered; the exact-r75/T64 fallback keeps all-context
completion; all-local launches skip the remote combine path.

Retired expert-wave settings other than 0 and cooperative-barrier settings
other than 1 are rejected, including while ctx1 is disabled. Their names exist
only as rejection/metadata guards, not JIT fields or device implementations.
The constructor and test harness validate protocol choices collectively before
launch. Subsequent same-allocation flag changes require a collective schedule,
as provided by the 0/1/0 sweep.

See [the ordering and lifecycle rationale](tests/MEGAMOE_SINGLE_COMBINE_CONTEXT.md).

## Validation commands

The separate artifact script `build_single_context_clean_node.sh` builds this
tree and runs only relevant CPU contracts. `smoke` compiles ctx1 diagnostics
0/1; `full` compiles baseline and ctx1 at diagnostics 0/1/2. Every TU includes
the T64 fallback. These are compile-only checks, not GPU correctness tests.

Use a rebuilt extension, a new process, and a fresh `DG_JIT_CACHE_DIR`. Then
run the unchanged-input `bench_mega_moe_single_combine_context.py` 0/1/0 sweep,
dense/heterogeneous-payload gates, T48/T64/T48 transitions, and isolated
DeepEP/TRTLLM comparator for MNS8/10/12. Diagnostic captures remain separate
from performance measurements. This candidate must establish its own accuracy
and timing evidence; experimental-tree results do not automatically transfer.
