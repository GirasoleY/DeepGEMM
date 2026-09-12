// Compile-only EP8/E448 coverage; the existing EP16 TU is unchanged.
// Compile twice with DG_MEGAMOE_GIN_COMBINE_OVERLAP=0/1 for R7b controls.
// DG_EP8_COMPILE_NATIVE=1 selects the native EP8 specialization instead.
// These representative template profiles do not override runtime heuristics:
// actual selected stages/ring sizes must still be recorded by the GPU harness.
#ifndef DG_EP8_COMPILE_NATIVE
#define DG_EP8_COMPILE_NATIVE 0
#endif

#if !DG_EP8_COMPILE_NATIVE
#ifndef DG_MEGAMOE_GIN_ACTIVE_FAST_PATH
#define DG_MEGAMOE_GIN_ACTIVE_FAST_PATH 1
#endif
#ifndef DG_MEGAMOE_GIN_ACTIVITY_GATE_OPT
#define DG_MEGAMOE_GIN_ACTIVITY_GATE_OPT 1
#endif
#ifndef DG_MEGAMOE_GIN_BULK_COMBINE
#define DG_MEGAMOE_GIN_BULK_COMBINE 1
#endif
#ifndef DG_MEGAMOE_GIN_DIRECT_DISPATCH
#define DG_MEGAMOE_GIN_DIRECT_DISPATCH 1
#endif
#ifndef DG_MEGAMOE_GIN_DISPATCH_WARP_SCAN
#define DG_MEGAMOE_GIN_DISPATCH_WARP_SCAN 1
#endif
#ifndef DG_MEGAMOE_GIN_COOP_DIRECT_PACK
#define DG_MEGAMOE_GIN_COOP_DIRECT_PACK 1
#endif
#ifndef DG_MEGAMOE_GIN_PRECONSENSUS_PACK
#define DG_MEGAMOE_GIN_PRECONSENSUS_PACK 1
#endif
#ifndef DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT
#define DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT 1
#endif
#ifndef DG_MEGAMOE_GIN_DISPATCH_OVERLAP
#define DG_MEGAMOE_GIN_DISPATCH_OVERLAP 1
#endif
#ifndef DG_MEGAMOE_GIN_COMBINE_OVERLAP
#define DG_MEGAMOE_GIN_COMBINE_OVERLAP 1
#endif
#ifndef DG_MEGAMOE_GIN_STRONGVA_COMBINE_TERMINAL
#define DG_MEGAMOE_GIN_STRONGVA_COMBINE_TERMINAL 1
#endif
#endif

#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh>

using namespace deep_gemm;

namespace {

template <uint32_t BlockM, uint32_t BlockK, uint32_t StoreBlockM,
          uint32_t NumStages, uint32_t NumEpilogueThreads = 256>
void instantiate_ep8_decode() {
    auto* kernel = &sm100_fp8_fp4_mega_moe_impl<
        384,                         // unchanged public capacity
        3584, 3072,
        448, 0, 16,                  // 56 local experts, original top-k
        BlockM, 128, BlockK,
        StoreBlockM,
        128, 128,
        47616, 761856,               // conservative representative ring bounds
        NumStages,
        3584,
        128, 128, NumEpilogueThreads,
        152, 8,                      // validated GB200 SMs, actual EP8 world
        10.0f,
        false, !DG_EP8_COMPILE_NATIVE>;
    (void)kernel;
}

void instantiate_all_ep8_decode_tiles() {
    instantiate_ep8_decode<16, 256, 8, 5>();
    instantiate_ep8_decode<32, 128, 16, 10>(); // T32/40/48
    instantiate_ep8_decode<64, 128, 32, 9, 128>(); // T64 row fallback
}

}  // namespace
