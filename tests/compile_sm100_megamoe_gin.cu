// Compile-only coverage for the MegaMoE decode tiles and the T64 exact-r75
// fallback used by the same-context 48->64->48 lifetime gate.
// Keep the Stage-2 switches here, in the TU used by the established NVCC
// compile check, so it instantiates the canonical combined direct-dispatch and
// bulk-combine source-to-owner branches rather than only proving the default
// r75 specialization parses.

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

#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh>

using namespace deep_gemm;

namespace {

template <uint32_t BlockM,
          uint32_t BlockK,
          uint32_t StoreBlockM,
          uint32_t NumStages,
          uint32_t NumEpilogueThreads = 256>
void instantiate_k3_decode_gin() {
    auto* kernel = &sm100_fp8_fp4_mega_moe_impl<
        384,                         // max tokens/rank
        3584, 3072,                  // hidden, intermediate
        896, 0, 16,                  // experts, shared experts, top-k
        BlockM, 128, BlockK,
        StoreBlockM,
        128, 128,                    // UTCCP SF blocks
        47616, 761856,               // ring tokens, SF ring tokens
        NumStages,
        3584,                        // bytes per legacy NVL pull
        128, 128, NumEpilogueThreads, // dispatch, non-epilogue, epilogue threads
        148, 16,                     // B300 SMs, ranks
        10.0f,                       // matched runtime activation clamp
        false, true>;                // fast math, GIN
    (void)kernel;
}

void instantiate_all_k3_decode_gin_tiles() {
    instantiate_k3_decode_gin<16, 256, 8, 5>();
    instantiate_k3_decode_gin<32, 128, 16, 10>();
    // T64 averages 18.28 rows/expert: BM64 with one epilogue warpgroup.
    // Runtime fast-path ineligibility retains the all-context Put barrier,
    // including when the compile-time single-combine-context flag is enabled.
    instantiate_k3_decode_gin<64, 128, 32, 9, 128>();
}

}  // namespace
