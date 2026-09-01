// Compile-only coverage for the two MegaMoE tile configurations used by the
// initial Kimi K3 decode target (MNS8 with three draft tokens, then MNS10/12).
// This file deliberately does not fabricate a GIN communicator or launch a
// kernel; it forces NVCC to instantiate the complete feature-on device path.

#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh>

using namespace deep_gemm;

namespace {

template <uint32_t BlockM,
          uint32_t BlockK,
          uint32_t StoreBlockM,
          uint32_t NumStages>
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
        128, 128, 256,               // dispatch, non-epilogue, epilogue threads
        148, 16,                     // B300 SMs, ranks
        cute::numeric_limits<float>::infinity(),
        true, true>;                 // fast math, GIN
    (void)kernel;
}

void instantiate_all_k3_decode_gin_tiles() {
    instantiate_k3_decode_gin<16, 256, 8, 5>();
    instantiate_k3_decode_gin<32, 128, 16, 10>();
}

}  // namespace
