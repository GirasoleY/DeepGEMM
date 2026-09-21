#pragma once

#include <cstdint>

#ifdef __CUDACC__
#define DG_GIN_HOST_DEVICE __host__ __device__
#else
#define DG_GIN_HOST_DEVICE
#endif

namespace deep_gemm::layout {

// A compact peer index excludes the workspace owner's entire LSA team.
// Sender and receiver indices differ once there are more than two teams.
struct MegaMoeGinTopology {
    uint32_t world_size;
    uint32_t lsa_size;

    DG_GIN_HOST_DEVICE constexpr bool is_same_lsa(
            uint32_t rank, uint32_t peer) const {
        return rank / lsa_size == peer / lsa_size;
    }

    DG_GIN_HOST_DEVICE constexpr uint32_t remote_rank(
            uint32_t rank, uint32_t slot) const {
        const uint32_t base = rank / lsa_size * lsa_size;
        return slot < base ? slot : slot + lsa_size;
    }

    DG_GIN_HOST_DEVICE constexpr uint32_t remote_slot(
            uint32_t rank, uint32_t peer) const {
        const uint32_t base = rank / lsa_size * lsa_size;
        return peer < base ? peer : peer - lsa_size;
    }

    DG_GIN_HOST_DEVICE constexpr uint32_t return_slot(
            uint32_t rank, uint32_t slot) const {
        return remote_slot(remote_rank(rank, slot), rank);
    }
};

} // namespace deep_gemm::layout

#undef DG_GIN_HOST_DEVICE
