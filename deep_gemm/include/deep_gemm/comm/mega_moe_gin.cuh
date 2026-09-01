#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <deep_gemm/common/exception.cuh>

#ifdef DG_MEGAMOE_GIN

#include <nccl_device.h>

#if !defined(DG_NCCL_VERSION_CODE) || DG_NCCL_VERSION_CODE != 23007
#error "MegaMoE GIN prototype requires exactly the validated NCCL 2.30.7 Device API GET ABI"
#endif

namespace deep_gemm::comm {

// Trivially-copyable launch state owned by the host-side MegaMoE GIN context.
// NCCL's device communicator contains device pointers to all persistent GIN
// resources, while the window identifies the registered symmetric MegaMoE
// allocation.  Neither object may be recreated during CUDA Graph capture.
struct MegaMoeGinTransport {
    ncclDevComm_t dev_comm{};
    ncclWindow_t window = nullptr;
    uint32_t enabled = 0;
    uint32_t completion_batch = 1;
    uint32_t combine_chunk_bytes = 7168;
    uint32_t outbox_depth = 8;

#if NCCL_CHECK_CUDACC
    NCCL_DEVICE_INLINE bool is_enabled() const {
        return enabled != 0;
    }

    NCCL_DEVICE_INLINE bool is_same_lsa_peer(const uint32_t peer) const {
        // The target communicator is validated by the host to contain two
        // contiguous eight-rank LSA teams.  Keeping this predicate here avoids
        // materializing mapped pointers for peers outside the local LSA.
        return peer / static_cast<uint32_t>(dev_comm.lsaSize) ==
               static_cast<uint32_t>(dev_comm.rank) /
                   static_cast<uint32_t>(dev_comm.lsaSize);
    }

    NCCL_DEVICE_INLINE uint32_t data_context(const uint32_t peer) const {
        // Context zero is reserved for control/combine and world barriers.
        // The remaining eight contexts are stable by source LSA rank.
        return 1u + peer % static_cast<uint32_t>(dev_comm.lsaSize);
    }

    NCCL_DEVICE_INLINE size_t window_offset(const void* local_base,
                                            const void* ptr) const {
        return static_cast<size_t>(
            static_cast<const uint8_t*>(ptr) -
            static_cast<const uint8_t*>(local_base));
    }
#endif
};

static_assert(std::is_trivially_copyable_v<MegaMoeGinTransport>,
              "MegaMoE GIN launch state must be copied directly into kernel arguments");

#if NCCL_CHECK_CUDACC

// One correctness-first chained-SoA owner GET.  Callers coordinate one issuer
// at a time for each (context, peer), then scatter the returned scale words and
// publish L1 readiness only after this function returns.
NCCL_DEVICE_INLINE void mega_moe_gin_issue_get_route(
    const MegaMoeGinTransport& transport,
    const uint32_t peer,
    const void* local_window_base,
    const void* remote_activation,
    void* local_activation,
    const uint32_t activation_bytes,
    const void* remote_scale,
    void* local_scale_scratch,
    const uint32_t scale_bytes,
    const void* remote_weight,
    void* local_weight,
    const uint32_t weight_bytes,
    ncclGinRequest_t* request) {
    ncclGin gin{transport.dev_comm,
                static_cast<int>(transport.data_context(peer)),
                NCCL_GIN_RESOURCE_SHARING_GPU};
    const auto world = ncclTeamWorld(transport.dev_comm);
    gin.get(world, static_cast<int>(peer), transport.window,
            transport.window_offset(local_window_base, remote_activation),
            transport.window,
            transport.window_offset(local_window_base, local_activation),
            activation_bytes, ncclCoopThread{}, ncclGin_None{},
            ncclGinOptFlagsAggregateRequests);
    gin.get(world, static_cast<int>(peer), transport.window,
            transport.window_offset(local_window_base, remote_scale),
            transport.window,
            transport.window_offset(local_window_base, local_scale_scratch),
            scale_bytes, ncclCoopThread{}, ncclGin_None{},
            ncclGinOptFlagsAggregateRequests);
    gin.get(world, static_cast<int>(peer), transport.window,
            transport.window_offset(local_window_base, remote_weight),
            transport.window,
            transport.window_offset(local_window_base, local_weight),
            weight_bytes, ncclCoopThread{}, ncclGin_None{},
            ncclGinOptFlagsDefault);
    gin.flushAsync(world, peer, request, ncclCoopThread{},
                   ncclGinOptFlagsDefault, ncclGin_None{});
}

NCCL_DEVICE_INLINE void mega_moe_gin_wait_get_route(
    const MegaMoeGinTransport& transport,
    const uint32_t peer,
    ncclGinRequest_t& request) {
    ncclGin gin{transport.dev_comm,
                static_cast<int>(transport.data_context(peer)),
                NCCL_GIN_RESOURCE_SHARING_GPU};
    gin.wait(request, ncclCoopThread{}, ncclGin_None{},
             cuda::memory_order_acquire);
}

NCCL_DEVICE_INLINE void mega_moe_gin_put(
    const MegaMoeGinTransport& transport,
    const uint32_t peer,
    const void* local_window_base,
    const void* local_source,
    void* remote_destination,
    const uint32_t bytes,
    const uint32_t opt_flags = ncclGinOptFlagsDefault) {
    ncclGin gin{transport.dev_comm, 0, NCCL_GIN_RESOURCE_SHARING_GPU};
    gin.put(ncclTeamWorld(transport.dev_comm), static_cast<int>(peer),
            transport.window,
            transport.window_offset(local_window_base, remote_destination),
            transport.window,
            transport.window_offset(local_window_base, local_source), bytes,
            ncclGin_None{}, ncclGin_None{}, ncclCoopThread{}, ncclGin_None{},
            cuda::thread_scope_device, cuda::thread_scope_device, opt_flags);
}

// Publish one BF16 combine row as a chained sequence of equally sized PUTs.
// The final PUT closes the aggregate chain.  The caller must flush/wait the
// destination peer before allowing the source outbox slot to be reused.
NCCL_DEVICE_INLINE void mega_moe_gin_put_row_chunks(
    const MegaMoeGinTransport& transport,
    const uint32_t peer,
    const void* local_window_base,
    const void* local_source,
    void* remote_destination,
    const uint32_t row_bytes) {
    const uint32_t chunk_bytes = transport.combine_chunk_bytes;
    DG_DEVICE_ASSERT(
        (chunk_bytes == 256u or chunk_bytes == 1792u or
         chunk_bytes == 3584u or chunk_bytes == 7168u) and
        row_bytes % chunk_bytes == 0);

    for (uint32_t offset = 0; offset < row_bytes; offset += chunk_bytes) {
        const bool is_last = offset + chunk_bytes == row_bytes;
        mega_moe_gin_put(
            transport, peer, local_window_base,
            static_cast<const uint8_t*>(local_source) + offset,
            static_cast<uint8_t*>(remote_destination) + offset,
            chunk_bytes,
            is_last ? ncclGinOptFlagsDefault :
                      ncclGinOptFlagsAggregateRequests);
    }
}

NCCL_DEVICE_INLINE void mega_moe_gin_put_value(
    const MegaMoeGinTransport& transport,
    const uint32_t peer,
    const void* local_window_base,
    void* remote_destination,
    const uint64_t value,
    const uint32_t bytes,
    const uint32_t opt_flags = ncclGinOptFlagsDefault) {
    ncclGin gin{transport.dev_comm, 0, NCCL_GIN_RESOURCE_SHARING_GPU};
    const auto offset =
        transport.window_offset(local_window_base, remote_destination);
    if (bytes == sizeof(uint32_t)) {
        gin.putValue(ncclTeamWorld(transport.dev_comm), static_cast<int>(peer),
                     transport.window, offset,
                     static_cast<uint32_t>(value), ncclGin_None{},
                     ncclCoopThread{}, ncclGin_None{},
                     cuda::thread_scope_device, cuda::thread_scope_device,
                     opt_flags);
    } else {
        gin.putValue(ncclTeamWorld(transport.dev_comm), static_cast<int>(peer),
                     transport.window, offset, value, ncclGin_None{},
                     ncclCoopThread{}, ncclGin_None{},
                     cuda::thread_scope_device, cuda::thread_scope_device,
                     opt_flags);
    }
}

NCCL_DEVICE_INLINE void mega_moe_gin_flush_peer(
    const MegaMoeGinTransport& transport,
    const uint32_t peer) {
    ncclGin gin{transport.dev_comm, 0, NCCL_GIN_RESOURCE_SHARING_GPU};
    ncclGinRequest_t request{};
    gin.flushAsync(ncclTeamWorld(transport.dev_comm), peer, &request,
                   ncclCoopThread{}, ncclGinOptFlagsDefault, ncclGin_None{});
    gin.wait(request, ncclCoopThread{}, ncclGin_None{},
             cuda::memory_order_acquire);
}

// A world GIN barrier fences every data context.  It is used only at the
// existing three logical MegaMoE phase boundaries; same-LSA payload movement
// remains on the original mapped-pointer/TMA path.
NCCL_DEVICE_INLINE void mega_moe_gin_world_barrier(
    const MegaMoeGinTransport& transport,
    const uint32_t barrier_index,
    const ncclGinFenceLevel fence =
        ncclGinFenceLevel::Put | ncclGinFenceLevel::Get) {
    // The all-context/world barrier has more work items than a single thread
    // can safely drive: a lone worker may signal one ring neighbor and wait
    // before reaching the remaining peers.  A converged warp cooperatively
    // stripes every (context, peer) pair and avoids that dependency cycle.
    ncclGinBarrier(ncclCoopWarp{}, ncclGinAllContexts(transport.dev_comm),
                   ncclTeamTagWorld{}, barrier_index,
                   cuda::memory_order_acq_rel, fence);
}

#endif  // NCCL_CHECK_CUDACC

}  // namespace deep_gemm::comm

#endif  // DG_MEGAMOE_GIN
