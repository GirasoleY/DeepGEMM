#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/layout/mega_moe_gin.cuh>

#ifdef DG_MEGAMOE_GIN

#include <nccl_device.h>

#if !defined(NCCL_VERSION_CODE) || NCCL_VERSION_CODE != 23007
#error "MegaMoE GIN Device API headers must come from NCCL 2.30.7"
#endif

namespace deep_gemm::comm {

// The caller owns every referenced object and keeps it alive until all kernel
// launches and captured graphs using this descriptor have completed.  The
// registered window covers the native MegaMoE allocation and its appended GIN
// workspace at identical offsets on all eight ranks.
struct MegaMoeEp8GinTransport {
    ncclDevComm_t dev_comm{};
    ncclWindow_t window = nullptr;
    const void* window_base = nullptr;
    uint64_t window_bytes = 0;

#if NCCL_CHECK_CUDACC
    NCCL_DEVICE_INLINE bool is_same_lsa_peer(const uint32_t peer) const {
        DG_DEVICE_ASSERT(peer < layout::kMegaMoEEp8GinWorldSize);
        return peer / layout::kMegaMoEEp8GinLsaSize ==
               static_cast<uint32_t>(dev_comm.rank) /
                   layout::kMegaMoEEp8GinLsaSize;
    }

    NCCL_DEVICE_INLINE uint32_t local_lsa_lane() const {
        return static_cast<uint32_t>(dev_comm.rank) %
               layout::kMegaMoEEp8GinLsaSize;
    }

    NCCL_DEVICE_INLINE uint32_t other_lsa_rank(
            const uint32_t peer_lsa_lane) const {
        DG_DEVICE_ASSERT(peer_lsa_lane < layout::kMegaMoEEp8GinLsaSize);
        const uint32_t remote_lsa =
            1u - static_cast<uint32_t>(dev_comm.rank) /
                     layout::kMegaMoEEp8GinLsaSize;
        return remote_lsa * layout::kMegaMoEEp8GinLsaSize + peer_lsa_lane;
    }

    NCCL_DEVICE_INLINE uint32_t data_context(
            const uint32_t peer_lsa_lane) const {
        DG_DEVICE_ASSERT(peer_lsa_lane < layout::kMegaMoEEp8GinLsaSize);
        // Context zero is reserved for collectives and cleanup rendezvous;
        // contexts one through four carry dispatch traffic.
        return 1u + peer_lsa_lane;
    }

    NCCL_DEVICE_INLINE uint32_t combine_context() const {
        // Keep the full combine packet independent of every dispatch QP.
        return layout::kMegaMoEEp8GinLsaSize + 1u;
    }

    NCCL_DEVICE_INLINE size_t window_offset(
            const void* ptr, const uint64_t bytes) const {
        const auto base = reinterpret_cast<uintptr_t>(window_base);
        const auto address = reinterpret_cast<uintptr_t>(ptr);
        DG_DEVICE_ASSERT(address >= base);
        const auto offset = static_cast<uint64_t>(address - base);
        DG_DEVICE_ASSERT(offset <= window_bytes);
        DG_DEVICE_ASSERT(bytes <= window_bytes - offset);
        return static_cast<size_t>(offset);
    }
#endif
};

static_assert(std::is_standard_layout_v<MegaMoeEp8GinTransport>);
static_assert(std::is_trivially_copyable_v<MegaMoeEp8GinTransport>);

#if NCCL_CHECK_CUDACC

// Publish compact expert counts and assignments first.  The StrongVA control
// terminal authorizes scheduling metadata reconstruction, but never payload
// reads.  This function intentionally neither flushes nor waits; the caller
// must preserve the send packet until flush_wait_dispatch_peer returns.
NCCL_DEVICE_INLINE void mega_moe_ep8_gin_publish_dispatch_control_async(
    const MegaMoeEp8GinTransport& transport,
    const uint32_t remote_owner_lane,
    const uint32_t num_expert_assignments,
    const void* local_dispatch_control,
    void* remote_dispatch_control,
    void* remote_control_epoch) {
    DG_DEVICE_ASSERT(
        num_expert_assignments <=
        layout::kMegaMoEEp8GinMaxExpertAssignments);
    const uint32_t remote_owner_rank =
        transport.other_lsa_rank(remote_owner_lane);
    const uint32_t control_bytes =
        layout::kMegaMoEEp8GinDispatchExpertCountBytes +
        num_expert_assignments * sizeof(uint32_t);

    ncclGin gin{
        transport.dev_comm,
        static_cast<int>(transport.data_context(remote_owner_lane)),
        NCCL_GIN_RESOURCE_SHARING_GPU};
    gin.put(
        ncclTeamWorld(transport.dev_comm),
        static_cast<int>(remote_owner_rank), transport.window,
        transport.window_offset(remote_dispatch_control, control_bytes),
        transport.window,
        transport.window_offset(local_dispatch_control, control_bytes),
        control_bytes,
        ncclGin_StrongVASignalInc{
            transport.window,
            transport.window_offset(remote_control_epoch, sizeof(uint64_t))},
        ncclGin_None{}, ncclCoopThread{}, ncclGin_None{},
        cuda::thread_scope_device, cuda::thread_scope_device,
        ncclGinOptFlagsDefault);
}

// Queue one full-source chained-SoA payload after control on the same
// (context, peer).  A distinct StrongVA terminal authorizes mirror reads.  An
// empty owner still receives one signal so all cumulative generations remain
// aligned across eager launches and graph replays.
NCCL_DEVICE_INLINE void mega_moe_ep8_gin_publish_dispatch_payload_async(
    const MegaMoeEp8GinTransport& transport,
    const uint32_t remote_owner_lane,
    const uint32_t num_tokens,
    const uint32_t num_expert_assignments,
    const void* local_input_activations,
    void* remote_input_activations,
    const void* local_input_scales,
    void* remote_input_scales,
    const void* local_topk_weights,
    void* remote_topk_weights,
    void* remote_payload_epoch) {
    DG_DEVICE_ASSERT(num_tokens <= layout::kMegaMoEEp8GinMaxTokens);
    DG_DEVICE_ASSERT(
        num_expert_assignments <=
        layout::kMegaMoEEp8GinMaxExpertAssignments);
    const uint32_t remote_owner_rank =
        transport.other_lsa_rank(remote_owner_lane);

    const uint32_t activation_bytes =
        num_tokens * layout::kMegaMoEEp8GinInputActivationBytes;
    const uint32_t scale_bytes =
        num_tokens * layout::kMegaMoEEp8GinInputScaleBytes;
    const uint32_t weight_bytes =
        num_tokens * layout::kMegaMoEEp8GinTopKWeightBytes;
    ncclGin gin{
        transport.dev_comm,
        static_cast<int>(transport.data_context(remote_owner_lane)),
        NCCL_GIN_RESOURCE_SHARING_GPU};
    const auto world = ncclTeamWorld(transport.dev_comm);
    const ncclGin_StrongVASignalInc payload_ready{
        transport.window,
        transport.window_offset(remote_payload_epoch, sizeof(uint64_t))};

    if (num_expert_assignments != 0) {
        DG_DEVICE_ASSERT(num_tokens != 0);
        gin.put(
            world, static_cast<int>(remote_owner_rank), transport.window,
            transport.window_offset(
                remote_input_activations, activation_bytes),
            transport.window,
            transport.window_offset(
                local_input_activations, activation_bytes),
            activation_bytes,
            ncclGin_None{}, ncclGin_None{}, ncclCoopThread{}, ncclGin_None{},
            cuda::thread_scope_device, cuda::thread_scope_device,
            ncclGinOptFlagsAggregateRequests);
        gin.put(
            world, static_cast<int>(remote_owner_rank), transport.window,
            transport.window_offset(remote_input_scales, scale_bytes),
            transport.window,
            transport.window_offset(local_input_scales, scale_bytes),
            scale_bytes,
            ncclGin_None{}, ncclGin_None{}, ncclCoopThread{}, ncclGin_None{},
            cuda::thread_scope_device, cuda::thread_scope_device,
            ncclGinOptFlagsAggregateRequests);
        gin.put(
            world, static_cast<int>(remote_owner_rank), transport.window,
            transport.window_offset(remote_topk_weights, weight_bytes),
            transport.window,
            transport.window_offset(local_topk_weights, weight_bytes),
            weight_bytes,
            payload_ready, ncclGin_None{}, ncclCoopThread{}, ncclGin_None{},
            cuda::thread_scope_device, cuda::thread_scope_device,
            ncclGinOptFlagsDefault);
    } else {
        gin.signal(
            world, static_cast<int>(remote_owner_rank), payload_ready,
            ncclCoopThread{}, ncclGin_None{}, cuda::thread_scope_device,
            cuda::thread_scope_device, ncclGinOptFlagsDefault);
    }
}

// Retire the control and payload chains together once the dispatch source
// ranges may be reused.  Four peer-owned dispatch warps can call this helper
// concurrently because each uses a distinct (context, peer) queue.
NCCL_DEVICE_INLINE void mega_moe_ep8_gin_flush_wait_dispatch_peer(
    const MegaMoeEp8GinTransport& transport,
    const uint32_t remote_owner_lane) {
    const uint32_t remote_owner_rank =
        transport.other_lsa_rank(remote_owner_lane);
    ncclGin gin{
        transport.dev_comm,
        static_cast<int>(transport.data_context(remote_owner_lane)),
        NCCL_GIN_RESOURCE_SHARING_GPU};
    const auto world = ncclTeamWorld(transport.dev_comm);
    ncclGinRequest_t request{};
    gin.flushAsync(
        world, remote_owner_rank, &request, ncclCoopThread{},
        ncclGinOptFlagsDefault, ncclGin_None{});
    gin.wait(
        request, ncclCoopThread{}, ncclGin_None{},
        cuda::memory_order_acquire);
}

NCCL_DEVICE_INLINE void mega_moe_ep8_gin_wait_dispatch(
    const MegaMoeEp8GinTransport& transport,
    const void* local_ready_epoch,
    const uint64_t expected_epoch) {
    DG_DEVICE_ASSERT(expected_epoch != 0);
    ncclGin gin{
        transport.dev_comm,
        static_cast<int>(transport.data_context(transport.local_lsa_lane())),
        NCCL_GIN_RESOURCE_SHARING_GPU};
    gin.waitSignal(
        ncclCoopThread{}, transport.window,
        transport.window_offset(local_ready_epoch, sizeof(uint64_t)),
        expected_epoch, 64, cuda::memory_order_acquire);
}

// Every L2 CTA joins all of its epilogue writers before one release increment
// for the completed expert/N fragment. The source-peer issuer acquires every
// contributing expert counter before publishing a record span.
NCCL_DEVICE_INLINE void mega_moe_ep8_gin_combine_producer_barrier(
    const uint32_t num_threads,
    const uint32_t barrier_idx) {
    asm volatile("bar.sync %0, %1;"
                 :: "r"(barrier_idx), "r"(num_threads) : "memory");
}

NCCL_DEVICE_INLINE void mega_moe_ep8_gin_combine_completion_release(
    uint32_t* completion) {
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;"
                 :: "l"(completion), "r"(1u) : "memory");
}

NCCL_DEVICE_INLINE uint32_t mega_moe_ep8_gin_combine_completion_acquire(
    const uint32_t* completion) {
    uint32_t value;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];"
                 : "=r"(value) : "l"(completion) : "memory");
    return value;
}

// Publish a contiguous set of complete fixed-stride records.
NCCL_DEVICE_INLINE void mega_moe_ep8_gin_publish_combine_span_async(
    const MegaMoeEp8GinTransport& transport,
    const uint32_t remote_source_lane,
    const void* local_span,
    void* remote_span,
    const uint32_t num_records) {
    DG_DEVICE_ASSERT(num_records != 0);
    DG_DEVICE_ASSERT(
        num_records <= layout::kMegaMoEEp8GinMaxExpertAssignments);
    const uint32_t remote_source_rank =
        transport.other_lsa_rank(remote_source_lane);
    const uint32_t span_bytes =
        num_records * layout::kMegaMoEEp8GinCombineRecordBytes;
    ncclGin gin{
        transport.dev_comm,
        static_cast<int>(transport.combine_context()),
        NCCL_GIN_RESOURCE_SHARING_GPU};
    gin.put(
        ncclTeamWorld(transport.dev_comm),
        static_cast<int>(remote_source_rank), transport.window,
        transport.window_offset(remote_span, span_bytes),
        transport.window,
        transport.window_offset(local_span, span_bytes), span_bytes,
        ncclGin_None{}, ncclGin_None{}, ncclCoopThread{}, ncclGin_None{},
        cuda::thread_scope_device,
        cuda::thread_scope_system, ncclGinOptFlagsDefault);
}

// Complete one peer queue only after all four owner waves have been submitted.
// Four source-peer lanes may call this concurrently because their peer QPs are
// independent within the dedicated combine context.
NCCL_DEVICE_INLINE void mega_moe_ep8_gin_flush_wait_combine_peer(
    const MegaMoeEp8GinTransport& transport,
    const uint32_t remote_source_lane) {
    const uint32_t remote_source_rank =
        transport.other_lsa_rank(remote_source_lane);
    ncclGin gin{
        transport.dev_comm,
        static_cast<int>(transport.combine_context()),
        NCCL_GIN_RESOURCE_SHARING_GPU};
    const auto world = ncclTeamWorld(transport.dev_comm);
    ncclGinRequest_t request{};
    gin.flushAsync(
        world, static_cast<int>(remote_source_rank), &request, ncclCoopThread{},
        ncclGinOptFlagsDefault, ncclGin_None{});
    gin.wait(
        request, ncclCoopThread{}, ncclGin_None{},
        cuda::memory_order_acquire);
}

// Fence only the dedicated combine context. Dispatch contexts are retired by
// their own late per-peer completions and must not delay combine visibility.
NCCL_DEVICE_INLINE void mega_moe_ep8_gin_world_put_barrier(
    const MegaMoeEp8GinTransport& transport,
    const uint32_t barrier_index) {
    ncclGin gin{
        transport.dev_comm, static_cast<int>(transport.combine_context()),
        NCCL_GIN_RESOURCE_SHARING_GPU};
    ncclGinBarrier(
        ncclCoopWarp{}, gin, ncclTeamTagWorld{}, barrier_index,
        cuda::memory_order_acq_rel, ncclGinFenceLevel::Put);
}

// Run after every rank has finished consuming dispatch mirrors and combine
// receive packets.  Context zero carries only the collective milestone; the
// earlier StrongVA wait and world PUT barrier supplied data visibility.
NCCL_DEVICE_INLINE void mega_moe_ep8_gin_world_cleanup_barrier(
    const MegaMoeEp8GinTransport& transport,
    const uint32_t barrier_index) {
    ncclGin gin{
        transport.dev_comm, 0, NCCL_GIN_RESOURCE_SHARING_GPU};
    ncclGinBarrier(
        ncclCoopWarp{}, gin, ncclTeamTagWorld{}, barrier_index,
        cuda::memory_order_acq_rel, ncclGinFenceLevel::None);
}

#endif  // NCCL_CHECK_CUDACC

} // namespace deep_gemm::comm

#endif  // DG_MEGAMOE_GIN
