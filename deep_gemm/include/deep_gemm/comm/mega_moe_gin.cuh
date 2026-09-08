#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/ptx/ld_st.cuh>

#ifndef DG_MEGAMOE_GIN_DIAGNOSTICS
#define DG_MEGAMOE_GIN_DIAGNOSTICS 0
#endif

#ifdef DG_MEGAMOE_GIN

#include <nccl_device.h>

#if !defined(DG_NCCL_VERSION_CODE) || DG_NCCL_VERSION_CODE != 23007
#error "MegaMoE GIN prototype requires exactly the validated NCCL 2.30.7 Device API GET ABI"
#endif

namespace deep_gemm::comm {

// Diagnostic-only, local (not registered) storage. Columns [0,128) are
// fixed phase slots; [128+2*expert,130+2*expert) are first/last L2
// L2 fragment-store-phase completion samples for that logical SM. These
// observations do not prove NIC-visible readiness. Zero means unobserved.
// Phase map (ranges indexed by peer lane or warp as indicated):
// 0 entry, 1 route grid, 2 prepack, 3 consensus, 4 pre-pull barrier entry,
// 5 pre-pull barrier exit, 6 cached scheduler counts, 7 count publication;
// 8:16 direct issue, 16:24 direct PUTs queued, 24:32 direct local completion;
// 32:40 receiver signal observed, 40:48 source unpack leader done;
// 48:52 dispatch-warp pull done; 52 first epilogue task, 53 epilogue loop exit,
// 54 TMEM free, 55 combine grid1, 56:58 world barrier begin/end,
// 58 combine grid2, 59 scatter leader done, 60 scatter grid, 61 reduction begin,
// 62 cleanup begins, 63 cleanup world barrier done;
// 64:72 combine issue, 72:80 combine queued, 80:88 combine local completion;
// 88:96 reduction-loop exit per warp (last TMA STORE ISSUED, not settled);
// 96 first MMA task acquired, 97 first MMA operands ready, 98 MMA loop exit;
// 99 first ready expert selected, 100 all expert payload PUTs queued,
// 101 expert payload flushes complete (same dispatch warp; not late headers);
// 104:112 receiver wait entry, 112:120 scatter done per epilogue warp,
// 120:128 reduction begin per epilogue warp. Other slots are reserved.
// Times use each GPU's %globaltimer; never subtract across GPUs. Clear the
// tensor before a diagnostic replay; writes overwrite the most recent launch.
static constexpr uint32_t kMegaMoeGinDiagnosticColumns = 240;

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
    uint32_t combine_issue_wave = 8;
    uint32_t active_fast_path = 0;
    uint32_t bulk_combine = 0;
    uint32_t direct_dispatch = 0;
    uint64_t* diagnostic_buffer = nullptr;
    uint32_t diagnostic_num_sms = 0;

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

    NCCL_DEVICE_INLINE uint32_t data_context(const uint32_t stripe) const {
        // Context zero is reserved for control and cleanup rendezvous.
        // Stripe paired/direct input publication and combine payloads across
        // the remaining eight contexts.  A context owns one QP per peer, so
        // combine batches for a hot peer can use eight independent QPs instead
        // of one serialized queue.
        return 1u + stripe % static_cast<uint32_t>(dev_comm.lsaSize);
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

NCCL_DEVICE_INLINE void mega_moe_gin_trace(
    const MegaMoeGinTransport& transport, const uint32_t sm,
    const uint32_t column, const bool first_only = false) {
#if DG_MEGAMOE_GIN_DIAGNOSTICS
    uint64_t tick;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tick) :: "memory");
    auto* slot = reinterpret_cast<volatile uint64_t*>(
        transport.diagnostic_buffer +
        sm * kMegaMoeGinDiagnosticColumns + column);
    if (not first_only or *slot == 0)
        *slot = tick;
#endif
}

// One correctness-first chained-SoA owner GET.  Callers coordinate one issuer
// at a time for each (context, peer), then scatter the returned scale words and
// publish L1 readiness only after this function returns.
NCCL_DEVICE_INLINE void mega_moe_gin_issue_get_route(
    const MegaMoeGinTransport& transport,
    const uint32_t peer,
    const uint32_t context_stripe,
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
                static_cast<int>(transport.data_context(context_stripe)),
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
    const uint32_t context_stripe,
    ncclGinRequest_t& request) {
    ncclGin gin{transport.dev_comm,
                static_cast<int>(transport.data_context(context_stripe)),
                NCCL_GIN_RESOURCE_SHARING_GPU};
    gin.wait(request, ncclCoopThread{}, ncclGin_None{},
             cuda::memory_order_acquire);
}

// Publish the active input prefix once per remote peer.  The three SoA ranges
// are one aggregate chain on a peer-private data context, so the queue carries
// three operations and one completion instead of one completion per route.
NCCL_DEVICE_INLINE void mega_moe_gin_publish_inputs(
    const MegaMoeGinTransport& transport,
    const uint32_t peer,
    const uint32_t context_stripe,
    const void* local_window_base,
    const void* local_activation,
    void* remote_activation,
    const uint32_t activation_bytes,
    const void* local_scale,
    void* remote_scale,
    const uint32_t scale_bytes,
    const void* local_weights,
    void* remote_weights,
    const uint32_t weight_bytes) {
    ncclGin gin{transport.dev_comm,
                static_cast<int>(transport.data_context(context_stripe)),
                NCCL_GIN_RESOURCE_SHARING_GPU};
    const auto world = ncclTeamWorld(transport.dev_comm);
    gin.put(world, static_cast<int>(peer), transport.window,
            transport.window_offset(local_window_base, remote_activation),
            transport.window,
            transport.window_offset(local_window_base, local_activation),
            activation_bytes, ncclGin_None{}, ncclGin_None{},
            ncclCoopThread{}, ncclGin_None{}, cuda::thread_scope_device,
            cuda::thread_scope_device, ncclGinOptFlagsAggregateRequests);
    gin.put(world, static_cast<int>(peer), transport.window,
            transport.window_offset(local_window_base, remote_scale),
            transport.window,
            transport.window_offset(local_window_base, local_scale),
            scale_bytes, ncclGin_None{}, ncclGin_None{}, ncclCoopThread{},
            ncclGin_None{}, cuda::thread_scope_device,
            cuda::thread_scope_device, ncclGinOptFlagsAggregateRequests);
    gin.put(world, static_cast<int>(peer), transport.window,
            transport.window_offset(local_window_base, remote_weights),
            transport.window,
            transport.window_offset(local_window_base, local_weights),
            weight_bytes, ncclGin_None{}, ncclGin_None{}, ncclCoopThread{},
            ncclGin_None{}, cuda::thread_scope_device,
            cuda::thread_scope_device, ncclGinOptFlagsDefault);

    ncclGinRequest_t request{};
    gin.flushAsync(world, peer, &request, ncclCoopThread{},
                   ncclGinOptFlagsDefault, ncclGin_None{});
    gin.wait(request, ncclCoopThread{}, ncclGin_None{},
             cuda::memory_order_acquire);
}

// Stage-2 direct source-to-owner publication.  An active owner receives the
// three source SoA prefixes followed by compact expert-major control.  An
// inactive owner still receives a fresh all-zero count vector.  The strong VA
// terminal on that final control PUT proves that it and every preceding PUT on
// this context to this peer have settled before the owner observes readiness.
NCCL_DEVICE_INLINE void mega_moe_gin_publish_direct_dispatch(
    const MegaMoeGinTransport& transport,
    const uint32_t peer,
    const uint32_t context_stripe,
    const void* local_window_base,
    const bool publish_inputs,
    const void* local_activation,
    void* remote_activation,
    const uint32_t activation_bytes,
    const void* local_scale,
    void* remote_scale,
    const uint32_t scale_bytes,
    const void* local_weights,
    void* remote_weights,
    const uint32_t weight_bytes,
    const void* local_control,
    void* remote_control,
    const uint32_t control_bytes,
    void* remote_ready) {
    mega_moe_gin_trace(transport, blockIdx.x, 8u + context_stripe);
    constexpr uint32_t kDirectCountBytes = 56u * sizeof(uint64_t);
    constexpr uint32_t kDirectMaxControlBytes =
        kDirectCountBytes + 48u * 16u * sizeof(uint32_t);
    DG_DEVICE_ASSERT(context_stripe <
                     static_cast<uint32_t>(transport.dev_comm.lsaSize));
    DG_DEVICE_ASSERT(control_bytes >= kDirectCountBytes and
                     control_bytes <= kDirectMaxControlBytes);
    ncclGin gin{transport.dev_comm,
                static_cast<int>(transport.data_context(context_stripe)),
                NCCL_GIN_RESOURCE_SHARING_GPU};
    const auto world = ncclTeamWorld(transport.dev_comm);
    if (publish_inputs) {
        DG_DEVICE_ASSERT(activation_bytes > 0 and scale_bytes > 0 and
                         weight_bytes > 0);
        gin.put(world, static_cast<int>(peer), transport.window,
                transport.window_offset(local_window_base, remote_activation),
                transport.window,
                transport.window_offset(local_window_base, local_activation),
                activation_bytes, ncclGin_None{}, ncclGin_None{},
                ncclCoopThread{}, ncclGin_None{}, cuda::thread_scope_device,
                cuda::thread_scope_device,
                ncclGinOptFlagsAggregateRequests);
        gin.put(world, static_cast<int>(peer), transport.window,
                transport.window_offset(local_window_base, remote_scale),
                transport.window,
                transport.window_offset(local_window_base, local_scale),
                scale_bytes, ncclGin_None{}, ncclGin_None{}, ncclCoopThread{},
                ncclGin_None{}, cuda::thread_scope_device,
                cuda::thread_scope_device,
                ncclGinOptFlagsAggregateRequests);
        gin.put(world, static_cast<int>(peer), transport.window,
                transport.window_offset(local_window_base, remote_weights),
                transport.window,
                transport.window_offset(local_window_base, local_weights),
                weight_bytes, ncclGin_None{}, ncclGin_None{}, ncclCoopThread{},
                ncclGin_None{}, cuda::thread_scope_device,
                cuda::thread_scope_device,
                ncclGinOptFlagsAggregateRequests);
    }

    gin.put(
        world, static_cast<int>(peer), transport.window,
        transport.window_offset(local_window_base, remote_control),
        transport.window,
        transport.window_offset(local_window_base, local_control),
        control_bytes,
        ncclGin_StrongVASignalInc{
            transport.window,
            transport.window_offset(local_window_base, remote_ready)},
        ncclGin_None{}, ncclCoopThread{}, ncclGin_None{},
        cuda::thread_scope_device, cuda::thread_scope_device,
        ncclGinOptFlagsDefault);

    mega_moe_gin_trace(transport, blockIdx.x, 16u + context_stripe);

    ncclGinRequest_t request{};
    gin.flushAsync(world, peer, &request, ncclCoopThread{},
                   ncclGinOptFlagsDefault, ncclGin_None{});
    gin.wait(request, ncclCoopThread{}, ncclGin_None{},
             cuda::memory_order_acquire);
    mega_moe_gin_trace(transport, blockIdx.x, 24u + context_stripe);
}

// Control-first experiment: close the compact count/route PUT with its own
// StrongVA terminal. This deliberately does not flush or wait: the receiver
// can reconstruct exact scheduling counts while the later payload progresses.
// The caller must publish all control writers before entry and preserve the
// source slab until its eventual per-(context, peer) flush has completed.
NCCL_DEVICE_INLINE void mega_moe_gin_publish_direct_control_async(
    const MegaMoeGinTransport& transport,
    const uint32_t peer,
    const uint32_t context_stripe,
    const void* local_window_base,
    const void* local_control,
    void* remote_control,
    const uint32_t control_bytes,
    void* remote_control_ready) {
    constexpr uint32_t kDirectCountBytes = 56u * sizeof(uint64_t);
    constexpr uint32_t kDirectMaxControlBytes =
        kDirectCountBytes + 48u * 16u * sizeof(uint32_t);
    DG_DEVICE_ASSERT(context_stripe <
                     static_cast<uint32_t>(transport.dev_comm.lsaSize));
    DG_DEVICE_ASSERT(control_bytes >= kDirectCountBytes and
                     control_bytes <= kDirectMaxControlBytes);
    DG_DEVICE_ASSERT(transport.window_offset(
                         local_window_base, remote_control_ready) %
                         sizeof(uint64_t) == 0);
    ncclGin gin{transport.dev_comm,
                static_cast<int>(transport.data_context(context_stripe)),
                NCCL_GIN_RESOURCE_SHARING_GPU};
    gin.put(
        ncclTeamWorld(transport.dev_comm), static_cast<int>(peer),
        transport.window,
        transport.window_offset(local_window_base, remote_control),
        transport.window,
        transport.window_offset(local_window_base, local_control),
        control_bytes,
        ncclGin_StrongVASignalInc{
            transport.window,
            transport.window_offset(local_window_base, remote_control_ready)},
        ncclGin_None{}, ncclCoopThread{}, ncclGin_None{},
        cuda::thread_scope_device, cuda::thread_scope_device,
        ncclGinOptFlagsDefault);
}

// One full-source SoA payload, not one request/completion per assignment.
// Use the same (context, peer) issuer as control_async, after its control PUT.
// The final weight PUT closes this aggregate chain with a DISTINCT StrongVA
// terminal. An inactive pair still signals once, keeping payload generations
// aligned across empty-route and CUDA Graph replays. No local completion is
// requested here; the caller later flushes/waits once before source reuse.
// Control readiness alone never authorizes activation/SF/weight reads.
NCCL_DEVICE_INLINE void mega_moe_gin_publish_direct_payload_async(
    const MegaMoeGinTransport& transport,
    const uint32_t peer,
    const uint32_t context_stripe,
    const void* local_window_base,
    const bool publish_inputs,
    const void* local_activation,
    void* remote_activation,
    const uint32_t activation_bytes,
    const void* local_scale,
    void* remote_scale,
    const uint32_t scale_bytes,
    const void* local_weights,
    void* remote_weights,
    const uint32_t weight_bytes,
    void* remote_payload_ready) {
    DG_DEVICE_ASSERT(context_stripe <
                     static_cast<uint32_t>(transport.dev_comm.lsaSize));
    DG_DEVICE_ASSERT(transport.window_offset(
                         local_window_base, remote_payload_ready) %
                         sizeof(uint64_t) == 0);
    ncclGin gin{transport.dev_comm,
                static_cast<int>(transport.data_context(context_stripe)),
                NCCL_GIN_RESOURCE_SHARING_GPU};
    const auto world = ncclTeamWorld(transport.dev_comm);
    const ncclGin_StrongVASignalInc ready{
        transport.window,
        transport.window_offset(local_window_base, remote_payload_ready)};
    if (publish_inputs) {
        DG_DEVICE_ASSERT(activation_bytes > 0 and scale_bytes > 0 and
                         weight_bytes > 0);
        gin.put(world, static_cast<int>(peer), transport.window,
                transport.window_offset(local_window_base, remote_activation),
                transport.window,
                transport.window_offset(local_window_base, local_activation),
                activation_bytes, ncclGin_None{}, ncclGin_None{},
                ncclCoopThread{}, ncclGin_None{}, cuda::thread_scope_device,
                cuda::thread_scope_device, ncclGinOptFlagsAggregateRequests);
        gin.put(world, static_cast<int>(peer), transport.window,
                transport.window_offset(local_window_base, remote_scale),
                transport.window,
                transport.window_offset(local_window_base, local_scale),
                scale_bytes, ncclGin_None{}, ncclGin_None{}, ncclCoopThread{},
                ncclGin_None{}, cuda::thread_scope_device,
                cuda::thread_scope_device, ncclGinOptFlagsAggregateRequests);
        gin.put(world, static_cast<int>(peer), transport.window,
                transport.window_offset(local_window_base, remote_weights),
                transport.window,
                transport.window_offset(local_window_base, local_weights),
                weight_bytes, ready, ncclGin_None{}, ncclCoopThread{},
                ncclGin_None{}, cuda::thread_scope_device,
                cuda::thread_scope_device, ncclGinOptFlagsDefault);
    } else {
        // NCCL 2.30.7 gin.h exposes signal(RemoteAction, Coop, Descriptor,
        // givenRelease, requiredRelease, optFlags); no dummy/zero-byte PUT.
        gin.signal(world, static_cast<int>(peer), ready, ncclCoopThread{},
                   ncclGin_None{}, cuda::thread_scope_device,
                   cuda::thread_scope_device, ncclGinOptFlagsDefault);
    }
}

// Keep the default publication (including its completion and diagnostics)
// untouched. Only the opt-in specialization splits control and payload.
// The two remote words are cumulative, never cleared per invocation: control
// advances on every direct launch, payload only on control-first launches.
// Callers must therefore track separate expected epochs when modes alternate.
template <bool kControlFirst>
NCCL_DEVICE_INLINE void mega_moe_gin_publish_direct_dispatch_ordered(
    const MegaMoeGinTransport& transport,
    const uint32_t peer,
    const uint32_t context_stripe,
    const void* local_window_base,
    const bool publish_inputs,
    const void* local_activation,
    void* remote_activation,
    const uint32_t activation_bytes,
    const void* local_scale,
    void* remote_scale,
    const uint32_t scale_bytes,
    const void* local_weights,
    void* remote_weights,
    const uint32_t weight_bytes,
    const void* local_control,
    void* remote_control,
    const uint32_t control_bytes,
    void* remote_control_ready,
    void* remote_payload_ready) {
    if constexpr (kControlFirst) {
        DG_DEVICE_ASSERT(remote_control_ready != remote_payload_ready);
        mega_moe_gin_trace(transport, blockIdx.x, 8u + context_stripe);
        mega_moe_gin_publish_direct_control_async(
            transport, peer, context_stripe, local_window_base,
            local_control, remote_control, control_bytes,
            remote_control_ready);
        mega_moe_gin_publish_direct_payload_async(
            transport, peer, context_stripe, local_window_base, publish_inputs,
            local_activation, remote_activation, activation_bytes,
            local_scale, remote_scale, scale_bytes,
            local_weights, remote_weights, weight_bytes,
            remote_payload_ready);
        mega_moe_gin_trace(transport, blockIdx.x, 16u + context_stripe);
    } else {
        mega_moe_gin_publish_direct_dispatch(
            transport, peer, context_stripe, local_window_base, publish_inputs,
            local_activation, remote_activation, activation_bytes,
            local_scale, remote_scale, scale_bytes,
            local_weights, remote_weights, weight_bytes,
            local_control, remote_control, control_bytes,
            remote_control_ready);
    }
}

NCCL_DEVICE_INLINE void mega_moe_gin_wait_direct_dispatch(
    const MegaMoeGinTransport& transport,
    const uint32_t context_stripe,
    const void* local_window_base,
    const void* local_ready,
    const uint64_t expected_epoch) {
    ncclGin gin{transport.dev_comm,
                static_cast<int>(transport.data_context(context_stripe)),
                NCCL_GIN_RESOURCE_SHARING_GPU};
    gin.waitSignal(
        ncclCoopThread{}, transport.window,
        transport.window_offset(local_window_base, local_ready),
        expected_epoch, 64, cuda::memory_order_acquire);
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

NCCL_DEVICE_INLINE void mega_moe_gin_put_data(
    const MegaMoeGinTransport& transport,
    const uint32_t peer,
    const uint32_t context_stripe,
    const void* local_window_base,
    const void* local_source,
    void* remote_destination,
    const uint32_t bytes,
    const uint32_t opt_flags = ncclGinOptFlagsDefault) {
    ncclGin gin{transport.dev_comm,
                static_cast<int>(transport.data_context(context_stripe)),
                NCCL_GIN_RESOURCE_SHARING_GPU};
    gin.put(ncclTeamWorld(transport.dev_comm), static_cast<int>(peer),
            transport.window,
            transport.window_offset(local_window_base, remote_destination),
            transport.window,
            transport.window_offset(local_window_base, local_source), bytes,
            ncclGin_None{}, ncclGin_None{}, ncclCoopThread{}, ncclGin_None{},
            cuda::thread_scope_device, cuda::thread_scope_device, opt_flags);
}

// These scoped publication helpers deliberately do not reuse the legacy PTX
// wrappers: every participating producer needs compiler ordering around the
// complete CTA epilogue barrier, then a device-release RMW per N fragment.
// Each issuing peer lane acquires the completed counter before constructing
// any span PUT. Counter storage must not be reset until all consumers finish.
NCCL_DEVICE_INLINE void mega_moe_gin_combine_producer_barrier(
    const uint32_t num_threads, const uint32_t barrier_idx) {
    asm volatile("bar.sync %0, %1;"
                 :: "r"(barrier_idx), "r"(num_threads) : "memory");
}

NCCL_DEVICE_INLINE void mega_moe_gin_combine_ready_release(uint32_t* ptr) {
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;"
                 :: "l"(ptr), "r"(1u) : "memory");
}

NCCL_DEVICE_INLINE uint32_t mega_moe_gin_combine_ready_acquire(
    const uint32_t* ptr) {
    uint32_t value;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];"
                 : "=r"(value) : "l"(ptr) : "memory");
    return value;
}

// Publish only complete fixed-stride records (including destination headers).
// The caller owns all readiness acquires, exactly-once span selection and the
// later flush. Default rings the doorbell and retains automatic queue-credit
// checking; an open aggregate chain across dynamically discovered spans is
// expressly forbidden. NCCL2.30.7/GDAKI adds the required system release after
// the issuer's device acquire, cumulatively publishing producer payload stores.
NCCL_DEVICE_INLINE void mega_moe_gin_put_bulk_combine_span(
    const MegaMoeGinTransport& transport,
    const uint32_t peer,
    const uint32_t context_stripe,
    const void* local_window_base,
    const void* local_span,
    void* remote_span,
    const uint32_t span_bytes,
    const uint32_t diagnostic_peer_lane) {
    DG_DEVICE_ASSERT(span_bytes > 0u and span_bytes % 16u == 0u);
    mega_moe_gin_trace(
        transport, blockIdx.x, 64u + diagnostic_peer_lane, true);
    asm volatile("" ::: "memory");
    ncclGin gin{transport.dev_comm,
                static_cast<int>(transport.data_context(context_stripe)),
                NCCL_GIN_RESOURCE_SHARING_GPU};
    gin.put(ncclTeamWorld(transport.dev_comm), static_cast<int>(peer),
            transport.window,
            transport.window_offset(local_window_base, remote_span),
            transport.window,
            transport.window_offset(local_window_base, local_span), span_bytes,
            ncclGin_None{}, ncclGin_None{}, ncclCoopThread{}, ncclGin_None{},
            cuda::thread_scope_device, cuda::thread_scope_system,
            ncclGinOptFlagsDefault);
    asm volatile("" ::: "memory");
    mega_moe_gin_trace(transport, blockIdx.x, 72u + diagnostic_peer_lane);
}

// Publish one BF16 combine row as a chained sequence of equally sized PUTs.
// The final PUT closes the aggregate chain.  The caller must flush/wait the
// destination peer before allowing the source outbox slot to be reused.
NCCL_DEVICE_INLINE void mega_moe_gin_put_row_chunks(
    const MegaMoeGinTransport& transport,
    const uint32_t peer,
    const uint32_t context_stripe,
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
        mega_moe_gin_put_data(
            transport, peer, context_stripe, local_window_base,
            static_cast<const uint8_t*>(local_source) + offset,
            static_cast<uint8_t*>(remote_destination) + offset,
            chunk_bytes,
            is_last ? ncclGinOptFlagsDefault :
                      ncclGinOptFlagsAggregateRequests);
    }
}

NCCL_DEVICE_INLINE void mega_moe_gin_flush_data_peer_async(
    const MegaMoeGinTransport& transport,
    const uint32_t peer,
    const uint32_t context_stripe,
    ncclGinRequest_t* request) {
    ncclGin gin{transport.dev_comm,
                static_cast<int>(transport.data_context(context_stripe)),
                NCCL_GIN_RESOURCE_SHARING_GPU};
    gin.flushAsync(ncclTeamWorld(transport.dev_comm), peer, request,
                   ncclCoopThread{}, ncclGinOptFlagsDefault, ncclGin_None{});
}

NCCL_DEVICE_INLINE void mega_moe_gin_wait_data_peer(
    const MegaMoeGinTransport& transport,
    const uint32_t context_stripe,
    ncclGinRequest_t& request) {
    ncclGin gin{transport.dev_comm,
                static_cast<int>(transport.data_context(context_stripe)),
                NCCL_GIN_RESOURCE_SHARING_GPU};
    gin.wait(request, ncclCoopThread{}, ncclGin_None{},
             cuda::memory_order_acquire);
}

// Stage-1 bulk combine emits exactly one packet PUT for an active
// owner/source pair, followed by one local completion.  Remote visibility is
// left to the matching Put-fenced world barrier. Trace slots remain indexed
// by peer lane independently of the context stripe selected for transport.
NCCL_DEVICE_INLINE void mega_moe_gin_put_bulk_combine_packet(
    const MegaMoeGinTransport& transport,
    const uint32_t peer,
    const uint32_t context_stripe,
    const void* local_window_base,
    const void* local_packet,
    void* remote_packet,
    const uint32_t packet_bytes,
    const uint32_t diagnostic_peer_lane) {
    DG_DEVICE_ASSERT(packet_bytes >= 16u and packet_bytes % 16u == 0);
    mega_moe_gin_trace(transport, blockIdx.x, 64u + diagnostic_peer_lane);
    mega_moe_gin_put_data(
        transport, peer, context_stripe, local_window_base, local_packet,
        remote_packet, packet_bytes);
    mega_moe_gin_trace(transport, blockIdx.x, 72u + diagnostic_peer_lane);
    ncclGinRequest_t request{};
    mega_moe_gin_flush_data_peer_async(
        transport, peer, context_stripe, &request);
    mega_moe_gin_wait_data_peer(transport, context_stripe, request);
    mega_moe_gin_trace(transport, blockIdx.x, 80u + diagnostic_peer_lane);
}

// Early-record mode leaves packet-count publication at the original late
// epilogue point. Send exactly the existing 16-byte packet header, never the
// payload again, with the original device/device and flush/wait semantics.
// The existing world Put fence still establishes remote visibility.
NCCL_DEVICE_INLINE void mega_moe_gin_put_bulk_combine_header(
    const MegaMoeGinTransport& transport,
    const uint32_t peer,
    const uint32_t context_stripe,
    const void* local_window_base,
    const void* local_packet,
    void* remote_packet,
    const uint32_t diagnostic_peer_lane) {
    mega_moe_gin_trace(
        transport, blockIdx.x, 64u + diagnostic_peer_lane, true);
    mega_moe_gin_put_data(
        transport, peer, context_stripe, local_window_base, local_packet,
        remote_packet, 16u);
    mega_moe_gin_trace(transport, blockIdx.x, 72u + diagnostic_peer_lane);
    ncclGinRequest_t request{};
    mega_moe_gin_flush_data_peer_async(
        transport, peer, context_stripe, &request);
    mega_moe_gin_wait_data_peer(transport, context_stripe, request);
    mega_moe_gin_trace(transport, blockIdx.x, 80u + diagnostic_peer_lane);
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

NCCL_DEVICE_INLINE void mega_moe_gin_flush_peer_async(
    const MegaMoeGinTransport& transport,
    const uint32_t peer,
    ncclGinRequest_t* request) {
    ncclGin gin{transport.dev_comm, 0, NCCL_GIN_RESOURCE_SHARING_GPU};
    gin.flushAsync(ncclTeamWorld(transport.dev_comm), peer, request,
                   ncclCoopThread{}, ncclGinOptFlagsDefault, ncclGin_None{});
}

NCCL_DEVICE_INLINE void mega_moe_gin_wait_flush_peer(
    const MegaMoeGinTransport& transport,
    ncclGinRequest_t& request) {
    ncclGin gin{transport.dev_comm, 0, NCCL_GIN_RESOURCE_SHARING_GPU};
    gin.wait(request, ncclCoopThread{}, ncclGin_None{},
             cuda::memory_order_acquire);
}

// Exchange a two-bit launch decision with the same-GPU-index rank in the
// other LSA. Bit 0 is route activity and bit 1 is bulk-combine ineligibility.
// An indexed GIN signal did not progress on the target NCCL 2.30.7/B300
// deployment, including when attached to a PUT.  Carry both the generation
// and flags in one naturally aligned inline PUT instead.  The default (not
// aggregate) GDAKI operation rings the QP doorbell, and the peer consumes the
// aligned 64-bit word with a system-acquire load.  The two flag bits are also
// mirrored in the high bits, so a transient partial observation cannot be
// accepted as a complete decision.
//
// The caller alternates two mailbox slots.  With serialized launches on one
// symmetric buffer, two slots are sufficient: a rank cannot issue generation
// N+2 into N's slot until it has consumed its peer's N+1, and that peer cannot
// issue N+1 until it has consumed N.  Concurrent streams sharing a workspace
// are outside the existing MegaMoE workspace contract.
NCCL_DEVICE_INLINE uint32_t mega_moe_gin_exchange_pair_flags(
    const MegaMoeGinTransport& transport,
    const uint32_t peer,
    const void* local_window_base,
    uint64_t* local_mailbox_ptr,
    const uint64_t generation,
    const uint32_t local_flags) {
    constexpr uint64_t kGenerationMask = (1ull << 60) - 1;
    DG_DEVICE_ASSERT(generation > 0 and generation <= kGenerationMask);
    DG_DEVICE_ASSERT(local_flags < 4);
    ncclGin gin{transport.dev_comm, 0, NCCL_GIN_RESOURCE_SHARING_GPU};
    const auto world = ncclTeamWorld(transport.dev_comm);
    const uint64_t encoded =
        (generation << 2) | local_flags |
        (static_cast<uint64_t>(local_flags) << 62);
    gin.putValue<uint64_t>(
        world, static_cast<int>(peer), transport.window,
        transport.window_offset(local_window_base, local_mailbox_ptr),
        encoded, ncclGin_None{}, ncclCoopThread{}, ncclGin_None{},
        cuda::thread_scope_device,
        cuda::thread_scope_device, ncclGinOptFlagsDefault);

    uint64_t received = 0;
    while (true) {
        received = ptx::ld_acq_sys(local_mailbox_ptr);
        const uint64_t received_generation =
            (received >> 2) & kGenerationMask;
        const uint32_t received_flags =
            static_cast<uint32_t>(received & 3ull);
        const uint32_t received_flags_copy =
            static_cast<uint32_t>(received >> 62);
        if (received_generation == generation and
            received_flags == received_flags_copy)
            break;
    }
    const uint32_t received_flags = static_cast<uint32_t>(received & 3ull);
    DG_DEVICE_ASSERT(received_flags < 4);
    return received_flags;
}

NCCL_DEVICE_INLINE bool mega_moe_gin_exchange_pair_activity(
    const MegaMoeGinTransport& transport,
    const uint32_t peer,
    const void* local_window_base,
    uint64_t* local_mailbox_ptr,
    const uint64_t generation,
    const bool local_active) {
    return mega_moe_gin_exchange_pair_flags(
               transport, peer, local_window_base, local_mailbox_ptr,
               generation,
               static_cast<uint32_t>(local_active)) != 0;
}

NCCL_DEVICE_INLINE void mega_moe_gin_flush_peer(
    const MegaMoeGinTransport& transport,
    const uint32_t peer) {
    ncclGinRequest_t request{};
    mega_moe_gin_flush_peer_async(transport, peer, &request);
    mega_moe_gin_wait_flush_peer(transport, request);
}

// A GIN flush only makes local source storage reusable; it does not guarantee
// that a PUT has settled remotely.  The fused publication and combine phases
// therefore use this all-context Put fence before their consumers proceed.
NCCL_DEVICE_INLINE void mega_moe_gin_world_barrier_all_contexts(
    const MegaMoeGinTransport& transport,
    const uint32_t barrier_index,
    const ncclGinFenceLevel fence) {
    ncclGinBarrier(ncclCoopWarp{}, ncclGinAllContexts(transport.dev_comm),
                   ncclTeamTagWorld{}, barrier_index,
                   cuda::memory_order_acq_rel, fence);
}

// Every combine PUT/flush must use data stripe 0 (context 1). Direct-input
// StrongVA terminals have already discharged the other contexts before
// compute. Keep the existing barrier index: context 1 advances in both modes;
// other signal/shadow sequences pause together during this uniform mode.
NCCL_DEVICE_INLINE void mega_moe_gin_world_barrier_single_combine_context(
    const MegaMoeGinTransport& transport,
    const uint32_t barrier_index) {
    ncclGin gin{transport.dev_comm,
                static_cast<int>(transport.data_context(0)),
                NCCL_GIN_RESOURCE_SHARING_GPU};
    ncclGinBarrier(ncclCoopWarp{}, gin, ncclTeamTagWorld{}, barrier_index,
                   cuda::memory_order_acq_rel, ncclGinFenceLevel::Put);
}

// Use a context-zero rendezvous when a preceding target-local fence has
// discharged payload visibility and ranks only need to exchange a milestone,
// as in ingress-ready handoff and final cleanup.  Callers choose the required
// fence level explicitly.
NCCL_DEVICE_INLINE void mega_moe_gin_world_barrier(
    const MegaMoeGinTransport& transport,
    const uint32_t barrier_index,
    const ncclGinFenceLevel fence =
        ncclGinFenceLevel::Put | ncclGinFenceLevel::Get) {
    ncclGin gin{transport.dev_comm, 0, NCCL_GIN_RESOURCE_SHARING_GPU};
    ncclGinBarrier(ncclCoopWarp{}, gin,
                   ncclTeamTagWorld{}, barrier_index,
                   cuda::memory_order_acq_rel, fence);
}

#endif  // NCCL_CHECK_CUDACC

}  // namespace deep_gemm::comm

#endif  // DG_MEGAMOE_GIN
