#pragma once

#include <cstdint>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#ifdef DG_MEGAMOE_GIN
#include <deep_gemm/comm/mega_moe_gin.cuh>
#endif
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>

// Every phase slot has one writer per logical SM. Disabled specializations
// compile out both timestamp reads and predicates. These observations are not
// readiness signals and introduce no fence, barrier, or atomic operation.
#if defined(DG_MEGAMOE_GIN) && DG_MEGAMOE_GIN_DIAGNOSTICS
#define DG_GIN_TRACE_IF(pred, col) do { if constexpr (kUseGin) { \
    if (pred) comm::mega_moe_gin_trace(gin_transport, sm_idx, col); } } while (0)
#define DG_GIN_TRACE_FIRST_IF(pred, col) do { if constexpr (kUseGin) { \
    if (pred) comm::mega_moe_gin_trace(gin_transport, sm_idx, col, true); } } while (0)
#else
#define DG_GIN_TRACE_IF(pred, col) ((void)0)
#define DG_GIN_TRACE_FIRST_IF(pred, col) ((void)0)
#endif

#ifndef DG_MEGAMOE_GIN_LOCAL_ABLATION_STAGE
#define DG_MEGAMOE_GIN_LOCAL_ABLATION_STAGE 0
#endif

#ifndef DG_MEGAMOE_GIN_ACTIVE_FAST_PATH
#define DG_MEGAMOE_GIN_ACTIVE_FAST_PATH 0
#endif

#ifndef DG_MEGAMOE_GIN_BULK_COMBINE
#define DG_MEGAMOE_GIN_BULK_COMBINE 0
#endif

#ifndef DG_MEGAMOE_GIN_DIRECT_DISPATCH
#define DG_MEGAMOE_GIN_DIRECT_DISPATCH 0
#endif

#ifndef DG_MEGAMOE_GIN_ACTIVITY_GATE_OPT
#define DG_MEGAMOE_GIN_ACTIVITY_GATE_OPT 0
#endif

#ifndef DG_MEGAMOE_GIN_DISPATCH_WARP_SCAN
#define DG_MEGAMOE_GIN_DISPATCH_WARP_SCAN 0
#endif

#ifndef DG_MEGAMOE_GIN_COOP_DIRECT_PACK
#define DG_MEGAMOE_GIN_COOP_DIRECT_PACK 0
#endif

#ifndef DG_MEGAMOE_GIN_PRECONSENSUS_PACK
#define DG_MEGAMOE_GIN_PRECONSENSUS_PACK 0
#endif

#ifndef DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT
#define DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT 0
#endif

#ifndef DG_MEGAMOE_GIN_DISPATCH_OVERLAP
#define DG_MEGAMOE_GIN_DISPATCH_OVERLAP 0
#endif

#ifndef DG_MEGAMOE_GIN_COMBINE_OVERLAP
#define DG_MEGAMOE_GIN_COMBINE_OVERLAP 0
#endif

#ifndef DG_MEGAMOE_GIN_STRONGVA_COMBINE_TERMINAL
#define DG_MEGAMOE_GIN_STRONGVA_COMBINE_TERMINAL 0
#endif

#ifndef DG_MEGAMOE_GIN_COMBINE_OWNER_WAVES
#define DG_MEGAMOE_GIN_COMBINE_OWNER_WAVES 0
#endif

#ifndef DG_MEGAMOE_GIN_COMBINE_OWNER_SLOT_READY
#define DG_MEGAMOE_GIN_COMBINE_OWNER_SLOT_READY 0
#endif

// Retired experiment switches must not silently select an unsupported path.
#if defined(DG_MEGAMOE_GIN_COMBINE_EXPERTS_PER_WAVE) && DG_MEGAMOE_GIN_COMBINE_EXPERTS_PER_WAVE != 0
#error "Expert-wave combine is not part of the clean single-context candidate"
#endif
#if defined(DG_MEGAMOE_GIN_COMBINE_BARRIER_WARPS) && DG_MEGAMOE_GIN_COMBINE_BARRIER_WARPS != 1
#error "Cooperative combine barriers are not part of the clean single-context candidate"
#endif

namespace deep_gemm {

static constexpr bool kMegaMoeGinSingleCombineContext =
    DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT != 0;
static_assert(DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT == 0 or
              DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT == 1,
              "Invalid MegaMoE GIN single-combine-context flag");

static constexpr uint32_t kMegaMoeGinLocalAblationStage =
    DG_MEGAMOE_GIN_LOCAL_ABLATION_STAGE;
static_assert(kMegaMoeGinLocalAblationStage <= 4,
              "Invalid MegaMoE GIN local ablation stage");
static constexpr bool kMegaMoeGinActiveFastPath =
    DG_MEGAMOE_GIN_ACTIVE_FAST_PATH != 0;
static_assert(DG_MEGAMOE_GIN_ACTIVE_FAST_PATH == 0 or
              DG_MEGAMOE_GIN_ACTIVE_FAST_PATH == 1,
              "Invalid MegaMoE GIN active-fast-path value");
static_assert(not kMegaMoeGinActiveFastPath or
              kMegaMoeGinLocalAblationStage == 0,
              "GIN active fast path cannot be combined with local ablation");
static constexpr bool kMegaMoeGinBulkCombine =
    DG_MEGAMOE_GIN_BULK_COMBINE != 0;
static_assert(DG_MEGAMOE_GIN_BULK_COMBINE == 0 or
              DG_MEGAMOE_GIN_BULK_COMBINE == 1,
              "Invalid MegaMoE GIN bulk-combine value");
static_assert(not kMegaMoeGinBulkCombine or
              kMegaMoeGinLocalAblationStage == 0,
              "GIN bulk combine cannot be combined with local ablation");
static_assert(not kMegaMoeGinBulkCombine or kMegaMoeGinActiveFastPath,
              "GIN bulk combine requires the global activity consensus");
static constexpr bool kMegaMoeGinDirectDispatch =
    DG_MEGAMOE_GIN_DIRECT_DISPATCH != 0;
static_assert(not kMegaMoeGinSingleCombineContext or
              (kMegaMoeGinBulkCombine and kMegaMoeGinDirectDispatch),
              "Single combine context requires direct dispatch and bulk combine");
static_assert(DG_MEGAMOE_GIN_DIRECT_DISPATCH == 0 or
              DG_MEGAMOE_GIN_DIRECT_DISPATCH == 1,
              "Invalid MegaMoE GIN direct-dispatch value");
static_assert(not kMegaMoeGinDirectDispatch or
              kMegaMoeGinLocalAblationStage == 0,
              "GIN direct dispatch cannot be combined with local ablation");
static_assert(not kMegaMoeGinDirectDispatch or
              kMegaMoeGinActiveFastPath,
              "GIN direct dispatch requires the global activity consensus");
static constexpr bool kMegaMoeGinActivityGateOpt =
    DG_MEGAMOE_GIN_ACTIVITY_GATE_OPT != 0;
static_assert(DG_MEGAMOE_GIN_ACTIVITY_GATE_OPT == 0 or
              DG_MEGAMOE_GIN_ACTIVITY_GATE_OPT == 1,
              "Invalid MegaMoE GIN activity-gate optimization value");
static_assert(not kMegaMoeGinActivityGateOpt or
              kMegaMoeGinActiveFastPath,
              "GIN activity-gate optimization requires activity consensus");
static constexpr bool kMegaMoeGinDispatchWarpScan =
    DG_MEGAMOE_GIN_DISPATCH_WARP_SCAN != 0;
static_assert(DG_MEGAMOE_GIN_DISPATCH_WARP_SCAN == 0 or
              DG_MEGAMOE_GIN_DISPATCH_WARP_SCAN == 1,
              "Invalid MegaMoE GIN dispatch warp-scan value");
static_assert(not kMegaMoeGinDispatchWarpScan or
              kMegaMoeGinDirectDispatch,
              "GIN dispatch warp scan requires direct dispatch");
static constexpr bool kMegaMoeGinCoopDirectPack =
    DG_MEGAMOE_GIN_COOP_DIRECT_PACK != 0;
static_assert(DG_MEGAMOE_GIN_COOP_DIRECT_PACK == 0 or
              DG_MEGAMOE_GIN_COOP_DIRECT_PACK == 1,
              "Invalid MegaMoE GIN cooperative direct-pack value");
static_assert(not kMegaMoeGinCoopDirectPack or
              kMegaMoeGinDirectDispatch,
              "GIN cooperative direct pack requires direct dispatch");
static constexpr bool kMegaMoeGinPreconsensusPack =
    DG_MEGAMOE_GIN_PRECONSENSUS_PACK != 0;
static_assert(DG_MEGAMOE_GIN_PRECONSENSUS_PACK == 0 or
              DG_MEGAMOE_GIN_PRECONSENSUS_PACK == 1,
              "Invalid MegaMoE GIN pre-consensus pack value");
static_assert(not kMegaMoeGinPreconsensusPack or
              (kMegaMoeGinCoopDirectPack and
               kMegaMoeGinActiveFastPath),
              "GIN pre-consensus pack requires cooperative direct pack and "
              "activity consensus");
static constexpr bool kMegaMoeGinDispatchOverlap =
    DG_MEGAMOE_GIN_DISPATCH_OVERLAP != 0;
static_assert(DG_MEGAMOE_GIN_DISPATCH_OVERLAP == 0 or
              DG_MEGAMOE_GIN_DISPATCH_OVERLAP == 1,
              "Invalid MegaMoE GIN dispatch-overlap flag");
static_assert(not kMegaMoeGinDispatchOverlap or
              (kMegaMoeGinDirectDispatch and kMegaMoeGinBulkCombine and
               kMegaMoeGinPreconsensusPack),
              "GIN dispatch overlap requires prepacked direct dispatch and "
              "bulk combine");
static constexpr bool kMegaMoeGinCombineOverlap =
    DG_MEGAMOE_GIN_COMBINE_OVERLAP != 0;
static_assert(DG_MEGAMOE_GIN_COMBINE_OVERLAP == 0 or
              DG_MEGAMOE_GIN_COMBINE_OVERLAP == 1,
              "Invalid MegaMoE GIN combine-overlap flag");
static_assert(not kMegaMoeGinCombineOverlap or
              (kMegaMoeGinDispatchOverlap and
               kMegaMoeGinSingleCombineContext and
               kMegaMoeGinDirectDispatch and kMegaMoeGinBulkCombine),
              "GIN combine overlap requires split direct dispatch and "
              "single-context bulk combine");
static constexpr bool kMegaMoeGinStrongVACombineTerminal =
    DG_MEGAMOE_GIN_STRONGVA_COMBINE_TERMINAL != 0;
static_assert(DG_MEGAMOE_GIN_STRONGVA_COMBINE_TERMINAL == 0 or
              DG_MEGAMOE_GIN_STRONGVA_COMBINE_TERMINAL == 1,
              "Invalid MegaMoE GIN StrongVA combine-terminal flag");
static_assert(not kMegaMoeGinStrongVACombineTerminal or
              kMegaMoeGinCombineOverlap,
              "GIN StrongVA combine terminal requires combine overlap");
static constexpr uint32_t kMegaMoeGinCombineOwnerWaves =
    DG_MEGAMOE_GIN_COMBINE_OWNER_WAVES;
static_assert(kMegaMoeGinCombineOwnerWaves == 0 or
              kMegaMoeGinCombineOwnerWaves == 2 or
              kMegaMoeGinCombineOwnerWaves == 4 or
              kMegaMoeGinCombineOwnerWaves == 8,
              "Invalid MegaMoE GIN combine owner-wave count");
static_assert(kMegaMoeGinCombineOwnerWaves == 0 or
              kMegaMoeGinStrongVACombineTerminal,
              "GIN combine owner waves require the StrongVA terminal protocol");
static constexpr bool kMegaMoeGinCombineOwnerSlotReady =
    DG_MEGAMOE_GIN_COMBINE_OWNER_SLOT_READY != 0;
static_assert(DG_MEGAMOE_GIN_COMBINE_OWNER_SLOT_READY == 0 or
              DG_MEGAMOE_GIN_COMBINE_OWNER_SLOT_READY == 1,
              "Invalid MegaMoE GIN combine owner-slot-ready flag");
static_assert(not kMegaMoeGinCombineOwnerSlotReady or
              kMegaMoeGinCombineOwnerWaves == 4,
              "GIN owner-slot readiness requires the W4 sender");
static_assert(not kMegaMoeGinCombineOwnerSlotReady or
              kMegaMoeGinStrongVACombineTerminal,
              "GIN owner-slot readiness requires StrongVA terminals");

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumSharedExperts,
    uint32_t kNumTopk,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t STORE_BLOCK_M,
    uint32_t SF_BLOCK_M, uint32_t SF_BLOCK_N,
    uint32_t kNumRingTokens,
    uint32_t kNumSFRingTokens,
    uint32_t kNumStages,
    uint32_t kNumBytesPerPull,
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kNumSMs, uint32_t kNumRanks,
    float kActivationClamp,
    bool kFastMath,
    bool kUseGin = false,
    bool kHasShared = (kNumSharedExperts > 0),
    uint32_t L1_SHAPE_N = kIntermediateHidden * 2,
    uint32_t L1_SHAPE_K = kHidden,
    uint32_t L2_SHAPE_N = kHidden,
    uint32_t L2_SHAPE_K = kIntermediateHidden,
    uint32_t SHARED_L2_SHAPE_K = L2_SHAPE_K * kNumSharedExperts,
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32,
    uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32,
    uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32,
    uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4,
    uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads,
    uint32_t kNumTokensPerWarp = 32 / kNumTopk,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks,
    uint32_t kNumRingBlocks = kNumRingTokens / BLOCK_M,
    uint32_t kNumSharedSFTokens = layout::get_num_max_shared_sf_tokens(kNumMaxTokensPerRank),
    typename task_info_t = sched::TaskInfo<kHasShared>
>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm100_fp8_fp4_mega_moe_impl(void* y,
                            int* cumulative_local_expert_recv_stats,
                            const uint32_t num_tokens,
                            const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
#ifdef DG_MEGAMOE_GIN
                            const __grid_constant__ comm::MegaMoeGinTransport gin_transport,
#endif
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_weights_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_output,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_weights_sf) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::TMEM::Allocator2Sm;
    // Actual peers in each of the two GIN LSA teams. The wire/storage layout
    // still reserves eight peer slots and nine registered contexts.
    constexpr uint32_t kGinPeerCount = kNumRanks / 2u;
    constexpr bool kGinOwnerSlotShape =
        kUseGin and kMegaMoeGinCombineOwnerSlotReady and
        kNumRanks == 8 and kNumExperts == 448 and
        kNumExpertsPerRank == 56 and kNumTopk == 16 and
        kHidden == 3584 and kIntermediateHidden == 3072 and
        not kHasShared and kNumEpilogueWarps == 8 and
        kNumSMs > layout::kMegaMoeGinDirectDispatchMaxTokens;

    // Template checks
    DG_STATIC_ASSERT(kNumDispatchThreads % 128 == 0, "Invalid number of dispatch threads");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 128, "Invalid number of MMA non-epilogue threads");
    DG_STATIC_ASSERT(kNumEpilogueThreads % 128 == 0, "Invalid number of MMA epilogue and combine threads");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");
    DG_STATIC_ASSERT(not kMegaMoeGinBulkCombine or kUseGin,
                     "GIN bulk combine requires the GIN specialization");
    DG_STATIC_ASSERT(not kMegaMoeGinDirectDispatch or kUseGin,
                     "GIN direct dispatch requires the GIN specialization");
#ifndef DG_MEGAMOE_GIN
    DG_STATIC_ASSERT(not kUseGin, "GIN kernel instantiated without DG_MEGAMOE_GIN");
#else
    DG_STATIC_ASSERT(kMegaMoeGinLocalAblationStage == 0 or kUseGin,
                     "GIN local ablation requires the GIN specialization");
    if constexpr (kUseGin) {
        DG_STATIC_ASSERT(kNumRanks == 8 or kNumRanks == 16,
                         "GIN requires EP8/LSA4 or EP16/LSA8");
        DG_STATIC_ASSERT(kNumDispatchWarps == layout::kMegaMoeGinNumDispatchWarps,
                         "GIN workspace must match the four dispatch warps");
        DG_STATIC_ASSERT(BLOCK_M <= layout::kMegaMoeGinMaxOutboxBlockM,
                         "GIN combine outbox row capacity is too small");
        DG_DEVICE_ASSERT(gin_transport.is_enabled());
        DG_DEVICE_ASSERT(gin_transport.dev_comm.rank == sym_buffer.rank_idx);
        DG_DEVICE_ASSERT(gin_transport.dev_comm.nRanks == kNumRanks);
        DG_DEVICE_ASSERT(gin_transport.dev_comm.lsaSize == kGinPeerCount);
        DG_DEVICE_ASSERT(
            gin_transport.dev_comm.nRanks == 2 * gin_transport.dev_comm.lsaSize);
        DG_DEVICE_ASSERT(gin_transport.dev_comm.ginContextCount >= 9);
        DG_DEVICE_ASSERT(gin_transport.dev_comm.ginSignalCount >= 2);
        if constexpr (kMegaMoeGinBulkCombine) {
            DG_STATIC_ASSERT(kNumExpertsPerRank == 56 and
                             kNumTopk == 16 and kHidden == 3584 and
                             kIntermediateHidden == 3072 and not kHasShared,
                             "GIN bulk combine requires the MNS8 decode shape");
            DG_DEVICE_ASSERT(gin_transport.bulk_combine != 0);
            DG_DEVICE_ASSERT(gin_transport.outbox_depth == 64);
        }
        if constexpr (kMegaMoeGinDirectDispatch) {
            DG_STATIC_ASSERT(
                (kNumRanks == 8 or kNumRanks == 16) and
                kNumExpertsPerRank ==
                    layout::kMegaMoeGinDirectDispatchExpertsPerRank and
                kNumTopk == 16 and kHidden == 3584 and
                kIntermediateHidden == 3072 and not kHasShared,
                "GIN direct dispatch requires the MNS8/10/12 decode shape");
            DG_STATIC_ASSERT(
                kNumMaxTokensPerRank >=
                    layout::kMegaMoeGinDirectDispatchNumPeers *
                        layout::kMegaMoeGinDirectDispatchMaxTokens,
                "GIN direct dispatch requires eight 48-row input lanes");
            DG_STATIC_ASSERT(
                static_cast<uint64_t>(kNumSMs) *
                        layout::kMegaMoeGinNumDispatchWarps *
                        (kHidden / 32u) >=
                    layout::kMegaMoeGinDirectDispatchStorageBytes,
                "GIN direct-dispatch control alias exceeds scale scratch");
            DG_DEVICE_ASSERT(gin_transport.direct_dispatch != 0);
        }
        DG_DEVICE_ASSERT(
            gin_transport.outbox_depth == 4 or
            gin_transport.outbox_depth == 8 or
            gin_transport.outbox_depth == 16 or
            gin_transport.outbox_depth == 64);
        DG_DEVICE_ASSERT(
            (gin_transport.combine_chunk_bytes == 256 or
             gin_transport.combine_chunk_bytes == 1792 or
             gin_transport.combine_chunk_bytes == 3584 or
             gin_transport.combine_chunk_bytes == 7168) and
            (kHidden * sizeof(nv_bfloat16)) %
                gin_transport.combine_chunk_bytes == 0);
        DG_DEVICE_ASSERT(
            gin_transport.completion_batch == 1 or
            gin_transport.completion_batch == 2 or
            gin_transport.completion_batch == 4 or
            gin_transport.completion_batch == 8);
        if constexpr (kMegaMoeGinLocalAblationStage > 0) {
            DG_STATIC_ASSERT(kNumRanks == 8 or kNumRanks == 16,
                             "GIN local ablation requires world size 8 or 16");
            DG_STATIC_ASSERT(not kHasShared,
                             "GIN local ablation excludes shared experts");
            DG_DEVICE_ASSERT(gin_transport.dev_comm.lsaSize == kGinPeerCount);
        }
    }
#endif

    // Thread indices
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();
    DG_GIN_TRACE_IF(warp_idx == 0 and lane_idx == 0, 0);
    constexpr bool kUseGinOutbox =
        kUseGin and kMegaMoeGinLocalAblationStage < 4;

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_weights);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_weights_sf);
    }

    // Workspaces and Buffer
    const auto buffer = layout::MegaMoEBuffer(
        sym_buffer.get_base_ptr(),
        kHidden, kIntermediateHidden,
        kNumRanks, kNumExperts,
        kNumMaxTokensPerRank, kNumTopk,
        kNumRingTokens, kNumSFRingTokens,
        /*with_sf=*/ true,
        kNumSharedExperts,
        /*with_gin=*/ kUseGin,
        /*num_sms=*/ kUseGin ? kNumSMs : 0u,
#ifdef DG_MEGAMOE_GIN
        /*gin_completion_batch=*/ kUseGin ? gin_transport.completion_batch : 1u,
        /*gin_outbox_depth=*/ kUseGin ? gin_transport.outbox_depth : 8u,
        /*gin_bulk_combine=*/ kUseGin and kMegaMoeGinBulkCombine
#else
        /*gin_completion_batch=*/ 1u,
        /*gin_outbox_depth=*/ 8u,
        /*gin_bulk_combine=*/ false
#endif
    );
    if constexpr (kUseGin and kMegaMoeGinDirectDispatch) {
        // The compact direct-control packets alias this concrete GIN layout,
        // not the generic symmetric-address map passed to the kernel.  Check
        // the runtime base after MegaMoEBuffer has materialized all offsets.
        DG_DEVICE_ASSERT(
            reinterpret_cast<uintptr_t>(
                buffer.gin_workspace.scale_scratch_buffer.base) %
                layout::kMegaMoeGinDirectDispatchPacketAlignment == 0);
    }
    if constexpr (kUseGin and kMegaMoeGinStrongVACombineTerminal) {
        // This candidate removes a world collective, so a rank-local fit
        // fallback is forbidden. Host/JIT validation checks the same bound;
        // keep a device assertion at the concrete registered layout too.
        DG_DEVICE_ASSERT(
            buffer.gin_workspace.combine_overlap_alias_fits());
        DG_DEVICE_ASSERT(
            buffer.gin_workspace.combine_terminal_signal_buffer
                    .get_num_bytes() ==
                kGinPeerCount *
                    layout::kMegaMoeGinCombineTerminalSignalStride);
        if constexpr (kGinOwnerSlotShape) {
            DG_DEVICE_ASSERT(
                buffer.gin_workspace.combine_direct_reduce_alias_fits());
        }
    }
    const auto workspace = buffer.workspace;
    const auto use_gin_bulk_combine_this_launch = [&]() {
        if constexpr (kUseGin and kMegaMoeGinBulkCombine) {
            return ptx::ld_acq(workspace.get_gin_world_active_ptr()) != 0 and
                   ptx::ld_acq(
                       workspace.get_gin_world_bulk_ineligible_ptr()) == 0;
        }
        return false;
    };
    const auto use_gin_direct_dispatch_this_launch = [&]() {
        if constexpr (kUseGin and kMegaMoeGinDirectDispatch) {
            return ptx::ld_acq(workspace.get_gin_world_active_ptr()) != 0 and
                   ptx::ld_acq(
                       workspace.get_gin_world_bulk_ineligible_ptr()) == 0;
        }
        return false;
    };
    const auto use_gin_combine_overlap_this_launch = [&]() {
        if constexpr (kUseGin and kMegaMoeGinCombineOverlap) {
            const bool world_eligible =
                use_gin_bulk_combine_this_launch() and
                use_gin_direct_dispatch_this_launch();
            if constexpr (kMegaMoeGinStrongVACombineTerminal) {
                // Host preflight and the concrete assertion above guarantee
                // the fixed scratch extent. Once receiver waits replace the
                // world collective, no rank-local fit branch is permissible.
                return world_eligible;
            }
            // Local fit fallback is safe only on the accepted path: it changes
            // this owner's send mechanism while every receiver still reaches
            // the same late header and world Put barrier.
            return world_eligible and
                   buffer.gin_workspace.combine_overlap_alias_fits();
        }
        return false;
    };
    const auto use_gin_strongva_combine_terminal_this_launch = [&]() {
        if constexpr (kUseGin and kMegaMoeGinStrongVACombineTerminal) {
            // Both inputs are outputs of the existing world consensus. The
            // concrete scratch fit is asserted above, never used as a local
            // branch after this protocol removes the world Put barrier.
            return use_gin_bulk_combine_this_launch() and
                   use_gin_direct_dispatch_this_launch();
        }
        return false;
    };
    const auto use_gin_combine_owner_slot_ready_this_launch = [&]() {
        if constexpr (kGinOwnerSlotShape) {
            return use_gin_strongva_combine_terminal_this_launch() and
                   num_tokens <=
                       layout::kMegaMoeGinDirectDispatchMaxTokens;
        }
        return false;
    };
    // Pre-consensus packing may prepare only local storage. The same full
    // alias bound and local token guard also gate every later inverse lookup.
    const bool prepare_gin_direct_reduce_ordinals = [&]() {
        if constexpr (kUseGin and kMegaMoeGinCombineOverlap)
            return num_tokens <= layout::kMegaMoeGinDirectDispatchMaxTokens and
                   buffer.gin_workspace.combine_direct_reduce_alias_fits();
        return false;
    }();

    // SF and its buffer configs
    constexpr uint32_t kGranK = 32;
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    DG_STATIC_ASSERT(SF_BLOCK_M == math::constexpr_align(BLOCK_M, kNumUTCCPAlignedElems), "Invalid SF_BLOCK_M");
    DG_STATIC_ASSERT(SF_BLOCK_N == BLOCK_N, "No padding is needed for SFB");

    // UTCCP 4x32 transpose index mapping within each 128-element group
    const auto transform_sf_token_idx = [](const uint32_t& token_idx_in_expert) {
        const uint32_t idx = token_idx_in_expert % BLOCK_M;
        return token_idx_in_expert / BLOCK_M * SF_BLOCK_M +
               (idx & ~127u) + (idx & 31u) * 4 + ((idx >> 5) & 3u);
    };

    // Data types
    // NOTES: activations are FP8 (e4m3), weights are FP4 (e2m1)
    using a_dtype_t = cutlass::float_e4m3_t;
    using b_dtype_t = cutlass::detail::float_e2m1_unpacksmem_t;
    using shared_b_dtype_t = cutlass::float_e4m3_t;

    // MMA configs
    // NOTES: always swap A/B, 2-CTA MMA, and matrices are K-major
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * 2;
    constexpr uint32_t UMMA_N = BLOCK_M;  // Swap AB
    constexpr uint32_t UMMA_BLOCK_K = 128;
    constexpr uint32_t UMMA_K = 32;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / 2;  // Multicast on A
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    DG_STATIC_ASSERT(BLOCK_M % 16 == 0, "Invalid block M");
    DG_STATIC_ASSERT(BLOCK_N == LAYOUT_AD_M, "Invalid block N");

    // Swizzle configs
    constexpr uint32_t kSwizzleAMode = 128;
    constexpr uint32_t kSwizzleBMode = 128;
    constexpr uint32_t kSwizzleCDMode = 128;
    DG_STATIC_ASSERT(BLOCK_N % kSwizzleCDMode == 0, "Invalid block N");

    // Epilogue configs
    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumTMAStoreStages = 2;

    // Shared memory
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    // Scheduler configs
    constexpr uint32_t kNumScheduleStages = 2;
    constexpr uint32_t kNumScheduleConsumerThreads = 2 * kNumEpilogueThreads;

    // Shared memory sizes
    // NOTES: FP8 CD output for L1 (2 TMA stages, BLOCK_N/2 post-SwiGLU), BF16 output for L2 (no TMA, a single stage)
    constexpr uint32_t L1_OUT_BLOCK_N = BLOCK_N / 2;
    constexpr uint32_t AMAX_REDUCTION_WARP_BUFFER_SIZE = STORE_BLOCK_M / 2; // float2

    struct SharedStorage {
        alignas(kSharedMemoryAlignment) uint32_t expert_token_count[kNumExperts];
        alignas(kSharedMemoryAlignment) uint8_t dispatch_send_buffer[kNumDispatchWarps][kNumBytesPerPull];
        union {
            alignas(kSharedMemoryAlignment) cutlass::float_e4m3_t l1[kNumEpilogueWarpgroups][kNumTMAStoreStages][STORE_BLOCK_M * L1_OUT_BLOCK_N];
            alignas(kSharedMemoryAlignment) nv_bfloat16 l2[kNumEpilogueWarpgroups][STORE_BLOCK_M * BLOCK_N];
        } smem_d;
        alignas(kSharedMemoryAlignment) a_dtype_t smem_a[kNumStages][LOAD_BLOCK_M * BLOCK_K];
        alignas(kSharedMemoryAlignment) b_dtype_t smem_b[kNumStages][LOAD_BLOCK_N * BLOCK_K];
        uint32_t smem_sfa[kNumStages][SF_BLOCK_M * (BLOCK_K / 128)];
        uint32_t smem_sfb[kNumStages][SF_BLOCK_N * (BLOCK_K / 128)];
        float2 amax_reduction[kNumEpilogueWarps][AMAX_REDUCTION_WARP_BUFFER_SIZE];
        task_info_t task_infos[kNumScheduleStages];
        Barrier dispatch_barriers[kNumDispatchWarps];
        Barrier full_barriers[kNumStages];
        Barrier empty_barriers[kNumStages];
        Barrier tmem_full_barriers[kNumEpilogueStages];
        Barrier tmem_empty_barriers[kNumEpilogueStages];
        Barrier combine_barriers[kNumEpilogueWarps * 2];
        Barrier task_info_full_barriers[kNumScheduleStages];
        Barrier task_info_empty_barriers[kNumScheduleStages];
        uint32_t tmem_ptr_in_smem;
    };
    constexpr uint32_t kNumReusableSmemBytes = offsetof(SharedStorage, dispatch_barriers);
    SharedStorage &shared_storage = *reinterpret_cast<SharedStorage*>(smem_buffer);

    // Send buffers
    constexpr auto pull_layout = layout::Data(kNumBytesPerPull);
    const auto smem_send_buffers = layout::Buffer(
        pull_layout, kNumDispatchWarps, 1,
        static_cast<void*>(shared_storage.dispatch_send_buffer));

    // Tensor memory size
    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // A cluster sync is essential for 2CTA tensor memory allocation
    comm::cluster_sync_with_relaxed_arrive();

    // Initialization
    if (warp_idx == 0) {
        // Clean shared memory
        if (cute::elect_one_sync()) {
            // The bytes must be 8 bytes aligned
            ptx::st_shared_bulk(
                shared_storage.expert_token_count,
                math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment)
            );
        }
    } else if (warp_idx == 1) {
        // Init m-barriers for dispatch
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32)
            shared_storage.dispatch_barriers[i].init(1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Init GEMM barriers
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++ i) {
                // Arrive at 2 CTAs, A + B
                shared_storage.full_barriers[i].init(2 * 2);
                shared_storage.empty_barriers[i].init(1);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
                // Arrive at all CTAs
                shared_storage.tmem_full_barriers[i].init(1);
                // Arrive only at the leader CTA
                shared_storage.tmem_empty_barriers[i].init(2 * kNumEpilogueThreads);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++ i)
                shared_storage.combine_barriers[i].init(1);
            #pragma unroll
            for (uint32_t i = 0; i < kNumScheduleStages; ++ i) {
                shared_storage.task_info_full_barriers[i].init(1);
                shared_storage.task_info_empty_barriers[i].init(kNumScheduleConsumerThreads);
            }
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 3) {
        // Allocate tensor memory
        Allocator().allocate(kNumTmemCols, &shared_storage.tmem_ptr_in_smem);
    }
    // NOTES: Using `.relaxed` is allowed here since `fence_barrier_init` is `.release.cluster`,
    // and `barrier.cluster.wait.aligned` is by default `.acquire`
    comm::cluster_sync_with_relaxed_arrive();

    // Task scheduler
    auto scheduler = sched::MegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank,
        kNumSMs, kNumRanks,
        kNumRingBlocks,
        kNumSharedExperts>(
            workspace,
            shared_storage.task_info_full_barriers,
            shared_storage.task_info_empty_barriers,
            shared_storage.task_infos
    );

    // MMA pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // Intra-SM Barrier indices
    constexpr uint32_t kDispatchBarrierIdx = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx = 2;
    constexpr uint32_t kEpilogueWGBarrierStartIdx = 3;

    // NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag = 1;
    constexpr uint32_t kBeforeCombineReduceBarrierTag = 2;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag = 3;
    constexpr uint32_t kGinActiveDecisionBarrierTag = 4;

    // GIN world-barrier slots.  Input publication needs two collectives:
    // first make inbound PUTs visible at each paired ingress, then prove that
    // every ingress has completed that target-local fence before an arbitrary
    // owner rank reads the ingress mirror over NVLink.
    constexpr uint32_t kGinInputPutBarrierIdx = 0;
    constexpr uint32_t kGinInputIngressReadyBarrierIdx = 1;
    constexpr uint32_t kGinCombinePutBarrierIdx = 2;
    constexpr uint32_t kGinCleanupBarrierIdx = 3;

    // Adjust registers
    // NOTES: more experts per rank will cost more schedulers' registers
    constexpr bool kUseMoreEpilogueRegisters = kNumExpertsPerRank <= 64;
    constexpr uint32_t kNumDispatchRegisters = kUseMoreEpilogueRegisters ? 48 : 96;
    constexpr uint32_t kNumNonEpilogueRegisters = kUseMoreEpilogueRegisters ? 40 : 88;
    constexpr uint32_t kNumEpilogueRegisters = kUseMoreEpilogueRegisters ? 208 : 160;
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                     "Too many registers");

    // Grid sync index assignments (dispatch and epilogue use separate counters to avoid conflicts)
    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;

    // Different warp roles
    if (warp_idx < kNumDispatchWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();

        // Dispatch warps
        DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto read_topk_idx = [&](const auto& process) {
            // TODO: figure out better unrolling
            // Now, `unroll` is better than `unroll 8`
            #pragma unroll
            for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                 i < num_tokens;
                 i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
                // Allocate slots for each token-topk
                int expert_idx = -1;
                if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                    expert_idx = static_cast<int>(
                        __ldg(buffer.input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx));
                    if (expert_idx >= 0)
                        process(i * kNumTopk + lane_idx, expert_idx);
                }
                __syncwarp();
            }
        };

        // Count experts' tokens
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
           atomicAdd_block(shared_storage.expert_token_count + expert_idx, 1);
        });
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Get SM offset (~6.5 us)
        #pragma unroll
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(shared_storage.expert_token_count[i]);
            shared_storage.expert_token_count[i] = static_cast<uint32_t>(
                ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Write source indices (~2 us with 512 tokens)
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
            const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
            const auto dst_slot_idx = atomicAdd_block(shared_storage.expert_token_count + expert_idx, 1);
            // Every token must route to a given expert at most once.  This is
            // the existing top-k contract and bounds both the legacy route
            // array and the appended cross-host staging array.
            DG_DEVICE_ASSERT(dst_slot_idx < kNumMaxTokensPerRank);
            const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
            if constexpr (kUseGin) {
#ifdef DG_MEGAMOE_GIN
                if (gin_transport.is_same_lsa_peer(dst_rank_idx)) {
                    *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
                } else {
                    if constexpr (kMegaMoeGinLocalAblationStage > 0) {
                        DG_DEVICE_ASSERT(
                            false and
                            "GIN local ablation received a cross-LSA route");
                    } else {
                        *buffer.gin_workspace.get_route_staging_ptr(
                            expert_idx, dst_slot_idx) = token_topk_idx;
                    }
                }
#endif
            } else {
                *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
            }
        });

        // Grid sync
        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );

        DG_GIN_TRACE_IF(warp_idx == 0 and lane_idx == 0, 1);
        if constexpr (kUseGin and kMegaMoeGinPreconsensusPack) {
#ifdef DG_MEGAMOE_GIN
            // Packet construction depends only on the route grid rendezvous
            // above, not on the world activity decision.  Let SM1 build the
            // exact cooperative owner packets while SM0 exchanges the pair
            // flags below.  No GIN payload is issued until the final decision
            // selects direct dispatch.
            //
            // The local token guard prevents an ineligible T64 transition
            // from exceeding the fixed T48 packet capacity.  World-uniform
            // eligibility later guarantees that every direct issuer prepared
            // its packet during this launch.  The unused send-side readiness
            // prefix holds only the local route-count scratch; it is outside
            // the transmitted count/route interval and is never a signal
            // target.
            if (num_tokens <= layout::kMegaMoeGinDirectDispatchMaxTokens and
                sm_idx == 1 and warp_idx == 0) {
                const uint32_t lsa_size = static_cast<uint32_t>(
                    gin_transport.dev_comm.lsaSize);
                DG_DEVICE_ASSERT(lsa_size == kGinPeerCount);
                DG_STATIC_ASSERT(
                    kNumExpertsPerRank == 56 and
                    layout::kMegaMoeGinNumDataContexts == 8,
                    "Pre-consensus pack requires 8x56 layout");
                constexpr uint32_t kExpertsPerHelper = 14;
                const uint32_t peer_in_lsa = lane_idx & 7u;
                const uint32_t helper_idx = lane_idx >> 3;
                // Keep the EP16 lane/shuffle map. EP8's unused peer columns
                // contribute zero but must still execute every warp collective.
                const bool active_pack_lane = kGinPeerCount == 8 or
                    peer_in_lsa < kGinPeerCount;
                const uint32_t remote_lsa_base =
                    (1u - sym_buffer.rank_idx / lsa_size) * lsa_size;
                const uint32_t peer = remote_lsa_base + peer_in_lsa;
                auto* packed_counts = buffer.gin_workspace
                    .get_direct_dispatch_count_ptr(
                        /*send=*/ true, peer_in_lsa);
                auto* packed_routes = buffer.gin_workspace
                    .get_direct_dispatch_route_ptr(
                        /*send=*/ true, peer_in_lsa);

                const uint32_t first_expert =
                    helper_idx * kExpertsPerHelper;
                uint32_t segment_count = 0;
                #pragma unroll
                for (uint32_t helper_expert = 0;
                     active_pack_lane and helper_expert < kExpertsPerHelper;
                     ++helper_expert) {
                    const uint32_t local_expert =
                        first_expert + helper_expert;
                    const uint32_t global_expert =
                        peer * kNumExpertsPerRank + local_expert;
                    const uint32_t count = static_cast<uint32_t>(
                        *workspace.get_expert_send_count_ptr(global_expert));
                    packed_counts[local_expert] = count;
                    segment_count += count;
                }

                const uint32_t segment_count_0 = __shfl_sync(
                    0xffffffffu, segment_count, peer_in_lsa);
                const uint32_t segment_count_1 = __shfl_sync(
                    0xffffffffu, segment_count, peer_in_lsa + 8u);
                const uint32_t segment_count_2 = __shfl_sync(
                    0xffffffffu, segment_count, peer_in_lsa + 16u);
                const uint32_t segment_count_3 = __shfl_sync(
                    0xffffffffu, segment_count, peer_in_lsa + 24u);
                const uint32_t route_count =
                    segment_count_0 + segment_count_1 + segment_count_2 +
                    segment_count_3;
                uint32_t route_prefix =
                    (helper_idx > 0 ? segment_count_0 : 0u) +
                    (helper_idx > 1 ? segment_count_1 : 0u) +
                    (helper_idx > 2 ? segment_count_2 : 0u);
                const uint32_t segment_begin = route_prefix;

                #pragma unroll
                for (uint32_t helper_expert = 0;
                     active_pack_lane and helper_expert < kExpertsPerHelper;
                     ++helper_expert) {
                    const uint32_t local_expert =
                        first_expert + helper_expert;
                    const uint32_t global_expert =
                        peer * kNumExpertsPerRank + local_expert;
                    const uint32_t count = static_cast<uint32_t>(
                        *workspace.get_expert_send_count_ptr(global_expert));
                    DG_DEVICE_ASSERT(
                        route_prefix + count <=
                        layout::kMegaMoeGinDirectDispatchMaxRoutes);
                    for (uint32_t slot = 0; slot < count; ++slot) {
                        const uint32_t token_topk_idx =
                            *buffer.gin_workspace.get_route_staging_ptr(
                                global_expert, slot);
                        packed_routes[route_prefix + slot] = token_topk_idx;
                        if (prepare_gin_direct_reduce_ordinals) {
                            DG_DEVICE_ASSERT(token_topk_idx < num_tokens * kNumTopk);
                            *buffer.gin_workspace.get_combine_direct_reduce_ordinal_ptr(
                                token_topk_idx) = route_prefix + slot;
                        }
                    }
                    route_prefix += count;
                }
                DG_DEVICE_ASSERT(
                    route_prefix == segment_begin + segment_count);
                DG_DEVICE_ASSERT(route_count <= num_tokens * kNumTopk);

                if (active_pack_lane and helper_idx == 0) {
                    auto* route_count_scratch = reinterpret_cast<uint32_t*>(
                        buffer.gin_workspace
                            .get_direct_dispatch_ready_ptr(
                                /*send=*/ true, peer_in_lsa));
                    *route_count_scratch = route_count;
                }
                __threadfence_system();
                __syncwarp();
                DG_GIN_TRACE_IF(lane_idx == 0, 2);
            }
#endif
        }

        if constexpr (kUseGin and kMegaMoeGinActiveFastPath) {
#ifdef DG_MEGAMOE_GIN
            // A rank-local route test cannot safely select between an LSA
            // barrier and a world collective.  The same consensus also carries
            // a small-decode ineligibility bit, because num_tokens may differ
            // by rank.  Both LSAs consequently select direct dispatch and/or
            // bulk combine, or their original r75 paths, uniformly.
            if (sm_idx == 0 and warp_idx == 0) {
                bool lane_has_remote = false;
                for (uint32_t expert = lane_idx;
                     expert < kNumExperts; expert += 32) {
                    const uint32_t dst_rank =
                        expert / kNumExpertsPerRank;
                    lane_has_remote |=
                        not gin_transport.is_same_lsa_peer(dst_rank) and
                        static_cast<uint32_t>(
                            *workspace.get_expert_send_count_ptr(expert)) > 0;
                }
                const bool local_has_remote =
                    __any_sync(0xffffffffu, lane_has_remote);
                if (lane_idx == 0) {
                    const uint32_t local_flags =
                        static_cast<uint32_t>(local_has_remote) |
                        (static_cast<uint32_t>(
                             (kMegaMoeGinBulkCombine or
                              kMegaMoeGinDirectDispatch) and
                             num_tokens > layout::
                                 kMegaMoeGinBulkCombineMaxTokens)
                         << 1);
                    auto* epoch_ptr =
                        workspace.get_gin_active_launch_epoch_ptr();
                    const uint64_t epoch = *epoch_ptr;
                    DG_DEVICE_ASSERT(epoch < ((1ull << 60) - 1));
                    const uint64_t generation = epoch + 1;
                    *epoch_ptr = generation;
                    const uint32_t lsa_size = static_cast<uint32_t>(
                        gin_transport.dev_comm.lsaSize);
                    const uint32_t paired_rank =
                        (sym_buffer.rank_idx + lsa_size) % kNumRanks;
                    auto* pair_active_ptr =
                        workspace.get_gin_pair_active_ptr();
                    auto* pair_mailbox_ptr =
                        workspace.get_gin_pair_mailbox_ptr(
                            static_cast<uint32_t>(epoch & 1ull));
                    const uint32_t paired_flags =
                        comm::mega_moe_gin_exchange_pair_flags(
                            gin_transport, paired_rank,
                            sym_buffer.get_base_ptr(), pair_mailbox_ptr,
                            generation, local_flags);
                    *pair_active_ptr = local_flags | paired_flags;
                    __threadfence_system();
                }
                __syncwarp();
            }

            if constexpr (kMegaMoeGinActivityGateOpt) {
                // Route construction already completed a grid rendezvous
                // immediately above.  Only SM0 changes state before this LSA
                // rendezvous, and the final decision is followed by another
                // full grid rendezvous.  Keep the cross-rank signal/acquire,
                // but remove its redundant grid prologue and epilogue.
                comm::nvlink_lsa_barrier<
                    kNumRanks, kGinPeerCount, kNumSMs, kNumDispatchThreads,
                    kDispatchGridSyncIndex, kGinActiveDecisionBarrierTag>(
                        workspace, sym_buffer, sm_idx, thread_idx,
                        [=]() {
                            ptx::sync_aligned(
                                kNumDispatchThreads, kDispatchBarrierIdx);
                        },
                        /* Pair publication is warp-synchronized above */ false,
                        /* Final decision has its own grid rendezvous */ false);
            } else {
                comm::nvlink_lsa_barrier<
                    kNumRanks, kGinPeerCount, kNumSMs, kNumDispatchThreads,
                    kDispatchGridSyncIndex, kGinActiveDecisionBarrierTag>(
                        workspace, sym_buffer, sm_idx, thread_idx,
                        [=]() {
                            ptx::sync_aligned(
                                kNumDispatchThreads, kDispatchBarrierIdx);
                        });
            }

            if (sm_idx == 0 and warp_idx == 0) {
                uint32_t world_flags = 0;
                const uint32_t lsa_size = static_cast<uint32_t>(
                    gin_transport.dev_comm.lsaSize);
                const uint32_t lsa_base =
                    (sym_buffer.rank_idx / lsa_size) * lsa_size;

                if constexpr (kMegaMoeGinActivityGateOpt) {
                    // Lanes 1..31 can return from the LSA helper while lane 0
                    // is still polling its arrival counter when the helper's
                    // grid epilogue is disabled.  Reconverge before letting
                    // eight lanes acquire one pair decision each.
                    __syncwarp();
                    uint32_t lane_pair_flags = 0;
                    if (lane_idx < lsa_size) {
                        const uint32_t peer = lsa_base + lane_idx;
                        lane_pair_flags = ptx::ld_acq_sys(
                            sym_buffer.map(
                                workspace.get_gin_pair_active_ptr(), peer));
                    }
                    world_flags = __reduce_or_sync(
                        0xffffffffu, lane_pair_flags);
                } else if (lane_idx == 0) {
                    for (uint32_t local_rank = 0;
                         local_rank < lsa_size; ++local_rank) {
                        const uint32_t peer = lsa_base + local_rank;
                        world_flags |= ptx::ld_acq_sys(
                            sym_buffer.map(
                                workspace.get_gin_pair_active_ptr(), peer));
                    }
                }

                if (lane_idx == 0) {
                    auto* world_active_ptr =
                        workspace.get_gin_world_active_ptr();
                    asm volatile("st.release.gpu.global.u32 [%0], %1;"
                                 :: "l"(world_active_ptr),
                                    "r"(world_flags & 1u)
                                 : "memory");
                    auto* world_ineligible_ptr =
                        workspace.get_gin_world_bulk_ineligible_ptr();
                    asm volatile("st.release.gpu.global.u32 [%0], %1;"
                                 :: "l"(world_ineligible_ptr),
                                    "r"((world_flags >> 1) & 1u)
                                 : "memory");
                }
            }
            comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
                workspace, sm_idx, thread_idx,
                [=]() {
                    ptx::sync_aligned(
                        kNumDispatchThreads, kDispatchBarrierIdx);
                });
#endif
        }

        DG_GIN_TRACE_IF(warp_idx == 0 and lane_idx == 0, 3);
        // The activity/ineligibility decision is world-uniform after the
        // consensus grid rendezvous above.  Cache it once per dispatch thread;
        // repeated acquire loads in the pull loops noticeably perturb the
        // decode floor, including launches that select the row fallback.
        const bool use_gin_bulk_combine =
            use_gin_bulk_combine_this_launch();
        const bool use_gin_direct_dispatch =
            use_gin_direct_dispatch_this_launch();
        const bool use_gin_combine_overlap =
            use_gin_combine_overlap_this_launch();
        const bool use_gin_strongva_combine_terminal =
            use_gin_strongva_combine_terminal_this_launch();
        const bool use_gin_combine_owner_slot_ready =
            use_gin_combine_owner_slot_ready_this_launch();

        if constexpr (kUseGin and kMegaMoeGinStrongVACombineTerminal) {
#ifdef DG_MEGAMOE_GIN
            // Cumulative terminal cells are never reset. Advance this expected
            // generation exactly once only on world-uniform eligible launches;
            // local-only, T64 and ordinary fallback invocations pause it.
            if (use_gin_strongva_combine_terminal and sm_idx == 0 and
                warp_idx == 0 and lane_idx == 0) {
                auto* epoch_ptr =
                    workspace.get_gin_combine_terminal_epoch_ptr();
                DG_DEVICE_ASSERT(
                    *epoch_ptr < static_cast<uint64_t>(-1));
                *epoch_ptr += 1;
                __threadfence();
            }
            __syncwarp();
#endif
        }

        if constexpr (kUseGin and kMegaMoeGinCombineOverlap) {
#ifdef DG_MEGAMOE_GIN
            // The tail of GET scale scratch is unused only in direct mode.
            // Initialize every eligible invocation: a row fallback may have
            // overwritten it since the last eligible graph replay. The
            // existing pre-pull grid/handoff publishes this initialization
            // before any producer can increment an expert fragment count.
            if (use_gin_combine_overlap and sm_idx == 0 and warp_idx == 0) {
                for (uint32_t expert = lane_idx;
                     expert < layout::kMegaMoeGinCombineOverlapNumExperts;
                     expert += 32u) {
                    *buffer.gin_workspace.get_combine_overlap_ready_ptr(
                        expert) = 0;
                    *buffer.gin_workspace.get_combine_overlap_sent_ptr(
                        expert) = 0;
                }
                __threadfence();
                __syncwarp();
            }
#endif
        }

        if constexpr (kUseGin and kMegaMoeGinDirectDispatch) {
#ifdef DG_MEGAMOE_GIN
            // VA readiness counters are cumulative and never reset.  The
            // world-uniform direct branch therefore advances this rank-local
            // expected generation exactly once per direct invocation, across
            // eager launches and CUDA Graph replay alike.
            if (use_gin_direct_dispatch and sm_idx == 0 and warp_idx == 0 and
                lane_idx == 0) {
                auto* epoch_ptr =
                    workspace.get_gin_direct_dispatch_epoch_ptr();
                *epoch_ptr += 1;
                if constexpr (kMegaMoeGinDispatchOverlap)
                    *workspace.get_gin_dispatch_payload_epoch_ptr() += 1;
                __threadfence();
            }
            __syncwarp();
#endif
        }

        if constexpr (kUseGin and kMegaMoeGinBulkCombine) {
#ifdef DG_MEGAMOE_GIN
            // The accepted path may send no packet for an empty pair, so keep
            // its startup clear. StrongVA instead queues one exact header for
            // every pair, including zero, before its ordered terminal. Do not
            // race that early NIC write with an unordered local clear; stale
            // counts are never consumed before the current terminal acquire.
            if (use_gin_bulk_combine and sm_idx == 0 and
                not use_gin_strongva_combine_terminal and warp_idx == 0 and
                lane_idx < kGinPeerCount) {
                *buffer.gin_workspace.get_bulk_combine_packet_count_ptr(
                    /*send=*/ false, lane_idx) = 0;
            }
            __syncwarp();
#endif
        }

        if constexpr (kUseGin and kMegaMoeGinLocalAblationStage < 1) {
#ifdef DG_MEGAMOE_GIN
            // SM 1 is otherwise idle while SM 0 publishes same-LSA counts.  In
            // direct mode its first kGinPeerCount lanes each build one compact
            // expert-major control slab and publish the source SoA prefix
            // straight to that owner.  The fallback keeps the paired-ingress
            // publication byte-for-byte unchanged.
            if (sm_idx == 1 and warp_idx == 0) {
                const uint32_t lsa_size =
                    static_cast<uint32_t>(gin_transport.dev_comm.lsaSize);
                const bool run_remote_path =
                    not kMegaMoeGinActiveFastPath or
                    ptx::ld_acq(
                        workspace.get_gin_world_active_ptr()) != 0;
                if constexpr (kMegaMoeGinDirectDispatch) {
                    if constexpr (kMegaMoeGinCoopDirectPack) {
                        if constexpr (kMegaMoeGinPreconsensusPack) {
                            // The complete packet was prepared before the
                            // activity consensus. Only the active peer
                            // leaders issue GIN publication, and only after
                            // the world selected direct mode. The overlap
                            // experiment posts control first and defers source
                            // completion until after this CTA's pulls.
                            if (use_gin_direct_dispatch and
                                lane_idx < lsa_size) {
                                DG_DEVICE_ASSERT(lsa_size == kGinPeerCount);
                                const uint32_t peer_in_lsa = lane_idx;
                                const uint32_t remote_lsa_base =
                                    (1u - sym_buffer.rank_idx / lsa_size) *
                                    lsa_size;
                                const uint32_t peer =
                                    remote_lsa_base + peer_in_lsa;
                                auto* packed_counts = buffer.gin_workspace
                                    .get_direct_dispatch_count_ptr(
                                        /*send=*/ true, peer_in_lsa);
                                const auto* route_count_scratch =
                                    reinterpret_cast<const uint32_t*>(
                                        buffer.gin_workspace
                                            .get_direct_dispatch_ready_ptr(
                                                /*send=*/ true,
                                                peer_in_lsa));
                                const uint32_t route_count =
                                    ptx::ld_acq(route_count_scratch);
                                DG_DEVICE_ASSERT(
                                    route_count <= num_tokens * kNumTopk);
                                const uint32_t source_lane =
                                    sym_buffer.rank_idx % lsa_size;
                                comm::mega_moe_gin_publish_direct_dispatch_ordered<
                                    kMegaMoeGinDispatchOverlap>(
                                    gin_transport, peer,
                                    /*context_stripe=*/ peer_in_lsa,
                                    sym_buffer.get_base_ptr(),
                                    /*publish_inputs=*/ route_count > 0,
                                    buffer.input_token_buffer.get_base_ptr(),
                                    buffer.gin_workspace
                                        .get_direct_input_token_ptr(
                                            source_lane),
                                    num_tokens *
                                        buffer.input_token_buffer.data_layout
                                            .template get_num_bytes<uint32_t>(),
                                    buffer.input_sf_buffer.get_base_ptr(),
                                    buffer.gin_workspace
                                        .get_direct_input_sf_ptr(source_lane),
                                    num_tokens *
                                        buffer.input_sf_buffer.data_layout
                                            .template get_num_bytes<uint32_t>(),
                                    buffer.input_topk_weights_buffer
                                        .get_base_ptr(),
                                    buffer.gin_workspace
                                        .get_direct_input_topk_weights_ptr(
                                            source_lane),
                                    num_tokens *
                                        buffer.input_topk_weights_buffer
                                            .data_layout
                                            .template get_num_bytes<uint32_t>(),
                                    packed_counts,
                                    buffer.gin_workspace
                                        .get_direct_dispatch_count_ptr(
                                            /*send=*/ false, source_lane),
                                    buffer.gin_workspace
                                        .get_direct_dispatch_control_bytes(
                                            route_count),
                                    buffer.gin_workspace
                                        .get_direct_dispatch_ready_ptr(
                                            /*send=*/ false, source_lane),
                                    buffer.gin_workspace
                                        .get_direct_dispatch_payload_ready_ptr(
                                            source_lane));
                            }
                        } else {
                        // Preserve lanes 0-7 as the sole GIN issuers while the
                        // full warp cooperatively builds eight owner packets.
                        // Lanes owner+8*h own consecutive 14-expert segments,
                        // so the four segment totals reproduce the original
                        // expert-major route prefix exactly. EP8 leaves peer
                        // columns4..7 inactive, not absent from the shuffles.
                        DG_STATIC_ASSERT(
                            kNumExpertsPerRank == 56 and
                            layout::kMegaMoeGinNumDataContexts == 8,
                            "Cooperative direct pack requires 8x56 layout");
                        if (use_gin_direct_dispatch) {
                            DG_DEVICE_ASSERT(lsa_size == kGinPeerCount);
                            constexpr uint32_t kExpertsPerHelper = 14;
                            const uint32_t peer_in_lsa = lane_idx & 7u;
                            const uint32_t helper_idx = lane_idx >> 3;
                            const bool active_pack_lane = kGinPeerCount == 8 or
                                peer_in_lsa < kGinPeerCount;
                            const uint32_t remote_lsa_base =
                                (1u - sym_buffer.rank_idx / lsa_size) *
                                lsa_size;
                            const uint32_t peer =
                                remote_lsa_base + peer_in_lsa;
                            auto* packed_counts = buffer.gin_workspace
                                .get_direct_dispatch_count_ptr(
                                    /*send=*/ true, peer_in_lsa);
                            auto* packed_routes = buffer.gin_workspace
                                .get_direct_dispatch_route_ptr(
                                    /*send=*/ true, peer_in_lsa);

                            const uint32_t first_expert =
                                helper_idx * kExpertsPerHelper;
                            uint32_t segment_count = 0;
                            #pragma unroll
                            for (uint32_t helper_expert = 0;
                                 active_pack_lane and helper_expert < kExpertsPerHelper;
                                 ++helper_expert) {
                                const uint32_t local_expert =
                                    first_expert + helper_expert;
                                const uint32_t global_expert =
                                    peer * kNumExpertsPerRank + local_expert;
                                const uint32_t count = static_cast<uint32_t>(
                                    *workspace.get_expert_send_count_ptr(
                                        global_expert));
                                packed_counts[local_expert] = count;
                                segment_count += count;
                            }

                            const uint32_t segment_count_0 = __shfl_sync(
                                0xffffffffu, segment_count, peer_in_lsa);
                            const uint32_t segment_count_1 = __shfl_sync(
                                0xffffffffu, segment_count,
                                peer_in_lsa + 8u);
                            const uint32_t segment_count_2 = __shfl_sync(
                                0xffffffffu, segment_count,
                                peer_in_lsa + 16u);
                            const uint32_t segment_count_3 = __shfl_sync(
                                0xffffffffu, segment_count,
                                peer_in_lsa + 24u);
                            const uint32_t route_count =
                                segment_count_0 + segment_count_1 +
                                segment_count_2 + segment_count_3;
                            uint32_t route_prefix =
                                (helper_idx > 0 ? segment_count_0 : 0u) +
                                (helper_idx > 1 ? segment_count_1 : 0u) +
                                (helper_idx > 2 ? segment_count_2 : 0u);
                            const uint32_t segment_begin = route_prefix;

                            // A second pass keeps the candidate's register
                            // footprint bounded while parallelizing the route
                            // copies.  The count cells are hot from the first
                            // pass, and every output slice remains disjoint.
                            #pragma unroll
                            for (uint32_t helper_expert = 0;
                                 active_pack_lane and helper_expert < kExpertsPerHelper;
                                 ++helper_expert) {
                                const uint32_t local_expert =
                                    first_expert + helper_expert;
                                const uint32_t global_expert =
                                    peer * kNumExpertsPerRank + local_expert;
                                const uint32_t count = static_cast<uint32_t>(
                                    *workspace.get_expert_send_count_ptr(
                                        global_expert));
                                DG_DEVICE_ASSERT(
                                    route_prefix + count <=
                                    layout::kMegaMoeGinDirectDispatchMaxRoutes);
                                for (uint32_t slot = 0; slot < count; ++slot) {
                                    const uint32_t token_topk_idx =
                                        *buffer.gin_workspace
                                             .get_route_staging_ptr(
                                                 global_expert, slot);
                                    packed_routes[route_prefix + slot] = token_topk_idx;
                                    if (prepare_gin_direct_reduce_ordinals) {
                                        DG_DEVICE_ASSERT(token_topk_idx < num_tokens * kNumTopk);
                                        *buffer.gin_workspace.get_combine_direct_reduce_ordinal_ptr(
                                            token_topk_idx) = route_prefix + slot;
                                    }
                                }
                                route_prefix += count;
                            }
                            DG_DEVICE_ASSERT(
                                route_prefix ==
                                segment_begin + segment_count);
                            DG_DEVICE_ASSERT(
                                route_count <= num_tokens * kNumTopk);

                            // Every packet writer, not just its GIN leader,
                            // must publish its local stores before NIC DMA can
                            // read the shared packet.
                            __threadfence_system();
                            __syncwarp();

                            if (active_pack_lane and helper_idx == 0) {
                                const uint32_t source_lane =
                                    sym_buffer.rank_idx % lsa_size;
                                comm::mega_moe_gin_publish_direct_dispatch(
                                    gin_transport, peer,
                                    /*context_stripe=*/ peer_in_lsa,
                                    sym_buffer.get_base_ptr(),
                                    /*publish_inputs=*/ route_count > 0,
                                    buffer.input_token_buffer.get_base_ptr(),
                                    buffer.gin_workspace
                                        .get_direct_input_token_ptr(source_lane),
                                    num_tokens *
                                        buffer.input_token_buffer.data_layout
                                            .template get_num_bytes<uint32_t>(),
                                    buffer.input_sf_buffer.get_base_ptr(),
                                    buffer.gin_workspace
                                        .get_direct_input_sf_ptr(source_lane),
                                    num_tokens *
                                        buffer.input_sf_buffer.data_layout
                                            .template get_num_bytes<uint32_t>(),
                                    buffer.input_topk_weights_buffer
                                        .get_base_ptr(),
                                    buffer.gin_workspace
                                        .get_direct_input_topk_weights_ptr(
                                            source_lane),
                                    num_tokens *
                                        buffer.input_topk_weights_buffer
                                            .data_layout
                                            .template get_num_bytes<uint32_t>(),
                                    packed_counts,
                                    buffer.gin_workspace
                                        .get_direct_dispatch_count_ptr(
                                            /*send=*/ false, source_lane),
                                    buffer.gin_workspace
                                        .get_direct_dispatch_control_bytes(route_count),
                                    buffer.gin_workspace
                                        .get_direct_dispatch_ready_ptr(
                                            /*send=*/ false, source_lane));
                            }
                        }
                        }
                    } else {
                        if (use_gin_direct_dispatch and lane_idx < lsa_size) {
                            const uint32_t remote_lsa_base =
                                (1u - sym_buffer.rank_idx / lsa_size) *
                                lsa_size;
                            const uint32_t peer_in_lsa = lane_idx;
                            const uint32_t peer =
                                remote_lsa_base + peer_in_lsa;
                            auto* packed_counts = buffer.gin_workspace
                                .get_direct_dispatch_count_ptr(
                                    /*send=*/ true, peer_in_lsa);
                            auto* packed_routes = buffer.gin_workspace
                                .get_direct_dispatch_route_ptr(
                                    /*send=*/ true, peer_in_lsa);

                            uint32_t route_count = 0;
                            #pragma unroll
                            for (uint32_t local_expert = 0;
                                 local_expert < kNumExpertsPerRank;
                                 ++local_expert) {
                                const uint32_t global_expert =
                                    peer * kNumExpertsPerRank + local_expert;
                                const uint32_t count = static_cast<uint32_t>(
                                    *workspace.get_expert_send_count_ptr(
                                        global_expert));
                                packed_counts[local_expert] = count;
                                DG_DEVICE_ASSERT(
                                    route_count + count <=
                                    layout::kMegaMoeGinDirectDispatchMaxRoutes);
                                for (uint32_t slot = 0; slot < count; ++slot) {
                                    const uint32_t token_topk_idx =
                                        *buffer.gin_workspace
                                             .get_route_staging_ptr(
                                                 global_expert, slot);
                                    packed_routes[route_count + slot] = token_topk_idx;
                                    if (prepare_gin_direct_reduce_ordinals) {
                                        DG_DEVICE_ASSERT(token_topk_idx < num_tokens * kNumTopk);
                                        *buffer.gin_workspace.get_combine_direct_reduce_ordinal_ptr(
                                            token_topk_idx) = route_count + slot;
                                    }
                                }
                                route_count += count;
                            }
                            DG_DEVICE_ASSERT(
                                route_count <= num_tokens * kNumTopk);
                            __threadfence_system();

                            const uint32_t source_lane =
                                sym_buffer.rank_idx % lsa_size;
                            comm::mega_moe_gin_publish_direct_dispatch(
                                gin_transport, peer,
                                /*context_stripe=*/ peer_in_lsa,
                                sym_buffer.get_base_ptr(),
                                /*publish_inputs=*/ route_count > 0,
                                buffer.input_token_buffer.get_base_ptr(),
                                buffer.gin_workspace.get_direct_input_token_ptr(
                                    source_lane),
                                num_tokens *
                                    buffer.input_token_buffer.data_layout
                                        .template get_num_bytes<uint32_t>(),
                                buffer.input_sf_buffer.get_base_ptr(),
                                buffer.gin_workspace.get_direct_input_sf_ptr(
                                    source_lane),
                                num_tokens *
                                    buffer.input_sf_buffer.data_layout
                                        .template get_num_bytes<uint32_t>(),
                                buffer.input_topk_weights_buffer.get_base_ptr(),
                                buffer.gin_workspace
                                    .get_direct_input_topk_weights_ptr(
                                        source_lane),
                                num_tokens *
                                    buffer.input_topk_weights_buffer.data_layout
                                        .template get_num_bytes<uint32_t>(),
                                packed_counts,
                                buffer.gin_workspace
                                    .get_direct_dispatch_count_ptr(
                                        /*send=*/ false, source_lane),
                                buffer.gin_workspace
                                    .get_direct_dispatch_control_bytes(route_count),
                                buffer.gin_workspace
                                    .get_direct_dispatch_ready_ptr(
                                        /*send=*/ false, source_lane));
                        }
                    }
                }
                if (not use_gin_direct_dispatch and lane_idx == 0 and
                    num_tokens > 0 and run_remote_path) {
                    const uint32_t peer =
                        (sym_buffer.rank_idx + lsa_size) % kNumRanks;
                    const auto published_tokens =
                        buffer.gin_workspace.published_input_token_buffer
                            .get_base_ptr();
                    const auto published_sf =
                        buffer.gin_workspace.published_input_sf_buffer
                            .get_base_ptr();
                    const auto published_weights =
                        buffer.gin_workspace.published_input_topk_weights_buffer
                            .get_base_ptr();
                    comm::mega_moe_gin_publish_inputs(
                        gin_transport, peer, /*context_stripe=*/ 0,
                        sym_buffer.get_base_ptr(),
                        buffer.input_token_buffer.get_base_ptr(),
                        published_tokens,
                        num_tokens * buffer.input_token_buffer.data_layout
                                         .template get_num_bytes<uint32_t>(),
                        buffer.input_sf_buffer.get_base_ptr(), published_sf,
                        num_tokens * buffer.input_sf_buffer.data_layout
                                         .template get_num_bytes<uint32_t>(),
                        buffer.input_topk_weights_buffer.get_base_ptr(),
                        published_weights,
                        num_tokens *
                            buffer.input_topk_weights_buffer.data_layout
                                .template get_num_bytes<uint32_t>());
                }
                __syncwarp();
            }
#endif
        }

        // Write expert count
        if (sm_idx == 0) {
            if constexpr (kUseGin) {
#ifdef DG_MEGAMOE_GIN
                const bool run_local_count_path =
                    kMegaMoeGinLocalAblationStage >= 3 or
                    (kMegaMoeGinActiveFastPath and
                     ptx::ld_acq(
                         workspace.get_gin_world_active_ptr()) == 0);
                if (run_local_count_path) {
                    // The matched local-floor specialization forms two
                    // independent eight-rank NVLink teams.  Seed the missing
                    // remote team's ready-tag contribution, then use the
                    // original local atomic count publication.  All additions
                    // commute, and the scheduler still observes its unchanged
                    // kNumSMs * kNumRanks tag.
                    constexpr uint32_t kLocalLsaSize = kGinPeerCount;
                    const uint64_t missing_sources_tag =
                        static_cast<uint64_t>(
                            kNumSMs * (kNumRanks - kLocalLsaSize)) << 32;
                    for (uint32_t i = thread_idx;
                         i < kNumExpertsPerRank;
                         i += kNumDispatchThreads) {
                        ptx::atomic_add_sys(
                            workspace.get_expert_recv_count_sum_ptr(i),
                            missing_sources_tag);
                    }

                    #pragma unroll
                    for (uint32_t i = thread_idx;
                         i < kNumExperts;
                         i += kNumDispatchThreads) {
                        const uint32_t dst_rank_idx =
                            i / kNumExpertsPerRank;
                        const uint32_t dst_local_expert_idx =
                            i % kNumExpertsPerRank;
                        const uint64_t expert_status =
                            *workspace.get_expert_send_count_ptr(i);
                        if (gin_transport.is_same_lsa_peer(dst_rank_idx)) {
                            *sym_buffer.map(
                                workspace.get_expert_recv_count_ptr(
                                    sym_buffer.rank_idx,
                                    dst_local_expert_idx),
                                dst_rank_idx) =
                                    static_cast<uint32_t>(expert_status);
                            ptx::atomic_add_sys(
                                sym_buffer.map(
                                    workspace.get_expert_recv_count_sum_ptr(
                                        dst_local_expert_idx),
                                    dst_rank_idx),
                                expert_status);
                        } else {
                            DG_DEVICE_ASSERT(
                                static_cast<uint32_t>(expert_status) == 0 and
                                "GIN local ablation received a cross-LSA count");
                        }
                    }
                } else {
                    // Same-LSA count cells retain direct stores. Cross-host
                    // counts are packed into one contiguous uint64 vector per
                    // destination rank.
                    #pragma unroll
                    for (uint32_t i = thread_idx;
                         i < kNumExperts;
                         i += kNumDispatchThreads) {
                        const auto dst_rank_idx = i / kNumExpertsPerRank;
                        const auto dst_local_expert_idx =
                            i % kNumExpertsPerRank;
                        const uint64_t count = static_cast<uint32_t>(
                            *workspace.get_expert_send_count_ptr(i));
                        buffer.gin_workspace.count_staging_buffer
                            .get_rank_buffer(dst_rank_idx)
                            .template get_base_ptr<uint64_t>()
                                [dst_local_expert_idx] = count;
                        if (gin_transport.is_same_lsa_peer(dst_rank_idx)) {
                            *sym_buffer.map(
                                workspace.get_expert_recv_count_ptr(
                                    sym_buffer.rank_idx,
                                    dst_local_expert_idx),
                                dst_rank_idx) = count;
                        } else if constexpr (
                                kMegaMoeGinLocalAblationStage > 0) {
                            DG_DEVICE_ASSERT(
                                count == 0 and
                                "GIN local ablation received a cross-LSA count");
                        }
                    }
                    ptx::sync_aligned(
                        kNumDispatchThreads, kDispatchBarrierIdx);

                    // Stages 2+ remove the known-empty remote route/count
                    // control while leaving the rest of the common path intact.
                    if constexpr (kMegaMoeGinLocalAblationStage < 2) {
                        if (not use_gin_direct_dispatch and warp_idx == 0) {
                            const uint32_t lsa_size = static_cast<uint32_t>(
                                gin_transport.dev_comm.lsaSize);
                            const bool is_remote_peer_lane = lane_idx < lsa_size;
                            const uint32_t remote_lsa_base =
                                (1u - sym_buffer.rank_idx / lsa_size) * lsa_size;
                            const uint32_t peer = remote_lsa_base + lane_idx;
                            const uint64_t* staged_counts = nullptr;
                            if (is_remote_peer_lane) {
                                staged_counts =
                                    buffer.gin_workspace.count_staging_buffer
                                        .get_rank_buffer(peer)
                                        .template get_base_ptr<uint64_t>();
                            }

                            uint32_t local_expert = 0;
                            while (true) {
                                while (is_remote_peer_lane and
                                       local_expert < kNumExpertsPerRank and
                                       static_cast<uint32_t>(
                                           staged_counts[local_expert]) == 0)
                                    ++local_expert;
                                const bool has_route =
                                    is_remote_peer_lane and
                                    local_expert < kNumExpertsPerRank;
                                const uint32_t route_mask = __ballot_sync(
                                    0xffffffffu, has_route);
                                if (route_mask == 0)
                                    break;
                                if (has_route) {
                                    const uint32_t count =
                                        static_cast<uint32_t>(
                                            staged_counts[local_expert]);
                                    const uint32_t global_expert =
                                        peer * kNumExpertsPerRank +
                                        local_expert;
                                    comm::mega_moe_gin_put(
                                        gin_transport, peer,
                                        sym_buffer.get_base_ptr(),
                                        buffer.gin_workspace
                                            .get_route_staging_ptr(
                                                global_expert, 0),
                                        workspace
                                            .get_src_token_topk_idx_ptr(
                                                local_expert,
                                                sym_buffer.rank_idx, 0),
                                        count * sizeof(uint32_t),
                                        ncclGinOptFlagsAggregateRequests);
                                    ++local_expert;
                                }
                            }

                            if (is_remote_peer_lane) {
                                comm::mega_moe_gin_put(
                                    gin_transport, peer,
                                    sym_buffer.get_base_ptr(), staged_counts,
                                    workspace.get_expert_recv_count_ptr(
                                        sym_buffer.rank_idx, 0),
                                    kNumExpertsPerRank * sizeof(uint64_t),
                                    ncclGinOptFlagsDefault);
                                comm::mega_moe_gin_flush_peer(
                                    gin_transport, peer);
                            }
                            __syncwarp();
                        }
                    }
                }
#endif
            } else {
                #pragma unroll
                for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                    const auto dst_rank_idx = i / kNumExpertsPerRank;
                    const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                    const auto expert_status = *workspace.get_expert_send_count_ptr(i);
                    *sym_buffer.map(
                        workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                        dst_rank_idx) = expert_status & 0xffffffff;
                    ptx::atomic_add_sys(
                        sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                        expert_status);
                }
            }
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Barrier before pulling
        DG_GIN_TRACE_IF(warp_idx == 0 and lane_idx == 0, 4);
        if constexpr (kUseGin and kMegaMoeGinLocalAblationStage < 3) {
#ifdef DG_MEGAMOE_GIN
            const bool run_remote_path =
                not kMegaMoeGinActiveFastPath or
                ptx::ld_acq(
                    workspace.get_gin_world_active_ptr()) != 0;
            if (run_remote_path) {
                if (use_gin_direct_dispatch) {
                    // Strong VA terminals cover only cross-LSA GIN traffic.
                    // Publish and rendezvous the unchanged same-LSA route/count
                    // stores separately before any owner starts pulling.
                    __threadfence_system();
                    comm::nvlink_lsa_barrier<
                        kNumRanks, kGinPeerCount, kNumSMs, kNumDispatchThreads,
                        kDispatchGridSyncIndex,
                        kBeforeDispatchPullBarrierTag>(
                            workspace, sym_buffer, sm_idx, thread_idx,
                            [=]() {
                                ptx::sync_aligned(
                                    kNumDispatchThreads,
                                    kDispatchBarrierIdx);
                            },
                            /* Publish every source's local stores */ true,
                            /* Pullers require a post-barrier grid sync */ true);

                    // Every remote source emits exactly one terminal, even for
                    // an inactive pair. In the baseline this also covers all
                    // payloads. With dispatch overlap it covers ONLY exact
                    // counts and assignment metadata; pullers acquire each
                    // source's separate payload terminal before mirror reads.
                    if (sm_idx == 0 and warp_idx == 0 and lane_idx < kGinPeerCount) {
                        const uint32_t owner_lane =
                            sym_buffer.rank_idx % kGinPeerCount;
                        const uint64_t expected_epoch =
                            *workspace.get_gin_direct_dispatch_epoch_ptr();
                        DG_GIN_TRACE_IF(true, 104u + lane_idx);
                        comm::mega_moe_gin_wait_direct_dispatch(
                            gin_transport,
                            /*context_stripe=*/ owner_lane,
                            sym_buffer.get_base_ptr(),
                            buffer.gin_workspace
                                .get_direct_dispatch_ready_ptr(
                                    /*send=*/ false, lane_idx),
                            expected_epoch);
                        DG_GIN_TRACE_IF(true, 32u + lane_idx);
                    }
                    ptx::sync_aligned(
                        kNumDispatchThreads, kDispatchBarrierIdx);

                    // Restore the legacy per-source/per-expert route and count
                    // layout locally.  This deliberately keeps the scheduler,
                    // round-robin source choice, and all later pull code exact.
                    if (sm_idx == 0) {
                        const uint32_t remote_lsa_base =
                            (1u - sym_buffer.rank_idx / kGinPeerCount) * kGinPeerCount;
                        if constexpr (kMegaMoeGinDispatchWarpScan) {
                            // Four warps restore one remote source apiece in
                            // one EP8 wave or two EP16 waves. Both 32-lane scans execute
                            // with a full mask; inactive lanes in the second
                            // expert segment contribute zero.
                            DG_STATIC_ASSERT(
                                kNumDispatchWarps == 4 and
                                kNumExpertsPerRank == 56,
                                "Dispatch warp scan requires four warps and "
                                "56 experts per rank");
                            #pragma unroll
                            for (uint32_t source_wave = 0;
                                 source_wave < kGinPeerCount / kNumDispatchWarps;
                                 ++source_wave) {
                                const uint32_t source_lane =
                                    warp_idx + source_wave *
                                                   kNumDispatchWarps;
                                const uint32_t remote_source =
                                    remote_lsa_base + source_lane;
                                const auto* packed_counts =
                                    buffer.gin_workspace
                                        .get_direct_dispatch_count_ptr(
                                            /*send=*/ false, source_lane);

                                const uint32_t first_expert = lane_idx;
                                const uint32_t first_count =
                                    static_cast<uint32_t>(
                                        packed_counts[first_expert]);
                                const uint32_t first_inclusive =
                                    math::warp_inclusive_sum(
                                        first_count, lane_idx);
                                const uint32_t first_prefix =
                                    first_inclusive - first_count;
                                const uint32_t first_total = __shfl_sync(
                                    0xffffffffu, first_inclusive, 31);

                                const uint32_t second_expert =
                                    lane_idx + 32u;
                                const uint32_t second_count =
                                    second_expert < kNumExpertsPerRank
                                        ? static_cast<uint32_t>(
                                              packed_counts[second_expert])
                                        : 0u;
                                const uint32_t second_inclusive =
                                    math::warp_inclusive_sum(
                                        second_count, lane_idx);
                                const uint32_t second_prefix =
                                    first_total + second_inclusive -
                                    second_count;

                                if constexpr (kMegaMoeGinCombineOverlap) {
                                    if (use_gin_combine_overlap) {
                                        // All 32 lanes participate, including
                                        // zero-padded experts56..63. Save the
                                        // masks during the existing count scan,
                                        // not in a new pre-submission scan.
                                        const uint32_t first_nonempty = __ballot_sync(
                                            0xffffffffu, first_count != 0);
                                        const uint32_t second_nonempty = __ballot_sync(
                                            0xffffffffu, second_count != 0);
                                        if (lane_idx == 0) {
                                            *buffer.gin_workspace.get_combine_overlap_nonempty_mask_ptr(
                                                source_lane, 0u) = first_nonempty;
                                            *buffer.gin_workspace.get_combine_overlap_nonempty_mask_ptr(
                                                source_lane, 1u) = second_nonempty;
                                        }
                                    }
                                }

                                DG_DEVICE_ASSERT(
                                    first_count <=
                                    layout::kMegaMoeGinDirectDispatchMaxTokens);
                                DG_DEVICE_ASSERT(
                                    first_prefix + first_count <=
                                    layout::kMegaMoeGinDirectDispatchMaxRoutes);
                                *workspace.get_expert_recv_count_ptr(
                                    remote_source, first_expert) = first_count;
                                if constexpr (kMegaMoeGinCombineOverlap) {
                                    if (use_gin_combine_overlap)
                                        *buffer.gin_workspace.get_combine_overlap_prefix_ptr(
                                            source_lane, first_expert) = first_prefix;
                                }
                                const auto* first_packed_routes =
                                    buffer.gin_workspace
                                        .get_direct_dispatch_route_ptr(
                                            /*send=*/ false, source_lane,
                                            first_prefix);
                                auto* first_routes =
                                    workspace.get_src_token_topk_idx_ptr(
                                        first_expert, remote_source, 0);
                                for (uint32_t slot = 0;
                                     slot < first_count;
                                     ++slot) {
                                    first_routes[slot] =
                                        first_packed_routes[slot];
                                }

                                if (second_expert < kNumExpertsPerRank) {
                                    DG_DEVICE_ASSERT(
                                        second_count <=
                                        layout::
                                            kMegaMoeGinDirectDispatchMaxTokens);
                                    DG_DEVICE_ASSERT(
                                        second_prefix + second_count <=
                                        layout::
                                            kMegaMoeGinDirectDispatchMaxRoutes);
                                    *workspace.get_expert_recv_count_ptr(
                                        remote_source, second_expert) =
                                        second_count;
                                    if constexpr (kMegaMoeGinCombineOverlap) {
                                        if (use_gin_combine_overlap)
                                            *buffer.gin_workspace.get_combine_overlap_prefix_ptr(
                                                source_lane, second_expert) = second_prefix;
                                    }
                                    const auto* second_packed_routes =
                                        buffer.gin_workspace
                                            .get_direct_dispatch_route_ptr(
                                                /*send=*/ false, source_lane,
                                                second_prefix);
                                    auto* second_routes =
                                        workspace
                                            .get_src_token_topk_idx_ptr(
                                                second_expert, remote_source,
                                                0);
                                    for (uint32_t slot = 0;
                                         slot < second_count;
                                         ++slot) {
                                        second_routes[slot] =
                                            second_packed_routes[slot];
                                    }
                                }
                                DG_GIN_TRACE_IF(lane_idx == 0, 40u + source_lane);
                            }
                        } else {
                            constexpr uint32_t kNumRemoteExpertPairs =
                                kGinPeerCount * kNumExpertsPerRank;
                            for (uint32_t pair = thread_idx;
                                 pair < kNumRemoteExpertPairs;
                                 pair += kNumDispatchThreads) {
                                const uint32_t source_lane =
                                    pair / kNumExpertsPerRank;
                                const uint32_t local_expert =
                                    pair % kNumExpertsPerRank;
                                const uint32_t remote_source =
                                    remote_lsa_base + source_lane;
                                const auto* packed_counts =
                                    buffer.gin_workspace
                                        .get_direct_dispatch_count_ptr(
                                            /*send=*/ false, source_lane);
                                const uint32_t count = static_cast<uint32_t>(
                                    packed_counts[local_expert]);
                                DG_DEVICE_ASSERT(
                                    count <= layout::
                                                 kMegaMoeGinDirectDispatchMaxTokens);
                                *workspace.get_expert_recv_count_ptr(
                                    remote_source, local_expert) = count;

                                uint32_t route_prefix = 0;
                                uint32_t nonempty_mask = 0;
                                const uint32_t expert_group = local_expert / 32u;
                                const bool mask_writer = local_expert == 31u or
                                    local_expert + 1u == kNumExpertsPerRank;
                                #pragma unroll
                                for (uint32_t prior_expert = 0;
                                     prior_expert < local_expert;
                                     ++prior_expert) {
                                    const uint32_t prior_count = static_cast<uint32_t>(
                                        packed_counts[prior_expert]);
                                    route_prefix += prior_count;
                                    if constexpr (kMegaMoeGinCombineOverlap) {
                                        // Exactly experts31/55 write one mask
                                        // per source, using count loads already
                                        // needed for their scalar prefix. No
                                        // atomic OR or additional count scan.
                                        if (use_gin_combine_overlap and mask_writer and
                                            prior_expert / 32u == expert_group and
                                            prior_count != 0)
                                            nonempty_mask |= 1u << (prior_expert % 32u);
                                    }
                                }
                                DG_DEVICE_ASSERT(
                                    route_prefix + count <= layout::
                                        kMegaMoeGinDirectDispatchMaxRoutes);
                                if constexpr (kMegaMoeGinCombineOverlap) {
                                    if (use_gin_combine_overlap) {
                                        *buffer.gin_workspace.get_combine_overlap_prefix_ptr(
                                            source_lane, local_expert) = route_prefix;
                                        if (mask_writer) {
                                            if (count != 0)
                                                nonempty_mask |= 1u << (local_expert % 32u);
                                            *buffer.gin_workspace.get_combine_overlap_nonempty_mask_ptr(
                                                source_lane, expert_group) = nonempty_mask;
                                        }
                                    }
                                }
                                const auto* packed_routes =
                                    buffer.gin_workspace
                                        .get_direct_dispatch_route_ptr(
                                            /*send=*/ false, source_lane,
                                            route_prefix);
                                auto* routes =
                                    workspace.get_src_token_topk_idx_ptr(
                                        local_expert, remote_source, 0);
                                for (uint32_t slot = 0; slot < count; ++slot)
                                    routes[slot] = packed_routes[slot];
                            }
                        }
                        __threadfence();
                    }
                    ptx::sync_aligned(
                        kNumDispatchThreads, kDispatchBarrierIdx);
                } else {
                    // The exact r75 fallback retains the paired ingress and its
                    // two-phase remote-visibility protocol.
                    __threadfence_system();
                    comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
                        workspace, sm_idx, thread_idx,
                        [=]() {
                            ptx::sync_aligned(
                                kNumDispatchThreads, kDispatchBarrierIdx);
                        });
                    if (sm_idx == 0 and warp_idx == 0) {
                        // GIN flush/wait protects local source reuse but does
                        // not itself prove that a PUT settled remotely.
                        comm::mega_moe_gin_world_barrier_all_contexts(
                            gin_transport, kGinInputPutBarrierIdx,
                            ncclGinFenceLevel::Put);
                        __threadfence_system();

                        // The paired ingress must finish its target-local fence
                        // before a third-party owner reads it over NVLink.
                        comm::mega_moe_gin_world_barrier(
                            gin_transport, kGinInputIngressReadyBarrierIdx,
                            ncclGinFenceLevel::None);
                    }
                    comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
                        workspace, sm_idx, thread_idx,
                        [=]() {
                            ptx::sync_aligned(
                                kNumDispatchThreads, kDispatchBarrierIdx);
                        });
                }

                // GIN publishes per-source cells.  The owner reconstructs the
                // original tagged sum locally because GIN has no arbitrary-
                // address fetch-add equivalent.  The scheduler wait remains
                // unchanged and also publishes completed direct unpacking.
                if (sm_idx == 0) {
                    for (uint32_t expert = thread_idx;
                         expert < kNumExpertsPerRank;
                         expert += kNumDispatchThreads) {
                        uint32_t total = 0;
                        #pragma unroll
                        for (uint32_t source = 0;
                             source < kNumRanks; ++source) {
                            total += static_cast<uint32_t>(
                                *workspace.get_expert_recv_count_ptr(
                                    source, expert));
                        }
                        const uint64_t tagged =
                            (static_cast<uint64_t>(kNumSMs * kNumRanks) << 32) |
                            total;
                        auto* ptr =
                            workspace.get_expert_recv_count_sum_ptr(expert);
                        asm volatile("st.release.gpu.global.u64 [%0], %1;"
                                     :: "l"(ptr), "l"(tagged) : "memory");
                    }
                }
                ptx::sync_aligned(
                    kNumDispatchThreads, kDispatchBarrierIdx);
                DG_GIN_TRACE_IF(sm_idx == 0 and warp_idx == 0 and lane_idx == 0, 7);
            } else {
                comm::nvlink_lsa_barrier<
                    kNumRanks, kGinPeerCount, kNumSMs, kNumDispatchThreads,
                    kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
                        workspace, sym_buffer, sm_idx, thread_idx,
                        [=]() {
                            ptx::sync_aligned(
                                kNumDispatchThreads, kDispatchBarrierIdx);
                        },
                        /* Route consensus already grid-synchronized */ false,
                        /* Pullers require a post-barrier grid sync */ true);
            }
#endif
        } else if constexpr (kUseGin) {
#ifdef DG_MEGAMOE_GIN
            comm::nvlink_lsa_barrier<
                kNumRanks, kGinPeerCount, kNumSMs, kNumDispatchThreads,
                kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
                workspace, sym_buffer, sm_idx, thread_idx,
                [=]() {
                    ptx::sync_aligned(
                        kNumDispatchThreads, kDispatchBarrierIdx);
                },
                /* Prior route grid sync covers all non-SM0 writes */ false,
                /* Pullers require a post-barrier grid sync */ true);
#endif
        } else {
            comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                                 kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
                workspace, sym_buffer, sm_idx, thread_idx,
                [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
                /* After the grid sync above, there is no more writes by other SMs (except 0) */ false,
                /* After the NVLink barrier, there is a grid sync */ true
            );
        }

        DG_GIN_TRACE_IF(warp_idx == 0 and lane_idx == 0, 5);
        // Ensure the epilogue barrier cannot run with the pull barrier
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Pull token data and SF from remote ranks into local L1 buffer
        uint32_t pull_mbarrier_phase = 0;
        const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
        const auto pull_mbarrier = &shared_storage.dispatch_barriers[warp_idx];

        // Per-rank counts for current expert (re-loaded when expert changes)
        constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
        int current_expert_idx = -1;
        uint32_t stored_rank_count[kNumRanksPerLane] = {};
        uint32_t stored_rank_prefix[kNumRanksPerLane] = {};
        uint32_t expert_start_idx = 0, expert_end_idx = 0;
        uint32_t expert_pool_block_offset = 0;

        // A whole-source payload is immutable through this launch. Cache one
        // acquire per source per pull warp, not one poll per assignment. The
        // baseline specialization eliminates this state and the waits below.
        uint32_t gin_acquired_payload_sources = 0;

        // Wait token data arrival
        scheduler.fetch_expert_recv_count();
        DG_GIN_TRACE_IF(warp_idx == 0 and lane_idx == 0, 6);

        // In the GIN path SM0/warp0 drains combine output concurrently with
        // owner pulls.  Reserving it removes a pull -> ring reuse -> epilogue
        // -> outbox reuse -> drainer cycle; the remaining global warps cover
        // every logical token with a compressed index.
        const bool use_gin_outbox =
            kUseGinOutbox and
            not use_gin_bulk_combine and
            (not kMegaMoeGinActiveFastPath or
             ptx::ld_acq(workspace.get_gin_world_active_ptr()) != 0);
        const bool is_gin_outbox_drainer =
            use_gin_outbox and sm_idx == 0 and warp_idx == 0;
        if (not is_gin_outbox_drainer) {
        constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;
        const uint32_t num_global_pull_warps =
            kNumGlobalWarps - (use_gin_outbox ? 1u : 0u);
        const uint32_t physical_global_warp_idx =
            sm_idx * kNumDispatchWarps + warp_idx;
        const uint32_t global_pull_warp_idx =
            use_gin_outbox ?
                physical_global_warp_idx - 1u : physical_global_warp_idx;
        for (uint32_t token_idx = global_pull_warp_idx; ;
             token_idx += num_global_pull_warps) {
            // Advance expert until within the range
            const int old_expert_idx = current_expert_idx;
            while (token_idx >= expert_end_idx) {
                // Preserve a compact per-source prefix in ascending local
                // expert order.  Loading counts inside this loop is required:
                // a pull warp may skip multiple empty or short experts.  Keep
                // the legacy/fallback pull path's final-expert-only load exact.
                if constexpr (kUseGin and kMegaMoeGinBulkCombine) {
                    if (use_gin_bulk_combine and current_expert_idx >= 0) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumRanksPerLane; ++i)
                            stored_rank_prefix[i] += stored_rank_count[i];
                    }
                }
                if (++ current_expert_idx >= kNumExpertsPerRank)
                    break;

                // Update pool block offset for the new expert
                expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);

                // Move start and end to the next expert
                expert_start_idx = expert_end_idx;
                expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
                if constexpr (kUseGin and kMegaMoeGinBulkCombine) {
                    if (use_gin_bulk_combine) {
                        #pragma unroll
                        for (uint32_t i = 0;
                             i < kNumRanksPerLane; ++i) {
                            const uint32_t j = i * 32 + lane_idx;
                            // TODO: this is not coalesced
                            stored_rank_count[i] = j < kNumRanks ?
                                static_cast<uint32_t>(
                                    *workspace.get_expert_recv_count_ptr(
                                        j, current_expert_idx)) : 0;
                        }
                    }
                }
            }

            // Finish all tokens
            if (current_expert_idx >= kNumExpertsPerRank)
                break;

            // The original row path only loads the final selected expert.
            // Bulk mode already loaded every traversed expert above so it can
            // retain the exact per-source prefix used as the slab row index.
            bool load_selected_expert_counts =
                old_expert_idx != current_expert_idx;
            if constexpr (kUseGin and kMegaMoeGinBulkCombine)
                load_selected_expert_counts &=
                    not use_gin_bulk_combine;
            if (load_selected_expert_counts) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++i) {
                    const uint32_t j = i * 32 + lane_idx;
                    // TODO: this is not coalesced
                    stored_rank_count[i] = j < kNumRanks ?
                        static_cast<uint32_t>(
                            *workspace.get_expert_recv_count_ptr(
                                j, current_expert_idx)) : 0;
                }
            }

            // Round-robin rank selection via iterative min-peeling
            uint32_t current_rank_in_expert_idx;
            uint32_t remaining[kNumRanksPerLane];
            #pragma unroll
            for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                remaining[i] = stored_rank_count[i];
            uint32_t offset = 0;
            uint32_t token_idx_in_expert = token_idx - expert_start_idx;
            uint32_t slot_idx = token_idx_in_expert;
            uint32_t token_idx_in_rank;
            while (true) {
                // Compute active count and min across all ranks
                // NOTES: reduce within each lane first, then warp-reduce once
                uint32_t num_actives_in_lane = 0;
                uint32_t min_in_lane = 0xffffffff;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    num_actives_in_lane += remaining[i] > 0;
                    if (remaining[i] > 0)
                        min_in_lane = cute::min(min_in_lane, remaining[i]);
                }
                const uint32_t num_active_ranks = __reduce_add_sync(0xffffffff, num_actives_in_lane);
                const uint32_t length = __reduce_min_sync(0xffffffff, min_in_lane);

                // Hit in the current round
                const uint32_t num_round_tokens = length * num_active_ranks;
                if (slot_idx < num_round_tokens) {
                    const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                    uint32_t num_seen_ranks = 0;
                    current_rank_in_expert_idx = 0;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        const uint32_t mask = __ballot_sync(0xffffffff, remaining[i] > 0);
                        const uint32_t num_active_lanes = __popc(mask);
                        if (slot_idx_in_round >= num_seen_ranks and slot_idx_in_round < num_seen_ranks + num_active_lanes)
                            current_rank_in_expert_idx = i * 32 + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                        num_seen_ranks += num_active_lanes;
                    }
                    token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                    break;
                }

                // Move into the next round
                slot_idx -= num_round_tokens;
                offset += length;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                    remaining[i] -= cute::min(remaining[i], length);
            }

            uint32_t bulk_return_idx = 0;
            if constexpr (kUseGin and kMegaMoeGinBulkCombine) {
#ifdef DG_MEGAMOE_GIN
                if (use_gin_bulk_combine and
                    not gin_transport.is_same_lsa_peer(
                        current_rank_in_expert_idx)) {
                    const uint32_t rank_group =
                        current_rank_in_expert_idx / 32u;
                    const uint32_t rank_lane =
                        current_rank_in_expert_idx % 32u;
                    const uint32_t source_prefix = __shfl_sync(
                        0xffffffffu, stored_rank_prefix[rank_group],
                        rank_lane);
                    bulk_return_idx = source_prefix + token_idx_in_rank;
                    DG_DEVICE_ASSERT(
                        bulk_return_idx <
                        layout::kMegaMoeGinBulkCombineMaxTokens * kNumTopk);
                }
#endif
            }

            // Read source token-topk index (written by remote dispatch via NVLink)
            const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
            const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
            const uint32_t src_topk_idx = src_token_topk_idx % kNumTopk;

            // Hidden bytes are divided into chunks
            constexpr uint32_t kNumChunks = kHidden / kNumBytesPerPull;
            DG_STATIC_ASSERT(kNumChunks * kNumBytesPerPull == kHidden, "kNumBytesPerPull must divide hidden");

            // TMA load token from remote rank and store into local
            const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
            const uint32_t pool_block_idx = pool_token_idx / BLOCK_M;

            // Wait for ring buffer slot to be available (previous consumer must have finished all N blocks)
            constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
            const auto l1_empty_count_target = (pool_block_idx / kNumRingBlocks) * kNumL1BlockNs;
            if (l1_empty_count_target > 0) {
                const auto empty_ptr = workspace.get_l1_empty_count_ptr(pool_block_idx % kNumRingBlocks);
                while (ptx::ld_acq(empty_ptr) < l1_empty_count_target);
            }

            // Same-LSA sources retain mapped NVLink loads.  Cross-host sources
            // point at the local source-published mirror, after which the TMA,
            // SF transpose, ring readiness, and metadata path is identical.
            void* src_base_ptr = nullptr;
            const uint32_t* source_sf_ptr = nullptr;
            const float* source_weights_ptr = nullptr;
            if constexpr (kUseGin and
                          kMegaMoeGinLocalAblationStage >= 3) {
#ifdef DG_MEGAMOE_GIN
                const bool is_same_lsa = gin_transport.is_same_lsa_peer(
                    current_rank_in_expert_idx);
                DG_DEVICE_ASSERT(
                    is_same_lsa and
                    "GIN local ablation pulled a cross-LSA source");
                if (is_same_lsa) {
                    src_base_ptr = sym_buffer.map(
                        buffer.input_token_buffer
                            .get_data_buffer(src_token_idx).get_base_ptr(),
                        current_rank_in_expert_idx);
                    source_sf_ptr = sym_buffer.map(
                        buffer.input_sf_buffer
                            .get_data_buffer(src_token_idx)
                            .template get_base_ptr<uint32_t>(),
                        current_rank_in_expert_idx);
                    source_weights_ptr = sym_buffer.map(
                        buffer.input_topk_weights_buffer
                                .get_base_ptr<float>() +
                            src_token_topk_idx,
                        current_rank_in_expert_idx);
                }
#endif
            } else if constexpr (kUseGin) {
#ifdef DG_MEGAMOE_GIN
                if (not gin_transport.is_same_lsa_peer(
                        current_rank_in_expert_idx)) {
                    const uint32_t lsa_size = static_cast<uint32_t>(
                        gin_transport.dev_comm.lsaSize);
                    if (use_gin_direct_dispatch) {
                        const uint32_t source_lane =
                            current_rank_in_expert_idx % lsa_size;
                        if constexpr (kMegaMoeGinDispatchOverlap) {
                            const uint32_t source_bit = 1u << source_lane;
                            if ((gin_acquired_payload_sources & source_bit) == 0) {
                                // Control arrival permits scheduling, not
                                // reading activation/SF/weights. The elected
                                // thread acquires the StrongVA payload terminal
                                // and hands that visibility to the full warp
                                // before any mirror load or TMA issue.
                                if (cute::elect_one_sync()) {
                                    comm::mega_moe_gin_wait_direct_dispatch(
                                        gin_transport,
                                        /*context_stripe=*/ sym_buffer.rank_idx % lsa_size,
                                        sym_buffer.get_base_ptr(),
                                        buffer.gin_workspace
                                            .get_direct_dispatch_payload_ready_ptr(
                                                source_lane),
                                        *workspace.get_gin_dispatch_payload_epoch_ptr());
                                }
                                __syncwarp();
                                gin_acquired_payload_sources |= source_bit;
                            }
                        }
                        src_base_ptr = buffer.gin_workspace
                            .get_direct_input_token_ptr(
                                source_lane, src_token_idx);
                        source_sf_ptr = static_cast<const uint32_t*>(
                            buffer.gin_workspace.get_direct_input_sf_ptr(
                                source_lane, src_token_idx));
                        source_weights_ptr = static_cast<const float*>(
                            buffer.gin_workspace
                                .get_direct_input_topk_weights_ptr(
                                    source_lane, src_token_idx)) +
                            src_topk_idx;
                    } else {
                        const uint32_t local_ingress_rank =
                            (sym_buffer.rank_idx / lsa_size) * lsa_size +
                            current_rank_in_expert_idx % lsa_size;
                        src_base_ptr = sym_buffer.map(
                            buffer.gin_workspace.published_input_token_buffer
                                .get_data_buffer(src_token_idx).get_base_ptr(),
                            local_ingress_rank);
                        source_sf_ptr = sym_buffer.map(
                            buffer.gin_workspace.published_input_sf_buffer
                                .get_data_buffer(src_token_idx)
                                .template get_base_ptr<uint32_t>(),
                            local_ingress_rank);
                        source_weights_ptr = sym_buffer.map(
                            buffer.gin_workspace
                                    .published_input_topk_weights_buffer
                                    .template get_base_ptr<float>() +
                                src_token_topk_idx,
                            local_ingress_rank);
                    }
                } else {
                    src_base_ptr = sym_buffer.map(
                        buffer.input_token_buffer.get_data_buffer(src_token_idx)
                            .get_base_ptr(),
                        current_rank_in_expert_idx);
                    source_sf_ptr = sym_buffer.map(
                        buffer.input_sf_buffer.get_data_buffer(src_token_idx)
                            .template get_base_ptr<uint32_t>(),
                        current_rank_in_expert_idx);
                    source_weights_ptr = sym_buffer.map(
                        buffer.input_topk_weights_buffer.get_base_ptr<float>() +
                            src_token_topk_idx,
                        current_rank_in_expert_idx);
                }
#endif
            } else {
                src_base_ptr = sym_buffer.map(
                    buffer.input_token_buffer.get_data_buffer(src_token_idx)
                        .get_base_ptr(),
                    current_rank_in_expert_idx);
                source_sf_ptr = sym_buffer.map(
                    buffer.input_sf_buffer.get_data_buffer(src_token_idx)
                        .template get_base_ptr<uint32_t>(),
                    current_rank_in_expert_idx);
                source_weights_ptr = sym_buffer.map(
                    buffer.input_topk_weights_buffer.get_base_ptr<float>() +
                        src_token_topk_idx,
                    current_rank_in_expert_idx);
            }

            const auto dst_base_ptr = buffer.l1_token_buffer.get_data_buffer(pool_token_idx % kNumRingTokens).get_base_ptr();
            const auto issue_and_wait_pull_store = [&](const uint32_t& i) {
                ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);
                ptx::tma_store_1d(
                    math::advance_ptr(dst_base_ptr, i * kNumBytesPerPull),
                    pull_buffer.get_base_ptr(), kNumBytesPerPull
                );
                cute::tma_store_arrive();
                ptx::tma_store_wait<0>();
            };
            if (cute::elect_one_sync()) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumChunks; ++ i) {
                    ptx::tma_load_1d(
                        pull_buffer.get_base_ptr(),
                        math::advance_ptr(src_base_ptr, i * kNumBytesPerPull),
                        pull_mbarrier, kNumBytesPerPull
                    );
                    ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kNumBytesPerPull);
                    i != (kNumChunks - 1) ? issue_and_wait_pull_store(i) : void();
                }
            }
            __syncwarp();

            // Load and store SF (overlaps with last chunk's TMA load from remote)
            constexpr uint32_t kNumSFUint32 = kHidden / 128;
            DG_STATIC_ASSERT(kNumSFUint32 > 0 and kHidden % 128 == 0, "Invalid SF");
            const auto local_sf_ptr = buffer.l1_sf_buffer.get_base_ptr<uint32_t>();
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t token_idx_in_block = token_idx_in_expert % BLOCK_M;
            const auto sf_ring_token_idx = ring_block_idx * SF_BLOCK_M +
                transform_sf_token_idx(token_idx_in_block);
            #pragma unroll
            for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFUint32, 32u); ++ i) {
                const uint32_t j = i * 32 + lane_idx;
                if (j < kNumSFUint32)
                    local_sf_ptr[j * kNumSFRingTokens + sf_ring_token_idx] =
                        source_sf_ptr[j];
            }
            __syncwarp();

            // Store weights and metadata
            if (cute::elect_one_sync()) {
                // Load weights
                const auto weight = *source_weights_ptr;
                *buffer.l1_topk_weights_buffer.get_data_buffer(pool_token_idx % kNumRingTokens).template get_base_ptr<float>() = weight;

                // Write source metadata for combine write-back (logical pool token)
                *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                    {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                if constexpr (kUseGin and kMegaMoeGinBulkCombine) {
#ifdef DG_MEGAMOE_GIN
                    if (use_gin_bulk_combine and
                        not gin_transport.is_same_lsa_peer(
                            current_rank_in_expert_idx)) {
                        *buffer.gin_workspace
                             .get_bulk_combine_return_index_ptr(pool_token_idx) =
                            bulk_return_idx;
                    }
#endif
                }

                // Complete last chunk's store
                issue_and_wait_pull_store(kNumChunks - 1);
                const bool is_last_token = (token_idx == expert_end_idx - 1);
                ptx::red_add_rel(
                    workspace.get_l1_full_count_ptr(pool_block_idx % kNumRingBlocks), 
                    is_last_token ? BLOCK_M - (token_idx_in_expert % BLOCK_M) : 1u
                );
            }
            __syncwarp();
        }
        }

        DG_GIN_TRACE_IF(lane_idx == 0, 48u + warp_idx);
        if constexpr (kUseGin and kMegaMoeGinCombineOverlap) {
#ifdef DG_MEGAMOE_GIN
            // Reuse a dispatch warp only AFTER all its pulls. Producers do
            // not wait for send credits or reuse these immutable output slabs;
            // all readiness publications precede their end handoff. Thus the
            // drainer may wait here without creating a producer/handoff cycle.
            if (use_gin_combine_overlap and sm_idx == 0 and warp_idx == 0) {
                constexpr uint32_t kNumL2Fragments = L2_SHAPE_N / BLOCK_N;
                constexpr uint32_t kMaxReadyExpertsPerPut = 8u;
                constexpr uint32_t kNumExpertGroups =
                    math::constexpr_ceil_div(kNumExpertsPerRank, 32u);
                DG_STATIC_ASSERT(kNumExpertsPerRank ==
                    layout::kMegaMoeGinCombineOverlapNumExperts,
                    "Expert-ready combine requires the fixed 56 local experts");
                const uint32_t lsa_size =
                    static_cast<uint32_t>(gin_transport.dev_comm.lsaSize);
                DG_DEVICE_ASSERT(lsa_size == kGinPeerCount);
                const uint32_t remote_base =
                    (1u - sym_buffer.rank_idx / lsa_size) * lsa_size;
                const uint32_t owner_lane = sym_buffer.rank_idx % lsa_size;
                const bool peer_lane = lane_idx < lsa_size;
                const uint32_t peer = remote_base + lane_idx;
                uint32_t sent_records = 0;
                uint32_t expected_records = 0;
                uint32_t expected_fragments[kNumExpertGroups];
                bool discovered[kNumExpertGroups];
                uint32_t ready_masks[kNumExpertGroups];
                uint32_t pending_first = 0;
                uint32_t pending_second = 0;
                if constexpr (kMegaMoeGinStrongVACombineTerminal) {
                    if (peer_lane) {
                        // This exact count is immutable after input consensus.
                        // Queue it before every dynamically ordered payload PUT
                        // on the same context/peer. The actual last submitted
                        // span (or empty-pair signal) later closes the chain.
                        #pragma unroll
                        for (uint32_t expert = 0;
                             expert < kNumExpertsPerRank; ++expert)
                            expected_records += static_cast<uint32_t>(
                                *workspace.get_expert_recv_count_ptr(
                                    peer, expert));
                        constexpr uint32_t kBulkCapacity =
                            layout::kMegaMoeGinBulkCombineMaxTokens * kNumTopk;
                        DG_DEVICE_ASSERT(expected_records <= kBulkCapacity);
                        auto* local_packet = buffer.gin_workspace
                            .get_bulk_combine_packet_ptr(
                                /*send=*/ true, lane_idx);
                        *static_cast<uint32_t*>(local_packet) = expected_records;
                        __threadfence_system();
                        comm::mega_moe_gin_put_bulk_combine_header_async(
                            gin_transport, peer, /*context_stripe=*/ 0u,
                            sym_buffer.get_base_ptr(), local_packet,
                            buffer.gin_workspace.get_bulk_combine_packet_ptr(
                                /*send=*/ false, owner_lane),
                            /*diagnostic_peer_lane=*/ lane_idx);
                    }
                    __syncwarp();
                }
                // Counts are exact, published before pulls, and remain live
                // until the existing second dispatch/epilogue handoff. Read
                // lane-owned cells directly: no varying-index scheduler query
                // or row/metadata reconstruction belongs in this progress loop.
                // Each peer loads just its two saved nonempty masks, avoiding
                // a new 56-expert setup scan before the first submission.
                #pragma unroll
                for (uint32_t group = 0; group < kNumExpertGroups; ++group) {
                    const uint32_t expert = group * 32u + lane_idx;
                    uint32_t target = 0;
                    if (expert < kNumExpertsPerRank) {
                        const uint32_t tokens = static_cast<uint32_t>(
                            *workspace.get_expert_recv_count_sum_ptr(expert));
                        target = math::ceil_div(tokens, BLOCK_M) * kNumL2Fragments;
                    }
                    expected_fragments[group] = target;
                    discovered[group] = target == 0;
                    ready_masks[group] = 0;
                    const uint32_t nonempty = peer_lane
                        ? *buffer.gin_workspace.get_combine_overlap_nonempty_mask_ptr(
                              lane_idx, group)
                        : 0u;
                    if (group == 0u) {
                        pending_first = nonempty;
                    } else {
                        DG_DEVICE_ASSERT((nonempty >> (kNumExpertsPerRank - 32u)) == 0);
                        pending_second = nonempty;
                    }
                }

                if constexpr (kMegaMoeGinCombineOwnerWaves == 0) {
                // Every lane stays in the uniform outer loop, including idle
                // peers and lanes8..31 which discover the other experts. Peer
                // lanes may choose DIFFERENT expert spans in the same round.
                while (__any_sync(0xffffffffu, (pending_first | pending_second) != 0)) {
                    // Parallel readiness probes for experts0..31 and32..55.
                    // Discovery is monotonic and common to the warp; sending
                    // to one peer never retires readiness for another peer.
                    // Producers never wait for the NIC or reuse these slabs.
                    #pragma unroll
                    for (uint32_t group = 0; group < kNumExpertGroups; ++group) {
                        const uint32_t expert = group * 32u + lane_idx;
                        bool newly_ready = false;
                        if (not discovered[group]) {
                            const uint32_t completed =
                                comm::mega_moe_gin_combine_ready_acquire(
                                    buffer.gin_workspace.get_combine_overlap_ready_ptr(expert));
                            DG_DEVICE_ASSERT(completed <= expected_fragments[group]);
                            newly_ready = completed == expected_fragments[group];
                            discovered[group] = newly_ready;
                        }
                        ready_masks[group] |= __ballot_sync(0xffffffffu, newly_ready);
                    }

                    // Freeze this round's ready snapshot. Coalesce at most
                    // eight already-ready nonempty experts per peer; never
                    // wait for future readiness or keep a batch across polls.
                    uint32_t available_first = pending_first & ready_masks[0];
                    uint32_t available_second = pending_second & ready_masks[1];
                    uint32_t accepted_first = 0;
                    uint32_t accepted_second = 0;
                    uint32_t batch_prefix = 0;
                    uint32_t batch_records = 0;
                    bool can_extend = peer_lane and
                        (available_first | available_second) != 0;
                    for (uint32_t stage = 0; stage < kMaxReadyExpertsPerPut and
                         __any_sync(0xffffffffu, can_extend); ++stage) {
                        const uint32_t selected_group = available_first != 0 ? 0u : 1u;
                        const uint32_t selected_mask = can_extend
                            ? (available_first != 0 ? available_first : available_second)
                            : 0u;
                        const uint32_t selected_lane = selected_mask != 0
                            ? static_cast<uint32_t>(__ffs(selected_mask) - 1) : 0u;
                        // Every lane participates, even when its batch ended.
                        // Gather fixed groups separately: one varying-group
                        // shuffle would use the SOURCE lane's selected group.
                        const uint32_t first_target = __shfl_sync(
                            0xffffffffu, expected_fragments[0], selected_lane);
                        const uint32_t second_target = __shfl_sync(
                            0xffffffffu, expected_fragments[1], selected_lane);
                        const uint32_t target = selected_group == 0u
                            ? first_target : second_target;
                        const uint32_t expert = selected_group * 32u + selected_lane;
                        uint32_t count = 0;
                        uint32_t prefix = 0;
                        bool accept = false;
                        if (can_extend) {
                            DG_DEVICE_ASSERT(expert < kNumExpertsPerRank and target != 0);
                            count = static_cast<uint32_t>(
                                *workspace.get_expert_recv_count_ptr(peer, expert));
                            prefix = *buffer.gin_workspace.get_combine_overlap_prefix_ptr(
                                lane_idx, expert);
                            DG_DEVICE_ASSERT(count != 0 and prefix + count <=
                                layout::kMegaMoeGinBulkCombineMaxTokens * kNumTopk);
                            // Zero-count experts occupy no records and may be
                            // skipped. An unready or previously sent positive
                            // span leaves a gap, which must never be bridged.
                            accept = batch_records == 0 or
                                prefix == batch_prefix + batch_records;
                            if (not accept)
                                can_extend = false;
                        }
#if DG_MEGAMOE_GIN_DIAGNOSTICS
                        const uint32_t accepting_lanes = __ballot_sync(0xffffffffu, accept);
                        // Includes first-batch lookahead in99-to-first-issue,
                        // even when peer lane0 itself has nothing to send.
                        DG_GIN_TRACE_FIRST_IF(lane_idx == 0 and accepting_lanes != 0, 99);
#endif
                        if (accept) {
                            // The SAME issuer acquires ALL M×N contributors
                            // for EVERY included expert before the one PUT's
                            // required-system publication. Discovery alone
                            // never substitutes for these per-issuer acquires.
                            uint32_t completed;
                            do {
                                completed = comm::mega_moe_gin_combine_ready_acquire(
                                    buffer.gin_workspace.get_combine_overlap_ready_ptr(expert));
                                DG_DEVICE_ASSERT(completed <= target);
                            } while (completed != target);
                            if (batch_records == 0)
                                batch_prefix = prefix;
                            batch_records += count;
                            const uint32_t bit = 1u << selected_lane;
                            if (selected_group == 0u) {
                                accepted_first |= bit;
                                available_first &= ~bit;
                            } else {
                                accepted_second |= bit;
                                available_second &= ~bit;
                            }
                            can_extend = (available_first | available_second) != 0;
                        }
                    }

                    if (batch_records != 0) {
                        DG_DEVICE_ASSERT(peer_lane and batch_prefix + batch_records <=
                            layout::kMegaMoeGinBulkCombineMaxTokens * kNumTopk);
                        // One closed Default PUT; no aggregate flag or new
                        // context. A singleton is sent as soon as this bounded
                        // ready-only planning finishes, with no batch-fill wait.
                        const uint32_t remaining_first =
                            pending_first & ~accepted_first;
                        const uint32_t remaining_second =
                            pending_second & ~accepted_second;
                        if constexpr (kMegaMoeGinStrongVACombineTerminal) {
                            DG_DEVICE_ASSERT(
                                sent_records + batch_records <= expected_records);
                            const bool is_final_submission =
                                sent_records + batch_records == expected_records;
                            // Readiness order is not address order. The terminal
                            // belongs to whichever submitted span exhausts the
                            // exact record count and both pending masks.
                            DG_DEVICE_ASSERT(
                                is_final_submission ==
                                ((remaining_first | remaining_second) == 0));
                            if (is_final_submission) {
                                comm::mega_moe_gin_put_bulk_combine_terminal_span(
                                    gin_transport, peer,
                                    /*context_stripe=*/ 0u,
                                    sym_buffer.get_base_ptr(),
                                    buffer.gin_workspace
                                        .get_bulk_combine_record_ptr(
                                            /*send=*/ true, lane_idx,
                                            batch_prefix),
                                    buffer.gin_workspace
                                        .get_bulk_combine_record_ptr(
                                            /*send=*/ false, owner_lane,
                                            batch_prefix),
                                    buffer.gin_workspace
                                        .get_combine_terminal_signal_ptr(
                                            owner_lane),
                                    batch_records *
                                        buffer.gin_workspace.bulk_record_bytes,
                                    /*diagnostic_peer_lane=*/ lane_idx);
                            } else {
                                comm::mega_moe_gin_put_bulk_combine_span(
                                    gin_transport, peer,
                                    /*context_stripe=*/ 0u,
                                    sym_buffer.get_base_ptr(),
                                    buffer.gin_workspace
                                        .get_bulk_combine_record_ptr(
                                            /*send=*/ true, lane_idx,
                                            batch_prefix),
                                    buffer.gin_workspace
                                        .get_bulk_combine_record_ptr(
                                            /*send=*/ false, owner_lane,
                                            batch_prefix),
                                    batch_records *
                                        buffer.gin_workspace.bulk_record_bytes,
                                    /*diagnostic_peer_lane=*/ lane_idx);
                            }
                        } else {
                            comm::mega_moe_gin_put_bulk_combine_span(
                                gin_transport, peer,
                                /*context_stripe=*/ 0u,
                                sym_buffer.get_base_ptr(),
                                buffer.gin_workspace
                                    .get_bulk_combine_record_ptr(
                                        /*send=*/ true, lane_idx,
                                        batch_prefix),
                                buffer.gin_workspace
                                    .get_bulk_combine_record_ptr(
                                        /*send=*/ false, owner_lane,
                                        batch_prefix),
                                batch_records *
                                    buffer.gin_workspace.bulk_record_bytes,
                                /*diagnostic_peer_lane=*/ lane_idx);
                        }
                        sent_records += batch_records;
                        // Keep pending state live until submission returns.
                        // Scalar masks avoid a varying-index local-array store.
                        pending_first = remaining_first;
                        pending_second = remaining_second;
                    }
                    __syncwarp();
                }
                } else {
                    // Divide the fixed 56 owner experts into equal, contiguous
                    // ranges.  Each source-peer lane owns an independent pending
                    // range mask: a nonempty range is submitted exactly once as
                    // soon as every expert contributing records to that peer is
                    // complete. Different peers and ranges may therefore issue
                    // in different (including non-address) orders.
                    constexpr uint32_t kOwnerWaveDivisor =
                        kMegaMoeGinCombineOwnerWaves == 0
                            ? 1u : kMegaMoeGinCombineOwnerWaves;
                    DG_STATIC_ASSERT(
                        kNumRanks == 8 and kNumExperts == 448 and
                        kGinPeerCount == 4 and kNumExpertsPerRank == 56 and
                        kNumExpertsPerRank % kOwnerWaveDivisor == 0,
                        "Owner waves require EP8/E448 with 56 evenly "
                        "partitioned experts");
                    constexpr uint32_t kExpertsPerOwnerWave =
                        kNumExpertsPerRank / kOwnerWaveDivisor;
                    const auto get_owner_wave_group_mask = [=](
                            const uint32_t& owner_wave,
                            const uint32_t& group) {
                        const uint32_t range_begin =
                            owner_wave * kExpertsPerOwnerWave;
                        const uint32_t range_end =
                            range_begin + kExpertsPerOwnerWave;
                        const uint32_t group_begin = group * 32u;
                        const uint32_t group_end = group_begin + 32u;
                        const uint32_t begin = range_begin > group_begin
                            ? range_begin : group_begin;
                        const uint32_t end = range_end < group_end
                            ? range_end : group_end;
                        if (begin >= end)
                            return 0u;
                        const uint32_t width = end - begin;
                        const uint32_t low_bits = width == 32u
                            ? 0xffffffffu : (1u << width) - 1u;
                        return low_bits << (begin - group_begin);
                    };
                    uint32_t pending_owner_waves = 0;
                    if (peer_lane) {
                        #pragma unroll
                        for (uint32_t owner_wave = 0;
                             owner_wave < kMegaMoeGinCombineOwnerWaves;
                             ++owner_wave) {
                            const bool nonempty_wave =
                                (pending_first &
                                 get_owner_wave_group_mask(owner_wave, 0u)) != 0 or
                                (pending_second &
                                 get_owner_wave_group_mask(owner_wave, 1u)) != 0;
                            if (nonempty_wave)
                                pending_owner_waves |= 1u << owner_wave;
                        }
                    }

                    while (__any_sync(
                            0xffffffffu, pending_owner_waves != 0)) {
                        // Preserve the r4 warp-parallel monotonic readiness
                        // discovery.  A peer-specific range becomes eligible
                        // only when its complete nonempty expert subset is in
                        // these common ready masks.
                        #pragma unroll
                        for (uint32_t group = 0;
                             group < kNumExpertGroups; ++group) {
                            const uint32_t expert = group * 32u + lane_idx;
                            bool newly_ready = false;
                            if (not discovered[group]) {
                                const uint32_t completed =
                                    comm::mega_moe_gin_combine_ready_acquire(
                                        buffer.gin_workspace
                                            .get_combine_overlap_ready_ptr(
                                                expert));
                                DG_DEVICE_ASSERT(
                                    completed <= expected_fragments[group]);
                                newly_ready =
                                    completed == expected_fragments[group];
                                discovered[group] = newly_ready;
                            }
                            ready_masks[group] |= __ballot_sync(
                                0xffffffffu, newly_ready);
                        }

                        uint32_t ready_owner_waves = 0;
                        if (peer_lane) {
                            #pragma unroll
                            for (uint32_t owner_wave = 0;
                                 owner_wave < kMegaMoeGinCombineOwnerWaves;
                                 ++owner_wave) {
                                const uint32_t owner_wave_bit =
                                    1u << owner_wave;
                                if ((pending_owner_waves & owner_wave_bit) == 0)
                                    continue;
                                const uint32_t contributing_first =
                                    pending_first &
                                    get_owner_wave_group_mask(owner_wave, 0u);
                                const uint32_t contributing_second =
                                    pending_second &
                                    get_owner_wave_group_mask(owner_wave, 1u);
                                const bool all_contributing_experts_ready =
                                    (contributing_first & ~ready_masks[0]) == 0 and
                                    (contributing_second & ~ready_masks[1]) == 0;
                                if (all_contributing_experts_ready)
                                    ready_owner_waves |= owner_wave_bit;
                            }
                        }

                        const bool has_ready_owner_wave =
                            ready_owner_waves != 0;
#if DG_MEGAMOE_GIN_DIAGNOSTICS
                        const uint32_t ready_peer_lanes = __ballot_sync(
                            0xffffffffu, has_ready_owner_wave);
                        DG_GIN_TRACE_FIRST_IF(
                            lane_idx == 0 and ready_peer_lanes != 0, 99);
#endif
                        if (has_ready_owner_wave) {
                            const uint32_t owner_wave =
                                static_cast<uint32_t>(
                                    __ffs(ready_owner_waves) - 1);
                            const uint32_t owner_wave_bit = 1u << owner_wave;
                            const uint32_t expert_begin =
                                owner_wave * kExpertsPerOwnerWave;
                            const uint32_t expert_end =
                                expert_begin + kExpertsPerOwnerWave;
                            uint32_t batch_prefix = 0;
                            uint32_t batch_records = 0;
                            for (uint32_t expert = expert_begin;
                                 expert < expert_end; ++expert) {
                                const uint32_t count = static_cast<uint32_t>(
                                    *workspace.get_expert_recv_count_ptr(
                                        peer, expert));
                                if (count == 0)
                                    continue;
                                const uint32_t prefix =
                                    *buffer.gin_workspace
                                         .get_combine_overlap_prefix_ptr(
                                             lane_idx, expert);
                                const uint32_t target =
                                    math::ceil_div(static_cast<uint32_t>(
                                        *workspace
                                             .get_expert_recv_count_sum_ptr(
                                                 expert)), BLOCK_M) *
                                    kNumL2Fragments;
                                DG_DEVICE_ASSERT(
                                    target != 0 and
                                    prefix + count <=
                                        layout::kMegaMoeGinBulkCombineMaxTokens *
                                            kNumTopk);
                                uint32_t completed;
                                do {
                                    completed =
                                        comm::mega_moe_gin_combine_ready_acquire(
                                            buffer.gin_workspace
                                                .get_combine_overlap_ready_ptr(
                                                    expert));
                                    DG_DEVICE_ASSERT(completed <= target);
                                } while (completed != target);
                                if (batch_records == 0) {
                                    batch_prefix = prefix;
                                } else {
                                    DG_DEVICE_ASSERT(
                                        prefix == batch_prefix + batch_records);
                                }
                                batch_records += count;
                            }
                            DG_DEVICE_ASSERT(batch_records != 0);
                            const uint32_t remaining_owner_waves =
                                pending_owner_waves & ~owner_wave_bit;
                            DG_DEVICE_ASSERT(
                                sent_records + batch_records <=
                                    expected_records);
                            const bool is_final_submission =
                                remaining_owner_waves == 0;
                            DG_DEVICE_ASSERT(
                                is_final_submission ==
                                (sent_records + batch_records ==
                                 expected_records));
                            if (is_final_submission) {
                                // The terminal follows submission order, not
                                // expert-address order: StrongVA settles every
                                // preceding same-context/peer range and header.
                                comm::mega_moe_gin_put_bulk_combine_terminal_span(
                                    gin_transport, peer,
                                    /*context_stripe=*/ 0u,
                                    sym_buffer.get_base_ptr(),
                                    buffer.gin_workspace
                                        .get_bulk_combine_record_ptr(
                                            /*send=*/ true, lane_idx,
                                            batch_prefix),
                                    buffer.gin_workspace
                                        .get_bulk_combine_record_ptr(
                                            /*send=*/ false, owner_lane,
                                            batch_prefix),
                                    buffer.gin_workspace
                                        .get_combine_terminal_signal_ptr(
                                            owner_lane),
                                    batch_records *
                                        buffer.gin_workspace.bulk_record_bytes,
                                    /*diagnostic_peer_lane=*/ lane_idx);
                            } else {
                                comm::mega_moe_gin_put_bulk_combine_span(
                                    gin_transport, peer,
                                    /*context_stripe=*/ 0u,
                                    sym_buffer.get_base_ptr(),
                                    buffer.gin_workspace
                                        .get_bulk_combine_record_ptr(
                                            /*send=*/ true, lane_idx,
                                            batch_prefix),
                                    buffer.gin_workspace
                                        .get_bulk_combine_record_ptr(
                                            /*send=*/ false, owner_lane,
                                            batch_prefix),
                                    batch_records *
                                        buffer.gin_workspace.bulk_record_bytes,
                                    /*diagnostic_peer_lane=*/ lane_idx);
                            }
                            sent_records += batch_records;
                            pending_owner_waves = remaining_owner_waves;
                        }
                        __syncwarp();
                    }
                }

                DG_GIN_TRACE_IF(lane_idx == 0, 100);
                // Preserve the old scratch partition with one writer per
                // expert. These entries now mean ALL remote submissions are
                // queued (vacuously true for local/empty experts), not that
                // computation or remote visibility has completed. They are
                // bookkeeping only, never a producer/receiver readiness flag.
                if (not use_gin_combine_owner_slot_ready_this_launch()) {
                    #pragma unroll
                    for (uint32_t group = 0; group < kNumExpertGroups; ++group) {
                        const uint32_t expert = group * 32u + lane_idx;
                        if (expert < kNumExpertsPerRank)
                            *buffer.gin_workspace.get_combine_overlap_sent_ptr(expert) = 1;
                    }
                }
                if (peer_lane) {
                    if constexpr (not kMegaMoeGinStrongVACombineTerminal) {
                        for (uint32_t expert = 0;
                             expert < kNumExpertsPerRank; ++expert)
                            expected_records += static_cast<uint32_t>(
                                *workspace.get_expert_recv_count_ptr(
                                    peer, expert));
                    }
                    DG_DEVICE_ASSERT(sent_records == expected_records);
                    if constexpr (kMegaMoeGinStrongVACombineTerminal) {
                        auto* remote_signal = buffer.gin_workspace
                            .get_combine_terminal_signal_ptr(owner_lane);
                        if (expected_records == 0) {
                            comm::mega_moe_gin_signal_bulk_combine_terminal(
                                gin_transport, peer,
                                /*context_stripe=*/ 0u,
                                sym_buffer.get_base_ptr(), remote_signal);
                        }
                        // Do not gate compute on sender-local completion. The
                        // exact header and record sources remain immutable
                        // until the same peer chain is flushed after dispatch
                        // cleanup, immediately before its final world barrier.
                    }
                }
                // The accepted path still defers completion to its late header.
                // StrongVA mode instead defers each same-context/peer completion
                // until cleanup; receiver visibility remains a distinct wait.
                __syncwarp();
            }
#endif
        }
        if constexpr (kUseGin and kMegaMoeGinDispatchOverlap) {
#ifdef DG_MEGAMOE_GIN
            // All active peer chains were posted before the count rendezvous.
            // Keep the input/control send storage alive, then complete each
            // chain once here before cleanup may overwrite it. Crucially,
            // no outbound flush gates count publication or compute startup.
            if (use_gin_direct_dispatch and sm_idx == 1 and warp_idx == 0 and
                lane_idx < kGinPeerCount) {
                // Retire every inbound terminal, including zero-assignment
                // peers which no pull warp needed to read. This preserves the
                // baseline's all-input-context remote completion before the
                // later context-1-only combine rendezvous and graph reuse.
                comm::mega_moe_gin_wait_direct_dispatch(
                    gin_transport,
                    /*context_stripe=*/ sym_buffer.rank_idx % kGinPeerCount,
                    sym_buffer.get_base_ptr(),
                    buffer.gin_workspace
                        .get_direct_dispatch_payload_ready_ptr(lane_idx),
                    *workspace.get_gin_dispatch_payload_epoch_ptr());
                const uint32_t peer =
                    (1u - sym_buffer.rank_idx / kGinPeerCount) * kGinPeerCount + lane_idx;
                ncclGinRequest_t request{};
                comm::mega_moe_gin_flush_data_peer_async(
                    gin_transport, peer, lane_idx, &request);
                comm::mega_moe_gin_wait_data_peer(
                    gin_transport, lane_idx, request);
                DG_GIN_TRACE_IF(true, 24u + lane_idx);
            }
#endif
        }
        if constexpr (kUseGinOutbox) {
#ifdef DG_MEGAMOE_GIN
            // One warp drains the bounded block outbox in logical pool order.
            // Keep the scheduler queries warp-converged because their cached
            // expert counts are distributed across lanes. The active peer
            // lanes each own one remote peer, so Device API PUT construction
            // and independent peer QPs progress in parallel instead of one
            // lane serially issuing every remote row.  Each peer lane carries
            // completion batches across up to outbox_depth consecutive blocks
            // that are already ready.  The warp releases the current wave
            // before waiting for an unready block, so producers always regain
            // their slots and no scheduler dependency cycle is introduced.
            if (is_gin_outbox_drainer) {
                constexpr uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N;
                constexpr uint32_t kCombineRowBytes =
                    kHidden * sizeof(nv_bfloat16);
                constexpr uint32_t kNumRemotePeers =
                    kGinPeerCount;
                uint32_t pool_block_idx = 0;

                const uint32_t lsa_size = static_cast<uint32_t>(
                    gin_transport.dev_comm.lsaSize);
                const uint32_t remote_lsa_base =
                    (1u - sym_buffer.rank_idx / lsa_size) * lsa_size;
                const bool is_remote_peer_lane = lane_idx < kNumRemotePeers;
                const uint32_t remote_peer = remote_lsa_base + lane_idx;
                uint32_t pending_rows = 0;
                uint32_t active_context = 0;
                bool completion_outstanding
                    [layout::kMegaMoeGinNumDataContexts] = {};
                ncclGinRequest_t completion_requests
                    [layout::kMegaMoeGinNumDataContexts] = {};
                uint32_t wave_start = 0;
                uint32_t wave_blocks = 0;
                uint32_t issue_group_start = 0;
                uint32_t issue_group_blocks = 0;
                uint64_t issue_group_valid_ms = 0;
                DG_STATIC_ASSERT(BLOCK_M <= 255,
                                 "GIN issue-wave valid_m must fit in one byte");

                const auto release_wave = [&]() {
                    // Ring every residual peer batch first so all peer QPs can
                    // overlap, then join the warp before publishing slot reuse.
                    if (is_remote_peer_lane) {
                        if (pending_rows > 0) {
                            const uint32_t context_stripe = active_context;
                            DG_DEVICE_ASSERT(
                                not completion_outstanding[context_stripe]);
                            comm::mega_moe_gin_flush_data_peer_async(
                                gin_transport, remote_peer,
                                context_stripe,
                                &completion_requests[context_stripe]);
                            completion_outstanding[context_stripe] = true;
                            pending_rows = 0;
                        }
                        #pragma unroll
                        for (uint32_t context_stripe = 0;
                             context_stripe < kGinPeerCount;
                             ++context_stripe) {
                            if (completion_outstanding[context_stripe]) {
                                comm::mega_moe_gin_wait_data_peer(
                                    gin_transport, context_stripe,
                                    completion_requests[context_stripe]);
                                completion_outstanding[context_stripe] = false;
                            }
                        }
                    }

                    __syncwarp();
                    if (lane_idx == 0) {
                        __threadfence();
                        for (uint32_t block = 0; block < wave_blocks; ++block) {
                            const uint32_t completed_pool_block = wave_start + block;
                            const uint32_t completed_slot =
                                completed_pool_block % gin_transport.outbox_depth;
                            const uint32_t completed_generation =
                                completed_pool_block / gin_transport.outbox_depth;
                            atomicExch(
                                buffer.gin_workspace
                                    .get_outbox_empty_count_ptr(completed_slot),
                                completed_generation + 1);
                        }
                    }
                    __syncwarp();
                    wave_start += wave_blocks;
                    wave_blocks = 0;
                };

                const auto drain_issue_group = [&]() {
                    DG_DEVICE_ASSERT(issue_group_blocks > 0);
                    DG_DEVICE_ASSERT(
                        issue_group_start == wave_start + wave_blocks);
                    uint32_t group_block = 0;
                    uint32_t row = 0;
                    while (true) {
                        // Every peer lane independently advances across all
                        // ready blocks in this issue group.  The full-warp
                        // ballot forms converged rounds in which rows from
                        // different blocks and peers enter GIN together.
                        while (is_remote_peer_lane and
                               group_block < issue_group_blocks) {
                            const uint32_t valid_m = static_cast<uint32_t>(
                                (issue_group_valid_ms >>
                                 (group_block * 8u)) & 0xffu);
                            while (row < valid_m) {
                                const uint32_t logical_pool_block =
                                    issue_group_start + group_block;
                                const auto src_metadata =
                                    *workspace.get_token_src_metadata_ptr(
                                        logical_pool_block * BLOCK_M + row);
                                if (src_metadata.rank_idx == remote_peer)
                                    break;
                                ++row;
                            }
                            if (row < valid_m)
                                break;
                            ++group_block;
                            row = 0;
                        }

                        const bool has_peer_row =
                            is_remote_peer_lane and
                            group_block < issue_group_blocks;
                        const uint32_t peer_row_mask = __ballot_sync(
                            0xffffffffu, has_peer_row);
                        if (peer_row_mask == 0)
                            break;

                        if (has_peer_row) {
                            const uint32_t logical_pool_block =
                                issue_group_start + group_block;
                            const uint32_t slot = logical_pool_block %
                                gin_transport.outbox_depth;
                            const auto src_metadata =
                                *workspace.get_token_src_metadata_ptr(
                                    logical_pool_block * BLOCK_M + row);
                            const uint32_t dst_rank_idx = src_metadata.rank_idx;
                            DG_DEVICE_ASSERT(dst_rank_idx == remote_peer);

                            const uint32_t context_stripe = active_context;
                            if (pending_rows == 0 and
                                completion_outstanding[context_stripe]) {
                                comm::mega_moe_gin_wait_data_peer(
                                    gin_transport, context_stripe,
                                    completion_requests[context_stripe]);
                                completion_outstanding[context_stripe] = false;
                            }
                            const auto local_source =
                                buffer.gin_workspace
                                    .get_combine_outbox_row_ptr(slot, row);
                            const auto remote_destination =
                                buffer.combine_token_buffer
                                    .get_rank_buffer(src_metadata.topk_idx)
                                    .get_data_buffer(src_metadata.token_idx)
                                    .get_base_ptr();
                            comm::mega_moe_gin_put_row_chunks(
                                gin_transport, dst_rank_idx, context_stripe,
                                sym_buffer.get_base_ptr(), local_source,
                                remote_destination, kCombineRowBytes);
                            if (++pending_rows ==
                                gin_transport.completion_batch) {
                                comm::mega_moe_gin_flush_data_peer_async(
                                    gin_transport, dst_rank_idx,
                                    context_stripe,
                                    &completion_requests[context_stripe]);
                                completion_outstanding[context_stripe] = true;
                                pending_rows = 0;
                                active_context =
                                    (context_stripe + 1) %
                                    kGinPeerCount;
                            }
                            ++row;
                        }
                    }
                    __syncwarp();
                    wave_blocks += issue_group_blocks;
                    issue_group_blocks = 0;
                    issue_group_valid_ms = 0;
                };

                for (uint32_t expert_idx = 0;
                     expert_idx < kNumExpertsPerRank; ++expert_idx) {
                    const uint32_t num_expert_tokens =
                        scheduler.get_num_tokens(expert_idx);
                    const uint32_t num_expert_blocks =
                        math::ceil_div(num_expert_tokens, BLOCK_M);

                    for (uint32_t expert_block_idx = 0;
                         expert_block_idx < num_expert_blocks;
                         ++expert_block_idx, ++pool_block_idx) {
                        const uint32_t slot =
                            pool_block_idx % gin_transport.outbox_depth;
                        const uint32_t generation =
                            pool_block_idx / gin_transport.outbox_depth;
                        const uint32_t full_target =
                            (generation + 1) * kNumL2BlockNs;
                        const auto full_ptr =
                            buffer.gin_workspace.get_outbox_full_count_ptr(slot);
                        uint32_t block_ready = 0;
                        if (lane_idx == 0)
                            block_ready = ptx::ld_acq(full_ptr) >= full_target;
                        block_ready = __shfl_sync(0xffffffffu, block_ready, 0);
                        if (not block_ready) {
                            if (issue_group_blocks > 0)
                                drain_issue_group();
                            if (wave_blocks > 0)
                                release_wave();
                        }

                        // Each consuming lane performs its own acquire before
                        // reading outbox rows produced by the epilogue blocks.
                        while (ptx::ld_acq(full_ptr) < full_target) {}

                        const uint32_t row_base = expert_block_idx * BLOCK_M;
                        const uint32_t valid_m = cute::min(
                            num_expert_tokens - row_base, BLOCK_M);
                        if (issue_group_blocks == 0)
                            issue_group_start = pool_block_idx;
                        DG_DEVICE_ASSERT(
                            pool_block_idx ==
                                issue_group_start + issue_group_blocks);
                        issue_group_valid_ms |=
                            static_cast<uint64_t>(valid_m) <<
                            (issue_group_blocks * 8u);
                        ++issue_group_blocks;

                        if (issue_group_blocks ==
                                gin_transport.combine_issue_wave or
                            wave_blocks + issue_group_blocks ==
                                gin_transport.outbox_depth)
                            drain_issue_group();
                        if (wave_blocks == gin_transport.outbox_depth)
                            release_wave();
                        __syncwarp();
                    }
                }
                if (issue_group_blocks > 0)
                    drain_issue_group();
                if (wave_blocks > 0)
                    release_wave();
                __syncwarp();
            }
#endif
        }

        // Clean workspace for the next usage, and also do cumulative stats
        // NOTES: it is overlapped with combine reduction epilogue
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
        if constexpr (kUseGin and kMegaMoeGinLocalAblationStage < 3) {
            // Phase 2 is driven by the epilogue below.  Do not let dispatch
            // cleanup (and its phase-3 barrier) overtake that collective.
            if (not kMegaMoeGinActiveFastPath or
                ptx::ld_acq(
                    workspace.get_gin_world_active_ptr()) != 0) {
                ptx::sync_unaligned(
                    kNumDispatchThreads + kNumEpilogueThreads,
                    kDispatchWithEpilogueBarrierIdx);
            }
        }

        DG_GIN_TRACE_IF(warp_idx == 0 and lane_idx == 0, 62);
        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if (sm_idx == 0) {
            // SM 0: clear expert send count and schedule task counters
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
                *workspace.get_expert_send_count_ptr(i) = 0;
            if (warp_idx == 0 and cute::elect_one_sync()) {
                *workspace.get_l1_task_count_ptr() = 0;
                *workspace.get_l2_task_count_ptr() = 0;
                *workspace.get_shared_l1_task_count_ptr() = 0;
                *workspace.get_shared_l2_task_count_ptr() = 0;
            }
            __syncwarp();
            for (uint32_t i = thread_idx; i < workspace.num_shared_l2_pool_blocks; i += kNumDispatchThreads)
                *workspace.get_shared_l2_full_count_ptr(i) = 0;
            __syncwarp();
        } else {
            // Other SMs: clean blocks
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                // Read expert token count before clearing
                const auto num_recv_tokens = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(i));
                const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);

                // Compute expert pool block offset
                expert_pool_block_offset = scheduler.get_pool_block_offset(i);

                // Wait read count ready
                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

                // Clean expert token count, and add cumulative results
                DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                if (warp_idx == 0) {
                    *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                } else if (warp_idx == 1) {
                    if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                        ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                    __syncwarp();
                }

                // Clean per-rank token count
                for (uint32_t j = thread_idx; j < kNumRanks; j += kNumDispatchThreads)
                    *workspace.get_expert_recv_count_ptr(j, i) = 0;
                __syncwarp();

                // Clean L1 and L2 full stuffs and ring buffer counts
                for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                    *workspace.get_l1_full_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l1_empty_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l2_full_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l2_empty_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                }
                __syncwarp();
            }
        }

        // Wait for all ranks to finish cleaning.  Reset the cumulative outbox
        // generations only after every slot has been drained so graph replay
        // begins from the same all-zero state as the first launch.
        if constexpr (kUseGin and kMegaMoeGinLocalAblationStage < 3) {
#ifdef DG_MEGAMOE_GIN
            const bool run_remote_path =
                not kMegaMoeGinActiveFastPath or
                ptx::ld_acq(
                    workspace.get_gin_world_active_ptr()) != 0;
            if (run_remote_path) {
            if (sm_idx == 0) {
                for (uint32_t i = thread_idx;
                     i < gin_transport.outbox_depth;
                     i += kNumDispatchThreads) {
                    *buffer.gin_workspace.get_outbox_full_count_ptr(i) = 0;
                    *buffer.gin_workspace.get_outbox_empty_count_ptr(i) = 0;
                }
            }
            __threadfence_system();
            comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
                workspace, sm_idx, thread_idx,
                [=]() {
                    ptx::sync_aligned(
                        kNumDispatchThreads, kDispatchBarrierIdx);
                });
            if (sm_idx == 0 and warp_idx == 0) {
                if constexpr (kMegaMoeGinStrongVACombineTerminal) {
                    if (use_gin_strongva_combine_terminal) {
                        // The earlier dispatch/epilogue handoff retired every
                        // local packet reader; this grid proves all dispatch
                        // CTAs finished workspace cleanup. Retire each sender
                        // chain now, immediately before the cleanup world
                        // rendezvous permits kernel/replay source reuse.
                        const uint32_t lsa_size = static_cast<uint32_t>(
                            gin_transport.dev_comm.lsaSize);
                        DG_DEVICE_ASSERT(lsa_size == kGinPeerCount);
                        if (lane_idx < lsa_size) {
                            const uint32_t remote_base =
                                (1u - sym_buffer.rank_idx / lsa_size) *
                                lsa_size;
                            const uint32_t peer = remote_base + lane_idx;
                            ncclGinRequest_t request{};
                            comm::mega_moe_gin_flush_data_peer_async(
                                gin_transport, peer,
                                /*context_stripe=*/ 0u, &request);
                            comm::mega_moe_gin_wait_data_peer(
                                gin_transport, /*context_stripe=*/ 0u,
                                request);
                            DG_GIN_TRACE_IF(true, 80u + lane_idx);
                        }
                        // The world barrier below is warp-cooperative. Do not
                        // let inactive peer lanes enter it before all active
                        // LSA-peer completions have returned.
                        __syncwarp();
                    }
                }
                comm::mega_moe_gin_world_barrier(
                    gin_transport, kGinCleanupBarrierIdx,
                    ncclGinFenceLevel::None);
                DG_GIN_TRACE_IF(lane_idx == 0, 63);
            }
            } else {
                comm::nvlink_lsa_barrier<
                    kNumRanks, kGinPeerCount, kNumSMs, kNumDispatchThreads,
                    kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
                        workspace, sym_buffer, sm_idx, thread_idx,
                        [=]() {
                            ptx::sync_aligned(
                                kNumDispatchThreads, kDispatchBarrierIdx);
                        },
                        /* Publish all cleanup writes */ true,
                        /* No work follows in dispatch warps */ false);
            }
#endif
        } else if constexpr (kUseGin) {
#ifdef DG_MEGAMOE_GIN
            if constexpr (kUseGinOutbox) {
                if (sm_idx == 0) {
                    for (uint32_t i = thread_idx;
                         i < gin_transport.outbox_depth;
                         i += kNumDispatchThreads) {
                        *buffer.gin_workspace
                             .get_outbox_full_count_ptr(i) = 0;
                        *buffer.gin_workspace
                             .get_outbox_empty_count_ptr(i) = 0;
                    }
                }
            }
            comm::nvlink_lsa_barrier<
                kNumRanks, kGinPeerCount, kNumSMs, kNumDispatchThreads,
                kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
                workspace, sym_buffer, sm_idx, thread_idx,
                [=]() {
                    ptx::sync_aligned(
                        kNumDispatchThreads, kDispatchBarrierIdx);
                },
                /* Publish all cleanup writes */ true,
                /* No work follows in dispatch warps */ false);
#endif
        } else {
            comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                                 kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
                workspace, sym_buffer, sm_idx, thread_idx,
                [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
                /* Before the NVLink barrier, there is a grid sync */ true,
                /* At the end of kernel does not need to sync */ false
            );
        }
    } else if (warp_idx == kNumDispatchWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM TMA load warp for tokens with SFA
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            const auto tensor_map_a_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_acts :
                                          task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_acts :
                                          task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_acts :
                                        /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_acts;
            const auto tensor_map_sfa_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_acts_sf :
                                            task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_acts_sf :
                                            task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_acts_sf :
                                          /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_acts_sf;
            const auto num_k_blocks = math::ceil_div(task_info.shape_k, BLOCK_K);

            // Compute pool block offset for this expert
            const uint32_t pool_block_idx = task_info.pool_block_idx;
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t block_idx = task_info.is_shared() ? pool_block_idx : ring_block_idx;

            // Wait the entire token arrival
            if (task_info.block_phase == sched::BlockPhase::Linear1) {
                const auto ptr = workspace.get_l1_full_count_ptr(block_idx);
                const auto num_expected_tokens = BLOCK_M * (pool_block_idx / kNumRingBlocks + 1);
                while (ptx::ld_acq(ptr) != num_expected_tokens);
            } else if (task_info.block_phase == sched::BlockPhase::Linear2) {
                const auto ptr = workspace.get_l2_full_count_ptr(block_idx);
                const auto num_expected_blocks = (L2_SHAPE_K / BLOCK_N) * 2 * (pool_block_idx / kNumRingBlocks + 1);
                while (ptx::ld_acq(ptr) != num_expected_blocks);
            } else if (task_info.block_phase == sched::BlockPhase::SharedLinear2) {
                const auto ptr = workspace.get_shared_l2_full_count_ptr(block_idx);
                const auto num_expected_blocks = (SHARED_L2_SHAPE_K / BLOCK_N) * 2;
                while (ptx::ld_acq(ptr) != num_expected_blocks);
            }

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait consumer release
                shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);

                // Compute token offsets from block index
                uint32_t m_idx = block_idx * BLOCK_M;
                uint32_t k_idx = k_block_idx * BLOCK_K;
                const uint32_t sfa_m_idx = block_idx * SF_BLOCK_M;
                uint32_t sfa_k_idx = k_block_idx * (BLOCK_K / 128);

                // Add 2 CTA offsets for non-leader CTA
                if (not is_leader_cta)
                    m_idx += task_info.get_umma_aligned_valid_m() / 2;

                // TMA copy tokens and SFA, then arrive at full barrier
                if (cute::elect_one_sync()) {
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                        tensor_map_a_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_a[stage_idx], k_idx, m_idx, 2);
                    tma::copy<SF_BLOCK_M, 1, 0>(
                        tensor_map_sfa_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_sfa[stage_idx], sfa_m_idx, sfa_k_idx, 2);
                    if (is_leader_cta) {
                        shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(sizeof(SharedStorage::smem_a[0]) * 2 + sizeof(SharedStorage::smem_sfa[0]) * 2);
                    } else {
                        shared_storage.full_barriers[stage_idx].arrive(0u);
                    }
                }
                __syncwarp();
            }
        }
    } else if (warp_idx == kNumDispatchWarps + 1) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM TMA load warp for weights with SF
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            const auto tensor_map_b_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_weights :
                                          task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights :
                                          task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_weights :
                                        /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_weights;
            const auto tensor_map_sfb_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_weights_sf :
                                            task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights_sf :
                                            task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_weights_sf :
                                          /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_weights_sf;

            const auto shape_k = task_info.shape_k;
            const auto shape_n = task_info.shape_n;
            const auto shape_sfb_k = math::ceil_div(shape_k, kGranK * 4u);
            const auto n_block_idx = task_info.n_cluster_idx * 2 + (is_leader_cta ? 0u : 1u);
            const auto num_k_blocks = math::ceil_div(shape_k, BLOCK_K);

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait consumer release
                shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);

                // Compute weight offset
                uint32_t n_idx = task_info.is_shared() ? n_block_idx * BLOCK_N : task_info.local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                uint32_t k_idx = k_block_idx * BLOCK_K;
                uint32_t sfb_n_idx = n_block_idx * BLOCK_N;
                uint32_t sfb_k_idx = task_info.is_shared() ? k_block_idx * (BLOCK_K / 128) : task_info.local_expert_idx * shape_sfb_k + k_block_idx * (BLOCK_K / 128);

                // TMA copy weights with SF
                if (cute::elect_one_sync()) {
                    if (task_info.is_shared()) {
                        tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, shared_b_dtype_t>(
                            tensor_map_b_ptr, &shared_storage.full_barriers[stage_idx], reinterpret_cast<shared_b_dtype_t*>(shared_storage.smem_b[stage_idx]), k_idx, n_idx, 2);
                        tma::copy<BLOCK_N, 1, 0>(
                            tensor_map_sfb_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_sfb[stage_idx], sfb_n_idx, sfb_k_idx, 2);
                        if (is_leader_cta) {
                            shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(sizeof(SharedStorage::smem_b[0]) * 2 + sizeof(SharedStorage::smem_sfb[0]) * 2);
                        } else {
                            shared_storage.full_barriers[stage_idx].arrive(0u);
                        }
                    } else {
                        tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
                            tensor_map_b_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_b[stage_idx], k_idx, n_idx, 2);
                        tma::copy<BLOCK_N, 1, 0>(
                            tensor_map_sfb_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_sfb[stage_idx], sfb_n_idx, sfb_k_idx, 2);
                        if (is_leader_cta) {
                            shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(sizeof(SharedStorage::smem_b[0]) + sizeof(SharedStorage::smem_sfb[0]) * 2);
                        } else {
                            shared_storage.full_barriers[stage_idx].arrive(0u);
                        }
                    }
                }
                __syncwarp();
            }
        }
    } else if (warp_idx == kNumDispatchWarps + 2) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM MMA issue warp (only the leader CTA will run)
        if (is_leader_cta) {
            // Make instruction descriptor with block scaling
            // NOTES: always swap A/B
            auto routed_instr_desc = cute::UMMA::make_instr_desc_block_scaled<
                    b_dtype_t, a_dtype_t, float, cutlass::float_ue8m0_t,
                    UMMA_M, UMMA_N,
                    cute::UMMA::Major::K, cute::UMMA::Major::K
                >();
            auto shared_instr_desc = cute::UMMA::make_instr_desc_block_scaled<
                shared_b_dtype_t, a_dtype_t, float, cutlass::float_ue8m0_t,
                UMMA_M, UMMA_N,
                cute::UMMA::Major::K, cute::UMMA::Major::K
            >();
            auto sf_desc = mma::sm100::make_sf_desc(nullptr);

            DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
            auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_M, UMMA_BLOCK_K, kSwizzleAMode>(shared_storage.smem_a[0], 0, 0);
            auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_N, UMMA_BLOCK_K, kSwizzleBMode>(shared_storage.smem_b[0], 0, 0);
            auto shared_b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_N, UMMA_BLOCK_K, kSwizzleBMode>(reinterpret_cast<shared_b_dtype_t*>(shared_storage.smem_b[0]), 0, 0);
            uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * sizeof(SharedStorage::smem_a[0]) / 16 : 0u;
            uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * sizeof(SharedStorage::smem_b[0]) / 16 : 0u;
            uint32_t shared_b_desc_lo = lane_idx < kNumStages ? shared_b_desc.lo + lane_idx * sizeof(SharedStorage::smem_b[0]) / 16 : 0u;

            // Checks for MMA instructions
            DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                             "Invalid MMA instruction shape");

            // Persistently schedule over blocks
            uint32_t current_iter_idx = 0;
            task_info_t task_info;
            while (scheduler.get_next_task(task_info)) {
                const auto num_k_blocks = task_info.shape_k / BLOCK_K;
                DG_GIN_TRACE_IF(lane_idx == 0 and current_iter_idx == 0, 96);

                // Dynamic update of UMMA N based on effective M
                auto& instr_desc = task_info.is_shared() ? shared_instr_desc : routed_instr_desc;
                mma::sm100::update_instr_desc_with_umma_n(instr_desc, task_info.get_umma_aligned_valid_m());

                // Wait tensor memory empty barrier arrival
                const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
                const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
                shared_storage.tmem_empty_barriers[accum_stage_idx].wait(accum_phase ^ 1);
                ptx::tcgen05_after_thread_sync();

                // Empty barrier arrival
                auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                    auto umma_arrive = [](const uint64_t* barrier) {
                        constexpr uint16_t kCTAMask = (1 << 2) - 1;
                        cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                    };
                    umma_arrive(reinterpret_cast<uint64_t*>(&shared_storage.empty_barriers[stage_idx]));

                    // NOTES: the tensor memory accumulator pipeline has nothing to do with multicasting
                    if (do_tmem_full_arrive)
                        umma_arrive(reinterpret_cast<uint64_t*>(&shared_storage.tmem_full_barriers[accum_stage_idx]));
                    __syncwarp();
                };

                // Launch MMAs
                #pragma unroll 2
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    // Wait TMA load completion
                    shared_storage.full_barriers[stage_idx].wait(phase);
                    ptx::tcgen05_after_thread_sync();

                    DG_GIN_TRACE_IF(lane_idx == 0 and current_iter_idx == 1 and
                                    k_block_idx == 0, 97);

                    const auto a_desc_base_lo = ptx::exchange(a_desc_lo, stage_idx);
                    const auto b_desc_base_lo = ptx::exchange(task_info.is_shared() ? shared_b_desc_lo : b_desc_lo, stage_idx);
                    if (cute::elect_one_sync()) {
                        #pragma unroll
                        for (uint32_t umma_k_block_idx = 0; umma_k_block_idx < BLOCK_K / UMMA_BLOCK_K; ++ umma_k_block_idx) {
                            // UTCCP copy SFA and SFB to TMEM
                            using cute_utccp_t = cute::SM100_UTCCP_4x32dp128bit_2cta;
                            #pragma unroll
                            for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                                auto smem_ptr = shared_storage.smem_sfa[stage_idx] + umma_k_block_idx * SF_BLOCK_M + i * kNumUTCCPAlignedElems;
                                mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                                cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
                            }
                            #pragma unroll
                            for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                                auto smem_ptr = shared_storage.smem_sfb[stage_idx] + umma_k_block_idx * SF_BLOCK_N + i * kNumUTCCPAlignedElems;
                                mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                                cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
                            }

                            // Issue UMMA
                            #pragma unroll
                            for (uint32_t k = 0; k < UMMA_BLOCK_K / UMMA_K; ++ k) {
                                const auto runtime_instr_desc =
                                    mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k, k);
                                a_desc.lo = mma::sm100::advance_umma_desc_lo<
                                    cute::UMMA::Major::K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(a_desc_base_lo, umma_k_block_idx * UMMA_BLOCK_K * LOAD_BLOCK_M * sizeof(a_dtype_t), k * UMMA_K);
                                if (task_info.is_shared()) {
                                    b_desc.lo = mma::sm100::advance_umma_desc_lo<
                                        cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleBMode, shared_b_dtype_t>(b_desc_base_lo, umma_k_block_idx * UMMA_BLOCK_K * LOAD_BLOCK_N * sizeof(shared_b_dtype_t), k * UMMA_K);
                                } else {
                                    b_desc.lo = mma::sm100::advance_umma_desc_lo<
                                        cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(b_desc_base_lo, umma_k_block_idx * UMMA_BLOCK_K * LOAD_BLOCK_N * sizeof(b_dtype_t), k * UMMA_K);
                                }
                                ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma(
                                    b_desc, a_desc, accum_stage_idx * UMMA_N,
                                    k_block_idx > 0 or umma_k_block_idx > 0 or k > 0, runtime_instr_desc,
                                    kTmemStartColOfSFB, kTmemStartColOfSFA);
                            }
                        }
                    }
                    __syncwarp();

                    // Commit to the mbarrier object
                    // No explicit `tcgen05.fence::before_thread_sync` is needed, as this is implicitly performed by `tcgen05.commit`
                    empty_barrier_arrive(k_block_idx == num_k_blocks - 1);
                }
            }

            // To safely deconstruct barriers, we need another round of waits
            if (current_iter_idx > 0) {
                const auto accum_phase_idx = ((current_iter_idx - 1) / kNumEpilogueStages) & 1;
                shared_storage.tmem_empty_barriers[(current_iter_idx - 1) % kNumEpilogueStages].wait(accum_phase_idx);
            }
        }
        DG_GIN_TRACE_IF(lane_idx == 0 and is_leader_cta, 98);
    } else if (warp_idx == kNumDispatchWarps + 3) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // Do mainloop by the leader CTA
        if (is_leader_cta)
            scheduler.mainloop(num_tokens);
    } else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();

        // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits,
        // i.e., no need for `tmem_ptr |= (epilogue_warp_idx * 32) << 16`.
        // NOTES: we also forbid two CTAs to share the same SM and its tensor memory
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(&shared_storage.tmem_ptr_in_smem) == 0);

        // GEMM epilogue warps
        const auto epilogue_warp_idx = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        const auto epilogue_wg_idx = epilogue_warp_idx / 4;
        const auto epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        const auto warp_idx_in_wg = epilogue_warp_idx % 4;
        DG_STATIC_ASSERT((kNumDispatchWarps + kNumMMANonEpilogueWarps) % 4 == 0 and
                         kNumEpilogueWarps % 4 == 0, "Invalid epilogue warps");

        // TODO: support effective block M
        // NOTES:
        //  - 2 warpgroups divide the whole BM into BM / 2
        //  - 4 warps divide the whole BN into BN / 4
        //  - BM / 2 is further divided into stored blocks, i.e. with `STORE_BLOCK_M` size
        //  - `STORE_BLOCK_M` in further divided into `ATOM_M`
        constexpr uint32_t WG_BLOCK_M = BLOCK_M / kNumEpilogueWarpgroups;
        constexpr uint32_t ATOM_M = 8;
        constexpr uint32_t kNumBankGroupBytes = 16u;
        constexpr uint32_t kNumAtomsPerStore = STORE_BLOCK_M / ATOM_M;
        DG_STATIC_ASSERT(BLOCK_M % kNumEpilogueWarpgroups == 0, "Invalid block M");
        DG_STATIC_ASSERT(WG_BLOCK_M % STORE_BLOCK_M == 0, "Invalid warpgroup block M");
        DG_STATIC_ASSERT(STORE_BLOCK_M % ATOM_M == 0, "Invalid store block M");
        DG_STATIC_ASSERT(BLOCK_N == 128, "Invalid block N");

        // Ensure the epilogue barrier cannot run with the pull barrier
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Dispatch has completed the world-uniform activity consensus before
        // reaching this handoff.  Cache the decision once per epilogue thread
        // so output rows and the scatter phase do not repeatedly acquire-load
        // the same two workspace words.
        const bool use_gin_bulk_combine =
            use_gin_bulk_combine_this_launch();

        const bool use_gin_combine_overlap =
            use_gin_combine_overlap_this_launch();
        const bool use_gin_direct_reduce =
            prepare_gin_direct_reduce_ordinals and use_gin_combine_overlap;

        // Persistently schedule over blocks
        uint32_t current_iter_idx = 0;
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            // Wait UMMA arrival
            const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
            const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
            shared_storage.tmem_full_barriers[accum_stage_idx].wait(accum_phase);
            ptx::tcgen05_after_thread_sync();
            DG_GIN_TRACE_IF(epilogue_warp_idx == 0 and lane_idx == 0 and
                            current_iter_idx == 1, 52);

            // Now we can release the task
            scheduler.release_task_info();

            // Compute offsets
            // NOTES: use shuffle here to let NVCC know warp divergence won't happen
            const uint32_t valid_m = ptx::exchange(task_info.valid_m, 0);
            const uint32_t pool_block_idx = task_info.pool_block_idx;
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t block_idx = task_info.is_shared() ? pool_block_idx : ring_block_idx;
            const uint32_t ring_m_idx = ring_block_idx * BLOCK_M;  // Ring-buffer offset for reusable data buffers
            const uint32_t m_idx = block_idx * BLOCK_M;
            const uint32_t pool_m_idx = pool_block_idx * BLOCK_M;  // Full-pool offset for non-ring metadata
            const uint32_t n_block_idx = task_info.n_cluster_idx * 2 + (is_leader_cta ? 0u : 1u);
            uint32_t n_idx = n_block_idx * BLOCK_N;

            if (task_info.block_phase == sched::BlockPhase::Linear1 or task_info.block_phase == sched::BlockPhase::SharedLinear1) {
                if (not task_info.is_shared()) {
                    // Wait L2 block empty
                    const auto l2_empty_ptr = workspace.get_l2_empty_count_ptr(ring_block_idx);
                    const auto num_expected_blocks = (L2_SHAPE_N / BLOCK_N) * (pool_block_idx / kNumRingBlocks);
                    while (ptx::ld_acq(l2_empty_ptr) != num_expected_blocks);
                }

                // Unified L1 epilogue: SwiGLU in-place using granularity 8 interleaved weights
                // With `SM100_TMEM_LOAD_16dp256b1x`, gate/up pairs are:
                float stored_cached_weight = 1.0f;

                #pragma unroll
                for (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++ s) {
                    // Early break if the entire store block is beyond the valid token range
                    if (epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M >= valid_m) {
                        ptx::tcgen05_before_thread_sync();
                        shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                        break;
                    }

                    // Iterate all atoms in the store block
                    float2 activation_values[kNumAtomsPerStore][2];
                    float2 amax_values[kNumAtomsPerStore];
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                        const uint32_t j = s * kNumAtomsPerStore + i;

                        // Load weights from global into register cache per 32 tokens
                        DG_STATIC_ASSERT(32 % ATOM_M == 0, "Invalid block size");
                        if (not task_info.is_shared() and (j * ATOM_M) % 32 == 0 and
                            (WG_BLOCK_M % 32 == 0 or j * ATOM_M + lane_idx < WG_BLOCK_M)) {
                            stored_cached_weight = *buffer.l1_topk_weights_buffer
                                .get_data_buffer(ring_m_idx + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M + lane_idx)
                                .template get_base_ptr<float>();
                        }

                        // Load weights from register cache
                        const float2 weights = {
                            ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 0),
                            ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 1)
                        };

                        // Load from TMEM
                        uint2 raw_values[4];
                        uint32_t tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M;
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                               raw_values[0].x, raw_values[0].y, raw_values[1].x, raw_values[1].y);
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                               raw_values[2].x, raw_values[2].y, raw_values[3].x, raw_values[3].y);
                        cutlass::arch::fence_view_async_tmem_load();

                        // Signal tensor memory consumed on the last atom
                        if (j == WG_BLOCK_M / ATOM_M - 1) {
                            ptx::tcgen05_before_thread_sync();
                            shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                        }

                        // Apply SwiGLU: silu(gate) * up
                        auto fp32_values = reinterpret_cast<float2*>(raw_values);
                        #pragma unroll
                        for (uint32_t k = 0; k < 2; ++ k) {
                            auto bf16_gate = __float22bfloat162_rn(fp32_values[k * 2 + 0]);
                            auto bf16_up =   __float22bfloat162_rn(fp32_values[k * 2 + 1]);

                            // Clamp
                            if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity()) {
                                bf16_gate = __hmin2(bf16_gate, {kActivationClamp, kActivationClamp});
                                bf16_up = __hmax2(bf16_up, {-kActivationClamp, -kActivationClamp});
                                bf16_up = __hmin2(bf16_up, {kActivationClamp, kActivationClamp});
                            }

                            // SwiGLU
                            auto gate = __bfloat1622float2(bf16_gate);
                            auto neg_gate_exp = make_float2(
                                kFastMath ? __expf(-gate.x) : expf(-gate.x),
                                kFastMath ? __expf(-gate.y) : expf(-gate.y));
                            const auto denom = __fadd2_rn({1.0f, 1.0f}, neg_gate_exp);
                            if constexpr (kFastMath) {
                                gate = __fmul2_rn(gate, {math::fast_rcp(denom.x), math::fast_rcp(denom.y)});
                            } else {
                                gate = {gate.x / denom.x, gate.y / denom.y};
                            }
                            const auto up = __bfloat1622float2(bf16_up);
                            activation_values[i][k] = __fmul2_rn(__fmul2_rn(gate, up), weights);
                        }

                        // Amax reduction (thread-level)
                        float2 thread_local_amax = {0.f, 0.f};
                        #pragma unroll
                        for (uint32_t k = 0; k < 2; ++ k) {
                            thread_local_amax.x = cute::max(thread_local_amax.x, cute::abs(activation_values[i][k].x));
                            thread_local_amax.y = cute::max(thread_local_amax.y, cute::abs(activation_values[i][k].y));
                        }

                        // Amax reduction (warp-level)
                        amax_values[i].x = math::warp_reduce<4, true>(
                            thread_local_amax.x, math::ReduceMax<float>());
                        amax_values[i].y = math::warp_reduce<4, true>(
                            thread_local_amax.y, math::ReduceMax<float>());

                        // Reduce amax (warp-pair-level)
                        if (lane_idx < 4)
                            shared_storage.amax_reduction[epilogue_warp_idx][i * (ATOM_M / 2) + lane_idx] = amax_values[i];
                        __syncwarp();
                    }

                    // Wait shared memory release from previous TMA store
                    // And fence `shared_storage.amax_reduction`
                    const uint32_t tma_stage_idx = s % kNumTMAStoreStages;
                    ptx::tma_store_wait<kNumTMAStoreStages - 1>();
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Cast to FP8 E4M3 and store into shared memory
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                        // Reduce amax (warp-pair-level)
                        const float2 wp_amax =
                            shared_storage.amax_reduction[epilogue_warp_idx ^ 1][i * (ATOM_M / 2) + lane_idx % 4];
                        amax_values[i].x = cute::max(amax_values[i].x, wp_amax.x);
                        amax_values[i].y = cute::max(amax_values[i].y, wp_amax.y);

                        // Calculate SF
                        float2 sf, sf_inv;
                        math::get_e4m3_sf_and_sf_inv(amax_values[i], sf, sf_inv);

                        // Cast
                        const float2 upper = __fmul2_rn(activation_values[i][0], sf_inv);
                        const float2 lower = __fmul2_rn(activation_values[i][1], sf_inv);
                        const auto fp8x4_values = __nv_fp8x4_e4m3(make_float4(upper.x, upper.y, lower.x, lower.y));

                        // STSM
                        uint32_t row = lane_idx;
                        uint32_t col = warp_idx_in_wg;
                        const auto smem_ptr = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l1[epilogue_wg_idx][tma_stage_idx])
                            + i * ATOM_M * L1_OUT_BLOCK_N
                            + row * L1_OUT_BLOCK_N
                            // Use 64B swizzle for SwiGLU, so divided by 2
                            + (col ^ (row / 2)) * kNumBankGroupBytes;
                        ptx::SM100_U8x4_STSM_T<__nv_fp8x4_e4m3>::copy(fp8x4_values, smem_ptr);

                        // Store SF to `buffer.l2_sf_buffer` as UE8M0 (MN-major layout)
                        // Only one warp per pair writes (both hold the same SF after cross-warp reduce)
                        // Each lane < 4 holds SF for 2 rows (sf.x and sf.y)
                        if (warp_idx_in_wg % 2 == 0 and lane_idx < 4) {
                            const uint32_t k_idx = n_block_idx * 2 + warp_idx_in_wg / 2;
                            const uint32_t k_uint_idx = k_idx / 4, byte_idx = k_idx % 4;
                            const uint32_t mn_stride = (task_info.is_shared() ? kNumSharedSFTokens : kNumSFRingTokens) * sizeof(uint32_t);
                            const auto sf_base_ptr = task_info.is_shared() ?
                                buffer.shared_l2_sf_buffer.get_base_ptr<uint8_t>() : buffer.l2_sf_buffer.get_base_ptr<uint8_t>();
                            // NOTES: consecutive tokens (t, t + 1) are in the same 32-group, so `sf_idx` differs by 4
                            // NOTES: originally there was:
                            //   - `const uint32_t token_idx_in_expert = task_info.m_block_idx * BLOCK_M + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M + lane_idx * 2
                            //   - `task_info.pool_block_idx * SF_BLOCK_M + transform_sf_token_idx(token_idx_in_expert)`
                            // We find out that
                            //   1. `task_info.m_block_idx * BLOCK_M` mod `BLOCK_M` is 0, and `epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M + lane_idx * 2` is always < `BLOCK_M`, so we can put `task_info.m_block_idx * BLOCK_M` outside
                            //   2. `lane_idx * 2` controls the lowest 3 bit of `token_idx_in_expert`, and `transform_sf_token_idx` is a bitwise-independent transformation if the input is less than `BLOCK_M`, so we can put `lane_idx * 2` outside
                            // This reduce the number of computation instructions.
                            const uint32_t token_base_idx = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M;
                            __builtin_assume(token_base_idx < BLOCK_M);
                            const auto sf_token_idx = block_idx * SF_BLOCK_M
                                + transform_sf_token_idx(token_base_idx) + (lane_idx * 2) * 4;
                            const auto sf_addr = k_uint_idx * mn_stride + sf_token_idx * static_cast<uint32_t>(sizeof(uint32_t)) + byte_idx;
                            sf_base_ptr[sf_addr] =
                                (*reinterpret_cast<const uint32_t*>(&sf.x) >> 23);
                            sf_base_ptr[sf_addr + 4 * static_cast<uint32_t>(sizeof(uint32_t))] =
                                (*reinterpret_cast<const uint32_t*>(&sf.y) >> 23);
                        }
                        __syncwarp();
                    }
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Issue TMA store after all atoms in this store block
                    if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                        const auto tensor_map_l1_output_ptr = task_info.is_shared() ? &tensor_map_shared_l1_output : &tensor_map_l1_output;
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            tensor_map_l1_output_ptr,
                            shared_storage.smem_d.l1[epilogue_wg_idx][tma_stage_idx],
                            out_n_idx,
                            m_idx + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                }

                // Notify L2 and increment L1 empty count
                // TODO: less epilogue sync scope
                ptx::tma_store_wait<0>();
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                    if (task_info.is_shared()) {
                        ptx::red_add_rel(
                            workspace.get_shared_l2_full_count_ptr(pool_block_idx), 1u);
                    } else {
                        ptx::red_add_rel(
                            workspace.get_l2_full_count_ptr(ring_block_idx), 1u);

                        // Increment L1 empty count for this physical slot (one per N block)
                        ptx::red_add(
                            workspace.get_l1_empty_count_ptr(ring_block_idx), 1u);
                    }
                }
                __syncwarp();
            } else {
                // Increment L2 empty count for this physical slot (one per N block)
                if (not task_info.is_shared()) {
                    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                        ptx::red_add(
                            workspace.get_l2_empty_count_ptr(ring_block_idx), 1u);
                    }
                    __syncwarp();
                }

                DG_STATIC_ASSERT(STORE_BLOCK_M % 8 == 0, "Invalid store M");
                constexpr uint32_t kNumRowsPerWarp = STORE_BLOCK_M / 8;

                uint32_t gin_outbox_slot = 0;
                if constexpr (kUseGinOutbox) {
#ifdef DG_MEGAMOE_GIN
                    const bool run_gin_outbox =
                        not use_gin_bulk_combine and
                        (not kMegaMoeGinActiveFastPath or
                         ptx::ld_acq(
                             workspace.get_gin_world_active_ptr()) != 0);
                    if (not task_info.is_shared() and run_gin_outbox) {
                        gin_outbox_slot =
                            pool_block_idx % gin_transport.outbox_depth;
                        const uint32_t generation =
                            pool_block_idx / gin_transport.outbox_depth;
                        const auto empty_ptr = buffer.gin_workspace
                            .get_outbox_empty_count_ptr(gin_outbox_slot);
                        while (ptx::ld_acq(empty_ptr) < generation) {}
                    }
#endif
                }

                // L2 BF16 epilogue: same-LSA rows retain mapped NVLink stores;
                // cross-LSA rows are staged in the bounded registered outbox.
                #pragma unroll
                for (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++ s) {
                    // Early break if the entire store block is beyond the valid token range
                    // TODO: check performance
                    if (epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M >= valid_m) {
                        ptx::tcgen05_before_thread_sync();
                        shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                        break;
                    }

                    #pragma unroll
                    for (uint32_t i = 0; i < STORE_BLOCK_M / ATOM_M; ++ i) {
                        // Load from TMEM using .16x256b shape to satisfy STSM layout requirements
                        // Start from lane index 0 and 16
                        uint32_t tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M;
                        uint32_t values[ATOM_M];
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                               values[0], values[1], values[2], values[3]);
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                               values[4], values[5], values[6], values[7]);
                        cutlass::arch::fence_view_async_tmem_load();

                        // Wait shared memory release from previous NVLink store
                        // NOTES: skip for the first store block since the prior full barrier already ensures completion
                        if (i == 0 and s > 0)
                            ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                        // Signal tensor memory consumed
                        if (s == WG_BLOCK_M / STORE_BLOCK_M - 1 and i == STORE_BLOCK_M / ATOM_M - 1) {
                            ptx::tcgen05_before_thread_sync();
                            shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                        }

                        // Store into shared memory
                        // NOTES: each lane provides its own address for stmatrix; 2 warps share a BF16 swizzle atom
                        uint32_t row = lane_idx % 8;
                        uint32_t col = (epilogue_warp_idx % 2) * 4 + lane_idx / 8;
                        const auto smem_ptr = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l2[epilogue_wg_idx]) +
                            (warp_idx_in_wg / 2) * STORE_BLOCK_M * kSwizzleCDMode +
                            i * ATOM_M * kSwizzleCDMode +
                            row * (kNumBankGroupBytes * 8) +
                            (col ^ row) * kNumBankGroupBytes;
                        ptx::SM90_U32x4_STSM_T<uint32_t>::copy(
                            math::cast_into_bf16_and_pack(values[0], values[1]),
                            math::cast_into_bf16_and_pack(values[2], values[3]),
                            math::cast_into_bf16_and_pack(values[4], values[5]),
                            math::cast_into_bf16_and_pack(values[6], values[7]),
                            smem_ptr
                        );
                    }

                    // Wait shared memory ready
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Write into remote buffers
                    // Each warp writes 2 rows (lane_idx/16 splits the warp into two halves, one per row)
                    const uint32_t row_in_atom = (warp_idx_in_wg * 2 + lane_idx / 16) % ATOM_M;
                    const uint32_t bank_group_idx = lane_idx % 8;

                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                        const uint32_t row_in_store = j * 8 + warp_idx_in_wg * 2 + lane_idx / 16;
                        const uint32_t m_idx_in_block = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + row_in_store;

                        // Skip padding rows beyond the actual token count for this expert
                        if (m_idx_in_block >= valid_m)
                            break;

                        uint32_t dst_rank_idx, dst_token_idx, dst_topk_idx;
                        if (task_info.is_shared()) {
                            dst_rank_idx = sym_buffer.rank_idx;
                            dst_token_idx = pool_m_idx + m_idx_in_block;
                            dst_topk_idx = kNumTopk;
                        } else {
                            const auto src_metadata = *workspace.get_token_src_metadata_ptr(pool_m_idx + m_idx_in_block);
                            dst_rank_idx = src_metadata.rank_idx;
                            dst_token_idx = src_metadata.token_idx;
                            dst_topk_idx = src_metadata.topk_idx;
                        }

                        // Read from shared memory
                        const auto smem_ptr = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l2[epilogue_wg_idx]) +
                            (lane_idx % 16 / 8) * STORE_BLOCK_M * kSwizzleCDMode +
                            row_in_store * kSwizzleCDMode +
                            (bank_group_idx ^ row_in_atom) * kNumBankGroupBytes;
                        const auto packed = ptx::ld_shared(reinterpret_cast<float4*>(smem_ptr));

                        // Write into remote
                        const auto dst_token = buffer.combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                               .get_data_buffer(dst_token_idx);
                        const auto dst_ptr = math::advance_ptr<float4>(
                            dst_token.get_base_ptr(),
                            n_idx * static_cast<uint32_t>(sizeof(nv_bfloat16)) + (lane_idx % 16) * static_cast<uint32_t>(sizeof(float4)));
                        if constexpr (kUseGin and
                                      kMegaMoeGinLocalAblationStage >= 3) {
#ifdef DG_MEGAMOE_GIN
                            const bool is_same_lsa =
                                gin_transport.is_same_lsa_peer(dst_rank_idx);
                            DG_DEVICE_ASSERT(
                                is_same_lsa and
                                "GIN local ablation produced a cross-LSA output");
                            if (is_same_lsa)
                                *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
#endif
                        } else if constexpr (kUseGin) {
#ifdef DG_MEGAMOE_GIN
                            if (gin_transport.is_same_lsa_peer(dst_rank_idx)) {
                                *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                            } else if (use_gin_bulk_combine) {
                                const uint32_t return_idx =
                                    *buffer.gin_workspace
                                         .get_bulk_combine_return_index_ptr(
                                             pool_m_idx + m_idx_in_block);
                                const uint32_t peer_in_lsa =
                                    dst_rank_idx % static_cast<uint32_t>(
                                        gin_transport.dev_comm.lsaSize);
                                const auto packet_payload =
                                    buffer.gin_workspace
                                        .get_bulk_combine_record_payload_ptr(
                                            /*send=*/ true, peer_in_lsa,
                                            return_idx);
                                const auto bulk_ptr = math::advance_ptr<float4>(
                                    packet_payload,
                                    n_idx * static_cast<uint32_t>(
                                                sizeof(nv_bfloat16)) +
                                        (lane_idx % 16) *
                                            static_cast<uint32_t>(
                                                sizeof(float4)));
                                *bulk_ptr = packed;
                                if (n_idx == 0 and lane_idx % 16 == 0) {
                                    *buffer.gin_workspace
                                         .get_bulk_combine_record_destination_ptr(
                                             /*send=*/ true, peer_in_lsa,
                                             return_idx) =
                                        dst_token_idx * kNumTopk +
                                        dst_topk_idx;
                                }
                            } else {
                                const auto outbox_ptr = math::advance_ptr<float4>(
                                    buffer.gin_workspace
                                        .get_combine_outbox_row_ptr(
                                            gin_outbox_slot, m_idx_in_block),
                                    n_idx * static_cast<uint32_t>(sizeof(nv_bfloat16)) +
                                        (lane_idx % 16) *
                                            static_cast<uint32_t>(sizeof(float4)));
                                *outbox_ptr = packed;
                            }
#endif
                        } else {
                            *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                        }
                    }
                }

                // Ensure the next epilogue safe to use shared memory
                if constexpr (kUseGin and kMegaMoeGinCombineOverlap) {
#ifdef DG_MEGAMOE_GIN
                    if (use_gin_combine_overlap and not task_info.is_shared()) {
                        // Same named barrier/count as the original epilogue,
                        // now with explicit compiler ordering for publication.
                        comm::mega_moe_gin_combine_producer_barrier(
                            kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                        DG_DEVICE_ASSERT(task_info.local_expert_idx <
                            layout::kMegaMoeGinCombineOverlapNumExperts);
                        if (epilogue_warp_idx == 0 and lane_idx == 0)
                            comm::mega_moe_gin_combine_ready_release(
                                buffer.gin_workspace.get_combine_overlap_ready_ptr(
                                    task_info.local_expert_idx));
                    } else {
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    }
#endif
                } else {
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                }

#if DG_MEGAMOE_GIN_DIAGNOSTICS >= 2
                // One writer per (logical SM, expert). Sampling the first and
                // last routed L2 task on every SM reconstructs expert-range
                // production opportunity without contended ready counters.
                // This is a timing point, not NIC-visible readiness proof.
                if (not task_info.is_shared()) {
                    DG_GIN_TRACE_FIRST_IF(epilogue_warp_idx == 0 and lane_idx == 0,
                                         128u + 2u * task_info.local_expert_idx);
                    DG_GIN_TRACE_IF(epilogue_warp_idx == 0 and lane_idx == 0,
                                   129u + 2u * task_info.local_expert_idx);
                }
#endif

                if constexpr (kUseGinOutbox) {
#ifdef DG_MEGAMOE_GIN
                    // One release increment per routed L2 N tile.  The block
                    // drainer waits for all L2_SHAPE_N / BLOCK_N producers,
                    // including tiles whose rows all remain on the local LSA.
                    if (not task_info.is_shared() and
                        not use_gin_bulk_combine and
                        (not kMegaMoeGinActiveFastPath or
                         ptx::ld_acq(
                             workspace.get_gin_world_active_ptr()) != 0) and
                        epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                        ptx::red_add_rel(
                            buffer.gin_workspace
                                .get_outbox_full_count_ptr(gin_outbox_slot),
                            1u);
                    }
                    __syncwarp();
#endif
                }
            }
        }

        // Deallocate tensor memory
        DG_GIN_TRACE_IF(epilogue_warp_idx == 0 and lane_idx == 0, 53);
        // NOTES: must be called by the same logical warp ID on both CTAs
        if (epilogue_warp_idx == 0)
            Allocator().free(0, kNumTmemCols);

        DG_GIN_TRACE_IF(epilogue_warp_idx == 0 and lane_idx == 0, 54);

        const bool use_gin_strongva_combine_terminal =
            use_gin_strongva_combine_terminal_this_launch();
        const bool use_gin_combine_owner_slot_ready =
            use_gin_combine_owner_slot_ready_this_launch();
        const bool use_gin_owner_slot_pair_reduce =
            kGinOwnerSlotShape and
            num_tokens <= layout::kMegaMoeGinDirectDispatchMaxTokens;
        if constexpr (kGinOwnerSlotShape) {
            if (use_gin_combine_owner_slot_ready) {
                DG_DEVICE_ASSERT(
                    use_gin_direct_reduce and
                    prepare_gin_direct_reduce_ordinals);
            }
        }

        if constexpr (kUseGin and kMegaMoeGinLocalAblationStage < 3) {
#ifdef DG_MEGAMOE_GIN
            const bool run_remote_path =
                not kMegaMoeGinActiveFastPath or
                ptx::ld_acq(
                    workspace.get_gin_world_active_ptr()) != 0;
            if (run_remote_path) {
            // Publish every same-LSA mapped store before handing phase 2 to
            // its visibility protocol. The first local barrier joins this CTA's
            // dispatch phase; the grid below joins all early PUT submissions.
            // StrongVA mode has queued every peer terminal here, while its
            // sender-local completion remains deferred until late cleanup.
            __threadfence_system();
            ptx::sync_unaligned(
                kNumDispatchThreads + kNumEpilogueThreads,
                kDispatchWithEpilogueBarrierIdx);

            // All epilogue CTAs have completed their direct stores and every
            // CTA has observed its dispatch side at the local handoff.  Only
            // one converged warp per rank enters the NCCL world barrier.
            comm::grid_sync<kNumSMs, kEpilogueGridSyncIndex>(
                workspace, sm_idx, epilogue_thread_idx,
                [&]() {
                    ptx::sync_aligned(
                        kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            });
            DG_GIN_TRACE_IF(epilogue_warp_idx == 0 and lane_idx == 0, 55);
            bool use_single_combine_context = false;
            if constexpr (kMegaMoeGinSingleCombineContext) {
                // Both decisions are world-uniform. T64/non-direct fallback
                // keeps its original data striping and all-context Put fence.
                use_single_combine_context = use_gin_bulk_combine and
                    use_gin_direct_dispatch_this_launch();
            }
            if (sm_idx == 0 and epilogue_warp_idx == 0 and
                not use_gin_strongva_combine_terminal) {
                if constexpr (kMegaMoeGinBulkCombine) {
                    if (use_gin_bulk_combine) {
                        const uint32_t lsa_size = static_cast<uint32_t>(
                            gin_transport.dev_comm.lsaSize);
                        if (lane_idx < lsa_size) {
                            const uint32_t remote_lsa_base =
                                (1u - sym_buffer.rank_idx / lsa_size) *
                                lsa_size;
                            const uint32_t remote_source =
                                remote_lsa_base + lane_idx;
                            uint32_t route_count = 0;
                            #pragma unroll
                            for (uint32_t expert = 0;
                                 expert < kNumExpertsPerRank; ++expert) {
                                route_count += static_cast<uint32_t>(
                                    *workspace.get_expert_recv_count_ptr(
                                        remote_source, expert));
                            }
                            constexpr uint32_t kBulkCapacity =
                                layout::kMegaMoeGinBulkCombineMaxTokens *
                                kNumTopk;
                            DG_DEVICE_ASSERT(route_count <= kBulkCapacity);
                            if (route_count > 0) {
                                auto* local_packet =
                                    buffer.gin_workspace
                                        .get_bulk_combine_packet_ptr(
                                            /*send=*/ true, lane_idx);
                                *static_cast<uint32_t*>(local_packet) =
                                    route_count;
                                __threadfence_system();
                                const uint32_t owner_in_lsa =
                                    sym_buffer.rank_idx % lsa_size;
                                auto* remote_packet =
                                    buffer.gin_workspace
                                        .get_bulk_combine_packet_ptr(
                                            /*send=*/ false, owner_in_lsa);
                                const uint32_t packet_bytes =
                                    layout::kMegaMoeGinBulkCombineRecordAreaOffset +
                                    route_count *
                                        buffer.gin_workspace.bulk_record_bytes;
                                DG_DEVICE_ASSERT(
                                    packet_bytes <=
                                    buffer.gin_workspace.bulk_packet_bytes);
                                if (use_gin_combine_overlap) {
                                    // Records were already sent by dispatch.
                                    // Retain late count publication to avoid a
                                    // race with the receiver's startup clear.
                                    comm::mega_moe_gin_put_bulk_combine_header(
                                        gin_transport, remote_source,
                                        /*context_stripe=*/ 0u,
                                        sym_buffer.get_base_ptr(), local_packet,
                                        remote_packet,
                                        /*diagnostic_peer_lane=*/ lane_idx);
                                } else {
                                    comm::mega_moe_gin_put_bulk_combine_packet(
                                        gin_transport, remote_source,
                                        /*context_stripe=*/
                                            use_single_combine_context ? 0u : lane_idx,
                                        sym_buffer.get_base_ptr(), local_packet,
                                        remote_packet, packet_bytes,
                                        /*diagnostic_peer_lane=*/ lane_idx);
                                }
                            }
                        }
                        __syncwarp();
                    }
                }

                // Local flushes protect source reuse. The matching context-1
                // or all-context Put fence establishes remote visibility.
                DG_GIN_TRACE_IF(lane_idx == 0, 56);
                if (use_single_combine_context) {
                    comm::mega_moe_gin_world_barrier_single_combine_context(
                        gin_transport, kGinCombinePutBarrierIdx);
                } else {
                    comm::mega_moe_gin_world_barrier_all_contexts(
                        gin_transport, kGinCombinePutBarrierIdx,
                        ncclGinFenceLevel::Put);
                }
                __threadfence_system();
                DG_GIN_TRACE_IF(lane_idx == 0, 57);
            }
            if constexpr (kMegaMoeGinStrongVACombineTerminal) {
                if (use_gin_strongva_combine_terminal) {
                    // Grid1 above is the prologue for same-LSA output stores;
                    // grid2 below is the epilogue. StrongVA terminals establish
                    // cross-LSA packet visibility without a world collective.
                    comm::nvlink_lsa_barrier<
                        kNumRanks, kGinPeerCount, kNumSMs,
                        kNumEpilogueThreads, kEpilogueGridSyncIndex,
                        kBeforeCombineReduceBarrierTag>(
                            workspace, sym_buffer, sm_idx,
                            epilogue_thread_idx,
                            [&]() {
                                ptx::sync_aligned(
                                    kNumEpilogueThreads,
                                    kEpilogueFullBarrierIdx);
                            },
                            /* Existing grid1 is the prologue */ false,
                            /* Existing grid2 is the epilogue */ false);

                    if (use_gin_combine_owner_slot_ready) {
                        // Finish the existing LSA handoff before independent
                        // token CTAs consume same-LSA results. Cross-LSA owner
                        // terminals remain asynchronous after this point.
                        comm::grid_sync<kNumSMs, kEpilogueGridSyncIndex>(
                            workspace, sm_idx, epilogue_thread_idx,
                            [&]() {
                                ptx::sync_aligned(
                                    kNumEpilogueThreads,
                                    kEpilogueFullBarrierIdx);
                            });
                    }

                    if (not use_gin_combine_owner_slot_ready and
                        sm_idx == 0 and epilogue_warp_idx == 0) {
                        // Lane0 may still be polling the LSA barrier arrival;
                        // reconverge before lanes independently wait for all
                        // remote owner terminals on shared data context 0.
                        __syncwarp();
                        DG_GIN_TRACE_IF(lane_idx == 0, 56);
                        if (lane_idx < kGinPeerCount) {
                            comm::mega_moe_gin_wait_bulk_combine_terminal(
                                gin_transport, /*context_stripe=*/ 0u,
                                sym_buffer.get_base_ptr(),
                                buffer.gin_workspace
                                    .get_combine_terminal_signal_ptr(lane_idx),
                                *workspace
                                     .get_gin_combine_terminal_epoch_ptr());
                        }
                        __syncwarp();
                        if (lane_idx == 0)
                            __threadfence_system();
                        DG_GIN_TRACE_IF(lane_idx == 0, 57);
                    }
                }
            }
            if (not use_gin_combine_owner_slot_ready) {
                comm::grid_sync<kNumSMs, kEpilogueGridSyncIndex>(
                    workspace, sm_idx, epilogue_thread_idx,
                    [&]() {
                        ptx::sync_aligned(
                            kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    });
            }

            DG_GIN_TRACE_IF(epilogue_warp_idx == 0 and lane_idx == 0, 58);

            if constexpr (kGinOwnerSlotShape) {
                constexpr uint32_t kNumOwnerProgressWarps = kGinPeerCount;
                constexpr uint32_t kNumGlobalEpilogueWarps =
                    kNumSMs * kNumEpilogueWarps;
                constexpr uint32_t kOwnerProgressWarpBase =
                    kNumGlobalEpilogueWarps - kNumOwnerProgressWarps;
                const uint32_t global_epilogue_warp =
                    sm_idx * kNumEpilogueWarps + epilogue_warp_idx;
                if (use_gin_combine_owner_slot_ready and
                    global_epilogue_warp >= kOwnerProgressWarpBase) {
                    // Four tail warps independently acquire the existing four
                    // whole-owner terminals. No additional network action or
                    // registered storage is introduced by this experiment.
                    const uint32_t owner_in_lsa =
                        global_epilogue_warp - kOwnerProgressWarpBase;
                    if (lane_idx == 0) {
                        comm::mega_moe_gin_wait_bulk_combine_terminal(
                            gin_transport, /*context_stripe=*/ 0u,
                            sym_buffer.get_base_ptr(),
                            buffer.gin_workspace
                                .get_combine_terminal_signal_ptr(owner_in_lsa),
                            *workspace.get_gin_combine_terminal_epoch_ptr());
                        // The StrongVA acquire precedes both the proxy bridge
                        // and rank-local release publication. Token-pair warps
                        // perform a matching acquire before resolving packet
                        // metadata or issuing a payload TMA load.
                        asm volatile(
                            "fence.proxy.async.global;" ::: "memory");
                        ptx::red_add_rel(
                            buffer.gin_workspace
                                .get_combine_receiver_owner_ready_ptr(
                                    owner_in_lsa),
                            1u);

                        // Keep the exact packet-count audit, but move it off
                        // the token critical path and after acquired visibility.
                        const uint32_t lsa_size = static_cast<uint32_t>(
                            gin_transport.dev_comm.lsaSize);
                        const uint32_t remote_lsa_base =
                            (1u - sym_buffer.rank_idx / lsa_size) * lsa_size;
                        const uint32_t remote_owner =
                            remote_lsa_base + owner_in_lsa;
                        const auto* expected_counts =
                            buffer.gin_workspace.count_staging_buffer
                                .get_rank_buffer(remote_owner)
                                .template get_base_ptr<uint64_t>();
                        uint32_t expected_count = 0;
                        #pragma unroll
                        for (uint32_t expert = 0;
                             expert < kNumExpertsPerRank; ++expert) {
                            expected_count += static_cast<uint32_t>(
                                expected_counts[expert]);
                        }
                        const uint32_t received_count = ptx::ld_acq_sys(
                            buffer.gin_workspace
                                .get_bulk_combine_packet_count_ptr(
                                    /*send=*/ false, owner_in_lsa));
                        DG_DEVICE_ASSERT(received_count == expected_count);
                    }
                    __syncwarp();
                }
            }

            if constexpr (kMegaMoeGinBulkCombine) {
                if (use_gin_bulk_combine) {
                    // Scatter compact owner packets into the unchanged
                    // [top-k, token, hidden] combine buffer.  One warp owns one
                    // record at a time; shipped token/top-k headers avoid
                    // replaying nondeterministic source-side atomics.
                    constexpr uint32_t kBulkCapacity =
                        layout::kMegaMoeGinBulkCombineMaxTokens * kNumTopk;
                    constexpr uint32_t kCombineRowUint4 =
                        kHidden * sizeof(nv_bfloat16) / sizeof(uint4);
                    constexpr uint32_t kNumGlobalEpilogueWarps =
                        kNumSMs * kNumEpilogueWarps;
                    const uint32_t global_epilogue_warp =
                        sm_idx * kNumEpilogueWarps + epilogue_warp_idx;
                    const uint32_t lsa_size = static_cast<uint32_t>(
                        gin_transport.dev_comm.lsaSize);

                    // Cross-check the received packet count against this
                    // source rank's preserved outbound count vector.  Only
                    // active peer lanes perform the check, so it does not inflate
                    // the per-record scatter loop.
                    if (not use_gin_combine_owner_slot_ready and
                        sm_idx == 0 and epilogue_warp_idx == 0 and
                        lane_idx < lsa_size) {
                        const uint32_t remote_lsa_base =
                            (1u - sym_buffer.rank_idx / lsa_size) * lsa_size;
                        const uint32_t remote_owner =
                            remote_lsa_base + lane_idx;
                        const auto* expected_counts =
                            buffer.gin_workspace.count_staging_buffer
                                .get_rank_buffer(remote_owner)
                                .template get_base_ptr<uint64_t>();
                        uint32_t expected_count = 0;
                        #pragma unroll
                        for (uint32_t expert = 0;
                             expert < kNumExpertsPerRank; ++expert)
                            expected_count += static_cast<uint32_t>(
                                expected_counts[expert]);
                        const uint32_t received_count = ptx::ld_acq_sys(
                            buffer.gin_workspace
                                .get_bulk_combine_packet_count_ptr(
                                    /*send=*/ false, lane_idx));
                        DG_DEVICE_ASSERT(received_count == expected_count);
                    }
                    __syncwarp();

                    if (not use_gin_direct_reduce) {
                    #pragma unroll
                    for (uint32_t owner_in_lsa = 0;
                         owner_in_lsa < kGinPeerCount;
                         ++owner_in_lsa) {
                        auto* recv_count_ptr =
                            buffer.gin_workspace
                                .get_bulk_combine_packet_count_ptr(
                                    /*send=*/ false, owner_in_lsa);
                        const uint32_t recv_count =
                            ptx::ld_acq_sys(recv_count_ptr);
                        DG_DEVICE_ASSERT(recv_count <= kBulkCapacity);
                        for (uint32_t return_idx = global_epilogue_warp;
                             return_idx < recv_count;
                             return_idx += kNumGlobalEpilogueWarps) {
                            const uint32_t destination = ptx::ld_acq_sys(
                                buffer.gin_workspace
                                    .get_bulk_combine_record_destination_ptr(
                                        /*send=*/ false, owner_in_lsa,
                                        return_idx));
                            DG_DEVICE_ASSERT(
                                destination < num_tokens * kNumTopk);
                            const uint32_t dst_token_idx =
                                destination / kNumTopk;
                            const uint32_t dst_topk_idx =
                                destination % kNumTopk;
                            const auto* src = static_cast<const uint4*>(
                                buffer.gin_workspace
                                    .get_bulk_combine_record_payload_ptr(
                                        /*send=*/ false, owner_in_lsa,
                                        return_idx));
                            auto* dst = buffer.combine_token_buffer
                                .get_rank_buffer(dst_topk_idx)
                                .get_data_buffer(dst_token_idx)
                                .template get_base_ptr<uint4>();
                            #pragma unroll
                            for (uint32_t element = lane_idx;
                                 element < kCombineRowUint4;
                                 element += 32)
                                dst[element] = src[element];
                            __syncwarp();
                        }
                    }
                    __threadfence();

                    DG_GIN_TRACE_IF(lane_idx == 0, 112u + epilogue_warp_idx);
                    DG_GIN_TRACE_IF(epilogue_warp_idx == 0 and lane_idx == 0, 59);

                    // The reduction may be scheduled on a different CTA from
                    // the record scatter, so retain a third epilogue grid
                    // rendezvous only for the scatter path. Every CTA on this
                    // rank makes the same local alias-fit/eligibility choice.
                    comm::grid_sync<kNumSMs, kEpilogueGridSyncIndex>(
                        workspace, sm_idx, epilogue_thread_idx,
                        [&]() {
                            ptx::sync_aligned(
                                kNumEpilogueThreads,
                                kEpilogueFullBarrierIdx);
                        });
                    DG_GIN_TRACE_IF(epilogue_warp_idx == 0 and lane_idx == 0, 60);
                    }
                }
            }

            // Release dispatch cleanup here only for the scatter/fallback path
            // after phase 2 is locally complete. Direct packet readers keep this
            // existing second handoff pending until reduction ends.
            if (not use_gin_direct_reduce)
                ptx::sync_unaligned(
                    kNumDispatchThreads + kNumEpilogueThreads,
                    kDispatchWithEpilogueBarrierIdx);
            } else {
                if (not use_gin_strongva_combine_terminal) {
                    comm::nvlink_lsa_barrier<
                        kNumRanks, kGinPeerCount, kNumSMs,
                        kNumEpilogueThreads, kEpilogueGridSyncIndex,
                        kBeforeCombineReduceBarrierTag>(
                            workspace, sym_buffer, sm_idx,
                            epilogue_thread_idx,
                            [&]() {
                                ptx::sync_aligned(
                                    kNumEpilogueThreads,
                                    kEpilogueFullBarrierIdx);
                            });
                }

                ptx::sync_unaligned(
                    kNumDispatchThreads + kNumEpilogueThreads,
                    kDispatchWithEpilogueBarrierIdx);
            }
#endif
        } else if constexpr (kUseGin) {
#ifdef DG_MEGAMOE_GIN
            comm::nvlink_lsa_barrier<
                kNumRanks, kGinPeerCount, kNumSMs, kNumEpilogueThreads,
                kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag>(
                workspace, sym_buffer, sm_idx, epilogue_thread_idx,
                [&]() {
                    ptx::sync_aligned(
                        kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                });

            ptx::sync_unaligned(
                kNumDispatchThreads + kNumEpilogueThreads,
                kDispatchWithEpilogueBarrierIdx);
#endif
        } else {
            // NVLink barrier (grid sync + cross-rank signal + grid sync): ~4 us
            comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads,
                                 kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag>(
                workspace, sym_buffer, sm_idx, epilogue_thread_idx,
                [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
            );

            // Barrier with dispatch warps, so that they can do clean workspace
            ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
        }

        if constexpr (kUseGin and kMegaMoeGinCombineOverlap) {
            if (use_gin_direct_reduce and
                not use_gin_combine_owner_slot_ready) {
                // The selected world Put fence or StrongVA waits plus grid2
                // establish target visibility. Bridge that acquired global
                // memory into TMA's async proxy once on every potential issuer,
                // before reading immutable received packets.
                asm volatile("fence.proxy.async.global;" ::: "memory");
            }
        }

        // Combine: reduce top-k results and write back
        // NOTES: reuse shared memory from start up to the barriers
        // 1 token, 1 topk latency: ~3 us
        constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
        constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);

        // 3 slots of chunk is needed: 2 load stages and 1 store
        constexpr uint32_t kNumChunkSlots = 3;
        constexpr uint32_t kNumMaxRegistersForBuffer = 128;

        // NOTES: either 1 or 2 chunks for simplicity
        // NOTES: Restrict on both smem and register
        constexpr uint32_t kNumChunks =
            kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes <= kNumReusableSmemBytes and kHidden <= 32 * kNumMaxRegistersForBuffer ? 1 : 2;
        constexpr uint32_t kNumChunkBytes = kNumHiddenBytes / kNumChunks;
        constexpr uint32_t kNumChunkUint4 = kNumChunkBytes / sizeof(uint4);
        constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;
        DG_STATIC_ASSERT(kHidden % kNumChunks == 0, "Hidden must be divisible by number of chunks");
        DG_STATIC_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes / kNumChunks <= kNumReusableSmemBytes, "Hidden is too large");
        DG_STATIC_ASSERT(kNumChunkBytes % 16 == 0, "Combine chunk must be TMA-aligned (16 bytes)");
        DG_STATIC_ASSERT(kNumChunkBytes % sizeof(uint4) == 0, "Combine chunk must be divisible by 16 bytes");
        DG_STATIC_ASSERT(kNumChunkUint4 % 32 == 0, "Combine chunk must be a multiple of 32 16-byte elements (one per lane)");
        DG_STATIC_ASSERT(kNumTopk + (kNumSharedExperts > 0 ? 1u : 0u) <= 32u, "Top-k + shared must fit in a single warp");

        // Verify combined shared memory budget at runtime
        DG_DEVICE_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumChunkBytes <= kNumReusableSmemBytes);

        // Per-warp buffer: 2 stage load buffers + 1 store buffer
        const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
            return math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + i * kNumEpilogueWarps) * kNumChunkBytes);
        });
        const auto combine_store_buffer  = math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + kNumEpilogueWarps * 2) * kNumChunkBytes);

        // Per-warp barriers
        auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
            return &shared_storage.combine_barriers[i + epilogue_warp_idx * 2];
        });

        // Iterate over all tokens
        DG_GIN_TRACE_IF(lane_idx == 0, 120u + epilogue_warp_idx);
        DG_GIN_TRACE_IF(epilogue_warp_idx == 0 and lane_idx == 0, 61);
        if (use_gin_owner_slot_pair_reduce) {
            // One CTA owns one token. Its eight epilogue warps preserve broad
            // slot-level parallelism: warp w reduces assignments {2w,2w+1}.
            // Two independent BF16 load stages per fixed assignment pair let
            // either owner start its TMA as soon as it becomes ready. After
            // both loads retire, those two chunks are reused as the pair's
            // equally-sized FP32 partial. One separate output chunk makes 17
            // chunks total, within the original 24-chunk shared-memory budget.
            DG_DEVICE_ASSERT(kNumEpilogueWarps == 8 and kNumTopk == 16);
            DG_DEVICE_ASSERT(
                (2u * kNumEpilogueWarps + 1u) * kNumChunkBytes <=
                kNumReusableSmemBytes);
            const auto pair_load_buffers = utils::PatternVisitor(
                [&](const uint32_t& pair_slot) {
                    return math::advance_ptr<uint4>(
                        smem_buffer,
                        (epilogue_warp_idx * 2u + pair_slot) *
                            kNumChunkBytes);
                });
            auto* pair_partial_buffer = math::advance_ptr<float2>(
                smem_buffer,
                epilogue_warp_idx * 2u * kNumChunkBytes);
            auto* pair_output_buffer = math::advance_ptr<uint4>(
                smem_buffer,
                2u * kNumEpilogueWarps * kNumChunkBytes);
            const auto pair_load_barriers = utils::PatternVisitor(
                [&](const uint32_t& pair_slot) {
                    return &shared_storage.combine_barriers[
                        epilogue_warp_idx * 2u + pair_slot];
                });
            uint32_t pair_load_phases[2] = {};

            for (uint32_t token_idx = sm_idx;
                 token_idx < num_tokens; token_idx += kNumSMs) {
                const uint32_t first_slot = epilogue_warp_idx * 2u;
                int pair_experts[2] = {-1, -1};
                if (lane_idx == 0) {
                    pair_experts[0] = static_cast<int>(__ldg(
                        buffer.input_topk_idx_buffer.get_base_ptr<int64_t>() +
                        token_idx * kNumTopk + first_slot));
                    pair_experts[1] = static_cast<int>(__ldg(
                        buffer.input_topk_idx_buffer.get_base_ptr<int64_t>() +
                        token_idx * kNumTopk + first_slot + 1u));
                }
                pair_experts[0] = __shfl_sync(
                    0xffffffffu, pair_experts[0], 0);
                pair_experts[1] = __shfl_sync(
                    0xffffffffu, pair_experts[1], 0);

                uint64_t pair_row_ptrs[2] = {};

                for (uint32_t chunk = 0; chunk < kNumChunks; ++chunk) {
                    const uint32_t chunk_byte_offset =
                        chunk * kNumChunkBytes;
                    if (epilogue_warp_idx == 0) {
                        // Retire the preceding store before reusing the
                        // separate output stage for this chunk.
                        ptx::tma_store_wait<0>();
                        __syncwarp();
                    }

                    if (lane_idx == 0) {
                        if (chunk == 0) {
                            // Poll both owners together. Same-LSA assignments
                            // are immediately ready; if both remote owners are
                            // ready in the same observation, the lower fixed
                            // assignment slot wins the deterministic tie.
                            uint32_t pending_mask =
                                (pair_experts[0] >= 0 ? 1u : 0u) |
                                (pair_experts[1] >= 0 ? 2u : 0u);
                            while (pending_mask != 0) {
                                int selected_pair_slot = -1;
                                #pragma unroll
                                for (uint32_t pair_slot = 0;
                                     pair_slot < 2; ++pair_slot) {
                                    if ((pending_mask &
                                         (1u << pair_slot)) == 0)
                                        continue;
                                    DG_DEVICE_ASSERT(
                                        pair_experts[pair_slot] <
                                        kNumExperts);

                                    bool assignment_ready = true;
#ifdef DG_MEGAMOE_GIN
                                    if (use_gin_direct_reduce) {
                                        const uint32_t owner =
                                            static_cast<uint32_t>(
                                                pair_experts[pair_slot]) /
                                            kNumExpertsPerRank;
                                        if (not gin_transport
                                                    .is_same_lsa_peer(owner) and
                                            use_gin_combine_owner_slot_ready) {
                                            const uint32_t owner_in_lsa =
                                                owner %
                                                static_cast<uint32_t>(
                                                    gin_transport.dev_comm
                                                        .lsaSize);
                                            assignment_ready =
                                                ptx::ld_acq(
                                                    buffer.gin_workspace
                                                        .get_combine_receiver_owner_ready_ptr(
                                                            owner_in_lsa)) != 0;
                                        }
                                    }
#endif
                                    if (assignment_ready) {
                                        selected_pair_slot =
                                            static_cast<int>(pair_slot);
                                        break;
                                    }
                                }
                                if (selected_pair_slot < 0)
                                    continue;

                                const uint32_t pair_slot =
                                    static_cast<uint32_t>(selected_pair_slot);
                                const uint32_t slot_idx =
                                    first_slot + pair_slot;
                                const int expert = pair_experts[pair_slot];
                                DG_DEVICE_ASSERT(expert < kNumExperts);
                                pair_row_ptrs[pair_slot] =
                                    reinterpret_cast<uint64_t>(
                                        buffer.combine_token_buffer
                                            .get_rank_buffer(slot_idx)
                                            .get_data_buffer(token_idx)
                                            .get_base_ptr());
#ifdef DG_MEGAMOE_GIN
                                if (use_gin_direct_reduce) {
                                    const uint32_t owner =
                                        static_cast<uint32_t>(expert) /
                                        kNumExpertsPerRank;
                                    if (not gin_transport
                                                .is_same_lsa_peer(owner)) {
                                        const uint32_t owner_in_lsa = owner %
                                            static_cast<uint32_t>(
                                                gin_transport.dev_comm.lsaSize);
                                        // Owner readiness is the release/acquire
                                        // handoff for the immutable packet
                                        // metadata below. Never resolve it
                                        // before that owner's terminal arrives.
                                        asm volatile(
                                            "fence.proxy.async.global;" :::
                                            "memory");
                                        const uint32_t token_topk_idx =
                                            token_idx * kNumTopk + slot_idx;
                                        const uint32_t ordinal =
                                            *buffer.gin_workspace
                                                 .get_combine_direct_reduce_ordinal_ptr(
                                                     token_topk_idx);
                                        const uint32_t received_count =
                                            ptx::ld_acq_sys(
                                                buffer.gin_workspace
                                                    .get_bulk_combine_packet_count_ptr(
                                                        /*send=*/ false,
                                                        owner_in_lsa));
                                        DG_DEVICE_ASSERT(
                                            ordinal < received_count);
                                        const uint32_t destination =
                                            ptx::ld_acq_sys(
                                                buffer.gin_workspace
                                                    .get_bulk_combine_record_destination_ptr(
                                                        /*send=*/ false,
                                                        owner_in_lsa,
                                                        ordinal));
                                        DG_DEVICE_ASSERT(
                                            destination == token_topk_idx);
                                        pair_row_ptrs[pair_slot] =
                                            reinterpret_cast<uint64_t>(
                                                buffer.gin_workspace
                                                    .get_bulk_combine_record_payload_ptr(
                                                        /*send=*/ false,
                                                        owner_in_lsa,
                                                        ordinal));
                                    } else {
                                        // Bridge same-LSA direct-store
                                        // visibility for this fixed TMA
                                        // issuer too.
                                        asm volatile(
                                            "fence.proxy.async.global;" :::
                                            "memory");
                                    }
                                }
#endif
                                auto* src_ptr =
                                    math::advance_ptr<uint8_t>(
                                        reinterpret_cast<void*>(
                                            pair_row_ptrs[pair_slot]),
                                        chunk_byte_offset);
                                ptx::tma_load_1d(
                                    pair_load_buffers[pair_slot], src_ptr,
                                    pair_load_barriers[pair_slot],
                                    kNumChunkBytes);
                                ptx::mbarrier_arrive_and_set_tx(
                                    pair_load_barriers[pair_slot],
                                    kNumChunkBytes);
                                pending_mask &= ~(1u << pair_slot);
                            }
                        } else {
                            // Both immutable row pointers were cached during
                            // chunk zero. Issue both independent stages without
                            // serializing either TMA behind a completion wait.
                            #pragma unroll
                            for (uint32_t pair_slot = 0;
                                 pair_slot < 2; ++pair_slot) {
                                if (pair_experts[pair_slot] < 0)
                                    continue;
                                auto* src_ptr =
                                    math::advance_ptr<uint8_t>(
                                        reinterpret_cast<void*>(
                                            pair_row_ptrs[pair_slot]),
                                        chunk_byte_offset);
                                ptx::tma_load_1d(
                                    pair_load_buffers[pair_slot], src_ptr,
                                    pair_load_barriers[pair_slot],
                                    kNumChunkBytes);
                                ptx::mbarrier_arrive_and_set_tx(
                                    pair_load_barriers[pair_slot],
                                    kNumChunkBytes);
                            }
                        }
                    }
                    __syncwarp();

                    float2 pair_reduced[
                        kNumUint4PerLane * kNumElemsPerUint4] = {};
                    // TMA issue order is readiness-driven, but accumulation is
                    // always fixed ascending assignment-slot order so the r11
                    // FP32 pair association remains bitwise unchanged.
                    #pragma unroll
                    for (uint32_t pair_slot = 0; pair_slot < 2;
                         ++pair_slot) {
                        if (pair_experts[pair_slot] < 0)
                            continue;
                        pair_load_barriers[pair_slot]->wait(
                            pair_load_phases[pair_slot]);
                        pair_load_phases[pair_slot] ^= 1u;
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumUint4PerLane; ++j) {
                            const auto uint4_values = pair_load_buffers[pair_slot][
                                j * 32u + lane_idx];
                            const auto* bf16_values =
                                reinterpret_cast<const nv_bfloat162*>(
                                    &uint4_values);
                            #pragma unroll
                            for (uint32_t l = 0;
                                 l < kNumElemsPerUint4; ++l) {
                                ptx::accumulate(
                                    pair_reduced[
                                        j * kNumElemsPerUint4 + l],
                                    bf16_values[l]);
                            }
                        }
                    }
                    __syncwarp();

                    #pragma unroll
                    for (uint32_t value = 0;
                         value <
                             kNumUint4PerLane * kNumElemsPerUint4;
                         ++value) {
                        ptx::st_shared(
                            pair_partial_buffer +
                                value * 32u + lane_idx,
                            pair_reduced[value]);
                    }
                    __syncwarp();
                    ptx::sync_aligned(
                        kNumEpilogueThreads, kEpilogueFullBarrierIdx);

                    if (epilogue_warp_idx == 0) {
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumUint4PerLane; ++j) {
                            uint4 casted;
                            auto* casted_bf16 =
                                reinterpret_cast<nv_bfloat162*>(&casted);
                            #pragma unroll
                            for (uint32_t l = 0;
                                 l < kNumElemsPerUint4; ++l) {
                                const uint32_t value =
                                    j * kNumElemsPerUint4 + l;
                                float2 reduced = {};
                                #pragma unroll
                                for (uint32_t pair = 0; pair < 8;
                                     ++pair) {
                                    const auto* partial =
                                        math::advance_ptr<float2>(
                                            smem_buffer,
                                                pair * 2u *
                                                    kNumChunkBytes);
                                    const float2 next = ptx::ld_shared(
                                        partial + value * 32u + lane_idx);
                                    reduced.x += next.x;
                                    reduced.y += next.y;
                                }
                                casted_bf16[l] =
                                    __float22bfloat162_rn(reduced);
                            }
                            ptx::st_shared(
                                pair_output_buffer + j * 32u + lane_idx,
                                casted.x, casted.y,
                                casted.z, casted.w);
                        }
                        __syncwarp();
                        if (lane_idx == 0) {
                            cute::tma_store_fence();
                            ptx::tma_store_1d(
                                math::advance_ptr(
                                    y,
                                    static_cast<uint64_t>(token_idx) *
                                            kNumHiddenBytes +
                                        chunk_byte_offset),
                                pair_output_buffer, kNumChunkBytes);
                            cute::tma_store_arrive();
                        }
                        __syncwarp();
                    }

                    // Do not reuse the aliased BF16 stage / FP32 pair-partial
                    // region until warp0 has consumed all fixed pairs in
                    // ascending assignment order and queued the output.
                    ptx::sync_aligned(
                        kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                }
            }
        } else {
        uint32_t combine_phase = 0;
        uint32_t load_stage_idx = 0;
        for (uint32_t token_idx = sm_idx * kNumEpilogueWarps + epilogue_warp_idx;
             token_idx < num_tokens;
             token_idx += kNumSMs * kNumEpilogueWarps) {
            // Read top-k slot indices: each lane reads one slot, then broadcast via exchange
            const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(buffer.input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)) :
                (kNumSharedExperts > 0 and lane_idx == kNumTopk ? static_cast<int>(kNumTopk) : -1);
            const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);

            // Each valid slot's lane resolves its row once per token, in
            // parallel, then reuses one pointer across all chunks. Inactive
            // lanes never consult the inverse map; local/shared slots retain
            // their original combine-buffer row. The final Put/grid2/proxy
            // handoff above has already published the immutable packet data.
            uint64_t cached_combine_row_ptr = 0;
            if constexpr (kUseGin and kMegaMoeGinCombineOverlap) {
#ifdef DG_MEGAMOE_GIN
                if (use_gin_direct_reduce and stored_topk_slot_idx >= 0) {
                    cached_combine_row_ptr = reinterpret_cast<uint64_t>(
                        buffer.combine_token_buffer.get_rank_buffer(lane_idx)
                            .get_data_buffer(token_idx).get_base_ptr());
                    if (lane_idx < kNumTopk) {
                        DG_DEVICE_ASSERT(stored_topk_slot_idx < kNumExperts);
                        const uint32_t owner =
                            static_cast<uint32_t>(stored_topk_slot_idx) / kNumExpertsPerRank;
                        if (not gin_transport.is_same_lsa_peer(owner)) {
                            const uint32_t token_topk_idx = token_idx * kNumTopk + lane_idx;
                            // Only current active remote assignments read
                            // these source-packed ordinals. Validate both the
                            // packet bound and original destination identity.
                            const uint32_t ordinal =
                                *buffer.gin_workspace.get_combine_direct_reduce_ordinal_ptr(
                                    token_topk_idx);
                            const uint32_t owner_in_lsa = owner %
                                static_cast<uint32_t>(gin_transport.dev_comm.lsaSize);
                            const uint32_t received_count = ptx::ld_acq_sys(
                                buffer.gin_workspace.get_bulk_combine_packet_count_ptr(
                                    /*send=*/ false, owner_in_lsa));
                            DG_DEVICE_ASSERT(ordinal < received_count);
                            const uint32_t destination = ptx::ld_acq_sys(
                                buffer.gin_workspace.get_bulk_combine_record_destination_ptr(
                                    /*send=*/ false, owner_in_lsa, ordinal));
                            DG_DEVICE_ASSERT(destination == token_topk_idx);
                            cached_combine_row_ptr = reinterpret_cast<uint64_t>(
                                buffer.gin_workspace.get_bulk_combine_record_payload_ptr(
                                    /*send=*/ false, owner_in_lsa, ordinal));
                        }
                    }
                }
#endif
            }

            // Iterate all chunks
            for (uint32_t chunk = 0; chunk < kNumChunks; ++ chunk) {
                const uint32_t chunk_byte_offset = chunk * kNumChunkBytes;

                // Move mask and load
                uint32_t mask = total_mask;
                const auto move_mask_and_load = [&](const uint32_t& i) {
                    if (mask) {
                        // Move
                        const uint32_t slot_idx = __ffs(mask) - 1;
                        mask ^= 1 << slot_idx;

                        uint64_t selected_combine_row_ptr = 0;
                        if constexpr (kUseGin and kMegaMoeGinCombineOverlap) {
                            if (use_gin_direct_reduce) {
                                // This is a full-warp gather, never inside the
                                // elected TMA issuer's divergent branch. The
                                // original ascending slot order is unchanged.
                                selected_combine_row_ptr = __shfl_sync(
                                    0xffffffffu,
                                    static_cast<unsigned long long>(cached_combine_row_ptr),
                                    slot_idx);
                            }
                        }

                        // Load
                        if (cute::elect_one_sync()) {
                            auto src_ptr = math::advance_ptr<uint8_t>(
                                buffer.combine_token_buffer.get_rank_buffer(slot_idx)
                                                    .get_data_buffer(token_idx).get_base_ptr(),
                                chunk_byte_offset);
                            if constexpr (kUseGin and kMegaMoeGinCombineOverlap) {
                                if (use_gin_direct_reduce) {
                                    src_ptr = math::advance_ptr<uint8_t>(
                                        reinterpret_cast<void*>(selected_combine_row_ptr),
                                        chunk_byte_offset);
                                }
                            }
                            ptx::tma_load_1d(combine_load_buffer[i], src_ptr, combine_load_barriers[i], kNumChunkBytes);
                            ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i], kNumChunkBytes);
                        }
                        __syncwarp();
                        return true;
                    }
                    return false;
                };

                // Load the first selection
                bool do_reduce = move_mask_and_load(load_stage_idx);

                // Accumulate all top-k contributions for this chunk in float registers
                float2 reduced[kNumUint4PerLane * kNumElemsPerUint4] = {};
                while (do_reduce) {
                    // Prefetch next top-k into the buffer while current is being accumulated
                    do_reduce = move_mask_and_load(load_stage_idx ^ 1);

                    // Accumulate
                    combine_load_barriers[load_stage_idx]->wait(combine_phase);
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
                        const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                            ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                    }
                    combine_phase ^= load_stage_idx;
                    load_stage_idx ^= 1;
                }

                // Cast
                #pragma unroll
                for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                    uint4 casted;
                    auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                    #pragma unroll
                    for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                        casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);

                    // Wait share memory release and write
                    if (j == 0) {
                        ptx::tma_store_wait<0>();
                        __syncwarp();
                    }
                    ptx::st_shared(combine_store_buffer + j * 32 + lane_idx,
                                   casted.x, casted.y, casted.z, casted.w);
                }
                __syncwarp();

                // TMA store the token chunk
                if (cute::elect_one_sync()) {
                    cute::tma_store_fence();
                    ptx::tma_store_1d(
                        math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + chunk_byte_offset),
                        combine_store_buffer, kNumChunkBytes);
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }
        }
        }
        DG_GIN_TRACE_IF(lane_idx == 0, 88u + epilogue_warp_idx);
        if constexpr (kUseGin and kMegaMoeGinCombineOverlap) {
            if (use_gin_direct_reduce) {
                // All packet-input TMA loads have completed their mbarrier
                // waits, including on CTAs with no output tokens. Move only
                // the epilogue side of the EXISTING second handoff: dispatch
                // then cleans and joins its old grid/world cleanup barriers,
                // which now also retire every rank's packet consumers.
                asm volatile("" ::: "memory");
                ptx::sync_unaligned(
                    kNumDispatchThreads + kNumEpilogueThreads,
                    kDispatchWithEpilogueBarrierIdx);
                asm volatile("" ::: "memory");
            }
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

} // namespace deep_gemm

#undef DG_GIN_TRACE_IF
#undef DG_GIN_TRACE_FIRST_IF
