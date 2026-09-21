#pragma once

#include <deep_gemm/comm/mega_moe_gin.cuh>
#include <deep_gemm/ptx/ld_st.cuh>

#ifdef DG_MEGAMOE_GIN
#if NCCL_CHECK_CUDACC

namespace deep_gemm::comm {

// Shared by the FP8/FP4 and BF16 kernels. These helpers own only transport;
// each caller retains its native warp roles, scheduling and phase barriers.
template <typename GinWorkspace>
CUTLASS_DEVICE void mega_moe_gin_pack_dispatch(
    const MegaMoeGinTransport& gin_transport,
    const GinWorkspace& gin_workspace,
    const layout::MegaMoEBuffer& buffer,
    const uint32_t remote_owner_slot, const uint32_t num_tokens) {
    constexpr uint32_t kNumExpertsPerRank = GinWorkspace::kNumExpertsPerRank;
    const auto& workspace = buffer.workspace;
    const uint32_t remote_owner_rank =
        gin_transport.remote_rank(remote_owner_slot);
    auto* packed_counts =
        gin_workspace.get_dispatch_send_expert_count_ptr(remote_owner_slot);
    auto* packed_assignments =
        gin_workspace.get_dispatch_send_expert_assignment_ptr(remote_owner_slot);
    uint32_t num_expert_assignments = 0;
    for (uint32_t local_expert = 0;
         local_expert < kNumExpertsPerRank; ++local_expert) {
        const uint32_t global_expert =
            remote_owner_rank * kNumExpertsPerRank + local_expert;
        const uint64_t expert_status =
            *workspace.get_expert_send_count_ptr(global_expert);
        const uint32_t count = static_cast<uint32_t>(expert_status);
        packed_counts[local_expert] = count;
        DG_DEVICE_ASSERT(count <= GinWorkspace::kMaxActiveTokens);
        DG_DEVICE_ASSERT(
            num_expert_assignments + count <=
            GinWorkspace::kMaxExpertAssignments);
        for (uint32_t assignment_idx = 0;
             assignment_idx < count; ++assignment_idx) {
            packed_assignments[num_expert_assignments + assignment_idx] =
                *gin_workspace.get_staged_expert_assignment_ptr(
                    remote_owner_slot, local_expert, assignment_idx);
        }
        num_expert_assignments += count;
    }
    *gin_workspace.get_dispatch_send_total_assignment_count_ptr(
        remote_owner_slot) = num_expert_assignments;
    __threadfence_system();

    const uint32_t source_slot = gin_transport.return_slot(remote_owner_slot);
    comm::mega_moe_gin_publish_dispatch_control_async<
        GinWorkspace>(
        gin_transport,
        remote_owner_slot,
        num_expert_assignments,
        packed_counts,
        gin_workspace.get_dispatch_receive_expert_count_ptr(
            source_slot),
        gin_workspace.get_dispatch_receive_control_epoch_ptr(
            source_slot));
    comm::mega_moe_gin_publish_dispatch_payload_async<
        GinWorkspace>(
        gin_transport,
        remote_owner_slot,
        num_tokens, num_expert_assignments,
        buffer.input_token_buffer.get_base_ptr(),
        gin_workspace.get_remote_input_activation_ptr(source_slot),
        buffer.input_sf_buffer.get_base_ptr(),
        gin_workspace.get_remote_input_scale_ptr(source_slot),
        buffer.input_topk_weights_buffer.get_base_ptr(),
        gin_workspace.get_remote_topk_weight_ptr(source_slot),
        gin_workspace.get_dispatch_receive_payload_epoch_ptr(
            source_slot));
}

template <typename GinWorkspace>
CUTLASS_DEVICE void mega_moe_gin_unpack_dispatch(
    const MegaMoeGinTransport& gin_transport,
    const GinWorkspace& gin_workspace,
    const layout::Workspace& workspace,
    const uint32_t remote_source_slot, const uint32_t lane_idx) {
    constexpr uint32_t kNumExpertsPerRank = GinWorkspace::kNumExpertsPerRank;
    if (lane_idx == 0)
        *gin_workspace.get_dispatch_payload_acquired_epoch_ptr(
            remote_source_slot) = 0;
    const uint32_t remote_source_rank =
        gin_transport.remote_rank(remote_source_slot);
    if (lane_idx == 0) {
        comm::mega_moe_gin_wait_dispatch(
            gin_transport,
            gin_workspace.get_dispatch_receive_control_epoch_ptr(
                remote_source_slot),
            *gin_workspace.get_dispatch_epoch_ptr());
    }
    __syncwarp();

    const auto* packed_counts =
        gin_workspace.get_dispatch_receive_expert_count_ptr(
            remote_source_slot);
    const auto* packed_assignments =
        gin_workspace.get_dispatch_receive_expert_assignment_ptr(
            remote_source_slot);
    constexpr uint32_t kExpertUnpackRounds =
        math::constexpr_ceil_div(kNumExpertsPerRank, 32u);
    uint32_t assignment_base = 0;
    #pragma unroll
    for (uint32_t round = 0; round < kExpertUnpackRounds;
         ++round) {
        const uint32_t local_expert = round * 32 + lane_idx;
        const uint32_t count = local_expert < kNumExpertsPerRank ?
            packed_counts[local_expert] : 0u;
        const uint32_t inclusive =
            math::warp_inclusive_sum(count, lane_idx);
        const uint32_t prefix =
            assignment_base + inclusive - count;
        const uint32_t round_total =
            __shfl_sync(0xffffffffu, inclusive, 31);

        if (local_expert < kNumExpertsPerRank) {
            DG_DEVICE_ASSERT(
                count <= GinWorkspace::kMaxActiveTokens);
            DG_DEVICE_ASSERT(
                prefix + count <=
                GinWorkspace::kMaxExpertAssignments);
            *workspace.get_expert_recv_count_ptr(
                remote_source_rank, local_expert) = count;
            *gin_workspace.get_combine_record_prefix_ptr(
                remote_source_slot, local_expert) = prefix;
            auto* dst = workspace.get_src_token_topk_idx_ptr(
                local_expert, remote_source_rank, 0);
            for (uint32_t assignment_idx = 0;
                 assignment_idx < count; ++assignment_idx)
                dst[assignment_idx] =
                    packed_assignments[prefix + assignment_idx];
        }
        assignment_base += round_total;
    }
    DG_DEVICE_ASSERT(
        assignment_base <= GinWorkspace::kMaxExpertAssignments);
    __threadfence();
}

template <uint32_t BLOCK_M, uint32_t BLOCK_N, typename GinWorkspace>
CUTLASS_DEVICE void mega_moe_gin_drain_combine(
    const MegaMoeGinTransport& gin_transport,
    const GinWorkspace& gin_workspace,
    const layout::Workspace& workspace, const uint32_t lane_idx) {
    constexpr uint32_t L2_SHAPE_N = GinWorkspace::kHidden;
    constexpr uint32_t kNumExpertsPerRank = GinWorkspace::kNumExpertsPerRank;
    constexpr uint32_t kNumL2Fragments =
        L2_SHAPE_N / BLOCK_N;
    DG_STATIC_ASSERT(
        L2_SHAPE_N % BLOCK_N == 0,
        "L2 output must contain complete N fragments");
    constexpr uint32_t kExpertsPerOwnerWave =
        math::constexpr_ceil_div(
            kNumExpertsPerRank,
            layout::kMegaMoEGinCombineOwnerWaves);

    for (uint32_t peer_base = 0;
         peer_base < GinWorkspace::kNumRemotePeers; peer_base += 32) {
        const bool has_remote_source =
            peer_base + lane_idx < GinWorkspace::kNumRemotePeers;
        const uint32_t remote_source_slot =
            has_remote_source ? peer_base + lane_idx : 0u;
        const uint32_t remote_source_rank =
            gin_transport.remote_rank(remote_source_slot);
        const uint32_t owner_return_slot =
            gin_transport.return_slot(remote_source_slot);
        uint32_t pending_owner_waves = 0;
        uint32_t total_expert_assignments = 0;

        if (has_remote_source) {
            #pragma unroll
            for (uint32_t owner_wave = 0;
                 owner_wave <
                     layout::kMegaMoEGinCombineOwnerWaves;
                 ++owner_wave) {
                uint32_t wave_expert_assignments = 0;
                for (uint32_t wave_expert = 0;
                     wave_expert < kExpertsPerOwnerWave;
                     ++wave_expert) {
                    const uint32_t local_expert =
                        owner_wave * kExpertsPerOwnerWave +
                        wave_expert;
                    if (local_expert >= kNumExpertsPerRank)
                        continue;
                    wave_expert_assignments += static_cast<uint32_t>(
                        *workspace.get_expert_recv_count_ptr(
                            remote_source_rank, local_expert));
                }
                if (wave_expert_assignments != 0)
                    pending_owner_waves |= 1u << owner_wave;
                total_expert_assignments += wave_expert_assignments;
            }
            DG_DEVICE_ASSERT(
                total_expert_assignments <=
                GinWorkspace::kMaxExpertAssignments);
        }
        __syncwarp();

        // Different peer lanes may discover and submit different
        // owner waves in the same iteration. A wave is eligible only
        // after this issuer has acquired every contributing expert's
        // complete M-block x N-fragment target.
        while (__any_sync(
                0xffffffffu, pending_owner_waves != 0)) {
            uint32_t ready_owner_waves = 0;
            if (has_remote_source) {
                #pragma unroll
                for (uint32_t owner_wave = 0;
                     owner_wave <
                         layout::kMegaMoEGinCombineOwnerWaves;
                     ++owner_wave) {
                    const uint32_t owner_wave_bit = 1u << owner_wave;
                    if ((pending_owner_waves & owner_wave_bit) == 0)
                        continue;
                    bool wave_ready = true;
                    for (uint32_t wave_expert = 0;
                         wave_expert < kExpertsPerOwnerWave;
                         ++wave_expert) {
                        const uint32_t local_expert =
                            owner_wave * kExpertsPerOwnerWave +
                            wave_expert;
                        if (local_expert >= kNumExpertsPerRank)
                            continue;
                        const uint32_t source_assignments =
                            static_cast<uint32_t>(
                                *workspace
                                     .get_expert_recv_count_ptr(
                                         remote_source_rank,
                                         local_expert));
                        if (source_assignments == 0)
                            continue;
                        const uint32_t total_assignments =
                            static_cast<uint32_t>(
                                *workspace
                                     .get_expert_recv_count_sum_ptr(
                                         local_expert));
                        const uint32_t expected_fragments =
                            math::ceil_div(
                                total_assignments, BLOCK_M) *
                            kNumL2Fragments;
                        const uint32_t completed_fragments =
                            comm::mega_moe_gin_combine_completion_acquire(
                                gin_workspace
                                    .get_combine_expert_completion_ptr(
                                        local_expert));
                        DG_DEVICE_ASSERT(
                            completed_fragments <= expected_fragments);
                        wave_ready &=
                            completed_fragments == expected_fragments;
                    }
                    if (wave_ready)
                        ready_owner_waves |= owner_wave_bit;
                }
            }

            if (ready_owner_waves != 0) {
                const uint32_t owner_wave =
                    static_cast<uint32_t>(
                        __ffs(ready_owner_waves) - 1);
                const uint32_t owner_wave_bit = 1u << owner_wave;
                uint32_t span_record_ordinal = 0;
                uint32_t span_records = 0;
                for (uint32_t wave_expert = 0;
                     wave_expert < kExpertsPerOwnerWave;
                     ++wave_expert) {
                    const uint32_t local_expert =
                        owner_wave * kExpertsPerOwnerWave +
                        wave_expert;
                    if (local_expert >= kNumExpertsPerRank)
                        continue;
                    const uint32_t count = static_cast<uint32_t>(
                        *workspace.get_expert_recv_count_ptr(
                            remote_source_rank, local_expert));
                    if (count == 0)
                        continue;
                    const uint32_t prefix =
                        *gin_workspace.get_combine_record_prefix_ptr(
                            remote_source_slot, local_expert);
                    if (span_records == 0) {
                        span_record_ordinal = prefix;
                    } else {
                        DG_DEVICE_ASSERT(
                            prefix ==
                            span_record_ordinal + span_records);
                    }
                    span_records += count;
                }
                DG_DEVICE_ASSERT(span_records != 0);
                DG_DEVICE_ASSERT(
                    span_record_ordinal + span_records <=
                    total_expert_assignments);

                auto* local_span =
                    gin_workspace.get_combine_send_record_ptr(
                        remote_source_slot,
                        span_record_ordinal);
                auto* remote_span =
                    gin_workspace.get_combine_receive_record_ptr(
                        owner_return_slot,
                        span_record_ordinal);
                comm::mega_moe_gin_publish_combine_span_async<
                    GinWorkspace>(
                    gin_transport, remote_source_slot,
                    local_span, remote_span, span_records);
                pending_owner_waves &= ~owner_wave_bit;
            }
            __syncwarp();
        }

        if (has_remote_source and
            total_expert_assignments != 0) {
            comm::mega_moe_gin_flush_wait_combine_peer(
                gin_transport, remote_source_slot);
        }
        __syncwarp();
    }
}

template <uint32_t kNumSMs, uint32_t kNumEpilogueWarps, typename GinWorkspace>
CUTLASS_DEVICE void mega_moe_gin_scatter_combine(
    const GinWorkspace& gin_workspace, const layout::MegaMoEBuffer& buffer,
    const uint32_t num_tokens, const uint32_t sm_idx,
    const uint32_t epilogue_warp_idx, const uint32_t lane_idx) {
    constexpr uint32_t kNumHiddenBytes = GinWorkspace::kOutputActivationBytes;
    constexpr uint32_t kNumTopk = GinWorkspace::kNumTopK;
    constexpr uint32_t kCombineRecordUint4 =
        kNumHiddenBytes / sizeof(uint4);
    constexpr uint32_t kNumGlobalEpilogueWarps =
        kNumSMs * kNumEpilogueWarps;
    const uint32_t global_epilogue_warp =
        sm_idx * kNumEpilogueWarps + epilogue_warp_idx;
    for (uint32_t remote_owner_slot = 0;
         remote_owner_slot < GinWorkspace::kNumRemotePeers;
         ++remote_owner_slot) {
        // The remote owner returns records in the expert-major order
        // of this immutable dispatch-send packet. It therefore needs
        // neither a count header nor per-record destination metadata.
        const uint32_t received_count = ptx::ld_acq_sys(
            gin_workspace.get_dispatch_send_total_assignment_count_ptr(
                remote_owner_slot));
        DG_DEVICE_ASSERT(
            received_count <=
            GinWorkspace::kMaxExpertAssignments);
        for (uint32_t record_ordinal = global_epilogue_warp;
             record_ordinal < received_count;
             record_ordinal += kNumGlobalEpilogueWarps) {
            const uint32_t destination = ptx::ld_acq_sys(
                gin_workspace.get_dispatch_send_expert_assignment_ptr(
                    remote_owner_slot, record_ordinal));
            DG_DEVICE_ASSERT(destination < num_tokens * kNumTopk);
            const uint32_t dst_token_idx = destination / kNumTopk;
            const uint32_t dst_topk_idx = destination % kNumTopk;
            const auto* src = static_cast<const uint4*>(
                gin_workspace.get_combine_receive_record_ptr(
                    remote_owner_slot, record_ordinal));
            auto* dst = buffer.combine_token_buffer
                .get_rank_buffer(dst_topk_idx)
                .get_data_buffer(dst_token_idx)
                .template get_base_ptr<uint4>();
            #pragma unroll
            for (uint32_t element = lane_idx;
                 element < kCombineRecordUint4;
                 element += 32)
                dst[element] = src[element];
            __syncwarp();
        }
    }
}

} // namespace deep_gemm::comm

#endif
#endif
