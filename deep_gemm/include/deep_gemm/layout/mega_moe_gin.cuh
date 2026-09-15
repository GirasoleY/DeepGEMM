#pragma once

#include <cstddef>
#include <cstdint>

#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/layout/mega_moe.cuh>

namespace deep_gemm::layout {

// This layout is intentionally a single model specialization.  Runtime token
// capacity in the native MegaMoE allocation may be larger, but the GIN path
// accepts at most this many active source tokens.
static constexpr uint32_t kMegaMoEEp8GinWorldSize = 8;
static constexpr uint32_t kMegaMoEEp8GinLsaSize = 4;
static constexpr uint32_t kMegaMoEEp8GinNumRemotePeers = 4;
static constexpr uint32_t kMegaMoEEp8GinNumExperts = 448;
static constexpr uint32_t kMegaMoEEp8GinNumExpertsPerRank = 56;
static constexpr uint32_t kMegaMoEEp8GinCombineOwnerWaves = 4;
static constexpr uint32_t kMegaMoEEp8GinExpertsPerOwnerWave =
    kMegaMoEEp8GinNumExpertsPerRank / kMegaMoEEp8GinCombineOwnerWaves;
static constexpr uint32_t kMegaMoEEp8GinTopK = 16;
static constexpr uint32_t kMegaMoEEp8GinHidden = 3584;
static constexpr uint32_t kMegaMoEEp8GinIntermediateHidden = 3072;
static constexpr uint32_t kMegaMoEEp8GinMaxTokens = 48;
static constexpr uint32_t kMegaMoEEp8GinMaxExpertAssignments =
    kMegaMoEEp8GinMaxTokens * kMegaMoEEp8GinTopK;
static constexpr uint32_t kMegaMoEEp8GinMaxPoolTokens =
    get_num_max_pool_tokens<uint32_t>(
        kMegaMoEEp8GinWorldSize,
        kMegaMoEEp8GinMaxTokens,
        kMegaMoEEp8GinTopK,
        kMegaMoEEp8GinNumExpertsPerRank);
static constexpr uint32_t kMegaMoEEp8GinAlignment = 128;

static constexpr uint32_t kMegaMoEEp8GinInputActivationBytes =
    kMegaMoEEp8GinHidden;
static constexpr uint32_t kMegaMoEEp8GinInputScaleBytes =
    kMegaMoEEp8GinHidden / 32;
static constexpr uint32_t kMegaMoEEp8GinTopKWeightBytes =
    kMegaMoEEp8GinTopK * sizeof(float);
static constexpr uint32_t kMegaMoEEp8GinOutputActivationBytes =
    kMegaMoEEp8GinHidden * sizeof(uint16_t);

// A dispatch packet's first 16 bytes are reserved for the cumulative remote
// ready epoch. The send packet retains its first uint32_t as the matching
// combine-record count through end-of-launch cleanup; no NIC signals it.
static constexpr uint32_t kMegaMoEEp8GinDispatchReadyBytes = 16;
static constexpr uint32_t kMegaMoEEp8GinDispatchExpertCountBytes =
    kMegaMoEEp8GinNumExpertsPerRank * sizeof(uint32_t);
static constexpr uint32_t kMegaMoEEp8GinDispatchExpertAssignmentBytes =
    kMegaMoEEp8GinMaxExpertAssignments * sizeof(uint32_t);
static constexpr uint32_t kMegaMoEEp8GinDispatchPacketDataBytes =
    kMegaMoEEp8GinDispatchReadyBytes +
    kMegaMoEEp8GinDispatchExpertCountBytes +
    kMegaMoEEp8GinDispatchExpertAssignmentBytes;
static constexpr uint32_t kMegaMoEEp8GinDispatchPacketBytes =
    math::constexpr_align(
        kMegaMoEEp8GinDispatchPacketDataBytes,
        kMegaMoEEp8GinAlignment);

// Return records retain the expert-major ordinal of the dispatch assignment.
// The source therefore derives both the record count and its flattened
// token/top-k destination from its immutable dispatch-send packet.
static constexpr uint32_t kMegaMoEEp8GinCombineRecordBytes =
    math::constexpr_align(
        kMegaMoEEp8GinOutputActivationBytes,
        kMegaMoEEp8GinAlignment);
static constexpr uint32_t kMegaMoEEp8GinCombinePacketBytes =
    kMegaMoEEp8GinMaxExpertAssignments *
        kMegaMoEEp8GinCombineRecordBytes;

static_assert(kMegaMoEEp8GinWorldSize == 2 * kMegaMoEEp8GinLsaSize);
static_assert(kMegaMoEEp8GinNumRemotePeers == kMegaMoEEp8GinLsaSize);
static_assert(kMegaMoEEp8GinNumExperts ==
              kMegaMoEEp8GinWorldSize *
                  kMegaMoEEp8GinNumExpertsPerRank);
static_assert(
    kMegaMoEEp8GinNumExpertsPerRank %
        kMegaMoEEp8GinCombineOwnerWaves == 0);
static_assert(kMegaMoEEp8GinExpertsPerOwnerWave == 14);
static_assert(kMegaMoEEp8GinDispatchPacketBytes == 3328);
static_assert(kMegaMoEEp8GinCombineRecordBytes == 7168);
static_assert(kMegaMoEEp8GinCombinePacketBytes == 5505024);
static_assert(kMegaMoEEp8GinMaxPoolTokens == 21120);

// Registered transport storage appended after an unmodified MegaMoEBuffer.
// Persistent NIC-updated epochs are initialized once with the allocation and
// are never reset between launches.  All other arrays are ordinary per-launch
// scratch owned by a serialized stream.
struct MegaMoeEp8GinWorkspace {
    void* base = nullptr;

    static constexpr uint64_t kDispatchEpochOffset = 0;
    // A designated local waiter acquires each source's payload StrongVA
    // terminal once, then publishes that generation here for all pull CTAs.
    // These bridge words are local-only: the NIC never reads or writes them.
    static constexpr uint64_t kDispatchPayloadAcquiredEpochOffset = 16;
    static constexpr uint64_t kDispatchPayloadAcquiredEpochBytes =
        static_cast<uint64_t>(kMegaMoEEp8GinNumRemotePeers) *
        sizeof(uint64_t);
    static constexpr uint64_t kControlBytes = kMegaMoEEp8GinAlignment;

    static constexpr uint64_t kStagedExpertAssignmentOffset = kControlBytes;
    static constexpr uint64_t kStagedExpertAssignmentBytes =
        static_cast<uint64_t>(kMegaMoEEp8GinNumRemotePeers) *
        kMegaMoEEp8GinNumExpertsPerRank * kMegaMoEEp8GinMaxTokens *
        sizeof(uint32_t);

    static constexpr uint64_t kDispatchPacketOffset =
        kStagedExpertAssignmentOffset + kStagedExpertAssignmentBytes;
    static constexpr uint64_t kDispatchPacketStorageBytes =
        2ull * kMegaMoEEp8GinNumRemotePeers *
        kMegaMoEEp8GinDispatchPacketBytes;

    static constexpr uint64_t kRemoteInputActivationOffset =
        kDispatchPacketOffset + kDispatchPacketStorageBytes;
    static constexpr uint64_t kRemoteInputActivationBytes =
        static_cast<uint64_t>(kMegaMoEEp8GinNumRemotePeers) *
        kMegaMoEEp8GinMaxTokens * kMegaMoEEp8GinInputActivationBytes;

    static constexpr uint64_t kRemoteInputScaleOffset =
        kRemoteInputActivationOffset + kRemoteInputActivationBytes;
    static constexpr uint64_t kRemoteInputScaleStorageBytes =
        static_cast<uint64_t>(kMegaMoEEp8GinNumRemotePeers) *
        kMegaMoEEp8GinMaxTokens * kMegaMoEEp8GinInputScaleBytes;

    static constexpr uint64_t kRemoteTopKWeightOffset =
        kRemoteInputScaleOffset + kRemoteInputScaleStorageBytes;
    static constexpr uint64_t kRemoteTopKWeightStorageBytes =
        static_cast<uint64_t>(kMegaMoEEp8GinNumRemotePeers) *
        kMegaMoEEp8GinMaxTokens * kMegaMoEEp8GinTopKWeightBytes;

    static constexpr uint64_t kCombineRecordPrefixOffset =
        kRemoteTopKWeightOffset + kRemoteTopKWeightStorageBytes;
    static constexpr uint64_t kCombineRecordPrefixBytes =
        static_cast<uint64_t>(kMegaMoEEp8GinNumRemotePeers) *
        kMegaMoEEp8GinNumExpertsPerRank * sizeof(uint32_t);

    static constexpr uint64_t kCombinePacketOffset =
        kCombineRecordPrefixOffset + kCombineRecordPrefixBytes;
    static constexpr uint64_t kCombinePacketStorageBytes =
        2ull * kMegaMoEEp8GinNumRemotePeers *
        kMegaMoEEp8GinCombinePacketBytes;

    static constexpr uint64_t kCombineReturnOrdinalOffset =
        kCombinePacketOffset + kCombinePacketStorageBytes;
    static constexpr uint64_t kCombineExpertCompletionOffset =
        kCombineReturnOrdinalOffset +
            static_cast<uint64_t>(kMegaMoEEp8GinMaxPoolTokens) *
                sizeof(uint32_t);
    static constexpr uint64_t kCombineExpertCompletionBytes =
        math::constexpr_align(
            static_cast<uint64_t>(kMegaMoEEp8GinNumExpertsPerRank) *
                sizeof(uint32_t),
            static_cast<uint64_t>(kMegaMoEEp8GinAlignment));
    static constexpr uint64_t kNumBytes = math::constexpr_align(
        kCombineExpertCompletionOffset + kCombineExpertCompletionBytes,
        static_cast<uint64_t>(kMegaMoEEp8GinAlignment));

    static_assert(
        kStagedExpertAssignmentOffset % kMegaMoEEp8GinAlignment == 0);
    static_assert(
        kDispatchPayloadAcquiredEpochOffset +
            kDispatchPayloadAcquiredEpochBytes <=
        kControlBytes);
    static_assert(kDispatchPacketOffset % kMegaMoEEp8GinAlignment == 0);
    static_assert(
        kRemoteInputActivationOffset % kMegaMoEEp8GinAlignment == 0);
    static_assert(kRemoteInputScaleOffset % kMegaMoEEp8GinAlignment == 0);
    static_assert(kRemoteTopKWeightOffset % kMegaMoEEp8GinAlignment == 0);
    static_assert(kCombineRecordPrefixOffset % kMegaMoEEp8GinAlignment == 0);
    static_assert(kCombinePacketOffset % kMegaMoEEp8GinAlignment == 0);
    static_assert(kCombineReturnOrdinalOffset % kMegaMoEEp8GinAlignment == 0);
    static_assert(
        kCombineExpertCompletionOffset % kMegaMoEEp8GinAlignment == 0);
    static_assert(kCombineExpertCompletionBytes == 256);
    static_assert(kNumBytes == 44917504);

    MegaMoeEp8GinWorkspace() = default;

    CUTLASS_HOST_DEVICE
    explicit MegaMoeEp8GinWorkspace(void* aligned_base): base(aligned_base) {
        DG_UNIFIED_ASSERT(
            reinterpret_cast<uintptr_t>(base) %
                kMegaMoEEp8GinAlignment == 0);
    }

    CUTLASS_HOST_DEVICE
    static MegaMoeEp8GinWorkspace from_native(
            const MegaMoEBuffer& native) {
        const auto native_bytes = static_cast<uint64_t>(native.get_num_bytes());
        auto* aligned_base = math::advance_ptr(
            native.workspace.signals,
            math::align<uint64_t>(native_bytes, kMegaMoEEp8GinAlignment));
        return MegaMoeEp8GinWorkspace(aligned_base);
    }

    CUTLASS_HOST_DEVICE
    uint64_t* get_dispatch_epoch_ptr() const {
        return math::advance_ptr<uint64_t>(base, kDispatchEpochOffset);
    }

    CUTLASS_HOST_DEVICE
    uint64_t* get_dispatch_payload_acquired_epoch_ptr(
            const uint32_t remote_source_lane) const {
        DG_UNIFIED_ASSERT(
            remote_source_lane < kMegaMoEEp8GinNumRemotePeers);
        return math::advance_ptr<uint64_t>(
                   base, kDispatchPayloadAcquiredEpochOffset) +
               remote_source_lane;
    }

    CUTLASS_HOST_DEVICE
    uint32_t* get_combine_expert_completion_ptr(
            const uint32_t local_expert) const {
        DG_UNIFIED_ASSERT(local_expert < kMegaMoEEp8GinNumExpertsPerRank);
        return math::advance_ptr<uint32_t>(
                   base, kCombineExpertCompletionOffset) +
               local_expert;
    }

    CUTLASS_HOST_DEVICE
    uint32_t* get_staged_expert_assignment_ptr(
            const uint32_t remote_owner_lane,
            const uint32_t local_expert,
            const uint32_t assignment_idx = 0) const {
        DG_UNIFIED_ASSERT(remote_owner_lane < kMegaMoEEp8GinNumRemotePeers);
        DG_UNIFIED_ASSERT(local_expert < kMegaMoEEp8GinNumExpertsPerRank);
        DG_UNIFIED_ASSERT(assignment_idx < kMegaMoEEp8GinMaxTokens);
        const uint64_t index =
            (static_cast<uint64_t>(remote_owner_lane) *
                 kMegaMoEEp8GinNumExpertsPerRank +
             local_expert) *
                kMegaMoEEp8GinMaxTokens +
            assignment_idx;
        return math::advance_ptr<uint32_t>(
                   base, kStagedExpertAssignmentOffset) +
               index;
    }

    CUTLASS_HOST_DEVICE
    void* get_dispatch_send_packet_ptr(
            const uint32_t remote_owner_lane) const {
        return get_dispatch_packet_ptr(/*send=*/ true, remote_owner_lane);
    }

    CUTLASS_HOST_DEVICE
    void* get_dispatch_receive_packet_ptr(
            const uint32_t remote_source_lane) const {
        return get_dispatch_packet_ptr(/*send=*/ false, remote_source_lane);
    }

    CUTLASS_HOST_DEVICE
    uint32_t* get_dispatch_send_total_assignment_count_ptr(
            const uint32_t remote_owner_lane) const {
        return static_cast<uint32_t*>(
            get_dispatch_send_packet_ptr(remote_owner_lane));
    }

    CUTLASS_HOST_DEVICE
    uint64_t* get_dispatch_receive_control_epoch_ptr(
            const uint32_t remote_source_lane) const {
        return static_cast<uint64_t*>(
            get_dispatch_receive_packet_ptr(remote_source_lane));
    }

    CUTLASS_HOST_DEVICE
    uint64_t* get_dispatch_receive_payload_epoch_ptr(
            const uint32_t remote_source_lane) const {
        static_assert(
            kMegaMoEEp8GinDispatchReadyBytes >= 2 * sizeof(uint64_t));
        return get_dispatch_receive_control_epoch_ptr(remote_source_lane) + 1;
    }

    CUTLASS_HOST_DEVICE
    uint32_t* get_dispatch_send_expert_count_ptr(
            const uint32_t remote_owner_lane) const {
        return math::advance_ptr<uint32_t>(
            get_dispatch_send_packet_ptr(remote_owner_lane),
            kMegaMoEEp8GinDispatchReadyBytes);
    }

    CUTLASS_HOST_DEVICE
    uint32_t* get_dispatch_receive_expert_count_ptr(
            const uint32_t remote_source_lane) const {
        return math::advance_ptr<uint32_t>(
            get_dispatch_receive_packet_ptr(remote_source_lane),
            kMegaMoEEp8GinDispatchReadyBytes);
    }

    CUTLASS_HOST_DEVICE
    uint32_t* get_dispatch_send_expert_assignment_ptr(
            const uint32_t remote_owner_lane,
            const uint32_t assignment_idx = 0) const {
        DG_UNIFIED_ASSERT(
            assignment_idx <= kMegaMoEEp8GinMaxExpertAssignments);
        return math::advance_ptr<uint32_t>(
                   get_dispatch_send_packet_ptr(remote_owner_lane),
                   kMegaMoEEp8GinDispatchReadyBytes +
                       kMegaMoEEp8GinDispatchExpertCountBytes) +
               assignment_idx;
    }

    CUTLASS_HOST_DEVICE
    uint32_t* get_dispatch_receive_expert_assignment_ptr(
            const uint32_t remote_source_lane,
            const uint32_t assignment_idx = 0) const {
        DG_UNIFIED_ASSERT(
            assignment_idx <= kMegaMoEEp8GinMaxExpertAssignments);
        return math::advance_ptr<uint32_t>(
                   get_dispatch_receive_packet_ptr(remote_source_lane),
                   kMegaMoEEp8GinDispatchReadyBytes +
                       kMegaMoEEp8GinDispatchExpertCountBytes) +
               assignment_idx;
    }

    CUTLASS_HOST_DEVICE
    void* get_remote_input_activation_ptr(
            const uint32_t remote_source_lane,
            const uint32_t token_idx = 0) const {
        return get_remote_input_record_ptr(
            kRemoteInputActivationOffset,
            kMegaMoEEp8GinInputActivationBytes,
            remote_source_lane, token_idx);
    }

    CUTLASS_HOST_DEVICE
    void* get_remote_input_scale_ptr(
            const uint32_t remote_source_lane,
            const uint32_t token_idx = 0) const {
        return get_remote_input_record_ptr(
            kRemoteInputScaleOffset,
            kMegaMoEEp8GinInputScaleBytes,
            remote_source_lane, token_idx);
    }

    CUTLASS_HOST_DEVICE
    void* get_remote_topk_weight_ptr(
            const uint32_t remote_source_lane,
            const uint32_t token_idx = 0) const {
        return get_remote_input_record_ptr(
            kRemoteTopKWeightOffset,
            kMegaMoEEp8GinTopKWeightBytes,
            remote_source_lane, token_idx);
    }

    CUTLASS_HOST_DEVICE
    uint32_t* get_combine_record_prefix_ptr(
            const uint32_t remote_source_lane,
            const uint32_t local_expert = 0) const {
        DG_UNIFIED_ASSERT(remote_source_lane < kMegaMoEEp8GinNumRemotePeers);
        DG_UNIFIED_ASSERT(local_expert < kMegaMoEEp8GinNumExpertsPerRank);
        const uint64_t index =
            static_cast<uint64_t>(remote_source_lane) *
                kMegaMoEEp8GinNumExpertsPerRank +
            local_expert;
        return math::advance_ptr<uint32_t>(
                   base, kCombineRecordPrefixOffset) +
               index;
    }

    CUTLASS_HOST_DEVICE
    uint32_t* get_combine_return_ordinal_ptr(
            const uint32_t pool_token_idx) const {
        DG_UNIFIED_ASSERT(pool_token_idx < kMegaMoEEp8GinMaxPoolTokens);
        return math::advance_ptr<uint32_t>(
                   base, kCombineReturnOrdinalOffset) +
               pool_token_idx;
    }

    CUTLASS_HOST_DEVICE
    void* get_combine_send_record_ptr(
            const uint32_t remote_source_lane,
            const uint32_t record_ordinal) const {
        return get_combine_record_ptr(
            /*send=*/ true, remote_source_lane, record_ordinal);
    }

    CUTLASS_HOST_DEVICE
    void* get_combine_receive_record_ptr(
            const uint32_t remote_owner_lane,
            const uint32_t record_ordinal) const {
        return get_combine_record_ptr(
            /*send=*/ false, remote_owner_lane, record_ordinal);
    }

private:
    CUTLASS_HOST_DEVICE
    void* get_dispatch_packet_ptr(
            const bool send, const uint32_t peer_lane) const {
        DG_UNIFIED_ASSERT(peer_lane < kMegaMoEEp8GinNumRemotePeers);
        const uint32_t packet_idx =
            (send ? 0u : kMegaMoEEp8GinNumRemotePeers) + peer_lane;
        return math::advance_ptr(
            base,
            kDispatchPacketOffset +
                static_cast<uint64_t>(packet_idx) *
                    kMegaMoEEp8GinDispatchPacketBytes);
    }

    CUTLASS_HOST_DEVICE
    void* get_remote_input_record_ptr(
            const uint64_t storage_offset,
            const uint32_t record_bytes,
            const uint32_t remote_source_lane,
            const uint32_t token_idx) const {
        DG_UNIFIED_ASSERT(remote_source_lane < kMegaMoEEp8GinNumRemotePeers);
        DG_UNIFIED_ASSERT(token_idx < kMegaMoEEp8GinMaxTokens);
        const uint64_t record_idx =
            static_cast<uint64_t>(remote_source_lane) *
                kMegaMoEEp8GinMaxTokens +
            token_idx;
        return math::advance_ptr(
            base, storage_offset + record_idx * record_bytes);
    }

    CUTLASS_HOST_DEVICE
    void* get_combine_packet_ptr(
            const bool send, const uint32_t peer_lane) const {
        DG_UNIFIED_ASSERT(peer_lane < kMegaMoEEp8GinNumRemotePeers);
        const uint32_t packet_idx =
            (send ? 0u : kMegaMoEEp8GinNumRemotePeers) + peer_lane;
        return math::advance_ptr(
            base,
            kCombinePacketOffset +
                static_cast<uint64_t>(packet_idx) *
                    kMegaMoEEp8GinCombinePacketBytes);
    }

    CUTLASS_HOST_DEVICE
    void* get_combine_record_ptr(
            const bool send,
            const uint32_t peer_lane,
            const uint32_t record_ordinal) const {
        DG_UNIFIED_ASSERT(
            record_ordinal < kMegaMoEEp8GinMaxExpertAssignments);
        return math::advance_ptr(
            get_combine_packet_ptr(send, peer_lane),
            static_cast<uint64_t>(record_ordinal) *
                kMegaMoEEp8GinCombineRecordBytes);
    }
};

} // namespace deep_gemm::layout
