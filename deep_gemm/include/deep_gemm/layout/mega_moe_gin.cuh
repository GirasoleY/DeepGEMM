#pragma once

#include <cstddef>
#include <cstdint>

#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/layout/mega_moe.cuh>

namespace deep_gemm::layout {

// Defaults preserve the original EP8/two-LSA4 specialization. Model geometry,
// dtype, world size and the contiguous LSA partition are specialization inputs.
static constexpr uint32_t kMegaMoEGinCombineOwnerWaves = 4;
static constexpr uint32_t kMegaMoEGinAlignment = 128;
static constexpr uint32_t kMegaMoEGinDispatchReadyBytes = 16;

// One constexpr calculator is shared by runtime allocation sizing and the
// compile-time workspace specialization. Every storage boundary is aligned
// explicitly so arbitrary supported geometries do not rely on the default
// model's coincidental divisibility.
struct MegaMoeGinLayout {
    uint32_t num_experts_per_rank;
    uint32_t num_topk;
    uint32_t hidden;
    uint32_t max_active_tokens;
    uint32_t world_size;
    uint32_t lsa_size;
    bool with_sf;

    CUTLASS_HOST_DEVICE constexpr MegaMoeGinLayout(
            const uint32_t num_experts_per_rank,
            const uint32_t num_topk,
            const uint32_t hidden,
            const uint32_t max_active_tokens,
            const uint32_t world_size = 8,
            const uint32_t lsa_size = 4,
            const bool with_sf = true)
        : num_experts_per_rank(num_experts_per_rank),
          num_topk(num_topk),
          hidden(hidden),
          max_active_tokens(max_active_tokens),
          world_size(world_size), lsa_size(lsa_size), with_sf(with_sf) {}

    CUTLASS_HOST_DEVICE constexpr uint32_t num_remote_peers() const {
        return world_size - lsa_size;
    }

    // This footer is located at the end of the underlying allocation, not the
    // shape-dependent tail. Aliases retain the same cumulative NIC counters.
    CUTLASS_HOST_DEVICE constexpr uint64_t persistent_bytes() const {
        return align((1ull + 2ull * num_remote_peers()) * sizeof(uint64_t));
    }

    CUTLASS_HOST_DEVICE constexpr uint64_t align(const uint64_t bytes) const {
        return math::constexpr_align(
            bytes, static_cast<uint64_t>(kMegaMoEGinAlignment));
    }

    CUTLASS_HOST_DEVICE constexpr uint64_t max_expert_assignments() const {
        // A source token can select a local expert at most once. Therefore one
        // peer packet needs at most min(top-k, experts-per-rank) assignments
        // per active source token.
        return static_cast<uint64_t>(max_active_tokens) *
               math::constexpr_min(
                   static_cast<uint64_t>(num_topk),
                   static_cast<uint64_t>(num_experts_per_rank));
    }

    CUTLASS_HOST_DEVICE constexpr uint64_t max_pool_tokens() const {
        return get_num_max_pool_tokens<uint64_t>(
            world_size,
            max_active_tokens,
            num_topk,
            num_experts_per_rank);
    }

    CUTLASS_HOST_DEVICE constexpr uint64_t input_activation_bytes() const {
        return static_cast<uint64_t>(hidden) * (with_sf ? 1u : 2u);
    }

    CUTLASS_HOST_DEVICE constexpr uint64_t input_scale_bytes() const {
        return with_sf ? hidden / 32 : 0;
    }

    CUTLASS_HOST_DEVICE constexpr uint64_t topk_weight_bytes() const {
        return static_cast<uint64_t>(num_topk) * sizeof(float);
    }

    CUTLASS_HOST_DEVICE constexpr uint64_t output_activation_bytes() const {
        return static_cast<uint64_t>(hidden) * sizeof(uint16_t);
    }

    CUTLASS_HOST_DEVICE constexpr uint64_t dispatch_expert_count_bytes() const {
        return static_cast<uint64_t>(num_experts_per_rank) *
               sizeof(uint32_t);
    }

    CUTLASS_HOST_DEVICE constexpr uint64_t
    dispatch_expert_assignment_bytes() const {
        return max_expert_assignments() * sizeof(uint32_t);
    }

    CUTLASS_HOST_DEVICE constexpr uint64_t dispatch_packet_bytes() const {
        return align(
            kMegaMoEGinDispatchReadyBytes +
            dispatch_expert_count_bytes() +
            dispatch_expert_assignment_bytes());
    }

    // Return records retain the expert-major ordinal of the dispatch
    // assignment. The source derives both the record count and flattened
    // token/top-k destination from its immutable dispatch-send packet.
    CUTLASS_HOST_DEVICE constexpr uint64_t combine_record_bytes() const {
        return align(output_activation_bytes());
    }

    CUTLASS_HOST_DEVICE constexpr uint64_t combine_packet_bytes() const {
        return max_expert_assignments() * combine_record_bytes();
    }

    CUTLASS_HOST_DEVICE constexpr uint64_t control_bytes() const {
        return align(16ull + num_remote_peers() * sizeof(uint64_t));
    }

    CUTLASS_HOST_DEVICE constexpr uint64_t staged_expert_assignment_offset() const {
        return control_bytes();
    }

    CUTLASS_HOST_DEVICE constexpr uint64_t staged_expert_assignment_bytes() const {
        return static_cast<uint64_t>(num_remote_peers()) *
               num_experts_per_rank * max_active_tokens * sizeof(uint32_t);
    }

    CUTLASS_HOST_DEVICE constexpr uint64_t dispatch_packet_offset() const {
        return align(staged_expert_assignment_offset() +
                     staged_expert_assignment_bytes());
    }

    CUTLASS_HOST_DEVICE constexpr uint64_t dispatch_packet_storage_bytes() const {
        return 2ull * num_remote_peers() * dispatch_packet_bytes();
    }

    CUTLASS_HOST_DEVICE constexpr uint64_t remote_input_activation_offset() const {
        return align(dispatch_packet_offset() +
                     dispatch_packet_storage_bytes());
    }

    CUTLASS_HOST_DEVICE constexpr uint64_t remote_input_activation_storage_bytes() const {
        return static_cast<uint64_t>(num_remote_peers()) *
               max_active_tokens * input_activation_bytes();
    }

    CUTLASS_HOST_DEVICE constexpr uint64_t remote_input_scale_offset() const {
        return align(remote_input_activation_offset() +
                     remote_input_activation_storage_bytes());
    }

    CUTLASS_HOST_DEVICE constexpr uint64_t remote_input_scale_storage_bytes() const {
        return static_cast<uint64_t>(num_remote_peers()) *
               max_active_tokens * input_scale_bytes();
    }

    CUTLASS_HOST_DEVICE constexpr uint64_t remote_topk_weight_offset() const {
        return align(remote_input_scale_offset() +
                     remote_input_scale_storage_bytes());
    }

    CUTLASS_HOST_DEVICE constexpr uint64_t remote_topk_weight_storage_bytes() const {
        return static_cast<uint64_t>(num_remote_peers()) *
               max_active_tokens * topk_weight_bytes();
    }

    CUTLASS_HOST_DEVICE constexpr uint64_t combine_record_prefix_offset() const {
        return align(remote_topk_weight_offset() +
                     remote_topk_weight_storage_bytes());
    }

    CUTLASS_HOST_DEVICE constexpr uint64_t combine_record_prefix_bytes() const {
        return static_cast<uint64_t>(num_remote_peers()) *
               num_experts_per_rank * sizeof(uint32_t);
    }

    CUTLASS_HOST_DEVICE constexpr uint64_t combine_packet_offset() const {
        return align(combine_record_prefix_offset() +
                     combine_record_prefix_bytes());
    }

    CUTLASS_HOST_DEVICE constexpr uint64_t combine_packet_storage_bytes() const {
        return 2ull * num_remote_peers() * combine_packet_bytes();
    }

    CUTLASS_HOST_DEVICE constexpr uint64_t combine_return_ordinal_offset() const {
        return align(combine_packet_offset() +
                     combine_packet_storage_bytes());
    }

    CUTLASS_HOST_DEVICE constexpr uint64_t combine_return_ordinal_bytes() const {
        return max_pool_tokens() * sizeof(uint32_t);
    }

    CUTLASS_HOST_DEVICE constexpr uint64_t
    combine_expert_completion_offset() const {
        return align(combine_return_ordinal_offset() +
                     combine_return_ordinal_bytes());
    }

    CUTLASS_HOST_DEVICE constexpr uint64_t
    combine_expert_completion_bytes() const {
        return align(
            static_cast<uint64_t>(num_experts_per_rank) * sizeof(uint32_t));
    }

    CUTLASS_HOST_DEVICE constexpr uint64_t num_bytes() const {
        return align(combine_expert_completion_offset() +
                     combine_expert_completion_bytes());
    }

    CUTLASS_HOST_DEVICE constexpr bool is_valid() const {
        constexpr uint64_t kMaxUint32 =
            static_cast<uint64_t>(~static_cast<uint32_t>(0));
        return world_size > 0 and world_size <= kNumMaxRanks and
               lsa_size > 0 and lsa_size <= world_size and
               world_size % lsa_size == 0 and
               num_experts_per_rank > 0 and num_topk > 0 and hidden > 0 and
               hidden % 32 == 0 and max_active_tokens > 0 and
               max_expert_assignments() <= kMaxUint32 and
               max_pool_tokens() <= kMaxUint32;
    }
};

// Registered transport storage appended after an unmodified MegaMoEBuffer.
// Persistent NIC-updated epochs live in a separate allocation footer so every
// geometry alias shares their addresses. The tail contains per-launch scratch
// owned by serialized launches across all aliases.
template <uint32_t kNumExpertsPerRank_,
          uint32_t kNumTopK_,
          uint32_t kHidden_,
          uint32_t kMaxActiveTokens_,
          uint32_t kWorldSize_ = 8,
          uint32_t kLsaSize_ = 4,
          bool kWithSf_ = true>
struct MegaMoeGinWorkspaceT {
    void* base = nullptr;
    void* persistent = nullptr;

    static constexpr uint32_t kWorldSize = kWorldSize_;
    static constexpr uint32_t kLsaSize = kLsaSize_;
    static constexpr uint32_t kNumRemotePeers = kWorldSize - kLsaSize;
    static constexpr bool kWithSf = kWithSf_;
    static constexpr uint32_t kNumExpertsPerRank = kNumExpertsPerRank_;
    static constexpr uint32_t kNumTopK = kNumTopK_;
    static constexpr uint32_t kHidden = kHidden_;
    static constexpr uint32_t kMaxActiveTokens = kMaxActiveTokens_;
    inline static constexpr MegaMoeGinLayout kLayout{
        kNumExpertsPerRank, kNumTopK, kHidden, kMaxActiveTokens,
        kWorldSize, kLsaSize, kWithSf};

    static_assert(kLayout.is_valid(), "Invalid GIN workspace geometry");
    static constexpr uint32_t kMaxExpertAssignments =
        static_cast<uint32_t>(kLayout.max_expert_assignments());
    static constexpr uint32_t kMaxPoolTokens =
        static_cast<uint32_t>(kLayout.max_pool_tokens());
    static constexpr uint64_t kInputActivationBytes =
        kLayout.input_activation_bytes();
    static constexpr uint64_t kInputScaleBytes =
        kLayout.input_scale_bytes();
    static constexpr uint64_t kTopKWeightBytes =
        kLayout.topk_weight_bytes();
    static constexpr uint64_t kOutputActivationBytes =
        kLayout.output_activation_bytes();
    static constexpr uint64_t kDispatchReadyBytes =
        kMegaMoEGinDispatchReadyBytes;
    static constexpr uint64_t kDispatchExpertCountBytes =
        kLayout.dispatch_expert_count_bytes();
    static constexpr uint64_t kDispatchExpertAssignmentBytes =
        kLayout.dispatch_expert_assignment_bytes();
    static constexpr uint64_t kDispatchPacketBytes =
        kLayout.dispatch_packet_bytes();
    static constexpr uint64_t kCombineRecordBytes =
        kLayout.combine_record_bytes();
    static constexpr uint64_t kCombinePacketBytes =
        kLayout.combine_packet_bytes();

    // A designated local waiter acquires each source's payload StrongVA
    // terminal once, then publishes that generation here for all pull CTAs.
    // These bridge words are local-only: the NIC never reads or writes them.
    static constexpr uint64_t kDispatchPayloadAcquiredEpochOffset = 16;
    static constexpr uint64_t kDispatchPayloadAcquiredEpochBytes =
        static_cast<uint64_t>(kNumRemotePeers) *
        sizeof(uint64_t);
    static constexpr uint64_t kControlBytes = kLayout.control_bytes();

    static constexpr uint64_t kStagedExpertAssignmentOffset =
        kLayout.staged_expert_assignment_offset();
    static constexpr uint64_t kStagedExpertAssignmentBytes =
        kLayout.staged_expert_assignment_bytes();

    static constexpr uint64_t kDispatchPacketOffset =
        kLayout.dispatch_packet_offset();
    static constexpr uint64_t kDispatchPacketStorageBytes =
        kLayout.dispatch_packet_storage_bytes();

    static constexpr uint64_t kRemoteInputActivationOffset =
        kLayout.remote_input_activation_offset();
    static constexpr uint64_t kRemoteInputActivationStorageBytes =
        kLayout.remote_input_activation_storage_bytes();

    static constexpr uint64_t kRemoteInputScaleOffset =
        kLayout.remote_input_scale_offset();
    static constexpr uint64_t kRemoteInputScaleStorageBytes =
        kLayout.remote_input_scale_storage_bytes();

    static constexpr uint64_t kRemoteTopKWeightOffset =
        kLayout.remote_topk_weight_offset();
    static constexpr uint64_t kRemoteTopKWeightStorageBytes =
        kLayout.remote_topk_weight_storage_bytes();

    static constexpr uint64_t kCombineRecordPrefixOffset =
        kLayout.combine_record_prefix_offset();
    static constexpr uint64_t kCombineRecordPrefixBytes =
        kLayout.combine_record_prefix_bytes();

    static constexpr uint64_t kCombinePacketOffset =
        kLayout.combine_packet_offset();
    static constexpr uint64_t kCombinePacketStorageBytes =
        kLayout.combine_packet_storage_bytes();

    static constexpr uint64_t kCombineReturnOrdinalOffset =
        kLayout.combine_return_ordinal_offset();
    static constexpr uint64_t kCombineExpertCompletionOffset =
        kLayout.combine_expert_completion_offset();
    static constexpr uint64_t kCombineExpertCompletionBytes =
        kLayout.combine_expert_completion_bytes();
    static constexpr uint64_t kNumBytes = kLayout.num_bytes();
    static constexpr uint64_t kPersistentBytes = kLayout.persistent_bytes();

    static_assert(
        kStagedExpertAssignmentOffset % kMegaMoEGinAlignment == 0);
    static_assert(
        kDispatchPayloadAcquiredEpochOffset +
            kDispatchPayloadAcquiredEpochBytes <=
        kControlBytes);
    static_assert(kDispatchPacketOffset % kMegaMoEGinAlignment == 0);
    static_assert(
        kRemoteInputActivationOffset % kMegaMoEGinAlignment == 0);
    static_assert(kRemoteInputScaleOffset % kMegaMoEGinAlignment == 0);
    static_assert(kRemoteTopKWeightOffset % kMegaMoEGinAlignment == 0);
    static_assert(kCombineRecordPrefixOffset % kMegaMoEGinAlignment == 0);
    static_assert(kCombinePacketOffset % kMegaMoEGinAlignment == 0);
    static_assert(kCombineReturnOrdinalOffset % kMegaMoEGinAlignment == 0);
    static_assert(
        kCombineExpertCompletionOffset % kMegaMoEGinAlignment == 0);
    MegaMoeGinWorkspaceT() = default;

    CUTLASS_HOST_DEVICE
    explicit MegaMoeGinWorkspaceT(void* aligned_base, void* persistent_base)
        : base(aligned_base), persistent(persistent_base) {
        DG_UNIFIED_ASSERT(
            reinterpret_cast<uintptr_t>(base) %
                kMegaMoEGinAlignment == 0);
    }

    CUTLASS_HOST_DEVICE
    static MegaMoeGinWorkspaceT from_native(
            const MegaMoEBuffer& native, void* persistent_base) {
        const auto native_bytes = static_cast<uint64_t>(native.get_num_bytes());
        auto* aligned_base = math::advance_ptr(
            native.workspace.signals,
            math::align<uint64_t>(native_bytes, kMegaMoEGinAlignment));
        return MegaMoeGinWorkspaceT(aligned_base, persistent_base);
    }

    CUTLASS_HOST_DEVICE
    uint64_t* get_dispatch_epoch_ptr() const {
        return static_cast<uint64_t*>(persistent);
    }

    CUTLASS_HOST_DEVICE
    uint64_t* get_dispatch_payload_acquired_epoch_ptr(
            const uint32_t remote_source_slot) const {
        DG_UNIFIED_ASSERT(
            remote_source_slot < kNumRemotePeers);
        return math::advance_ptr<uint64_t>(
                   base, kDispatchPayloadAcquiredEpochOffset) +
               remote_source_slot;
    }

    CUTLASS_HOST_DEVICE
    uint32_t* get_combine_expert_completion_ptr(
            const uint32_t local_expert) const {
        DG_UNIFIED_ASSERT(local_expert < kNumExpertsPerRank);
        return math::advance_ptr<uint32_t>(
                   base, kCombineExpertCompletionOffset) +
               local_expert;
    }

    CUTLASS_HOST_DEVICE
    uint32_t* get_staged_expert_assignment_ptr(
            const uint32_t remote_owner_slot,
            const uint32_t local_expert,
            const uint32_t assignment_idx = 0) const {
        DG_UNIFIED_ASSERT(remote_owner_slot < kNumRemotePeers);
        DG_UNIFIED_ASSERT(local_expert < kNumExpertsPerRank);
        DG_UNIFIED_ASSERT(assignment_idx < kMaxActiveTokens);
        const uint64_t index =
            (static_cast<uint64_t>(remote_owner_slot) *
                 kNumExpertsPerRank +
             local_expert) *
                kMaxActiveTokens +
            assignment_idx;
        return math::advance_ptr<uint32_t>(
                   base, kStagedExpertAssignmentOffset) +
               index;
    }

    CUTLASS_HOST_DEVICE
    void* get_dispatch_send_packet_ptr(
            const uint32_t remote_owner_slot) const {
        return get_dispatch_packet_ptr(/*send=*/ true, remote_owner_slot);
    }

    CUTLASS_HOST_DEVICE
    void* get_dispatch_receive_packet_ptr(
            const uint32_t remote_source_slot) const {
        return get_dispatch_packet_ptr(/*send=*/ false, remote_source_slot);
    }

    CUTLASS_HOST_DEVICE
    uint32_t* get_dispatch_send_total_assignment_count_ptr(
            const uint32_t remote_owner_slot) const {
        return static_cast<uint32_t*>(
            get_dispatch_send_packet_ptr(remote_owner_slot));
    }

    CUTLASS_HOST_DEVICE
    uint64_t* get_dispatch_receive_control_epoch_ptr(
            const uint32_t remote_source_slot) const {
        DG_UNIFIED_ASSERT(remote_source_slot < kNumRemotePeers);
        return static_cast<uint64_t*>(persistent) + 1 + remote_source_slot;
    }

    CUTLASS_HOST_DEVICE
    uint64_t* get_dispatch_receive_payload_epoch_ptr(
            const uint32_t remote_source_slot) const {
        static_assert(
            kDispatchReadyBytes >= 2 * sizeof(uint64_t));
        return get_dispatch_receive_control_epoch_ptr(remote_source_slot) + kNumRemotePeers;
    }

    CUTLASS_HOST_DEVICE
    uint32_t* get_dispatch_send_expert_count_ptr(
            const uint32_t remote_owner_slot) const {
        return math::advance_ptr<uint32_t>(
            get_dispatch_send_packet_ptr(remote_owner_slot),
            kDispatchReadyBytes);
    }

    CUTLASS_HOST_DEVICE
    uint32_t* get_dispatch_receive_expert_count_ptr(
            const uint32_t remote_source_slot) const {
        return math::advance_ptr<uint32_t>(
            get_dispatch_receive_packet_ptr(remote_source_slot),
            kDispatchReadyBytes);
    }

    CUTLASS_HOST_DEVICE
    uint32_t* get_dispatch_send_expert_assignment_ptr(
            const uint32_t remote_owner_slot,
            const uint32_t assignment_idx = 0) const {
        DG_UNIFIED_ASSERT(
            assignment_idx <= kMaxExpertAssignments);
        return math::advance_ptr<uint32_t>(
                   get_dispatch_send_packet_ptr(remote_owner_slot),
                   kDispatchReadyBytes + kDispatchExpertCountBytes) +
               assignment_idx;
    }

    CUTLASS_HOST_DEVICE
    uint32_t* get_dispatch_receive_expert_assignment_ptr(
            const uint32_t remote_source_slot,
            const uint32_t assignment_idx = 0) const {
        DG_UNIFIED_ASSERT(
            assignment_idx <= kMaxExpertAssignments);
        return math::advance_ptr<uint32_t>(
                   get_dispatch_receive_packet_ptr(remote_source_slot),
                   kDispatchReadyBytes + kDispatchExpertCountBytes) +
               assignment_idx;
    }

    CUTLASS_HOST_DEVICE
    void* get_remote_input_activation_ptr(
            const uint32_t remote_source_slot,
            const uint32_t token_idx = 0) const {
        return get_remote_input_record_ptr(
            kRemoteInputActivationOffset,
            kInputActivationBytes,
            remote_source_slot, token_idx);
    }

    CUTLASS_HOST_DEVICE
    void* get_remote_input_scale_ptr(
            const uint32_t remote_source_slot,
            const uint32_t token_idx = 0) const {
        return get_remote_input_record_ptr(
            kRemoteInputScaleOffset,
            kInputScaleBytes,
            remote_source_slot, token_idx);
    }

    CUTLASS_HOST_DEVICE
    void* get_remote_topk_weight_ptr(
            const uint32_t remote_source_slot,
            const uint32_t token_idx = 0) const {
        return get_remote_input_record_ptr(
            kRemoteTopKWeightOffset,
            kTopKWeightBytes,
            remote_source_slot, token_idx);
    }

    CUTLASS_HOST_DEVICE
    uint32_t* get_combine_record_prefix_ptr(
            const uint32_t remote_source_slot,
            const uint32_t local_expert = 0) const {
        DG_UNIFIED_ASSERT(remote_source_slot < kNumRemotePeers);
        DG_UNIFIED_ASSERT(local_expert < kNumExpertsPerRank);
        const uint64_t index =
            static_cast<uint64_t>(remote_source_slot) *
                kNumExpertsPerRank +
            local_expert;
        return math::advance_ptr<uint32_t>(
                   base, kCombineRecordPrefixOffset) +
               index;
    }

    CUTLASS_HOST_DEVICE
    uint32_t* get_combine_return_ordinal_ptr(
            const uint32_t pool_token_idx) const {
        DG_UNIFIED_ASSERT(pool_token_idx < kMaxPoolTokens);
        return math::advance_ptr<uint32_t>(
                   base, kCombineReturnOrdinalOffset) +
               pool_token_idx;
    }

    CUTLASS_HOST_DEVICE
    void* get_combine_send_record_ptr(
            const uint32_t remote_source_slot,
            const uint32_t record_ordinal) const {
        return get_combine_record_ptr(
            /*send=*/ true, remote_source_slot, record_ordinal);
    }

    CUTLASS_HOST_DEVICE
    void* get_combine_receive_record_ptr(
            const uint32_t remote_owner_slot,
            const uint32_t record_ordinal) const {
        return get_combine_record_ptr(
            /*send=*/ false, remote_owner_slot, record_ordinal);
    }

private:
    CUTLASS_HOST_DEVICE
    void* get_dispatch_packet_ptr(
            const bool send, const uint32_t peer_slot) const {
        DG_UNIFIED_ASSERT(peer_slot < kNumRemotePeers);
        const uint32_t packet_idx =
            (send ? 0u : kNumRemotePeers) + peer_slot;
        return math::advance_ptr(
            base,
            kDispatchPacketOffset +
                static_cast<uint64_t>(packet_idx) *
                    kDispatchPacketBytes);
    }

    CUTLASS_HOST_DEVICE
    void* get_remote_input_record_ptr(
            const uint64_t storage_offset,
            const uint64_t record_bytes,
            const uint32_t remote_source_slot,
            const uint32_t token_idx) const {
        DG_UNIFIED_ASSERT(remote_source_slot < kNumRemotePeers);
        DG_UNIFIED_ASSERT(token_idx < kMaxActiveTokens);
        const uint64_t record_idx =
            static_cast<uint64_t>(remote_source_slot) *
                kMaxActiveTokens +
            token_idx;
        return math::advance_ptr(
            base, storage_offset + record_idx * record_bytes);
    }

    CUTLASS_HOST_DEVICE
    void* get_combine_packet_ptr(
            const bool send, const uint32_t peer_slot) const {
        DG_UNIFIED_ASSERT(peer_slot < kNumRemotePeers);
        const uint32_t packet_idx =
            (send ? 0u : kNumRemotePeers) + peer_slot;
        return math::advance_ptr(
            base,
            kCombinePacketOffset +
                static_cast<uint64_t>(packet_idx) *
                    kCombinePacketBytes);
    }

    CUTLASS_HOST_DEVICE
    void* get_combine_record_ptr(
            const bool send,
            const uint32_t peer_slot,
            const uint32_t record_ordinal) const {
        DG_UNIFIED_ASSERT(
            record_ordinal < kMaxExpertAssignments);
        return math::advance_ptr(
            get_combine_packet_ptr(send, peer_slot),
            static_cast<uint64_t>(record_ordinal) *
                kCombineRecordBytes);
    }
};

// Preserve the prototype's original specialization as the default while the
// fused kernel selects MegaMoeGinWorkspaceT with its own compile-time
// geometry. Compile-time checks below pin the established packet ABI and
// offsets without exposing the old model dimensions as transport constraints.
using MegaMoeGinWorkspace =
    MegaMoeGinWorkspaceT<56, 16, 3584, 48>;

static_assert(MegaMoeGinWorkspace::kDispatchPacketBytes == 3328);
static_assert(MegaMoeGinWorkspace::kCombineRecordBytes == 7168);
static_assert(MegaMoeGinWorkspace::kCombinePacketBytes == 5505024);
static_assert(MegaMoeGinWorkspace::kMaxPoolTokens == 21120);
static_assert(MegaMoeGinWorkspace::kDispatchPacketOffset == 43136);
static_assert(MegaMoeGinWorkspace::kCombinePacketOffset == 792576);
static_assert(
    MegaMoeGinWorkspace::kCombineReturnOrdinalOffset == 44832768);
static_assert(
    MegaMoeGinWorkspace::kCombineExpertCompletionOffset == 44917248);
static_assert(
    MegaMoeGinWorkspace::kCombineExpertCompletionBytes == 256);
static_assert(MegaMoeGinWorkspace::kNumBytes == 44917504);

} // namespace deep_gemm::layout
