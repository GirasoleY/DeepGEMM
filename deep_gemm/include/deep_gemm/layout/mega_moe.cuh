#pragma once

#include <cute/numeric/math.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::layout {

static constexpr int kNumCandidateBlockMs = 7;
static constexpr int kCandidateBlockM[kNumCandidateBlockMs] = {8, 16, 32, 64, 96, 128, 192};
static constexpr int kMaxCandidateBlockM = 192;
static constexpr int kMinCandidateBlockM = 8;
static constexpr int kLCMCandidateBlockM = 384;

// The first GIN prototype reuses the existing four MegaMoE dispatch warps and
// supports the agreed completion sweep up to B=8.  The outbox remains bounded
// while reserving enough row capacity for any runtime-selected BLOCK_M.
static constexpr uint32_t kMegaMoeGinNumDispatchWarps = 4;
static constexpr uint32_t kMegaMoeGinMaxCompletionBatch = 8;
static constexpr uint32_t kMegaMoeGinMaxOutboxBlockM = kMaxCandidateBlockM;
static constexpr uint32_t kMegaMoeGinNumDataContexts = 8;
// Stage-1 bulk combine targets the decode regime only.  The public MegaMoE
// allocation is still aligned to 384 tokens, so keep the packet capacity
// independent from num_max_tokens_per_rank and select this path only when the
// actual launch contains at most 48 tokens.
static constexpr uint32_t kMegaMoeGinBulkCombineMaxTokens = 48;
static constexpr uint32_t kMegaMoeGinBulkCombineHeaderBytes = 16;
static constexpr uint32_t kMegaMoeGinBulkCombineRecordHeaderBytes = 16;
// Keep the semantic packet and record headers unchanged, but isolate the
// fixed-stride record array on 128-byte boundaries.  The packet header-only
// PUT therefore remains exactly 16 bytes, while every record begins on its
// own aligned address family.
static constexpr uint32_t kMegaMoeGinBulkCombineRecordAlignment = 128;
static constexpr uint32_t kMegaMoeGinBulkCombineRecordAreaOffset = 128;
static_assert(kMegaMoeGinBulkCombineRecordAreaOffset >=
              kMegaMoeGinBulkCombineHeaderBytes);
static_assert(kMegaMoeGinBulkCombineRecordAreaOffset %
                  kMegaMoeGinBulkCombineRecordAlignment == 0);
// StrongVA combine visibility uses persistent NIC-written counters. Keep each
// owner lane on its own 128-byte address family and, unlike the historical
// packet/scratch aliases, reserve these bytes explicitly at the end of the
// registered GIN extension. The first eight bytes of each stride are the cell.
static constexpr uint32_t kMegaMoeGinCombineTerminalSignalStride = 128;
static_assert(kMegaMoeGinCombineTerminalSignalStride >= sizeof(uint64_t));
static_assert(kMegaMoeGinCombineTerminalSignalStride % 128u == 0);

// Stage-2 direct dispatch reinterprets the existing 384-row paired-ingress
// mirrors as eight source-private lanes.  Its compact control packets alias
// the otherwise-unused fine-grained GET scale scratch; no registered bytes are
// appended.  Pad each packet stride to a 128-byte line so one source's route
// tail cannot share a line with another source's VA terminal.  Keep these
// constants exact so host and device code can prove that the alias fits before
// enabling the specialization.
static constexpr uint32_t kMegaMoeGinDirectDispatchMaxTokens = 48;
static constexpr uint32_t kMegaMoeGinDirectDispatchNumPeers = 8;
static constexpr uint32_t kMegaMoeGinDirectDispatchExpertsPerRank = 56;
static constexpr uint32_t kMegaMoeGinDirectDispatchMaxRoutes =
    kMegaMoeGinDirectDispatchMaxTokens * 16;
static constexpr uint32_t kMegaMoeGinDirectDispatchReadyBytes = 16;
static constexpr uint32_t kMegaMoeGinDirectDispatchCountBytes =
    kMegaMoeGinDirectDispatchExpertsPerRank * sizeof(uint64_t);
static constexpr uint32_t kMegaMoeGinDirectDispatchRouteBytes =
    kMegaMoeGinDirectDispatchMaxRoutes * sizeof(uint32_t);
static constexpr uint32_t kMegaMoeGinDirectDispatchPacketAlignment = 128;
static constexpr uint32_t kMegaMoeGinDirectDispatchPacketDataBytes =
    kMegaMoeGinDirectDispatchReadyBytes +
    kMegaMoeGinDirectDispatchCountBytes +
    kMegaMoeGinDirectDispatchRouteBytes;
static constexpr uint32_t kMegaMoeGinDirectDispatchPacketBytes =
    ((kMegaMoeGinDirectDispatchPacketDataBytes +
      kMegaMoeGinDirectDispatchPacketAlignment - 1) /
     kMegaMoeGinDirectDispatchPacketAlignment) *
    kMegaMoeGinDirectDispatchPacketAlignment;
static_assert(kMegaMoeGinDirectDispatchPacketBytes %
                  kMegaMoeGinDirectDispatchPacketAlignment == 0);
static_assert(kMegaMoeGinDirectDispatchPacketDataBytes <=
              kMegaMoeGinDirectDispatchPacketBytes);
static constexpr uint32_t kMegaMoeGinDirectDispatchStorageBytes =
    2 * kMegaMoeGinDirectDispatchNumPeers *
    kMegaMoeGinDirectDispatchPacketBytes;

// Dynamic expert-ready combine aliases only the otherwise-unused tail of the
// direct-dispatch scale scratch. Expert counters include ALL actual M blocks;
// their storage bound is independent of imbalance or the selected BLOCK_M.
static constexpr uint32_t kMegaMoeGinCombineOverlapNumExperts =
    kMegaMoeGinDirectDispatchExpertsPerRank;
static constexpr uint32_t kMegaMoeGinCombineOverlapNumExpertGroups =
    (kMegaMoeGinCombineOverlapNumExperts + 31u) / 32u;
CUTLASS_HOST_DEVICE constexpr uint64_t
get_mega_moe_gin_combine_overlap_scratch_bytes() {
    // 56 ready counts, 56 sent entries, and one saved prefix for each of the
    // eight remote sources ×56 experts, followed by two nonempty-expert masks
    // per source. Counts stay in their existing cells; the appended masks do
    // not change any existing prefix, registered allocation or legacy offset.
    return (2ull + kMegaMoeGinDirectDispatchNumPeers) * sizeof(uint32_t) *
               kMegaMoeGinCombineOverlapNumExperts +
           uint64_t(kMegaMoeGinDirectDispatchNumPeers) * sizeof(uint32_t) *
               kMegaMoeGinCombineOverlapNumExpertGroups;
}
static constexpr uint32_t kMegaMoeGinDirectReduceOrdinalBytes =
    kMegaMoeGinDirectDispatchMaxRoutes * sizeof(uint32_t);

// Pool capacity for shared expert token pool: worst-case total tokens + per-expert BLOCK_M alignment padding, among all possible BLOCK_M
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_max_pool_tokens(T num_ranks, T num_max_tokens_per_rank, T num_topk,
                                                        T num_experts_per_rank) {
    const auto num_max_recv_tokens = num_ranks * num_max_tokens_per_rank;
    const auto num_max_experts_per_token = math::constexpr_min(num_topk, num_experts_per_rank);
    return math::constexpr_align(
        num_max_recv_tokens * num_max_experts_per_token + num_experts_per_rank * (static_cast<T>(kMaxCandidateBlockM) - 1),
        static_cast<T>(kLCMCandidateBlockM));
}

// SF pool capacity: all experts share a contiguous SF region, sized by pool blocks × SF_BLOCK_M
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_sf_ring_tokens(T num_ring_tokens, T block_m) {
    return (num_ring_tokens / block_m) * math::constexpr_align(block_m, static_cast<T>(128));
}

// Shared L2 input SF capacity: worst-case aligned SF pages over all candidate BLOCK_M.
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_max_shared_sf_tokens(const T& num_max_tokens_per_rank) {
    return math::constexpr_ceil_div<T>(num_max_tokens_per_rank, kMinCandidateBlockM) * 128;
}

// Per-token source metadata for combine write-back
struct TokenSrcMetadata {
    uint32_t rank_idx;
    uint32_t token_idx;
    uint32_t topk_idx;
};

struct Workspace {
    void* base;
    uint32_t num_ranks, num_experts;
    uint32_t num_experts_per_rank;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_max_recv_tokens_per_expert;

    // Ring-buffer capacity used by reusable token/data buffers
    uint32_t num_ring_tokens;
    uint32_t num_ring_blocks;
    uint32_t num_shared_l2_pool_blocks;

    // Full-pool span used by non-ring token metadata
    uint32_t num_max_pool_tokens;

    // Keep grid/NVLink/schedule counters separated from expert counters.
    // NVIDIA L2 cache lines are 128B, and these counters are hot atomics.
    static constexpr uint64_t kNumBarrierSignalBytes = 128;

    Workspace() = default;

    CUTLASS_HOST_DEVICE
    Workspace(void* base,
              const uint32_t& num_ranks,
              const uint32_t& num_experts,
              const uint32_t& num_max_tokens_per_rank,
              const uint32_t& num_topk,
              const uint32_t& num_ring_tokens):
        base(base),
        num_ranks(num_ranks), num_experts(num_experts),
        num_max_tokens_per_rank(num_max_tokens_per_rank),
        num_ring_tokens(num_ring_tokens) {
        num_experts_per_rank = num_experts / num_ranks;
        num_max_recv_tokens_per_expert = num_ranks * num_max_tokens_per_rank;
        num_max_pool_tokens = get_num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
        num_ring_blocks = num_ring_tokens / kMinCandidateBlockM;
        num_shared_l2_pool_blocks = math::ceil_div<uint32_t>(num_max_tokens_per_rank, kMinCandidateBlockM);
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        uint64_t num_bytes = 0;

        // Barrier and in-kernel task scheduling counters
        num_bytes += kNumBarrierSignalBytes;

        // Expert send/recv count
        num_bytes += num_experts * sizeof(uint64_t) * 2;

        // Expert recv count sum
        num_bytes += num_experts_per_rank * sizeof(uint64_t);

        // L1 full token count (ring)
        num_bytes += num_ring_blocks * sizeof(uint32_t);

        // L1 empty block count (ring)
        num_bytes += num_ring_blocks * sizeof(uint32_t);

        // L2 full block count (ring)
        num_bytes += num_ring_blocks * sizeof(uint32_t);

        // L2 empty block count (ring)
        num_bytes += num_ring_blocks * sizeof(uint32_t);

        // Shared L2 full block count
        num_bytes += num_shared_l2_pool_blocks * sizeof(uint32_t);

        // Dispatch pulling source token-topk
        num_bytes += num_experts_per_rank * num_ranks * num_max_recv_tokens_per_expert * sizeof(int);

        // Combine push source indices (full)
        num_bytes += num_max_pool_tokens * sizeof(TokenSrcMetadata);

        // Align to TMA descriptor requirements
        num_bytes = math::align<uint64_t>(num_bytes, 16);
        return num_bytes;
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    // Grid sync counters: `kNumBarrierSignalBytes` layout
    // [ 0..15]: 4 x `uint32_t` grid sync counters
    // [16..20]: `uint32_t` NVLink barrier counter
    // [20..27]: 2 x `int` NVLink barrier signals (phase 0 and 1)
    // [28..31]: `uint32_t` L1 schedule task counter
    // [32..35]: `uint32_t` L2 schedule task counter
    // [36..39]: `uint32_t` shared L1 schedule task counter
    // [40..43]: `uint32_t` shared L2 schedule task counter
    // [44..47]: reserved (former 32-bit GIN activity epoch)
    // [48..51]: GIN paired-rank activity decision
    // [52..55]: GIN world activity decision
    // [56..59]: GIN world small-decode ineligibility decision
    // [60..63]: reserved
    // [64..71]: `uint64_t` StrongVA combine-terminal invocation epoch
    // [72..87]: 2 x `uint64_t` GIN paired-decision mailboxes
    // [88..95]: reserved (keeps the NIC mailboxes in their own 32-byte sector)
    // [96..103]: `uint64_t` GIN paired-decision launch epoch
    // [104..111]: `uint64_t` GIN direct-dispatch invocation epoch
    // [112..119]: `uint64_t` GIN control-first payload invocation epoch
    // [120..127]: padding to isolate hot expert counters from barrier/schedule counters
    static constexpr uint32_t kNumMaxGridSyncCounters = 4;

    template <uint32_t kIndex = 0>
    CUTLASS_DEVICE
    uint32_t* get_grid_sync_count_ptr() const {
        DG_STATIC_ASSERT(kIndex < kNumMaxGridSyncCounters, "Grid sync index out of bounds");
        return static_cast<uint32_t*>(base) + kIndex;
    }

    CUTLASS_DEVICE
    uint32_t* get_nvl_barrier_counter_ptr() const {
        return static_cast<uint32_t*>(base) + kNumMaxGridSyncCounters;
    }

    CUTLASS_DEVICE
    int* get_nvl_barrier_signal_ptr(const uint32_t& phase) const {
        // NOTES: the signal is signed, as we may minus
        return math::advance_ptr<int>(base, (kNumMaxGridSyncCounters + 1) * sizeof(uint32_t) + phase * sizeof(int));
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_task_count_ptr() const {
        return math::advance_ptr<uint32_t>(base, 28u);
    }

    CUTLASS_DEVICE
    uint32_t* get_l2_task_count_ptr() const {
        return math::advance_ptr<uint32_t>(base, 32u);
    }

    CUTLASS_DEVICE
    uint32_t* get_shared_l1_task_count_ptr() const {
        return math::advance_ptr<uint32_t>(base, 36u);
    }

    CUTLASS_DEVICE
    uint32_t* get_shared_l2_task_count_ptr() const {
        return math::advance_ptr<uint32_t>(base, 40u);
    }

    CUTLASS_DEVICE
    uint64_t* get_gin_active_launch_epoch_ptr() const {
        return math::advance_ptr<uint64_t>(base, 96u);
    }

    CUTLASS_DEVICE
    uint32_t* get_gin_pair_active_ptr() const {
        return math::advance_ptr<uint32_t>(base, 48u);
    }

    CUTLASS_DEVICE
    uint32_t* get_gin_world_active_ptr() const {
        return math::advance_ptr<uint32_t>(base, 52u);
    }

    CUTLASS_DEVICE
    uint32_t* get_gin_world_bulk_ineligible_ptr() const {
        return math::advance_ptr<uint32_t>(base, 56u);
    }

    CUTLASS_DEVICE
    uint64_t* get_gin_pair_mailbox_ptr(const uint32_t& parity) const {
        DG_DEVICE_ASSERT(parity < 2);
        return math::advance_ptr<uint64_t>(base, 72u + parity * sizeof(uint64_t));
    }

    CUTLASS_DEVICE
    uint64_t* get_gin_direct_dispatch_epoch_ptr() const {
        return math::advance_ptr<uint64_t>(base, 104u);
    }

    CUTLASS_DEVICE
    uint64_t* get_gin_dispatch_payload_epoch_ptr() const {
        // Separate from the control epoch: baseline direct invocations still
        // increment the control terminal, but do not publish this terminal.
        return math::advance_ptr<uint64_t>(base, 112u);
    }

    CUTLASS_DEVICE
    uint64_t* get_gin_combine_terminal_epoch_ptr() const {
        return math::advance_ptr<uint64_t>(base, 64u);
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_send_count_ptr(const uint32_t& expert_idx = 0) const {
        return math::advance_ptr<uint64_t>(base, kNumBarrierSignalBytes) + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_ptr(
        const uint32_t& rank_idx = 0, const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts) + rank_idx * num_experts_per_rank + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_sum_ptr(const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts * 2) + expert_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_full_count_ptr(const uint32_t& ring_block_idx = 0) const {
        const auto base = get_expert_recv_count_sum_ptr(num_experts_per_rank);
        return reinterpret_cast<uint32_t*>(base) + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_empty_count_ptr(const uint32_t& ring_block_idx = 0) const {
        const auto base = get_l1_full_count_ptr(num_ring_blocks);
        return reinterpret_cast<uint32_t*>(base) + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l2_full_count_ptr(const uint32_t& ring_block_idx = 0) const {
        const auto base = get_l1_empty_count_ptr(num_ring_blocks);
        return reinterpret_cast<uint32_t*>(base) + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l2_empty_count_ptr(const uint32_t& ring_block_idx = 0) const {
        const auto base = get_l2_full_count_ptr(num_ring_blocks);
        return reinterpret_cast<uint32_t*>(base) + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_shared_l2_full_count_ptr(const uint32_t& block_idx = 0) const {
        const auto base = get_l2_empty_count_ptr(num_ring_blocks);
        return reinterpret_cast<uint32_t*>(base) + block_idx;
    }

    // For dispatch pulling
    CUTLASS_DEVICE
    uint32_t* get_src_token_topk_idx_ptr(
        const uint32_t& expert_idx = 0, const uint32_t& rank_idx = 0, const uint32_t& token_idx = 0) const {
        const auto base = get_shared_l2_full_count_ptr(num_shared_l2_pool_blocks);
        return reinterpret_cast<uint32_t*>(base) +
            expert_idx * (num_ranks * num_max_recv_tokens_per_expert) +
            rank_idx * num_max_recv_tokens_per_expert + token_idx;
    }

    // For combine usages (full)
    CUTLASS_DEVICE
    TokenSrcMetadata* get_token_src_metadata_ptr(const uint32_t& pool_token_idx = 0) const {
        const auto base = reinterpret_cast<TokenSrcMetadata*>(get_src_token_topk_idx_ptr(num_experts_per_rank));
        return base + pool_token_idx;
    }
};

struct Data {
    uint32_t num_bytes;
    bool require_tma_alignment;
    void* base;

    Data() = default;

    CUTLASS_HOST_DEVICE
    constexpr explicit Data(
        const uint32_t& num_bytes,
        const bool& require_tma_alignment = true,
        void* base = nullptr) :
        num_bytes(num_bytes), require_tma_alignment(require_tma_alignment), base(base) {
        DG_UNIFIED_ASSERT(num_bytes % 16 == 0 or not require_tma_alignment);
    }

    template <typename dtype_t = uint32_t>
    CUTLASS_HOST_DEVICE constexpr dtype_t get_num_bytes() const {
        return static_cast<dtype_t>(num_bytes);
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    CUTLASS_HOST_DEVICE void set_base_ptr(void* ptr) {
        base = ptr;
    }
};

struct Buffer {
    Data data_layout;
    uint32_t num_ranks;
    uint32_t num_max_tokens_per_rank;

    void* base;

    Buffer() = default;

    CUTLASS_HOST_DEVICE
    Buffer(const Data& data_layout,
           const uint32_t& num_ranks,
           const uint32_t& num_max_tokens_per_rank,
           void* base = nullptr) :
        data_layout(data_layout),
        num_ranks(num_ranks), num_max_tokens_per_rank(num_max_tokens_per_rank),
        base(base) {}

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes_per_rank() const {
        return num_max_tokens_per_rank * data_layout.get_num_bytes<uint64_t>();
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        return get_num_bytes_per_rank() * num_ranks;
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    CUTLASS_HOST_DEVICE
    Buffer get_rank_buffer(const uint32_t& rank_idx) const {
        return {
            data_layout,
            1, num_max_tokens_per_rank,
            math::advance_ptr(base, get_num_bytes_per_rank() * rank_idx)
        };
    }

    CUTLASS_HOST_DEVICE
    Data get_data_buffer(const uint32_t& token_idx, const bool& global = false) const {
        DG_DEVICE_ASSERT(num_ranks == 1 or global);
        return Data(
            data_layout.num_bytes,
            data_layout.require_tma_alignment,
            math::advance_ptr(base, data_layout.get_num_bytes<uint64_t>() * token_idx)
        );
    }
};

// Registered transport-only storage appended after the existing MegaMoE
// layout.  Appending keeps every legacy offset unchanged.  The extension is
// absent unless the opt-in GIN path is requested by the host.
struct MegaMoeGinWorkspace {
    void* base;
    uint32_t num_ranks;
    uint32_t num_sms;
    uint32_t completion_batch;
    uint32_t outbox_depth;
    uint32_t bulk_record_bytes;

    Buffer published_input_token_buffer;
    Buffer published_input_sf_buffer;
    Buffer published_input_topk_weights_buffer;
    Buffer count_staging_buffer;
    Buffer route_staging_buffer;
    Buffer scale_scratch_buffer;
    Buffer combine_outbox_alignment_padding_buffer;
    Buffer combine_outbox_buffer;
    Buffer bulk_combine_packet_tail_buffer;
    Buffer bulk_combine_return_index_buffer;
    Buffer combine_terminal_signal_buffer;
    bool bulk_combine;
    uint32_t bulk_packet_bytes;

    MegaMoeGinWorkspace() = default;

    CUTLASS_HOST_DEVICE
    MegaMoeGinWorkspace(void* base,
                        const uint32_t& hidden,
                        const uint32_t& num_ranks,
                        const uint32_t& num_experts,
                        const uint32_t& num_max_tokens_per_rank,
                        const uint32_t& num_topk,
                        const uint32_t& num_sms,
                        const uint32_t& completion_batch,
                        const uint32_t& outbox_depth,
                        const bool& bulk_combine):
        base(base), num_ranks(num_ranks), num_sms(num_sms),
        completion_batch(completion_batch), outbox_depth(outbox_depth),
        bulk_record_bytes(0), bulk_combine(bulk_combine),
        bulk_packet_bytes(0) {
        DG_UNIFIED_ASSERT(completion_batch > 0 and
                          completion_batch <= kMegaMoeGinMaxCompletionBatch);
        DG_UNIFIED_ASSERT(outbox_depth > 0);

        const auto input_token_layout = Data(hidden);
        const auto input_sf_layout = Data(hidden / 32);
        const auto input_topk_weights_layout =
            Data(num_topk * sizeof(float), false);
        const auto count_layout =
            Data((num_experts / num_ranks) * sizeof(uint64_t));
        const auto route_layout = Data(sizeof(uint32_t), false);
        const auto scale_layout = Data(hidden / 32);
        const auto combine_row_layout = Data(hidden * sizeof(uint16_t));
        const auto bulk_return_index_layout = Data(sizeof(uint32_t), false);

        // The r75 fallback interprets these rows as one inbound mirror for the
        // same-GPU-index peer in the other LSA.  The opt-in direct-dispatch
        // specialization instead partitions the first 384 rows into eight
        // source-private 48-row lanes.  The allocation and all later offsets
        // remain identical in both interpretations.
        published_input_token_buffer = Buffer(
            input_token_layout, 1, num_max_tokens_per_rank,
            get_control_end_ptr());
        published_input_sf_buffer = Buffer(
            input_sf_layout, 1, num_max_tokens_per_rank,
            published_input_token_buffer.get_end_ptr());
        published_input_topk_weights_buffer = Buffer(
            input_topk_weights_layout, 1, num_max_tokens_per_rank,
            published_input_sf_buffer.get_end_ptr());
        count_staging_buffer = Buffer(
            count_layout, num_ranks, 1,
            published_input_topk_weights_buffer.get_end_ptr());
        route_staging_buffer = Buffer(
            route_layout, num_experts, num_max_tokens_per_rank,
            count_staging_buffer.get_end_ptr());
        scale_scratch_buffer = Buffer(
            scale_layout,
            num_sms * kMegaMoeGinNumDispatchWarps * completion_batch, 1,
            route_staging_buffer.get_end_ptr());

        // The public GIN context requires the symmetric allocation base to be
        // NCCL_WIN_REQUIRED_ALIGNMENT-aligned (4096 bytes in the supported
        // runtime), so the nullptr-based size layout and the concrete layout
        // have the same residue modulo 128. Some public capacities place the
        // end of scale scratch at +64 modulo 128; account that explicitly
        // instead of relying on a target-shape assertion. Legacy/non-bulk
        // layouts retain their exact offsets.
        const auto unaligned_outbox_address = reinterpret_cast<uintptr_t>(
            scale_scratch_buffer.get_end_ptr());
        const uint32_t outbox_alignment_padding_bytes = bulk_combine ?
            static_cast<uint32_t>(
                (kMegaMoeGinBulkCombineRecordAlignment -
                 unaligned_outbox_address %
                     kMegaMoeGinBulkCombineRecordAlignment) %
                kMegaMoeGinBulkCombineRecordAlignment) : 0u;
        DG_UNIFIED_ASSERT(
            outbox_alignment_padding_bytes <
                kMegaMoeGinBulkCombineRecordAlignment and
            outbox_alignment_padding_bytes % 16u == 0u);
        combine_outbox_alignment_padding_buffer = Buffer(
            Data(outbox_alignment_padding_bytes), 1, 1,
            scale_scratch_buffer.get_end_ptr());
        combine_outbox_buffer = Buffer(
            combine_row_layout, outbox_depth,
            static_cast<uint32_t>(kMegaMoeGinMaxOutboxBlockM),
            combine_outbox_alignment_padding_buffer.get_end_ptr());
        if (bulk_combine) {
            DG_UNIFIED_ASSERT(
                reinterpret_cast<uintptr_t>(combine_outbox_buffer.base) %
                    kMegaMoeGinBulkCombineRecordAlignment == 0);
        }

        // A compact owner->source packet has one semantic 16-byte packet
        // header, padding through byte 128, and at most (48 tokens * top-k)
        // fixed-stride records. Each record's first 16 bytes carry the final
        // token/top-k index, followed by one BF16 output row and tail padding
        // through the next 128-byte boundary.
        // Only actual peers in the other LSA need packets. Keep the complete
        // fallback outbox even when EP8's four-peer packet pool is smaller.
        const auto bulk_base = combine_outbox_buffer.get_end_ptr();
        if (bulk_combine) {
            const uint32_t bulk_capacity =
                kMegaMoeGinBulkCombineMaxTokens * num_topk;
            const uint32_t num_remote_peers = num_ranks / 2;
            const uint64_t bulk_record_data_bytes =
                kMegaMoeGinBulkCombineRecordHeaderBytes +
                static_cast<uint64_t>(hidden) * sizeof(uint16_t);
            DG_UNIFIED_ASSERT(
                bulk_record_data_bytes <=
                static_cast<uint64_t>(UINT32_MAX) -
                    (kMegaMoeGinBulkCombineRecordAlignment - 1u));
            bulk_record_bytes = math::align<uint32_t>(
                static_cast<uint32_t>(bulk_record_data_bytes),
                kMegaMoeGinBulkCombineRecordAlignment);
            DG_UNIFIED_ASSERT(
                bulk_record_bytes >= bulk_record_data_bytes and
                bulk_record_bytes %
                    kMegaMoeGinBulkCombineRecordAlignment == 0);
            const uint64_t bulk_packet_bytes_u64 =
                kMegaMoeGinBulkCombineRecordAreaOffset +
                static_cast<uint64_t>(bulk_capacity) * bulk_record_bytes;
            DG_UNIFIED_ASSERT(bulk_packet_bytes_u64 <=
                              static_cast<uint64_t>(UINT32_MAX));
            bulk_packet_bytes =
                static_cast<uint32_t>(bulk_packet_bytes_u64);
            DG_UNIFIED_ASSERT(
                bulk_packet_bytes %
                    kMegaMoeGinBulkCombineRecordAlignment == 0);
            const uint64_t bulk_packet_storage_bytes =
                2ull * num_remote_peers * bulk_packet_bytes;
            const uint64_t outbox_storage_bytes =
                combine_outbox_buffer.get_num_bytes();
            const uint64_t reserved_storage_bytes =
                bulk_packet_storage_bytes > outbox_storage_bytes ?
                bulk_packet_storage_bytes : outbox_storage_bytes;
            DG_UNIFIED_ASSERT(
                reserved_storage_bytes - outbox_storage_bytes <=
                static_cast<uint64_t>(UINT32_MAX));
            const uint32_t packet_tail_bytes = static_cast<uint32_t>(
                reserved_storage_bytes - outbox_storage_bytes);
            bulk_combine_packet_tail_buffer = Buffer(
                Data(packet_tail_bytes), 1, 1, bulk_base);
            bulk_combine_return_index_buffer = Buffer(
                bulk_return_index_layout, 1,
                get_num_max_pool_tokens(
                    num_ranks, num_max_tokens_per_rank, num_topk,
                    num_experts / num_ranks),
                bulk_combine_packet_tail_buffer.get_end_ptr());
            DG_UNIFIED_ASSERT(
                reinterpret_cast<uintptr_t>(
                    bulk_combine_return_index_buffer.get_end_ptr()) %
                    kMegaMoeGinCombineTerminalSignalStride == 0);
            combine_terminal_signal_buffer = Buffer(
                Data(kMegaMoeGinCombineTerminalSignalStride),
                num_remote_peers, 1,
                bulk_combine_return_index_buffer.get_end_ptr());
        } else {
            const auto empty_layout = Data(0, false);
            bulk_combine_packet_tail_buffer = Buffer(
                empty_layout, 0, 0, bulk_base);
            bulk_combine_return_index_buffer = Buffer(
                empty_layout, 0, 0, bulk_base);
            combine_terminal_signal_buffer = Buffer(
                empty_layout, 0, 0, bulk_base);
        }
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_control_num_bytes() const {
        // Per-(world peer, data-context) owner-GET issue locks, followed by
        // cumulative outbox full and empty generations.  Keep the appended
        // data buffers 16-byte aligned for registered-window/TMA-friendly
        // addressing.
        const uint64_t bytes =
            (static_cast<uint64_t>(num_ranks) * kMegaMoeGinNumDataContexts +
             2ull * outbox_depth) *
            sizeof(uint32_t);
        return math::align<uint64_t>(bytes, 16);
    }

    CUTLASS_HOST_DEVICE
    void* get_control_end_ptr() const {
        return math::advance_ptr(base, get_control_num_bytes());
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        return static_cast<uint8_t*>(
                   combine_terminal_signal_buffer.get_end_ptr()) -
               static_cast<uint8_t*>(base);
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return combine_terminal_signal_buffer.get_end_ptr();
    }

#if defined(__CUDA_ARCH__) or defined(__CLION_IDE__)
    CUTLASS_DEVICE
    uint32_t* get_peer_issue_lock_ptr(const uint32_t& peer,
                                      const uint32_t& context_stripe) const {
        DG_DEVICE_ASSERT(context_stripe < kMegaMoeGinNumDataContexts);
        return static_cast<uint32_t*>(base) +
               peer * kMegaMoeGinNumDataContexts + context_stripe;
    }

    CUTLASS_DEVICE
    uint32_t* get_outbox_full_count_ptr(const uint32_t& slot) const {
        return static_cast<uint32_t*>(base) +
               num_ranks * kMegaMoeGinNumDataContexts + slot;
    }

    CUTLASS_DEVICE
    uint32_t* get_outbox_empty_count_ptr(const uint32_t& slot) const {
        return static_cast<uint32_t*>(base) +
               num_ranks * kMegaMoeGinNumDataContexts + outbox_depth + slot;
    }

    CUTLASS_DEVICE
    uint32_t* get_route_staging_ptr(const uint32_t& expert_idx,
                                    const uint32_t& slot_idx) const {
        return route_staging_buffer.get_rank_buffer(expert_idx)
            .get_data_buffer(slot_idx).template get_base_ptr<uint32_t>();
    }

    CUTLASS_DEVICE
    void* get_scale_scratch_ptr(const uint32_t& sm_idx,
                                const uint32_t& dispatch_warp_idx,
                                const uint32_t& batch_idx = 0) const {
        const uint32_t record =
            (sm_idx * kMegaMoeGinNumDispatchWarps + dispatch_warp_idx) *
                completion_batch +
            batch_idx;
        return scale_scratch_buffer.get_rank_buffer(record)
            .get_data_buffer(0).get_base_ptr();
    }

    CUTLASS_HOST_DEVICE
    bool direct_dispatch_alias_fits() const {
        return scale_scratch_buffer.get_num_bytes() >=
               kMegaMoeGinDirectDispatchStorageBytes;
    }

    CUTLASS_HOST_DEVICE
    bool combine_overlap_alias_fits() const {
        return scale_scratch_buffer.get_num_bytes() >=
                   kMegaMoeGinDirectDispatchStorageBytes +
                       get_mega_moe_gin_combine_overlap_scratch_bytes();
    }

    CUTLASS_HOST_DEVICE
    uint32_t* get_combine_overlap_ready_ptr(
            const uint32_t local_expert) const {
        DG_UNIFIED_ASSERT(combine_overlap_alias_fits());
        DG_UNIFIED_ASSERT(local_expert < kMegaMoeGinCombineOverlapNumExperts);
        return static_cast<uint32_t*>(math::advance_ptr(
            scale_scratch_buffer.base,
            kMegaMoeGinDirectDispatchStorageBytes)) + local_expert;
    }

    CUTLASS_HOST_DEVICE
    uint32_t* get_combine_overlap_sent_ptr(
            const uint32_t local_expert) const {
        return get_combine_overlap_ready_ptr(local_expert) +
            kMegaMoeGinCombineOverlapNumExperts;
    }

    // The owner-slot reducer reuses four already-reset bookkeeping cells after
    // the W4 sender has finished.  These are rank-local handoff flags, not new
    // registered-window signals and not part of the wire protocol.
    CUTLASS_HOST_DEVICE
    uint32_t* get_combine_receiver_owner_ready_ptr(
            const uint32_t owner_in_lsa) const {
        DG_UNIFIED_ASSERT(
            owner_in_lsa < kMegaMoeGinDirectDispatchNumPeers);
        return get_combine_overlap_sent_ptr(owner_in_lsa);
    }

    CUTLASS_HOST_DEVICE
    uint32_t* get_combine_overlap_prefix_ptr(
            const uint32_t source_lane, const uint32_t local_expert) const {
        DG_UNIFIED_ASSERT(source_lane < kMegaMoeGinDirectDispatchNumPeers);
        return get_combine_overlap_ready_ptr(local_expert) +
            (2u + source_lane) * kMegaMoeGinCombineOverlapNumExperts;
    }

    CUTLASS_HOST_DEVICE
    uint32_t* get_combine_overlap_nonempty_mask_ptr(
            const uint32_t source_lane, const uint32_t expert_group) const {
        DG_UNIFIED_ASSERT(source_lane < kMegaMoeGinDirectDispatchNumPeers);
        DG_UNIFIED_ASSERT(expert_group < kMegaMoeGinCombineOverlapNumExpertGroups);
        return get_combine_overlap_ready_ptr(0) +
            (2u + kMegaMoeGinDirectDispatchNumPeers) *
                kMegaMoeGinCombineOverlapNumExperts +
            source_lane * kMegaMoeGinCombineOverlapNumExpertGroups + expert_group;
    }

    CUTLASS_HOST_DEVICE
    bool combine_direct_reduce_alias_fits() const {
        return scale_scratch_buffer.get_num_bytes() >=
                   kMegaMoeGinDirectDispatchStorageBytes +
                       get_mega_moe_gin_combine_overlap_scratch_bytes() +
                       kMegaMoeGinDirectReduceOrdinalBytes;
    }

    CUTLASS_HOST_DEVICE
    uint32_t* get_combine_direct_reduce_ordinal_ptr(
            const uint32_t token_topk_idx) const {
        DG_UNIFIED_ASSERT(combine_direct_reduce_alias_fits());
        DG_UNIFIED_ASSERT(token_topk_idx < kMegaMoeGinDirectDispatchMaxRoutes);
        // Source-local inverse of the actually packed assignment ordering.
        // Distinct from the owner's simultaneously live return-index buffer.
        return static_cast<uint32_t*>(math::advance_ptr(
            scale_scratch_buffer.base,
            kMegaMoeGinDirectDispatchStorageBytes +
                get_mega_moe_gin_combine_overlap_scratch_bytes())) + token_topk_idx;
    }

    CUTLASS_HOST_DEVICE
    bool direct_dispatch_mirrors_fit() const {
        return published_input_token_buffer.num_max_tokens_per_rank >=
                   kMegaMoeGinDirectDispatchNumPeers *
                       kMegaMoeGinDirectDispatchMaxTokens and
               published_input_sf_buffer.num_max_tokens_per_rank >=
                   kMegaMoeGinDirectDispatchNumPeers *
                       kMegaMoeGinDirectDispatchMaxTokens and
               published_input_topk_weights_buffer.num_max_tokens_per_rank >=
                   kMegaMoeGinDirectDispatchNumPeers *
                       kMegaMoeGinDirectDispatchMaxTokens;
    }

    CUTLASS_HOST_DEVICE
    void* get_direct_dispatch_packet_ptr(
            const bool send, const uint32_t& peer_in_lsa) const {
        DG_UNIFIED_ASSERT(direct_dispatch_alias_fits());
        DG_UNIFIED_ASSERT(
            peer_in_lsa < kMegaMoeGinDirectDispatchNumPeers);
        const uint32_t packet_idx =
            (send ? 0u : kMegaMoeGinDirectDispatchNumPeers) + peer_in_lsa;
        return math::advance_ptr(
            scale_scratch_buffer.base,
            static_cast<uint64_t>(packet_idx) *
                kMegaMoeGinDirectDispatchPacketBytes);
    }

    CUTLASS_HOST_DEVICE
    uint64_t* get_direct_dispatch_ready_ptr(
            const bool send, const uint32_t& peer_in_lsa) const {
        return static_cast<uint64_t*>(
            get_direct_dispatch_packet_ptr(send, peer_in_lsa));
    }

    CUTLASS_HOST_DEVICE
    uint64_t* get_direct_dispatch_payload_ready_ptr(
            const uint32_t& peer_in_lsa) const {
        static_assert(kMegaMoeGinDirectDispatchReadyBytes >=
                      2u * sizeof(uint64_t));
        // Use the reserved second uint64_t of the existing receive header.
        // No allocation or packet stride change; never reset between launches.
        return get_direct_dispatch_ready_ptr(/*send=*/ false, peer_in_lsa) + 1;
    }

    CUTLASS_HOST_DEVICE
    uint64_t* get_direct_dispatch_count_ptr(
            const bool send, const uint32_t& peer_in_lsa,
            const uint32_t& local_expert_idx = 0) const {
        DG_UNIFIED_ASSERT(
            local_expert_idx <=
            kMegaMoeGinDirectDispatchExpertsPerRank);
        return math::advance_ptr<uint64_t>(
            get_direct_dispatch_packet_ptr(send, peer_in_lsa),
            kMegaMoeGinDirectDispatchReadyBytes) + local_expert_idx;
    }

    CUTLASS_HOST_DEVICE
    uint32_t* get_direct_dispatch_route_ptr(
            const bool send, const uint32_t& peer_in_lsa,
            const uint32_t& route_idx = 0) const {
        DG_UNIFIED_ASSERT(
            route_idx <= kMegaMoeGinDirectDispatchMaxRoutes);
        return math::advance_ptr<uint32_t>(
            get_direct_dispatch_packet_ptr(send, peer_in_lsa),
            kMegaMoeGinDirectDispatchReadyBytes +
                kMegaMoeGinDirectDispatchCountBytes) + route_idx;
    }

    CUTLASS_HOST_DEVICE
    uint32_t get_direct_dispatch_control_bytes(
            const uint32_t& route_count) const {
        DG_UNIFIED_ASSERT(
            route_count <= kMegaMoeGinDirectDispatchMaxRoutes);
        return kMegaMoeGinDirectDispatchCountBytes +
               route_count * sizeof(uint32_t);
    }

    CUTLASS_HOST_DEVICE
    void* get_direct_input_token_ptr(
            const uint32_t& source_lane,
            const uint32_t& token_idx = 0) const {
        DG_UNIFIED_ASSERT(
            source_lane < kMegaMoeGinDirectDispatchNumPeers and
            token_idx < kMegaMoeGinDirectDispatchMaxTokens);
        DG_UNIFIED_ASSERT(direct_dispatch_mirrors_fit());
        return published_input_token_buffer
            .get_data_buffer(
                source_lane * kMegaMoeGinDirectDispatchMaxTokens + token_idx)
            .get_base_ptr();
    }

    CUTLASS_HOST_DEVICE
    void* get_direct_input_sf_ptr(
            const uint32_t& source_lane,
            const uint32_t& token_idx = 0) const {
        DG_UNIFIED_ASSERT(
            source_lane < kMegaMoeGinDirectDispatchNumPeers and
            token_idx < kMegaMoeGinDirectDispatchMaxTokens);
        DG_UNIFIED_ASSERT(direct_dispatch_mirrors_fit());
        return published_input_sf_buffer
            .get_data_buffer(
                source_lane * kMegaMoeGinDirectDispatchMaxTokens + token_idx)
            .get_base_ptr();
    }

    CUTLASS_HOST_DEVICE
    void* get_direct_input_topk_weights_ptr(
            const uint32_t& source_lane,
            const uint32_t& token_idx = 0) const {
        DG_UNIFIED_ASSERT(
            source_lane < kMegaMoeGinDirectDispatchNumPeers and
            token_idx < kMegaMoeGinDirectDispatchMaxTokens);
        DG_UNIFIED_ASSERT(direct_dispatch_mirrors_fit());
        return published_input_topk_weights_buffer
            .get_data_buffer(
                source_lane * kMegaMoeGinDirectDispatchMaxTokens + token_idx)
            .get_base_ptr();
    }

    CUTLASS_DEVICE
    void* get_combine_outbox_row_ptr(const uint32_t& slot,
                                     const uint32_t& row) const {
        return combine_outbox_buffer.get_rank_buffer(slot)
            .get_data_buffer(row).get_base_ptr();
    }

    CUTLASS_DEVICE
    uint32_t* get_bulk_combine_return_index_ptr(
        const uint32_t& pool_token_idx) const {
        DG_DEVICE_ASSERT(bulk_combine);
        return bulk_combine_return_index_buffer
            .get_data_buffer(pool_token_idx)
            .template get_base_ptr<uint32_t>();
    }

    CUTLASS_HOST_DEVICE
    void* get_bulk_combine_packet_ptr(const bool send,
                                      const uint32_t& peer_in_lsa) const {
        DG_UNIFIED_ASSERT(bulk_combine);
        const uint32_t num_remote_peers = num_ranks / 2;
        DG_UNIFIED_ASSERT(peer_in_lsa < num_remote_peers);
        const uint32_t packet_idx =
            (send ? 0u : num_remote_peers) + peer_in_lsa;
        // Bulk mode exclusively aliases the existing row-outbox allocation
        // plus a small appended tail.  Uniform fallback launches continue to
        // use combine_outbox_buffer's unchanged slot/row formula.
        auto* packet = math::advance_ptr(
            combine_outbox_buffer.base,
            static_cast<uint64_t>(packet_idx) * bulk_packet_bytes);
        DG_UNIFIED_ASSERT(
            reinterpret_cast<uintptr_t>(packet) %
                kMegaMoeGinBulkCombineRecordAlignment == 0);
        return packet;
    }

    CUTLASS_DEVICE
    uint32_t* get_bulk_combine_packet_count_ptr(
        const bool send, const uint32_t& peer_in_lsa) const {
        DG_DEVICE_ASSERT(bulk_combine);
        return static_cast<uint32_t*>(
            get_bulk_combine_packet_ptr(send, peer_in_lsa));
    }

    CUTLASS_HOST_DEVICE
    uint64_t* get_combine_terminal_signal_ptr(
        const uint32_t& owner_in_lsa) const {
        DG_UNIFIED_ASSERT(bulk_combine);
        DG_UNIFIED_ASSERT(owner_in_lsa < num_ranks / 2u);
        auto* signal = combine_terminal_signal_buffer
            .get_rank_buffer(owner_in_lsa)
            .get_data_buffer(0)
            .template get_base_ptr<uint64_t>();
        DG_UNIFIED_ASSERT(
            reinterpret_cast<uintptr_t>(signal) %
                kMegaMoeGinCombineTerminalSignalStride == 0);
        return signal;
    }

    CUTLASS_DEVICE
    void* get_bulk_combine_record_ptr(const bool send,
                                      const uint32_t& peer_in_lsa,
                                      const uint32_t& return_idx) const {
        DG_DEVICE_ASSERT(bulk_combine);
        DG_DEVICE_ASSERT(
            kMegaMoeGinBulkCombineRecordAreaOffset +
                    (static_cast<uint64_t>(return_idx) + 1ull) *
                        bulk_record_bytes <=
                bulk_packet_bytes);
        auto* record = math::advance_ptr(
            get_bulk_combine_packet_ptr(send, peer_in_lsa),
            kMegaMoeGinBulkCombineRecordAreaOffset +
                return_idx * bulk_record_bytes);
        DG_DEVICE_ASSERT(
            reinterpret_cast<uintptr_t>(record) %
                kMegaMoeGinBulkCombineRecordAlignment == 0);
        return record;
    }

    CUTLASS_DEVICE
    uint32_t* get_bulk_combine_record_destination_ptr(
        const bool send, const uint32_t& peer_in_lsa,
        const uint32_t& return_idx) const {
        return static_cast<uint32_t*>(
            get_bulk_combine_record_ptr(send, peer_in_lsa, return_idx));
    }

    CUTLASS_DEVICE
    void* get_bulk_combine_record_payload_ptr(
        const bool send, const uint32_t& peer_in_lsa,
        const uint32_t& return_idx) const {
        DG_DEVICE_ASSERT(kMegaMoeGinBulkCombineRecordHeaderBytes <=
                         bulk_record_bytes);
        return math::advance_ptr(
            get_bulk_combine_record_ptr(send, peer_in_lsa, return_idx),
            kMegaMoeGinBulkCombineRecordHeaderBytes);
    }
#endif
};

struct MegaMoEBuffer {
    Workspace workspace;

    // Input buffers (per-rank)
    Buffer input_token_buffer,
           input_sf_buffer,
           input_topk_idx_buffer,
           input_topk_weights_buffer;

    // Routed expert ring buffers
    // NOTE: shared L1 tokens reuse `input_token_buffer`.
    Buffer shared_l1_token_buffer, shared_l1_sf_buffer,
           shared_l2_token_buffer, shared_l2_sf_buffer;

    // Routed expert ring buffers
    Buffer l1_token_buffer,
           l1_sf_buffer,
           l1_topk_weights_buffer,
           l2_token_buffer,
           l2_sf_buffer,
           combine_token_buffer;

    MegaMoeGinWorkspace gin_workspace;
    bool with_gin;

    CUTLASS_HOST_DEVICE
    MegaMoEBuffer(void* base,
                  const uint32_t& hidden,
                  const uint32_t& intermediate_hidden,
                  const uint32_t& num_ranks,
                  const uint32_t& num_experts,
                  const uint32_t& num_max_tokens_per_rank,
                  const uint32_t& num_topk,
                  const uint32_t& num_ring_tokens,
                  const uint32_t& num_sf_ring_tokens,
                  const bool& with_sf,
                  const uint32_t& num_shared_experts = 0,
                  const bool& with_gin = false,
                  const uint32_t& num_sms = 0,
                  const uint32_t& gin_completion_batch = 1,
                  const uint32_t& gin_outbox_depth = 8,
                  const bool& gin_bulk_combine = false):
        with_gin(with_gin) {
        // Workspace
        workspace = Workspace(base, num_ranks, num_experts,
                              num_max_tokens_per_rank, num_topk, num_ring_tokens);

        // Shared
        const auto shared_intermediate_hidden = intermediate_hidden * num_shared_experts;
        const auto num_max_shared_sf_tokens = with_sf ? get_num_max_shared_sf_tokens(num_max_tokens_per_rank) : 0u;

        // Layouts
        const uint32_t num_mma_elem_bytes = with_sf ? 1 : 2;
        const auto input_token_layout = layout::Data(hidden * num_mma_elem_bytes);
        const auto bf16_token_layout = layout::Data(hidden * 2);
        const auto intermediate_token_layout = layout::Data(intermediate_hidden * num_mma_elem_bytes);
        const auto shared_intermediate_token_layout = layout::Data(shared_intermediate_hidden * num_mma_elem_bytes);
        const auto input_sf_layout = layout::Data(with_sf ? hidden / 32 : 0);
        const auto intermediate_sf_layout = layout::Data(with_sf ? intermediate_hidden / 32 : 0);
        const auto shared_intermediate_sf_layout = layout::Data(with_sf ? shared_intermediate_hidden / 32 : 0);
        const auto input_topk_idx_layout = layout::Data(num_topk * sizeof(int64_t), false);
        const auto input_topk_weights_layout = layout::Data(num_topk * sizeof(float), false);
        const auto l1_topk_weights_layout = layout::Data(sizeof(float), false);

        // Input buffers
        input_token_buffer = Buffer(
            input_token_layout, 1, num_max_tokens_per_rank,
            workspace.get_end_ptr());
        input_sf_buffer = Buffer(
            input_sf_layout, 1, num_max_tokens_per_rank,
            input_token_buffer.get_end_ptr());
        input_topk_idx_buffer = Buffer(
            input_topk_idx_layout, 1, num_max_tokens_per_rank,
            with_sf ? input_sf_buffer.get_end_ptr() : input_token_buffer.get_end_ptr());
        input_topk_weights_buffer = Buffer(
            input_topk_weights_layout, 1, num_max_tokens_per_rank,
            input_topk_idx_buffer.get_end_ptr());

        // Shared expert buffers
        shared_l1_token_buffer = input_token_buffer;
        shared_l1_sf_buffer = Buffer(
            input_sf_layout, 1, num_shared_experts > 0 ? num_max_shared_sf_tokens : 0,
            input_topk_weights_buffer.get_end_ptr());
        shared_l2_token_buffer = Buffer(
            shared_intermediate_token_layout, 1, num_shared_experts > 0 ? num_max_tokens_per_rank : 0,
            with_sf ? shared_l1_sf_buffer.get_end_ptr() : input_topk_weights_buffer.get_end_ptr());
        shared_l2_sf_buffer = Buffer(
            shared_intermediate_sf_layout, 1, num_shared_experts > 0 ? num_max_shared_sf_tokens : 0,
            shared_l2_token_buffer.get_end_ptr());

        // Routed expert ring buffers
        l1_token_buffer = Buffer(
            input_token_layout, 1, num_ring_tokens,
            num_shared_experts > 0 ?
                (with_sf ? shared_l2_sf_buffer.get_end_ptr() : shared_l2_token_buffer.get_end_ptr()) :
                input_topk_weights_buffer.get_end_ptr()
        );
        l1_sf_buffer = Buffer(
            input_sf_layout, 1, num_sf_ring_tokens,
            l1_token_buffer.get_end_ptr());
        l1_topk_weights_buffer = Buffer(
            l1_topk_weights_layout, 1, num_ring_tokens,
            with_sf ? l1_sf_buffer.get_end_ptr() : l1_token_buffer.get_end_ptr());

        l2_token_buffer = Buffer(
            intermediate_token_layout, 1, num_ring_tokens,
            l1_topk_weights_buffer.get_end_ptr());
        l2_sf_buffer = Buffer(
            intermediate_sf_layout, 1, num_sf_ring_tokens,
            l2_token_buffer.get_end_ptr());

        combine_token_buffer = Buffer(
            bf16_token_layout, num_topk + (num_shared_experts > 0 ? 1u : 0u), num_max_tokens_per_rank,
            with_sf ? l2_sf_buffer.get_end_ptr() : l2_token_buffer.get_end_ptr());

        if (with_gin) {
            DG_UNIFIED_ASSERT(with_sf);
            DG_UNIFIED_ASSERT(num_sms > 0);
            gin_workspace = MegaMoeGinWorkspace(
                combine_token_buffer.get_end_ptr(), hidden, num_ranks,
                num_experts, num_max_tokens_per_rank, num_topk, num_sms,
                gin_completion_batch, gin_outbox_depth, gin_bulk_combine);
        }
    }

    CUTLASS_HOST_DEVICE
    int64_t get_num_bytes() const {
        return static_cast<uint8_t*>(with_gin ? gin_workspace.get_end_ptr() :
                                              combine_token_buffer.get_end_ptr())
               - static_cast<uint8_t*>(workspace.base);
    }
};

} // namespace deep_gemm::layout
