#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/system.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#ifdef DG_MEGAMOE_GIN
#include <deep_gemm/comm/mega_moe_gin.cuh>
#endif

#include "../heuristics/mega_moe.hpp"

namespace deep_gemm {

class SM100FP8FP4MegaMoERuntime final : public LaunchRuntime<SM100FP8FP4MegaMoERuntime> {
public:
    struct Args {
        // Templated arguments
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_shared_experts, num_topk;
        int num_ranks;
        float activation_clamp;
        bool fast_math;
        bool use_gin;
        int gin_local_ablation_stage;
        bool gin_active_fast_path;
        bool gin_activity_gate_opt;
        bool gin_bulk_combine;
        bool gin_direct_dispatch;
        bool gin_dispatch_warp_scan;
        bool gin_coop_direct_pack;
        bool gin_preconsensus_pack;
        int gin_diagnostics;
        bool gin_single_combine_context;
        bool gin_dispatch_overlap;
        bool gin_combine_overlap;
        bool gin_strongva_combine_terminal;
        int gin_combine_owner_waves;
        bool gin_combine_owner_slot_ready;
        MegaMoEConfig config;

        // Runtime arguments
        void* y;
        int* cumulative_local_expert_recv_stats;
        int num_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;
#ifdef DG_MEGAMOE_GIN
        comm::MegaMoeGinTransport gin_transport;
#endif

        // Tensormap
        CUtensorMap tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_acts_sf;
        CUtensorMap tensor_map_l1_weights;
        CUtensorMap tensor_map_l1_weights_sf;
        CUtensorMap tensor_map_l1_output;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_acts_sf;
        CUtensorMap tensor_map_l2_weights;
        CUtensorMap tensor_map_l2_weights_sf;
        CUtensorMap tensor_map_shared_l1_acts;
        CUtensorMap tensor_map_shared_l1_acts_sf;
        CUtensorMap tensor_map_shared_l1_weights;
        CUtensorMap tensor_map_shared_l1_weights_sf;
        CUtensorMap tensor_map_shared_l1_output;
        CUtensorMap tensor_map_shared_l2_acts;
        CUtensorMap tensor_map_shared_l2_acts_sf;
        CUtensorMap tensor_map_shared_l2_weights;
        CUtensorMap tensor_map_shared_l2_weights_sf;

        // Launch configs
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#define DG_MEGAMOE_GIN_LOCAL_ABLATION_STAGE {}
#define DG_MEGAMOE_GIN_ACTIVE_FAST_PATH {}
#define DG_MEGAMOE_GIN_ACTIVITY_GATE_OPT {}
#define DG_MEGAMOE_GIN_BULK_COMBINE {}
#define DG_MEGAMOE_GIN_DIRECT_DISPATCH {}
#define DG_MEGAMOE_GIN_DISPATCH_WARP_SCAN {}
#define DG_MEGAMOE_GIN_COOP_DIRECT_PACK {}
#define DG_MEGAMOE_GIN_PRECONSENSUS_PACK {}
#define DG_MEGAMOE_GIN_DIAGNOSTICS {}
#define DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT {}
#define DG_MEGAMOE_GIN_DISPATCH_OVERLAP {}
#define DG_MEGAMOE_GIN_COMBINE_OVERLAP {}
#define DG_MEGAMOE_GIN_STRONGVA_COMBINE_TERMINAL {}
#define DG_MEGAMOE_GIN_COMBINE_OWNER_WAVES {}
#define DG_MEGAMOE_GIN_COMBINE_OWNER_SLOT_READY {}
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_fp4_mega_moe_impl<
        {},
        {}, {},
        {}, {},
        {}, {}, {},
        {},
        {}, {},
        {},
        {},
        {},
        {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {},
        {}
    >);
}};
)", args.gin_local_ablation_stage,
    args.gin_active_fast_path ? "1" : "0",
    args.gin_activity_gate_opt ? "1" : "0",
    args.gin_bulk_combine ? "1" : "0",
    args.gin_direct_dispatch ? "1" : "0",
    args.gin_dispatch_warp_scan ? "1" : "0",
    args.gin_coop_direct_pack ? "1" : "0",
    args.gin_preconsensus_pack ? "1" : "0",
    args.gin_diagnostics,
    args.gin_single_combine_context ? "1" : "0",
    args.gin_dispatch_overlap ? "1" : "0",
    args.gin_combine_overlap ? "1" : "0",
    args.gin_strongva_combine_terminal ? "1" : "0",
    args.gin_combine_owner_waves,
    args.gin_combine_owner_slot_ready ? "1" : "0",
    args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_shared_experts,
    args.num_topk,
    args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.store_block_m,
    args.config.sf_block_m, args.config.sf_block_n,
    args.config.num_ring_tokens,
    args.config.num_sf_ring_tokens,
    args.config.num_stages,
    args.config.num_bytes_per_pull,
    args.config.num_dispatch_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.launch_args.grid_dim.first, args.num_ranks,
    to_string(args.activation_clamp),
    args.fast_math ? "true" : "false",
    args.use_gin ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // TODO: optimize `args` copy
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.y,
            args.cumulative_local_expert_recv_stats,
            args.num_tokens,
            args.sym_buffer_ptrs,
#ifdef DG_MEGAMOE_GIN
            args.gin_transport,
#endif
            args.tensor_map_l1_acts,
            args.tensor_map_l1_acts_sf,
            args.tensor_map_l1_weights,
            args.tensor_map_l1_weights_sf,
            args.tensor_map_l1_output,
            args.tensor_map_l2_acts,
            args.tensor_map_l2_acts_sf,
            args.tensor_map_l2_weights,
            args.tensor_map_l2_weights_sf,
            args.tensor_map_shared_l1_acts,
            args.tensor_map_shared_l1_acts_sf,
            args.tensor_map_shared_l1_weights,
            args.tensor_map_shared_l1_weights_sf,
            args.tensor_map_shared_l1_output,
            args.tensor_map_shared_l2_acts,
            args.tensor_map_shared_l2_acts_sf,
            args.tensor_map_shared_l2_weights,
            args.tensor_map_shared_l2_weights_sf
        ));
    }
};

static void sm100_fp8_fp4_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& shared_l1_acts, const torch::Tensor& shared_l1_acts_sf,
    const torch::Tensor& shared_l2_acts, const torch::Tensor& shared_l2_acts_sf,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
    const torch::Tensor& shared_l1_weights, const torch::Tensor& shared_l2_weights,
    const torch::Tensor& shared_l1_weights_sf, const torch::Tensor& shared_l2_weights_sf,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_shared_experts,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math
#ifdef DG_MEGAMOE_GIN
    , const std::optional<comm::MegaMoeGinTransport>& gin_transport_opt = std::nullopt
#endif
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_ring_tokens = static_cast<int>(l1_acts.size(0));
    const auto num_sf_ring_tokens = static_cast<int>(l1_acts_sf.size(0));
    const auto shared_intermediate_hidden = intermediate_hidden * num_shared_experts;

    // Heuristics
    const auto config = get_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk, hidden, intermediate_hidden,
        num_ring_tokens, num_sf_ring_tokens,
        MmaKind::MXFP8FP4);

    // Make tensormap
    constexpr int kGranK = 32;
    const int sf_smem_outer_dim = config.block_k / (kGranK * 4);
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, config.num_ring_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        config.num_sf_ring_tokens, hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0, 0, false,
                                                        sf_smem_outer_dim);
    const auto tensor_map_l1_weights = make_tma_2d_desc(l1_weights,
                                                        hidden, num_experts_per_rank * intermediate_hidden * 2,
                                                        config.block_k, config.load_block_n,
                                                        static_cast<int>(l1_weights.stride(-2)),
                                                        config.swizzle_weights_mode);
    const auto tensor_map_l1_weights_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_weights_sf,
                                                           intermediate_hidden * 2, hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0, 0, false,
                                                        sf_smem_outer_dim);
    // NOTES: L1 output and L2 activations are essentially the same tensor.
    // Post-SwiGLU output has half the N width (`BLOCK_N / 2` per input tile),
    // so the swizzle mode is also halved (128 -> 64).
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, config.num_ring_tokens,
                                                       config.block_n / 2, config.store_block_m,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       config.swizzle_acts_mode / 2);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden, config.num_ring_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        config.num_sf_ring_tokens, intermediate_hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0, 0, false,
                                                        sf_smem_outer_dim);
    const auto tensor_map_l2_weights = make_tma_2d_desc(l2_weights,
                                                        intermediate_hidden, num_experts_per_rank * hidden,
                                                        config.block_k, config.load_block_n,
                                                        static_cast<int>(l2_weights.stride(-2)),
                                                        config.swizzle_weights_mode);
    const auto tensor_map_l2_weights_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_weights_sf,
                                                           hidden, intermediate_hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0, 0, false,
                                                        sf_smem_outer_dim);

    const auto tensor_map_shared_l1_acts = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l1_acts,
        hidden, num_max_tokens_per_rank,
        config.block_k, config.load_block_m,
        static_cast<int>(shared_l1_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l1_acts;
    const auto tensor_map_shared_l1_acts_sf = num_shared_experts > 0 ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l1_acts_sf,
        static_cast<int>(shared_l1_acts_sf.size(0)), hidden,
        config.sf_block_m, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim) : tensor_map_l1_acts_sf;
    const auto tensor_map_shared_l1_weights = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l1_weights,
        hidden, shared_intermediate_hidden * 2,
        config.block_k, config.load_block_n,
        static_cast<int>(shared_l1_weights.stride(-2)),
        config.swizzle_weights_mode) : tensor_map_l1_weights;
    const auto tensor_map_shared_l1_weights_sf = num_shared_experts > 0 ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l1_weights_sf,
        shared_intermediate_hidden * 2, hidden,
        config.block_n, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim) : tensor_map_l1_weights_sf;
    const auto tensor_map_shared_l1_output = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_acts,
        shared_intermediate_hidden, num_max_tokens_per_rank,
        config.block_n / 2, config.store_block_m,
        static_cast<int>(shared_l2_acts.stride(-2)),
        config.swizzle_acts_mode / 2) : tensor_map_l1_output;
    const auto tensor_map_shared_l2_acts = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_acts,
        shared_intermediate_hidden, num_max_tokens_per_rank,
        config.block_k, config.load_block_m,
        static_cast<int>(shared_l2_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l2_acts;
    const auto tensor_map_shared_l2_acts_sf = num_shared_experts > 0 ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l2_acts_sf,
        static_cast<int>(shared_l2_acts_sf.size(0)), shared_intermediate_hidden,
        config.sf_block_m, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim) : tensor_map_l2_acts_sf;
    const auto tensor_map_shared_l2_weights = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_weights,
        shared_intermediate_hidden, hidden,
        config.block_k, config.load_block_n,
        static_cast<int>(shared_l2_weights.stride(-2)),
        config.swizzle_weights_mode) : tensor_map_l2_weights;
    const auto tensor_map_shared_l2_weights_sf = num_shared_experts > 0 ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l2_weights_sf,
        hidden, shared_intermediate_hidden,
        config.block_n, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim) : tensor_map_l2_weights_sf;

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    // Launch
    const auto num_sms = device_runtime->get_num_sms();
    // Fail closed for removed experimental paths, even with context mode off.
    DG_HOST_ASSERT(get_env<std::string>("DG_MEGAMOE_GIN_COMBINE_EXPERTS_PER_WAVE", "0") == "0");
    DG_HOST_ASSERT(get_env<std::string>("DG_MEGAMOE_GIN_COMBINE_BARRIER_WARPS", "1") == "1");
    const auto gin_combine_overlap_value =
        get_env<std::string>("DG_MEGAMOE_GIN_COMBINE_OVERLAP", "0");
    DG_HOST_ASSERT(gin_combine_overlap_value == "0" or
                   gin_combine_overlap_value == "1");
    const auto gin_strongva_combine_terminal_value =
        get_env<std::string>(
            "DG_MEGAMOE_GIN_STRONGVA_COMBINE_TERMINAL", "0");
    DG_HOST_ASSERT(gin_strongva_combine_terminal_value == "0" or
                   gin_strongva_combine_terminal_value == "1");
    const auto gin_combine_owner_waves_value =
        get_env<std::string>(
            "DG_MEGAMOE_GIN_COMBINE_OWNER_WAVES", "0");
    DG_HOST_ASSERT(gin_combine_owner_waves_value == "0" or
                   gin_combine_owner_waves_value == "2" or
                   gin_combine_owner_waves_value == "4" or
                   gin_combine_owner_waves_value == "8");
    const auto gin_combine_owner_slot_ready_value =
        get_env<std::string>(
            "DG_MEGAMOE_GIN_COMBINE_OWNER_SLOT_READY", "0");
    DG_HOST_ASSERT(gin_combine_owner_slot_ready_value == "0" or
                   gin_combine_owner_slot_ready_value == "1");
#ifdef DG_MEGAMOE_GIN
    const int gin_local_ablation_stage = gin_transport_opt.has_value() ?
        get_env<int>("DG_MEGAMOE_GIN_LOCAL_ABLATION_STAGE", 0) : 0;
    DG_HOST_ASSERT(gin_local_ablation_stage >= 0 and
                   gin_local_ablation_stage <= 4);
    DG_HOST_ASSERT(gin_local_ablation_stage == 0 or
                   (gin_transport_opt.has_value() and
                    (num_ranks == 8 or num_ranks == 16) and
                    num_shared_experts == 0));
    const bool gin_active_fast_path = gin_transport_opt.has_value() and
        gin_transport_opt->active_fast_path != 0;
    DG_HOST_ASSERT(not gin_active_fast_path or
                   (gin_local_ablation_stage == 0 and
                    (num_ranks == 8 or num_ranks == 16) and
                    num_shared_experts == 0));
    const int gin_activity_gate_opt_value =
        get_env<int>("DG_MEGAMOE_GIN_ACTIVITY_GATE_OPT", 0);
    DG_HOST_ASSERT(gin_activity_gate_opt_value == 0 or
                   gin_activity_gate_opt_value == 1);
    const bool gin_activity_gate_opt =
        gin_transport_opt.has_value() and gin_activity_gate_opt_value != 0;
    DG_HOST_ASSERT(not gin_activity_gate_opt or gin_active_fast_path);
    const bool gin_bulk_combine = gin_transport_opt.has_value() and
        gin_transport_opt->bulk_combine != 0 and
        gin_local_ablation_stage == 0 and
        ((num_ranks == 16 and num_experts == 896) or
         (num_ranks == 8 and num_experts == 448)) and
        num_topk == 16 and hidden == 3584 and
        intermediate_hidden == 3072 and num_shared_experts == 0 and
        gin_transport_opt->outbox_depth == 64;
    DG_HOST_ASSERT(not gin_transport_opt.has_value() or
                   gin_transport_opt->bulk_combine == 0 or
                   (gin_bulk_combine and gin_active_fast_path));
    const bool gin_direct_dispatch = gin_transport_opt.has_value() and
        gin_transport_opt->direct_dispatch != 0 and
        gin_local_ablation_stage == 0 and
        ((num_ranks == 16 and num_experts == 896) or
         (num_ranks == 8 and num_experts == 448)) and
        num_topk == 16 and hidden == 3584 and
        intermediate_hidden == 3072 and num_shared_experts == 0 and
        num_max_tokens_per_rank >= 384 and
        static_cast<int64_t>(num_sms) *
                layout::kMegaMoeGinNumDispatchWarps * (hidden / 32) >=
            layout::kMegaMoeGinDirectDispatchStorageBytes;
    DG_HOST_ASSERT(not gin_transport_opt.has_value() or
                   gin_transport_opt->direct_dispatch == 0 or
                   (gin_direct_dispatch and gin_active_fast_path));
    const int gin_dispatch_warp_scan_value =
        get_env<int>("DG_MEGAMOE_GIN_DISPATCH_WARP_SCAN", 0);
    DG_HOST_ASSERT(gin_dispatch_warp_scan_value == 0 or
                   gin_dispatch_warp_scan_value == 1);
    const bool gin_dispatch_warp_scan =
        gin_dispatch_warp_scan_value != 0;
    DG_HOST_ASSERT(not gin_dispatch_warp_scan or gin_direct_dispatch);
    const int gin_coop_direct_pack_value =
        get_env<int>("DG_MEGAMOE_GIN_COOP_DIRECT_PACK", 0);
    DG_HOST_ASSERT(gin_coop_direct_pack_value == 0 or
                   gin_coop_direct_pack_value == 1);
    const bool gin_coop_direct_pack = gin_coop_direct_pack_value != 0;
    DG_HOST_ASSERT(not gin_coop_direct_pack or gin_direct_dispatch);
    const int gin_preconsensus_pack_value =
        get_env<int>("DG_MEGAMOE_GIN_PRECONSENSUS_PACK", 0);
    DG_HOST_ASSERT(gin_preconsensus_pack_value == 0 or
                   gin_preconsensus_pack_value == 1);
    const bool gin_preconsensus_pack =
        gin_preconsensus_pack_value != 0;
    DG_HOST_ASSERT(not gin_preconsensus_pack or
                   (gin_coop_direct_pack and gin_active_fast_path));
    const int gin_diagnostics = get_env<int>("DG_MEGAMOE_GIN_DIAGNOSTICS", 0);
    DG_HOST_ASSERT(gin_diagnostics >= 0 and gin_diagnostics <= 2);
    DG_HOST_ASSERT(gin_diagnostics == 0 or
                   (gin_transport_opt.has_value() and
                    gin_transport_opt->diagnostic_buffer != nullptr and
                    gin_transport_opt->diagnostic_num_sms >= num_sms and
                    num_experts_per_rank == 56 and
                    config.num_dispatch_threads == 128 and
                    config.num_epilogue_threads <= 256));
    const auto gin_single_combine_context_value =
        get_env<std::string>("DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT", "0");
    DG_HOST_ASSERT(gin_single_combine_context_value == "0" or
                   gin_single_combine_context_value == "1");
    const bool gin_single_combine_context = gin_single_combine_context_value == "1";
    DG_HOST_ASSERT(not gin_single_combine_context or
                   (gin_bulk_combine and gin_direct_dispatch));
    const auto gin_dispatch_overlap_value =
        get_env<std::string>("DG_MEGAMOE_GIN_DISPATCH_OVERLAP", "0");
    DG_HOST_ASSERT(gin_dispatch_overlap_value == "0" or
                   gin_dispatch_overlap_value == "1");
    const bool gin_dispatch_overlap = gin_dispatch_overlap_value == "1";
    DG_HOST_ASSERT(not gin_dispatch_overlap or
                   (gin_direct_dispatch and gin_bulk_combine and
                    gin_preconsensus_pack));
    const bool gin_combine_overlap = gin_combine_overlap_value == "1";
    DG_HOST_ASSERT(not gin_combine_overlap or
                   (gin_single_combine_context and gin_dispatch_overlap and
                    gin_direct_dispatch and gin_bulk_combine));
    // Fixed expert-counter/prefix scratch fit is checked in the kernel.
    // Insufficient tail space uses the unchanged whole-packet path; never
    // resize registered storage or change the selected compute configuration.
    const bool gin_strongva_combine_terminal =
        gin_strongva_combine_terminal_value == "1";
    DG_HOST_ASSERT(not gin_strongva_combine_terminal or
                   (gin_combine_overlap and gin_single_combine_context and
                    gin_dispatch_overlap and gin_direct_dispatch and
                    gin_bulk_combine));
    // Removing a world collective cannot depend on the kernel's rank-local
    // alias fallback. Fail closed on the conservative completion-batch=1
    // scratch bound, then let runtime eligibility use only world consensus.
    DG_HOST_ASSERT(
        not gin_strongva_combine_terminal or
        static_cast<int64_t>(num_sms) *
                layout::kMegaMoeGinNumDispatchWarps * (hidden / 32) >=
            layout::kMegaMoeGinDirectDispatchStorageBytes +
                layout::get_mega_moe_gin_combine_overlap_scratch_bytes());
    const int gin_combine_owner_waves =
        gin_combine_owner_waves_value == "2" ? 2 :
        gin_combine_owner_waves_value == "4" ? 4 :
        gin_combine_owner_waves_value == "8" ? 8 : 0;
    DG_HOST_ASSERT(not gin_combine_owner_waves or
                   (num_ranks == 8 and num_experts == 448 and
                    gin_strongva_combine_terminal and gin_combine_overlap and
                    gin_single_combine_context and gin_dispatch_overlap and
                    gin_direct_dispatch and gin_bulk_combine and
                    num_experts_per_rank == 56));
    const bool gin_combine_owner_slot_ready =
        gin_combine_owner_slot_ready_value == "1";
    DG_HOST_ASSERT(not gin_combine_owner_slot_ready or
                   (gin_combine_owner_waves == 4 and num_ranks == 8 and
                    num_experts == 448 and num_experts_per_rank == 56 and
                    num_topk == 16 and hidden == 3584 and
                    intermediate_hidden == 3072 and
                    num_shared_experts == 0 and
                    gin_strongva_combine_terminal and gin_combine_overlap and
                    gin_single_combine_context and gin_dispatch_overlap and
                    gin_direct_dispatch and gin_bulk_combine and
                    config.num_epilogue_threads == 256));
#else
    DG_HOST_ASSERT(gin_combine_overlap_value == "0");
    DG_HOST_ASSERT(gin_strongva_combine_terminal_value == "0");
    DG_HOST_ASSERT(gin_combine_owner_waves_value == "0");
    DG_HOST_ASSERT(gin_combine_owner_slot_ready_value == "0");
    constexpr int gin_local_ablation_stage = 0;
    constexpr bool gin_active_fast_path = false;
    constexpr bool gin_activity_gate_opt = false;
    constexpr bool gin_bulk_combine = false;
    constexpr bool gin_direct_dispatch = false;
    constexpr bool gin_dispatch_warp_scan = false;
    constexpr bool gin_coop_direct_pack = false;
    constexpr bool gin_preconsensus_pack = false;
    constexpr int gin_diagnostics = 0;
    constexpr bool gin_single_combine_context = false;
    constexpr bool gin_dispatch_overlap = false;
    constexpr bool gin_combine_overlap = false;
    constexpr bool gin_strongva_combine_terminal = false;
    constexpr int gin_combine_owner_waves = 0;
    constexpr bool gin_combine_owner_slot_ready = false;
#endif
    const SM100FP8FP4MegaMoERuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_shared_experts = num_shared_experts,
        .num_topk = num_topk,
        .num_ranks = num_ranks,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
#ifdef DG_MEGAMOE_GIN
        .use_gin = gin_transport_opt.has_value(),
#else
        .use_gin = false,
#endif
        .gin_local_ablation_stage = gin_local_ablation_stage,
        .gin_active_fast_path = gin_active_fast_path,
        .gin_activity_gate_opt = gin_activity_gate_opt,
        .gin_bulk_combine = gin_bulk_combine,
        .gin_direct_dispatch = gin_direct_dispatch,
        .gin_dispatch_warp_scan = gin_dispatch_warp_scan,
        .gin_coop_direct_pack = gin_coop_direct_pack,
        .gin_preconsensus_pack = gin_preconsensus_pack,
        .gin_diagnostics = gin_diagnostics,
        .gin_single_combine_context = gin_single_combine_context,
        .gin_dispatch_overlap = gin_dispatch_overlap,
        .gin_combine_overlap = gin_combine_overlap,
        .gin_strongva_combine_terminal = gin_strongva_combine_terminal,
        .gin_combine_owner_waves = gin_combine_owner_waves,
        .gin_combine_owner_slot_ready = gin_combine_owner_slot_ready,
        .config = config,
        .y = y.data_ptr(),
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats_ptr,
        .num_tokens = num_tokens,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
#ifdef DG_MEGAMOE_GIN
        .gin_transport = gin_transport_opt.value_or(comm::MegaMoeGinTransport{}),
#endif
        .tensor_map_l1_acts = tensor_map_l1_acts,
        .tensor_map_l1_acts_sf = tensor_map_l1_acts_sf,
        .tensor_map_l1_weights = tensor_map_l1_weights,
        .tensor_map_l1_weights_sf = tensor_map_l1_weights_sf,
        .tensor_map_l1_output = tensor_map_l1_output,
        .tensor_map_l2_acts = tensor_map_l2_acts,
        .tensor_map_l2_acts_sf = tensor_map_l2_acts_sf,
        .tensor_map_l2_weights = tensor_map_l2_weights,
        .tensor_map_l2_weights_sf = tensor_map_l2_weights_sf,
        .tensor_map_shared_l1_acts = tensor_map_shared_l1_acts,
        .tensor_map_shared_l1_acts_sf = tensor_map_shared_l1_acts_sf,
        .tensor_map_shared_l1_weights = tensor_map_shared_l1_weights,
        .tensor_map_shared_l1_weights_sf = tensor_map_shared_l1_weights_sf,
        .tensor_map_shared_l1_output = tensor_map_shared_l1_output,
        .tensor_map_shared_l2_acts = tensor_map_shared_l2_acts,
        .tensor_map_shared_l2_acts_sf = tensor_map_shared_l2_acts_sf,
        .tensor_map_shared_l2_weights = tensor_map_shared_l2_weights,
        .tensor_map_shared_l2_weights_sf = tensor_map_shared_l2_weights_sf,
        .launch_args = LaunchArgs(num_sms,
                                  config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size, 2)
    };

    const auto code = SM100FP8FP4MegaMoERuntime::generate(args);
    const auto runtime = compiler->build("sm100_fp8_fp4_mega_moe", code);
    SM100FP8FP4MegaMoERuntime::launch(runtime, args);
}

} // namespace deep_gemm
