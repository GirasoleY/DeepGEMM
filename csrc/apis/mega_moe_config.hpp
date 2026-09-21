#pragma once

#include <stdexcept>
#include <string>
#include <tuple>

namespace deep_gemm::mega {

// The immutable SymmBuffer allocation snapshot, shared by both transports.
using MegaMoeWorkspaceConfigTuple =
    std::tuple<int, int, int, int, int, int, std::string, std::string, int>;

struct MegaMoeWorkspaceConfig {
    int num_experts, num_max_tokens_per_rank, num_topk;
    int hidden, intermediate_hidden, num_shared_experts;
    std::string mma_type, activation;
    int max_active_tokens;

    explicit MegaMoeWorkspaceConfig(const MegaMoeWorkspaceConfigTuple& values) {
        std::tie(num_experts, num_max_tokens_per_rank, num_topk,
                 hidden, intermediate_hidden, num_shared_experts,
                 mma_type, activation, max_active_tokens) = values;
    }

    void validate(const int num_ranks) const {
        require(num_ranks > 0 and num_experts > 0 and num_experts % num_ranks == 0,
                "expert count must be positive and divisible by ranks");
        require(max_active_tokens > 0 and num_max_tokens_per_rank >= max_active_tokens,
                "storage capacity must cover the positive requested token capacity");
        require(num_topk > 0 and num_topk <= 32 and num_shared_experts >= 0 and
                    num_topk + (num_shared_experts > 0 ? 1 : 0) <= 32,
                "top-k plus the shared-expert result must fit one warp");
        require(hidden > 0 and intermediate_hidden > 0,
                "hidden dimensions must be positive");
        require(mma_type == "fp8xfp4" or mma_type == "fp8xfp8" or mma_type == "bf16xbf16",
                "unsupported MMA type");
        require(activation == "swiglu", "only SwiGLU is supported");
    }

    // Called after the existing weight-layout checks have derived the actual
    // geometry. This is allocation compatibility, not a transport restriction.
    void validate_launch(const int num_tokens, const int actual_hidden,
                         const int actual_intermediate_hidden,
                         const int actual_num_shared_experts,
                         const std::string& actual_mma_type,
                         const std::string& actual_activation) const {
        require(num_tokens >= 0 and num_tokens <= max_active_tokens,
                "launch exceeds the requested token capacity");
        require(actual_hidden == hidden and actual_intermediate_hidden == intermediate_hidden,
                "weight dimensions do not match the workspace");
        require(actual_num_shared_experts == num_shared_experts,
                "shared-expert count does not match the workspace");
        require(actual_mma_type == mma_type, "weight dtype does not match the workspace");
        require(actual_activation == activation, "activation does not match the workspace");
    }

private:
    static void require(const bool condition, const char* message) {
        if (not condition)
            throw std::invalid_argument(std::string("MegaMoE ") + message);
    }
};

} // namespace deep_gemm::mega
