#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <torch/python.h>

#include <deep_gemm/comm/mega_moe_gin_transport.h>

#include "mega_moe.hpp"

#ifdef DG_MEGAMOE_GIN
#include <deep_gemm/comm/mega_moe_gin.cuh>
#endif

namespace deep_gemm::mega::gin {

namespace py = pybind11;

static constexpr int kRequiredNcclVersion =
    DG_MEGAMOE_GIN_REQUIRED_NCCL_VERSION_CODE;
static constexpr int kWorldSize = 8;
static constexpr int kLsaSize = 4;
static constexpr int kNumExperts = 448;
static constexpr int kNumTopk = 16;
static constexpr int kHidden = 3584;
static constexpr int kIntermediateHidden = 3072;
static constexpr int kMaxTokensPerRank = 48;
static constexpr int kRequiredContextCount = 6;
static constexpr int kRequiredQueueDepth = 64;
static constexpr int kRequiredWorldBarrierCount = 2;

static std::runtime_error unsupported(const std::string& reason,
                                      const std::string& detail) {
    return std::runtime_error(
        "MegaMoE GIN unsupported [reason=" + reason + "]: " + detail);
}

#ifdef DG_MEGAMOE_GIN

static_assert(NCCL_VERSION_CODE == kRequiredNcclVersion,
              "MegaMoE GIN requires exactly NCCL 2.30.7");

static void require(const bool condition, const std::string& message) {
    if (not condition)
        throw std::invalid_argument(message);
}

static auto get_symm_buffer_size_for_mega_moe_gin() {
    // Keep every native input/ring offset byte-for-byte identical, then append
    // the fixed EP8 transport workspace at its required alignment.
    auto [native_num_bytes, slice_input_buffers] =
        ::deep_gemm::mega::get_symm_buffer_size_for_mega_moe(
            kWorldSize, kNumExperts,
            layout::kLCMCandidateBlockM, kNumTopk,
            kHidden, kIntermediateHidden,
            "fp8xfp4", "swiglu", 0);
    const auto gin_offset = math::align<uint64_t>(
        static_cast<uint64_t>(native_num_bytes),
        layout::kMegaMoEEp8GinAlignment);
    return std::make_tuple(
        static_cast<int64_t>(
            gin_offset + layout::MegaMoeEp8GinWorkspace::kNumBytes),
        std::move(slice_input_buffers));
}

class MegaMoeGinTransportHandle final {
public:
    static std::shared_ptr<MegaMoeGinTransportHandle> import(
        const py::object& capsule_object,
        const torch::Tensor& buffer,
        const int rank,
        const py::object& owner) {
        if (owner.is_none())
            throw std::invalid_argument(
                "MegaMoE GIN requires a non-None external owner");
        if (not PyCapsule_CheckExact(capsule_object.ptr()) or
            not PyCapsule_IsValid(
                capsule_object.ptr(), DG_MEGAMOE_GIN_TRANSPORT_CAPSULE_NAME)) {
            throw std::invalid_argument(
                "MegaMoE GIN requires a PyCapsule named \""
                DG_MEGAMOE_GIN_TRANSPORT_CAPSULE_NAME "\"");
        }

        const auto capsule = py::reinterpret_borrow<py::capsule>(capsule_object);
        const auto* raw_descriptor = static_cast<const uint8_t*>(
            PyCapsule_GetPointer(
                capsule.ptr(), DG_MEGAMOE_GIN_TRANSPORT_CAPSULE_NAME));
        if (raw_descriptor == nullptr)
            throw py::error_already_set();

        struct DescriptorHeader {
            uint32_t abi_version;
            uint32_t struct_bytes;
        } header{};
        std::memcpy(&header, raw_descriptor, sizeof(header));
        if (header.abi_version != DG_MEGAMOE_GIN_TRANSPORT_ABI_VERSION) {
            throw unsupported(
                "descriptor_abi_version",
                "expected ABI version 1, got " +
                    std::to_string(header.abi_version));
        }
        if (header.struct_bytes != sizeof(DGMegaMoEGinTransportV1)) {
            throw unsupported(
                "descriptor_size",
                "descriptor has " + std::to_string(header.struct_bytes) +
                    " bytes, but ABI v1 requires exactly " +
                    std::to_string(sizeof(DGMegaMoEGinTransportV1)));
        }

        DGMegaMoEGinTransportV1 descriptor{};
        std::memcpy(&descriptor, raw_descriptor, sizeof(descriptor));
        validate_descriptor(descriptor, buffer, rank);
        return std::shared_ptr<MegaMoeGinTransportHandle>(
            new MegaMoeGinTransportHandle(
                descriptor, buffer, owner, rank));
    }

    MegaMoeGinTransportHandle(const MegaMoeGinTransportHandle&) = delete;
    MegaMoeGinTransportHandle& operator=(
        const MegaMoeGinTransportHandle&) = delete;

    void validate_binding(const torch::Tensor& buffer, const int rank) const {
        require(buffer.defined() and buffer.is_cuda() and
                    buffer.scalar_type() == torch::kInt8 and
                    buffer.dim() == 1 and buffer.is_contiguous(),
                "MegaMoE GIN transport requires its original CUDA int8 buffer");
        require(buffer.data_ptr() == buffer_.data_ptr() and
                    buffer.nbytes() == buffer_.nbytes(),
                "MegaMoE GIN transport is bound to a different symmetric buffer");
        require(buffer.get_device() == buffer_.get_device(),
                "MegaMoE GIN transport buffer device changed after import");
        require(rank == rank_,
                "MegaMoE GIN transport rank changed after import");
    }

    comm::MegaMoeEp8GinTransport launch_descriptor() const {
        return comm::MegaMoeEp8GinTransport{
            descriptor_.dev_comm,
            descriptor_.window,
            descriptor_.window_base,
            descriptor_.window_bytes,
        };
    }

private:
    MegaMoeGinTransportHandle(
        const DGMegaMoEGinTransportV1& descriptor,
        torch::Tensor buffer,
        py::object owner,
        const int rank)
        : descriptor_(descriptor),
          buffer_(std::move(buffer)),
          owner_(std::move(owner)),
          rank_(rank) {}

    static void validate_descriptor(
        const DGMegaMoEGinTransportV1& descriptor,
        const torch::Tensor& buffer,
        const int rank) {
        require(descriptor.nccl_version_code == kRequiredNcclVersion,
                "MegaMoE GIN descriptor requires NCCL version code 23007");
        require(descriptor.dev_comm_bytes == sizeof(ncclDevComm_t),
                "MegaMoE GIN descriptor has a different ncclDevComm_t size");
        require(descriptor.nccl_headers_fingerprint ==
                    DG_NCCL_HEADERS_FINGERPRINT,
                "MegaMoE GIN descriptor has a different NCCL Device API header fingerprint");
        require(descriptor.queue_depth >= kRequiredQueueDepth,
                "MegaMoE GIN queue depth must be at least 64");
        require(descriptor.world_barrier_count >=
                    kRequiredWorldBarrierCount,
                "MegaMoE GIN requires at least 2 world barrier slots");
        require(descriptor.window != nullptr,
                "MegaMoE GIN descriptor window must not be null");
        require(descriptor.window_base != nullptr and
                    descriptor.window_bytes > 0,
                "MegaMoE GIN registered window extent must not be empty");
        require(
            reinterpret_cast<uintptr_t>(descriptor.window_base) %
                    NCCL_WIN_REQUIRED_ALIGNMENT ==
                0,
            "MegaMoE GIN registered window base has invalid NCCL alignment");

        require(buffer.defined() and buffer.is_cuda() and
                    buffer.scalar_type() == torch::kInt8 and
                    buffer.dim() == 1 and buffer.is_contiguous() and
                    buffer.nbytes() > 0,
                "MegaMoE GIN buffer must be a nonempty contiguous CUDA int8 tensor");
        require(
            buffer.nbytes() >= static_cast<size_t>(
                std::get<0>(get_symm_buffer_size_for_mega_moe_gin())),
            "MegaMoE GIN symmetric buffer is smaller than the fixed EP8 workspace");
        require(descriptor.cuda_device == buffer.get_device(),
                "MegaMoE GIN descriptor CUDA device does not match the buffer");
        require(rank >= 0 and rank < kWorldSize,
                "MegaMoE GIN rank must be in [0, 8)");
        require(descriptor.dev_comm.rank == rank and
                    descriptor.dev_comm.nRanks == kWorldSize,
                "MegaMoE GIN device communicator must match the EP8 rank ordering");
        require(descriptor.dev_comm.lsaSize == kLsaSize and
                    descriptor.dev_comm.lsaRank == rank % kLsaSize,
                "MegaMoE GIN requires contiguous two-by-LSA4 rank ordering");
        require(descriptor.dev_comm.ginContextCount >=
                    kRequiredContextCount,
                "MegaMoE GIN requires at least 6 GIN contexts");
        require(descriptor.dev_comm.ginConnectionCount > 0,
                "MegaMoE GIN device communicator has no GIN connections");
        require(not descriptor.dev_comm.ginConnectionsRailed and
                    not descriptor.dev_comm.ginContextsRailed,
                "MegaMoE GIN requires unrailed connections and contexts");

        const auto tensor_begin =
            reinterpret_cast<uintptr_t>(buffer.data_ptr());
        const auto window_begin =
            reinterpret_cast<uintptr_t>(descriptor.window_base);
        require(tensor_begin >= window_begin,
                "MegaMoE GIN buffer begins before the registered window");
        const uint64_t window_offset = tensor_begin - window_begin;
        require(window_offset <= descriptor.window_bytes and
                    buffer.nbytes() <=
                        descriptor.window_bytes - window_offset,
                "MegaMoE GIN buffer is not contained in the registered window");
        require(tensor_begin % 128 == 0 and window_offset % 128 == 0,
                "MegaMoE GIN buffer and window offset must be 128-byte aligned");
    }

    const DGMegaMoEGinTransportV1 descriptor_;
    const torch::Tensor buffer_;
    const py::object owner_;
    const int rank_;
};

static void fp8_fp4_mega_moe_gin(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int rank_idx,
    const int num_max_tokens_per_rank,
    const int gin_num_max_tokens_per_rank,
    const int num_experts,
    const int num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool fast_math,
    const std::shared_ptr<MegaMoeGinTransportHandle>& gin_transport) {
    require(gin_transport != nullptr,
            "MegaMoE GIN transport handle must not be null");
    gin_transport->validate_binding(sym_buffer, rank_idx);

    require(num_experts == kNumExperts and num_topk == kNumTopk,
            "MegaMoE GIN requires E448 and top-k 16");
    require(gin_num_max_tokens_per_rank >= 1 and
                gin_num_max_tokens_per_rank <= kMaxTokensPerRank,
            "MegaMoE GIN token capacity must be in [1, 48]");
    require(num_max_tokens_per_rank == layout::kLCMCandidateBlockM,
            "MegaMoE GIN native token capacity must use the 1920-token alignment");
    require(sym_buffer_ptrs.size() == kWorldSize,
            "MegaMoE GIN requires exactly 8 symmetric buffer pointers");
    for (int peer = 0; peer < kWorldSize; ++peer) {
        const bool same_lsa = peer / kLsaSize == rank_idx / kLsaSize;
        require((sym_buffer_ptrs[peer] != 0) == same_lsa,
                "MegaMoE GIN requires local-LSA pointers and null cross-LSA pointers");
    }
    require(sym_buffer_ptrs[rank_idx] ==
                static_cast<int64_t>(
                    reinterpret_cast<uintptr_t>(sym_buffer.data_ptr())),
            "MegaMoE GIN self pointer must use the original symmetric tensor VA");

    require(y.defined() and y.is_cuda() and
                y.scalar_type() == torch::kBFloat16 and
                y.dim() == 2 and y.size(1) == kHidden and
                y.is_contiguous() and y.device() == sym_buffer.device(),
            "MegaMoE GIN output must be contiguous CUDA BF16 [tokens, 3584] on the buffer device");
    const auto num_tokens = static_cast<int>(y.size(0));
    require(num_tokens <= gin_num_max_tokens_per_rank,
            "MegaMoE GIN launch exceeds the requested token capacity");

    const auto [l1_weights, l1_weights_sf] = l1_weights_tuple;
    const auto [l2_weights, l2_weights_sf] = l2_weights_tuple;
    require(l1_weights.device() == sym_buffer.device() and
                l2_weights.device() == sym_buffer.device() and
                l1_weights_sf.device() == sym_buffer.device() and
                l2_weights_sf.device() == sym_buffer.device(),
            "MegaMoE GIN weights and scales must be on the buffer device");

    if (cumulative_local_expert_recv_stats.has_value()) {
        require(cumulative_local_expert_recv_stats->is_cuda() and
                    cumulative_local_expert_recv_stats->device() ==
                        sym_buffer.device(),
                "MegaMoE GIN cumulative expert stats must be a CUDA tensor on the buffer device");
    }

    const auto launch_transport = gin_transport->launch_descriptor();
    ::deep_gemm::mega::fp8_fp4_mega_moe_impl(
        y, l1_weights_tuple, l2_weights_tuple,
        std::nullopt, std::nullopt,
        cumulative_local_expert_recv_stats,
        sym_buffer, sym_buffer_ptrs, rank_idx,
        num_max_tokens_per_rank, num_experts, num_topk,
        recipe, activation, activation_clamp_opt, fast_math,
        &launch_transport);

}

#endif  // DG_MEGAMOE_GIN

static void register_apis(py::module_& m) {
    m.def("megamoe_gin_build_info", []() {
        py::dict result;
#ifdef DG_MEGAMOE_GIN
        result["enabled"] = true;
        result["compiled_nccl_version"] = NCCL_VERSION_CODE;
#else
        result["enabled"] = false;
        result["compiled_nccl_version"] = py::none();
#endif
        result["required_nccl_version"] = kRequiredNcclVersion;
#ifdef DG_MEGAMOE_GIN
        result["nccl_headers_fingerprint"] =
            py::int_(DG_NCCL_HEADERS_FINGERPRINT);
        result["dev_comm_bytes"] = sizeof(ncclDevComm_t);
#else
        result["nccl_headers_fingerprint"] = py::none();
        result["dev_comm_bytes"] = py::none();
#endif
        result["transport_abi_version"] =
            DG_MEGAMOE_GIN_TRANSPORT_ABI_VERSION;
        result["transport_capsule_name"] =
            DG_MEGAMOE_GIN_TRANSPORT_CAPSULE_NAME;
        result["world_size"] = kWorldSize;
        result["lsa_size"] = kLsaSize;
        result["max_tokens_per_rank"] = kMaxTokensPerRank;
        result["minimum_context_count"] = kRequiredContextCount;
        result["minimum_queue_depth"] = kRequiredQueueDepth;
        result["minimum_world_barrier_count"] =
            kRequiredWorldBarrierCount;
        return result;
    });

#ifdef DG_MEGAMOE_GIN
    py::class_<MegaMoeGinTransportHandle,
               std::shared_ptr<MegaMoeGinTransportHandle>>(
        m, "_MegaMoeGinTransportHandle");

    m.def("import_megamoe_gin_transport",
          &MegaMoeGinTransportHandle::import,
          py::arg("capsule"), py::arg("buffer"), py::arg("rank"),
          py::arg("owner"));
    m.def("get_symm_buffer_size_for_mega_moe_gin",
          &get_symm_buffer_size_for_mega_moe_gin);
    m.def("fp8_fp4_mega_moe_gin", &fp8_fp4_mega_moe_gin);
#else
    const auto build_disabled = [](py::args, py::kwargs) -> py::object {
        throw unsupported(
            "build_disabled",
            "rebuild with DG_MEGAMOE_GIN=1 and matching NCCL 2.30.7 Device API headers");
    };
    m.def("import_megamoe_gin_transport",
          build_disabled);
    m.def("get_symm_buffer_size_for_mega_moe_gin", build_disabled);
    m.def("fp8_fp4_mega_moe_gin", build_disabled);
#endif
}

}  // namespace deep_gemm::mega::gin
