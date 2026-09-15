#pragma once

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cutlass/version.h>
#include <deep_jit/backend/cuda/backend.hpp>

namespace deep_gemm {

#define DG_STRINGIFY_IMPL(value) #value
#define DG_STRINGIFY(value) DG_STRINGIFY_IMPL(value)

#ifdef DG_MEGAMOE_GIN
#if DG_MEGAMOE_GIN != 1
#error "DG_MEGAMOE_GIN must be defined as 1"
#endif
#ifndef DG_NCCL_HEADERS_FINGERPRINT
#error "MegaMoE GIN requires an NCCL Device API header fingerprint"
#endif
#ifndef DG_NCCL_BUILD_INCLUDE_DIR
#error "MegaMoE GIN requires an NCCL Device API build include directory"
#endif
#endif

inline deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>> jit(nullptr);

inline void init_jit(const std::string& library_root_path) {
    const auto library_root = std::filesystem::absolute(library_root_path);
    const auto include_dir = library_root / "include";
    auto extra_signature = "cutlass-" + std::to_string(CUTLASS_VERSION);
    std::vector<std::filesystem::path> include_dirs{include_dir};
#ifdef DG_MEGAMOE_GIN
    auto nccl_include_dir = include_dir / "nccl";
    if (not std::filesystem::is_regular_file(nccl_include_dir / "nccl_device.h"))
        nccl_include_dir = DG_NCCL_BUILD_INCLUDE_DIR;
    if (not std::filesystem::is_regular_file(nccl_include_dir / "nccl_device.h"))
        throw std::runtime_error("NCCL Device API headers are unavailable to the JIT");
    include_dirs.emplace_back(std::move(nccl_include_dir));
    extra_signature += "-nccl-" DG_STRINGIFY(DG_NCCL_HEADERS_FINGERPRINT);
#endif
    const auto config = deep_jit::Config(
        library_root,
        "DG",
        std::move(extra_signature),
        std::move(include_dirs),
        // NCCL uses quoted internal includes, which the current parser does
        // not fingerprint. The build-time Device API tree digest above does.
        {"deep_gemm/"});

    jit = deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>>([config] {
        auto runtime = std::make_shared<deep_jit::Runtime<deep_jit::CUDA>>(config);
        runtime->default_compiler_options.nvcc_flags->
            emplace_back("--diag-suppress=39,161,174,177,186,940");
        runtime->default_compiler_options.nvcc_flags->
            emplace_back("--compiler-options=-Wno-deprecated-declarations,-Wno-abi");
#ifdef DG_MEGAMOE_GIN
        runtime->default_compiler_options.extra_nvcc_flags.emplace_back(
            "-DDG_MEGAMOE_GIN=1");
        runtime->default_compiler_options.extra_nvcc_flags.emplace_back(
            "-DDG_NCCL_HEADERS_FINGERPRINT="
            DG_STRINGIFY(DG_NCCL_HEADERS_FINGERPRINT));
#endif
        return runtime;
    });
}

#undef DG_STRINGIFY
#undef DG_STRINGIFY_IMPL

}  // namespace deep_gemm
