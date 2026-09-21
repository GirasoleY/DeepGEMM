#pragma once

#include <cstdint>
#include <type_traits>

#define DG_MEGAMOE_GIN_TRANSPORT_CAPSULE_NAME \
    "deep_gemm.mega_moe_gin_transport.v1"
#define DG_MEGAMOE_GIN_TRANSPORT_ABI_VERSION 1u
#define DG_MEGAMOE_GIN_REQUIRED_NCCL_VERSION_CODE 23007u

#ifdef DG_MEGAMOE_GIN

#include <nccl_device.h>

#if !defined(NCCL_VERSION_CODE) || \
    NCCL_VERSION_CODE != DG_MEGAMOE_GIN_REQUIRED_NCCL_VERSION_CODE
#error "MegaMoE GIN requires exactly the validated NCCL 2.30.7 Device API ABI"
#endif
#ifndef DG_NCCL_HEADERS_FINGERPRINT
#error "MegaMoE GIN requires the validated NCCL Device API header fingerprint"
#endif

// Caller-owned descriptor passed through a PyCapsule named
// DG_MEGAMOE_GIN_TRANSPORT_CAPSULE_NAME. The capsule points directly to this
// structure. DeepGEMM copies it while importing the capsule and never destroys
// any referenced NCCL resource. The caller provisions full, unrailed,
// exclusive GDAKI contexts with strong VA signals; registers a strict-order
// symmetric window; collectively validates equal contiguous LSA teams; and
// keeps every resource alive until all launches and captured graphs complete.
struct DGMegaMoEGinTransportV1 final {
    uint32_t abi_version = DG_MEGAMOE_GIN_TRANSPORT_ABI_VERSION;
    uint32_t struct_bytes = sizeof(DGMegaMoEGinTransportV1);
    uint32_t nccl_version_code = NCCL_VERSION_CODE;
    uint32_t dev_comm_bytes = sizeof(ncclDevComm_t);
    uint64_t nccl_headers_fingerprint = DG_NCCL_HEADERS_FINGERPRINT;

    int32_t cuda_device = -1;
    uint32_t queue_depth = 0;
    uint32_t world_barrier_count = 0;

    ncclDevComm_t dev_comm{};
    ncclWindow_t window = nullptr;
    const void* window_base = nullptr;
    uint64_t window_bytes = 0;
};

static_assert(std::is_standard_layout_v<DGMegaMoEGinTransportV1>);
static_assert(std::is_trivially_copyable_v<DGMegaMoEGinTransportV1>);

#endif  // DG_MEGAMOE_GIN
