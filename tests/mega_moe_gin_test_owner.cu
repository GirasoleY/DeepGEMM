#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_runtime_api.h>
#include <nccl.h>
#include <nccl_device.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <deep_gemm/comm/mega_moe_gin_transport.h>

namespace py = pybind11;

static void check_nccl(ncclResult_t result, const char* operation) {
    if (result != ncclSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + ncclGetErrorString(result));
    }
}

static void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

class TestGinOwner final {
public:
    TestGinOwner(torch::Tensor buffer, py::bytes unique_id, int rank,
                 uint64_t window_base, uint64_t window_bytes,
                 int world_size, int lsa_size)
        : buffer_(std::move(buffer)), rank_(rank),
          world_size_(world_size), lsa_size_(lsa_size),
          window_base_(reinterpret_cast<void*>(window_base)),
          window_bytes_(window_bytes) {
        try {
            initialize(static_cast<std::string>(unique_id));
        } catch (...) {
            abort();
            throw;
        }
    }

    TestGinOwner(const TestGinOwner&) = delete;
    TestGinOwner& operator=(const TestGinOwner&) = delete;
    ~TestGinOwner() { abort(); }

    py::dict facts() const {
        require_live();
        py::dict result;
        result["rank"] = dev_comm_.rank;
        result["world_size"] = dev_comm_.nRanks;
        result["lsa_rank"] = dev_comm_.lsaRank;
        result["lsa_size"] = dev_comm_.lsaSize;
        result["contexts"] = dev_comm_.ginContextCount;
        result["connections"] =
            static_cast<uint32_t>(dev_comm_.ginConnectionCount);
        result["buffer_bytes"] = buffer_.nbytes();
        result["window_bytes"] = window_bytes_;
        result["window_offset"] =
            reinterpret_cast<uintptr_t>(buffer_.data_ptr()) -
            reinterpret_cast<uintptr_t>(window_base_);
        result["cuda_device"] = buffer_.get_device();
        return result;
    }

    py::capsule capsule() {
        require_live();
        return py::capsule(
            &descriptor_, DG_MEGAMOE_GIN_TRANSPORT_CAPSULE_NAME);
    }

    void close() {
        if (comm_ == nullptr)
            return;
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        if (dev_comm_created_) {
            check_nccl(ncclDevCommDestroy(comm_, &dev_comm_),
                       "ncclDevCommDestroy");
            dev_comm_created_ = false;
        }
        check_nccl(ncclCommWindowDeregister(comm_, window_),
                   "ncclCommWindowDeregister");
        window_ = nullptr;
        check_nccl(ncclCommFinalize(comm_), "ncclCommFinalize");
        check_nccl(ncclCommDestroy(comm_), "ncclCommDestroy");
        comm_ = nullptr;
        buffer_ = torch::Tensor();
    }

    void abort() noexcept {
        if (comm_ != nullptr)
            ncclCommAbort(comm_);
        comm_ = nullptr;
        window_ = nullptr;
        dev_comm_created_ = false;
        buffer_ = torch::Tensor();
    }

private:
    void initialize(const std::string& unique_id) {
        if (!buffer_.defined() || !buffer_.is_cuda() ||
            buffer_.scalar_type() != torch::kInt8 ||
            !buffer_.is_contiguous() || buffer_.dim() != 1 ||
            buffer_.nbytes() == 0) {
            throw std::invalid_argument(
                "owner requires a nonempty contiguous CUDA int8 buffer");
        }
        const auto buffer_begin =
            reinterpret_cast<uintptr_t>(buffer_.data_ptr());
        const auto window_begin = reinterpret_cast<uintptr_t>(window_base_);
        if (rank_ < 0 || rank_ >= world_size_ || world_size_ > 72 ||
            lsa_size_ <= 0 || lsa_size_ >= world_size_ ||
            world_size_ % lsa_size_ != 0 || window_begin > buffer_begin ||
            window_begin %
                    NCCL_WIN_REQUIRED_ALIGNMENT != 0) {
            throw std::invalid_argument("invalid rank or window alignment");
        }
        const auto window_offset = buffer_begin - window_begin;
        if (window_offset > window_bytes_ ||
            buffer_.nbytes() > window_bytes_ - window_offset)
            throw std::invalid_argument("buffer is outside its NCCL window");
        if (unique_id.size() != NCCL_UNIQUE_ID_BYTES)
            throw std::invalid_argument("invalid NCCL unique ID size");

        int runtime_version = 0, current_device = -1;
        check_nccl(ncclGetVersion(&runtime_version), "ncclGetVersion");
        check_cuda(cudaGetDevice(&current_device), "cudaGetDevice");
        if (runtime_version != 23007 || current_device != buffer_.get_device())
            throw std::runtime_error("NCCL version or CUDA device mismatch");

        ncclUniqueId id{};
        std::memcpy(id.internal, unique_id.data(), NCCL_UNIQUE_ID_BYTES);
        ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
        config.blocking = 1;
        check_nccl(ncclCommInitRankConfig(&comm_, world_size_, id, rank_, &config),
                   "ncclCommInitRankConfig");

        ncclCommProperties_t properties = NCCL_COMM_PROPERTIES_INITIALIZER;
        check_nccl(ncclCommQueryProperties(comm_, &properties),
                   "ncclCommQueryProperties");
        if (!properties.deviceApiSupport ||
            properties.ginType != NCCL_GIN_TYPE_GDAKI ||
            properties.nLsaTeams != world_size_ / lsa_size_ || properties.rank != rank_ ||
            properties.nRanks != world_size_ ||
            properties.cudaDev != current_device) {
            throw std::runtime_error("NCCL communicator lacks the requested GDAKI topology");
        }

        check_nccl(ncclCommWindowRegister(
                       comm_, window_base_, window_bytes_, &window_,
                       NCCL_WIN_COLL_SYMMETRIC | NCCL_WIN_STRICT_ORDERING),
                   "ncclCommWindowRegister");
        ncclDevCommRequirements_t requirements =
            NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        requirements.ginForceEnable = true;
        requirements.ginContextCount = 6;
        requirements.ginSignalCount = 0;
        requirements.ginCounterCount = 0;
        requirements.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
        requirements.ginExclusiveContexts = true;
        requirements.ginQueueDepth = 64;
        requirements.worldGinBarrierCount = 2;
        check_nccl(ncclDevCommCreate(comm_, &requirements, &dev_comm_),
                   "ncclDevCommCreate");
        dev_comm_created_ = true;

        if (dev_comm_.rank != rank_ || dev_comm_.nRanks != world_size_ ||
            dev_comm_.lsaRank != rank_ % lsa_size_ || dev_comm_.lsaSize != lsa_size_ ||
            dev_comm_.ginContextCount < 6 ||
            dev_comm_.ginConnectionCount == 0 ||
            dev_comm_.ginConnectionsRailed || dev_comm_.ginContextsRailed) {
            throw std::runtime_error("unexpected NCCL device communicator");
        }

        descriptor_.cuda_device = current_device;
        descriptor_.queue_depth = 64;
        descriptor_.world_barrier_count = 2;
        descriptor_.dev_comm = dev_comm_;
        descriptor_.window = window_;
        descriptor_.window_base = window_base_;
        descriptor_.window_bytes = window_bytes_;
    }

    void require_live() const {
        if (comm_ == nullptr)
            throw std::runtime_error("test GIN owner is closed");
    }

    torch::Tensor buffer_;
    int rank_ = -1;
    int world_size_;
    int lsa_size_;
    void* window_base_ = nullptr;
    uint64_t window_bytes_ = 0;
    ncclComm_t comm_ = nullptr;
    ncclWindow_t window_ = nullptr;
    ncclDevComm_t dev_comm_{};
    bool dev_comm_created_ = false;
    DGMegaMoEGinTransportV1 descriptor_{};
};

static py::bytes get_unique_id() {
    ncclUniqueId id{};
    check_nccl(ncclGetUniqueId(&id), "ncclGetUniqueId");
    return py::bytes(id.internal, NCCL_UNIQUE_ID_BYTES);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("get_unique_id", &get_unique_id);
    py::class_<TestGinOwner>(module, "TestGinOwner")
        .def(py::init<torch::Tensor, py::bytes, int, uint64_t, uint64_t, int, int>(),
             py::arg("buffer"), py::arg("unique_id"), py::arg("rank"),
             py::arg("window_base"), py::arg("window_bytes"),
             py::arg("world_size") = 8, py::arg("lsa_size") = 4)
        .def("facts", &TestGinOwner::facts)
        .def("capsule", &TestGinOwner::capsule)
        .def("close", &TestGinOwner::close)
        .def("abort", &TestGinOwner::abort);
}
