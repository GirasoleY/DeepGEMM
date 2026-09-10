#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

#include <pybind11/pybind11.h>
#include <torch/python.h>

#ifdef DG_MEGAMOE_GIN
#include <deep_gemm/comm/mega_moe_gin.cuh>
#endif

namespace deep_gemm::mega::gin {

namespace py = pybind11;

static constexpr int kRequiredNcclVersion = 23007;
static constexpr int kDefaultContextCount = 9;
static constexpr int kDefaultQueueDepth = 64;
static constexpr int kDefaultWorldBarrierCount = 4;
static constexpr int kDefaultExpectedLsaSize = 8;
static constexpr int kRequiredSignalCount = 2;

static std::runtime_error unsupported(const std::string& reason, const std::string& detail) {
    return std::runtime_error(
        "MegaMoE GIN unsupported [reason=" + reason + "]: " + detail);
}

#ifdef DG_MEGAMOE_GIN

static_assert(NCCL_VERSION_CODE == kRequiredNcclVersion,
              "DG_MEGAMOE_GIN prototype requires exactly NCCL 2.30.7");
#ifdef DG_NCCL_VERSION_CODE
static_assert(NCCL_VERSION_CODE == DG_NCCL_VERSION_CODE,
              "setup.py NCCL version does not match the included NCCL headers");
#endif

static void check_nccl(const ncclResult_t result, const char* operation) {
    if (result != ncclSuccess) {
        std::ostringstream message;
        message << operation << " failed with " << ncclGetErrorString(result)
                << " (ncclResult_t=" << static_cast<int>(result) << ")";
        throw std::runtime_error("MegaMoE GIN lifecycle error: " + message.str());
    }
}

static void check_cuda(const cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        std::ostringstream message;
        message << operation << " failed with " << cudaGetErrorString(result)
                << " (cudaError_t=" << static_cast<int>(result) << ")";
        throw std::runtime_error("MegaMoE GIN lifecycle error: " + message.str());
    }
}

static const char* gin_type_name(const ncclGinType_t gin_type) {
    switch (gin_type) {
        case NCCL_GIN_TYPE_NONE: return "none";
        case NCCL_GIN_TYPE_PROXY: return "proxy";
        case NCCL_GIN_TYPE_GDAKI: return "gdaki";
        case NCCL_GIN_TYPE_GPI: return "gpi";
        default: return "unknown";
    }
}

static ncclGinType_t parse_required_gin_type(const std::string& value) {
    if (value == "gdaki")
        return NCCL_GIN_TYPE_GDAKI;
    if (value == "proxy")
        return NCCL_GIN_TYPE_PROXY;
    if (value == "gpi")
        return NCCL_GIN_TYPE_GPI;
    if (value == "any")
        return NCCL_GIN_TYPE_NONE;
    throw std::invalid_argument(
        "required_gin_type must be one of: gdaki, proxy, gpi, any");
}

class MegaMoeGinContext;

static std::mutex& context_registry_mutex() {
    static std::mutex mutex;
    return mutex;
}

static std::unordered_map<const void*, std::weak_ptr<MegaMoeGinContext>>&
context_registry() {
    static std::unordered_map<const void*, std::weak_ptr<MegaMoeGinContext>> registry;
    return registry;
}

static std::shared_ptr<MegaMoeGinContext> find_megamoe_gin_context(
    const void* buffer_data_ptr) {
    if (buffer_data_ptr == nullptr)
        return nullptr;
    std::lock_guard<std::mutex> lock(context_registry_mutex());
    auto& registry = context_registry();
    const auto iterator = registry.find(buffer_data_ptr);
    if (iterator == registry.end())
        return nullptr;
    auto context = iterator->second.lock();
    if (context == nullptr)
        registry.erase(iterator);
    return context;
}

static void register_megamoe_gin_context(
    const void* buffer_data_ptr,
    const std::shared_ptr<MegaMoeGinContext>& context) {
    std::lock_guard<std::mutex> lock(context_registry_mutex());
    auto& slot = context_registry()[buffer_data_ptr];
    if (auto existing = slot.lock(); existing != nullptr)
        throw std::runtime_error(
            "MegaMoE GIN context already exists for this symmetric buffer");
    slot = context;
}

static void unregister_megamoe_gin_context(
    const void* buffer_data_ptr,
    const MegaMoeGinContext* expected_context) {
    if (buffer_data_ptr == nullptr)
        return;
    std::lock_guard<std::mutex> lock(context_registry_mutex());
    auto& registry = context_registry();
    const auto iterator = registry.find(buffer_data_ptr);
    if (iterator == registry.end())
        return;
    const auto existing = iterator->second.lock();
    if (existing == nullptr || existing.get() == expected_context)
        registry.erase(iterator);
}

class CudaCollectiveScratch final {
public:
    CudaCollectiveScratch(const size_t send_bytes, const size_t recv_bytes) {
        check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags(capability validation)");
        try {
            check_cuda(cudaMalloc(&send_, send_bytes),
                       "cudaMalloc(capability validation send)");
            check_cuda(cudaMalloc(&recv_, recv_bytes),
                       "cudaMalloc(capability validation receive)");
        } catch (...) {
            cleanup();
            throw;
        }
    }

    CudaCollectiveScratch(const CudaCollectiveScratch&) = delete;
    CudaCollectiveScratch& operator=(const CudaCollectiveScratch&) = delete;

    ~CudaCollectiveScratch() { cleanup(); }

    void* send() const { return send_; }
    void* recv() const { return recv_; }
    cudaStream_t stream() const { return stream_; }

private:
    void cleanup() noexcept {
        if (recv_ != nullptr)
            cudaFree(recv_);
        if (send_ != nullptr)
            cudaFree(send_);
        if (stream_ != nullptr)
            cudaStreamDestroy(stream_);
        recv_ = nullptr;
        send_ = nullptr;
        stream_ = nullptr;
    }

    void* send_ = nullptr;
    void* recv_ = nullptr;
    cudaStream_t stream_ = nullptr;
};

class MegaMoeGinContext final {
public:
    static std::shared_ptr<MegaMoeGinContext> create(
        const torch::Tensor& buffer,
        const std::string& unique_id_bytes,
        const int rank,
        const int world_size,
        const int context_count,
        const int queue_depth,
        const int world_barrier_count,
        const int expected_lsa_size,
        const std::string& required_gin_type,
        const uint32_t completion_batch,
        const uint32_t combine_chunk_bytes,
        const uint32_t outbox_depth,
        const uint32_t combine_issue_wave,
        const bool active_fast_path,
        const bool bulk_combine,
        const bool direct_dispatch) {
        auto context = std::shared_ptr<MegaMoeGinContext>(new MegaMoeGinContext(
            buffer, unique_id_bytes, rank, world_size, context_count,
            queue_depth, world_barrier_count, expected_lsa_size,
            required_gin_type, completion_batch, combine_chunk_bytes,
            outbox_depth, combine_issue_wave, active_fast_path,
            bulk_combine, direct_dispatch));
        context->initialize();
        register_megamoe_gin_context(context->buffer_data_ptr(), context);
        return context;
    }

    MegaMoeGinContext(const MegaMoeGinContext&) = delete;
    MegaMoeGinContext& operator=(const MegaMoeGinContext&) = delete;

    ~MegaMoeGinContext() {
        // Destructors cannot safely enter collectives. Explicit destroy() is
        // mandatory; abort is only an emergency, rank-local leak-prevention path.
        unregister_megamoe_gin_context(buffer_data_ptr(), this);
        abort();
    }

    void destroy() {
        if (comm_ == nullptr)
            return;

        // The caller collectively quiesces the owning process group first.
        // Synchronize again here so direct C++ users cannot release handles while
        // a kernel still owns a copied launch descriptor.
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(teardown)");

        ncclResult_t first_error = ncclSuccess;
        const char* first_operation = nullptr;
        const auto remember = [&](const ncclResult_t result, const char* operation) {
            if (first_error == ncclSuccess && result != ncclSuccess) {
                first_error = result;
                first_operation = operation;
            }
        };

        if (dev_comm_created_) {
            remember(ncclDevCommDestroy(comm_, &dev_comm_), "ncclDevCommDestroy");
            dev_comm_created_ = false;
        }
        if (window_ != nullptr) {
            remember(ncclCommWindowDeregister(comm_, window_),
                     "ncclCommWindowDeregister");
            window_ = nullptr;
        }
        remember(ncclCommFinalize(comm_), "ncclCommFinalize");
        remember(ncclCommDestroy(comm_), "ncclCommDestroy");
        comm_ = nullptr;
        buffer_ = torch::Tensor();

        if (first_error != ncclSuccess)
            check_nccl(first_error, first_operation);
    }

    void abort() noexcept {
        // Emergency rank-local teardown for a peer failure. Never synchronize
        // or enter a collective here: the other ranks may already be unwinding
        // a different failure. NCCL communicator abort owns best-effort cleanup
        // of the registered window and device communicator in this path.
        if (comm_ != nullptr)
            ncclCommAbort(comm_);
        comm_ = nullptr;
        window_ = nullptr;
        dev_comm_created_ = false;
        buffer_ = torch::Tensor();
    }

    void release_buffer_registration() noexcept {
        // destroy()/abort() intentionally leave an inactive registry entry so
        // a caller retaining the symmetric allocation cannot silently launch
        // through the legacy NVLink path. The owning SymmBuffer calls this only
        // when it also releases that allocation, making pointer reuse safe.
        unregister_megamoe_gin_context(buffer_data_ptr(), this);
    }

    bool active() const { return comm_ != nullptr; }
    int rank() const { return rank_; }
    int world_size() const { return world_size_; }
    int lsa_rank() const { return lsa_rank_; }
    int lsa_size() const { return lsa_size_; }
    int context_count() const { return actual_context_count_; }
    int requested_context_count() const { return context_count_; }
    int connection_count() const { return connection_count_; }
    int requested_signal_count() const { return kRequiredSignalCount; }
    int signal_count() const { return actual_signal_count_; }
    bool active_fast_path() const { return active_fast_path_; }
    bool bulk_combine() const { return bulk_combine_; }
    bool direct_dispatch() const { return direct_dispatch_; }
    int queue_depth() const { return queue_depth_; }
    int world_barrier_count() const { return world_barrier_count_; }
    int gin_type() const { return static_cast<int>(gin_type_); }
    std::string gin_type_string() const { return gin_type_name(gin_type_); }
    int64_t buffer_bytes() const { return buffer_bytes_; }
    const void* buffer_data_ptr() const { return buffer_data_ptr_; }

    const ncclDevComm_t& dev_comm() const {
        if (!active())
            throw std::runtime_error("MegaMoE GIN context has already been destroyed");
        return dev_comm_;
    }

    ncclWindow_t window() const {
        if (!active())
            throw std::runtime_error("MegaMoE GIN context has already been destroyed");
        return window_;
    }

    comm::MegaMoeGinTransport launch_descriptor() const {
        return comm::MegaMoeGinTransport{
            dev_comm(), window(), 1, completion_batch_,
            combine_chunk_bytes_, outbox_depth_, combine_issue_wave_,
            active_fast_path_ ? 1u : 0u,
            bulk_combine_ ? 1u : 0u,
            direct_dispatch_ ? 1u : 0u,
            diagnostic_buffer_.defined() ?
                reinterpret_cast<uint64_t*>(diagnostic_buffer_.data_ptr<int64_t>()) : nullptr,
            diagnostic_buffer_.defined() ?
                static_cast<uint32_t>(diagnostic_buffer_.size(0)) : 0u};
    }

    // Attach once, before graph capture. Owning the tensor here keeps captured
    // launch pointers valid for the context lifetime; replacement is forbidden.
    void set_diagnostic_buffer(const torch::Tensor& diagnostic_buffer) {
        if (not active())
            throw std::runtime_error("Cannot attach diagnostics to an inactive GIN context");
        if (diagnostic_buffer_.defined())
            throw std::runtime_error("GIN diagnostic buffer is immutable once attached");
        if (not diagnostic_buffer.is_cuda() or
            diagnostic_buffer.device() != buffer_.device() or
            diagnostic_buffer.scalar_type() != torch::kInt64 or
            not diagnostic_buffer.is_contiguous() or
            diagnostic_buffer.dim() != 2 or diagnostic_buffer.size(0) <= 0 or
            diagnostic_buffer.size(1) != comm::kMegaMoeGinDiagnosticColumns)
            throw std::invalid_argument(
                "GIN diagnostic buffer must be contiguous CUDA int64 [num_sms,240] on the buffer device");
        diagnostic_buffer_ = diagnostic_buffer;
    }

    comm::MegaMoeGinTransport launch_descriptor(
        const uint32_t completion_batch,
        const uint32_t combine_chunk_bytes,
        const uint32_t outbox_depth,
        const uint32_t combine_issue_wave) const {
        validate_launch_tuning(
            completion_batch, combine_chunk_bytes, outbox_depth,
            combine_issue_wave);
        if (completion_batch != completion_batch_ ||
            combine_chunk_bytes != combine_chunk_bytes_ ||
            outbox_depth != outbox_depth_ ||
            combine_issue_wave != combine_issue_wave_)
            throw std::invalid_argument(
                "MegaMoE GIN launch tuning differs from the registered context/layout tuning");
        return launch_descriptor();
    }

    py::dict launch_descriptor_snapshot() const {
        const auto descriptor = launch_descriptor();
        py::dict result;
        result["dev_comm"] = py::bytes(
            reinterpret_cast<const char*>(&descriptor.dev_comm),
            sizeof(descriptor.dev_comm));
        result["dev_comm_bytes"] = py::int_(sizeof(descriptor.dev_comm));
        result["window"] = py::int_(
            reinterpret_cast<uintptr_t>(descriptor.window));
        result["rank"] = rank_;
        result["world_size"] = world_size_;
        result["lsa_rank"] = lsa_rank_;
        result["lsa_size"] = lsa_size_;
        result["context_count"] = actual_context_count_;
        result["requested_context_count"] = context_count_;
        result["connection_count"] = connection_count_;
        result["requested_signal_count"] = kRequiredSignalCount;
        result["signal_count"] = actual_signal_count_;
        result["queue_depth"] = queue_depth_;
        result["world_barrier_count"] = world_barrier_count_;
        result["enabled"] = descriptor.enabled;
        result["completion_batch"] = descriptor.completion_batch;
        result["combine_chunk_bytes"] = descriptor.combine_chunk_bytes;
        result["outbox_depth"] = descriptor.outbox_depth;
        result["combine_issue_wave"] = descriptor.combine_issue_wave;
        result["active_fast_path"] = descriptor.active_fast_path != 0;
        result["bulk_combine"] = descriptor.bulk_combine != 0;
        result["direct_dispatch"] = descriptor.direct_dispatch != 0;
        return result;
    }

private:
    static constexpr int kCapabilityFields = 16;

    MegaMoeGinContext(
        torch::Tensor buffer,
        std::string unique_id_bytes,
        const int rank,
        const int world_size,
        const int context_count,
        const int queue_depth,
        const int world_barrier_count,
        const int expected_lsa_size,
        const std::string& required_gin_type,
        const uint32_t completion_batch,
        const uint32_t combine_chunk_bytes,
        const uint32_t outbox_depth,
        const uint32_t combine_issue_wave,
        const bool active_fast_path,
        const bool bulk_combine,
        const bool direct_dispatch)
        : buffer_(std::move(buffer)), unique_id_bytes_(std::move(unique_id_bytes)),
          rank_(rank), world_size_(world_size), context_count_(context_count),
          queue_depth_(queue_depth), world_barrier_count_(world_barrier_count),
          expected_lsa_size_(expected_lsa_size),
          required_gin_type_(parse_required_gin_type(required_gin_type)),
          completion_batch_(completion_batch),
          combine_chunk_bytes_(combine_chunk_bytes),
          outbox_depth_(outbox_depth),
          combine_issue_wave_(combine_issue_wave),
          active_fast_path_(active_fast_path),
          bulk_combine_(bulk_combine),
          direct_dispatch_(direct_dispatch) {
        validate_arguments();
        buffer_data_ptr_ = buffer_.data_ptr();
        buffer_bytes_ = buffer_.nbytes();
    }

    void validate_arguments() const {
        if (!buffer_.defined() || !buffer_.is_cuda() || !buffer_.is_contiguous())
            throw std::invalid_argument(
                "MegaMoE GIN buffer must be a defined, contiguous CUDA tensor");
        if (buffer_.nbytes() == 0)
            throw std::invalid_argument("MegaMoE GIN buffer must not be empty");
        if (reinterpret_cast<uintptr_t>(buffer_.data_ptr()) % NCCL_WIN_REQUIRED_ALIGNMENT != 0) {
            std::ostringstream message;
            message << "symmetric buffer pointer " << buffer_.data_ptr()
                    << " is not aligned to NCCL_WIN_REQUIRED_ALIGNMENT="
                    << NCCL_WIN_REQUIRED_ALIGNMENT;
            throw unsupported("window_alignment", message.str());
        }
        if (unique_id_bytes_.size() != NCCL_UNIQUE_ID_BYTES)
            throw std::invalid_argument(
                "MegaMoE GIN NCCL unique ID must contain exactly " +
                std::to_string(NCCL_UNIQUE_ID_BYTES) + " bytes");
        if ((world_size_ != 8 && world_size_ != 16) ||
            rank_ < 0 || rank_ >= world_size_)
            throw std::invalid_argument(
                "MegaMoE GIN requires world_size == 8 or 16 and 0 <= rank < world_size");
        if (context_count_ < kDefaultContextCount)
            throw std::invalid_argument(
                "MegaMoE GIN requires at least 9 contexts (control + 8 data)");
        if (queue_depth_ < 64)
            throw std::invalid_argument(
                "MegaMoE GIN queue_depth must be at least 64");
        if (world_barrier_count_ < kDefaultWorldBarrierCount)
            throw std::invalid_argument(
                "MegaMoE GIN requires at least 4 world/hybrid barrier slots");
        if (expected_lsa_size_ != world_size_ / 2)
            throw std::invalid_argument(
                "MegaMoE GIN requires expected_lsa_size == world_size / 2 (4 or 8)");
        validate_launch_tuning(
            completion_batch_, combine_chunk_bytes_, outbox_depth_,
            combine_issue_wave_);
        if (bulk_combine_ && !active_fast_path_)
            throw std::invalid_argument(
                "MegaMoE GIN bulk_combine requires active_fast_path for "
                "world-uniform runtime eligibility consensus");
        if (direct_dispatch_ && !active_fast_path_)
            throw std::invalid_argument(
                "MegaMoE GIN direct_dispatch requires active_fast_path for "
                "world-uniform runtime eligibility consensus");

        int current_device = -1;
        check_cuda(cudaGetDevice(&current_device), "cudaGetDevice");
        if (current_device != buffer_.get_device()) {
            std::ostringstream message;
            message << "current CUDA device " << current_device
                    << " does not own the symmetric buffer on device "
                    << buffer_.get_device();
            throw std::invalid_argument(message.str());
        }
    }

    static void validate_launch_tuning(
        const uint32_t completion_batch,
        const uint32_t combine_chunk_bytes,
        const uint32_t outbox_depth,
        const uint32_t combine_issue_wave) {
        if (completion_batch != 1 && completion_batch != 2 &&
            completion_batch != 4 && completion_batch != 8)
            throw std::invalid_argument(
                "MegaMoE GIN completion_batch must be one of 1, 2, 4, 8");
        if (combine_chunk_bytes != 256 && combine_chunk_bytes != 1792 &&
            combine_chunk_bytes != 3584 && combine_chunk_bytes != 7168)
            throw std::invalid_argument(
                "MegaMoE GIN combine_chunk_bytes must be one of 256, 1792, 3584, 7168");
        if (outbox_depth != 4 && outbox_depth != 8 &&
            outbox_depth != 16 && outbox_depth != 64)
            throw std::invalid_argument(
                "MegaMoE GIN outbox_depth must be one of 4, 8, 16, 64");
        if (combine_issue_wave != 1 && combine_issue_wave != 2 &&
            combine_issue_wave != 4 && combine_issue_wave != 8)
            throw std::invalid_argument(
                "MegaMoE GIN combine_issue_wave must be one of 1, 2, 4, 8");
    }

    void initialize() {
        try {
            int runtime_version = 0;
            check_nccl(ncclGetVersion(&runtime_version), "ncclGetVersion");
            if (runtime_version != NCCL_VERSION_CODE) {
                std::ostringstream message;
                message << "runtime NCCL " << runtime_version
                        << " does not exactly match compiled headers "
                        << NCCL_VERSION_CODE
                        << "; ncclDevComm ABI mixing is unsafe";
                throw unsupported("runtime_header_version_mismatch", message.str());
            }

            ncclUniqueId id{};
            std::memcpy(id.internal, unique_id_bytes_.data(), NCCL_UNIQUE_ID_BYTES);
            ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
            config.blocking = 1;
            check_nccl(ncclCommInitRankConfig(
                           &comm_, world_size_, id, rank_, &config),
                       "ncclCommInitRankConfig");

            ncclCommProperties_t properties = NCCL_COMM_PROPERTIES_INITIALIZER;
            check_nccl(ncclCommQueryProperties(comm_, &properties),
                       "ncclCommQueryProperties");

            int comm_rank = -1, comm_size = -1, comm_device = -1;
            check_nccl(ncclCommUserRank(comm_, &comm_rank), "ncclCommUserRank");
            check_nccl(ncclCommCount(comm_, &comm_size), "ncclCommCount");
            check_nccl(ncclCommCuDevice(comm_, &comm_device), "ncclCommCuDevice");

            std::array<int, kCapabilityFields> local = {
                runtime_version,
                properties.deviceApiSupport ? 1 : 0,
                static_cast<int>(properties.ginType),
                properties.nLsaTeams,
                properties.rank,
                properties.nRanks,
                properties.cudaDev,
                comm_rank,
                comm_size,
                comm_device,
                static_cast<int>(buffer_.nbytes() & 0x7fffffff),
                static_cast<int>((static_cast<uint64_t>(buffer_.nbytes()) >> 31) & 0x7fffffff),
                kRequiredSignalCount,
                active_fast_path_ ? 1 : 0,
                bulk_combine_ ? 1 : 0,
                direct_dispatch_ ? 1 : 0,
            };
            const auto gathered = allgather(local);
            validate_capabilities(gathered);

            check_nccl(ncclCommWindowRegister(
                           comm_, buffer_.data_ptr(), buffer_.nbytes(), &window_,
                           NCCL_WIN_COLL_SYMMETRIC | NCCL_WIN_STRICT_ORDERING),
                       "ncclCommWindowRegister");

            ncclDevCommRequirements_t requirements =
                NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
            requirements.ginForceEnable = true;
            requirements.ginContextCount = context_count_;
            requirements.ginSignalCount = kRequiredSignalCount;
            requirements.ginCounterCount = 0;
            requirements.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
            requirements.ginExclusiveContexts = true;
            requirements.ginQueueDepth = queue_depth_;
            requirements.worldGinBarrierCount = world_barrier_count_;
            check_nccl(ncclDevCommCreate(comm_, &requirements, &dev_comm_),
                       "ncclDevCommCreate");
            dev_comm_created_ = true;

            lsa_rank_ = dev_comm_.lsaRank;
            lsa_size_ = dev_comm_.lsaSize;
            actual_context_count_ = static_cast<int>(dev_comm_.ginContextCount);
            connection_count_ = static_cast<int>(dev_comm_.ginConnectionCount);
            actual_signal_count_ = static_cast<int>(dev_comm_.ginSignalCount);

            std::array<int, kCapabilityFields> resources = {
                dev_comm_.rank,
                dev_comm_.nRanks,
                lsa_rank_,
                lsa_size_,
                actual_context_count_,
                connection_count_,
                dev_comm_.ginConnectionsRailed ? 1 : 0,
                dev_comm_.ginContextsRailed ? 1 : 0,
                rank_,
                world_size_,
                context_count_,
                world_barrier_count_,
                actual_signal_count_,
                active_fast_path_ ? 1 : 0,
                bulk_combine_ ? 1 : 0,
                direct_dispatch_ ? 1 : 0,
            };
            validate_resources(allgather(resources));
            gin_type_ = properties.ginType;
        } catch (...) {
            cleanup_after_failed_initialize();
            throw;
        }
    }

    std::vector<int> allgather(
        const std::array<int, kCapabilityFields>& local) const {
        CudaCollectiveScratch scratch(
            local.size() * sizeof(int),
            static_cast<size_t>(world_size_) * local.size() * sizeof(int));
        check_cuda(cudaMemcpyAsync(
                       scratch.send(), local.data(), local.size() * sizeof(int),
                       cudaMemcpyHostToDevice, scratch.stream()),
                   "cudaMemcpyAsync(capability validation input)");
        check_nccl(ncclAllGather(
                       scratch.send(), scratch.recv(), local.size(), ncclInt,
                       comm_, scratch.stream()),
                   "ncclAllGather(capability validation)");
        std::vector<int> result(static_cast<size_t>(world_size_) * local.size());
        check_cuda(cudaMemcpyAsync(
                       result.data(), scratch.recv(), result.size() * sizeof(int),
                       cudaMemcpyDeviceToHost, scratch.stream()),
                   "cudaMemcpyAsync(capability validation output)");
        check_cuda(cudaStreamSynchronize(scratch.stream()),
                   "cudaStreamSynchronize(capability validation)");
        return result;
    }

    void validate_capabilities(const std::vector<int>& gathered) const {
        const int expected_lsa_teams = world_size_ / expected_lsa_size_;
        for (int peer = 0; peer < world_size_; ++peer) {
            const int* fields = gathered.data() + peer * kCapabilityFields;
            const uint64_t buffer_bytes =
                static_cast<uint32_t>(fields[10]) |
                (static_cast<uint64_t>(static_cast<uint32_t>(fields[11])) << 31);
            if (fields[0] != NCCL_VERSION_CODE)
                throw unsupported("runtime_header_version_mismatch",
                                  "rank " + std::to_string(peer) +
                                  " reports NCCL " + std::to_string(fields[0]));
            if (fields[1] == 0)
                throw unsupported("device_api_unavailable",
                                  "rank " + std::to_string(peer) +
                                  " reports deviceApiSupport=false");
            const auto peer_gin_type = static_cast<ncclGinType_t>(fields[2]);
            if (peer_gin_type == NCCL_GIN_TYPE_NONE)
                throw unsupported("gin_type_none",
                                  "rank " + std::to_string(peer) +
                                  " has no GIN-capable network device");
            if (required_gin_type_ != NCCL_GIN_TYPE_NONE &&
                peer_gin_type != required_gin_type_) {
                throw unsupported(
                    "gin_type_mismatch",
                    "rank " + std::to_string(peer) + " reports " +
                    gin_type_name(peer_gin_type) + ", required " +
                    gin_type_name(required_gin_type_));
            }
            if (fields[3] != expected_lsa_teams) {
                throw unsupported(
                    "topology_lsa_team_count",
                    "rank " + std::to_string(peer) + " reports " +
                    std::to_string(fields[3]) + " LSA teams, expected " +
                    std::to_string(expected_lsa_teams));
            }
            if (fields[4] != peer || fields[5] != world_size_ ||
                fields[7] != peer || fields[8] != world_size_) {
                throw unsupported(
                    "rank_mapping_mismatch",
                    "auxiliary NCCL rank ordering does not match the owning process group at rank " +
                    std::to_string(peer));
            }
            if (fields[6] != fields[9])
                throw unsupported("cuda_device_mismatch",
                                  "NCCL property and communicator devices differ at rank " +
                                  std::to_string(peer));
            if (buffer_bytes != static_cast<uint64_t>(buffer_.nbytes()))
                throw unsupported(
                    "asymmetric_window_size",
                    "rank " + std::to_string(peer) + " registered " +
                    std::to_string(buffer_bytes) + " bytes, local rank registered " +
                    std::to_string(buffer_.nbytes()));
            if (fields[12] != kRequiredSignalCount)
                throw unsupported(
                    "gin_signal_count_request_mismatch",
                    "rank " + std::to_string(peer) + " requested " +
                    std::to_string(fields[12]) +
                    " indexed GIN signals; expected " +
                    std::to_string(kRequiredSignalCount));
            if (fields[13] != static_cast<int>(active_fast_path_))
                throw unsupported(
                    "active_fast_path_mismatch",
                    "rank " + std::to_string(peer) + " selected active_fast_path=" +
                    std::to_string(fields[13]) + ", local rank selected " +
                    std::to_string(static_cast<int>(active_fast_path_)));
            if (fields[14] != static_cast<int>(bulk_combine_))
                throw unsupported(
                    "bulk_combine_mismatch",
                    "rank " + std::to_string(peer) + " selected bulk_combine=" +
                    std::to_string(fields[14]) + ", local rank selected " +
                    std::to_string(static_cast<int>(bulk_combine_)));
            if (fields[15] != static_cast<int>(direct_dispatch_))
                throw unsupported(
                    "direct_dispatch_mismatch",
                    "rank " + std::to_string(peer) + " selected direct_dispatch=" +
                    std::to_string(fields[15]) + ", local rank selected " +
                    std::to_string(static_cast<int>(direct_dispatch_)));
        }
    }

    void validate_resources(const std::vector<int>& gathered) const {
        for (int peer = 0; peer < world_size_; ++peer) {
            const int* fields = gathered.data() + peer * kCapabilityFields;
            if (fields[0] != peer || fields[1] != world_size_ ||
                fields[8] != peer || fields[9] != world_size_)
                throw unsupported("device_comm_rank_mapping",
                                  "ncclDevComm rank metadata differs at rank " +
                                  std::to_string(peer));
            if (fields[2] != peer % expected_lsa_size_ ||
                fields[3] != expected_lsa_size_)
                throw unsupported(
                    "topology_lsa_layout",
                    "rank " + std::to_string(peer) + " reports lsaRank=" +
                    std::to_string(fields[2]) + ", lsaSize=" +
                    std::to_string(fields[3]) + "; expected contiguous 2x" +
                    std::to_string(expected_lsa_size_) + " ordering");
            // ncclGinAllContexts iterates dev_comm.ginContextCount locally.
            // Requiring the exact collective request prevents ranks from
            // executing different signal/wait counts in a world barrier.
            if (fields[4] != context_count_ || fields[10] != context_count_)
                throw unsupported(
                    "gin_context_count_mismatch",
                    "rank " + std::to_string(peer) + " received " +
                    std::to_string(fields[4]) + " contexts and requested " +
                    std::to_string(fields[10]) + "; local request is " +
                    std::to_string(context_count_));
            if (fields[11] != world_barrier_count_)
                throw unsupported(
                    "gin_world_barrier_count_mismatch",
                    "rank " + std::to_string(peer) + " requested " +
                    std::to_string(fields[11]) +
                    " world GIN barrier slots; local request is " +
                    std::to_string(world_barrier_count_));
            if (fields[12] < kRequiredSignalCount)
                throw unsupported(
                    "gin_signal_count_mismatch",
                    "rank " + std::to_string(peer) + " received " +
                    std::to_string(fields[12]) +
                    " indexed GIN signals; at least " +
                    std::to_string(kRequiredSignalCount) + " are required");
            if (fields[13] != static_cast<int>(active_fast_path_))
                throw unsupported(
                    "active_fast_path_resource_mismatch",
                    "rank " + std::to_string(peer) + " initialized active_fast_path=" +
                    std::to_string(fields[13]) + ", local rank initialized " +
                    std::to_string(static_cast<int>(active_fast_path_)));
            if (fields[14] != static_cast<int>(bulk_combine_))
                throw unsupported(
                    "bulk_combine_resource_mismatch",
                    "rank " + std::to_string(peer) + " initialized bulk_combine=" +
                    std::to_string(fields[14]) + ", local rank initialized " +
                    std::to_string(static_cast<int>(bulk_combine_)));
            if (fields[15] != static_cast<int>(direct_dispatch_))
                throw unsupported(
                    "direct_dispatch_resource_mismatch",
                    "rank " + std::to_string(peer) + " initialized direct_dispatch=" +
                    std::to_string(fields[15]) + ", local rank initialized " +
                    std::to_string(static_cast<int>(direct_dispatch_)));
            if (fields[5] <= 0)
                throw unsupported("no_gin_connections",
                                  "rank " + std::to_string(peer) +
                                  " received no full GIN connections");
            if (fields[6] != 0 || fields[7] != 0)
                throw unsupported(
                    "unexpected_railed_gin",
                    "rank " + std::to_string(peer) +
                    " received railed resources for a full-connection design");
        }
    }

    void cleanup_after_failed_initialize() noexcept {
        if (comm_ == nullptr)
            return;
        if (dev_comm_created_) {
            ncclDevCommDestroy(comm_, &dev_comm_);
            dev_comm_created_ = false;
        }
        if (window_ != nullptr) {
            ncclCommWindowDeregister(comm_, window_);
            window_ = nullptr;
        }
        ncclCommAbort(comm_);
        comm_ = nullptr;
    }

    torch::Tensor buffer_;
    torch::Tensor diagnostic_buffer_;
    const void* buffer_data_ptr_ = nullptr;
    int64_t buffer_bytes_ = 0;
    std::string unique_id_bytes_;
    int rank_ = -1;
    int world_size_ = -1;
    int context_count_ = 0;
    int queue_depth_ = 0;
    int world_barrier_count_ = 0;
    int expected_lsa_size_ = 0;
    ncclGinType_t required_gin_type_ = NCCL_GIN_TYPE_NONE;
    ncclGinType_t gin_type_ = NCCL_GIN_TYPE_NONE;
    uint32_t completion_batch_ = 1;
    uint32_t combine_chunk_bytes_ = 7168;
    uint32_t outbox_depth_ = 8;
    uint32_t combine_issue_wave_ = 8;
    bool active_fast_path_ = false;
    bool bulk_combine_ = false;
    bool direct_dispatch_ = false;

    ncclComm_t comm_ = nullptr;
    ncclWindow_t window_ = nullptr;
    ncclDevComm_t dev_comm_{};
    bool dev_comm_created_ = false;
    int lsa_rank_ = -1;
    int lsa_size_ = 0;
    int actual_context_count_ = 0;
    int connection_count_ = 0;
    int actual_signal_count_ = 0;
};

// Typed bridge for csrc/apis/mega.hpp and its JIT wrapper. The shared owner is
// deliberately part of the call contract so the descriptor cannot outlive its
// NCCL communicator/window while a launch is being prepared.
static comm::MegaMoeGinTransport get_launch_descriptor(
    const std::shared_ptr<MegaMoeGinContext>& context) {
    if (context == nullptr)
        throw std::invalid_argument("MegaMoE GIN context must not be null");
    return context->launch_descriptor();
}

static py::bytes get_unique_id() {
    int runtime_version = 0;
    check_nccl(ncclGetVersion(&runtime_version), "ncclGetVersion");
    if (runtime_version != NCCL_VERSION_CODE) {
        throw unsupported(
            "runtime_header_version_mismatch",
            "runtime NCCL " + std::to_string(runtime_version) +
            " does not exactly match compiled headers " +
            std::to_string(NCCL_VERSION_CODE));
    }
    ncclUniqueId id{};
    check_nccl(ncclGetUniqueId(&id), "ncclGetUniqueId");
    return py::bytes(id.internal, NCCL_UNIQUE_ID_BYTES);
}

#endif // DG_MEGAMOE_GIN

static void register_apis(py::module_& m) {
    m.def("megamoe_gin_build_info", []() {
        py::dict result;
#ifdef DG_MEGAMOE_GIN
        result["enabled"] = true;
        result["compiled_nccl_version"] = NCCL_VERSION_CODE;
        result["minimum_nccl_version"] = kRequiredNcclVersion;
        result["required_nccl_version"] = kRequiredNcclVersion;
        result["default_context_count"] = kDefaultContextCount;
        result["default_queue_depth"] = kDefaultQueueDepth;
        result["default_world_barrier_count"] = kDefaultWorldBarrierCount;
        result["default_expected_lsa_size"] = kDefaultExpectedLsaSize;
        result["required_signal_count"] = kRequiredSignalCount;
#else
        result["enabled"] = false;
        result["compiled_nccl_version"] = py::none();
        result["minimum_nccl_version"] = kRequiredNcclVersion;
        result["required_nccl_version"] = kRequiredNcclVersion;
#endif
        return result;
    });

#ifdef DG_MEGAMOE_GIN
    py::class_<MegaMoeGinContext, std::shared_ptr<MegaMoeGinContext>>(
        m, "_MegaMoeGinContext")
        .def("destroy", &MegaMoeGinContext::destroy)
        .def("abort", &MegaMoeGinContext::abort)
        .def("set_diagnostic_buffer", &MegaMoeGinContext::set_diagnostic_buffer)
        .def("_release_buffer_registration",
             &MegaMoeGinContext::release_buffer_registration)
        .def_property_readonly("active", &MegaMoeGinContext::active)
        .def_property_readonly("rank", &MegaMoeGinContext::rank)
        .def_property_readonly("world_size", &MegaMoeGinContext::world_size)
        .def_property_readonly("lsa_rank", &MegaMoeGinContext::lsa_rank)
        .def_property_readonly("lsa_size", &MegaMoeGinContext::lsa_size)
        .def_property_readonly("context_count", &MegaMoeGinContext::context_count)
        .def_property_readonly("requested_context_count", &MegaMoeGinContext::requested_context_count)
        .def_property_readonly("connection_count", &MegaMoeGinContext::connection_count)
        .def_property_readonly("requested_signal_count", &MegaMoeGinContext::requested_signal_count)
        .def_property_readonly("signal_count", &MegaMoeGinContext::signal_count)
        .def_property_readonly("active_fast_path", &MegaMoeGinContext::active_fast_path)
        .def_property_readonly("bulk_combine", &MegaMoeGinContext::bulk_combine)
        .def_property_readonly("direct_dispatch", &MegaMoeGinContext::direct_dispatch)
        .def_property_readonly("queue_depth", &MegaMoeGinContext::queue_depth)
        .def_property_readonly("world_barrier_count", &MegaMoeGinContext::world_barrier_count)
        .def_property_readonly("gin_type", &MegaMoeGinContext::gin_type)
        .def_property_readonly("gin_type_string", &MegaMoeGinContext::gin_type_string)
        .def_property_readonly("buffer_bytes", &MegaMoeGinContext::buffer_bytes)
        .def("launch_descriptor_snapshot", &MegaMoeGinContext::launch_descriptor_snapshot);

    m.def("get_megamoe_gin_unique_id", &get_unique_id);
    m.def(
        "create_megamoe_gin_context",
        [](const torch::Tensor& buffer, const py::bytes& unique_id,
           const int rank, const int world_size, const int context_count,
           const int queue_depth, const int world_barrier_count,
           const py::object& expected_lsa_size, const std::string& required_gin_type,
           const uint32_t completion_batch,
           const uint32_t combine_chunk_bytes,
           const uint32_t outbox_depth,
           const uint32_t combine_issue_wave,
           const bool active_fast_path,
           const bool bulk_combine,
           const bool direct_dispatch) {
            const std::string unique_id_bytes = unique_id;
            // None selects the matching half-world LSA. An explicit value is
            // never overridden and remains subject to validate_arguments().
            const int resolved_lsa_size = expected_lsa_size.is_none() ?
                world_size / 2 : expected_lsa_size.cast<int>();
            py::gil_scoped_release release;
            return MegaMoeGinContext::create(
                buffer, unique_id_bytes, rank, world_size, context_count,
                queue_depth, world_barrier_count, resolved_lsa_size,
                required_gin_type, completion_batch, combine_chunk_bytes,
                outbox_depth, combine_issue_wave, active_fast_path,
                bulk_combine, direct_dispatch);
        },
        py::arg("buffer"), py::arg("unique_id"), py::arg("rank"),
        py::arg("world_size"), py::arg("context_count") = kDefaultContextCount,
        py::arg("queue_depth") = kDefaultQueueDepth,
        py::arg("world_barrier_count") = kDefaultWorldBarrierCount,
        py::arg("expected_lsa_size") = py::none(),
        py::arg("required_gin_type") = "gdaki",
        py::arg("completion_batch") = 1,
        py::arg("combine_chunk_bytes") = 7168,
        py::arg("outbox_depth") = 8,
        py::arg("combine_issue_wave") = 8,
        py::arg("active_fast_path") = false,
        py::arg("bulk_combine") = false,
        py::arg("direct_dispatch") = false);
#else
    m.def("get_megamoe_gin_unique_id", []() -> py::bytes {
        throw unsupported(
            "build_disabled",
            "rebuild with DG_MEGAMOE_GIN=1 and DG_NCCL_ROOT pointing to matching NCCL 2.30.7 headers and library");
    });
    m.def("create_megamoe_gin_context", [](py::args, py::kwargs) -> py::object {
        throw unsupported(
            "build_disabled",
            "rebuild with DG_MEGAMOE_GIN=1 and DG_NCCL_ROOT pointing to matching NCCL 2.30.7 headers and library");
    });
#endif
}

} // namespace deep_gemm::mega::gin
