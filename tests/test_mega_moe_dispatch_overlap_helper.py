"""Compile the actual publication helpers against a typed CPU NCCL stub.

Checks request ordering and completion placement, not GPU/network visibility.
The stub signatures follow the archived NCCL 2.30.7 gin.h put/signal API.
"""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "deep_gemm/include/deep_gemm/comm/mega_moe_gin.cuh"


def function_source(source, name, *, templated=False):
    start = source.index("NCCL_DEVICE_INLINE void " + name + "(")
    if templated:
        start = source.rfind("template <", 0, start)
    body = source.index("{", source.index(name + "(", start))
    depth, end = 1, body + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


class DispatchOverlapHelperContract(unittest.TestCase):
    def test_async_helpers_do_not_request_completion_or_wait(self):
        source = HELPER.read_text()
        for name in ("mega_moe_gin_publish_direct_control_async",
                     "mega_moe_gin_publish_direct_payload_async"):
            body = function_source(source, name)
            self.assertNotIn("flushAsync(", body)
            self.assertNotIn(".wait(", body)
            self.assertNotIn("ncclGinRequest_t", body)

    def test_compiled_ordering_empty_pair_and_baseline_delegation(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("requires a host C++17 compiler, not CUDA")
        source = HELPER.read_text()
        declarations = r'''
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>
#define NCCL_DEVICE_INLINE inline
#define DG_DEVICE_ASSERT(condition) assert(condition)
namespace cuda {
enum thread_scope { thread_scope_device };
enum memory_order { memory_order_acquire };
}
struct ncclGin_None {};
struct ncclCoopThread {};
struct ncclGinRequest_t {};
using ncclWindow_t = void*;
struct ncclGin_StrongVASignalInc { ncclWindow_t signalWindow; size_t signalOffset; };
constexpr int NCCL_GIN_RESOURCE_SHARING_GPU = 1;
constexpr uint32_t ncclGinOptFlagsDefault = 0;
constexpr uint32_t ncclGinOptFlagsAggregateRequests = 1;
struct DevComm { int lsaSize = 8; };
inline int ncclTeamWorld(DevComm) { return 0; }
struct MegaMoeGinTransport {
    DevComm dev_comm;
    ncclWindow_t window = nullptr;
    uint32_t data_context(uint32_t stripe) const { return 1 + stripe % 8; }
    size_t window_offset(const void* base, const void* ptr) const {
        return static_cast<const uint8_t*>(ptr) - static_cast<const uint8_t*>(base);
    }
};
struct { uint32_t x = 1; } blockIdx;
struct Event {
    char kind;
    int peer, context;
    size_t dst, src, bytes, signal;
    uint32_t flags;
    bool operator==(const Event& other) const {
        return kind == other.kind && peer == other.peer && context == other.context &&
               dst == other.dst && src == other.src && bytes == other.bytes &&
               signal == other.signal && flags == other.flags;
    }
};
std::vector<Event> events;
constexpr size_t no_signal = size_t(-1);
inline size_t signal_offset(ncclGin_None) { return no_signal; }
inline size_t signal_offset(ncclGin_StrongVASignalInc action) { return action.signalOffset; }
inline void mega_moe_gin_trace(const MegaMoeGinTransport&, uint32_t, uint32_t column) {
    events.push_back({'T', 0, 0, column, 0, 0, no_signal, 0});
}
struct ncclGin {
    int context;
    ncclGin(DevComm, int context, int) : context(context) {}
    template <class RemoteAction, class LocalAction, class Coop, class Descriptor>
    void put(int, int peer, ncclWindow_t, size_t dst, ncclWindow_t, size_t src,
             size_t bytes, RemoteAction action, LocalAction, Coop, Descriptor,
             cuda::thread_scope, cuda::thread_scope, uint32_t flags) const {
        events.push_back({'P', peer, context, dst, src, bytes, signal_offset(action), flags});
    }
    template <class RemoteAction, class Coop, class Descriptor>
    void signal(int, int peer, RemoteAction action, Coop, Descriptor,
                cuda::thread_scope, cuda::thread_scope, uint32_t flags) const {
        events.push_back({'S', peer, context, 0, 0, 0, signal_offset(action), flags});
    }
    void flushAsync(int, int peer, ncclGinRequest_t*, ncclCoopThread,
                    uint32_t flags, ncclGin_None) const {
        events.push_back({'F', peer, context, 0, 0, 0, no_signal, flags});
    }
    void wait(ncclGinRequest_t&, ncclCoopThread, ncclGin_None, cuda::memory_order) const {
        events.push_back({'W', -1, context, 0, 0, 0, no_signal, 0});
    }
};
'''
        functions = "\n".join(function_source(source, name, templated=templated)
            for name, templated in (
                ("mega_moe_gin_publish_direct_dispatch", False),
                ("mega_moe_gin_publish_direct_control_async", False),
                ("mega_moe_gin_publish_direct_payload_async", False),
                ("mega_moe_gin_publish_direct_dispatch_ordered", True),
                ("mega_moe_gin_flush_data_peer_async", False),
                ("mega_moe_gin_wait_data_peer", False)))
        main = r'''
int main() {
    alignas(16) std::array<uint8_t, 8192> storage{};
    auto* base = storage.data();
    MegaMoeGinTransport transport;
    transport.window = base;
    auto direct = [&](bool active) {
        mega_moe_gin_publish_direct_dispatch(transport, 9, 1, base, active,
            base + 64, base + 512, 3584, base + 1024, base + 2048, 112,
            base + 3072, base + 3584, 64, base + 4096, base + 4608, 448, base + 6144);
    };
    auto ordered_baseline = [&](bool active) {
        mega_moe_gin_publish_direct_dispatch_ordered<false>(transport, 9, 1, base, active,
            base + 64, base + 512, 3584, base + 1024, base + 2048, 112,
            base + 3072, base + 3584, 64, base + 4096, base + 4608, 448,
            base + 6144, nullptr);
    };
    auto ordered_overlap = [&](bool active) {
        mega_moe_gin_publish_direct_dispatch_ordered<true>(transport, 9, 1, base, active,
            base + 64, base + 512, 3584, base + 1024, base + 2048, 112,
            base + 3072, base + 3584, 64, base + 4096, base + 4608, 448,
            base + 6144, base + 6152);
    };
    for (bool active : {false, true}) {
        events.clear();
        direct(active);
        const auto baseline = events;
        events.clear();
        ordered_baseline(active);
        assert(events == baseline);  // Includes old completion and phase traces.
        events.clear();
        ordered_overlap(active);
        const auto queued = events;
        assert(queued.front().kind == 'T' && queued.front().dst == 9);
        assert(queued.back().kind == 'T' && queued.back().dst == 17);
        std::vector<Event> operations;
        for (const auto& event : queued) {
            assert(event.kind != 'F' && event.kind != 'W');
            if (event.kind != 'T') operations.push_back(event);
        }
        assert(operations.size() == (active ? 4u : 2u));
        assert(operations[0] == (Event{'P', 9, 2, 4608, 4096, 448, 6144, 0}));
        if (active) {
            assert(operations[1] == (Event{'P', 9, 2, 512, 64, 3584, no_signal, 1}));
            assert(operations[2] == (Event{'P', 9, 2, 2048, 1024, 112, no_signal, 1}));
            assert(operations[3] == (Event{'P', 9, 2, 3584, 3072, 64, 6152, 0}));
        } else {
            assert(operations[1] == (Event{'S', 9, 2, 0, 0, 0, 6152, 0}));
        }
        // Request storage is created only at the later caller-owned reuse gate.
        ncclGinRequest_t request;
        mega_moe_gin_flush_data_peer_async(transport, 9, 1, &request);
        mega_moe_gin_wait_data_peer(transport, 1, request);
        assert(events.size() == queued.size() + 2);
        assert(events[queued.size()] == (Event{'F', 9, 2, 0, 0, 0, no_signal, 0}));
        assert(events.back() == (Event{'W', -1, 2, 0, 0, 0, no_signal, 0}));
    }
}
'''
        with tempfile.TemporaryDirectory(prefix="megamoe-overlap-helper-") as temporary:
            executable = str(Path(temporary) / "test")
            result = subprocess.run(
                [compiler, "-std=c++17", "-O2", "-x", "c++", "-", "-o", executable],
                input="\n".join((declarations, functions, main)),
                capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([executable], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
