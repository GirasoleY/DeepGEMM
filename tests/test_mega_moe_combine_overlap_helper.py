"""CPU contracts for expert-ready combine support, not GPU visibility proof.

Compile the actual layout and PUT/flush helpers against typed host stubs; inspect
the PTX-only publication functions for their explicit compiler memory ordering.
Generated CUDA/PTX/SASS and device accuracy remain separate required gates.
"""

from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "deep_gemm/include/deep_gemm/comm/mega_moe_gin.cuh"
LAYOUT = ROOT / "deep_gemm/include/deep_gemm/layout/mega_moe.cuh"
HOST = ROOT / "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp"
PUBLIC = ROOT / "deep_gemm/mega/__init__.py"


def function_source(source, name):
    match = re.search(
        r"(?:NCCL_DEVICE_INLINE|CUTLASS_HOST_DEVICE)\s+[^{};]*?\b"
        + re.escape(name) + r"\s*\(", source)
    if match is None:
        raise AssertionError(f"missing source function {name}")
    body = source.index("{", match.end())
    depth, end = 1, body + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]


class CombineOverlapHelperContract(unittest.TestCase):
    def compile_and_run(self, source):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("requires a host C++17 compiler, not CUDA")
        with tempfile.TemporaryDirectory(prefix="megamoe-combine-helper-") as temporary:
            executable = str(Path(temporary) / "test")
            result = subprocess.run(
                [compiler, "-std=c++17", "-O2", "-x", "c++", "-", "-o", executable],
                input=source, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run(
                [executable], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_every_publication_asm_has_memory_clobber(self):
        source = HELPER.read_text()
        for name, instruction in (
            ("mega_moe_gin_combine_producer_barrier", "bar.sync %0, %1;"),
            ("mega_moe_gin_combine_ready_release",
             "red.release.gpu.global.add.u32 [%0], %1;"),
            ("mega_moe_gin_combine_ready_acquire",
             "ld.acquire.gpu.global.b32 %0, [%1];"),
        ):
            with self.subTest(name=name):
                body = function_source(source, name)
                self.assertIn(instruction, body)
                self.assertEqual(body.count("asm volatile("), 1)
                self.assertIn(': "memory");', body)
        barrier = function_source(source, "mega_moe_gin_combine_producer_barrier")
        self.assertIn('"r"(barrier_idx), "r"(num_threads)', barrier)
        release = function_source(source, "mega_moe_gin_combine_ready_release")
        self.assertIn('"r"(1u)', release)

    def test_span_has_no_aggregate_or_completion_and_scopes_are_explicit(self):
        body = function_source(HELPER.read_text(), "mega_moe_gin_put_bulk_combine_span")
        for forbidden in ("AggregateRequests", "MaySkipCreditCheck", "flushAsync(",
                          "flush_data_peer", "wait_data_peer", "ncclGinRequest_t"):
            self.assertNotIn(forbidden, body)
        self.assertIn("cuda::thread_scope_device, cuda::thread_scope_system", body)
        self.assertIn("ncclGinOptFlagsDefault);", body)
        self.assertEqual(body.count('asm volatile("" ::: "memory");'), 2)
        self.assertLess(body.index('asm volatile(""'), body.index("gin.put("))
        self.assertGreater(body.rindex('asm volatile(""'), body.index("gin.put("))
        self.assertIn("64u + diagnostic_peer_lane, true", body)

    def test_compiled_fixed_expert_prefix_alias_offsets_and_fit(self):
        source = LAYOUT.read_text()
        constants = source[source.index("static constexpr int kNumCandidateBlockMs"):
                           source.index("// Pool capacity")]
        methods = "\n".join(function_source(source, name) for name in (
            "combine_overlap_alias_fits", "get_combine_overlap_ready_ptr",
            "get_combine_overlap_sent_ptr", "get_combine_overlap_prefix_ptr"))
        self.compile_and_run(r'''
#include <array>
#include <cassert>
#include <cstdint>
#define CUTLASS_HOST_DEVICE
#define DG_UNIFIED_ASSERT(x) assert(x)
namespace math {
inline void* advance_ptr(void* ptr, uint64_t bytes) {
    return static_cast<uint8_t*>(ptr) + bytes;
}
}
''' + constants + r'''
struct Buffer {
    void* base;
    uint64_t bytes;
    uint64_t get_num_bytes() const { return bytes; }
};
struct Workspace {
    Buffer scale_scratch_buffer;
''' + methods + r'''
};
int main() {
    static_assert(kMegaMoeGinCombineOverlapNumExperts == 56);
    static_assert(get_mega_moe_gin_combine_overlap_scratch_bytes() == 2240);
    static_assert(kMegaMoeGinDirectDispatchStorageBytes == 57344);
    alignas(128) std::array<uint8_t, 100000> storage{};
    Workspace workspace{{storage.data(), storage.size()}};
    const uint64_t required = kMegaMoeGinDirectDispatchStorageBytes +
        get_mega_moe_gin_combine_overlap_scratch_bytes();
    workspace.scale_scratch_buffer.bytes = required - 1;
    assert(!workspace.combine_overlap_alias_fits());
    workspace.scale_scratch_buffer.bytes = required;
    assert(workspace.combine_overlap_alias_fits());
    auto* ready = workspace.get_combine_overlap_ready_ptr(0);
    auto* sent = workspace.get_combine_overlap_sent_ptr(0);
    auto* prefixes = workspace.get_combine_overlap_prefix_ptr(0, 0);
    assert(reinterpret_cast<uint8_t*>(ready) ==
           storage.data() + kMegaMoeGinDirectDispatchStorageBytes);
    assert(sent == ready + 56);
    assert(prefixes == sent + 56);
    assert(workspace.get_combine_overlap_ready_ptr(55) + 1 == sent);
    assert(workspace.get_combine_overlap_sent_ptr(55) + 1 == prefixes);
    for (uint32_t peer = 0; peer < 8; ++peer) {
        for (uint32_t expert = 0; expert < 56; ++expert) {
            auto* prefix = workspace.get_combine_overlap_prefix_ptr(peer, expert);
            assert(prefix == prefixes + peer * 56 + expert);
            *prefix = peer * 56 + expert;
        }
    }
    assert(reinterpret_cast<uint8_t*>(workspace.get_combine_overlap_prefix_ptr(7, 55) + 1)
           == storage.data() + required);
    *ready = 448; *sent = 1;
    assert(*ready == 448 && *sent == 1);
    assert(*workspace.get_combine_overlap_prefix_ptr(7, 55) == 447);
    workspace.scale_scratch_buffer.bytes = 128ull * 4 * 112;
    assert(!workspace.combine_overlap_alias_fits());
    workspace.scale_scratch_buffer.bytes = 148ull * 4 * 112;
    assert(workspace.combine_overlap_alias_fits());
}
''')

    def test_compiled_span_header_scopes_and_completion_order(self):
        source = HELPER.read_text()
        functions = "\n".join(function_source(source, name) for name in (
            "mega_moe_gin_put_data", "mega_moe_gin_put_bulk_combine_span",
            "mega_moe_gin_flush_data_peer_async", "mega_moe_gin_wait_data_peer",
            "mega_moe_gin_put_bulk_combine_header"))
        self.compile_and_run(r'''
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>
#define NCCL_DEVICE_INLINE inline
#define DG_DEVICE_ASSERT(x) assert(x)
namespace cuda {
enum thread_scope { thread_scope_system, thread_scope_device };
enum memory_order { memory_order_acquire };
}
struct ncclGin_None {};
struct ncclCoopThread {};
struct ncclGinRequest_t {};
using ncclWindow_t = void*;
constexpr int NCCL_GIN_RESOURCE_SHARING_GPU = 1;
constexpr uint32_t ncclGinOptFlagsDefault = 0;
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
struct { uint32_t x = 0; } blockIdx;
struct Event {
    char kind;
    int peer, context;
    size_t dst, src, bytes;
    cuda::thread_scope given, required;
    uint32_t flags;
};
std::vector<Event> events;
std::vector<std::pair<uint32_t, bool>> traces;
inline void mega_moe_gin_trace(const MegaMoeGinTransport&, uint32_t,
                               uint32_t column, bool first_only = false) {
    traces.emplace_back(column, first_only);
}
struct ncclGin {
    int context;
    ncclGin(DevComm, int context, int) : context(context) {}
    void put(int, int peer, ncclWindow_t, size_t dst, ncclWindow_t, size_t src,
             size_t bytes, ncclGin_None, ncclGin_None, ncclCoopThread, ncclGin_None,
             cuda::thread_scope given, cuda::thread_scope required, uint32_t flags) {
        events.push_back({'P', peer, context, dst, src, bytes, given, required, flags});
    }
    void flushAsync(int, int peer, ncclGinRequest_t*, ncclCoopThread,
                    uint32_t flags, ncclGin_None) {
        events.push_back({'F', peer, context, 0, 0, 0,
                         cuda::thread_scope_device, cuda::thread_scope_device, flags});
    }
    void wait(ncclGinRequest_t&, ncclCoopThread, ncclGin_None, cuda::memory_order order) {
        assert(order == cuda::memory_order_acquire);
        events.push_back({'W', -1, context, 0, 0, 0,
                         cuda::thread_scope_device, cuda::thread_scope_device, 0});
    }
};
''' + functions + r'''
int main() {
    alignas(16) std::array<uint8_t, 65536> storage{};
    auto* base = storage.data();
    MegaMoeGinTransport transport;
    transport.window = base;
    // Two disjoint ready spans, no implicit completion and no count header.
    mega_moe_gin_put_bulk_combine_span(
        transport, 11, 0, base, base + 16, base + 32784, 7184, 3);
    mega_moe_gin_put_bulk_combine_span(
        transport, 11, 0, base, base + 14384, base + 47152, 14368, 3);
    assert(events.size() == 2);
    assert(events[0].src == 16 && events[0].dst == 32784 && events[0].bytes == 7184);
    assert(events[1].src == 14384 && events[1].dst == 47152 && events[1].bytes == 14368);
    for (const auto& event : events) {
        assert(event.kind == 'P' && event.peer == 11 && event.context == 1);
        assert(event.given == cuda::thread_scope_device);
        assert(event.required == cuda::thread_scope_system && event.flags == 0);
    }
    assert(traces[0] == std::make_pair(67u, true));
    assert(traces[2] == std::make_pair(67u, true));
    // Caller retires payloads before its handoff, separately from the late header.
    ncclGinRequest_t request;
    mega_moe_gin_flush_data_peer_async(transport, 11, 0, &request);
    mega_moe_gin_wait_data_peer(transport, 0, request);
    assert(events[2].kind == 'F' && events[3].kind == 'W');
    mega_moe_gin_put_bulk_combine_header(transport, 11, 0, base, base, base + 32768, 3);
    assert(events.size() == 7);
    const auto& header = events[4];
    assert(header.kind == 'P' && header.peer == 11 && header.context == 1);
    assert(header.src == 0 && header.dst == 32768 && header.bytes == 16);
    assert(header.given == cuda::thread_scope_device &&
           header.required == cuda::thread_scope_device && header.flags == 0);
    assert(events[5].kind == 'F' && events[6].kind == 'W');
    assert(traces[4] == std::make_pair(67u, true));
    assert(traces.back() == std::make_pair(83u, false));
}
''')

    def test_host_flag_is_strict_keyed_and_preserves_selected_config(self):
        source = HOST.read_text()
        self.assertEqual(source.count(
            'get_env<std::string>("DG_MEGAMOE_GIN_COMBINE_OVERLAP", "0")'), 1)
        self.assertIn('gin_combine_overlap_value == "0" or', source)
        self.assertIn('gin_combine_overlap_value == "1"', source)
        self.assertIn('DG_HOST_ASSERT(gin_combine_overlap_value == "0");', source)
        self.assertIn("gin_single_combine_context and gin_dispatch_overlap and", source)
        self.assertIn("#define DG_MEGAMOE_GIN_COMBINE_OVERLAP {}", source)
        self.assertIn('args.gin_combine_overlap ? "1" : "0",', source)
        self.assertIn(".gin_combine_overlap = gin_combine_overlap,", source)
        self.assertIn(".config = config,", source)
        self.assertIn(".num_tokens = num_tokens,", source)
        defines = re.findall(r"#define DG_MEGAMOE_(GIN_[A-Z_]+) \{\}", source)
        format_begin = source.index(')", args.gin_local_ablation_stage,')
        format_end = source.index("args.num_max_tokens_per_rank,", format_begin)
        values = re.findall(r"args\.(gin_[a-z_]+)", source[format_begin:format_end])
        self.assertEqual([name.lower() for name in defines], values)

    def test_public_raw_flag_is_collected_before_validation(self):
        source = PUBLIC.read_text()
        raw = source.index("'combine_overlap': os.environ.get(")
        collective = source.index("dist.all_gather_object(gathered, local_config", raw)
        validation = source.index("canonical['combine_overlap'] not in ('0', '1')", collective)
        self.assertLess(raw, collective)
        self.assertLess(collective, validation)
        self.assertIn("canonical['single_combine_context'] == '1'", source[validation:])
        self.assertIn("canonical['dispatch_overlap'] == '1'", source[validation:])


if __name__ == "__main__":
    unittest.main()
