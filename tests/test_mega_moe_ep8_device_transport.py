"""CPU contracts for EP8/LSA4 and preserved EP16/LSA8 device indexing.

The compiled pack test executes the actual kernel's two packet-building
blocks against bounds-checked host storage. Shuffles use independently
computed lane totals; this is not a CUDA compiler or transport visibility test.
"""

from pathlib import Path
import hashlib
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh"
HELPER = ROOT / "deep_gemm/include/deep_gemm/comm/mega_moe_gin.cuh"


class Ep8DeviceTransportContract(unittest.TestCase):
    def test_compile_tu_covers_native_and_gin_without_changing_ep16_tu(self):
        ep8 = (ROOT / "tests/compile_sm100_megamoe_gin_ep8.cu").read_text()
        self.assertIn("#if !DG_EP8_COMPILE_NATIVE", ep8)
        self.assertIn("448, 0, 16,", ep8)
        self.assertIn("152, 8,", ep8)
        self.assertIn("false, !DG_EP8_COMPILE_NATIVE", ep8)
        for profile in ("<16, 256, 8, 5>", "<32, 128, 16, 10>",
                        "<64, 128, 32, 9, 128>"):
            self.assertIn("instantiate_ep8_decode" + profile, ep8)
        self.assertIn("#ifndef DG_MEGAMOE_GIN_COMBINE_OVERLAP", ep8)
        self.assertIn(
            "#ifndef DG_MEGAMOE_GIN_STRONGVA_COMBINE_TERMINAL", ep8)
        original = (ROOT / "tests/compile_sm100_megamoe_gin.cu").read_text()
        self.assertIn("896, 0, 16,", original)
        self.assertIn("148, 16,", original)

    def test_compute_and_ordered_reducer_source_match_frozen_ep16(self):
        source = KERNEL.read_text()
        start = source.index("        // GEMM TMA load warp for tokens with SFA")
        end = source.index("        // Deallocate tensor memory", start)
        end = source.index("\n", end) + 1
        # Source bytes from d22387a, not a compiled-cubin identity claim.
        self.assertEqual(hashlib.sha256(source[start:end].encode()).hexdigest(),
                         "07ff056e2e3d9f120c571ecdcb1e033b14172af516528d0fee0119c3e7dc5f21")
        reducer = source.index("        // Combine: reduce top-k results and write back")
        self.assertEqual(hashlib.sha256(source[reducer:].encode()).hexdigest(),
                         "7acd8cc615165b7183b4510d53e8f4e42d1cfd7ecc37aa8f69dc7a5265a71a17")

    def test_only_supported_two_team_shapes_and_capacity_remains_maximum(self):
        source = KERNEL.read_text()
        self.assertIn("constexpr uint32_t kGinPeerCount = kNumRanks / 2u;", source)
        self.assertIn("kNumRanks == 8 or kNumRanks == 16", source)
        self.assertIn("gin_transport.dev_comm.lsaSize == kGinPeerCount", source)
        self.assertIn("gin_transport.dev_comm.ginContextCount >= 9", source)
        self.assertIn("kNumExpertsPerRank == 56", source)
        self.assertIn("layout::kMegaMoeGinDirectDispatchNumPeers *", source)
        self.assertIn("constexpr uint32_t kMaxReadyExpertsPerPut = 8u;", source)

    def test_all_lsa_barriers_use_actual_team_width(self):
        source = KERNEL.read_text()
        calls = re.findall(r"comm::nvlink_lsa_barrier<\s*([^>]+)>", source)
        # The default-off StrongVA specialization moves the existing combine
        # LSA rendezvous ahead of its per-owner signal waits. Both compile-time
        # branches remain in source, although exactly one executes.
        self.assertEqual(len(calls), 10)
        for call in calls:
            self.assertTrue(call.startswith("kNumRanks, kGinPeerCount,"), call)

    def test_actual_peer_bounds_cover_wait_retirement_scan_scatter(self):
        source = KERNEL.read_text()
        self.assertNotRegex(source, r"lane_idx < 8u?\b")
        self.assertNotIn("sym_buffer.rank_idx % 8u", source)
        self.assertNotIn("sym_buffer.rank_idx / 8u", source)
        self.assertIn("source_wave < kGinPeerCount / kNumDispatchWarps", source)
        self.assertIn("kGinPeerCount * kNumExpertsPerRank", source)
        self.assertIn("owner_in_lsa < kGinPeerCount", source)
        self.assertIn("context_stripe < kGinPeerCount", source)
        self.assertRegex(source, r"active_context\s*=\s*\(context_stripe \+ 1\)\s*%\s*kGinPeerCount")

    def test_scan_waves_cover_every_actual_source_once(self):
        for world in (8, 16):
            width = world // 2
            scanned = [warp + wave * 4 for wave in range(width // 4)
                       for warp in range(4)]
            self.assertEqual(scanned, list(range(width)))
            for rank in range(world):
                remote_base = (1 - rank // width) * width
                remote = [remote_base + lane for lane in scanned]
                self.assertEqual(len(set(remote)), width)
                self.assertTrue(all(0 <= peer < world and
                                    peer // width != rank // width
                                    for peer in remote))

    def test_only_actual_peers_advance_epochs_across_direct_and_fallback(self):
        # Eligibility, not local token count alone, controls epoch advancement.
        for width in (4, 8):
            control = [0] * 8
            payload = [0] * 8
            expected_control = expected_payload = 0
            for direct, overlap in ((True, False), (True, True),
                                    (False, True), (True, True), (True, False)):
                if direct:
                    expected_control += 1
                    expected_payload += overlap
                    for peer in range(width):
                        # Includes peers with zero assignments.
                        control[peer] += 1
                        payload[peer] += overlap
                self.assertEqual(control[:width], [expected_control] * width)
                self.assertEqual(payload[:width], [expected_payload] * width)
                self.assertEqual(control[width:] + payload[width:], [0] * (2 * (8 - width)))

    def test_actual_pack_blocks_and_transport_context_execute_on_cpu(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("requires a C++17 compiler, not CUDA")
        source = KERNEL.read_text()
        blocks = []
        cursor = 0
        for _ in range(2):
            begin = source.index("constexpr uint32_t kExpertsPerHelper = 14;", cursor)
            end = source.index("__threadfence_system();", begin)
            blocks.append(source[begin:end])
            cursor = end
        helper = HELPER.read_text()
        methods = helper[helper.index("    NCCL_DEVICE_INLINE bool is_same_lsa_peer("):
                         helper.index("    NCCL_DEVICE_INLINE size_t window_offset(")]
        header = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>
#define NCCL_DEVICE_INLINE inline
#define DG_DEVICE_ASSERT(x) assert(x)
constexpr uint32_t poison = 0xdeadbeef;
namespace layout { constexpr uint32_t kMegaMoeGinDirectDispatchMaxRoutes = 768; }
struct Transport { struct { int rank, lsaSize; } dev_comm;
'''
        host = r'''
};
struct Workspace {
    std::vector<uint32_t> counts;
    explicit Workspace(uint32_t experts): counts(experts) {}
    uint32_t* get_expert_send_count_ptr(uint32_t expert) { return &counts.at(expert); }
};
struct GinWorkspace {
    std::array<std::array<uint32_t, 56>, 8> counts;
    std::array<std::array<uint32_t, 768>, 8> routes;
    std::array<uint32_t, 8> ready;
    std::array<uint32_t, 768> ordinal;
    std::vector<std::vector<uint32_t>> staging;
    explicit GinWorkspace(uint32_t experts): staging(experts) {
        for (auto& row: counts) row.fill(poison);
        for (auto& row: routes) row.fill(poison);
        ready.fill(poison); ordinal.fill(poison);
    }
    uint32_t* get_direct_dispatch_count_ptr(bool send, uint32_t peer) {
        assert(send); return counts.at(peer).data();
    }
    uint32_t* get_direct_dispatch_route_ptr(bool send, uint32_t peer) {
        assert(send); return routes.at(peer).data();
    }
    uint32_t* get_direct_dispatch_ready_ptr(bool send, uint32_t peer) {
        assert(send); return &ready.at(peer);
    }
    uint32_t* get_route_staging_ptr(uint32_t expert, uint32_t slot) {
        return &staging.at(expert).at(slot);
    }
    uint32_t* get_combine_direct_reduce_ordinal_ptr(uint32_t assignment) {
        return &ordinal.at(assignment);
    }
};
template<unsigned kGinPeerCount, bool prepack>
void check(uint32_t rank, uint32_t num_tokens, uint32_t pattern) {
    constexpr uint32_t kNumExpertsPerRank = 56, kNumTopk = 16;
    const uint32_t lsa_size = kGinPeerCount;
    const uint32_t remote_base = (1 - rank / lsa_size) * lsa_size;
    struct { uint32_t rank_idx; } sym_buffer{rank};
    Workspace workspace(2 * lsa_size * 56);
    struct { GinWorkspace gin_workspace; } buffer{GinWorkspace(2 * lsa_size * 56)};
    auto& data = buffer.gin_workspace;
    const bool prepare_gin_direct_reduce_ordinals = true;
    const uint32_t n = pattern == 0 ? 0 : num_tokens * kNumTopk;
    for (uint32_t item = 0; item < n; ++item) {
        const uint32_t peer = pattern == 1 ? 0 : item % lsa_size;
        const uint32_t expert = pattern == 1 ? item / num_tokens :
            (item / lsa_size * 13u) % 56;
        const uint32_t global = (remote_base + peer) * 56 + expert;
        ++workspace.counts.at(global);
        data.staging.at(global).push_back(item);
    }
    std::array<uint32_t, 32> segments{};
    for (uint32_t lane = 0; lane < 32; ++lane) {
        const uint32_t peer = lane & 7u, helper = lane >> 3;
        if (peer < lsa_size)
            for (uint32_t e = helper * 14; e < (helper + 1) * 14; ++e)
                segments[lane] += workspace.counts.at((remote_base + peer) * 56 + e);
    }
    for (uint32_t lane_idx = 0; lane_idx < 32; ++lane_idx) {
        auto __shfl_sync = [&](uint32_t mask, uint32_t value, uint32_t lane) {
            assert(mask == 0xffffffffu && value == segments[lane_idx]);
            return segments.at(lane);
        };
        if constexpr (prepack) {
'''
        tail = r'''
        }
    }
    for (uint32_t peer = 0; peer < 8; ++peer) {
        if (peer >= lsa_size) {
            for (auto value: data.counts[peer]) assert(value == poison);
            for (auto value: data.routes[peer]) assert(value == poison);
            assert(data.ready[peer] == poison);
            continue;
        }
        uint32_t prefix = 0;
        for (uint32_t e = 0; e < 56; ++e) {
            const uint32_t global = (remote_base + peer) * 56 + e;
            assert(data.counts[peer][e] == workspace.counts[global]);
            for (auto assignment: data.staging[global]) {
                assert(data.routes[peer][prefix] == assignment);
                assert(data.ordinal[assignment] == prefix);
                ++prefix;
            }
        }
        if constexpr (prepack) assert(data.ready[peer] == prefix);
        for (; prefix < 768; ++prefix) assert(data.routes[peer][prefix] == poison);
    }
    for (uint32_t assignment = n; assignment < 768; ++assignment)
        assert(data.ordinal[assignment] == poison);
    Transport transport{{int(rank), int(lsa_size)}};
    for (uint32_t peer = 0; peer < 2 * lsa_size; ++peer)
        assert(transport.is_same_lsa_peer(peer) == (peer / lsa_size == rank / lsa_size));
    for (uint32_t stripe = 0; stripe < lsa_size; ++stripe)
        assert(transport.data_context(stripe) == stripe + 1);
}
int main() {
    for (uint32_t rank = 0; rank < 16; ++rank)
        for (uint32_t tokens: {32u, 40u, 48u})
            for (uint32_t pattern = 0; pattern < 3; ++pattern) {
                check<8, true>(rank, tokens, pattern);
                check<8, false>(rank, tokens, pattern);
                if (rank < 8) {
                    check<4, true>(rank, tokens, pattern);
                    check<4, false>(rank, tokens, pattern);
                }
            }
}
'''
        program = header + methods + host + blocks[0] + "\n} else {\n" + blocks[1] + tail
        with tempfile.TemporaryDirectory(prefix="megamoe-ep8-pack-") as directory:
            source_path = Path(directory) / "pack.cpp"
            binary = Path(directory) / "pack"
            source_path.write_text(program)
            subprocess.run([compiler, "-std=c++17", "-Wno-unknown-pragmas",
                            str(source_path), "-o", str(binary)], check=True,
                           capture_output=True, text=True)
            subprocess.run([str(binary)], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
