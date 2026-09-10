"""CPU-only EP8 host/layout contracts; not CUDA, fabric, or accuracy proof."""

import ast
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "deep_gemm/mega/__init__.py"
HOST = ROOT / "csrc/apis/mega.hpp"
GIN = ROOT / "csrc/apis/mega_gin.hpp"
JIT = ROOT / "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp"
LAYOUT = ROOT / "deep_gemm/include/deep_gemm/layout/mega_moe.cuh"


class ReachedAllocation(Exception):
    pass


class ReachedUID(Exception):
    pass


def public_class(namespace):
    tree = ast.parse(PUBLIC.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == "SymmBuffer")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(PUBLIC), "exec"), namespace)
    return namespace["SymmBuffer"]


class EP8PublicContract(unittest.TestCase):
    def setUp(self):
        self.configs = []
        self.gather_hook = None

        def gather(output, local, group):
            output[:] = [local.copy() if isinstance(local, dict) else local
                         for _ in range(group.size())]
            if isinstance(local, dict):
                self.configs.append(local)
            if self.gather_hook is not None:
                self.gather_hook(output, local)

        extension = types.SimpleNamespace(
            get_symm_buffer_size_for_mega_moe=mock.Mock(side_effect=ReachedAllocation),
            megamoe_gin_build_info=lambda: {"enabled": True})
        self.namespace = {"os": os, "_C": extension,
                          "dist": types.SimpleNamespace(all_gather_object=gather)}
        self.cls = public_class(self.namespace)
        self.environment = mock.patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def buffer(self, world, rank=0):
        value = self.cls.__new__(self.cls)
        value.group = types.SimpleNamespace(size=lambda: world, rank=lambda: rank)
        value.buffer = types.SimpleNamespace(numel=lambda: 1024, element_size=lambda: 1)
        value._gin_context = None
        value._gin_layout_enabled = True
        value.num_experts = world * 56
        value.num_max_tokens_per_rank = 384
        value.num_topk, value.hidden, value.intermediate_hidden = 16, 3584, 3072
        value.num_shared_experts = 0
        value.gin_queue_depth, value.gin_completion_batch = 64, 8
        value.gin_combine_chunk_bytes, value.gin_outbox_depth = 7168, 64
        value.gin_combine_issue_wave = 8
        value.gin_active_fast_path = value.gin_bulk_combine = value.gin_direct_dispatch = True
        begin = rank // (world // 2) * (world // 2)
        value.buffer_ptrs = [0x1000 + i if begin <= i < begin + world // 2 else 0
                             for i in range(world)]
        value._collective_get_gin_unique_id = mock.Mock(side_effect=ReachedUID)
        return value

    def test_default_and_explicit_lsa_resolve_before_uid(self):
        for world in (8, 16):
            for rank in (0, world - 1):
                for explicit in (None, world // 2):
                    with self.subTest(world=world, rank=rank, explicit=explicit):
                        value = self.buffer(world, rank)
                        with self.assertRaises(ReachedUID):
                            value.enable_gin(expected_lsa_size=explicit)
                        self.assertEqual(self.configs[-1]["expected_lsa_size"], world // 2)
                        self.assertEqual(self.configs[-1]["context_count"], 9)

    def test_wrong_explicit_width_and_unsupported_world_fail_before_uid(self):
        for world, width in ((8, 8), (16, 4), (8, 0), (16, -1), (12, None)):
            value = self.buffer(world)
            with self.subTest(world=world, width=width):
                with self.assertRaisesRegex(RuntimeError, "collective configuration rejected"):
                    value.enable_gin(expected_lsa_size=width)
                value._collective_get_gin_unique_id.assert_not_called()

    def test_rank_skewed_width_is_collectively_rejected(self):
        def skew(output, local):
            if isinstance(local, dict):
                output[-1]["expected_lsa_size"] = 8
        self.gather_hook = skew
        value = self.buffer(8)
        with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
            value.enable_gin()
        value._collective_get_gin_unique_id.assert_not_called()

    def test_ep8_rejects_wrong_expert_count_and_insufficient_contexts(self):
        for changes in ({"num_experts": 896}, {"context_count": 8}):
            value = self.buffer(8)
            kwargs = {}
            for key, val in changes.items():
                if key == "context_count":
                    kwargs[key] = val
                else:
                    setattr(value, key, val)
            with self.assertRaisesRegex(RuntimeError, "collective configuration rejected"):
                value.enable_gin(**kwargs)
            value._collective_get_gin_unique_id.assert_not_called()

    def test_missing_lsa_alias_fails_before_uid(self):
        for world in (8, 16):
            value = self.buffer(world, world - 1)
            value.buffer_ptrs[world - 2] = 0
            with self.assertRaisesRegex(RuntimeError, "pointer alias validation"):
                value.enable_gin()
            value._collective_get_gin_unique_id.assert_not_called()

    def test_constructor_exact_bulk_and_direct_shapes_before_allocation(self):
        for world in (8, 16):
            for bulk, direct in ((True, False), (False, True), (True, True)):
                group = types.SimpleNamespace(size=lambda: world)
                with self.assertRaises(ReachedAllocation):
                    self.cls(group, world * 56, 384, 16, 3584, 3072,
                             enable_gin=True, gin_completion_batch=8,
                             gin_active_fast_path=True, gin_outbox_depth=64,
                             gin_bulk_combine=bulk, gin_direct_dispatch=direct)
        for world, experts, capacity in ((8, 896, 384), (16, 448, 384),
                                        (4, 224, 384), (8, 448, 383)):
            group = types.SimpleNamespace(size=lambda: world)
            with self.assertRaises(ValueError):
                self.cls(group, experts, capacity, 16, 3584, 3072,
                         enable_gin=True, gin_active_fast_path=True,
                         gin_outbox_depth=64, gin_bulk_combine=True,
                         gin_direct_dispatch=True)


class EP8CompiledHostLayoutContract(unittest.TestCase):
    def compile_run(self, source):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("requires a host C++17 compiler")
        with tempfile.TemporaryDirectory(prefix="megamoe-ep8-layout-") as temporary:
            binary = str(Path(temporary) / "contract")
            run = subprocess.run([compiler, "-std=c++17", "-O2", "-x", "c++", "-",
                                  "-o", binary], input=source, text=True,
                                 capture_output=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            run = subprocess.run([binary], capture_output=True, text=True, timeout=20)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_actual_host_size_predicates(self):
        source = HOST.read_text()
        checks = source[source.index("    DG_HOST_ASSERT(num_experts % num_ranks"):
                        source.index("    // Ring capacity:")]
        self.compile_run(r'''
#include <cassert>
#include <string>
#define DG_HOST_ASSERT(x) if (!(x)) throw 1
bool valid(int num_ranks, int num_experts, int num_max_tokens_per_rank,
           bool gin_bulk_combine, bool gin_direct_dispatch) {
    const int num_topk=16, hidden=3584, intermediate_hidden=3072;
    const int num_shared_experts=0, gin_outbox_depth=64;
    const bool enable_gin=true, gin_active_fast_path=true;
    const std::string activation="swiglu";
    try {
''' + checks + r'''
        return true;
    } catch (int) { return false; }
}
int main() {
    for (int world : {8,16}) {
        assert(valid(world,world*56,384,true,true));
        assert(valid(world,world*56,384,true,false));
        assert(valid(world,world*56,384,false,true));
        assert(!valid(world,world*56,383,true,true));
    }
    assert(!valid(8,896,384,true,true));
    assert(!valid(16,448,384,true,true));
    assert(!valid(4,224,384,true,true));
}
''')

    def test_actual_context_topology_guards_and_optional_binding(self):
        source = GIN.read_text()
        begin = source.index("        if ((world_size_ != 8")
        checks = source[begin:source.index("        validate_launch_tuning(", begin)]
        self.compile_run(r'''
#include <cassert>
#include <stdexcept>
constexpr int kDefaultContextCount=9, kDefaultWorldBarrierCount=4;
bool valid(int world_size_, int rank_, int expected_lsa_size_, int context_count_=9) {
    const int queue_depth_=64, world_barrier_count_=4;
    try {
''' + checks + r'''
        return true;
    } catch (const std::invalid_argument&) { return false; }
}
int main() {
    assert(valid(8,0,4)); assert(valid(8,7,4));
    assert(valid(16,0,8)); assert(valid(16,15,8));
    assert(!valid(8,0,8)); assert(!valid(16,0,4));
    assert(!valid(12,0,6)); assert(!valid(8,8,4));
    assert(!valid(8,-1,4)); assert(!valid(8,0,4,8));
}
''')
        binding = source[source.index('        "create_megamoe_gin_context",'):]
        self.assertIn('py::arg("expected_lsa_size") = py::none()', binding)
        self.assertIn("world_size / 2 : expected_lsa_size.cast<int>()", binding)
        self.assertLess(binding.index("expected_lsa_size.cast<int>()"),
                        binding.index("py::gil_scoped_release"))
        self.assertIn("fields[2] != peer % expected_lsa_size_", source)
        self.assertIn("fields[3] != expected_lsa_size_", source)

    def test_actual_jit_selects_requested_specializations_or_rejects(self):
        source = JIT.read_text()
        checks = source[source.index("    const int gin_local_ablation_stage ="):
                        source.index("    const int gin_dispatch_warp_scan_value =")]
        self.compile_run(r'''
#include <cassert>
#include <cstdint>
#include <optional>
#define DG_HOST_ASSERT(x) if (!(x)) throw 1
template<class T> T get_env(const char*, T value) { return value; }
namespace layout {
constexpr int kMegaMoeGinNumDispatchWarps=4;
constexpr int kMegaMoeGinDirectDispatchStorageBytes=57344;
}
struct Transport { int active_fast_path,bulk_combine,direct_dispatch,outbox_depth; };
int select(int num_ranks,int num_experts,int num_max_tokens_per_rank=384,
           int num_sms=152,bool enabled=true) {
    const int num_topk=16,hidden=3584,intermediate_hidden=3072,num_shared_experts=0;
    std::optional<Transport> gin_transport_opt;
    if (enabled) gin_transport_opt=Transport{1,1,1,64};
    try {
''' + checks + r'''
        return gin_active_fast_path+2*gin_bulk_combine+4*gin_direct_dispatch;
    } catch (int) { return -1; }
}
int main() {
    assert(select(8,448)==7); assert(select(16,896)==7);
    assert(select(8,448,384,128)==7); assert(select(16,896,384,128)==7);
    assert(select(8,448,384,127)==-1); assert(select(8,448,383)==-1);
    assert(select(8,896)==-1); assert(select(16,448)==-1);
    assert(select(4,224)==-1); assert(select(12,672)==-1);
    assert(select(8,448,384,152,false)==0);
    assert(select(16,896,384,152,false)==0);
}
''')

    def test_actual_layout_retains_outbox_and_max_peer_scratch(self):
        source = LAYOUT.read_text()
        constants = source[source.index("static constexpr int kNumCandidateBlockMs"):
                           source.index("// SF pool capacity:")]
        structs = source[source.index("struct Data {"):source.index("struct MegaMoEBuffer {")]
        self.compile_run(r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>
#define CUTLASS_HOST_DEVICE
#define CUTLASS_DEVICE
#define __CLION_IDE__
#define DG_UNIFIED_ASSERT(x) assert(x)
#define DG_DEVICE_ASSERT(x) assert(x)
namespace math {
template<class T> constexpr T constexpr_min(T a,T b) { return a < b ? a : b; }
template<class T> constexpr T constexpr_align(T a,T b) { return (a+b-1)/b*b; }
template<class T> T align(T a,T b) { return constexpr_align(a,b); }
template<class T=void> T* advance_ptr(void* p,uint64_t n) {
    return reinterpret_cast<T*>(static_cast<uint8_t*>(p)+n);
}
}
''' + constants + structs + r'''
int main() {
    static_assert(kMegaMoeGinNumDataContexts == 8);
    static_assert(kMegaMoeGinDirectDispatchNumPeers == 8);
    static_assert(kMegaMoeGinDirectDispatchStorageBytes == 57344);
    static_assert(get_mega_moe_gin_combine_overlap_scratch_bytes() == 2304);
    static_assert(kMegaMoeGinDirectReduceOrdinalBytes == 3072);
    std::vector<uint8_t> storage(128*1024*1024);
    for (uint32_t world : {8u,16u}) for (bool bulk : {false,true}) {
        MegaMoeGinWorkspace w(storage.data(),3584,world,world*56,384,16,152,8,64,bulk);
        const uint64_t outbox=64ull*192*3584*2;
        const uint64_t packets=world*(16ull+768*(16+3584*2));
        assert(w.combine_outbox_buffer.get_num_bytes()==outbox);
        assert(w.get_num_bytes()<=storage.size());
        assert(w.direct_dispatch_mirrors_fit());
        assert(w.combine_direct_reduce_alias_fits());
        auto* scratch=static_cast<uint8_t*>(w.scale_scratch_buffer.base);
        assert(reinterpret_cast<uint8_t*>(w.get_combine_direct_reduce_ordinal_ptr(767)+1)
               ==scratch+62720);
        assert(static_cast<uint8_t*>(w.get_direct_dispatch_packet_ptr(false,7))
               ==scratch+15*3584);
        if (!bulk) {
            assert(w.bulk_combine_packet_tail_buffer.get_num_bytes()==0);
            assert(w.bulk_combine_return_index_buffer.get_num_bytes()==0);
            continue;
        }
        const uint64_t reserved=std::max(outbox,packets);
        const auto base=static_cast<uint8_t*>(w.combine_outbox_buffer.base);
        assert(w.bulk_combine_packet_tail_buffer.get_num_bytes()==reserved-outbox);
        assert(w.bulk_combine_return_index_buffer.base==base+reserved);
        // EP16 retains the exact old packet-tail and return-index offsets.
        if (world==16) assert(reserved==packets);
        else assert(reserved==outbox && packets<outbox);
        for (uint32_t peer=0;peer<world/2;++peer) for (bool send : {false,true}) {
            auto* packet=static_cast<uint8_t*>(w.get_bulk_combine_packet_ptr(send,peer));
            assert(packet==base+((send?0:world/2)+peer)*uint64_t(w.bulk_packet_bytes));
            auto* end=static_cast<uint8_t*>(w.get_bulk_combine_record_payload_ptr(send,peer,767))
                +3584*2;
            assert(end==packet+w.bulk_packet_bytes);
            assert(end<=base+reserved);
        }
        auto* last=static_cast<uint8_t*>(w.get_combine_outbox_row_ptr(63,191));
        assert(last+3584*2==base+outbox);
        assert(last+3584*2<=static_cast<uint8_t*>(w.bulk_combine_return_index_buffer.base));
    }
}
''')


if __name__ == "__main__":
    unittest.main()
