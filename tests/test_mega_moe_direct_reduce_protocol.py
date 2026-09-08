"""Direct-reducer CPU models/source contracts, not CUDA runtime evidence."""

from pathlib import Path
import random
import unittest

from mega_moe_gin_direct_reduce_protocol import (
    DirectReduceGeneration, PreloadedTokenRows, ProtocolError, bf16, ordered_reduce,
)
import test_mega_moe_combine_overlap_helper as helper_contract


ROOT = Path(__file__).resolve().parents[1]


def fixture(tokens, seed=0, kind="mixed"):
    rng = random.Random(seed)
    ids, values = [], []
    for token in range(tokens):
        row, payload = [], []
        for slot in range(16):
            if kind == "masked":
                expert = -1
            elif kind == "hot":
                expert = 8 * 56 + slot
            elif kind == "local":
                expert = slot
            else:
                expert = rng.randrange(-1, 896)
            row.append(expert)
            payload.append(bf16(rng.uniform(-8, 8)))
        ids.append(row)
        values.append(payload)
    return ids, values


def complete(model):
    model.acquire_world_put()
    model.grid2()
    model.check_packet_counts()
    if model.direct:
        model.bridge_async_proxy()
    if not model.direct:
        model.scatter_fallback()
        model.second_handoff()
        model.dispatch_cleanup()
    result = model.reduce()
    if model.direct:
        model.second_handoff()
        model.dispatch_cleanup()
    model.retire()
    return result


def baseline(ids, values, shared=None):
    return tuple(ordered_reduce([value for expert, value in zip(row, payload) if expert >= 0] +
                                ([] if shared is None else [shared[token]]))
                 for token, (row, payload) in enumerate(zip(ids, values)))


class DirectReduceModelTests(unittest.TestCase):
    def test_random_packet_permutation_inverse_preserves_exact_topk_order(self):
        rng = random.Random(20260912)
        for trial in range(48):
            ids, values = fixture((trial % 3 + 2) * 8, seed=trial)
            source = trial % 16
            orders = {}
            for owner in range(16):
                routes = [token * 16 + slot for token, row in enumerate(ids)
                          for slot, expert in enumerate(row) if expert >= 0 and expert // 56 == owner]
                rng.shuffle(routes)
                orders[owner] = routes
            model = DirectReduceGeneration(source)
            model.begin(0, ids, values, pack_order=orders)
            self.assertEqual(complete(model), baseline(ids, values))
            expected = {token * 16 + slot for token, row in enumerate(ids)
                        for slot, expert in enumerate(row)
                        if expert >= 0 and expert // (56 * 8) != source // 8}
            self.assertEqual(model.inverse_reads, expected)

    def test_local_masked_shared_never_read_stale_inverse(self):
        ids, values = fixture(48, kind="hot")
        model = DirectReduceGeneration()
        model.begin(0, ids, values)
        complete(model)
        stale = dict(model.inverse)
        ids, values = fixture(48, kind="local")
        for token in range(48):
            ids[token][token % 16] = -1
        shared = [bf16(0.25 * token) for token in range(48)]
        model.begin(1, ids, values, shared=shared)
        self.assertEqual(complete(model), baseline(ids, values, shared))
        self.assertEqual(model.inverse, stale)
        self.assertFalse(model.inverse_reads)

    def test_masked_output_zero_and_empty_packets(self):
        model = DirectReduceGeneration()
        ids, values = fixture(32, kind="masked")
        model.begin(0, ids, values)
        self.assertEqual(complete(model), (0.0,) * 32)
        self.assertFalse(model.inverse_reads)
        self.assertTrue(all(not packet for packet in model.packets.values()))

    def test_hot_owner_all_768_assignments_are_retained(self):
        ids, values = fixture(48, kind="hot")
        model = DirectReduceGeneration()
        model.begin(0, ids, values)
        self.assertEqual(len(model.packets[8]), 768)
        self.assertEqual(set(model.inverse.values()), set(range(768)))
        self.assertEqual(complete(model), baseline(ids, values))

    def test_all_sources_hot_owner_counts_are_not_balanced_average_hint(self):
        remote_records, assignments = 0, 0
        for source in range(16):
            ids, values = fixture(48, kind="hot", seed=source)
            model = DirectReduceGeneration(source)
            model.begin(0, ids, values)
            assignments += len(ids) * 16
            remote_records += sum(map(len, model.packets.values()))
            self.assertEqual(complete(model), baseline(ids, values))
        self.assertEqual(assignments, 12288)
        self.assertEqual(remote_records, 6144)

    def test_asymmetric_source_owners_and_changed_routes_same_storage(self):
        model = DirectReduceGeneration(source=8)
        for generation, seed in enumerate((11, 27, 11)):
            ids, values = fixture(40, seed=seed)
            model.begin(generation, ids, values)
            self.assertEqual(complete(model), baseline(ids, values))

    def test_t48_t64_t48_and_exact_fit_fallback_have_no_truncation(self):
        model = DirectReduceGeneration()
        for generation, (tokens, available, direct) in enumerate(
                ((48, 62720, True), (64, 65536, False), (48, 62719, False),
                 (48, 62720, True))):
            ids, values = fixture(tokens, kind="hot", seed=generation)
            model.begin(generation, ids, values, available_scratch=available)
            self.assertEqual(model.direct, direct)
            self.assertEqual(len(model.packets[8]), tokens * 16)
            self.assertEqual(complete(model), baseline(ids, values))
            self.assertEqual(bool(model.inverse_reads), direct)

    def test_world_non_direct_and_disabled_paths_keep_scatter(self):
        for settings in ({"enabled": False}, {"direct_bulk": False}):
            model = DirectReduceGeneration()
            ids, values = fixture(32)
            model.begin(0, ids, values, **settings)
            self.assertFalse(model.direct)
            self.assertEqual(complete(model), baseline(ids, values))
            self.assertTrue(model.scatter_done)

    def test_ordinal_bounds_and_destination_assertions_detect_stale_map(self):
        for bad in (1000, 1):
            model = DirectReduceGeneration()
            ids, values = fixture(32, kind="hot")
            model.begin(0, ids, values)
            model.acquire_world_put()
            model.grid2()
            model.check_packet_counts()
            model.bridge_async_proxy()
            model.inverse[0] = bad
            with self.assertRaises(ProtocolError):
                model.begin_read(0, 0)

    def test_visibility_count_audit_and_retirement_are_distinct(self):
        model = DirectReduceGeneration()
        ids, values = fixture(32, kind="hot")
        model.begin(0, ids, values)
        with self.assertRaises(ProtocolError):
            model.grid2()
        with self.assertRaises(ProtocolError):
            model.begin_read(0, 0)
        model.acquire_world_put()
        model.grid2()
        model.check_packet_counts()
        model.bridge_async_proxy()
        model.begin_read(0, 0)
        with self.assertRaises(ProtocolError):
            model.second_handoff()
        with self.assertRaises(ProtocolError):
            model.dispatch_cleanup()
        with self.assertRaises(ProtocolError):
            model.begin(1, ids, values)

    def test_received_count_check_catches_missing_or_duplicate_record(self):
        ids, values = fixture(32, kind="hot")
        model = DirectReduceGeneration()
        model.begin(0, ids, values)
        model.packets[8].pop()
        model.acquire_world_put()
        model.grid2()
        with self.assertRaises(ProtocolError):
            model.check_packet_counts()

    def test_async_proxy_bridge_does_not_replace_target_visibility(self):
        model = DirectReduceGeneration()
        ids, values = fixture(32, kind="hot")
        model.begin(0, ids, values)
        with self.assertRaises(ProtocolError):
            model.bridge_async_proxy()
        model.acquire_world_put()
        model.grid2()
        model.check_packet_counts()
        with self.assertRaises(ProtocolError):
            model.begin_read(0, 0)
        model.bridge_async_proxy()
        self.assertEqual(model.begin_read(0, 0), values[0][0])

    def test_fallback_can_reduce_after_dispatch_count_cleanup(self):
        ids, values = fixture(64, kind="hot")
        model = DirectReduceGeneration()
        model.begin(0, ids, values)
        model.acquire_world_put()
        model.grid2()
        model.check_packet_counts()
        model.scatter_fallback()
        model.second_handoff()
        model.dispatch_cleanup()
        self.assertFalse(model.expected_counts)
        with self.assertRaises(ProtocolError):
            model.check_packet_counts()
        self.assertEqual(model.reduce(), baseline(ids, values))
        model.retire()

    def test_sum_is_original_slot_order_not_packet_or_owner_order(self):
        ids = [[-1] * 16]
        values = [[0.0] * 16]
        ids[0][:3] = [448, 0, 449]
        values[0][:3] = [bf16(2 ** 25), bf16(-2 ** 25), bf16(1.0)]
        model = DirectReduceGeneration()
        model.begin(0, ids, values)
        self.assertEqual(complete(model), (1.0,))
        self.assertEqual(ordered_reduce([values[0][0], values[0][2], values[0][1]]), 0.0)


class DirectPreloadModelTests(unittest.TestCase):
    @staticmethod
    def ready(model):
        model.acquire_world_put()
        model.grid2()
        model.check_packet_counts()
        model.bridge_async_proxy()

    def test_all_lanes_preload_once_and_both_chunks_keep_original_slot_order(self):
        for source in (0, 7, 8, 15):
            model = DirectReduceGeneration(source)
            ids, values = fixture(48, seed=source + 100)
            model.begin(0, ids, values)
            self.ready(model)
            for token, row in enumerate(ids):
                cached = PreloadedTokenRows(model, token)
                remote_count = sum(expert >= 0 and expert // 448 != source // 8
                                   for expert in row)
                self.assertEqual(len(cached.metadata_reads), 3 * remote_count)
                metadata = tuple(cached.metadata_reads)
                slots = [slot for slot, expert in enumerate(row) if expert >= 0]
                for chunk_offset in (0, 3584):
                    result = ordered_reduce([cached.read(slot, chunk_offset) for slot in slots])
                    self.assertEqual(result, baseline([row], [values[token]])[0])
                self.assertEqual(tuple(cached.metadata_reads), metadata)
                self.assertEqual(cached.chunk_reads,
                                 [(slot, offset) for offset in (0, 3584) for slot in slots])

    def test_preload_does_not_reorder_cancelling_remote_local_remote_sum(self):
        ids, values = [[-1] * 16], [[0.0] * 16]
        ids[0][:3], values[0][:3] = [448, 0, 449], [2 ** 25, -2 ** 25, 1.0]
        model = DirectReduceGeneration()
        model.begin(0, ids, values)
        self.ready(model)
        cached = PreloadedTokenRows(model, 0)
        for offset in (0, 3584):
            self.assertEqual(ordered_reduce([cached.read(slot, offset) for slot in (0, 1, 2)]), 1.0)
            self.assertEqual(ordered_reduce([cached.read(slot, offset) for slot in (0, 2, 1)]), 0.0)

    def test_local_masked_shared_lanes_never_consult_stale_inverse(self):
        ids, values = fixture(32, kind="local")
        ids[0][3] = -1
        model = DirectReduceGeneration()
        model.inverse = {slot: 99999 for slot in range(17)}
        model.begin(0, ids, values, shared=[0.25] * 32)
        self.ready(model)
        cached = PreloadedTokenRows(model, 0)
        self.assertEqual(cached.metadata_reads, [])
        self.assertFalse(model.inverse_reads)
        self.assertEqual(cached.read(16), 0.25)
        self.assertEqual(cached.read(1), values[0][1])
        for slot in (3, 17, 31):
            with self.assertRaisesRegex(ProtocolError, "masked/inactive"):
                cached.read(slot)

    def test_invalid_remote_ordinal_and_wrong_destination_fail_during_preload(self):
        for bad in (768, 1):
            model = DirectReduceGeneration()
            ids, values = fixture(48, kind="hot")
            model.begin(0, ids, values)
            self.ready(model)
            model.inverse[0] = bad
            with self.assertRaises(ProtocolError):
                PreloadedTokenRows(model, 0)

    def test_preload_requires_visibility_and_cannot_survive_generation_reuse(self):
        model = DirectReduceGeneration()
        ids, values = fixture(32, kind="hot")
        model.begin(0, ids, values)
        with self.assertRaises(ProtocolError):
            PreloadedTokenRows(model, 0)
        self.ready(model)
        cached = PreloadedTokenRows(model, 0)
        self.assertEqual(cached.read(0), values[0][0])
        complete(model)
        with self.assertRaises(ProtocolError):
            cached.read(0)
        ids1, values1 = fixture(32, kind="hot", seed=7)
        model.begin(1, ids1, values1)
        self.ready(model)
        with self.assertRaises(ProtocolError):
            cached.read(0)
        fresh = PreloadedTokenRows(model, 0)
        self.assertEqual(fresh.read(0), values1[0][0])

    def test_t48_t64_t48_only_eligible_generation_preloads(self):
        model = DirectReduceGeneration()
        for generation, tokens in enumerate((48, 64, 48)):
            ids, values = fixture(tokens, kind="hot", seed=generation)
            model.begin(generation, ids, values)
            model.acquire_world_put()
            model.grid2()
            model.check_packet_counts()
            if model.direct:
                model.bridge_async_proxy()
                cached = PreloadedTokenRows(model, tokens - 1)
                self.assertEqual(cached.read(15, 3584), values[-1][15])
            else:
                with self.assertRaises(ProtocolError):
                    PreloadedTokenRows(model, 0)
            self.assertEqual(complete(model), baseline(ids, values))


class DirectReduceSourceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel = (ROOT / "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh").read_text()
        cls.layout = (ROOT / "deep_gemm/include/deep_gemm/layout/mega_moe.cuh").read_text()

    def test_exact_append_only_layout_and_bounds(self):
        source = self.layout
        self.assertIn("kMegaMoeGinDirectReduceOrdinalBytes", source)
        start = source.index("bool combine_direct_reduce_alias_fits() const")
        end = source.index("bool direct_dispatch_mirrors_fit() const", start)
        body = source[start:end]
        self.assertIn("kMegaMoeGinDirectDispatchStorageBytes +", body)
        self.assertIn("get_mega_moe_gin_combine_overlap_scratch_bytes() +", body)
        self.assertIn("kMegaMoeGinDirectReduceOrdinalBytes", body)
        self.assertIn("token_topk_idx < kMegaMoeGinDirectDispatchMaxRoutes", body)
        self.assertIn("+ token_topk_idx", body)
        self.assertEqual(DirectReduceGeneration.scratch_bytes, 62720)

    def test_compiled_actual_layout_exact_fit_and_all_768_ordinal_addresses(self):
        source = self.layout
        constants = source[source.index("static constexpr int kNumCandidateBlockMs"):
                           source.index("// Pool capacity")]
        methods = "\n".join(helper_contract.function_source(source, name) for name in (
            "combine_direct_reduce_alias_fits", "get_combine_direct_reduce_ordinal_ptr"))
        helper_contract.CombineOverlapHelperContract.compile_and_run(self, r'''
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
    static_assert(kMegaMoeGinDirectDispatchMaxRoutes == 768);
    static_assert(kMegaMoeGinDirectReduceOrdinalBytes == 3072);
    static_assert(get_mega_moe_gin_combine_overlap_scratch_bytes() == 2304);
    alignas(128) std::array<uint8_t, 62720> storage{};
    Workspace workspace{{storage.data(), storage.size() - 1}};
    assert(!workspace.combine_direct_reduce_alias_fits());
    workspace.scale_scratch_buffer.bytes = storage.size();
    assert(workspace.combine_direct_reduce_alias_fits());
    for (uint32_t index = 0; index < 768; ++index) {
        auto* ordinal = workspace.get_combine_direct_reduce_ordinal_ptr(index);
        assert(reinterpret_cast<uint8_t*>(ordinal) == storage.data() + 59648 + 4 * index);
        *ordinal = 767 - index;
    }
    assert(reinterpret_cast<uint8_t*>(workspace.get_combine_direct_reduce_ordinal_ptr(767) + 1)
           == storage.data() + storage.size());
    for (uint32_t index = 0; index < 768; ++index)
        assert(*workspace.get_combine_direct_reduce_ordinal_ptr(index) == 767 - index);
}
''')

    def test_all_three_actual_pack_paths_write_inverse_before_existing_fence(self):
        source = self.kernel
        occurrences, start = [], 0
        while True:
            start = source.find("if (prepare_gin_direct_reduce_ordinals) {", start)
            if start < 0:
                break
            fence = source.index("__threadfence_system();", start)
            body = source[start:fence]
            self.assertIn("token_topk_idx < num_tokens * kNumTopk", body)
            self.assertIn("get_combine_direct_reduce_ordinal_ptr(", body)
            self.assertRegex(body, r"token_topk_idx\) = route_(?:prefix|count) \+ slot;")
            occurrences.append(start)
            start += 1
        self.assertEqual(len(occurrences), 3)

    def test_rank_local_fit_and_cached_existing_eligibility(self):
        source = self.kernel
        begin = source.index("const bool prepare_gin_direct_reduce_ordinals")
        end = source.index("// SF and its buffer configs", begin)
        body = source[begin:end]
        self.assertIn("if constexpr (kUseGin and kMegaMoeGinCombineOverlap)", body)
        self.assertIn("num_tokens <= layout::kMegaMoeGinDirectDispatchMaxTokens", body)
        self.assertIn("combine_direct_reduce_alias_fits()", body)
        self.assertRegex(source, r"const bool use_gin_direct_reduce =\s*"
                         r"prepare_gin_direct_reduce_ordinals and use_gin_combine_overlap;")

    def test_count_audit_stays_before_scatter_guard_and_grid3_inside(self):
        source = self.kernel
        audit = source.index("DG_DEVICE_ASSERT(received_count == expected_count);")
        guard = source.index("if (not use_gin_direct_reduce) {", audit)
        grid3 = source.index("DG_GIN_TRACE_IF(epilogue_warp_idx == 0 and lane_idx == 0, 60);", guard)
        early_handoff = source.index("if (not use_gin_direct_reduce)\n", grid3)
        self.assertLess(audit, guard)
        self.assertLess(guard, grid3)
        self.assertLess(grid3, early_handoff)

    def test_remote_only_inverse_and_fullwarp_gather_before_tma_issuer(self):
        source = self.kernel
        preload_start = source.index("uint64_t cached_combine_row_ptr = 0;")
        chunks = source.index("// Iterate all chunks", preload_start)
        preload = source[preload_start:chunks]
        self.assertIn("use_gin_direct_reduce and stored_topk_slot_idx >= 0", preload)
        self.assertLess(preload.index("if (lane_idx < kNumTopk)"),
                        preload.index("get_combine_direct_reduce_ordinal_ptr("))
        self.assertLess(preload.index("if (not gin_transport.is_same_lsa_peer(owner))"),
                        preload.index("get_combine_direct_reduce_ordinal_ptr("))
        self.assertIn("token_topk_idx = token_idx * kNumTopk + lane_idx", preload)
        self.assertIn("DG_DEVICE_ASSERT(stored_topk_slot_idx < kNumExperts)", preload)
        self.assertIn("DG_DEVICE_ASSERT(ordinal < received_count)", preload)
        self.assertIn("DG_DEVICE_ASSERT(destination == token_topk_idx)", preload)
        self.assertIn("get_bulk_combine_record_payload_ptr(", preload)
        self.assertIn("buffer.combine_token_buffer.get_rank_buffer(lane_idx)", preload)
        self.assertNotIn("cached_combine_row_ptr[", source)
        begin = source.index("const auto move_mask_and_load =")
        end = source.index("// Load the first selection", begin)
        body = source[begin:end]
        self.assertLess(body.index("selected_combine_row_ptr = __shfl_sync("),
                        body.index("if (cute::elect_one_sync())"))
        self.assertIn("static_cast<unsigned long long>(cached_combine_row_ptr)", body)
        self.assertIn("reinterpret_cast<void*>(selected_combine_row_ptr)", body)
        self.assertIn("chunk_byte_offset", body)
        self.assertNotIn("get_combine_direct_reduce_ordinal_ptr(", body)
        self.assertNotIn("get_bulk_combine_packet_count_ptr(", body)
        self.assertNotIn("get_bulk_combine_record_destination_ptr(", body)
        self.assertIn("buffer.combine_token_buffer.get_rank_buffer(slot_idx)", body)
        for forbidden in ("count_staging_buffer", "get_expert_send_count",
                          "get_bulk_combine_return_index"):
            self.assertNotIn(forbidden, preload + body)

    def test_one_pointer_per_lane_refreshed_per_token_before_chunks(self):
        source = self.kernel
        preload = source.index("uint64_t cached_combine_row_ptr = 0;")
        token_loop = source.rindex("for (uint32_t token_idx =", 0, preload)
        chunks = source.index("for (uint32_t chunk =", preload)
        mask = source.index("const uint32_t slot_idx = __ffs(mask) - 1;", chunks)
        gather = source.index("selected_combine_row_ptr = __shfl_sync(", mask)
        self.assertLess(token_loop, preload)
        self.assertLess(preload, chunks)
        self.assertLess(chunks, mask)
        self.assertLess(mask, gather)
        self.assertEqual(source.count("uint64_t cached_combine_row_ptr = 0;"), 1)
        self.assertNotIn("__shfl_sync", source[preload:chunks])

    def test_single_proxy_fence_only_for_direct_reader_after_visibility_before_tma(self):
        source = self.kernel
        fence = source.index('asm volatile("fence.proxy.async.global;" ::: "memory");')
        self.assertEqual(source.count('asm volatile("fence.proxy.async.global;" ::: "memory");'), 1)
        audit = source.index("DG_DEVICE_ASSERT(received_count == expected_count);")
        grid2 = source.rindex("comm::grid_sync<", 0, audit)
        reducer = source.index("const auto move_mask_and_load =")
        self.assertLess(grid2, audit)
        self.assertLess(audit, fence)
        self.assertLess(fence, reducer)
        self.assertIn("if (use_gin_direct_reduce)", source[audit:fence])
        self.assertNotIn("fence.proxy.async.global", source[reducer:])

    def test_deferred_existing_handoff_after_all_input_waits_not_new_barrier(self):
        source = self.kernel
        start = source.index("const auto move_mask_and_load =")
        end = source.index("// This kernel", start) if "// This kernel" in source[start:] else len(source)
        reducer = source[start:end]
        wait = reducer.index("combine_load_barriers[load_stage_idx]->wait(combine_phase);")
        trace = reducer.index("DG_GIN_TRACE_IF(lane_idx == 0, 88u + epilogue_warp_idx);")
        deferred = reducer.index("if (use_gin_direct_reduce) {", trace)
        handoff = reducer.index("ptx::sync_unaligned(", deferred)
        self.assertLess(wait, trace)
        self.assertLess(trace, deferred)
        self.assertEqual(reducer[deferred:].count("ptx::sync_unaligned("), 1)
        self.assertLess(reducer.index('asm volatile("" ::: "memory");', deferred), handoff)
        self.assertGreater(reducer.rindex('asm volatile("" ::: "memory");'), handoff)
        # Every newly selected load is either the initial load or the next
        # prefetch; the matching wait executes before the loop advances.
        self.assertIn("bool do_reduce = move_mask_and_load(load_stage_idx);", reducer)
        self.assertIn("do_reduce = move_mask_and_load(load_stage_idx ^ 1);", reducer)
        self.assertIn("ptx::accumulate(reduced[", reducer)
        self.assertIn("__float22bfloat162_rn", reducer)


if __name__ == "__main__":
    unittest.main()
