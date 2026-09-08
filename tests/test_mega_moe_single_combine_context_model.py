"""CPU-only scope/lifecycle contracts for the optional context-1 combine.

This model is not a device-ordering proof. The GPU gates must still exercise
same-allocation flag changes, graph payload changes, and T48/T64 transitions.
"""

from pathlib import Path
import random
import unittest


ROOT = Path(__file__).resolve().parents[1]
FLAG = "DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT"


def context_scope(enabled, remote, bulk, direct):
    if not remote:
        return ()
    return (1,) if enabled and bulk and direct else tuple(range(9))


class BarrierSequences:
    """Signals/shadows for one barrier index, independently stored per context."""

    def __init__(self, ranks=16):
        self.ranks = ranks
        self.signal = [[[0] * ranks for _ in range(9)] for _ in range(ranks)]
        self.shadow = [[[0] * ranks for _ in range(9)] for _ in range(ranks)]

    def complete(self, scope_by_rank):
        if any(scope != scope_by_rank[0] for scope in scope_by_rank):
            raise ValueError("world participants must agree before GPU launch")
        for source in range(self.ranks):
            for context in scope_by_rank[source]:
                for destination in range(self.ranks):
                    self.signal[destination][context][source] += 1
        for receiver in range(self.ranks):
            for context in scope_by_rank[receiver]:
                for sender in range(self.ranks):
                    self.shadow[receiver][context][sender] += 1
                    if self.signal[receiver][context][sender] != self.shadow[receiver][context][sender]:
                        raise AssertionError("signal/shadow drift")


class TestSingleCombineContext(unittest.TestCase):
    def test_uniform_scope_and_fallback(self):
        for enabled in (False, True):
            self.assertEqual(context_scope(enabled, False, True, True), ())
            self.assertEqual(context_scope(enabled, True, False, True), tuple(range(9)))
            self.assertEqual(context_scope(enabled, True, True, False), tuple(range(9)))
        self.assertEqual(context_scope(True, True, True, True), (1,))
        self.assertEqual(context_scope(False, True, True, True), tuple(range(9)))

    def test_same_allocation_off_on_t64_local_on_off_sequences(self):
        barriers = BarrierSequences()
        scopes = [context_scope(*flags) for flags in (
            (False, True, True, True), (True, True, True, True),
            (True, True, False, False), (True, False, True, True),
            (True, True, True, True), (False, True, True, True))]
        for scope in scopes:
            barriers.complete([scope] * 16)
        self.assertEqual(barriers.signal, barriers.shadow)
        self.assertEqual(barriers.shadow[0][1][8], 5)
        self.assertEqual(barriers.shadow[0][8][8], 3)

    def test_random_uniform_replay_lifetimes(self):
        rng = random.Random(20260906)
        barriers = BarrierSequences()
        for _ in range(100):
            scope = context_scope(*(bool(rng.randrange(2)) for _ in range(4)))
            barriers.complete([scope] * 16)
        self.assertEqual(barriers.signal, barriers.shadow)

    def test_nonuniform_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "before GPU launch"):
            BarrierSequences().complete([(1,)] * 8 + [tuple(range(9))] * 8)

    def test_single_context_retains_distinct_peer_qps_and_trace_slots(self):
        qps = {(1, peer) for peer in range(8, 16)}
        self.assertEqual(len(qps), 8)
        for base in (64, 72, 80):
            self.assertEqual(len({base + lane for lane in range(8)}), 8)
        helper = (ROOT / "deep_gemm/include/deep_gemm/comm/mega_moe_gin.cuh").read_text()
        packet = helper.split("void mega_moe_gin_put_bulk_combine_packet(", 1)[1].split("\n}", 1)[0]
        for base in (64, 72, 80):
            self.assertIn(f"{base}u + diagnostic_peer_lane", packet)
            self.assertNotIn(f"{base}u + context_stripe", packet)
        self.assertIn("transport, peer, context_stripe, &request", packet)
        self.assertIn("transport, context_stripe, request", packet)

    def test_public_single_context_put_barrier_and_unmodified_mapping(self):
        helper = (ROOT / "deep_gemm/include/deep_gemm/comm/mega_moe_gin.cuh").read_text()
        single = helper.split("void mega_moe_gin_world_barrier_single_combine_context(", 1)[1].split("\n}", 1)[0]
        self.assertIn("transport.data_context(0)", single)
        self.assertIn("NCCL_GIN_RESOURCE_SHARING_GPU", single)
        self.assertIn("ncclGinFenceLevel::Put", single)
        self.assertNotIn("ncclGinAllContexts", single)
        self.assertIn("return 1u + stripe % static_cast<uint32_t>(dev_comm.lsaSize)", helper)

    def test_flag_default_off_and_jit_keyed_with_isolated_experiment_guards(self):
        kernel = (ROOT / "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh").read_text()
        host = (ROOT / "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp").read_text()
        self.assertIn(f"#define {FLAG} 0", kernel)
        self.assertIn(f"#define {FLAG} {{}}", host)
        self.assertIn(f'get_env<std::string>("{FLAG}", "0")', host)
        self.assertIn(".gin_single_combine_context = gin_single_combine_context", host)
        self.assertIn('(gin_bulk_combine and gin_direct_dispatch)', host)
        self.assertIn('get_env<std::string>("DG_MEGAMOE_GIN_COMBINE_EXPERTS_PER_WAVE", "0") == "0"', host)
        self.assertIn('get_env<std::string>("DG_MEGAMOE_GIN_COMBINE_BARRIER_WARPS", "1") == "1"', host)
        self.assertIn("use_single_combine_context = use_gin_bulk_combine and\n                    use_gin_direct_dispatch_this_launch()", kernel)
        self.assertEqual(kernel.count("world_barrier_single_combine_context("), 1)
        self.assertIn("/*diagnostic_peer_lane=*/ lane_idx", kernel)

    def test_host_allocation_still_requires_nine_contexts(self):
        api = (ROOT / "csrc/apis/mega_gin.hpp").read_text()
        self.assertNotIn(FLAG, api)
        self.assertTrue("kDefaultContextCount = 9" in api)
        self.assertTrue("context_count_ < kDefaultContextCount" in api)

    def test_retired_paths_have_no_device_implementation_or_jit_fields(self):
        kernel = (ROOT / "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh").read_text()
        helper = (ROOT / "deep_gemm/include/deep_gemm/comm/mega_moe_gin.cuh").read_text()
        host = (ROOT / "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp").read_text()
        for token in ("kMegaMoeGinCombineExpertsPerWave", "kMegaMoeGinCombineBarrierWarps",
                      "mega_moe_gin_publish_expert_ready", "mega_moe_gin_load_expert_ready",
                      "ncclCoopWarpSpan", "world_barrier_all_contexts_coop",
                      "kGinCombineCoopBarrierId"):
            self.assertFalse(token in kernel or token in helper, token)
        for token in ("int gin_combine_experts_per_wave;", "int gin_combine_barrier_warps;",
                      "#define DG_MEGAMOE_GIN_COMBINE_EXPERTS_PER_WAVE {}",
                      "#define DG_MEGAMOE_GIN_COMBINE_BARRIER_WARPS {}"):
            self.assertNotIn(token, host)
        self.assertIn('#error "Expert-wave combine is not part', kernel)
        self.assertIn('#error "Cooperative combine barriers are not part', kernel)


if __name__ == "__main__":
    unittest.main()
