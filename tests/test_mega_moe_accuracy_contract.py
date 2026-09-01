import unittest
from types import SimpleNamespace

import test_mega_moe_accuracy as accuracy


def _args(**overrides):
    values = {
        "num_experts": 896,
        "num_topk": 16,
        "hidden": 3584,
        "intermediate_hidden": 3072,
        "mma_type": "fp8xfp4",
        "num_tokens": 24,
        "num_max_tokens_per_rank": 384,
        "num_shared_experts": 0,
        "require_gin": True,
        "gin_completion_batch": 1,
        "gin_combine_chunk_bytes": 7168,
        "gin_outbox_depth": 8,
        "gin_queue_depth": 64,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeTensor:
    def numel(self):
        return 4096

    def element_size(self):
        return 1


class _FakeContext:
    active = True
    gin_type_string = "gdaki"
    buffer_bytes = 4096

    def __init__(self, **overrides):
        self.snapshot = {
            "enabled": 1,
            "rank": 0,
            "world_size": 16,
            "lsa_rank": 0,
            "lsa_size": 8,
            "context_count": 9,
            "requested_context_count": 9,
            "connection_count": 8,
            "queue_depth": 64,
            "world_barrier_count": 3,
            "completion_batch": 1,
            "combine_chunk_bytes": 7168,
            "outbox_depth": 8,
            "window": 1234,
            "dev_comm_bytes": 256,
        }
        self.snapshot.update(overrides)

    def launch_descriptor_snapshot(self):
        return dict(self.snapshot)


class _FakeBuffer:
    def __init__(
        self, context=None, events=None, destroy_error=None, abort_error=None
    ):
        self.buffer = _FakeTensor()
        self.gin_context = context
        self.events = events if events is not None else []
        self.destroy_error = destroy_error
        self.abort_error = abort_error

    @property
    def gin_enabled(self):
        return self.gin_context is not None and self.gin_context.active

    def destroy(self):
        self.events.append("buffer.destroy")
        if self.destroy_error is not None:
            raise self.destroy_error

    def abort(self):
        self.events.append("buffer.abort")
        if self.abort_error is not None:
            raise self.abort_error


class _FakeDist:
    def __init__(self, events, initialized=True):
        self.events = events
        self.initialized = initialized

    def is_initialized(self):
        self.events.append("dist.is_initialized")
        return self.initialized

    def barrier(self):
        self.events.append("dist.barrier")

    def destroy_process_group(self):
        self.events.append("dist.destroy_process_group")


class TestMegaMoeAccuracyGinContract(unittest.TestCase):
    def test_required_gin_accepts_only_integrated_fp8_kernel(self):
        with self.assertRaisesRegex(ValueError, "BF16 MegaMoE launch"):
            accuracy._validate_args(_args(mma_type="bf16xbf16"), 16)

    def test_required_gin_accepts_only_initial_2x8_target(self):
        with self.assertRaisesRegex(ValueError, "exactly 16 ranks"):
            accuracy._validate_args(_args(), 8)

    def test_fused_accuracy_rejects_unimplemented_completion_batch(self):
        with self.assertRaisesRegex(ValueError, "standalone probe"):
            accuracy._validate_args(_args(gin_completion_batch=4), 16)

    def test_combine_chunk_must_tile_a_full_output_row(self):
        with self.assertRaisesRegex(ValueError, "must divide one BF16 output row"):
            accuracy._validate_args(
                _args(hidden=512, intermediate_hidden=512,
                      gin_combine_chunk_bytes=1792),
                16,
            )

    def test_transport_evidence_checks_and_reports_exact_tuning(self):
        evidence = accuracy._gin_transport_evidence(
            _FakeBuffer(_FakeContext()),
            _args(),
            rank=0,
            world_size=16,
            hostnames=["host-a"] * 8 + ["host-b"] * 8,
        )

        self.assertEqual(evidence["requested"], "gin")
        self.assertTrue(evidence["cross_host_payload_routes"])
        self.assertEqual(evidence["gin_type"], "gdaki")
        self.assertEqual(
            evidence["launch_descriptor"]["combine_chunk_bytes"], 7168
        )
        self.assertEqual(evidence["registered_buffer_bytes"], 4096)

    def test_transport_evidence_rejects_tuning_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "completion_batch=4"):
            accuracy._gin_transport_evidence(
                _FakeBuffer(_FakeContext(completion_batch=4)),
                _args(),
                rank=0,
                world_size=16,
                hostnames=["host-a"] * 8 + ["host-b"] * 8,
            )

    def test_transport_evidence_requires_cross_host_payload(self):
        with self.assertRaisesRegex(RuntimeError, "all ranks are on one host"):
            accuracy._gin_transport_evidence(
                _FakeBuffer(_FakeContext()),
                _args(),
                rank=0,
                world_size=16,
                hostnames=["host-a"] * 16,
            )

    def test_transport_evidence_requires_contiguous_2x8_host_ranks(self):
        with self.assertRaisesRegex(RuntimeError, "contiguous 2x8 rank placement"):
            accuracy._gin_transport_evidence(
                _FakeBuffer(_FakeContext()),
                _args(),
                rank=0,
                world_size=16,
                hostnames=["host-a", "host-b"] * 8,
            )


class TestMegaMoeAccuracyFailureSafeTeardown(unittest.TestCase):
    def test_synchronized_success_collectively_destroys_in_order(self):
        events = []
        buffer = _FakeBuffer(events=events)
        dist = _FakeDist(events)

        synchronized_success = accuracy._synchronize_worker_success(dist)
        accuracy._teardown_worker(buffer, dist, synchronized_success)

        self.assertEqual(
            events,
            [
                "dist.barrier",
                "buffer.destroy",
                "dist.is_initialized",
                "dist.destroy_process_group",
            ],
        )

    def test_failure_aborts_rank_locally_and_skips_distributed_teardown(self):
        events = []
        buffer = _FakeBuffer(events=events)
        dist = _FakeDist(events)

        accuracy._teardown_worker(buffer, dist, synchronized_success=False)

        self.assertEqual(events, ["buffer.abort"])

    def test_failure_before_buffer_creation_skips_distributed_teardown(self):
        events = []
        dist = _FakeDist(events)

        accuracy._teardown_worker(None, dist, synchronized_success=False)

        self.assertEqual(events, [])

    def test_rank_local_abort_failure_does_not_mask_worker_failure(self):
        events = []
        buffer = _FakeBuffer(events=events, abort_error=RuntimeError("abort failed"))
        dist = _FakeDist(events)

        accuracy._teardown_worker(buffer, dist, synchronized_success=False)

        self.assertEqual(events, ["buffer.abort"])

    def test_collective_buffer_destroy_failure_aborts_and_skips_process_group(self):
        events = []
        buffer = _FakeBuffer(events=events, destroy_error=RuntimeError("destroy failed"))
        dist = _FakeDist(events)

        with self.assertRaisesRegex(RuntimeError, "destroy failed"):
            accuracy._teardown_worker(buffer, dist, synchronized_success=True)

        self.assertEqual(events, ["buffer.destroy", "buffer.abort"])

    def test_success_with_uninitialized_process_group_does_not_destroy_it(self):
        events = []
        buffer = _FakeBuffer(events=events)
        dist = _FakeDist(events, initialized=False)

        accuracy._teardown_worker(buffer, dist, synchronized_success=True)

        self.assertEqual(events, ["buffer.destroy", "dist.is_initialized"])


if __name__ == "__main__":
    unittest.main()
