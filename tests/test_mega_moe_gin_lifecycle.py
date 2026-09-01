import inspect
import unittest
from unittest import mock

import torch

from deep_gemm import _C
from deep_gemm.mega import SymmBuffer, get_symm_buffer_for_mega_moe
import deep_gemm.mega as mega


class _FakeGroup:
    def __init__(self, rank=0, size=16, events=None):
        self._rank = rank
        self._size = size
        self._events = events

    def rank(self):
        return self._rank

    def size(self):
        return self._size

    def barrier(self):
        if self._events is not None:
            self._events.append('barrier')


class _FakeGinContext:
    active = True

    def __init__(self, events=None):
        self._events = events

    def destroy(self):
        if self._events is not None:
            self._events.append('context.destroy')
        self.active = False

    def abort(self):
        if self._events is not None:
            self._events.append('context.abort')
        self.active = False

    def _release_buffer_registration(self):
        if self._events is not None:
            self._events.append('context.release_registration')


class _FlakyGinContext(_FakeGinContext):
    def __init__(self, events=None):
        super().__init__(events)
        self.destroy_attempts = 0

    def destroy(self):
        self.destroy_attempts += 1
        if self.destroy_attempts == 1:
            if self._events is not None:
                self._events.append('context.destroy.failed')
            raise RuntimeError('injected teardown failure')
        super().destroy()


def _uninitialized_symm_buffer(rank=0):
    result = SymmBuffer.__new__(SymmBuffer)
    result.group = _FakeGroup(rank=rank)
    result.buffer = torch.empty(1, dtype=torch.uint8)
    result.handle = object()
    result.x = object()
    result.x_sf = object()
    result.num_experts = 896
    result.num_max_tokens_per_rank = 384
    result.num_topk = 16
    result.hidden = 3584
    result.intermediate_hidden = 3072
    result._gin_context = None
    result._gin_layout_enabled = True
    result.gin_completion_batch = 1
    result.gin_combine_chunk_bytes = 7168
    result.gin_outbox_depth = 8
    result.gin_queue_depth = 64
    return result


def _mirror_all_gather_object(output, local, group):
    output[:] = [local.copy() if isinstance(local, dict) else local
                 for _ in range(group.size())]


class TestMegaMoeGinLifecycle(unittest.TestCase):
    def setUp(self):
        self.all_gather_object = mock.patch.object(
            mega.dist, 'all_gather_object',
            side_effect=_mirror_all_gather_object).start()
        self.addCleanup(mock.patch.stopall)

    def test_default_api_is_opt_in(self):
        self.assertFalse(
            inspect.signature(SymmBuffer).parameters['enable_gin'].default)
        self.assertFalse(
            inspect.signature(get_symm_buffer_for_mega_moe)
            .parameters['enable_gin'].default)

    def test_build_info_and_disabled_diagnostic(self):
        info = _C.megamoe_gin_build_info()
        self.assertIn('enabled', info)
        self.assertEqual(info['minimum_nccl_version'], 23007)
        self.assertEqual(info['required_nccl_version'], 23007)
        if not info['enabled']:
            with self.assertRaisesRegex(
                    RuntimeError, r'reason=build_disabled.*DG_MEGAMOE_GIN=1'):
                _C.get_megamoe_gin_unique_id()

    def test_enabled_factory_rejects_cpu_buffer_before_collectives(self):
        info = _C.megamoe_gin_build_info()
        if not info['enabled']:
            self.skipTest('requires a GIN-enabled build, but no GIN hardware')
        with self.assertRaisesRegex(
                ValueError, 'defined, contiguous CUDA tensor'):
            _C.create_megamoe_gin_context(
                torch.empty(4096, dtype=torch.int8), bytes(128), 0, 16,
                9, 64, 3, 8, 'gdaki', 1, 7168, 8)

    def test_enable_broadcasts_uid_and_passes_target_defaults(self):
        symm_buffer = _uninitialized_symm_buffer()
        expected_context = _FakeGinContext()
        uid = bytes(range(128))

        with mock.patch.object(
                mega._C, 'megamoe_gin_build_info',
                return_value={'enabled': True}), mock.patch.object(
                mega._C, 'get_megamoe_gin_unique_id',
                return_value=uid) as get_uid, mock.patch.object(
                mega._C, 'create_megamoe_gin_context',
                return_value=expected_context) as create, mock.patch.object(
                mega.dist, 'broadcast') as broadcast:
            result = symm_buffer.enable_gin()

        self.assertIs(result, expected_context)
        self.assertIs(symm_buffer.gin_context, expected_context)
        get_uid.assert_called_once_with()
        broadcast.assert_called_once()
        (_, passed_uid, rank, world_size, contexts, queue_depth, barriers,
         lsa_size, gin_type, completion_batch, combine_chunk_bytes,
         outbox_depth) = create.call_args.args
        self.assertEqual(passed_uid, uid)
        self.assertEqual((rank, world_size), (0, 16))
        self.assertEqual((contexts, queue_depth, barriers), (9, 64, 3))
        self.assertEqual((lsa_size, gin_type), (8, 'gdaki'))
        self.assertEqual(
            (completion_batch, combine_chunk_bytes, outbox_depth),
            (1, 7168, 8))

    def test_enable_propagates_nondefault_registered_layout_tuning(self):
        symm_buffer = _uninitialized_symm_buffer()
        symm_buffer.gin_completion_batch = 8
        symm_buffer.gin_combine_chunk_bytes = 256
        symm_buffer.gin_outbox_depth = 16
        symm_buffer.gin_queue_depth = 128

        with mock.patch.object(
                mega._C, 'megamoe_gin_build_info',
                return_value={'enabled': True}), mock.patch.object(
                mega._C, 'get_megamoe_gin_unique_id',
                return_value=bytes(128)), mock.patch.object(
                mega._C, 'create_megamoe_gin_context',
                return_value=_FakeGinContext()) as create, mock.patch.object(
                mega.dist, 'broadcast'):
            symm_buffer.enable_gin()

        (_, _, _, _, contexts, queue_depth, barriers, lsa_size, gin_type,
         completion_batch, combine_chunk_bytes, outbox_depth) = create.call_args.args
        self.assertEqual((contexts, queue_depth, barriers), (9, 128, 3))
        self.assertEqual((lsa_size, gin_type), (8, 'gdaki'))
        self.assertEqual(
            (completion_batch, combine_chunk_bytes, outbox_depth),
            (8, 256, 16))

    def test_enable_uses_global_rank_fallback_for_old_torch(self):
        symm_buffer = _uninitialized_symm_buffer(rank=3)
        expected_context = _FakeGinContext()

        with mock.patch.object(
                mega._C, 'megamoe_gin_build_info',
                return_value={'enabled': True}), mock.patch.object(
                mega._C, 'get_megamoe_gin_unique_id') as get_uid, mock.patch.object(
                mega._C, 'create_megamoe_gin_context',
                return_value=expected_context), mock.patch.object(
                mega.dist, 'broadcast', side_effect=[TypeError, None]) as broadcast, mock.patch.object(
                mega.dist, 'get_global_rank', return_value=11) as get_global_rank:
            symm_buffer.enable_gin()

        # Every rank performs the noncollective runtime/header preflight, but
        # only group rank zero's unique ID is broadcast and used.
        get_uid.assert_called_once_with()
        get_global_rank.assert_called_once_with(symm_buffer.group, 0)
        self.assertEqual(broadcast.call_args_list[1].kwargs['src'], 11)

    def test_enable_rejects_second_context(self):
        symm_buffer = _uninitialized_symm_buffer()
        symm_buffer._gin_context = _FakeGinContext()
        with self.assertRaisesRegex(RuntimeError, 'already enabled'):
            symm_buffer.enable_gin()

    def test_gin_enabled_tracks_context_liveness_and_destroyed_context_is_terminal(self):
        symm_buffer = _uninitialized_symm_buffer()
        self.assertFalse(symm_buffer.gin_enabled)
        context = _FakeGinContext()
        symm_buffer._gin_context = context
        self.assertTrue(symm_buffer.gin_enabled)
        context.destroy()
        self.assertFalse(symm_buffer.gin_enabled)
        with self.assertRaisesRegex(RuntimeError, 'cannot be re-enabled'):
            symm_buffer.enable_gin()

    def test_enable_rejects_legacy_sized_buffer(self):
        symm_buffer = _uninitialized_symm_buffer()
        symm_buffer._gin_layout_enabled = False
        with self.assertRaisesRegex(RuntimeError, 'legacy-sized buffer'):
            symm_buffer.enable_gin()

    def test_enable_rejects_queue_depth_change_before_any_collective(self):
        symm_buffer = _uninitialized_symm_buffer()
        with mock.patch.object(
                mega._C, 'megamoe_gin_build_info',
                return_value={'enabled': True}) as build_info, mock.patch.object(
                mega.dist, 'broadcast') as broadcast:
            with self.assertRaisesRegex(RuntimeError, 'construction-time'):
                symm_buffer.enable_gin(queue_depth=32)
        build_info.assert_called_once_with()
        self.all_gather_object.assert_called_once()
        broadcast.assert_not_called()

    def test_rank_skewed_config_fails_before_uid_or_auxiliary_init(self):
        symm_buffer = _uninitialized_symm_buffer()

        def skew(output, local, group):
            output[:] = [local.copy() for _ in range(group.size())]
            output[7]['combine_chunk_bytes'] = 256

        self.all_gather_object.side_effect = skew
        with mock.patch.object(
                mega._C, 'megamoe_gin_build_info',
                return_value={'enabled': True}), mock.patch.object(
                mega._C, 'get_megamoe_gin_unique_id') as get_uid, mock.patch.object(
                mega._C, 'create_megamoe_gin_context') as create, mock.patch.object(
                mega.dist, 'broadcast') as broadcast:
            with self.assertRaisesRegex(RuntimeError, 'configuration mismatch across ranks'):
                symm_buffer.enable_gin()

        get_uid.assert_not_called()
        create.assert_not_called()
        broadcast.assert_not_called()

    def test_destroy_quiesces_collectively_before_releasing_context(self):
        events = []
        symm_buffer = _uninitialized_symm_buffer()
        symm_buffer.group = _FakeGroup(events=events)
        symm_buffer._gin_context = _FakeGinContext(events)

        with mock.patch.object(
                mega.torch.cuda, 'synchronize',
                side_effect=lambda: events.append('cuda.synchronize')):
            symm_buffer.destroy()

        self.assertEqual(events, [
            'cuda.synchronize', 'barrier', 'cuda.synchronize',
            'context.destroy', 'context.release_registration',
        ])
        self.assertIsNone(symm_buffer.gin_context)
        self.assertIsNone(symm_buffer.buffer)
        self.assertIsNone(symm_buffer.group)

    def test_destroyed_gin_context_cannot_reach_legacy_launch(self):
        symm_buffer = _uninitialized_symm_buffer()
        context = _FakeGinContext()
        symm_buffer._gin_context = context
        context.destroy()

        with mock.patch.object(mega._C, 'fp8_fp4_mega_moe') as launch:
            with self.assertRaisesRegex(RuntimeError, 'refusing legacy NVLink fallback'):
                mega.fp8_fp4_mega_moe(
                    object(), (object(), object()), (object(), object()),
                    symm_buffer)
        launch.assert_not_called()

    def test_bf16_rejects_even_a_live_gin_buffer(self):
        symm_buffer = _uninitialized_symm_buffer()
        symm_buffer._gin_context = _FakeGinContext()

        with mock.patch.object(mega._C, 'bf16_mega_moe') as launch:
            with self.assertRaisesRegex(RuntimeError, 'BF16 MegaMoE has no GIN transport'):
                mega.bf16_mega_moe(
                    object(), object(), object(), symm_buffer)
        launch.assert_not_called()

    def test_abort_is_rank_local_and_releases_registration(self):
        events = []
        symm_buffer = _uninitialized_symm_buffer()
        symm_buffer.group = _FakeGroup(events=events)
        symm_buffer._gin_context = _FakeGinContext(events)

        with mock.patch.object(mega.torch.cuda, 'synchronize') as synchronize:
            symm_buffer.abort()

        synchronize.assert_not_called()
        self.assertEqual(events, [
            'context.abort', 'context.release_registration',
        ])
        self.assertIsNone(symm_buffer.gin_context)
        self.assertIsNone(symm_buffer.buffer)
        self.assertIsNone(symm_buffer.group)

    def test_destroy_failure_preserves_ownership_for_collective_retry(self):
        events = []
        symm_buffer = _uninitialized_symm_buffer()
        symm_buffer.group = _FakeGroup(events=events)
        context = _FlakyGinContext(events)
        symm_buffer._gin_context = context

        with mock.patch.object(
                mega.torch.cuda, 'synchronize',
                side_effect=lambda: events.append('cuda.synchronize')):
            with self.assertRaisesRegex(RuntimeError, 'injected teardown failure'):
                symm_buffer.destroy()
            self.assertIs(symm_buffer.gin_context, context)
            self.assertIsNotNone(symm_buffer.buffer)
            self.assertIsNotNone(symm_buffer.group)
            symm_buffer.destroy()

        self.assertEqual(context.destroy_attempts, 2)
        self.assertIsNone(symm_buffer.gin_context)
        self.assertIsNone(symm_buffer.buffer)
        self.assertIsNone(symm_buffer.group)


if __name__ == '__main__':
    unittest.main()
