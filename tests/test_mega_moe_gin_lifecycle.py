import inspect
import os
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


class _FailingAbortGinContext(_FakeGinContext):
    def abort(self):
        if self._events is not None:
            self._events.append('context.abort.failed')
        raise RuntimeError('injected abort failure')


def _uninitialized_symm_buffer(rank=0):
    result = SymmBuffer.__new__(SymmBuffer)
    result.group = _FakeGroup(rank=rank)
    result.buffer = torch.empty(1, dtype=torch.uint8)
    result.handle = object()
    for name in SymmBuffer._RAW_BUFFER_VIEW_NAMES:
        setattr(result, name, object())
    lsa_begin = (rank // 8) * 8
    result.buffer_ptrs = [
        0x1000 + peer if lsa_begin <= peer < lsa_begin + 8 else 0
        for peer in range(16)
    ]
    result.num_experts = 896
    result.num_max_tokens_per_rank = 384
    result.num_topk = 16
    result.hidden = 3584
    result.intermediate_hidden = 3072
    result.num_shared_experts = 0
    result._gin_context = None
    result._gin_layout_enabled = True
    result.gin_completion_batch = 1
    result.gin_combine_chunk_bytes = 7168
    result.gin_outbox_depth = 8
    result.gin_combine_issue_wave = 8
    result.gin_queue_depth = 64
    result.gin_active_fast_path = False
    result.gin_bulk_combine = False
    result.gin_direct_dispatch = False
    return result


def _mirror_all_gather_object(output, local, group):
    output[:] = [local.copy() if isinstance(local, dict) else local
                 for _ in range(group.size())]


class TestMegaMoeGinLifecycle(unittest.TestCase):
    def setUp(self):
        mock.patch.dict(os.environ, {
            'DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT': '0',
            'DG_MEGAMOE_GIN_DISPATCH_OVERLAP': '0',
            'DG_MEGAMOE_GIN_COMBINE_OVERLAP': '0',
            'DG_MEGAMOE_GIN_ACTIVITY_GATE_OPT': '0',
            'DG_MEGAMOE_GIN_STRONGVA_COMBINE_TERMINAL': '0',
            'DG_MEGAMOE_GIN_COMBINE_OWNER_WAVES': '0',
            'DG_MEGAMOE_GIN_PRECONSENSUS_PACK': '0',
            'DG_MEGAMOE_GIN_COOP_DIRECT_PACK': '0',
            'DG_MEGAMOE_GIN_COMBINE_EXPERTS_PER_WAVE': '0',
            'DG_MEGAMOE_GIN_COMBINE_BARRIER_WARPS': '1',
        }).start()
        self.all_gather_object = mock.patch.object(
            mega.dist, 'all_gather_object',
            side_effect=_mirror_all_gather_object).start()
        self.addCleanup(mock.patch.stopall)

    def test_default_api_is_opt_in(self):
        for public_api in (SymmBuffer, get_symm_buffer_for_mega_moe):
            with self.subTest(public_api=public_api.__name__):
                parameters = inspect.signature(public_api).parameters
                self.assertFalse(parameters['enable_gin'].default)
                self.assertFalse(parameters['gin_active_fast_path'].default)
                self.assertFalse(parameters['gin_bulk_combine'].default)
                self.assertFalse(parameters['gin_direct_dispatch'].default)
                self.assertEqual(parameters['gin_combine_issue_wave'].default, 8)
                self.assertLess(
                    list(parameters).index('gin_queue_depth'),
                    list(parameters).index('gin_combine_issue_wave'))

    def test_constructor_rejects_unsupported_completion_before_allocation(self):
        with mock.patch.object(
                mega._C, 'get_symm_buffer_size_for_mega_moe') as get_size:
            for completion_batch in (0, 3, 16):
                with self.subTest(completion_batch=completion_batch):
                    with self.assertRaisesRegex(
                            ValueError, 'gin_completion_batch=1, 2, 4, or 8'):
                        SymmBuffer(
                            _FakeGroup(), 896, 384, 16, 3584, 3072,
                            enable_gin=True,
                            gin_completion_batch=completion_batch)

        get_size.assert_not_called()
        self.all_gather_object.assert_not_called()

    def test_constructor_rejects_unsupported_combine_issue_wave_before_allocation(self):
        with mock.patch.object(
                mega._C, 'get_symm_buffer_size_for_mega_moe') as get_size:
            for combine_issue_wave in (0, 3, 16):
                with self.subTest(combine_issue_wave=combine_issue_wave):
                    with self.assertRaisesRegex(
                            ValueError,
                            'gin_combine_issue_wave=1, 2, 4, or 8'):
                        SymmBuffer(
                            _FakeGroup(), 896, 384, 16, 3584, 3072,
                            enable_gin=True,
                            gin_combine_issue_wave=combine_issue_wave)

        get_size.assert_not_called()
        self.all_gather_object.assert_not_called()

    def test_constructor_rejects_undersized_queue_before_allocation(self):
        with mock.patch.object(
                mega._C, 'get_symm_buffer_size_for_mega_moe') as get_size:
            with self.assertRaisesRegex(ValueError, 'gin_queue_depth >= 64'):
                SymmBuffer(
                    _FakeGroup(), 896, 384, 16, 3584, 3072,
                    enable_gin=True, gin_queue_depth=32)

        get_size.assert_not_called()
        self.all_gather_object.assert_not_called()

    def test_constructor_rejects_direct_dispatch_without_gin_before_allocation(self):
        with mock.patch.object(
                mega._C, 'get_symm_buffer_size_for_mega_moe') as get_size:
            with self.assertRaisesRegex(
                    ValueError, 'gin_direct_dispatch requires enable_gin=True'):
                SymmBuffer(
                    _FakeGroup(), 896, 384, 16, 3584, 3072,
                    gin_active_fast_path=True,
                    gin_direct_dispatch=True)

        get_size.assert_not_called()
        self.all_gather_object.assert_not_called()

    def test_constructor_rejects_direct_dispatch_without_activity_consensus(self):
        with mock.patch.object(
                mega._C, 'get_symm_buffer_size_for_mega_moe') as get_size:
            with self.assertRaisesRegex(
                    ValueError, 'requires gin_active_fast_path=True'):
                SymmBuffer(
                    _FakeGroup(), 896, 384, 16, 3584, 3072,
                    enable_gin=True,
                    gin_direct_dispatch=True)

        get_size.assert_not_called()
        self.all_gather_object.assert_not_called()

    def test_constructor_rejects_direct_dispatch_outside_target_contract(self):
        cases = (
            (_FakeGroup(size=8), 896, 384, 16, 3584, 3072),
            (_FakeGroup(), 896, 383, 16, 3584, 3072),
        )
        with mock.patch.object(
                mega._C, 'get_symm_buffer_size_for_mega_moe') as get_size:
            for case in cases:
                with self.subTest(case=case[1:]):
                    with self.assertRaisesRegex(
                            ValueError,
                            'EP16/E896|num_max_tokens_per_rank >= 384'):
                        SymmBuffer(
                            *case,
                            enable_gin=True,
                            gin_active_fast_path=True,
                            gin_direct_dispatch=True)

        get_size.assert_not_called()
        self.all_gather_object.assert_not_called()

    def test_build_info_and_disabled_diagnostic(self):
        info = _C.megamoe_gin_build_info()
        self.assertIn('enabled', info)
        self.assertEqual(info['minimum_nccl_version'], 23007)
        self.assertEqual(info['required_nccl_version'], 23007)
        self.assertEqual(info['default_world_barrier_count'], 4)
        if info['enabled']:
            self.assertEqual(info['required_signal_count'], 2)
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
                9, 64, 4, 8, 'gdaki', 1, 7168, 8, 8)

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
         outbox_depth, combine_issue_wave, active_fast_path,
         bulk_combine, direct_dispatch) = create.call_args.args
        self.assertEqual(passed_uid, uid)
        self.assertEqual((rank, world_size), (0, 16))
        self.assertEqual((contexts, queue_depth, barriers), (9, 64, 4))
        self.assertEqual((lsa_size, gin_type), (8, 'gdaki'))
        self.assertEqual(
            (completion_batch, combine_chunk_bytes, outbox_depth,
             combine_issue_wave, active_fast_path, bulk_combine),
            (1, 7168, 8, 8, False, False))
        self.assertFalse(direct_dispatch)

    def test_enable_passes_direct_dispatch_independently_of_bulk_combine(self):
        symm_buffer = _uninitialized_symm_buffer()
        symm_buffer.gin_active_fast_path = True
        symm_buffer.gin_direct_dispatch = True

        with mock.patch.object(
                mega._C, 'megamoe_gin_build_info',
                return_value={'enabled': True}), mock.patch.object(
                mega._C, 'get_megamoe_gin_unique_id',
                return_value=bytes(128)), mock.patch.object(
                mega._C, 'create_megamoe_gin_context',
                return_value=_FakeGinContext()) as create, mock.patch.object(
                mega.dist, 'broadcast'):
            symm_buffer.enable_gin()

        self.assertEqual(create.call_args.args[-3:], (True, False, True))

    def test_enable_collectively_rejects_unsupported_completion_batch(self):
        symm_buffer = _uninitialized_symm_buffer()
        symm_buffer.gin_completion_batch = 3
        symm_buffer.gin_combine_chunk_bytes = 256
        symm_buffer.gin_outbox_depth = 16
        symm_buffer.gin_queue_depth = 128

        with mock.patch.object(
                mega._C, 'megamoe_gin_build_info',
                return_value={'enabled': True}), mock.patch.object(
                mega._C, 'get_megamoe_gin_unique_id') as get_uid, mock.patch.object(
                mega._C, 'create_megamoe_gin_context') as create, mock.patch.object(
                mega.dist, 'broadcast') as broadcast:
            with self.assertRaisesRegex(
                    RuntimeError, 'completion_batch must be 1, 2, 4, or 8'):
                symm_buffer.enable_gin()

        get_uid.assert_not_called()
        create.assert_not_called()
        broadcast.assert_not_called()

    def test_enable_collectively_rejects_unsupported_combine_issue_wave(self):
        symm_buffer = _uninitialized_symm_buffer()
        symm_buffer.gin_combine_issue_wave = 3

        with mock.patch.object(
                mega._C, 'megamoe_gin_build_info',
                return_value={'enabled': True}), mock.patch.object(
                mega._C, 'get_megamoe_gin_unique_id') as get_uid, mock.patch.object(
                mega._C, 'create_megamoe_gin_context') as create, mock.patch.object(
                mega.dist, 'broadcast') as broadcast:
            with self.assertRaisesRegex(
                    RuntimeError, 'combine_issue_wave must be 1, 2, 4, or 8'):
                symm_buffer.enable_gin()

        get_uid.assert_not_called()
        create.assert_not_called()
        broadcast.assert_not_called()

    def test_enable_collectively_rejects_three_barrier_slots(self):
        symm_buffer = _uninitialized_symm_buffer()

        with mock.patch.object(
                mega._C, 'megamoe_gin_build_info',
                return_value={'enabled': True}), mock.patch.object(
                mega._C, 'get_megamoe_gin_unique_id') as get_uid, mock.patch.object(
                mega._C, 'create_megamoe_gin_context') as create, mock.patch.object(
                mega.dist, 'broadcast') as broadcast:
            with self.assertRaisesRegex(
                    RuntimeError, 'world_barrier_count must be at least 4'):
                symm_buffer.enable_gin(world_barrier_count=3)

        get_uid.assert_not_called()
        create.assert_not_called()
        broadcast.assert_not_called()

    def test_enable_accepts_nonzero_aliases_outside_local_lsa(self):
        symm_buffer = _uninitialized_symm_buffer()
        symm_buffer.buffer_ptrs = [0x1000 + peer for peer in range(16)]

        with mock.patch.object(
                mega._C, 'megamoe_gin_build_info',
                return_value={'enabled': True}), mock.patch.object(
                mega._C, 'get_megamoe_gin_unique_id',
                return_value=bytes(128)), mock.patch.object(
                mega._C, 'create_megamoe_gin_context',
                return_value=_FakeGinContext()) as create, mock.patch.object(
                mega.dist, 'broadcast'):
            symm_buffer.enable_gin()

        create.assert_called_once()

    def test_missing_same_lsa_alias_fails_collectively_before_uid(self):
        symm_buffer = _uninitialized_symm_buffer(rank=11)
        symm_buffer.buffer_ptrs[12] = 0

        with mock.patch.object(
                mega._C, 'megamoe_gin_build_info',
                return_value={'enabled': True}), mock.patch.object(
                mega._C, 'get_megamoe_gin_unique_id') as get_uid, mock.patch.object(
                mega._C, 'create_megamoe_gin_context') as create, mock.patch.object(
                mega.dist, 'broadcast') as broadcast:
            with self.assertRaisesRegex(
                    RuntimeError,
                    r'contiguous LSA ranks \[8, 9, 10, 11, 12, 13, 14, 15\], '
                    r'missing ranks \[12\]'):
                symm_buffer.enable_gin()

        self.assertEqual(self.all_gather_object.call_count, 2)
        get_uid.assert_not_called()
        create.assert_not_called()
        broadcast.assert_not_called()

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

    def test_rank_skewed_active_fast_path_fails_before_uid_or_auxiliary_init(self):
        symm_buffer = _uninitialized_symm_buffer()
        symm_buffer.gin_active_fast_path = True

        def skew(output, local, group):
            output[:] = [local.copy() for _ in range(group.size())]
            output[7]['active_fast_path'] = False

        self.all_gather_object.side_effect = skew
        with mock.patch.object(
                mega._C, 'megamoe_gin_build_info',
                return_value={'enabled': True}), mock.patch.object(
                mega._C, 'get_megamoe_gin_unique_id') as get_uid, mock.patch.object(
                mega._C, 'create_megamoe_gin_context') as create, mock.patch.object(
                mega.dist, 'broadcast') as broadcast:
            with self.assertRaisesRegex(
                    RuntimeError, 'configuration mismatch across ranks'):
                symm_buffer.enable_gin()

        get_uid.assert_not_called()
        create.assert_not_called()
        broadcast.assert_not_called()

    def test_single_context_raw_rank_skew_fails_before_auxiliary_init(self):
        symm_buffer = _uninitialized_symm_buffer()

        def skew(output, local, group):
            output[:] = [local.copy() for _ in range(group.size())]
            output[7]['single_combine_context'] = '1'

        self.all_gather_object.side_effect = skew
        with mock.patch.object(
                mega._C, 'megamoe_gin_build_info',
                return_value={'enabled': True}), mock.patch.object(
                mega._C, 'get_megamoe_gin_unique_id') as get_uid, mock.patch.object(
                mega._C, 'create_megamoe_gin_context') as create:
            with self.assertRaisesRegex(RuntimeError, 'configuration mismatch across ranks'):
                symm_buffer.enable_gin()
        self.all_gather_object.assert_called_once()
        get_uid.assert_not_called()
        create.assert_not_called()

    def test_single_context_invalid_raw_values_fail_collectively(self):
        for raw in ('', 'true', '01', '2', '-1', ' 1'):
            with self.subTest(raw=raw), mock.patch.dict(os.environ, {
                    'DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT': raw}), mock.patch.object(
                    mega._C, 'megamoe_gin_build_info',
                    return_value={'enabled': True}), mock.patch.object(
                    mega._C, 'get_megamoe_gin_unique_id') as get_uid:
                self.all_gather_object.reset_mock()
                with self.assertRaisesRegex(RuntimeError, 'must be exactly 0 or 1'):
                    _uninitialized_symm_buffer().enable_gin()
                self.all_gather_object.assert_called_once()
                get_uid.assert_not_called()

    def test_single_context_dependencies_rejected_collectively(self):
        with mock.patch.dict(os.environ, {
                'DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT': '1'}):
            symm_buffer = _uninitialized_symm_buffer()
            with self.assertRaisesRegex(RuntimeError, 'requires bulk_combine and direct_dispatch'):
                symm_buffer._collective_validate_gin_config(
                    {'enabled': True}, 9, 64, 4, 8, 'gdaki')
            symm_buffer.gin_active_fast_path = True
            symm_buffer.gin_bulk_combine = True
            symm_buffer.gin_direct_dispatch = True
            symm_buffer.gin_outbox_depth = 64
            for flag, value in (
                    ('DG_MEGAMOE_GIN_COMBINE_EXPERTS_PER_WAVE', '1'),
                    ('DG_MEGAMOE_GIN_COMBINE_BARRIER_WARPS', '8')):
                with self.subTest(flag=flag), mock.patch.dict(os.environ, {flag: value}):
                    with self.assertRaisesRegex(RuntimeError, 'COMBINE_EXPERTS_PER_WAVE=0'):
                        symm_buffer._collective_validate_gin_config(
                            {'enabled': True}, 9, 64, 4, 8, 'gdaki')

    def test_single_context_valid_raw_mode_is_recorded_without_extra_collective(self):
        with mock.patch.dict(os.environ, {
                'DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT': '1'}):
            symm_buffer = _uninitialized_symm_buffer()
            symm_buffer.gin_active_fast_path = True
            symm_buffer.gin_bulk_combine = True
            symm_buffer.gin_direct_dispatch = True
            symm_buffer.gin_outbox_depth = 64
            symm_buffer._collective_validate_gin_config(
                {'enabled': True}, 9, 64, 4, 8, 'gdaki')
        self.all_gather_object.assert_called_once()
        gathered_config = self.all_gather_object.call_args.args[1]
        self.assertEqual(gathered_config['single_combine_context'], '1')
        self.assertEqual(gathered_config['combine_experts_per_wave'], '0')
        self.assertEqual(gathered_config['combine_barrier_warps'], '1')
        self.assertEqual(gathered_config['combine_owner_waves'], '0')

    def test_protocol_change_contract_is_documented_without_hot_path_collective(self):
        self.assertIn('caller-enforced rank agreement', SymmBuffer.enable_gin.__doc__)
        launch_guard = inspect.getsource(SymmBuffer._require_launchable_transport)
        self.assertNotIn('all_gather', launch_guard)

    def test_dispatch_overlap_raw_rank_skew_fails_before_uid(self):
        for local_mode, remote_mode in (('0', '1'), ('1', '0')):
            with self.subTest(local=local_mode, remote=remote_mode), mock.patch.dict(os.environ, {
                    'DG_MEGAMOE_GIN_DISPATCH_OVERLAP': local_mode,
                    'DG_MEGAMOE_GIN_PRECONSENSUS_PACK': '1',
                    'DG_MEGAMOE_GIN_COOP_DIRECT_PACK': '1'}):
                symm_buffer = _uninitialized_symm_buffer()
                symm_buffer.gin_active_fast_path = True
                symm_buffer.gin_bulk_combine = True
                symm_buffer.gin_direct_dispatch = True

                def skew(output, local, group):
                    _mirror_all_gather_object(output, local, group)
                    output[7]['dispatch_overlap'] = remote_mode

                self.all_gather_object.reset_mock()
                self.all_gather_object.side_effect = skew
                with mock.patch.object(
                        mega._C, 'megamoe_gin_build_info',
                        return_value={'enabled': True}), mock.patch.object(
                        mega._C, 'get_megamoe_gin_unique_id') as get_uid, mock.patch.object(
                        mega._C, 'create_megamoe_gin_context') as create, mock.patch.object(
                        mega.dist, 'broadcast') as broadcast:
                    with self.assertRaisesRegex(RuntimeError, 'configuration mismatch across ranks'):
                        symm_buffer.enable_gin()
                self.all_gather_object.assert_called_once()
                self.assertEqual(self.all_gather_object.call_args.args[1]['dispatch_overlap'], local_mode)
                get_uid.assert_not_called()
                create.assert_not_called()
                broadcast.assert_not_called()

    def test_combine_overlap_rank_skew_and_malformed_values_fail_before_uid(self):
        for raw, skewed in (("0", True), ("1", True), ("", False), ("01", False),
                            ("2", False), ("true", False), (" 1", False), ("1suffix", False)):
            with self.subTest(raw=raw, skewed=skewed), mock.patch.dict(os.environ, {
                    'DG_MEGAMOE_GIN_COMBINE_OVERLAP': raw,
                    'DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT': '1',
                    'DG_MEGAMOE_GIN_DISPATCH_OVERLAP': '1',
                    'DG_MEGAMOE_GIN_PRECONSENSUS_PACK': '1',
                    'DG_MEGAMOE_GIN_COOP_DIRECT_PACK': '1'}), mock.patch.object(
                    mega._C, 'megamoe_gin_build_info', return_value={'enabled': True}), mock.patch.object(
                    mega._C, 'get_megamoe_gin_unique_id') as get_uid, mock.patch.object(
                    mega._C, 'create_megamoe_gin_context') as create:
                symm_buffer = _uninitialized_symm_buffer()
                symm_buffer.gin_active_fast_path = symm_buffer.gin_bulk_combine = True
                symm_buffer.gin_direct_dispatch = True
                symm_buffer.gin_outbox_depth = 64
                def gather(output, local, group):
                    _mirror_all_gather_object(output, local, group)
                    if skewed:
                        output[7]['combine_overlap'] = '0' if raw == '1' else '1'
                self.all_gather_object.side_effect = gather
                self.all_gather_object.reset_mock()
                expected = 'configuration mismatch across ranks' if skewed else 'COMBINE_OVERLAP must be exactly 0 or 1'
                with self.assertRaisesRegex(RuntimeError, expected):
                    symm_buffer.enable_gin()
                self.all_gather_object.assert_called_once()
                self.assertEqual(self.all_gather_object.call_args.args[1]['combine_overlap'], raw)
                get_uid.assert_not_called()
                create.assert_not_called()

    def test_combine_overlap_each_dependency_is_collectively_rejected(self):
        for missing in ('single_combine_context', 'dispatch_overlap', 'bulk_combine', 'direct_dispatch'):
            env = {'DG_MEGAMOE_GIN_COMBINE_OVERLAP': '1',
                   'DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT': '1',
                   'DG_MEGAMOE_GIN_DISPATCH_OVERLAP': '1',
                   'DG_MEGAMOE_GIN_PRECONSENSUS_PACK': '1',
                   'DG_MEGAMOE_GIN_COOP_DIRECT_PACK': '1'}
            if missing in ('single_combine_context', 'dispatch_overlap'):
                env['DG_MEGAMOE_GIN_' + missing.upper()] = '0'
            with self.subTest(missing=missing), mock.patch.dict(os.environ, env), mock.patch.object(
                    mega._C, 'get_megamoe_gin_unique_id') as get_uid:
                symm_buffer = _uninitialized_symm_buffer()
                symm_buffer.gin_active_fast_path = True
                symm_buffer.gin_bulk_combine = missing != 'bulk_combine'
                symm_buffer.gin_direct_dispatch = missing != 'direct_dispatch'
                symm_buffer.gin_outbox_depth = 64
                self.all_gather_object.reset_mock()
                with self.assertRaisesRegex(RuntimeError, 'combine_overlap requires'):
                    symm_buffer._collective_validate_gin_config({'enabled': True}, 9, 64, 4, 8, 'gdaki')
                self.all_gather_object.assert_called_once()
                get_uid.assert_not_called()

    def test_combine_overlap_valid_mode_uses_existing_pre_uid_collective(self):
        with mock.patch.dict(os.environ, {
                'DG_MEGAMOE_GIN_COMBINE_OVERLAP': '1',
                'DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT': '1',
                'DG_MEGAMOE_GIN_DISPATCH_OVERLAP': '1',
                'DG_MEGAMOE_GIN_PRECONSENSUS_PACK': '1',
                'DG_MEGAMOE_GIN_COOP_DIRECT_PACK': '1'}):
            symm_buffer = _uninitialized_symm_buffer()
            symm_buffer.gin_active_fast_path = symm_buffer.gin_bulk_combine = True
            symm_buffer.gin_direct_dispatch = True
            symm_buffer.gin_outbox_depth = 64
            symm_buffer._collective_validate_gin_config({'enabled': True}, 9, 64, 4, 8, 'gdaki')
        self.all_gather_object.assert_called_once()
        self.assertEqual(self.all_gather_object.call_args.args[1]['combine_overlap'], '1')

    def test_combine_owner_waves_are_exact_collective_and_dependency_checked(self):
        dependency_env = {
            'DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT': '1',
            'DG_MEGAMOE_GIN_DISPATCH_OVERLAP': '1',
            'DG_MEGAMOE_GIN_COMBINE_OVERLAP': '1',
            'DG_MEGAMOE_GIN_ACTIVITY_GATE_OPT': '1',
            'DG_MEGAMOE_GIN_STRONGVA_COMBINE_TERMINAL': '1',
            'DG_MEGAMOE_GIN_PRECONSENSUS_PACK': '1',
            'DG_MEGAMOE_GIN_COOP_DIRECT_PACK': '1',
        }

        def configured_buffer():
            result = _uninitialized_symm_buffer()
            result.group = _FakeGroup(size=8)
            result.buffer_ptrs = [0x1000 + peer for peer in range(8)]
            result.num_experts = 448
            result.gin_active_fast_path = True
            result.gin_bulk_combine = True
            result.gin_direct_dispatch = True
            result.gin_outbox_depth = 64
            return result

        for raw in ('0', '2', '4', '8'):
            with self.subTest(raw=raw), mock.patch.dict(os.environ, {
                    **dependency_env,
                    'DG_MEGAMOE_GIN_COMBINE_OWNER_WAVES': raw}):
                self.all_gather_object.reset_mock()
                configured_buffer()._collective_validate_gin_config(
                    {'enabled': True}, 9, 64, 4, 4, 'gdaki')
                self.assertEqual(
                    self.all_gather_object.call_args.args[1]
                        ['combine_owner_waves'], raw)

        for raw in ('1', '3', '6', '02', '8 ', ''):
            with self.subTest(raw=raw), mock.patch.dict(os.environ, {
                    **dependency_env,
                    'DG_MEGAMOE_GIN_COMBINE_OWNER_WAVES': raw}):
                self.all_gather_object.reset_mock()
                with self.assertRaisesRegex(
                        RuntimeError, 'must be exactly 0, 2, 4, or 8'):
                    configured_buffer()._collective_validate_gin_config(
                        {'enabled': True}, 9, 64, 4, 4, 'gdaki')
                self.all_gather_object.assert_called_once()

        with mock.patch.dict(os.environ, {
                'DG_MEGAMOE_GIN_COMBINE_OWNER_WAVES': '2'}):
            with self.assertRaisesRegex(RuntimeError,
                                        'combine_owner_waves requires'):
                configured_buffer()._collective_validate_gin_config(
                    {'enabled': True}, 9, 64, 4, 4, 'gdaki')

        def skew(output, local, group):
            _mirror_all_gather_object(output, local, group)
            output[7]['combine_owner_waves'] = '8'

        with mock.patch.dict(os.environ, {
                **dependency_env,
                'DG_MEGAMOE_GIN_COMBINE_OWNER_WAVES': '4'}):
            self.all_gather_object.side_effect = skew
            with self.assertRaisesRegex(RuntimeError,
                                        'configuration mismatch across ranks'):
                configured_buffer()._collective_validate_gin_config(
                    {'enabled': True}, 9, 64, 4, 4, 'gdaki')

    def test_dispatch_overlap_invalid_raw_values_fail_collectively_before_uid(self):
        for raw in ('', 'true', '01', '2', '-1', ' 1', '1 ', '1suffix'):
            with self.subTest(raw=raw), mock.patch.dict(os.environ, {
                    'DG_MEGAMOE_GIN_DISPATCH_OVERLAP': raw}), mock.patch.object(
                    mega._C, 'megamoe_gin_build_info',
                    return_value={'enabled': True}), mock.patch.object(
                    mega._C, 'get_megamoe_gin_unique_id') as get_uid, mock.patch.object(
                    mega._C, 'create_megamoe_gin_context') as create:
                self.all_gather_object.reset_mock()
                with self.assertRaisesRegex(RuntimeError, 'DISPATCH_OVERLAP must be exactly 0 or 1'):
                    _uninitialized_symm_buffer().enable_gin()
                self.all_gather_object.assert_called_once()
                self.assertEqual(self.all_gather_object.call_args.args[1]['dispatch_overlap'], raw)
                get_uid.assert_not_called()
                create.assert_not_called()

    def test_dispatch_overlap_each_dependency_fails_collectively_before_uid(self):
        for missing in ('bulk_combine', 'direct_dispatch', 'preconsensus_pack', 'coop_direct_pack'):
            environment = {
                'DG_MEGAMOE_GIN_DISPATCH_OVERLAP': '1',
                'DG_MEGAMOE_GIN_PRECONSENSUS_PACK': '1',
                'DG_MEGAMOE_GIN_COOP_DIRECT_PACK': '1',
            }
            if missing in ('preconsensus_pack', 'coop_direct_pack'):
                environment['DG_MEGAMOE_GIN_' + missing.upper()] = '0'
            with self.subTest(missing=missing), mock.patch.dict(os.environ, environment), mock.patch.object(
                    mega._C, 'megamoe_gin_build_info',
                    return_value={'enabled': True}), mock.patch.object(
                    mega._C, 'get_megamoe_gin_unique_id') as get_uid, mock.patch.object(
                    mega._C, 'create_megamoe_gin_context') as create:
                symm_buffer = _uninitialized_symm_buffer()
                symm_buffer.gin_active_fast_path = True
                symm_buffer.gin_bulk_combine = missing != 'bulk_combine'
                symm_buffer.gin_direct_dispatch = missing != 'direct_dispatch'
                self.all_gather_object.reset_mock()
                with self.assertRaisesRegex(RuntimeError, 'dispatch_overlap requires bulk_combine, direct_dispatch'):
                    symm_buffer.enable_gin()
                self.all_gather_object.assert_called_once()
                get_uid.assert_not_called()
                create.assert_not_called()

    def test_dispatch_overlap_valid_mode_is_recorded_in_single_pre_uid_collective(self):
        with mock.patch.dict(os.environ, {
                'DG_MEGAMOE_GIN_DISPATCH_OVERLAP': '1',
                'DG_MEGAMOE_GIN_PRECONSENSUS_PACK': '1',
                'DG_MEGAMOE_GIN_COOP_DIRECT_PACK': '1'}), mock.patch.object(
                mega._C, 'get_megamoe_gin_unique_id') as get_uid:
            symm_buffer = _uninitialized_symm_buffer()
            symm_buffer.gin_active_fast_path = True
            symm_buffer.gin_bulk_combine = True
            symm_buffer.gin_direct_dispatch = True
            symm_buffer._collective_validate_gin_config(
                {'enabled': True}, 9, 64, 4, 8, 'gdaki')
        self.all_gather_object.assert_called_once()
        config = self.all_gather_object.call_args.args[1]
        self.assertEqual(config['dispatch_overlap'], '1')
        self.assertEqual(config['preconsensus_pack'], '1')
        self.assertEqual(config['coop_direct_pack'], '1')
        get_uid.assert_not_called()
        source = inspect.getsource(SymmBuffer.enable_gin)
        self.assertLess(source.index('_collective_validate_gin_config'),
                        source.index('_collective_get_gin_unique_id'))

    def test_retired_experiments_fail_collectively_even_when_single_mode_is_off(self):
        for mode in ('0', '1'):
            for flag, value in (
                    ('DG_MEGAMOE_GIN_COMBINE_EXPERTS_PER_WAVE', '8'),
                    ('DG_MEGAMOE_GIN_COMBINE_BARRIER_WARPS', '8'),
                    ('DG_MEGAMOE_GIN_COMBINE_EXPERTS_PER_WAVE', '00'),
                    ('DG_MEGAMOE_GIN_COMBINE_BARRIER_WARPS', '01')):
                with self.subTest(mode=mode, flag=flag), mock.patch.dict(os.environ, {
                        'DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT': mode, flag: value}):
                    self.all_gather_object.reset_mock()
                    with self.assertRaisesRegex(RuntimeError, 'retired experiments'):
                        _uninitialized_symm_buffer()._collective_validate_gin_config(
                            {'enabled': True}, 9, 64, 4, 8, 'gdaki')
                    self.all_gather_object.assert_called_once()

    def test_rank_skewed_bulk_combine_fails_before_uid_or_auxiliary_init(self):
        symm_buffer = _uninitialized_symm_buffer()
        symm_buffer.gin_active_fast_path = True
        symm_buffer.gin_bulk_combine = True
        symm_buffer.gin_outbox_depth = 64

        def skew(output, local, group):
            output[:] = [local.copy() for _ in range(group.size())]
            output[7]['bulk_combine'] = False

        self.all_gather_object.side_effect = skew
        with mock.patch.object(
                mega._C, 'megamoe_gin_build_info',
                return_value={'enabled': True}), mock.patch.object(
                mega._C, 'get_megamoe_gin_unique_id') as get_uid, mock.patch.object(
                mega._C, 'create_megamoe_gin_context') as create, mock.patch.object(
                mega.dist, 'broadcast') as broadcast:
            with self.assertRaisesRegex(
                    RuntimeError, 'configuration mismatch across ranks'):
                symm_buffer.enable_gin()

        get_uid.assert_not_called()
        create.assert_not_called()
        broadcast.assert_not_called()

    def test_rank_skewed_direct_dispatch_fails_before_uid_or_auxiliary_init(self):
        symm_buffer = _uninitialized_symm_buffer()
        symm_buffer.gin_active_fast_path = True
        symm_buffer.gin_direct_dispatch = True

        def skew(output, local, group):
            output[:] = [local.copy() for _ in range(group.size())]
            output[7]['direct_dispatch'] = False

        self.all_gather_object.side_effect = skew
        with mock.patch.object(
                mega._C, 'megamoe_gin_build_info',
                return_value={'enabled': True}), mock.patch.object(
                mega._C, 'get_megamoe_gin_unique_id') as get_uid, mock.patch.object(
                mega._C, 'create_megamoe_gin_context') as create, mock.patch.object(
                mega.dist, 'broadcast') as broadcast:
            with self.assertRaisesRegex(
                    RuntimeError, 'configuration mismatch across ranks'):
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
        for name in SymmBuffer._RAW_BUFFER_VIEW_NAMES:
            self.assertIsNone(getattr(symm_buffer, name))

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
        for name in SymmBuffer._RAW_BUFFER_VIEW_NAMES:
            self.assertIsNone(getattr(symm_buffer, name))

    def test_abort_failure_still_clears_every_raw_view_and_allocation(self):
        events = []
        symm_buffer = _uninitialized_symm_buffer()
        symm_buffer._gin_context = _FailingAbortGinContext(events)

        with self.assertRaisesRegex(RuntimeError, 'injected abort failure'):
            symm_buffer.abort()

        self.assertEqual(events, [
            'context.abort.failed', 'context.release_registration',
        ])
        self.assertIsNone(symm_buffer.gin_context)
        self.assertIsNone(symm_buffer.handle)
        self.assertIsNone(symm_buffer.buffer_ptrs)
        self.assertIsNone(symm_buffer.buffer)
        self.assertIsNone(symm_buffer.group)
        for name in SymmBuffer._RAW_BUFFER_VIEW_NAMES:
            self.assertIsNone(getattr(symm_buffer, name))

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
            for name in SymmBuffer._RAW_BUFFER_VIEW_NAMES:
                self.assertIsNotNone(getattr(symm_buffer, name))
            symm_buffer.destroy()

        self.assertEqual(context.destroy_attempts, 2)
        self.assertIsNone(symm_buffer.gin_context)
        self.assertIsNone(symm_buffer.buffer)
        self.assertIsNone(symm_buffer.group)
        for name in SymmBuffer._RAW_BUFFER_VIEW_NAMES:
            self.assertIsNone(getattr(symm_buffer, name))


if __name__ == '__main__':
    unittest.main()
