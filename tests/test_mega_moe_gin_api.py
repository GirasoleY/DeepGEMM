import gc
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock
import weakref


_REPO_ROOT = Path(__file__).resolve().parents[1]
_MEGA_MODULE_PATH = _REPO_ROOT / 'deep_gemm' / 'mega' / '__init__.py'


class _FakeTensorType:
    pass


class _FakeBuffer:
    def __init__(self, data_ptr=0x9000, nbytes=1 << 20, events=None):
        self._data_ptr = data_ptr
        self.nbytes = nbytes
        self._events = events

    def data_ptr(self):
        return self._data_ptr

    def zero_(self):
        if self._events is not None:
            self._events.append('zero')
        return self


class _FakeGroup:
    def __init__(self, rank=0, size=8, events=None):
        self._rank = rank
        self._size = size
        self._events = events
        self.barrier_calls = 0

    def rank(self):
        return self._rank

    def size(self):
        return self._size

    def barrier(self):
        self.barrier_calls += 1
        if self._events is not None:
            self._events.append('barrier')


def _load_mega_module():
    """Load the Python wrapper with CPU-only stand-ins for torch and _C."""
    package_name = '_deep_gemm_gin_api_test'
    package = types.ModuleType(package_name)
    package.__path__ = [str(_REPO_ROOT / 'deep_gemm')]

    fake_c = types.ModuleType(f'{package_name}._C')
    fake_c.get_token_alignment_for_mega_moe = mock.Mock(return_value=1920)
    fake_c.megamoe_gin_build_info = mock.Mock(return_value={'enabled': True})
    fake_c.get_symm_buffer_size_for_mega_moe_gin = mock.Mock(
        return_value=(4096, lambda _: tuple(object() for _ in range(12))))
    fake_c.import_megamoe_gin_transport = mock.Mock()
    fake_c.fp8_fp4_mega_moe = mock.Mock()
    fake_c.fp8_fp4_mega_moe_gin = mock.Mock()
    fake_c.bf16_mega_moe = mock.Mock()
    fake_c.bf16_mega_moe_gin = mock.Mock()
    fake_c.get_symm_buffer_size_for_mega_moe = mock.Mock(
        return_value=(1024, lambda _: tuple(object() for _ in range(12))))
    package._C = fake_c

    utils = types.ModuleType(f'{package_name}.utils')
    utils.__path__ = []
    math_module = types.ModuleType(f'{package_name}.utils.math')
    math_module.align = lambda value, alignment: (
        (value + alignment - 1) // alignment * alignment)

    torch_module = types.ModuleType('torch')
    torch_module.__path__ = []
    torch_module.Tensor = _FakeTensorType
    torch_module.int8 = object()
    torch_module.device = mock.Mock(return_value='cuda:0')
    torch_module.cuda = types.SimpleNamespace(
        current_device=mock.Mock(return_value=0),
        synchronize=mock.Mock(),
    )
    torch_module.empty = mock.Mock()

    distributed = types.ModuleType('torch.distributed')
    distributed.__path__ = []
    distributed.ProcessGroup = type('ProcessGroup', (), {})
    symmetric_memory = types.ModuleType('torch.distributed._symmetric_memory')
    symmetric_memory.empty = mock.Mock(return_value=_FakeBuffer(nbytes=4096))
    symmetric_memory.rendezvous = mock.Mock(return_value=types.SimpleNamespace(
        buffer_ptrs=[0x10000 * (peer + 1) for peer in range(8)],
        offset=0x180,
    ))
    torch_module.distributed = distributed

    module_name = f'{package_name}.mega'
    spec = importlib.util.spec_from_file_location(
        module_name, _MEGA_MODULE_PATH,
        submodule_search_locations=[str(_MEGA_MODULE_PATH.parent)])
    module = importlib.util.module_from_spec(spec)
    modules = {
        package_name: package,
        f'{package_name}._C': fake_c,
        f'{package_name}.utils': utils,
        f'{package_name}.utils.math': math_module,
        'torch': torch_module,
        'torch.distributed': distributed,
        'torch.distributed._symmetric_memory': symmetric_memory,
        module_name: module,
    }
    with mock.patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module, fake_c


class TestMegaMoeGinPythonApi(unittest.TestCase):
    def setUp(self):
        self.mega, self.extension = _load_mega_module()

    def test_factory_rejects_non_ep8_or_invalid_token_capacity(self):
        invalid = (
            (_FakeGroup(size=4), 48),
            (_FakeGroup(), 0),
        )
        for group, max_active_tokens in invalid:
            with self.subTest(size=group.size(), tokens=max_active_tokens):
                with self.assertRaisesRegex(ValueError, 'MegaMoE GIN requires'):
                    self.mega.SymmBuffer.for_ep8_gin(
                        group, max_active_tokens=max_active_tokens)

        self.extension.get_symm_buffer_size_for_mega_moe_gin.assert_not_called()

    def test_disabled_build_fails_before_sizing_or_allocation(self):
        self.extension.megamoe_gin_build_info.return_value = {'enabled': False}
        with self.assertRaisesRegex(RuntimeError, r'reason=build_disabled'):
            self.mega.SymmBuffer.for_ep8_gin(_FakeGroup())
        self.extension.get_symm_buffer_size_for_mega_moe_gin.assert_not_called()

    def test_factory_uses_default_geometry_and_local_lsa_mappings(self):
        events = []
        group = _FakeGroup(rank=5, events=events)
        self.mega.symm_mem.empty.return_value = _FakeBuffer(
            nbytes=4096, events=events)
        self.mega.torch.cuda.synchronize.side_effect = \
            lambda: events.append('synchronize')
        result = self.mega.SymmBuffer.for_ep8_gin(
            group, max_active_tokens=12)

        self.assertEqual(result._gin_max_active_tokens, 12)
        self.assertEqual(result.num_max_tokens_per_rank, 1920)
        self.assertEqual(
            (result.num_experts, result.num_topk,
             result.hidden, result.intermediate_hidden),
            (448, 16, 3584, 3072))
        self.assertEqual(result.buffer_ptrs[:4], [0, 0, 0, 0])
        self.assertEqual(result.buffer_ptrs[4], 0x50000 + 0x180)
        self.assertEqual(result.buffer_ptrs[5], result.buffer.data_ptr())
        self.assertEqual(result.buffer_ptrs[6], 0x70000 + 0x180)
        self.assertEqual(result.buffer_ptrs[7], 0x80000 + 0x180)
        self.extension.get_symm_buffer_size_for_mega_moe_gin.assert_called_once_with(
            448, 1920, 16, 3584, 3072,
            'fp8xfp4', 'swiglu', 0, 12, 8, 4)
        self.assertEqual(group.barrier_calls, 1)
        self.assertEqual(events, ['zero', 'synchronize', 'barrier'])

    def test_factory_forwards_nondefault_native_configuration(self):
        result = self.mega.SymmBuffer.for_ep8_gin(
            _FakeGroup(), max_active_tokens=1921,
            num_experts=264, num_topk=6,
            hidden=4096, intermediate_hidden=2048,
            num_shared_experts=1, mma_type='fp8xfp8')

        self.assertEqual(result.num_max_tokens_per_rank, 3840)
        self.assertEqual(
            result._gin_allocation_config,
            (264, 3840, 6, 4096, 2048, 1,
             'fp8xfp8', 'swiglu', 1921))
        self.extension.get_symm_buffer_size_for_mega_moe_gin.assert_called_once_with(
            264, 3840, 6, 4096, 2048,
            'fp8xfp8', 'swiglu', 1, 1921, 8, 4)

    def test_generic_factory_forwards_geometry_dtype_and_topology(self):
        # These are wrapper tests, not GPU correctness tests. Verify that the
        # original prototype shape is not substituted for caller parameters.
        cases = (
            (8, 4, 72, 4, 1024, 2048, 12, 0, 'fp8xfp4'),
            (12, 4, 780, 6, 4096, 1024, 64, 1, 'fp8xfp8'),
            (16, 8, 272, 6, 2048, 4096, 37, 2, 'bf16xbf16'),
        )
        for world, lsa, experts, topk, hidden, intermediate, capacity, shared, mma in cases:
            with self.subTest(world=world, mma=mma):
                self.extension.get_symm_buffer_size_for_mega_moe_gin.reset_mock()
                self.mega.symm_mem.rendezvous.return_value = types.SimpleNamespace(
                    buffer_ptrs=[0x10000 * (peer + 1) for peer in range(world)],
                    offset=128)
                result = self.mega.SymmBuffer.for_gin(
                    _FakeGroup(size=world), capacity, lsa_size=lsa,
                    num_experts=experts, num_topk=topk,
                    hidden=hidden, intermediate_hidden=intermediate,
                    num_shared_experts=shared, mma_type=mma)
                self.extension.get_symm_buffer_size_for_mega_moe_gin.assert_called_once_with(
                    experts, 1920, topk, hidden, intermediate,
                    mma, 'swiglu', shared, capacity, world, lsa)
                self.assertEqual(result._gin_allocation_config,
                                 (experts, 1920, topk, hidden, intermediate,
                                  shared, mma, 'swiglu', capacity))
                self.assertEqual(len(result.buffer_ptrs), world)
                self.assertEqual(sum(bool(ptr) for ptr in result.buffer_ptrs), lsa)

    def test_pointer_resolution_requires_every_local_lsa_mapping(self):
        handle = types.SimpleNamespace(
            buffer_ptrs=[0x1000] * 8,
            offset=128,
        )
        handle.buffer_ptrs[2] = 0
        with self.assertRaisesRegex(RuntimeError, 'local-LSA peer 2'):
            self.mega._make_gin_buffer_ptrs(
                _FakeBuffer(), handle, rank=1, world_size=8, lsa_size=4)

    def test_import_requires_and_pins_external_owner(self):
        group = _FakeGroup(rank=3)
        sym_buffer = self.mega.SymmBuffer.__new__(self.mega.SymmBuffer)
        sym_buffer.group = group
        sym_buffer.buffer = _FakeBuffer()
        sym_buffer.handle = object()
        sym_buffer._gin_max_active_tokens = 48
        sym_buffer._gin_transport = None

        with self.assertRaisesRegex(ValueError, 'non-None external owner'):
            sym_buffer.bind_gin_transport(object(), owner=None)
        self.extension.import_megamoe_gin_transport.assert_not_called()

        class Owner:
            pass

        owner = Owner()
        owner_ref = weakref.ref(owner)
        capsule = object()
        bound_handle = types.SimpleNamespace(owner=owner)
        self.extension.import_megamoe_gin_transport.return_value = bound_handle
        result = sym_buffer.bind_gin_transport(capsule, owner)
        self.extension.import_megamoe_gin_transport.assert_called_once_with(
            capsule, sym_buffer.buffer, 3, owner)
        del owner
        gc.collect()

        self.assertIs(result, sym_buffer)
        self.assertIs(sym_buffer._gin_transport, bound_handle)
        self.assertIsNotNone(owner_ref())
        with self.assertRaisesRegex(RuntimeError, 'already has a bound transport'):
            sym_buffer.bind_gin_transport(capsule, bound_handle)

    def test_normal_launch_api_selects_bound_gin_transport(self):
        sym_buffer = self.mega.SymmBuffer.__new__(self.mega.SymmBuffer)
        sym_buffer.group = _FakeGroup(rank=6)
        sym_buffer.buffer = _FakeBuffer()
        sym_buffer.handle = object()
        sym_buffer.buffer_ptrs = [0, 0, 0, 0, 1, 2, 3, 4]
        sym_buffer.num_max_tokens_per_rank = 1920
        sym_buffer._gin_max_active_tokens = 48
        sym_buffer.num_experts = 448
        sym_buffer.num_topk = 16
        sym_buffer.hidden = 3584
        sym_buffer.intermediate_hidden = 3072
        sym_buffer.num_shared_experts = 0
        sym_buffer.mma_type = 'fp8xfp4'
        sym_buffer.activation = 'swiglu'
        sym_buffer._gin_allocation_config = (
            448, 1920, 16, 3584, 3072, 0,
            'fp8xfp4', 'swiglu', 48)
        sym_buffer._gin_transport = object()

        y, l1, l2, stats = object(), object(), object(), object()
        self.mega.fp8_fp4_mega_moe(
            y, l1, l2, sym_buffer,
            cumulative_local_expert_recv_stats=stats)

        args = self.extension.fp8_fp4_mega_moe_gin.call_args.args
        self.assertEqual(
            args[:7],
            (y, l1, l2, None, None, stats, sym_buffer.buffer))
        self.assertEqual(
            args[7:17],
            (sym_buffer.buffer_ptrs, 6, 1920, 48, 448, 16,
             3584, 3072, 0, 'fp8xfp4'))
        self.assertEqual(
            args[17:22],
            ((1, 1, 32), 'swiglu', 'swiglu', None, True))
        self.assertIs(args[22], sym_buffer._gin_transport)
        self.extension.fp8_fp4_mega_moe.assert_not_called()

    def test_gin_launch_forwards_shared_weights(self):
        sym_buffer = self.mega.SymmBuffer.__new__(self.mega.SymmBuffer)
        sym_buffer.group = _FakeGroup(rank=2)
        sym_buffer.buffer = _FakeBuffer()
        sym_buffer.handle = object()
        sym_buffer.buffer_ptrs = [1, 2, 3, 4, 0, 0, 0, 0]
        sym_buffer._gin_max_active_tokens = 37
        sym_buffer._gin_allocation_config = (
            264, 1920, 6, 4096, 2048, 1,
            'fp8xfp8', 'swiglu', 37)
        sym_buffer._gin_transport = object()
        shared_l1, shared_l2 = object(), object()

        self.mega.fp8_fp4_mega_moe(
            object(), object(), object(), sym_buffer,
            shared_l1_weights=shared_l1,
            shared_l2_weights=shared_l2)

        args = self.extension.fp8_fp4_mega_moe_gin.call_args.args
        self.assertIs(args[3], shared_l1)
        self.assertIs(args[4], shared_l2)
        self.assertEqual(args[11:17], (264, 6, 4096, 2048, 1, 'fp8xfp8'))

    def test_gin_launch_requires_bound_transport(self):
        sym_buffer = self.mega.SymmBuffer.__new__(self.mega.SymmBuffer)
        sym_buffer._gin_max_active_tokens = 48
        sym_buffer._gin_transport = None
        with self.assertRaisesRegex(RuntimeError, 'no bound transport'):
            self.mega.fp8_fp4_mega_moe(
                object(), object(), object(), sym_buffer)

    def test_normal_launch_api_preserves_native_path(self):
        sym_buffer = self.mega.SymmBuffer.__new__(self.mega.SymmBuffer)
        sym_buffer._gin_max_active_tokens = None
        sym_buffer.buffer = _FakeBuffer()
        sym_buffer.handle = types.SimpleNamespace(buffer_ptrs=[0x9000])
        sym_buffer.group = _FakeGroup(rank=0, size=1)
        sym_buffer.num_max_tokens_per_rank = 1920
        sym_buffer.num_experts = 56
        sym_buffer.num_topk = 16
        y, l1, l2 = object(), object(), object()

        self.mega.fp8_fp4_mega_moe(y, l1, l2, sym_buffer)

        self.assertEqual(
            self.extension.fp8_fp4_mega_moe.call_args.args[:3],
            (y, l1, l2))
        self.extension.fp8_fp4_mega_moe_gin.assert_not_called()

    def test_transport_binding_rejects_native_workspace(self):
        sym_buffer = self.mega.SymmBuffer.__new__(self.mega.SymmBuffer)
        sym_buffer._gin_max_active_tokens = None
        with self.assertRaisesRegex(TypeError, 'GIN workspace'):
            sym_buffer.bind_gin_transport(object(), object())

    def test_bf16_public_launch_uses_bound_gin_and_all_options(self):
        buffer = self.mega.SymmBuffer.for_gin(
            _FakeGroup(), 24, lsa_size=4, mma_type='bf16xbf16',
            num_experts=136, num_topk=6, hidden=2048,
            intermediate_hidden=1024, num_shared_experts=2)
        handle = types.SimpleNamespace(owner=object())
        self.extension.import_megamoe_gin_transport.return_value = handle
        buffer.bind_gin_transport(object(), object())
        y, l1, l2, s1, s2, stats = (object() for _ in range(6))
        self.mega.bf16_mega_moe(
            y, l1, l2, buffer, s1, s2, stats,
            activation_clamp=1.0, fast_math=False)
        args = self.extension.bf16_mega_moe_gin.call_args.args
        self.assertEqual(args[:7], (y, l1, l2, s1, s2, stats, buffer.buffer))
        self.assertEqual(args[9:17],
                         (1920, 24, 136, 6, 2048, 1024, 2, 'bf16xbf16'))
        self.assertEqual(args[17:21], ('swiglu', 'swiglu', 1.0, False))
        self.assertIs(args[21], handle)
        self.extension.bf16_mega_moe.assert_not_called()

    def test_alias_shares_allocation_binding_and_survives_base_destroy(self):
        group = _FakeGroup()
        base = self.mega.SymmBuffer.for_ep8_gin(group)
        self.mega.symm_mem.empty.reset_mock()
        self.mega.symm_mem.rendezvous.reset_mock()
        self.mega.torch.cuda.synchronize.reset_mock()
        initial_barriers = group.barrier_calls
        alias = self.mega.SymmBuffer.for_gin(
            group, 24, lsa_size=4, base=base,
            num_experts=136, num_topk=6, hidden=2048,
            intermediate_hidden=1024, mma_type='bf16xbf16')
        self.mega.symm_mem.empty.assert_not_called()
        self.mega.symm_mem.rendezvous.assert_not_called()
        self.mega.torch.cuda.synchronize.assert_not_called()
        self.assertEqual(group.barrier_calls, initial_barriers)
        self.assertIs(alias.buffer, base.buffer)
        self.assertIs(alias.handle, base.handle)
        self.assertIs(alias.buffer_ptrs, base.buffer_ptrs)
        owner = object()
        imported = types.SimpleNamespace(owner=owner)
        self.extension.import_megamoe_gin_transport.return_value = imported
        alias.bind_gin_transport(object(), owner)
        self.assertIs(base._gin_transport, imported)
        with self.assertRaisesRegex(RuntimeError, 'already has'):
            base.bind_gin_transport(object(), owner)
        base.destroy()
        self.assertIs(alias._gin_transport, imported)
        self.assertIsNotNone(alias.buffer)
        self.mega.bf16_mega_moe(object(), object(), object(), alias)
        self.assertIs(self.extension.bf16_mega_moe_gin.call_args.args[-1], imported)

    def test_native_constructor_inherits_gin_allocation_on_reuse(self):
        group = _FakeGroup()
        base = self.mega.SymmBuffer.for_ep8_gin(group)
        alias = self.mega.SymmBuffer(
            group, 136, 24, 6, 2048, 1024,
            mma_type='bf16xbf16', base=base)
        self.assertIs(alias._gin_state, base._gin_state)
        self.assertEqual(alias._gin_max_active_tokens, 24)
        self.assertEqual(alias.mma_type, 'bf16xbf16')

    def test_alias_validates_group_capacity_partition_and_lifetime(self):
        group = _FakeGroup()
        base = self.mega.SymmBuffer.for_ep8_gin(group)
        with self.assertRaisesRegex(ValueError, 'invalid'):
            self.mega.SymmBuffer.for_ep8_gin(_FakeGroup(), base=base)
        for lsa_size in (2, 8):
            with self.subTest(lsa_size=lsa_size):
                with self.assertRaisesRegex(ValueError, 'partition'):
                    self.mega.SymmBuffer.for_gin(
                        group, lsa_size=lsa_size, base=base)
        self.extension.get_symm_buffer_size_for_mega_moe_gin.return_value = (
            8192, lambda _: tuple(object() for _ in range(12)))
        with self.assertRaisesRegex(ValueError, 'requires 8192'):
            self.mega.SymmBuffer.for_ep8_gin(group, base=base)
        base.destroy()
        with self.assertRaisesRegex(ValueError, 'invalid'):
            self.mega.SymmBuffer.for_ep8_gin(group, base=base)

    def test_pointer_mapping_supports_more_than_two_lsa_teams(self):
        for world, lsa in ((2, 1), (7, 1), (12, 4), (16, 8), (72, 8)):
            for rank in range(world):
                with self.subTest(world=world, lsa=lsa, rank=rank):
                    handle = types.SimpleNamespace(
                        buffer_ptrs=[0x10000 * (peer + 1) for peer in range(world)],
                        offset=128)
                    pointers = self.mega._make_gin_buffer_ptrs(
                        _FakeBuffer(), handle, rank, world, lsa)
                    for peer in range(world):
                        self.assertEqual(bool(pointers[peer]),
                                         peer // lsa == rank // lsa)
                    self.assertEqual(pointers[rank], 0x9000)

    def test_single_lsa_factory_uses_native_without_gin_build(self):
        self.extension.megamoe_gin_build_info.return_value = {'enabled': False}
        self.mega.torch.empty.return_value = _FakeBuffer(nbytes=1024)
        result = self.mega.SymmBuffer.for_gin(
            _FakeGroup(size=1), 16, lsa_size=1, num_experts=8,
            num_topk=2, hidden=2048, intermediate_hidden=1024,
            mma_type='bf16xbf16')
        self.assertIsNone(result._gin_max_active_tokens)
        self.extension.get_symm_buffer_size_for_mega_moe_gin.assert_not_called()
        self.mega.bf16_mega_moe(object(), object(), object(), result)
        self.extension.bf16_mega_moe.assert_called_once()

    def test_no_parallel_public_gin_types_or_launch(self):
        self.assertFalse(hasattr(self.mega, 'MegaMoEGinSymmBuffer'))
        self.assertFalse(hasattr(self.mega, 'GinTransport'))
        self.assertFalse(hasattr(self.mega, 'fp8_fp4_mega_moe_gin'))


class TestMegaMoeGinCppContract(unittest.TestCase):
    def test_public_descriptor_uses_named_capsule_abi(self):
        header = (_REPO_ROOT / 'deep_gemm' / 'include' / 'deep_gemm' /
                  'comm' / 'mega_moe_gin_transport.h').read_text()
        self.assertIn('deep_gemm.mega_moe_gin_transport.v1', header)
        self.assertIn('DG_MEGAMOE_GIN_TRANSPORT_ABI_VERSION 1u', header)
        self.assertIn('struct DGMegaMoEGinTransportV1 final', header)
        self.assertIn('uint32_t struct_bytes', header)
        self.assertIn('DG_MEGAMOE_GIN_REQUIRED_NCCL_VERSION_CODE 23007u',
                      header)
        self.assertIn('uint32_t dev_comm_bytes', header)
        self.assertIn('uint64_t nccl_headers_fingerprint', header)
        self.assertIn('ncclDevComm_t dev_comm', header)
        self.assertIn('const void* window_base', header)
        self.assertIn('uint64_t window_bytes', header)

    def test_deepgemm_api_never_owns_nccl_resources(self):
        source = (_REPO_ROOT / 'csrc' / 'apis' /
                  'mega_moe_gin.hpp').read_text()
        for forbidden_call in (
                'ncclCommInitRank(', 'ncclCommWindowRegister(',
                'ncclDevCommCreate(', 'ncclDevCommDestroy(',
                'ncclCommWindowDeregister(', 'ncclCommDestroy('):
            with self.subTest(forbidden_call=forbidden_call):
                self.assertNotIn(forbidden_call, source)
        self.assertNotIn('.def("destroy"', source)
        self.assertIn('PyCapsule_IsValid', source)
        self.assertIn('const DGMegaMoEGinTransportV1 descriptor_', source)
        self.assertIn('const py::object owner_', source)
        native_source = (_REPO_ROOT / 'csrc' / 'apis' / 'mega_moe.hpp').read_text()
        self.assertIn('and gin_transport == nullptr', native_source)

    def test_gin_specializes_native_model_geometry_not_transport_constants(self):
        native_api = (_REPO_ROOT / 'csrc' / 'apis' /
                      'mega_moe.hpp').read_text()
        kernel = (_REPO_ROOT / 'deep_gemm' / 'include' / 'deep_gemm' /
                  'impls' / 'sm100_fp8_fp4_mega_moe.cuh').read_text()
        layout = (_REPO_ROOT / 'deep_gemm' / 'include' / 'deep_gemm' /
                  'layout' / 'mega_moe_gin.cuh').read_text()

        self.assertNotIn(
            '\n        DG_HOST_ASSERT(weight_dtype == kPackedFP4);\n',
            native_api)
        for name in ('NumExpertsPerRank', 'Hidden', 'IntermediateHidden'):
            with self.subTest(prototype_constant=name):
                self.assertNotIn('kMegaMoEEp8Gin' + name, native_api)
                self.assertNotIn('kMegaMoEEp8Gin' + name, kernel)
                self.assertNotIn('kMegaMoEEp8Gin' + name, layout)
        for fixed_error in (
                'requires 448 experts', 'requires top-k 16',
                'requires H3584/I3072', 'does not support shared experts',
                'Owner-wave combine requires 56 local experts'):
            with self.subTest(fixed_error=fixed_error):
                self.assertNotIn(fixed_error, kernel)
        self.assertIn('MegaMoeGinWorkspaceT<', kernel)
        protocol = (_REPO_ROOT / 'deep_gemm' / 'include' / 'deep_gemm' /
                    'comm' / 'mega_moe_gin_kernels.cuh').read_text()
        self.assertIn('kExpertUnpackRounds', protocol)
        self.assertIn('kExpertsPerOwnerWave', protocol)
        self.assertIn('struct MegaMoeGinLayout', layout)
        self.assertIn(
            'using MegaMoeGinWorkspace =\n'
            '    MegaMoeGinWorkspaceT<56, 16, 3584, 48>;',
            layout)

    def test_distributed_gate_covers_public_api_and_target_matrix(self):
        spec = importlib.util.spec_from_file_location(
            '_mega_moe_gin_gate_contract',
            _REPO_ROOT / 'tests' / 'run_mega_moe_gin_ep8.py')
        harness = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(harness)

        self.assertEqual(harness.TOKENS, (32, 40, 48))
        self.assertEqual(
            harness.MODES,
            ('balanced', 'half_remote', 'all_remote', 'hot_owner0'))
        self.assertEqual(
            (harness.E, harness.K, harness.H, harness.I, harness.LOCAL_E),
            (448, 16, 3584, 3072, 56))
        config_fields = (
            'name', 'experts_per_rank', 'topk', 'hidden',
            'intermediate_hidden', 'max_active_tokens', 'mma_type',
            'num_shared_experts', 'activation_clamp', 'fast_math')
        as_tuple = lambda config: tuple(
            getattr(config, field) for field in config_fields)
        self.assertEqual(
            as_tuple(harness.DEFAULT),
            ('default', 56, 16, 3584, 3072, 48, 'fp8xfp4', 0,
             None, True))
        self.assertEqual(
            [(as_tuple(config), count, mode)
             for config, count, mode in harness.CAPABILITY_CASES],
            [
                (('dynamic_fp8', 65, 6, 2048, 1024, 64,
                  'fp8xfp8', 0, None, True), 64, 'all_remote'),
                (('shared_fp4', 17, 6, 2048, 1024, 24,
                  'fp8xfp4', 2, 1.0, False), 24, 'masked_remote'),
                (('shared_bf16', 17, 6, 2048, 1024, 24,
                  'bf16xbf16', 2, 1.0, False), 17, 'masked_remote'),
            ])

        for argv, expected in (
                (['run_mega_moe_gin_ep8.py'], False),
                (['run_mega_moe_gin_ep8.py', '--capability-matrix'], True)):
            with self.subTest(argv=argv), \
                    mock.patch.object(sys, 'argv', argv), \
                    mock.patch.object(harness, '_run') as run:
                harness.main()
                self.assertEqual(
                    run.call_args.args[0].capability_matrix, expected)


if __name__ == '__main__':
    unittest.main()
