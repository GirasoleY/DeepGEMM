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
            (_FakeGroup(), 49),
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

    def test_factory_uses_fixed_native_layout_and_local_lsa_mappings(self):
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
        self.extension.get_symm_buffer_size_for_mega_moe_gin.assert_called_once_with()
        self.assertEqual(group.barrier_calls, 1)
        self.assertEqual(events, ['zero', 'synchronize', 'barrier'])

    def test_pointer_resolution_requires_every_local_lsa_mapping(self):
        handle = types.SimpleNamespace(
            buffer_ptrs=[0x1000] * 8,
            offset=128,
        )
        handle.buffer_ptrs[2] = 0
        with self.assertRaisesRegex(RuntimeError, 'local-LSA peer 2'):
            self.mega._make_ep8_gin_buffer_ptrs(
                _FakeBuffer(), handle, rank=1)

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
        sym_buffer._gin_transport = object()

        y, l1, l2, stats = object(), object(), object(), object()
        self.mega.fp8_fp4_mega_moe(
            y, l1, l2, sym_buffer,
            cumulative_local_expert_recv_stats=stats)

        args = self.extension.fp8_fp4_mega_moe_gin.call_args.args
        self.assertEqual(args[:5], (y, l1, l2, stats, sym_buffer.buffer))
        self.assertEqual(args[5:11],
                         (sym_buffer.buffer_ptrs, 6, 1920, 48, 448, 16))
        self.assertEqual(args[11:15], ((1, 1, 32), 'swiglu', None, True))
        self.assertIs(args[15], sym_buffer._gin_transport)
        self.extension.fp8_fp4_mega_moe.assert_not_called()

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
        with self.assertRaisesRegex(TypeError, 'EP8 GIN workspace'):
            sym_buffer.bind_gin_transport(object(), object())

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


if __name__ == '__main__':
    unittest.main()
