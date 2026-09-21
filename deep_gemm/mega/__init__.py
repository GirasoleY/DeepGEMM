import torch
import types
import warnings
from typing import Tuple, Optional, Union
from ..utils.math import align

# noinspection PyBroadException
try:
    # noinspection PyProtectedMember
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load mega kernels, please check your PyTorch version: {exception}')

from .. import _C


_MEGAMOE_GIN_WORLD_SIZE = 8
_MEGAMOE_GIN_LSA_SIZE = 4
_MEGAMOE_GIN_DEFAULT_NUM_EXPERTS = 448
_MEGAMOE_GIN_DEFAULT_NUM_TOPK = 16
_MEGAMOE_GIN_DEFAULT_HIDDEN = 3584
_MEGAMOE_GIN_DEFAULT_INTERMEDIATE_HIDDEN = 3072
_MEGAMOE_GIN_DEFAULT_MAX_ACTIVE_TOKENS = 48


class SymmBuffer:
    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 num_shared_experts: int = 0,
                 mma_type: str = 'fp8xfp4',
                 activation: str = 'swiglu',
                 base: Optional['SymmBuffer'] = None):
        if base is not None and getattr(base, '_gin_max_active_tokens', None) is not None:
            alias = type(self).for_gin(
                group, max_active_tokens=num_max_tokens_per_rank,
                lsa_size=base._gin_lsa_size, base=base,
                num_experts=num_experts, num_topk=num_topk,
                hidden=hidden, intermediate_hidden=intermediate_hidden,
                num_shared_experts=num_shared_experts,
                mma_type=mma_type, activation=activation)
            self.__dict__.update(alias.__dict__)
            return
        self._gin_state = None
        self._gin_max_active_tokens = None
        self._gin_transport = None

        # Align token count
        num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_mega_moe())

        # Init
        assert activation == 'swiglu', f'Only `swiglu` activation is supported, got `{activation}`'
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.num_shared_experts = num_shared_experts
        self.mma_type = mma_type
        self.activation = activation

        # Allocate or reuse a symmetric buffer
        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_mega_moe(
            group.size(), num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            mma_type, activation,
            num_shared_experts
        )
        if base is None:
            allocator = torch if group.size() == 1 else symm_mem
            self.buffer = allocator.empty(num_bytes, dtype=torch.int8, device='cuda')
            self.handle = (
                types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
                if group.size() == 1
                else symm_mem.rendezvous(self.buffer, group=group)
            )
            self.buffer.zero_()
            self.group.barrier()
            torch.cuda.synchronize()
        else:
            assert base.buffer is not None and base.handle is not None and base.group is group, \
                'Cannot reuse an invalid symmetric buffer'
            assert num_bytes <= base.buffer.nbytes, \
                (f'The reused Mega MoE config requires {num_bytes} bytes, '
                 f'but the symmetric buffer only has {base.buffer.nbytes} bytes')
            self.buffer = base.buffer
            self.handle = base.handle

        # Create input buffer views
        (self.x, self.x_sf,
         self.topk_idx, self.topk_weights,
         self.shared_l1_acts, self.shared_l1_acts_sf,
         self.shared_l2_acts, self.shared_l2_acts_sf,
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf) = slice_input_buffers(self.buffer)

    @property
    def _gin_transport(self):
        state = getattr(self, '_gin_state', None)
        return state.transport if state is not None else getattr(
            self, '_gin_unbound_transport', None)

    @_gin_transport.setter
    def _gin_transport(self, value):
        state = getattr(self, '_gin_state', None)
        if state is not None:
            state.transport = value
        else:
            self._gin_unbound_transport = value

    def destroy(self):
        # Other aliases retain the allocation and the caller-owned transport.
        self._gin_state = None
        self._gin_transport = None
        self.handle = None
        self.buffer = None
        self.group = None
        self.buffer_ptrs = None
        for name in ('x', 'x_sf', 'topk_idx', 'topk_weights',
                     'shared_l1_acts', 'shared_l1_acts_sf',
                     'shared_l2_acts', 'shared_l2_acts_sf',
                     'l1_acts', 'l1_acts_sf', 'l2_acts', 'l2_acts_sf'):
            setattr(self, name, None)

    @classmethod
    def for_gin(cls, group: dist.ProcessGroup,
                    max_active_tokens: int = _MEGAMOE_GIN_DEFAULT_MAX_ACTIVE_TOKENS,
                    *,
                    lsa_size: int,
                    base: Optional['SymmBuffer'] = None,
                    num_experts: int = _MEGAMOE_GIN_DEFAULT_NUM_EXPERTS,
                    num_topk: int = _MEGAMOE_GIN_DEFAULT_NUM_TOPK,
                    hidden: int = _MEGAMOE_GIN_DEFAULT_HIDDEN,
                    intermediate_hidden: int = _MEGAMOE_GIN_DEFAULT_INTERMEDIATE_HIDDEN,
                    num_shared_experts: int = 0,
                    mma_type: str = 'fp8xfp4',
                    activation: str = 'swiglu'):
        """Allocate a GIN workspace for a native MegaMoE configuration.

        The external NCCL owner must subsequently be attached with
        :meth:`bind_gin_transport` before the normal MegaMoE launch API is used.
        Model geometry and MMA type retain the native MegaMoE constraints;
        FP8xFP4, FP8xFP8 and BF16xBF16 use the same launch APIs and transforms.
        LSA teams must be contiguous and equally sized. One-LSA groups use
        the native path. A GIN base may be reused for any configuration fitting
        its allocation, with the same process group and LSA partition. All
        ranks must use the same geometry and launch sequence. Launches sharing
        a base must be serialized, including captured graphs; StrongVA
        generations persist across aliases.
        """
        world_size = group.size()
        if not (1 <= world_size <= 72 and 1 <= lsa_size <= world_size
                and world_size % lsa_size == 0):
            raise ValueError(
                'MegaMoE GIN requires 1..72 ranks and equal contiguous LSA teams')
        if max_active_tokens < 1:
            raise ValueError('MegaMoE GIN requires max_active_tokens >= 1')
        if (base is not None and
                getattr(base, '_gin_max_active_tokens', None) is not None and
                lsa_size != base._gin_lsa_size):
            raise ValueError('GIN aliases must use the same LSA partition')
        if world_size == lsa_size:
            return cls(group, num_experts, max_active_tokens, num_topk,
                       hidden, intermediate_hidden, num_shared_experts,
                       mma_type, activation, base)
        if not _C.megamoe_gin_build_info()['enabled']:
            raise RuntimeError(
                'MegaMoE GIN unsupported [reason=build_disabled]: rebuild with '
                'DG_MEGAMOE_GIN=1 and matching NCCL 2.30.7 Device API headers')

        token_alignment = _C.get_token_alignment_for_mega_moe()
        num_max_tokens_per_rank = align(max_active_tokens, token_alignment)

        result = cls.__new__(cls)
        result._gin_state = None
        result._gin_lsa_size = lsa_size
        result._gin_max_active_tokens = max_active_tokens
        result._gin_transport = None
        result.group = group
        result.num_experts = num_experts
        result.num_max_tokens_per_rank = num_max_tokens_per_rank
        result.num_topk = num_topk
        result.hidden = hidden
        result.intermediate_hidden = intermediate_hidden
        result.num_shared_experts = num_shared_experts
        result.mma_type = mma_type
        result.activation = activation
        # Keep an immutable copy of the allocation geometry. The public fields
        # are retained for API compatibility, while the launch contract must
        # keep the native buffer and appended GIN tail at identical offsets.
        result._gin_allocation_config = (
            num_experts, num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden, num_shared_experts,
            mma_type, activation, max_active_tokens,
        )

        num_bytes, slice_input_buffers = \
            _C.get_symm_buffer_size_for_mega_moe_gin(
                num_experts, num_max_tokens_per_rank, num_topk,
                hidden, intermediate_hidden, mma_type, activation,
                num_shared_experts, max_active_tokens, world_size, lsa_size)
        if base is None:
            device = torch.device('cuda', torch.cuda.current_device())
            result.buffer = symm_mem.empty(
                num_bytes, dtype=torch.int8, device=device)
            result.handle = symm_mem.rendezvous(result.buffer, group=group)
            result.buffer_ptrs = _make_gin_buffer_ptrs(
                result.buffer, result.handle, group.rank(), world_size, lsa_size)
            result._gin_state = types.SimpleNamespace(
                transport=None, buffer=result.buffer, handle=result.handle,
                buffer_ptrs=result.buffer_ptrs)
        else:
            if (base.buffer is None or base.handle is None or
                    base.group is not group or
                    getattr(base, '_gin_state', None) is None):
                raise ValueError('Cannot reuse an invalid or non-GIN symmetric buffer')
            if num_bytes > base.buffer.nbytes:
                raise ValueError(
                    f'The reused MegaMoE config requires {num_bytes} bytes, '
                    f'but the buffer only has {base.buffer.nbytes} bytes')
            result._gin_state = base._gin_state
            result.buffer = base.buffer
            result.handle = base.handle
            result.buffer_ptrs = base.buffer_ptrs

        (result.x, result.x_sf,
         result.topk_idx, result.topk_weights,
         result.shared_l1_acts, result.shared_l1_acts_sf,
         result.shared_l2_acts, result.shared_l2_acts_sf,
         result.l1_acts, result.l1_acts_sf,
         result.l2_acts, result.l2_acts_sf) = slice_input_buffers(result.buffer)

        # Persistent NIC-updated epochs start at zero and are subsequently
        # advanced only by serialized launches on this workspace.
        if base is None:
            result.buffer.zero_()
            torch.cuda.synchronize()
            group.barrier()
        return result

    @classmethod
    def for_ep8_gin(cls, group: dist.ProcessGroup,
                    max_active_tokens: int = _MEGAMOE_GIN_DEFAULT_MAX_ACTIVE_TOKENS,
                    **kwargs):
        """Compatibility shorthand for GIN with two contiguous LSA4 teams."""
        if group.size() != _MEGAMOE_GIN_WORLD_SIZE:
            raise ValueError('MegaMoE GIN requires an EP8 process group')
        return cls.for_gin(group, max_active_tokens,
                           lsa_size=_MEGAMOE_GIN_LSA_SIZE, **kwargs)

    def bind_gin_transport(self, capsule, owner):
        """Bind caller-owned NCCL resources and retain their owner.

        The owner remains responsible for synchronizing launches and captured
        graphs before explicitly closing its NCCL communicator or window.
        """
        if self._gin_max_active_tokens is None:
            raise TypeError('MegaMoE GIN transport requires a GIN workspace')
        if owner is None:
            raise ValueError('MegaMoE GIN requires a non-None external owner')
        if self.buffer is None or self.handle is None or self.group is None:
            raise RuntimeError('MegaMoE GIN symmetric buffer has been released')
        if self._gin_transport is not None:
            raise RuntimeError('MegaMoE GIN workspace already has a bound transport')
        self._gin_transport = _C.import_megamoe_gin_transport(
            capsule, self.buffer, self.group.rank(), owner)
        return self


def _make_gin_buffer_ptrs(buffer, handle, rank: int,
                          world_size: int, lsa_size: int):
    """Resolve tensor pointers for the local LSA and force cross-LSA GIN."""
    if rank < 0 or rank >= world_size:
        raise ValueError('MegaMoE GIN rank is outside the process group')
    if not hasattr(handle, 'buffer_ptrs') or not hasattr(handle, 'offset'):
        raise RuntimeError(
            'MegaMoE GIN requires a symmetric-memory handle with buffer_ptrs and offset')

    allocation_ptrs = list(handle.buffer_ptrs)
    if len(allocation_ptrs) != world_size:
        raise ValueError(
            'MegaMoE GIN requires one symmetric allocation pointer per rank')

    buffer_offset = int(handle.offset)
    if buffer_offset < 0:
        raise ValueError('MegaMoE GIN symmetric allocation offset must be non-negative')

    lsa_begin = (rank // lsa_size) * lsa_size
    lsa_end = lsa_begin + lsa_size
    tensor_ptrs = [0] * world_size
    for peer in range(lsa_begin, lsa_end):
        allocation_ptr = allocation_ptrs[peer]
        allocation_ptr = 0 if allocation_ptr is None else int(allocation_ptr)
        if allocation_ptr == 0:
            raise RuntimeError(
                f'MegaMoE GIN local-LSA peer {peer} has no symmetric-memory mapping')
        tensor_ptrs[peer] = allocation_ptr + buffer_offset

    # A symmetric-memory handle may report the imported allocation VA for self,
    # which is not necessarily the original tensor VA used by TMA and window
    # registration. Preserve that original pointer explicitly.
    tensor_ptrs[rank] = int(buffer.data_ptr())
    return tensor_ptrs


# TODO: remove this function
def get_symm_buffer_for_mega_moe(group: dist.ProcessGroup,
                                 num_experts: int,
                                 num_max_tokens_per_rank: int, num_topk: int,
                                 hidden: int, intermediate_hidden: int,
                                 num_shared_experts: int = 0,
                                 use_fp8_dispatch: Union[bool, None] = None,
                                 mma_type: str = 'fp8xfp4',
                                 activation: str = 'swiglu') -> SymmBuffer:
    # Backward compat: derive `mma_type` from `use_fp8_dispatch` if provided
    if use_fp8_dispatch is not None:
        assert use_fp8_dispatch == (mma_type.split('x')[0] == 'fp8')
        warnings.warn(
            f'`use_fp8_dispatch` will be deprecated in the future, please use `mma_type`',
            DeprecationWarning, stacklevel=3
        )

    return SymmBuffer(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        num_shared_experts,
        mma_type=mma_type, activation=activation
    )


def _interleave_weights(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    # [gate: 0..7, up: 0..7, gate: 8..15, up: 8..15, ...] instead of [gate | up]
    # Unsqueeze for 2D
    assert t.dim() in (2, 3)
    squeeze_group_dim = t.dim() == 2
    if squeeze_group_dim:
        t = t.unsqueeze(0)

    # Transpose
    g, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up = t[:, half:].reshape(g, half // gran, gran, *rest)
    result = torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(g, n, *rest))
    return result.squeeze(0) if squeeze_group_dim else result


def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    # Unsqueeze for 2D
    assert sf.dtype == torch.int and sf.dim() in (2, 3)
    squeeze_group_dim = sf.dim() == 2
    if squeeze_group_dim:
        sf = sf.unsqueeze(0)

    # Transpose
    num_groups, mn, packed_sf_k = sf.shape
    assert mn % 128 == 0
    result = (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
                .transpose(2, 3)
                .reshape(num_groups, mn, packed_sf_k))
    result = torch.empty_like(sf).copy_(result)
    return result.squeeze(0) if squeeze_group_dim else result


def transform_weights_for_mega_moe(
    l1_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    l2_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    activation: str = 'swiglu'
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
           Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    assert activation == 'swiglu', f'Only `swiglu` activation is supported, got `{activation}`'
    if isinstance(l1_weights, tuple):
        # FP8/FP4: interleave gate/up for weight and SF, then transpose L1 SF for UTCCP
        l1_w = _interleave_weights(l1_weights[0])
        l1_sf = _transpose_sf_for_utccp(_interleave_weights(l1_weights[1]))
        l1_transformed = (l1_w, l1_sf)
        # L2: only transpose SF for UTCCP
        l2_transformed = (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1]))
    else:
        # BF16: L1 interleave gate/up, L2 unchanged
        l1_transformed = _interleave_weights(l1_weights)
        l2_transformed = l2_weights
    return l1_transformed, l2_transformed



def fp8_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: Tuple[torch.Tensor, torch.Tensor],
                     l2_weights: Tuple[torch.Tensor, torch.Tensor],
                     sym_buffer: SymmBuffer,
                     shared_l1_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                     shared_l2_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                     cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                     recipe: Tuple[int, int, int] = (1, 1, 32),
                     activation: str = 'swiglu',
                     activation_clamp: Optional[float] = None,
                     fast_math: bool = True):
    if getattr(sym_buffer, '_gin_max_active_tokens', None) is not None:
        if sym_buffer._gin_transport is None:
            raise RuntimeError(
                'MegaMoE GIN workspace has no bound transport; call '
                'bind_gin_transport(capsule, owner) first')
        (num_experts, num_max_tokens_per_rank, num_topk,
         hidden, intermediate_hidden, num_shared_experts,
         mma_type, allocated_activation, max_active_tokens) = \
            sym_buffer._gin_allocation_config
        _C.fp8_fp4_mega_moe_gin(
            y,
            l1_weights, l2_weights,
            shared_l1_weights, shared_l2_weights,
            cumulative_local_expert_recv_stats,
            sym_buffer.buffer,
            sym_buffer.buffer_ptrs, sym_buffer.group.rank(),
            num_max_tokens_per_rank, max_active_tokens,
            num_experts, num_topk,
            hidden, intermediate_hidden, num_shared_experts, mma_type,
            recipe,
            allocated_activation, activation, activation_clamp,
            fast_math,
            sym_buffer._gin_transport,
        )
        return

    _C.fp8_fp4_mega_moe(
        y,
        l1_weights, l2_weights,
        shared_l1_weights, shared_l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math
    )


def bf16_mega_moe(y: torch.Tensor,
                  l1_weights: torch.Tensor,
                  l2_weights: torch.Tensor,
                  sym_buffer: SymmBuffer,
                  shared_l1_weights: Optional[torch.Tensor] = None,
                  shared_l2_weights: Optional[torch.Tensor] = None,
                  cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                  activation: str = 'swiglu',
                  activation_clamp: Optional[float] = None,
                  fast_math: bool = True):
    if getattr(sym_buffer, '_gin_max_active_tokens', None) is not None:
        if sym_buffer._gin_transport is None:
            raise RuntimeError(
                'MegaMoE GIN workspace has no bound transport; call '
                'bind_gin_transport(capsule, owner) first')
        (num_experts, num_max_tokens_per_rank, num_topk,
         hidden, intermediate_hidden, num_shared_experts,
         mma_type, allocated_activation, max_active_tokens) = \
            sym_buffer._gin_allocation_config
        _C.bf16_mega_moe_gin(
            y,
            l1_weights, l2_weights,
            shared_l1_weights, shared_l2_weights,
            cumulative_local_expert_recv_stats,
            sym_buffer.buffer,
            sym_buffer.buffer_ptrs, sym_buffer.group.rank(),
            num_max_tokens_per_rank, max_active_tokens,
            num_experts, num_topk,
            hidden, intermediate_hidden, num_shared_experts, mma_type,
            allocated_activation, activation, activation_clamp,
            fast_math,
            sym_buffer._gin_transport,
        )
        return

    _C.bf16_mega_moe(
        y,
        l1_weights,
        l2_weights,
        shared_l1_weights,
        shared_l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts,
        sym_buffer.num_topk,
        activation, activation_clamp,
        fast_math
    )
