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


class SymmBuffer:
    _RAW_BUFFER_VIEW_NAMES = (
        'x', 'x_sf',
        'topk_idx', 'topk_weights',
        'shared_l1_acts', 'shared_l1_acts_sf',
        'shared_l2_acts', 'shared_l2_acts_sf',
        'l1_acts', 'l1_acts_sf',
        'l2_acts', 'l2_acts_sf',
    )

    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 num_shared_experts: int = 0,
                 mma_type: str = 'fp8xfp4',
                 activation: str = 'swiglu',
                 enable_gin: bool = False,
                 gin_completion_batch: int = 1,
                 gin_combine_chunk_bytes: int = 7168,
                 gin_outbox_depth: int = 8,
                 gin_queue_depth: int = 64):
        assert activation == 'swiglu', f'Only `swiglu` activation is supported, got `{activation}`'
        if enable_gin:
            # The first fused path has only been made correct for one
            # completion per readiness wave.  Reject other tunings before the
            # symmetric allocation/rendezvous: 2/4/8 remain available to the
            # standalone GIN probes, but must not silently enter MegaMoE.
            if gin_completion_batch != 1:
                raise ValueError(
                    'fused MegaMoE GIN currently requires '
                    'gin_completion_batch=1')
            if mma_type != 'fp8xfp4':
                raise ValueError('the first fused MegaMoE GIN path requires mma_type=fp8xfp4')
            if hidden % 512 != 0:
                raise ValueError('MegaMoE GIN requires hidden to be divisible by 512')
            if (2 * hidden) % gin_combine_chunk_bytes != 0:
                raise ValueError(
                    'MegaMoE GIN combine chunk bytes must divide one BF16 output row')
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden

        # Allocate a symmetric buffer
        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_mega_moe(
            group.size(), num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            mma_type, activation,
            num_shared_experts,
            enable_gin, gin_completion_batch, gin_outbox_depth
        )
        allocator = torch if group.size() == 1 else symm_mem
        # Keep the allocator's communicator lookup key identical to the
        # process-group device key.  In some PyTorch builds, passing the
        # indexless ``cuda`` device allocates on the current GPU but records
        # device index -1 in the NCCL symmetric-memory allocation.  The
        # communicator is registered under the explicit local device index,
        # so rendezvous cannot find it once more than one GPU is involved.
        device = torch.device('cuda', torch.cuda.current_device())
        self.buffer = allocator.empty(num_bytes, dtype=torch.int8, device=device)
        self.handle = (
            types.SimpleNamespace(
                buffer_ptrs=[self.buffer.data_ptr()], offset=0)
            if group.size() == 1
            else symm_mem.rendezvous(self.buffer, group=group)
        )
        # Symmetric-memory handles expose allocation/window bases plus the
        # offset of this tensor inside that allocation.  The MegaMoE kernels
        # expect pointers to the tensor itself, so apply the offset once for
        # every locally addressable peer.  Remote peers remain null and use
        # the GIN transport instead.
        buffer_offset = int(self.handle.offset)
        self.buffer_ptrs = [
            int(ptr) + buffer_offset if int(ptr) != 0 else 0
            for ptr in self.handle.buffer_ptrs
        ]
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

        # Create input buffer views
        (self.x, self.x_sf,
         self.topk_idx, self.topk_weights,
         self.shared_l1_acts, self.shared_l1_acts_sf,
         self.shared_l2_acts, self.shared_l2_acts_sf,
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf) = slice_input_buffers(self.buffer)

        # GIN owns an auxiliary NCCL communicator, a registered view of
        # `self.buffer`, and a device communicator. Keep it after all tensor
        # views so normal construction is byte-for-byte unchanged unless the
        # caller explicitly opts in.
        self._gin_context = None
        self._gin_layout_enabled = enable_gin
        self.gin_completion_batch = gin_completion_batch
        self.gin_combine_chunk_bytes = gin_combine_chunk_bytes
        self.gin_outbox_depth = gin_outbox_depth
        self.gin_queue_depth = gin_queue_depth
        if enable_gin:
            self.enable_gin(queue_depth=gin_queue_depth)

    @property
    def gin_context(self):
        return self._gin_context

    @property
    def gin_enabled(self) -> bool:
        """Whether this buffer currently owns a live GIN transport context."""
        return self._gin_context is not None and self._gin_context.active

    def _require_launchable_transport(self, *, supports_gin: bool):
        if self.buffer is None or self.handle is None or self.group is None:
            raise RuntimeError('MegaMoE symmetric buffer has already been released')
        if self._gin_layout_enabled and not supports_gin:
            raise RuntimeError(
                'BF16 MegaMoE has no GIN transport; refusing cross-host NVLink fallback')
        if self._gin_layout_enabled and not self.gin_enabled:
            raise RuntimeError(
                'MegaMoE GIN context is inactive; refusing legacy NVLink fallback')

    def _collective_validate_gin_config(
            self, build_info, context_count, queue_depth,
            world_barrier_count, expected_lsa_size, required_gin_type):
        context_state = (
            'none' if self._gin_context is None else
            'active' if self._gin_context.active else 'inactive'
        )
        local_config = {
            'protocol_version': 1,
            'build_enabled': bool(build_info['enabled']),
            'compiled_nccl_version': build_info.get('compiled_nccl_version'),
            'required_nccl_version': build_info.get('required_nccl_version'),
            'world_size': self.group.size(),
            'buffer_bytes': self.buffer.numel() * self.buffer.element_size(),
            'num_experts': self.num_experts,
            'num_max_tokens_per_rank': self.num_max_tokens_per_rank,
            'num_topk': self.num_topk,
            'hidden': self.hidden,
            'intermediate_hidden': self.intermediate_hidden,
            'gin_layout_enabled': self._gin_layout_enabled,
            'context_state': context_state,
            'context_count': context_count,
            'layout_queue_depth': self.gin_queue_depth,
            'queue_depth': queue_depth,
            'world_barrier_count': world_barrier_count,
            'expected_lsa_size': expected_lsa_size,
            'required_gin_type': required_gin_type,
            'completion_batch': self.gin_completion_batch,
            'combine_chunk_bytes': self.gin_combine_chunk_bytes,
            'outbox_depth': self.gin_outbox_depth,
        }
        gathered = [None] * self.group.size()
        dist.all_gather_object(gathered, local_config, group=self.group)
        canonical = gathered[0]
        mismatched_ranks = [
            rank for rank, config in enumerate(gathered)
            if config != canonical
        ]
        if mismatched_ranks:
            raise RuntimeError(
                'MegaMoE GIN configuration mismatch across ranks before '
                'auxiliary communicator initialization; mismatched ranks: '
                + ', '.join(map(str, mismatched_ranks)))

        errors = []
        if canonical['world_size'] != 16:
            errors.append('world_size must be exactly 16')
        if not canonical['gin_layout_enabled']:
            errors.append('legacy-sized buffer has no registered GIN workspace')
        if canonical['context_state'] == 'active':
            errors.append('GIN is already enabled for this buffer')
        elif canonical['context_state'] == 'inactive':
            errors.append('destroyed GIN context cannot be re-enabled')
        if canonical['context_count'] < 9:
            errors.append('context_count must be at least 9')
        if canonical['queue_depth'] != canonical['layout_queue_depth']:
            errors.append('queue_depth differs from construction-time layout tuning')
        if canonical['queue_depth'] < 0:
            errors.append('queue_depth must be non-negative')
        if canonical['world_barrier_count'] < 3:
            errors.append('world_barrier_count must be at least 3')
        if canonical['expected_lsa_size'] != 8:
            errors.append('expected_lsa_size must be exactly 8')
        if canonical['required_gin_type'] not in ('gdaki', 'proxy', 'gpi', 'any'):
            errors.append('required_gin_type must be gdaki, proxy, gpi, or any')
        if canonical['completion_batch'] != 1:
            errors.append('completion_batch must be exactly 1 for fused MegaMoE')
        if canonical['combine_chunk_bytes'] not in (256, 1792, 3584, 7168):
            errors.append('combine_chunk_bytes is unsupported')
        if canonical['outbox_depth'] not in (4, 8, 16):
            errors.append('outbox_depth must be 4, 8, or 16')
        if errors:
            raise RuntimeError(
                'MegaMoE GIN collective configuration rejected: '
                + '; '.join(errors))

    def _collective_validate_gin_pointer_aliases(
            self, expected_lsa_size: int):
        """Collectively reject missing aliases in each rank's contiguous LSA.

        PyTorch may expose aliases for ranks outside the local LSA, so those
        entries are deliberately not required to be zero.  A rank-local raise
        here would strand peers in the subsequent UID collectives; gather the
        diagnostics first so every rank takes the same failure path.
        """
        local_error = None
        try:
            rank = self.group.rank()
            world_size = self.group.size()
            pointers = self.buffer_ptrs
            if pointers is None:
                raise RuntimeError('symmetric-memory pointer list is unavailable')

            issues = []
            if len(pointers) != world_size:
                issues.append(
                    f'pointer count {len(pointers)} does not match world size '
                    f'{world_size}')

            lsa_begin = (rank // expected_lsa_size) * expected_lsa_size
            lsa_end = min(lsa_begin + expected_lsa_size, world_size)
            expected_ranks = list(range(lsa_begin, lsa_end))
            missing_ranks = [
                peer for peer in expected_ranks
                if peer >= len(pointers) or int(pointers[peer]) == 0
            ]
            if missing_ranks:
                issues.append(
                    'expected nonzero aliases for contiguous LSA ranks '
                    f'{expected_ranks}, missing ranks {missing_ranks}')
            if issues:
                local_error = '; '.join(issues)
        except BaseException as exception:
            local_error = f'{type(exception).__name__}: {exception}'

        errors = [None] * self.group.size()
        dist.all_gather_object(errors, local_error, group=self.group)
        if any(error is not None for error in errors):
            details = '; '.join(
                f'rank {rank}: {error}' for rank, error in enumerate(errors)
                if error is not None
            )
            raise RuntimeError(
                'MegaMoE GIN symmetric-memory pointer alias validation '
                'failed collectively: ' + details)

    def _collective_get_gin_unique_id(self):
        local_unique_id = bytes(128)
        local_error = None
        try:
            # Every rank validates its local NCCL runtime/header match. Only
            # group rank zero's ID is subsequently broadcast and used.
            local_unique_id = _C.get_megamoe_gin_unique_id()
        except BaseException as exception:
            local_error = f'{type(exception).__name__}: {exception}'

        errors = [None] * self.group.size()
        dist.all_gather_object(errors, local_error, group=self.group)
        if any(error is not None for error in errors):
            details = '; '.join(
                f'rank {rank}: {error}' for rank, error in enumerate(errors)
                if error is not None
            )
            raise RuntimeError(
                'MegaMoE GIN local NCCL preflight failed collectively: ' + details)
        return local_unique_id if self.group.rank() == 0 else bytes(128)

    def enable_gin(self,
                   context_count: int = 9,
                   queue_depth: Optional[int] = None,
                   world_barrier_count: int = 3,
                   expected_lsa_size: int = 8,
                   required_gin_type: str = 'gdaki'):
        """Collectively enable the 2x8 direct-GIN transport for this buffer.

        Every rank in ``self.group`` must call this method in the same order.
        Unsupported NCCL, GIN, or rank/topology configurations fail explicitly;
        this method never falls back to the existing NVLink path.
        """
        if queue_depth is None:
            queue_depth = self.gin_queue_depth

        build_info = _C.megamoe_gin_build_info()
        self._collective_validate_gin_config(
            build_info, context_count, queue_depth, world_barrier_count,
            expected_lsa_size, required_gin_type)
        if not build_info['enabled']:
            # Use the extension stub so C++ and Python callers receive exactly
            # the same actionable build-disabled diagnostic.
            _C.get_megamoe_gin_unique_id()

        self._collective_validate_gin_pointer_aliases(expected_lsa_size)

        rank = self.group.rank()
        unique_id = self._collective_get_gin_unique_id()
        unique_id_tensor = torch.tensor(
            list(unique_id), dtype=torch.uint8, device=self.buffer.device)
        try:
            # PyTorch 2.13 provides group_src, which avoids ambiguity for
            # subgroups whose group rank 0 is not global rank 0.
            dist.broadcast(unique_id_tensor, group=self.group, group_src=0)
        except TypeError:
            # Compatibility for older PyTorch releases.
            global_src = dist.get_global_rank(self.group, 0)
            dist.broadcast(unique_id_tensor, src=global_src, group=self.group)

        unique_id = unique_id_tensor.cpu().numpy().tobytes()
        self._gin_context = _C.create_megamoe_gin_context(
            self.buffer, unique_id, rank, self.group.size(),
            context_count, queue_depth, world_barrier_count,
            expected_lsa_size, required_gin_type,
            self.gin_completion_batch, self.gin_combine_chunk_bytes,
            self.gin_outbox_depth)
        return self._gin_context

    def _release_buffer_storage(self):
        # ``slice_input_buffers`` returns torch::from_blob views.  They do not
        # own the symmetric allocation, and CUDA graph execs retain the same
        # raw addresses independently of these Python objects.  Callers must
        # finish in-flight kernels and retire every captured graph that can be
        # replayed before destroy()/abort(); no view or graph may be used after
        # this method releases the allocation.
        for name in self._RAW_BUFFER_VIEW_NAMES:
            setattr(self, name, None)
        self.handle = None
        self.buffer_ptrs = None
        self.buffer = None
        self.group = None

    def destroy(self):
        if self._gin_context is not None:
            # All kernels must be locally quiescent before all ranks enter the
            # same teardown order. The barrier itself may be stream-backed, so
            # synchronize after it as well.
            torch.cuda.synchronize()
            self.group.barrier()
            torch.cuda.synchronize()
            self._gin_context.destroy()
            self._gin_context._release_buffer_registration()
            self._gin_context = None
        self._release_buffer_storage()

    def abort(self):
        """Release this buffer without collectives after a peer/rank failure."""
        try:
            if self._gin_context is not None:
                try:
                    self._gin_context.abort()
                finally:
                    self._gin_context._release_buffer_registration()
        finally:
            self._gin_context = None
            self._release_buffer_storage()


def get_symm_buffer_for_mega_moe(group: dist.ProcessGroup,
                                 num_experts: int,
                                 num_max_tokens_per_rank: int, num_topk: int,
                                 hidden: int, intermediate_hidden: int,
                                 num_shared_experts: int = 0,
                                 use_fp8_dispatch: Union[bool, None] = None,
                                 mma_type: str = 'fp8xfp4',
                                 activation: str = 'swiglu',
                                 enable_gin: bool = False,
                                 gin_completion_batch: int = 1,
                                 gin_combine_chunk_bytes: int = 7168,
                                 gin_outbox_depth: int = 8,
                                 gin_queue_depth: int = 64) -> SymmBuffer:
    # Align token count
    num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_mega_moe())

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
        mma_type=mma_type, activation=activation,
        enable_gin=enable_gin,
        gin_completion_batch=gin_completion_batch,
        gin_combine_chunk_bytes=gin_combine_chunk_bytes,
        gin_outbox_depth=gin_outbox_depth,
        gin_queue_depth=gin_queue_depth
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
        # FP8: interleave gate/up for weight and SF, then transpose L1 SF for UTCCP
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
    sym_buffer._require_launchable_transport(supports_gin=True)
    _C.fp8_fp4_mega_moe(
        y,
        l1_weights, l2_weights,
        shared_l1_weights, shared_l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.buffer_ptrs, sym_buffer.group.rank(),
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
    sym_buffer._require_launchable_transport(supports_gin=False)
    _C.bf16_mega_moe(
        y,
        l1_weights,
        l2_weights,
        shared_l1_weights,
        shared_l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts,
        sym_buffer.num_topk,
        activation, activation_clamp,
        fast_math
    )
