#!/usr/bin/env python3
"""Host-only contract model for Stage-2 MegaMoE direct GIN dispatch.

The model deliberately has no Torch or CUDA dependency.  It covers the exact
EP16 packet mapping, nondeterministic source atomic order, inactive terminals,
single-buffer generations, the <=48 world-uniform gate, and the zero-growth
workspace aliases used by the device implementation.
"""

from __future__ import annotations

import math
import random
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


NUM_RANKS = 16
LSA_SIZE = 8
EXPERTS_PER_RANK = 56
TOPK = 16
MAX_TOKENS = 48
MAX_ROUTES = MAX_TOKENS * TOPK

READY_BYTES = 16
COUNT_BYTES = EXPERTS_PER_RANK * 8
ROUTE_BYTES = MAX_ROUTES * 4
PACKET_DATA_BYTES = READY_BYTES + COUNT_BYTES + ROUTE_BYTES
PACKET_ALIGNMENT = 128
PACKET_BYTES = (
    (PACKET_DATA_BYTES + PACKET_ALIGNMENT - 1) // PACKET_ALIGNMENT
) * PACKET_ALIGNMENT
PACKET_STORAGE_BYTES = 2 * LSA_SIZE * PACKET_BYTES

HIDDEN = 3584
SF_BYTES = HIDDEN // 32
WEIGHT_BYTES = TOPK * 4
INPUT_BYTES_PER_TOKEN = HIDDEN + SF_BYTES + WEIGHT_BYTES
BULK_RECORD_BYTES = 16 + 2 * HIDDEN
BULK_HEADER_BYTES = 16


@dataclass(frozen=True)
class Route:
    source: int
    token: int
    topk: int
    owner: int
    local_expert: int

    @property
    def token_topk(self) -> int:
        return self.token * TOPK + self.topk


@dataclass(frozen=True)
class DirectPacket:
    source: int
    owner: int
    counts: Tuple[int, ...]
    routes: Tuple[int, ...]
    terminal_epoch: int

    @property
    def publishes_inputs(self) -> bool:
        return bool(self.routes)

    @property
    def control_bytes(self) -> int:
        return COUNT_BYTES + 4 * len(self.routes)


PacketKey = Tuple[int, int]  # (owner rank, remote source lane)
ExpertKey = Tuple[int, int, int]  # (owner, source, local expert)


def _same_lsa(lhs: int, rhs: int) -> bool:
    return lhs // LSA_SIZE == rhs // LSA_SIZE


def _standard_owner(mode: str, source: int, token: int, topk: int,
                    num_tokens: int) -> int:
    source_base = (source // LSA_SIZE) * LSA_SIZE
    local = list(range(source_base, source_base + LSA_SIZE))
    remote_base = (1 - source // LSA_SIZE) * LSA_SIZE
    remote = list(range(remote_base, remote_base + LSA_SIZE))
    if mode == "half_remote":
        route = token * TOPK + topk
        stride = 5
        while math.gcd(stride, num_tokens * TOPK) != 1:
            stride += 2
        mode = (
            "all_remote"
            if (route * stride) % (num_tokens * TOPK) < num_tokens * TOPK // 2
            else "all_same_host"
        )
    source_lane = source % LSA_SIZE
    offset = 1 + (token + topk) % (LSA_SIZE - 1)
    if mode == "all_remote":
        return remote[(source_lane + offset) % LSA_SIZE]
    if mode == "all_same_host":
        return local[(source_lane + offset) % LSA_SIZE]
    raise ValueError(mode)


def _standard_routes(mode: str, token_counts: Sequence[int]) -> List[Route]:
    routes: List[Route] = []
    for source, num_tokens in enumerate(token_counts):
        for token in range(num_tokens):
            for topk in range(TOPK):
                owner = _standard_owner(mode, source, token, topk, num_tokens)
                routes.append(
                    Route(
                        source=source,
                        token=token,
                        topk=topk,
                        owner=owner,
                        local_expert=(token * TOPK + topk) % EXPERTS_PER_RANK,
                    )
                )
    return routes


def _pack_direct(
    routes: Iterable[Route], epochs: Sequence[int], rng: random.Random
) -> Tuple[Dict[PacketKey, DirectPacket], Dict[ExpertKey, Tuple[int, ...]]]:
    """Pack all eight remote-owner terminals for every source.

    The shuffle models the existing source-side atomic slot assignment.  The
    compact packet must preserve that order inside each expert, rather than
    manufacture a new ordering while concatenating experts.
    """
    grouped: Dict[ExpertKey, List[int]] = {}
    for route in routes:
        if _same_lsa(route.source, route.owner):
            continue
        grouped.setdefault(
            (route.owner, route.source, route.local_expert), []
        ).append(route.token_topk)
    for values in grouped.values():
        rng.shuffle(values)

    packets: Dict[PacketKey, DirectPacket] = {}
    for source in range(NUM_RANKS):
        remote_base = (1 - source // LSA_SIZE) * LSA_SIZE
        for owner_lane in range(LSA_SIZE):
            owner = remote_base + owner_lane
            counts: List[int] = []
            packed_routes: List[int] = []
            for local_expert in range(EXPERTS_PER_RANK):
                values = grouped.get((owner, source, local_expert), [])
                counts.append(len(values))
                packed_routes.extend(values)
            if len(packed_routes) > MAX_ROUTES:
                raise OverflowError((source, owner, len(packed_routes)))
            key = (owner, source % LSA_SIZE)
            if key in packets:
                raise AssertionError(f"duplicate receive slab {key}")
            packets[key] = DirectPacket(
                source=source,
                owner=owner,
                counts=tuple(counts),
                routes=tuple(packed_routes),
                terminal_epoch=epochs[source],
            )
    frozen_groups = {key: tuple(values) for key, values in grouped.items()}
    return packets, frozen_groups


def _unpack_direct(
    packets: Mapping[PacketKey, DirectPacket]
) -> Dict[ExpertKey, Tuple[int, ...]]:
    unpacked: Dict[ExpertKey, Tuple[int, ...]] = {}
    for packet in packets.values():
        self_count = sum(packet.counts)
        if self_count != len(packet.routes):
            raise AssertionError("count vector does not cover route payload")
        prefix = 0
        for expert, count in enumerate(packet.counts):
            routes = packet.routes[prefix:prefix + count]
            unpacked[(packet.owner, packet.source, expert)] = routes
            prefix += count
        if prefix != len(packet.routes):
            raise AssertionError("packet has a trailing route payload")
    return unpacked


def _direct_decision(routes: Iterable[Route], token_counts: Sequence[int]) -> bool:
    active = any(not _same_lsa(route.source, route.owner) for route in routes)
    return active and all(count <= MAX_TOKENS for count in token_counts)


def _per_rank_traffic(
    packets: Mapping[PacketKey, DirectPacket], source: int,
    num_tokens: int
) -> Tuple[int, int, int]:
    source_packets = [p for p in packets.values() if p.source == source]
    active = sum(packet.publishes_inputs for packet in source_packets)
    puts = 3 * active + LSA_SIZE
    flushes = LSA_SIZE
    num_bytes = (
        active * num_tokens * INPUT_BYTES_PER_TOKEN
        + sum(packet.control_bytes for packet in source_packets)
    )
    return num_bytes, puts, flushes


class TestMegaMoeDirectDispatchModel(unittest.TestCase):
    def test_exact_layout_aliases_fit_without_growth(self) -> None:
        self.assertEqual(COUNT_BYTES, 448)
        self.assertEqual(ROUTE_BYTES, 3072)
        self.assertEqual(PACKET_DATA_BYTES, 3536)
        self.assertEqual(PACKET_BYTES, 3584)
        self.assertEqual(PACKET_STORAGE_BYTES, 57344)
        self.assertEqual(PACKET_BYTES % PACKET_ALIGNMENT, 0)

        ranges = [
            (packet * PACKET_BYTES, (packet + 1) * PACKET_BYTES)
            for packet in range(2 * LSA_SIZE)
        ]
        self.assertEqual(ranges[0], (0, 3584))
        self.assertEqual(ranges[7][1], 28672)
        self.assertEqual(ranges[8][0], 28672)
        self.assertEqual(ranges[-1][1], PACKET_STORAGE_BYTES)
        self.assertTrue(all(lhs[1] <= rhs[0] for lhs, rhs in zip(ranges, ranges[1:])))
        self.assertTrue(all(start % PACKET_ALIGNMENT == 0 for start, _ in ranges))

        # Actual B300 kernel specialization: 148 SMs, four dispatch warps,
        # completion batch >=1, and one 112-byte scale record.
        minimum_scale_scratch = 148 * 4 * 1 * SF_BYTES
        self.assertEqual(minimum_scale_scratch, 66304)
        self.assertGreaterEqual(minimum_scale_scratch, PACKET_STORAGE_BYTES)
        self.assertEqual(minimum_scale_scratch - PACKET_STORAGE_BYTES, 8960)

        mirror_rows = [
            range(lane * MAX_TOKENS, (lane + 1) * MAX_TOKENS)
            for lane in range(LSA_SIZE)
        ]
        self.assertEqual(mirror_rows[0].start, 0)
        self.assertEqual(mirror_rows[-1].stop, 384)
        self.assertEqual(len({row for rows in mirror_rows for row in rows}), 384)

    def test_packing_preserves_atomic_order_and_owner_mapping(self) -> None:
        rng = random.Random(20260903)
        routes: List[Route] = []
        for source in range(NUM_RANKS):
            num_tokens = rng.randrange(MAX_TOKENS + 1)
            for token in range(num_tokens):
                experts = rng.sample(
                    range(NUM_RANKS * EXPERTS_PER_RANK), TOPK
                )
                for topk, global_expert in enumerate(experts):
                    routes.append(
                        Route(
                            source,
                            token,
                            topk,
                            global_expert // EXPERTS_PER_RANK,
                            global_expert % EXPERTS_PER_RANK,
                        )
                    )
        epochs = [7] * NUM_RANKS
        packets, atomic_order = _pack_direct(
            routes, epochs, random.Random(99)
        )
        unpacked = _unpack_direct(packets)
        self.assertEqual(len(packets), NUM_RANKS * LSA_SIZE)
        for key, expected in atomic_order.items():
            self.assertEqual(unpacked[key], expected)
        for key, packet in packets.items():
            self.assertEqual(packet.terminal_epoch, 7)
            self.assertEqual(len(packet.counts), EXPERTS_PER_RANK)
            self.assertLessEqual(len(packet.routes), MAX_ROUTES)
            self.assertFalse(_same_lsa(packet.source, packet.owner))
            self.assertEqual(key, (packet.owner, packet.source % LSA_SIZE))

    def test_inactive_pairs_still_publish_fresh_terminal_and_zero_counts(self) -> None:
        routes = _standard_routes("all_remote", [32] * NUM_RANKS)
        packets, _ = _pack_direct(routes, [1] * NUM_RANKS, random.Random(1))
        for source in range(NUM_RANKS):
            remote_base = (1 - source // LSA_SIZE) * LSA_SIZE
            paired_owner = remote_base + source % LSA_SIZE
            paired = packets[(paired_owner, source % LSA_SIZE)]
            self.assertFalse(paired.publishes_inputs)
            self.assertEqual(paired.routes, ())
            self.assertEqual(paired.counts, (0,) * EXPERTS_PER_RANK)
            self.assertEqual(paired.control_bytes, COUNT_BYTES)
            self.assertEqual(paired.terminal_epoch, 1)

            active = [
                packet
                for packet in packets.values()
                if packet.source == source and packet.publishes_inputs
            ]
            self.assertEqual(len(active), 7)

    def test_world_uniform_limit_and_generation_transitions(self) -> None:
        epochs = [0] * NUM_RANKS
        ready: MutableMapping[PacketKey, int] = {
            (owner, source_lane): 0
            for owner in range(NUM_RANKS)
            for source_lane in range(LSA_SIZE)
        }

        def launch(num_tokens: int) -> bool:
            token_counts = [num_tokens] * NUM_RANKS
            routes = _standard_routes("all_remote", token_counts)
            direct = _direct_decision(routes, token_counts)
            if not direct:
                return False
            for source in range(NUM_RANKS):
                epochs[source] += 1
            packets, _ = _pack_direct(
                routes, epochs, random.Random(sum(epochs))
            )
            for key, packet in packets.items():
                self.assertEqual(packet.terminal_epoch, epochs[packet.source])
                ready[key] += 1
                self.assertEqual(ready[key], packet.terminal_epoch)
            return True

        self.assertTrue(launch(48))
        self.assertEqual(set(epochs), {1})
        self.assertFalse(launch(64))
        self.assertEqual(set(epochs), {1})
        # Reusing a separately captured <=48 graph advances device-resident
        # counters just like another eager direct invocation.
        self.assertTrue(launch(48))
        self.assertEqual(set(epochs), {2})
        self.assertEqual(set(ready.values()), {2})

        asymmetric_counts = [49] + [32] * (NUM_RANKS - 1)
        asymmetric_routes = _standard_routes("all_remote", asymmetric_counts)
        self.assertFalse(_direct_decision(asymmetric_routes, asymmetric_counts))

    def test_matched_traffic_counts(self) -> None:
        expected = {
            (32, "half_remote"): (846848, 29, 8, 2686064, 36, 15),
            (32, "all_remote"): (847872, 29, 8, 4526192, 36, 15),
            (40, "half_remote"): (1057664, 29, 8, 3356656, 36, 15),
            (40, "all_remote"): (1058944, 29, 8, 5656816, 36, 15),
            (48, "half_remote"): (1268480, 29, 8, 4027248, 36, 15),
            (48, "all_remote"): (1270016, 29, 8, 6787440, 36, 15),
        }
        for (num_tokens, mode), wanted in expected.items():
            with self.subTest(num_tokens=num_tokens, mode=mode):
                routes = _standard_routes(mode, [num_tokens] * NUM_RANKS)
                packets, _ = _pack_direct(
                    routes, [1] * NUM_RANKS, random.Random(num_tokens)
                )
                dispatch_bytes, dispatch_puts, dispatch_flushes = (
                    _per_rank_traffic(packets, 0, num_tokens)
                )
                remote_routes = sum(
                    route.source == 0
                    and not _same_lsa(route.source, route.owner)
                    for route in routes
                )
                active_owners = len({
                    route.owner
                    for route in routes
                    if route.source == 0
                    and not _same_lsa(route.source, route.owner)
                })
                self.assertEqual(active_owners, 7)
                bulk_bytes = (
                    remote_routes * BULK_RECORD_BYTES
                    + active_owners * BULK_HEADER_BYTES
                )
                observed = (
                    dispatch_bytes,
                    dispatch_puts,
                    dispatch_flushes,
                    dispatch_bytes + bulk_bytes,
                    dispatch_puts + active_owners,
                    dispatch_flushes + active_owners,
                )
                self.assertEqual(observed, wanted)

    def test_source_contains_strong_terminal_and_zero_growth_alias(self) -> None:
        root = Path(__file__).resolve().parents[1]
        comm = (root / "deep_gemm/include/deep_gemm/comm/mega_moe_gin.cuh").read_text()
        layout = (root / "deep_gemm/include/deep_gemm/layout/mega_moe.cuh").read_text()
        kernel = (root / "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh").read_text()

        self.assertIn("ncclGin_StrongVASignalInc", comm)
        self.assertIn("mega_moe_gin_wait_direct_dispatch", comm)
        self.assertIn("scale_scratch_buffer.base", layout)
        self.assertIn("kMegaMoeGinDirectDispatchStorageBytes", layout)
        self.assertIn("kMegaMoeGinDirectDispatchPacketAlignment", layout)
        self.assertIn("get_gin_direct_dispatch_epoch_ptr", layout)
        # Direct control is only an alias: workspace extent still ends at the
        # pre-existing bulk return-index buffer and gains no direct tail.
        self.assertIn(
            "bulk_combine_return_index_buffer.get_end_ptr()", layout
        )
        self.assertNotIn("direct_dispatch_packet_tail_buffer", layout)
        self.assertIn("route_count > 0", kernel)
        self.assertIn("get_direct_dispatch_control_bytes(route_count)", kernel)
        self.assertIn("get_src_token_topk_idx_ptr", kernel)
        self.assertIn("use_gin_direct_dispatch", kernel)
        self.assertIn(
            "kMegaMoeGinDirectDispatchPacketAlignment", kernel
        )
        self.assertNotIn("sym_buffer.gin_workspace", kernel)
        aligned_workspace = (
            "buffer.gin_workspace.scale_scratch_buffer.base"
        )
        self.assertIn(aligned_workspace, kernel)
        self.assertLess(
            kernel.index("const auto buffer = layout::MegaMoEBuffer("),
            kernel.index(aligned_workspace),
        )
        # The direct specialization must retain both pieces of the exact r75
        # dispatch fallback for a world-uniform >48-token decision.
        self.assertIn(
            "if (not use_gin_direct_dispatch and lane_idx == 0", kernel
        )
        self.assertIn(
            "if (not use_gin_direct_dispatch and warp_idx == 0)", kernel
        )
        self.assertIn("kGinInputPutBarrierIdx", kernel)
        self.assertIn("kGinInputIngressReadyBarrierIdx", kernel)


if __name__ == "__main__":
    unittest.main()
