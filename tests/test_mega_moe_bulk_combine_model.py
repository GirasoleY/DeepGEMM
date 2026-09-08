#!/usr/bin/env python3
"""Host-side contract model for the EP16 Stage-1 GIN bulk combine.

This deliberately does not import torch or DeepGEMM.  It models the indexing
contract shared by dispatch, the owner pull scheduler, owner packet packing,
and source packet scatter so failures are attributable to the protocol rather
than to a CUDA build.
"""

from __future__ import annotations

import math
import random
import unittest
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


NUM_RANKS = 16
LSA_SIZE = 8
EXPERTS_PER_RANK = 56
TOPK = 16
BULK_MAX_TOKENS = 48
BULK_CAPACITY = BULK_MAX_TOKENS * TOPK


@dataclass(frozen=True)
class Route:
    source: int
    token: int
    topk: int
    owner: int
    local_expert: int

    @property
    def destination(self) -> int:
        return self.token * TOPK + self.topk


PacketKey = Tuple[int, int]
PacketRecords = Dict[PacketKey, Dict[int, Route]]


def _same_lsa(lhs: int, rhs: int) -> bool:
    return lhs // LSA_SIZE == rhs // LSA_SIZE


def _standard_owner(mode: str, source: int, token: int, topk: int,
                    num_tokens: int) -> int:
    """Mirror the accuracy harness's matched 2x8 routing controls."""
    source_lsa_base = (source // LSA_SIZE) * LSA_SIZE
    local = list(range(source_lsa_base, source_lsa_base + LSA_SIZE))
    remote_lsa_base = (1 - source // LSA_SIZE) * LSA_SIZE
    remote = list(range(remote_lsa_base, remote_lsa_base + LSA_SIZE))

    if mode == "single_remote_source":
        mode = "all_remote" if source == 0 else "all_same_host"
    if mode == "half_remote":
        route = token * TOPK + topk
        total_routes = num_tokens * TOPK
        stride = 5
        while math.gcd(stride, total_routes) != 1:
            stride += 2
        mode = (
            "all_remote"
            if (route * stride) % total_routes < total_routes // 2
            else "all_same_host"
        )

    source_lane = source % LSA_SIZE
    offset = 1 + (token + topk) % (LSA_SIZE - 1)
    if mode == "all_same_host":
        return local[(source_lane + offset) % LSA_SIZE]
    if mode == "all_remote":
        return remote[(source_lane + offset) % LSA_SIZE]
    raise ValueError(f"unsupported standard mode: {mode}")


def _standard_routes(mode: str, token_counts: Sequence[int]) -> List[Route]:
    routes: List[Route] = []
    for source, num_tokens in enumerate(token_counts):
        for token in range(num_tokens):
            for topk in range(TOPK):
                owner = _standard_owner(
                    mode, source, token, topk, num_tokens
                )
                routes.append(
                    Route(
                        source,
                        token,
                        topk,
                        owner,
                        (token * TOPK + topk) % EXPERTS_PER_RANK,
                    )
                )
    return routes


def _random_routes(token_counts: Sequence[int], rng: random.Random) -> List[Route]:
    routes: List[Route] = []
    num_experts = NUM_RANKS * EXPERTS_PER_RANK
    for source, num_tokens in enumerate(token_counts):
        for token in range(num_tokens):
            # Top-k experts are unique, matching the kernel's route contract.
            for topk, global_expert in enumerate(
                rng.sample(range(num_experts), TOPK)
            ):
                routes.append(
                    Route(
                        source,
                        token,
                        topk,
                        global_expert // EXPERTS_PER_RANK,
                        global_expert % EXPERTS_PER_RANK,
                    )
                )
    return routes


def _owner_scheduler_order(counts: Sequence[int]) -> List[Tuple[int, int]]:
    """Model MegaMoE's min-peeling rank interleave for one expert.

    Each result is ``(source_rank, slot_within_source_and_expert)``.  The
    interleave changes pool order but must enumerate every source slot once.
    """
    remaining = list(counts)
    source_slot_base = 0
    result: List[Tuple[int, int]] = []
    while True:
        active = [source for source, count in enumerate(remaining) if count]
        if not active:
            return result
        length = min(remaining[source] for source in active)
        for slot_delta in range(length):
            for source in active:
                result.append((source, source_slot_base + slot_delta))
        source_slot_base += length
        for source in active:
            remaining[source] -= length


def _pack_owner_packets(routes: Iterable[Route], rng: random.Random) -> PacketRecords:
    """Pack owner send packets using the kernel's return-index formula.

    The source dispatch atomics may assign an arbitrary slot order within one
    (source, owner, expert) group.  Shuffling each group ensures the protocol
    does not accidentally depend on a deterministic atomic order.
    """
    grouped: Dict[Tuple[int, int, int], List[Route]] = {}
    all_routes = list(routes)
    for route in all_routes:
        grouped.setdefault(
            (route.source, route.owner, route.local_expert), []
        ).append(route)

    source_slots: Dict[Tuple[int, int, int, int], Route] = {}
    for group_key, group_routes in grouped.items():
        rng.shuffle(group_routes)
        for source_slot, route in enumerate(group_routes):
            source_slots[group_key + (source_slot,)] = route

    packets: PacketRecords = {}
    for owner in range(NUM_RANKS):
        prefix = [0] * NUM_RANKS
        for local_expert in range(EXPERTS_PER_RANK):
            counts = [
                len(grouped.get((source, owner, local_expert), ()))
                for source in range(NUM_RANKS)
            ]
            scheduled = _owner_scheduler_order(counts)
            _assert_scheduler_bijection(counts, scheduled)
            for source, source_slot in scheduled:
                route = source_slots[
                    (source, owner, local_expert, source_slot)
                ]
                if not _same_lsa(owner, source):
                    return_idx = prefix[source] + source_slot
                    if return_idx >= BULK_CAPACITY:
                        raise OverflowError(
                            f"owner={owner} source={source} return_idx={return_idx}"
                        )
                    # Owner send packets are indexed by the remote source lane.
                    packet = packets.setdefault(
                        (owner, source % LSA_SIZE), {}
                    )
                    if return_idx in packet:
                        raise AssertionError(
                            f"duplicate owner/source return_idx {return_idx}"
                        )
                    packet[return_idx] = route
            for source, count in enumerate(counts):
                prefix[source] += count
    return packets


def _assert_scheduler_bijection(
    counts: Sequence[int], scheduled: Sequence[Tuple[int, int]]
) -> None:
    expected = [
        (source, source_slot)
        for source, count in enumerate(counts)
        for source_slot in range(count)
    ]
    if sorted(scheduled) != expected:
        raise AssertionError(
            f"owner scheduler is not bijective: counts={counts}"
        )


def _transfer_to_source_recv(
    owner_packets: Mapping[PacketKey, Mapping[int, Route]],
) -> PacketRecords:
    """Apply the exact owner send-slot -> source receive-slot mapping."""
    received: PacketRecords = {}
    for (owner, source_lane), records in owner_packets.items():
        remote_lsa_base = (1 - owner // LSA_SIZE) * LSA_SIZE
        source = remote_lsa_base + source_lane
        recv_key = (source, owner % LSA_SIZE)
        if recv_key in received:
            raise AssertionError(f"duplicate receive packet {recv_key}")
        for route in records.values():
            if route.owner != owner or route.source != source:
                raise AssertionError(
                    "owner send slot did not map to the intended remote source"
                )
        received[recv_key] = dict(records)
    return received


def _bulk_decision(
    routes: Iterable[Route], token_counts: Sequence[int]
) -> Tuple[bool, bool, bool]:
    active = any(not _same_lsa(route.source, route.owner) for route in routes)
    ineligible = any(count > BULK_MAX_TOKENS for count in token_counts)
    return active, ineligible, active and not ineligible


def _expected_remote_destinations(
    routes: Iterable[Route],
) -> Dict[PacketKey, List[int]]:
    expected: Dict[PacketKey, List[int]] = {}
    for route in routes:
        if not _same_lsa(route.source, route.owner):
            expected.setdefault(
                (route.source, route.owner % LSA_SIZE), []
            ).append(route.destination)
    return expected


class TestMegaMoeBulkCombineModel(unittest.TestCase):
    def _check_model(self, routes: Sequence[Route], seed: int) -> PacketRecords:
        packets = _pack_owner_packets(routes, random.Random(seed))
        received = _transfer_to_source_recv(packets)
        expected = _expected_remote_destinations(routes)

        self.assertEqual(set(received), set(expected))
        for recv_key, expected_destinations in expected.items():
            records = received[recv_key]
            # Prefix + source slot must densely cover the packet, with no holes.
            self.assertEqual(sorted(records), list(range(len(records))))
            self.assertLessEqual(len(records), BULK_CAPACITY)

            shipped_destinations = [
                route.destination for route in records.values()
            ]
            self.assertCountEqual(shipped_destinations, expected_destinations)
            reconstructed = {}
            for destination in shipped_destinations:
                token, topk = divmod(destination, TOPK)
                self.assertGreaterEqual(token, 0)
                self.assertGreaterEqual(topk, 0)
                self.assertLess(topk, TOPK)
                # The record header reconstructs the exact source output slot.
                self.assertIn(destination, expected_destinations)
                self.assertNotIn((token, topk), reconstructed)
                reconstructed[(token, topk)] = destination
            self.assertEqual(
                set(reconstructed.values()), set(expected_destinations)
            )
        return received

    def test_standard_routes_have_dense_bijective_packets(self) -> None:
        for num_tokens in (1, 24, 30, 32, 36, 47, 48):
            token_counts = [num_tokens] * NUM_RANKS
            for mode in (
                "all_same_host",
                "half_remote",
                "all_remote",
                "single_remote_source",
            ):
                with self.subTest(num_tokens=num_tokens, mode=mode):
                    routes = _standard_routes(mode, token_counts)
                    num_remote = sum(
                        not _same_lsa(route.source, route.owner)
                        for route in routes
                    )
                    total_routes = NUM_RANKS * num_tokens * TOPK
                    expected_remote = {
                        "all_same_host": 0,
                        "half_remote": total_routes // 2,
                        "all_remote": total_routes,
                        "single_remote_source": num_tokens * TOPK,
                    }[mode]
                    self.assertEqual(num_remote, expected_remote)
                    received = self._check_model(
                        routes, seed=10_000 + num_tokens
                    )
                    if mode == "all_same_host":
                        self.assertEqual(received, {})

    def test_seeded_random_variable_token_matrices(self) -> None:
        rng = random.Random(20260903)
        for scenario in range(64):
            token_counts = [
                rng.randrange(BULK_MAX_TOKENS + 1)
                for _ in range(NUM_RANKS)
            ]
            routes = _random_routes(token_counts, rng)
            with self.subTest(scenario=scenario, token_counts=token_counts):
                active, ineligible, use_bulk = _bulk_decision(
                    routes, token_counts
                )
                self.assertFalse(ineligible)
                self.assertEqual(use_bulk, active)
                self._check_model(routes, seed=30_000 + scenario)

    def test_exact_owner_send_to_source_receive_slot_mapping(self) -> None:
        token_counts = [13, 7, 29, 1, 48, 0, 31, 11,
                        5, 42, 2, 33, 17, 9, 24, 46]
        routes = _random_routes(token_counts, random.Random(17))
        sends = _pack_owner_packets(routes, random.Random(23))
        receives = _transfer_to_source_recv(sends)
        for (owner, source_lane), send_records in sends.items():
            source = (1 - owner // LSA_SIZE) * LSA_SIZE + source_lane
            self.assertEqual(
                receives[(source, owner % LSA_SIZE)], send_records
            )

    def test_capacity_boundary_and_world_uniform_fallback(self) -> None:
        # Worst case for one owner/source pair: all 48*16 routes select owner 8.
        routes = [
            Route(0, token, topk, 8, (token * TOPK + topk) % EXPERTS_PER_RANK)
            for token in range(BULK_MAX_TOKENS)
            for topk in range(TOPK)
        ]
        received = self._check_model(routes, seed=768)
        records = received[(0, 0)]
        self.assertEqual(len(records), BULK_CAPACITY)
        self.assertEqual(min(records), 0)
        self.assertEqual(max(records), BULK_CAPACITY - 1)

        # A >48 rank forces fallback even if that rank itself has only local
        # routes while a different rank makes the launch globally active.
        token_counts = [49] + [32] * (NUM_RANKS - 1)
        mixed_routes = [
            Route(0, 0, 0, 1, 0),  # The ineligible rank has no remote route.
            Route(1, 0, 0, 8, 0),  # A different rank makes the world active.
        ]
        active, ineligible, use_bulk = _bulk_decision(
            mixed_routes, token_counts
        )
        self.assertTrue(active)
        self.assertTrue(ineligible)
        self.assertFalse(use_bulk)

        overflow_routes = [
            Route(0, token, topk, 8, (token * TOPK + topk) % EXPERTS_PER_RANK)
            for token in range(BULK_MAX_TOKENS + 1)
            for topk in range(TOPK)
        ]
        with self.assertRaises(OverflowError):
            _pack_owner_packets(overflow_routes, random.Random(769))

    def test_bulk_local_fallback_bulk_does_not_consume_stale_headers(self) -> None:
        header_counts: MutableMapping[PacketKey, int] = {
            (source, owner_lane): 0xBAD000 + source * LSA_SIZE + owner_lane
            for source in range(NUM_RANKS)
            for owner_lane in range(LSA_SIZE)
        }

        def launch(mode: str, token_counts: Sequence[int]) -> None:
            nonlocal header_counts
            routes = _standard_routes(mode, token_counts)
            active, _ineligible, use_bulk = _bulk_decision(
                routes, token_counts
            )
            if use_bulk:
                # The kernel clears every receive count before any owner PUT.
                header_counts = {
                    (source, owner_lane): 0
                    for source in range(NUM_RANKS)
                    for owner_lane in range(LSA_SIZE)
                }
                received = self._check_model(routes, seed=sum(token_counts))
                for recv_key, records in received.items():
                    header_counts[recv_key] = len(records)
                expected = _expected_remote_destinations(routes)
                for recv_key, count in header_counts.items():
                    self.assertEqual(count, len(expected.get(recv_key, ())))
            else:
                # Local and legacy fallback paths must not inspect packet
                # headers; legacy row-outbox traffic may overwrite their bytes.
                for recv_key in header_counts:
                    header_counts[recv_key] = (
                        header_counts[recv_key] ^ 0x5A5A5A5A
                    ) & 0xFFFFFFFF
                self.assertFalse(use_bulk)
                self.assertIn(active, (False, True))

        launch("all_remote", [32] * NUM_RANKS)
        launch("all_same_host", [32] * NUM_RANKS)
        launch("all_remote", [64] + [32] * (NUM_RANKS - 1))
        launch("single_remote_source", [48] * NUM_RANKS)

        # The final eligible launch cleared all corruption.  Only source zero's
        # seven non-paired remote owners have nonzero receive counts.
        nonzero = {key for key, count in header_counts.items() if count}
        self.assertEqual(
            nonzero,
            {(0, owner_lane) for owner_lane in range(1, LSA_SIZE)},
        )


if __name__ == "__main__":
    unittest.main()
