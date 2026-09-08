"""CPU specification of source-owned packet ordinals and deferred retirement.

This models exact route identity and an ordered scalar FP32/BF16 reduction. It
does not simulate CUDA TMA, compiled barriers, NIC visibility, or elapsed time.
"""

import struct


class ProtocolError(RuntimeError):
    pass


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def bf16(value):
    bits = struct.unpack("I", struct.pack("f", value))[0]
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack("f", struct.pack("I", bits))[0]


def ordered_reduce(values):
    total = 0.0
    for value in values:
        total = f32(total + value)
    return bf16(total)


class DirectReduceGeneration:
    """One source rank; packet rows are grouped by destination-local expert.

Inverse storage intentionally survives begin(): masked/local slots may contain
stale ordinals, but they must never be read. A new eligible remote route always
overwrites its ordinal with the actual current packing order.
"""
    scratch_bytes = 57344 + 2304 + 768 * 4

    def __init__(self, source=0):
        if not 0 <= source < 16:
            raise ValueError("EP16 source required")
        self.source, self.generation, self.retired = source, -1, True
        self.inverse = {}

    def begin(self, generation, ids, values, *, shared=None, enabled=True,
              direct_bulk=True, available_scratch=65536, pack_order=None):
        if not self.retired or generation != self.generation + 1:
            raise ProtocolError("previous generation must retire before reuse")
        ids, values = tuple(map(tuple, ids)), tuple(map(tuple, values))
        if not ids or len(ids) != len(values) or any(len(row) != 16 for row in ids + values):
            raise ValueError("matching nonempty [T,16] routes and values required")
        if any(type(expert) is not int or not -1 <= expert < 896 for row in ids for expert in row):
            raise ValueError("masked or EP16/E896 expert ids required")
        self.generation, self.retired = generation, False
        self.ids, self.values = ids, values
        self.tokens = len(ids)
        self.shared = tuple(shared) if shared is not None else None
        if self.shared is not None and len(self.shared) != self.tokens:
            raise ValueError("one shared-expert contribution per token required")
        self.direct = bool(enabled and direct_bulk and self.tokens <= 48 and
                           available_scratch >= self.scratch_bytes)
        self.packets, self.local, self.expected_counts = {}, {}, {}
        self.inverse_reads, self.inflight_reads, self.completed_reads = set(), set(), set()
        self.world_acquired = self.grid2_passed = self.counts_checked = False
        self.async_proxy_bridged = False
        self.cleanup_done = self.handoff_done = self.scatter_done = False
        self.results = {}
        for owner in range(16):
            routes = [token * 16 + slot for token, row in enumerate(ids)
                      for slot, expert in enumerate(row)
                      if expert >= 0 and expert // 56 == owner]
            if owner // 8 == self.source // 8:
                for destination in routes:
                    self.local[destination] = values[destination // 16][destination % 16]
                continue
            proposed = (pack_order or {}).get(owner, routes)
            if sorted(proposed) != sorted(routes):
                raise ProtocolError("packing must be an exact route permutation")
            # Actual packet construction first groups by expert, preserving the
            # chosen source-staging order within each expert.
            ordered = sorted(proposed, key=lambda d: ids[d // 16][d % 16] % 56)
            packet = []
            for ordinal, destination in enumerate(ordered):
                packet.append((destination, values[destination // 16][destination % 16]))
                if self.direct:
                    self.inverse[destination] = ordinal
            self.packets[owner] = packet
            self.expected_counts[owner] = len(packet)

    def _active(self):
        if self.retired:
            raise ProtocolError("retired generation")

    def acquire_world_put(self):
        self._active()
        self.world_acquired = True

    def grid2(self):
        if not self.world_acquired:
            raise ProtocolError("world Put visibility must precede grid2")
        self.grid2_passed = True

    def check_packet_counts(self):
        if not self.grid2_passed or self.cleanup_done:
            raise ProtocolError("audit preserved dispatch counts before cleanup")
        if any(len(self.packets[p]) != count for p, count in self.expected_counts.items()):
            raise ProtocolError("received packet count disagrees with dispatch")
        self.counts_checked = True

    def scatter_fallback(self):
        if self.direct or not self.counts_checked:
            raise ProtocolError("only fallback scatters after the retained count audit")
        self.scattered = dict(self.local)
        for packet in self.packets.values():
            for destination, value in packet:
                if destination in self.scattered:
                    raise ProtocolError("duplicate packet destination")
                self.scattered[destination] = value
        self.scatter_done = True  # Models scatter publication plus original grid3.

    def bridge_async_proxy(self):
        if not self.direct or not self.grid2_passed or not self.counts_checked:
            raise ProtocolError("direct reader proxy bridge follows target visibility and count audit")
        self.async_proxy_bridged = True

    def begin_read(self, token, slot):
        if not self.counts_checked or not 0 <= token < self.tokens or not 0 <= slot < 16:
            raise ProtocolError("read requires visibility, counts and a valid coordinate")
        expert, destination = self.ids[token][slot], token * 16 + slot
        if expert < 0:
            raise ProtocolError("masked slots must not be read")
        if self.direct and self.handoff_done:
            raise ProtocolError("direct packet read after storage retirement")
        if self.direct and not self.async_proxy_bridged:
            raise ProtocolError("generic visibility alone does not model async-proxy ordering")
        owner = expert // 56
        if not self.direct:
            if not self.scatter_done:
                raise ProtocolError("fallback requires published scatter")
            value = self.scattered[destination]
        elif owner // 8 == self.source // 8:
            value = self.local[destination]
        else:
            self.inverse_reads.add(destination)
            ordinal = self.inverse[destination]
            # This is the immutable received packet header, NOT cleared
            # dispatch source-count workspace.
            packet = self.packets[owner]
            if not 0 <= ordinal < len(packet):
                raise ProtocolError("ordinal outside received packet")
            actual_destination, value = packet[ordinal]
            if actual_destination != destination:
                raise ProtocolError("packet destination disagrees with original top-k slot")
        if destination in self.inflight_reads or destination in self.completed_reads:
            raise ProtocolError("duplicate modeled input read")
        self.inflight_reads.add(destination)
        return value

    def finish_read(self, token, slot):
        destination = token * 16 + slot
        if destination not in self.inflight_reads:
            raise ProtocolError("input completion without issued read")
        self.inflight_reads.remove(destination)
        self.completed_reads.add(destination)

    def reduce(self):
        for token, ids in enumerate(self.ids):
            values = []
            for slot, expert in enumerate(ids):
                if expert >= 0:
                    values.append(self.begin_read(token, slot))
                    self.finish_read(token, slot)
            if self.shared is not None:
                values.append(self.shared[token])
            self.results[token] = ordered_reduce(values)
        return tuple(self.results.values())

    def second_handoff(self):
        if self.handoff_done or not self.counts_checked:
            raise ProtocolError("exactly one second handoff after counts")
        if self.direct and (self.inflight_reads or len(self.results) != self.tokens):
            raise ProtocolError("all packet input reads and token loops precede handoff")
        if not self.direct and not self.scatter_done:
            raise ProtocolError("fallback keeps original scatter/grid3 handoff")
        self.handoff_done = True

    def dispatch_cleanup(self):
        if not self.handoff_done:
            raise ProtocolError("dispatch cleanup cannot overtake input TMA reads")
        self.expected_counts.clear()
        self.cleanup_done = True

    def retire(self):
        if not self.cleanup_done or self.inflight_reads or len(self.results) != self.tokens:
            raise ProtocolError("complete reduction and cleanup before next generation")
        self.retired = True


class PreloadedTokenRows:
    """R5 address-preparation model: one symbolic pointer per warp lane.

The tuple represents 32 distinct lanes, not an array held by each device
thread. Packet metadata is validated once at construction. Later chunk reads
use the retained row address, with original slot order supplied by the caller.
Existing DirectReduceGeneration tests separately cover read retirement.
"""

    def __init__(self, model, token):
        model._active()
        if not model.direct or not model.counts_checked or not model.async_proxy_bridged:
            raise ProtocolError("preload requires eligible direct visibility and proxy bridge")
        if model.handoff_done or not 0 <= token < model.tokens:
            raise ProtocolError("preload requires a live token before retirement")
        self.model, self.generation, self.token = model, model.generation, token
        self.metadata_reads, self.chunk_reads = [], []
        pointers = [None] * 32
        for lane, expert in enumerate(model.ids[token]):
            if expert < 0:
                continue
            destination, owner = token * 16 + lane, expert // 56
            if owner // 8 == model.source // 8:
                pointers[lane] = ("local", destination)
                continue
            model.inverse_reads.add(destination)
            self.metadata_reads.append(("inverse", destination))
            ordinal = model.inverse[destination]
            packet = model.packets[owner]
            self.metadata_reads.append(("count", destination))
            if not 0 <= ordinal < len(packet):
                raise ProtocolError("ordinal outside received packet")
            self.metadata_reads.append(("destination", destination))
            if packet[ordinal][0] != destination:
                raise ProtocolError("packet destination disagrees with original top-k slot")
            pointers[lane] = ("packet", owner, ordinal)
        if model.shared is not None:
            pointers[16] = ("shared", token)
        self.pointers = tuple(pointers)

    def read(self, slot, chunk_byte_offset=0):
        model = self.model
        if model.retired or model.generation != self.generation or model.handoff_done:
            raise ProtocolError("cached row pointer cannot cross storage retirement or generation")
        if not 0 <= slot < 32 or self.pointers[slot] is None:
            raise ProtocolError("masked/inactive lanes must not be selected")
        if chunk_byte_offset < 0 or chunk_byte_offset % 16:
            raise ProtocolError("chunk offset must retain TMA alignment")
        pointer = self.pointers[slot]
        self.chunk_reads.append((slot, chunk_byte_offset))
        if pointer[0] == "local":
            return model.local[pointer[1]]
        if pointer[0] == "shared":
            return model.shared[pointer[1]]
        return model.packets[pointer[1]][pointer[2]][1]
