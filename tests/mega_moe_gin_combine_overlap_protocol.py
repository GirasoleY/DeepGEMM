"""CPU specification of completed-block COMBINE, not CUDA/NIC simulation.

Uses the actual min-peeling-equivalent expert row ordering, full logical-M
blocks (not first fragments), per-peer expert-major return ordinals, and a
fair scan that coalesces only adjacent READY ordinals. The device helper's
release/acquire and ordered producer barrier require separate source/device
validation; this model does not prove those instructions or physical overlap.

Zero-count headers below model the receiver's current-generation zero state,
not extra zero-count network PUTs. The kernel clears receive counts at startup
and publishes late headers only for nonzero peers. Likewise, a model flush of
an empty peer is a vacuous lifetime transition, not a device flush request.
"""

from dataclasses import dataclass


class ProtocolError(RuntimeError):
    pass


def logical_blocks(counts, block_m, remote_peers):
    """Return valid rows (peer, return ordinal) or None for a local row.

    counts[source][expert] is exact. Each expert starts a new logical block;
    a partial last block contains only its valid rows, never padding records.
    """
    if type(block_m) is not int or block_m <= 0 or not counts:
        raise ValueError("positive block M and nonempty exact counts required")
    experts = len(counts[0])
    if not experts or any(len(row) != experts or any(type(n) is not int or n < 0 for n in row)
                          for row in counts):
        raise ValueError("rectangular nonnegative integer source/expert counts required")
    peers = tuple(remote_peers)
    if len(set(peers)) != len(peers) or any(type(p) is not int or not 0 <= p < len(counts) for p in peers):
        raise ValueError("remote peers must be unique valid sources")
    next_ordinal = dict.fromkeys(peers, 0)
    blocks = []
    for expert in range(experts):
        rows = []
        for ordinal in range(max(row[expert] for row in counts)):
            for source, row in enumerate(counts):
                if ordinal >= row[expert]:
                    continue
                if source in next_ordinal:
                    rows.append((source, next_ordinal[source]))
                    next_ordinal[source] += 1
                else:
                    rows.append(None)
        blocks.extend(tuple(rows[start:start + block_m]) for start in range(0, len(rows), block_m))
    return tuple(blocks)


@dataclass(frozen=True)
class Span:
    generation: int
    peer: int
    begin: int
    count: int


class CombineGeneration:
    """One owner's immutable send slabs and generation-scoped ready counters."""
    def __init__(self, peers, n_producers=28, issuing_lanes=32):
        self.peers = tuple(peers)
        if len(set(self.peers)) != len(self.peers) or any(type(p) is not int or p < 0 for p in self.peers):
            raise ValueError("unique nonnegative peers required")
        if any(type(n) is not int or n <= 0 for n in (n_producers, issuing_lanes)):
            raise ValueError("positive producer/lane counts required")
        self.n_producers, self.issuing_lanes = n_producers, issuing_lanes
        self.generation, self.retired = -1, True

    def begin(self, generation, blocks, *, enabled=True, eligible=True, storage_fits=True):
        if not self.retired or generation != self.generation + 1:
            raise ProtocolError("previous generation must fully drain before reuse")
        if any(type(value) is not bool for value in (enabled, eligible, storage_fits)):
            raise ValueError("mode, eligibility and per-rank fit must be booleans")
        blocks = tuple(tuple(block) for block in blocks)
        expected = dict.fromkeys(self.peers, 0)
        for block in blocks:
            if not block:
                raise ProtocolError("empty logical blocks do not exist")
            for record in block:
                if record is None:
                    continue
                peer, ordinal = record
                if peer not in expected or ordinal != expected[peer]:
                    raise ProtocolError("record metadata must preserve exact per-peer contiguous ordinals")
                expected[peer] += 1
        self.generation, self.retired = generation, False
        self.blocks, self.expected = blocks, expected
        self.early = enabled and eligible and storage_fits
        self.released = [set() for _ in blocks]
        self.acquired = [set() for _ in blocks]
        self.claimed, self.issued, self.settled = set(), set(), set()
        self.spans, self.flushed = [], set()
        self.headers, self.headers_settled = {}, set()
        self.final_acquired, self.consumed = False, set()

    def _check(self, generation):
        if self.retired or generation != self.generation:
            raise ProtocolError("stale or retired generation")

    def producer_release(self, block, producer, generation):
        self._check(generation)
        if not 0 <= block < len(self.blocks) or not 0 <= producer < self.n_producers:
            raise ProtocolError("unknown producer/block")
        if producer in self.released[block]:
            raise ProtocolError("duplicate fragment cannot complete a missing producer")
        self.released[block].add(producer)

    def acquire(self, block, lane, generation):
        self._check(generation)
        if not 0 <= lane < self.issuing_lanes or not 0 <= block < len(self.blocks):
            raise ProtocolError("unknown lane/block")
        if len(self.released[block]) == self.n_producers:
            self.acquired[block].add(lane)

    def scan(self, generation, order=None, acquiring_lanes=None):
        """One fair pass; never wait on an unfinished earlier block.

        A disjoint span in ANY lane flushes ALL pending peer spans, as in the
        warp-converged kernel. Every pass submits its tail before retrying.
        """
        self._check(generation)
        if self.headers:
            raise ProtocolError("payload issuance after late headers")
        order = tuple(range(len(self.blocks))) if order is None else tuple(order)
        if sorted(order) != list(range(len(self.blocks))):
            raise ProtocolError("scan must visit every logical block exactly once")
        if not self.early and any(len(ready) != self.n_producers for ready in self.released):
            return ()
        lanes = range(self.issuing_lanes) if acquiring_lanes is None else acquiring_lanes
        lanes = tuple(lanes)
        pending, emitted = {}, []

        def issue_pending():
            for peer, (begin, count) in pending.items():
                span = Span(generation, peer, begin, count)
                records = {(peer, index) for index in range(begin, begin + count)}
                if records & self.issued:
                    raise ProtocolError("record sent more than once")
                if not records <= {(p, i) for block in self.claimed for record in self.blocks[block]
                                   if record is not None for p, i in (record,)}:
                    raise ProtocolError("span contains unfinished or unclaimed holes")
                self.issued.update(records)
                self.spans.append(span)
                emitted.append(span)
            pending.clear()

        for block in order:
            if block in self.claimed:
                continue
            for lane in lanes:
                self.acquire(block, lane, generation)
            if len(self.acquired[block]) != self.issuing_lanes:
                continue
            spans = {}
            for record in self.blocks[block]:
                if record is not None:
                    peer, index = record
                    start, count = spans.get(peer, (index, 0))
                    if index != start + count:
                        raise ProtocolError("block's peer rows are not contiguous")
                    spans[peer] = start, count + 1
            if any(peer in pending and start != sum(pending[peer])
                   for peer, (start, _) in spans.items()):
                issue_pending()
            self.claimed.add(block)
            for peer, (start, count) in spans.items():
                begin, previous = pending.get(peer, (start, 0))
                pending[peer] = begin, previous + count
        issue_pending()
        return tuple(emitted)

    def flush_peer(self, peer, generation):
        self._check(generation)
        if peer not in self.expected or sum(p == peer for p, _ in self.issued) != self.expected[peer]:
            raise ProtocolError("source reuse before all peer records have been submitted")
        self.flushed.add(peer)

    def publish_headers(self, generation):
        self._check(generation)
        if self.headers or len(self.claimed) != len(self.blocks) or self.flushed != set(self.peers):
            raise ProtocolError("late headers require all blocks and local payload completion")
        self.headers = {peer: (generation, count) for peer, count in self.expected.items()}

    def settle_span(self, span):
        self._check(span.generation)
        if span not in self.spans:
            raise ProtocolError("unknown span")
        self.settled.update((span.peer, i) for i in range(span.begin, span.begin + span.count))

    def settle_headers(self, generation):
        self._check(generation)
        if len(self.headers) != len(self.peers):
            raise ProtocolError("headers not issued, including zero-count peers")
        self.headers_settled = set(self.peers)

    def acquire_final_put_barrier(self, generation):
        self._check(generation)
        if self.settled != self.issued or self.headers_settled != set(self.peers) or len(self.claimed) != len(self.blocks):
            raise ProtocolError("source flush/header alone does not prove remote payload settlement")
        self.final_acquired = True

    def consume_peer(self, peer, generation):
        self._check(generation)
        if not self.final_acquired or peer not in self.headers:
            raise ProtocolError("receiver needs final acquire before reading headers or records")
        self.consumed.add(peer)
        return tuple((generation, peer, index) for index in range(self.headers[peer][1]))

    def retire(self):
        if not self.final_acquired or self.consumed != set(self.peers) or self.flushed != set(self.peers):
            raise ProtocolError("no scratch/slab/counter reuse before local completion and remote consumers drain")
        self.retired = True
