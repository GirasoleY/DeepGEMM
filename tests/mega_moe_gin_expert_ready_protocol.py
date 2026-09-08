"""CPU specification of saved-prefix/expert-ready combine, not CUDA simulation.

Descriptors depend only on exact dispatch counts. Producer identities are
explicit here to detect missing/duplicate publications; the real counter is an
integer, not a set. Local completion and remote visibility remain distinct.
Fallback spans below describe logical record coverage, not the number of PUTs
in the unchanged row fallback. ExpertReadyGeneration retains the R1 common
expert-selection model; PeerReadyGeneration specifies R2 peer-local masks.
Neither model simulates warp scheduling, NIC credits, or elapsed time.
"""

from dataclasses import dataclass


class ProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExpertSpan:
    generation: int
    peer: int
    expert: int
    begin: int
    count: int


def saved_dispatch_prefixes(counts, peers):
    """Snapshot exact [source][expert] counts and exclusive source prefixes."""
    counts = tuple(tuple(row) for row in counts)
    if not counts or not counts[0]:
        raise ValueError("nonempty source/expert count matrix required")
    experts = len(counts[0])
    if any(len(row) != experts or
           any(type(value) is not int or value < 0 for value in row)
           for row in counts):
        raise ValueError("rectangular nonnegative integer counts required")
    peers = tuple(peers)
    if len(set(peers)) != len(peers) or any(
            type(peer) is not int or not 0 <= peer < len(counts) for peer in peers):
        raise ValueError("unique in-range source peers required")
    descriptors = {}
    for peer in peers:
        prefix = 0
        for expert, count in enumerate(counts[peer]):
            descriptors[peer, expert] = (prefix, count)
            prefix += count
    totals = tuple(sum(row[expert] for row in counts) for expert in range(experts))
    return counts, descriptors, totals


class ExpertReadyGeneration:
    """Exactly-once whole-expert spans over immutable permanent return slabs."""

    def __init__(self, peers, block_m=32, n_fragments=28, route_capacity=768):
        if any(type(value) is not int or value <= 0
               for value in (block_m, n_fragments, route_capacity)):
            raise ValueError("positive block, fragment and capacity values required")
        self.peers = tuple(peers)
        self.block_m, self.n_fragments = block_m, n_fragments
        self.route_capacity = route_capacity
        self.generation, self.retired = -1, True

    def begin(self, generation, counts, *, enabled=True, eligible=True,
              scratch_fits=True):
        if type(generation) is not int or not self.retired or generation != self.generation + 1:
            raise ProtocolError("previous generation must retire before reuse")
        if any(type(value) is not bool for value in (enabled, eligible, scratch_fits)):
            raise ValueError("mode and eligibility decisions must be Boolean")
        counts, descriptors, totals = saved_dispatch_prefixes(counts, self.peers)
        early = enabled and eligible and scratch_fits
        # A failed eligibility decision is an error, never silent truncation.
        # The unmodified full-packet/row fallback may support larger counts.
        if early and any(sum(row) > self.route_capacity for row in counts):
            raise ProtocolError("direct packet capacity must select fallback")
        self.generation, self.retired, self.early = generation, False, early
        self.counts, self.descriptors, self.totals = counts, descriptors, totals
        self.descriptor_constructions = 1
        self.blocks = tuple((n + self.block_m - 1) // self.block_m for n in totals)
        self.targets = tuple(n * self.n_fragments for n in self.blocks)
        self.releases = [set() for _ in totals]
        self.acquires = [set() for _ in totals]
        self.issued_experts, self.issued_records = set(), set()
        self.spans, self.settled_spans = [], set()
        self.flushed, self.headers, self.headers_settled = set(), {}, False
        self.final_acquired, self.consumed = False, set()
        self.expected = {peer: sum(counts[peer]) for peer in self.peers}

    def _check(self, generation):
        if self.retired or generation != self.generation:
            raise ProtocolError("stale or retired generation")

    def release(self, generation, expert, block, fragment):
        self._check(generation)
        if not 0 <= expert < len(self.totals) or not 0 <= block < self.blocks[expert] or \
                not 0 <= fragment < self.n_fragments:
            raise ProtocolError("unknown expert/block/N-fragment producer")
        producer = (block, fragment)
        if producer in self.releases[expert]:
            raise ProtocolError("duplicate producer cannot complete a missing fragment")
        self.releases[expert].add(producer)

    def acquire(self, generation, expert, peer):
        self._check(generation)
        if not 0 <= expert < len(self.totals) or peer not in self.peers:
            raise ProtocolError("unknown expert or issuing peer")
        ready = len(self.releases[expert]) == self.targets[expert]
        if ready:
            self.acquires[expert].add(peer)
        return ready

    def issue_expert(self, generation, expert):
        self._check(generation)
        if not self.early or self.headers or not 0 <= expert < len(self.totals):
            raise ProtocolError("invalid early publication phase")
        if expert in self.issued_experts:
            raise ProtocolError("expert already published")
        if len(self.releases[expert]) != self.targets[expert]:
            raise ProtocolError("all actual M-block/N-fragment producers required")
        active = {peer for peer in self.peers if self.descriptors[peer, expert][1]}
        if not active <= self.acquires[expert]:
            raise ProtocolError("every actual issuing peer must acquire readiness")
        emitted = []
        for peer in self.peers:
            begin, count = self.descriptors[peer, expert]
            if not count:
                continue
            span = ExpertSpan(generation, peer, expert, begin, count)
            records = {(peer, ordinal) for ordinal in range(begin, begin + count)}
            if records & self.issued_records:
                raise ProtocolError("overlapping or duplicate record span")
            self.issued_records.update(records)
            self.spans.append(span)
            emitted.append(span)
        self.issued_experts.add(expert)
        return tuple(emitted)

    def drain_ready(self, generation, order=None):
        """Ready high experts are emitted even when earlier experts are unready.

        Each selected expert is issued immediately. This models the selection
        semantics, not GPU ballot instruction cost or scheduling fairness.
        """
        self._check(generation)
        if not self.early:
            return ()
        order = tuple(range(len(self.totals))) if order is None else tuple(order)
        if sorted(order) != list(range(len(self.totals))):
            raise ProtocolError("discovery must cover every expert exactly once")
        emitted = []
        for expert in order:
            if expert in self.issued_experts or len(self.releases[expert]) != self.targets[expert]:
                continue
            for peer in self.peers:
                self.acquire(generation, expert, peer)
            emitted.extend(self.issue_expert(generation, expert))
        return tuple(emitted)

    def publish_fallback(self, generation):
        self._check(generation)
        if self.early or self.spans or any(len(done) != target
                                         for done, target in zip(self.releases, self.targets)):
            raise ProtocolError("fallback waits for all producer work")
        for peer, count in self.expected.items():
            if count:
                self.spans.append(ExpertSpan(generation, peer, -1, 0, count))
                self.issued_records.update((peer, ordinal) for ordinal in range(count))

    def flush_payloads(self, generation):
        self._check(generation)
        expected = {(peer, ordinal) for peer, count in self.expected.items()
                    for ordinal in range(count)}
        if self.issued_records != expected:
            raise ProtocolError("source reuse before every actual record is issued")
        self.flushed = {peer for peer, count in self.expected.items() if count}

    def publish_headers(self, generation):
        self._check(generation)
        active = {peer for peer, count in self.expected.items() if count}
        if self.flushed != active or any(len(done) != target
                                        for done, target in zip(self.releases, self.targets)):
            raise ProtocolError("late headers follow payload completion and all producer handoffs")
        self.headers = {peer: count for peer, count in self.expected.items() if count}

    def settle(self, generation, span):
        self._check(generation)
        if span.generation != generation or span not in self.spans:
            raise ProtocolError("foreign or stale span")
        self.settled_spans.add(span)

    def acquire_world_put(self, generation):
        self._check(generation)
        if not self.headers_settled or set(self.spans) != self.settled_spans or \
                self.headers != {peer: count for peer, count in self.expected.items() if count}:
            raise ProtocolError("local flush alone does not prove remote visibility")
        self.final_acquired = True

    def consume(self, generation, peer):
        self._check(generation)
        if not self.final_acquired or peer not in self.peers:
            raise ProtocolError("receive consumption requires final visibility")
        self.consumed.add(peer)

    def retire(self, generation):
        self._check(generation)
        if not self.final_acquired or self.consumed != set(self.peers) or \
                any(len(done) != target for done, target in zip(self.releases, self.targets)):
            raise ProtocolError("producer/consumer work remains before reset")
        self.retired = True


class PeerReadyGeneration(ExpertReadyGeneration):
    """R2: common monotonic readiness discovery, independent peer selection.

    A wave selects at most one ready expert per peer. Each peer issues only its
    own contiguous saved-prefix span and clears its own pending bit afterward.
    Two conceptual full-warp target gathers always occur before issue, even
    when peers select different groups or have no work. Actual CUDA collective
    participation/order is checked separately against the runtime source.
    """

    NUM_EXPERTS = 56
    GROUP_MASKS = (0xffffffff, 0x00ffffff)

    def begin(self, generation, counts, **kwargs):
        counts = tuple(tuple(row) for row in counts)
        if counts and len(counts[0]) > self.NUM_EXPERTS:
            raise ValueError("at most 56 expert mask bits are supported")
        super().begin(generation, counts, **kwargs)
        self.nonempty_masks = {
            peer: tuple(sum(1 << (expert % 32)
                            for expert in range(group * 32,
                                                min((group + 1) * 32, len(self.totals)))
                            if self.descriptors[peer, expert][1])
                        for group in range(2))
            for peer in self.peers
        }
        self.pending_masks = {peer: list(masks)
                              for peer, masks in self.nonempty_masks.items()}
        self.discovered_masks = [0, 0]
        self.target_gathers = []
        self.sent_bookkeeping = set()

    def _validate_masks(self):
        for peer in self.peers:
            for group in range(2):
                pending = self.pending_masks[peer][group]
                if type(pending) is not int or pending < 0 or \
                        pending & ~self.GROUP_MASKS[group] or \
                        pending & ~self.nonempty_masks[peer][group]:
                    raise ProtocolError("pending masks must retain only valid nonempty expert bits")

    def poll_ready_masks(self, generation):
        self._check(generation)
        self._validate_masks()
        for expert, target in enumerate(self.targets):
            # Runtime initializes zero-target discovery as already finished,
            # without setting a ready bit; no peer can need such an expert.
            if target and len(self.releases[expert]) == target:
                self.discovered_masks[expert // 32] |= 1 << (expert % 32)
        return tuple(self.discovered_masks)

    def select_wave(self, generation):
        """Select independently from ready AND pending, never pending alone."""
        ready = self.poll_ready_masks(generation)
        selections = {}
        for peer in self.peers:
            selections[peer] = None
            for group in range(2):
                available = ready[group] & self.pending_masks[peer][group]
                if available:
                    lane = (available & -available).bit_length() - 1
                    selections[peer] = group * 32 + lane
                    break
        # Both group gathers are unconditional; a selected lane may differ for
        # each peer. Out-of-range group-1 lanes and idle peers have safe values.
        gathered = [{}, {}]
        for group in range(2):
            for peer, expert in selections.items():
                lane = 0 if expert is None else expert % 32
                index = group * 32 + lane
                gathered[group][peer] = self.targets[index] if index < len(self.targets) else 0
        self.target_gathers.append(tuple(gathered))
        return {
            peer: (expert, None if expert is None else gathered[expert // 32][peer])
            for peer, expert in selections.items()
        }

    def issue_expert(self, generation, expert):
        raise ProtocolError("R2 issues each peer span independently, not one common expert")

    def issue_peer(self, generation, peer, expert, *, selected_target=None):
        self._check(generation)
        self._validate_masks()
        if not self.early or self.headers or peer not in self.peers or \
                not 0 <= expert < len(self.totals):
            raise ProtocolError("invalid peer publication phase")
        group, bit = expert // 32, 1 << (expert % 32)
        if not self.pending_masks[peer][group] & bit:
            raise ProtocolError("peer/expert span is empty or already published")
        if selected_target is not None and selected_target != self.targets[expert]:
            raise ProtocolError("selected target must come from the correct expert group")
        if len(self.releases[expert]) != self.targets[expert]:
            raise ProtocolError("all actual M-block/N-fragment producers required")
        if peer not in self.acquires[expert]:
            raise ProtocolError("each actual issuer must acquire selected readiness")
        begin, count = self.descriptors[peer, expert]
        span = ExpertSpan(generation, peer, expert, begin, count)
        records = {(peer, ordinal) for ordinal in range(begin, begin + count)}
        if records & self.issued_records:
            raise ProtocolError("overlapping or duplicate record span")
        self.spans.append(span)
        self.issued_records.update(records)
        # Represents helper return, not remote delivery or even local flush.
        self.pending_masks[peer][group] &= ~bit
        return span

    def finish_queuing(self, generation):
        self._check(generation)
        self._validate_masks()
        if any(any(masks) for masks in self.pending_masks.values()):
            raise ProtocolError("every peer must queue every nonempty span")
        # Sent storage is only end-of-drain bookkeeping. A local-only expert
        # may still be computing, so these words cannot imply compute-ready.
        self.sent_bookkeeping = set(range(len(self.totals)))

    def drain_ready(self, generation, peer_order=None):
        self._check(generation)
        if not self.early:
            return ()
        peer_order = self.peers if peer_order is None else tuple(peer_order)
        if len(peer_order) != len(self.peers) or set(peer_order) != set(self.peers):
            raise ProtocolError("wave must visit every peer exactly once")
        emitted = []
        while True:
            selections = self.select_wave(generation)
            if all(expert is None for expert, _ in selections.values()):
                break
            for peer in peer_order:
                expert, target = selections[peer]
                if expert is not None:
                    self.acquire(generation, expert, peer)
                    emitted.append(self.issue_peer(generation, peer, expert,
                                                   selected_target=target))
        if not any(any(masks) for masks in self.pending_masks.values()):
            self.finish_queuing(generation)
        return tuple(emitted)

    def flush_payloads(self, generation):
        if self.early:
            self.finish_queuing(generation)
        super().flush_payloads(generation)
