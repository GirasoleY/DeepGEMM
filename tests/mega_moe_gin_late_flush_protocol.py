"""R7 completion-order model over the unchanged R3 ready-span model.

No NIC/warp scheduling or elapsed-time simulation. Shared SQ reservations,
contiguous ready publication, Default submission, local CQ completion and
target visibility are distinct events. Queue counters persist across model
generations. Earlier R1/R2/R3 completion models remain unchanged.
The aggregate-input/combine interleaving is a backend stress case, not a claim
that the R5-based runtime moved input posting after its pre-pull rendezvous.
"""

from dataclasses import dataclass

from mega_moe_gin_expert_ready_protocol import ProtocolError, ReadyCoalesceGeneration


@dataclass(frozen=True)
class Ticket:
    context: int
    peer: int
    index: int


@dataclass(frozen=True)
class FlushSnapshot:
    context: int
    peer: int
    upper: int


class SharedSQ:
    """One GPU-shared (context, peer) SQ, not an issuer-private request list."""

    def __init__(self, context, peer):
        self.context, self.peer = context, peer
        self.reserved, self.constructed = [], set()
        self.ready = self.submitted = self.completed = 0

    def reserve(self, kind):
        ticket = Ticket(self.context, self.peer, len(self.reserved))
        self.reserved.append(kind)
        return ticket

    def _ticket(self, ticket):
        if (ticket.context, ticket.peer) != (self.context, self.peer) or not 0 <= ticket.index < len(self.reserved):
            raise ProtocolError("foreign SQ ticket")

    def construct(self, ticket):
        self._ticket(ticket)
        if ticket.index in self.constructed:
            raise ProtocolError("duplicate WQE construction")
        self.constructed.add(ticket.index)

    def finish_post(self, ticket, *, aggregate=False):
        self._ticket(ticket)
        if ticket.index not in self.constructed or ticket.index != self.ready:
            raise ProtocolError("posting waits for a contiguous constructed ready prefix")
        self.ready += 1
        if not aggregate:
            self.submitted = self.ready
        return ticket

    def post(self, kind, *, aggregate=False):
        ticket = self.reserve(kind)
        self.construct(ticket)
        return self.finish_post(ticket, aggregate=aggregate)

    def complete(self, upper=None):
        upper = self.submitted if upper is None else upper
        if type(upper) is not int or not self.completed <= upper <= self.submitted:
            raise ProtocolError("CQ cannot complete an unsubmitted WQE")
        self.completed = upper

    def flush(self):
        # Mirrors GDAKI's PUT-only shared sq_rsvd_index snapshot. It does NOT
        # submit an open aggregate chain or establish target visibility.
        return FlushSnapshot(self.context, self.peer, len(self.reserved))

    def wait(self, snapshot):
        if (snapshot.context, snapshot.peer) != (self.context, self.peer) or \
                type(snapshot.upper) is not int or not 0 <= snapshot.upper <= len(self.reserved) or \
                self.completed < snapshot.upper:
            raise ProtocolError("wrong SQ or local completion is still pending")


class LateFlushGeneration(ReadyCoalesceGeneration):
    """Eligible R7 spans retire through the later header's same-SQ snapshot."""

    def __init__(self, peers, **kwargs):
        super().__init__(peers, **kwargs)
        self.queues = {}

    def begin(self, generation, counts, **kwargs):
        super().begin(generation, counts, **kwargs)
        self.input_last, self.input_retired = {}, set()
        self.span_tickets, self.header_tickets = {}, {}
        self.joined = False

    def queue(self, peer, context=1):
        if peer not in self.peers:
            raise ProtocolError("unknown peer")
        return self.queues.setdefault((context, peer), SharedSQ(context, peer))

    def post_input(self, generation, peer, kind, *, aggregate=False):
        self._check(generation)
        if self.joined or peer in self.input_retired:
            raise ProtocolError("input posting after retirement")
        ticket = self.queue(peer, 1 + peer % 8).post("input:" + kind, aggregate=aggregate)
        self.input_last[peer] = ticket
        return ticket

    def retire_input(self, generation, peer, snapshot, *, inbound_terminal_acquired):
        self._check(generation)
        if inbound_terminal_acquired is not True or peer not in self.input_last:
            raise ProtocolError("all inbound payload terminals, including empty peers, are required")
        queue = self.queue(peer, 1 + peer % 8)
        queue.wait(snapshot)
        if snapshot.upper <= self.input_last[peer].index:
            raise ProtocolError("input completion snapshot is too early")
        self.input_retired.add(peer)

    def issue_batch(self, generation, batch, on_submit=None):
        def submit(model, span):
            self.span_tickets[span] = self.queue(span.peer).post("combine_payload")
            if on_submit is not None:
                on_submit(model, span)
        return super().issue_batch(generation, batch, on_submit=submit)

    def join_dispatch_and_epilogue(self, generation):
        self._check(generation)
        self.finish_queuing(generation)
        if self.input_retired != set(self.peers) or any(
                len(done) != target for done, target in zip(self.releases, self.targets)):
            raise ProtocolError("grid1 needs every dispatch/input and producer handoff")
        self.joined = True

    def flush_payloads(self, generation):
        if self.early:
            raise ProtocolError("R7 has no eligible early payload-only flush")
        return super().flush_payloads(generation)

    def publish_headers(self, generation):
        self._check(generation)
        if not self.early:
            return super().publish_headers(generation)
        if not self.joined or self.headers:
            raise ProtocolError("late header requires the queued-post handoff/grid1")
        self.headers = {peer: count for peer, count in self.expected.items() if count}
        for peer in self.headers:
            self.header_tickets[peer] = self.queue(peer).post("combine_header_16B")

    def finish_header_flush(self, generation, peer, snapshot):
        self._check(generation)
        if peer not in self.header_tickets or snapshot.upper <= self.header_tickets[peer].index:
            raise ProtocolError("header completion snapshot must include its header")
        self.queue(peer).wait(snapshot)
        for span, ticket in self.span_tickets.items():
            if span.peer == peer and ticket.index >= self.header_tickets[peer].index:
                raise ProtocolError("a payload was issued after its late header")
        self.flushed.add(peer)

    def check_send_source_reusable(self, generation):
        self._check(generation)
        if self.flushed != {peer for peer, count in self.expected.items() if count}:
            raise ProtocolError("send source still referenced by an incomplete header/payload snapshot")

    def acquire_world_put(self, generation):
        if self.early:
            self.check_send_source_reusable(generation)
        # Remote settlement and consumer retirement remain distinct from CQ.
        return super().acquire_world_put(generation)
