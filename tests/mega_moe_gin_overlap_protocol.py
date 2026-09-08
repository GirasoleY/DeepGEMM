"""CPU specification for split-control GIN dispatch; not a device simulator.

The model expresses the required ordering, not CUDA/NIC memory-model proof.
Counts determine exact expert-assignment tile membership. A control terminal
does not make activation/SF/weight bytes readable; their separate terminal
must be acquired. Local flush completion permits sender reuse, not receiver
loads. Consumer completion permits ring scratch reuse, not merely tile issue.
"""

from dataclasses import dataclass


PAYLOAD_COMPONENTS = frozenset(("activation", "scale", "weight"))


class ProtocolError(RuntimeError):
    pass


@dataclass
class TerminalEpochs:
    """Persistent per-peer counters; fallback does not post terminal1."""

    control: int = 0
    payload: int = 0

    def advance(self, split, direct_eligible=True):
        if type(split) is not bool or type(direct_eligible) is not bool:
            raise ValueError("split selection must be a collectively agreed boolean")
        if not direct_eligible:
            return None, None
        self.control += 1
        if split:
            self.payload += 1
        return self.control, self.payload if split else None


class DispatchGeneration:
    """One owner's abstract receive state with round-robin expert tiles.

    Assignment identity is (source, expert, ordinal), independent of delivery
    order. Payload tags include generation, preventing stale data from passing
    merely because two launches have identical expert assignment counts.
    """

    def __init__(self, sources=8, experts=56, block_m=32, scratch_slots=2):
        if any(type(value) is not int or value <= 0
               for value in (sources, experts, block_m, scratch_slots)):
            raise ValueError("model dimensions must be positive integers")
        self.sources, self.experts = sources, experts
        self.block_m, self.scratch_slots = block_m, scratch_slots
        self.generation = -1
        self.retired = True
        self.begin(0)

    def begin(self, generation):
        if not self.retired or generation != self.generation + 1:
            raise ProtocolError("previous generation must retire before reuse")
        self.generation, self.retired = generation, False
        self.controls, self.payloads = {}, {}
        self.control_acquired, self.payload_settled = set(), {}
        self.payload_signaled, self.payload_acquired = set(), set()
        self.local_send_complete = set()
        self.issued, self.completed, self.scratch = set(), set(), {}

    def _check(self, source, generation):
        if self.retired or generation != self.generation:
            raise ProtocolError("stale or retired generation")
        if type(source) is not int or not 0 <= source < self.sources:
            raise ProtocolError("unknown source")

    def publish_control(self, source, counts, generation):
        self._check(source, generation)
        if source in self.controls:
            raise ProtocolError("duplicate control")
        counts = tuple(counts)
        if len(counts) != self.experts or any(type(n) is not int or n < 0 for n in counts):
            raise ProtocolError("control needs an exact nonnegative count per expert")
        self.controls[source] = counts

    def acquire_control(self, source, generation):
        self._check(source, generation)
        if source not in self.controls:
            raise ProtocolError("control terminal not available")
        self.control_acquired.add(source)

    def expected_payload(self, source):
        if source not in self.controls:
            raise ProtocolError("payload issued before control")
        return tuple((self.generation, source, expert, ordinal)
                     for expert, count in enumerate(self.controls[source])
                     for ordinal in range(count))

    def issue_payload(self, source, generation, values=None):
        self._check(source, generation)
        if source in self.payloads:
            raise ProtocolError("duplicate payload")
        expected = self.expected_payload(source)
        values = expected if values is None else tuple(values)
        if values != expected:
            raise ProtocolError("payload does not match exact control membership/generation")
        self.payloads[source] = values

    def finish_local_send(self, source, generation):
        self._check(source, generation)
        if source not in self.payloads:
            raise ProtocolError("flush completed before send issue")
        self.local_send_complete.add(source)

    def sender_can_reuse(self, source):
        return source in self.local_send_complete

    def settle_payload(self, source, generation, components=PAYLOAD_COMPONENTS):
        self._check(source, generation)
        if source not in self.payloads:
            raise ProtocolError("payload has not been issued")
        components = set(components)
        if not components <= PAYLOAD_COMPONENTS:
            raise ProtocolError("unknown SoA payload component")
        self.payload_settled.setdefault(source, set()).update(components)

    def signal_payload(self, source, generation):
        self._check(source, generation)
        if self.payload_settled.get(source) != PAYLOAD_COMPONENTS:
            raise ProtocolError("StrongVA terminal cannot precede remote payload settlement")
        self.payload_signaled.add(source)

    def acquire_payload(self, source, generation):
        self._check(source, generation)
        if source not in self.payload_signaled:
            raise ProtocolError("payload terminal not available")
        self.payload_acquired.add(source)

    def tiles(self, expert):
        if len(self.control_acquired) != self.sources:
            raise ProtocolError("exact counts from every source are required")
        if not 0 <= expert < self.experts:
            raise ProtocolError("unknown expert")
        # Equivalent to the kernel's iterative min-peeling rank selection:
        # take one assignment from each still-active source, then repeat.
        counts = [self.controls[source][expert] for source in range(self.sources)]
        assignments = tuple((source, expert, ordinal)
                            for ordinal in range(max(counts, default=0))
                            for source in range(self.sources) if ordinal < counts[source])
        return tuple(assignments[start:start + self.block_m]
                     for start in range(0, len(assignments), self.block_m))

    def tile_ready(self, expert, tile):
        assignments = self.tiles(expert)[tile]
        return all(source in self.payload_acquired for source, _, _ in assignments)

    def issue_tile(self, expert, tile, scratch_slot):
        if self.retired:
            raise ProtocolError("generation retired")
        if type(scratch_slot) is not int or not 0 <= scratch_slot < self.scratch_slots:
            raise ProtocolError("unknown scratch slot")
        if scratch_slot in self.scratch:
            raise ProtocolError("scratch still belongs to an unfinished/unreleased consumer")
        key = (expert, tile)
        if key in self.issued or not self.tile_ready(expert, tile):
            raise ProtocolError("tile duplicate or a contributing payload is not acquired")
        self.issued.add(key)
        self.scratch[scratch_slot] = key
        return self.tiles(expert)[tile]

    def complete_tile(self, expert, tile):
        key = (expert, tile)
        if key not in self.issued or key in self.completed:
            raise ProtocolError("consumer completion without one outstanding tile")
        self.completed.add(key)

    def release_scratch(self, scratch_slot):
        if scratch_slot not in self.scratch or self.scratch[scratch_slot] not in self.completed:
            raise ProtocolError("consumer has not completed its scratch reads")
        del self.scratch[scratch_slot]

    def retire(self):
        expected = {(expert, tile) for expert in range(self.experts)
                    for tile in range(len(self.tiles(expert)))}
        if (len(self.payload_acquired) != self.sources or
                len(self.local_send_complete) != self.sources or
                self.completed != expected or self.scratch):
            raise ProtocolError("cannot reuse receive state before terminals and consumers drain")
        self.retired = True


def fixed_assignment_counts(mode, tokens, world=16, experts=56, topk=16):
    """Fixed-shape stress fixture, not the matched-performance routing fixture.

    Every active token retains top-k distinct experts. Single-hot-owner puts
    all assignments on 16 experts of rank8; paired-hot-owners keeps every
    assignment cross-host on the assumed contiguous 2x8 rank layout.
    """
    if (world, experts, topk) != (16, 56, 16) or tokens not in (32, 40, 48):
        raise ValueError("stress fixture requires EP16/E896/K16 and T32/40/48")
    counts = [[[0] * experts for _ in range(world)] for _ in range(world)]
    for source in range(world):
        for token in range(tokens):
            for slot in range(topk):
                if mode == "empty" or (mode == "asymmetric" and source != 0):
                    continue
                if mode == "balanced":
                    owner = (source + token + slot) % world
                    expert = (token * topk + slot) % experts
                elif mode in ("single_hot_owner", "asymmetric"):
                    owner, expert = 8, slot
                elif mode == "paired_hot_owners":
                    owner, expert = (8 if source < 8 else 0), slot
                else:
                    raise ValueError("unknown fixed-shape stress fixture")
                counts[owner][source][expert] += 1
    return tuple(tuple(tuple(expert_counts) for expert_counts in owner_counts)
                 for owner_counts in counts)
