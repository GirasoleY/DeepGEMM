import random
import unittest
from functools import reduce
from operator import or_


NUM_RANKS = 16
LSA_SIZE = 8
WARP_SIZE = 32


def _pair_decisions(local_flags):
    """Model the r3 cross-LSA mailbox exchange and local pair publication."""
    assert len(local_flags) == NUM_RANKS
    assert all(0 <= flags < 4 for flags in local_flags)
    return [
        local_flags[rank] | local_flags[(rank + LSA_SIZE) % NUM_RANKS]
        for rank in range(NUM_RANKS)
    ]


def _parallel_lsa_or(pair_flags, rank):
    """Model eight active load lanes followed by a full-warp OR reduction."""
    lsa_base = (rank // LSA_SIZE) * LSA_SIZE
    lanes = [
        pair_flags[lsa_base + lane] if lane < LSA_SIZE else 0
        for lane in range(WARP_SIZE)
    ]
    return reduce(or_, lanes, 0)


class TestMegaMoeActivityGateModel(unittest.TestCase):
    def _check_world_consensus(self, local_flags):
        pair_flags = _pair_decisions(local_flags)
        expected = reduce(or_, local_flags, 0)
        decisions = {
            _parallel_lsa_or(pair_flags, rank) for rank in range(NUM_RANKS)
        }
        self.assertEqual(decisions, {expected})

    def test_each_single_rank_flag_reaches_both_lsas(self):
        for source in range(NUM_RANKS):
            for flags in (1, 2, 3):
                with self.subTest(source=source, flags=flags):
                    local_flags = [0] * NUM_RANKS
                    local_flags[source] = flags
                    self._check_world_consensus(local_flags)

    def test_activity_and_ineligibility_bits_remain_independent(self):
        local_flags = [0] * NUM_RANKS
        local_flags[2] = 1
        local_flags[13] = 2
        pair_flags = _pair_decisions(local_flags)
        for rank in range(NUM_RANKS):
            world_flags = _parallel_lsa_or(pair_flags, rank)
            self.assertEqual(world_flags & 1, 1)
            self.assertEqual((world_flags >> 1) & 1, 1)

    def test_seeded_arbitrary_two_bit_decisions(self):
        rng = random.Random(20260904)
        for scenario in range(512):
            local_flags = [rng.randrange(4) for _ in range(NUM_RANKS)]
            with self.subTest(scenario=scenario):
                self._check_world_consensus(local_flags)

    def test_idle_world_stays_idle(self):
        self._check_world_consensus([0] * NUM_RANKS)


if __name__ == "__main__":
    unittest.main()
