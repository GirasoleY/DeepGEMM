#include <cassert>
#include <deep_gemm/layout/mega_moe_gin_topology.h>

int main() {
    // Check the production mapping against an enumerated peer list, including
    // every valid uniform partition in the native 1..72-rank domain.
    for (unsigned world = 1; world <= 72; ++world) {
        for (unsigned lsa = 1; lsa <= world; ++lsa) {
            if (world % lsa != 0)
                continue;
            const deep_gemm::layout::MegaMoeGinTopology topology{world, lsa};
            for (unsigned rank = 0; rank < world; ++rank) {
                unsigned slot = 0;
                bool seen[72] = {};
                for (unsigned peer = 0; peer < world; ++peer) {
                    if (topology.is_same_lsa(rank, peer))
                        continue;
                    assert(topology.remote_rank(rank, slot) == peer);
                    assert(topology.remote_slot(rank, peer) == slot);
                    const auto receive_slot = topology.return_slot(rank, slot);
                    assert(receive_slot < world - lsa);
                    assert(topology.remote_rank(peer, receive_slot) == rank);
                    assert(topology.return_slot(peer, receive_slot) == slot);
                    assert(!seen[peer]);
                    seen[peer] = true;
                    ++slot;
                }
                assert(slot == world - lsa);
            }
        }
    }
}
