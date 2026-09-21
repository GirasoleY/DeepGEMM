import unittest


def _align(value, alignment=128):
    return (value + alignment - 1) // alignment * alignment


def _layout(experts_per_rank, topk, hidden, max_active_tokens,
            world_size=8, lsa_size=4, with_sf=True):
    remote_peers = world_size - lsa_size
    max_assignments = max_active_tokens * min(topk, experts_per_rank)
    dispatch_packet_bytes = _align(
        16 + experts_per_rank * 4 + max_assignments * 4)
    combine_record_bytes = _align(hidden * 2)
    combine_packet_bytes = max_assignments * combine_record_bytes
    max_pool_tokens = _align(
        world_size * max_active_tokens * min(topk, experts_per_rank)
        + experts_per_rank * (240 - 1), 1920)

    offset = _align(16 + remote_peers * 8)
    staged_assignment_offset = offset
    offset = _align(
        offset + remote_peers * experts_per_rank
        * max_active_tokens * 4)
    dispatch_packet_offset = offset
    offset = _align(
        offset + 2 * remote_peers * dispatch_packet_bytes)
    remote_activation_offset = offset
    offset = _align(
        offset + remote_peers * max_active_tokens * hidden * (1 if with_sf else 2))
    remote_scale_offset = offset
    offset = _align(
        offset + remote_peers * max_active_tokens * (hidden // 32 if with_sf else 0))
    remote_topk_weight_offset = offset
    offset = _align(
        offset + remote_peers * max_active_tokens * topk * 4)
    combine_prefix_offset = offset
    offset = _align(offset + remote_peers * experts_per_rank * 4)
    combine_packet_offset = offset
    offset = _align(
        offset + 2 * remote_peers * combine_packet_bytes)
    combine_return_ordinal_offset = offset
    offset = _align(offset + max_pool_tokens * 4)
    combine_expert_completion_offset = offset
    workspace_bytes = _align(
        offset + _align(experts_per_rank * 4))
    return {
        'persistent_bytes': _align((1 + 2 * remote_peers) * 8),
        'max_assignments': max_assignments,
        'dispatch_packet_bytes': dispatch_packet_bytes,
        'combine_record_bytes': combine_record_bytes,
        'combine_packet_bytes': combine_packet_bytes,
        'max_pool_tokens': max_pool_tokens,
        'staged_assignment_offset': staged_assignment_offset,
        'dispatch_packet_offset': dispatch_packet_offset,
        'remote_activation_offset': remote_activation_offset,
        'remote_scale_offset': remote_scale_offset,
        'remote_topk_weight_offset': remote_topk_weight_offset,
        'combine_prefix_offset': combine_prefix_offset,
        'combine_packet_offset': combine_packet_offset,
        'combine_return_ordinal_offset': combine_return_ordinal_offset,
        'combine_expert_completion_offset':
            combine_expert_completion_offset,
        'workspace_bytes': workspace_bytes,
    }


class TestMegaMoeGinLayout(unittest.TestCase):
    def test_fixed_packet_and_workspace_arithmetic(self):
        layout = _layout(56, 16, 3584, 48)
        self.assertEqual(layout['dispatch_packet_bytes'], 3328)
        self.assertEqual(layout['combine_record_bytes'], 7168)
        self.assertEqual(layout['combine_packet_bytes'], 5_505_024)
        self.assertEqual(layout['max_pool_tokens'], 21_120)
        self.assertEqual(layout['dispatch_packet_offset'], 43_136)
        self.assertEqual(layout['combine_packet_offset'], 792_576)
        self.assertEqual(
            layout['combine_return_ordinal_offset'], 44_832_768)
        self.assertEqual(
            layout['combine_expert_completion_offset'], 44_917_248)
        self.assertEqual(layout['workspace_bytes'], 44_917_504)

    def test_dynamic_geometries_align_every_storage_boundary(self):
        cases = (
            (33, 6, 4096, 37),
            (9, 4, 1024, 12),
            (50, 6, 2048, 64),
            (4, 16, 2048, 7),
        )
        offset_names = (
            'staged_assignment_offset', 'dispatch_packet_offset',
            'remote_activation_offset', 'remote_scale_offset',
            'remote_topk_weight_offset', 'combine_prefix_offset',
            'combine_packet_offset', 'combine_return_ordinal_offset',
            'combine_expert_completion_offset', 'workspace_bytes',
        )
        for experts_per_rank, topk, hidden, max_active_tokens in cases:
            with self.subTest(
                    experts_per_rank=experts_per_rank, topk=topk,
                    hidden=hidden, max_active_tokens=max_active_tokens):
                layout = _layout(
                    experts_per_rank, topk, hidden, max_active_tokens)
                self.assertEqual(
                    layout['max_assignments'],
                    max_active_tokens * min(topk, experts_per_rank))
                self.assertTrue(all(
                    layout[name] % 128 == 0 for name in offset_names))
                self.assertGreater(
                    layout['workspace_bytes'],
                    layout['combine_expert_completion_offset'])

    def test_bf16_uses_two_byte_activations_without_scale_storage(self):
        fp8 = _layout(17, 6, 2048, 24)
        bf16 = _layout(17, 6, 2048, 24, with_sf=False)
        self.assertEqual(
            bf16['remote_scale_offset'] - bf16['remote_activation_offset'],
            2 * (fp8['remote_scale_offset'] - fp8['remote_activation_offset']))
        self.assertEqual(bf16['remote_scale_offset'],
                         bf16['remote_topk_weight_offset'])
        self.assertEqual(bf16['combine_record_bytes'], fp8['combine_record_bytes'])

    def test_persistent_footer_is_invariant_across_dtype_and_geometry_aliases(self):
        for world, lsa in ((2, 1), (7, 1), (8, 4), (12, 4), (16, 8), (72, 8)):
            with self.subTest(world=world, lsa=lsa):
                a = _layout(17, 6, 2048, 24, world, lsa, False)
                b = _layout(2, 2, 4096, 65, world, lsa, True)
                self.assertEqual(a['persistent_bytes'], b['persistent_bytes'])
                self.assertGreaterEqual(a['staged_assignment_offset'],
                                        16 + (world - lsa) * 8)
                # A -> B -> A aliases may use different scratch extents but
                # address counters relative to the same underlying allocation.
                allocation_bytes = max(a['workspace_bytes'], b['workspace_bytes']) + a['persistent_bytes']
                footer = allocation_bytes - a['persistent_bytes']
                for layout in (a, b, a):
                    self.assertLessEqual(layout['workspace_bytes'], footer)
                    self.assertEqual(allocation_bytes - layout['persistent_bytes'], footer)
                remote = world - lsa
                counters = [footer + index * 8 for index in range(1 + 2 * remote)]
                self.assertEqual(len(set(counters)), 1 + 2 * remote)
                self.assertLessEqual(counters[-1] + 8, allocation_bytes)

    def test_send_receive_packet_offsets_are_symmetric_and_disjoint(self):
        remote_peers = 4
        layouts = (
            (43_136, 3_328),
            (792_576, 5_505_024),
        )
        for base, stride in layouts:
            with self.subTest(base=base, stride=stride):
                send = [base + lane * stride for lane in range(remote_peers)]
                receive = [
                    base + (remote_peers + lane) * stride
                    for lane in range(remote_peers)
                ]
                self.assertEqual(
                    [dst - src for src, dst in zip(send, receive)],
                    [remote_peers * stride] * remote_peers)
                self.assertEqual(len(set(send + receive)), 2 * remote_peers)
                self.assertTrue(all(offset % 128 == 0
                                    for offset in send + receive))
                self.assertEqual(send[-1] + stride, receive[0])


if __name__ == '__main__':
    unittest.main()
