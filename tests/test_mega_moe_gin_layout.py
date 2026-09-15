import unittest


def _align(value, alignment=128):
    return (value + alignment - 1) // alignment * alignment


class TestMegaMoeEp8GinLayout(unittest.TestCase):
    def test_fixed_packet_and_workspace_arithmetic(self):
        remote_peers = 4
        experts_per_rank = 56
        max_tokens = 48
        topk = 16
        max_assignments = max_tokens * topk

        dispatch_packet_bytes = _align(
            16 + experts_per_rank * 4 + max_assignments * 4)
        combine_record_bytes = _align(3584 * 2)
        combine_packet_bytes = max_assignments * combine_record_bytes
        max_pool_tokens = _align(
            8 * max_tokens * topk + experts_per_rank * (240 - 1), 1920)

        offset = 128  # persistent dispatch epoch and cache-line padding
        offset += remote_peers * experts_per_rank * max_tokens * 4
        dispatch_packet_offset = offset
        offset += 2 * remote_peers * dispatch_packet_bytes
        offset += remote_peers * max_tokens * 3584
        offset += remote_peers * max_tokens * (3584 // 32)
        offset += remote_peers * max_tokens * topk * 4
        offset += remote_peers * experts_per_rank * 4
        combine_packet_offset = offset
        offset += 2 * remote_peers * combine_packet_bytes
        combine_return_ordinal_offset = offset
        offset += max_pool_tokens * 4
        combine_expert_completion_offset = offset
        offset += _align(experts_per_rank * 4)
        workspace_bytes = _align(offset)

        self.assertEqual(dispatch_packet_bytes, 3328)
        self.assertEqual(combine_record_bytes, 7168)
        self.assertEqual(combine_packet_bytes, 5_505_024)
        self.assertEqual(max_pool_tokens, 21_120)
        self.assertEqual(dispatch_packet_offset, 43_136)
        self.assertEqual(combine_packet_offset, 792_576)
        self.assertEqual(combine_return_ordinal_offset, 44_832_768)
        self.assertEqual(combine_expert_completion_offset, 44_917_248)
        self.assertEqual(workspace_bytes, 44_917_504)

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
