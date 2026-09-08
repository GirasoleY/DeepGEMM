"""CPU-only reconstruction of the September 6 comparator's worst element.

This is an independent scalar quantization diagnostic, not proof of the
installed TRTLLM kernel recipe and not an alternate correctness tolerance.
"""

import math
import struct
import unittest


def _fp32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def _bf16(value):
    bits = struct.unpack("I", struct.pack("f", value))[0]
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16
    return struct.unpack("f", struct.pack("I", rounded))[0]


def _e4m3_saturating_rne(value):
    candidates = [
        (code, (code % 8) * 2 ** -9 if code < 8 else
         (1 + (code % 8) / 8) * 2 ** ((code >> 3) - 7))
        for code in range(127)
    ]
    _, decoded = min(candidates, key=lambda item: (abs(item[1] - abs(value)), item[0] % 2))
    return math.copysign(decoded, value)


def _source(column):
    # rank 0, token 19, epoch 0, ordinary (non-heterogeneous) scales.
    code = (19 * 17 + column * 7) % 30
    return (code % 15 + 1) / 16 * (1 if code % 2 == 0 else -1)


def _worst_element(recipe, activation_bf16=False):
    output = 0.0
    for slot in range(16):
        expert = (19 * 16 + slot) % 56
        intermediate = (215 * 31 + expert * 11 + 5) % 3072
        group_start = intermediate // 32 * 32
        activation = []
        for column in range(group_start, group_start + 32):
            gate = _source((column * 17 + expert * 13 + 1) % 3584)
            gate *= 0.5 if expert % 2 == 0 else 1.0
            up = _source((column * 29 + expert * 7 + 3) % 3584)
            up *= 1.0 if expert % 3 == 0 else 0.5
            value = _fp32(_fp32(gate / _fp32(1 + _fp32(math.exp(-gate)))) * up)
            activation.append(_bf16(value) if activation_bf16 else value)
        amax = max(abs(value) for value in activation)
        if recipe == "deepgemm_ceil448":
            scale = 2 ** math.ceil(math.log2(amax / 448))
        elif recipe == "ocp_exponent_only":
            scale = 2 ** (math.floor(math.log2(amax)) - 8)
        else:
            raise ValueError(recipe)
        decoded = _e4m3_saturating_rne(activation[intermediate - group_start] / scale) * scale
        contribution = _bf16(decoded * (0.5 if expert % 4 < 2 else 1.0))
        output += contribution * 2 ** (-(slot % 3 + 1))
    return _bf16(output)


class Mxfp8RecipeReferenceTest(unittest.TestCase):
    def test_saturation_and_midpoint_round_to_even(self):
        self.assertEqual(_e4m3_saturating_rne(500), 448)
        self.assertEqual(_e4m3_saturating_rne(-500), -448)
        self.assertEqual(_e4m3_saturating_rne(1.0625), 1.0)
        self.assertEqual(_e4m3_saturating_rne(1.1875), 1.25)

    def test_scalar_deepgemm_recipe_reproduces_original_gpu_oracle(self):
        self.assertEqual(_worst_element("deepgemm_ceil448"), 0.66796875)

    def test_bf16_activation_alone_does_not_explain_gpu_result(self):
        self.assertEqual(_worst_element("deepgemm_ceil448", activation_bf16=True), 0.66796875)

    def test_exponent_only_candidate_reproduces_observed_gpu_result(self):
        self.assertEqual(_worst_element("ocp_exponent_only"), 0.63671875)


if __name__ == "__main__":
    unittest.main()
