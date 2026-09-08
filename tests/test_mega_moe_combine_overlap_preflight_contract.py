"""Execute the actual public preflight method without importing CUDA/PyTorch.

The existing test_mega_moe_gin_lifecycle.py also tests enable_gin's pre-UID
ordering using PyTorch objects. This CPU-only extraction does not claim to
exercise allocation, NCCL setup, graph lifecycle, or device code.
"""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / "deep_gemm/mega/__init__.py"


def actual_preflight():
    tree = ast.parse(SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SymmBuffer")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_collective_validate_gin_config")
    def gather(output, local, group):
        output[:] = [local.copy() for _ in range(group.size())]
    dist = SimpleNamespace(all_gather_object=Mock(side_effect=gather))
    namespace = {"os": os, "dist": dist}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[method.name], dist


def valid_self():
    return SimpleNamespace(
        _gin_context=None, group=SimpleNamespace(size=lambda: 16),
        buffer=SimpleNamespace(numel=lambda: 4096, element_size=lambda: 1),
        num_experts=896, num_max_tokens_per_rank=384, num_topk=16, hidden=3584,
        intermediate_hidden=3072, num_shared_experts=0, _gin_layout_enabled=True,
        gin_queue_depth=64, gin_completion_batch=8, gin_combine_chunk_bytes=7168,
        gin_outbox_depth=64, gin_combine_issue_wave=8, gin_active_fast_path=True,
        gin_bulk_combine=True, gin_direct_dispatch=True)


class ActualCombinePreflight(unittest.TestCase):
    def setUp(self):
        self.function, self.dist = actual_preflight()
        self.environment = {
            "DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT": "1",
            "DG_MEGAMOE_GIN_DISPATCH_OVERLAP": "1",
            "DG_MEGAMOE_GIN_PRECONSENSUS_PACK": "1",
            "DG_MEGAMOE_GIN_COOP_DIRECT_PACK": "1",
            "DG_MEGAMOE_GIN_COMBINE_EXPERTS_PER_WAVE": "0",
            "DG_MEGAMOE_GIN_COMBINE_BARRIER_WARPS": "1"}

    def run_preflight(self, mode=None, value=None):
        environment = dict(self.environment)
        if mode is not None:
            environment["DG_MEGAMOE_GIN_COMBINE_OVERLAP"] = mode
        with patch.dict(os.environ, environment, clear=True):
            return self.function(value or valid_self(), {"enabled": True}, 9, 64, 4, 8, "gdaki")

    def test_absent_flag_is_off_and_zero_one_zero_uses_one_existing_collective(self):
        for raw in (None, "0", "1", "0"):
            self.dist.all_gather_object.reset_mock()
            self.run_preflight(raw)
            self.dist.all_gather_object.assert_called_once()
            self.assertEqual(self.dist.all_gather_object.call_args.args[1]["combine_overlap"], raw or "0")

    def test_raw_invalid_values_are_collected_before_rejection(self):
        for raw in ("", "01", "true", "2", "-1", " 1", "1 ", "1suffix"):
            self.dist.all_gather_object.reset_mock()
            with self.assertRaisesRegex(RuntimeError, "COMBINE_OVERLAP must be exactly 0 or 1"):
                self.run_preflight(raw)
            self.dist.all_gather_object.assert_called_once()
            self.assertEqual(self.dist.all_gather_object.call_args.args[1]["combine_overlap"], raw)

    def test_rank_skew_both_directions_rejected(self):
        for raw in ("0", "1"):
            def skew(output, local, group):
                output[:] = [local.copy() for _ in range(group.size())]
                output[8]["combine_overlap"] = "0" if raw == "1" else "1"
            self.dist.all_gather_object.side_effect = skew
            with self.assertRaisesRegex(RuntimeError, "configuration mismatch across ranks"):
                self.run_preflight(raw)

    def test_all_four_dependencies_are_required(self):
        for missing in ("gin_bulk_combine", "gin_direct_dispatch",
                        "SINGLE_COMBINE_CONTEXT", "DISPATCH_OVERLAP"):
            value = valid_self()
            if missing.startswith("gin_"):
                setattr(value, missing, False)
            else:
                self.environment["DG_MEGAMOE_GIN_" + missing] = "0"
            with self.assertRaisesRegex(RuntimeError, "combine_overlap requires"):
                self.run_preflight("1", value)
            if not missing.startswith("gin_"):
                self.environment["DG_MEGAMOE_GIN_" + missing] = "1"


if __name__ == "__main__":
    unittest.main()
