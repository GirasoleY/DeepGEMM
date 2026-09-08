#!/usr/bin/env python3
"""Host-only contracts for JIT experiment feature evidence."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import test_mega_moe_accuracy as accuracy


class _FakeDist:
    def __init__(self, records=None):
        self.records = records

    def all_gather_object(self, output, local_record):
        records = self.records
        if records is None:
            records = [
                {**local_record, "rank": rank} for rank in range(len(output))
            ]
        if len(records) != len(output):
            raise AssertionError("fake collective record count mismatch")
        output[:] = records


def _args(*, direct_dispatch=False):
    return SimpleNamespace(gin_direct_dispatch=direct_dispatch)


def _records(*, warp="0", coop="0", prepack="0", single="0", direct=False,
             world_size=16):
    return [
        {
            "rank": rank,
            "direct_dispatch": direct,
            "expert_width": "0",
            "barrier_warps": "1",
            "flags": {
                accuracy.GIN_DISPATCH_WARP_SCAN_ENV: warp,
                accuracy.GIN_COOP_DIRECT_PACK_ENV: coop,
                accuracy.GIN_PRECONSENSUS_PACK_ENV: prepack,
                accuracy.GIN_SINGLE_COMBINE_CONTEXT_ENV: single,
            },
        }
        for rank in range(world_size)
    ]


class TestMegaMoeExperimentEvidence(unittest.TestCase):
    def test_single_mode_is_opt_in_and_not_implicitly_enabled_by_legacy_flags(self):
        self.assertNotIn(accuracy.GIN_SINGLE_COMBINE_CONTEXT_ENV,
                         accuracy.GIN_EXPERIMENT_FLAG_ENVS)
        evidence = accuracy._collect_gin_experiment_flags(
            _args(direct_dispatch=True), 0, 16,
            _FakeDist(_records(single="1", direct=True)))
        self.assertTrue(evidence[accuracy.GIN_SINGLE_COMBINE_CONTEXT_ENV])

    def test_single_mode_must_be_canonical_and_uniform(self):
        name = accuracy.GIN_SINGLE_COMBINE_CONTEXT_ENV
        for invalid in ("2", "01", "true", ""):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(RuntimeError, "must be 0 or 1"):
                accuracy._collect_gin_experiment_flags(
                    _args(direct_dispatch=True), 0, 16,
                    _FakeDist(_records(single=invalid, direct=True)))
        records = _records(single="1", direct=True)
        records[7]["flags"][name] = "0"
        with self.assertRaisesRegex(RuntimeError, "not uniform across ranks"):
            accuracy._collect_gin_experiment_flags(
                _args(direct_dispatch=True), 0, 16, _FakeDist(records))

    def test_retired_experiments_fail_collectively_in_both_modes(self):
        for mode in ("0", "1"):
            for key, value in (("expert_width", "8"), ("barrier_warps", "8"),
                               ("expert_width", "00"), ("barrier_warps", "01")):
                records = _records(single=mode, direct=True)
                records[7][key] = value
                with self.subTest(mode=mode, key=key), self.assertRaisesRegex(
                        RuntimeError, r"retired experiments.*incompatible ranks: \[7\]"):
                    accuracy._collect_gin_experiment_flags(
                        _args(direct_dispatch=True), 0, 16, _FakeDist(records))

    def test_defaults_are_collectively_disabled_and_emitted_by_exact_name(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            evidence = accuracy._collect_gin_experiment_flags(
                _args(), 0, 16, _FakeDist()
            )
        self.assertEqual(
            evidence,
            {
                "DG_MEGAMOE_GIN_DISPATCH_WARP_SCAN": False,
                "DG_MEGAMOE_GIN_COOP_DIRECT_PACK": False,
                "DG_MEGAMOE_GIN_PRECONSENSUS_PACK": False,
                "DG_MEGAMOE_GIN_SINGLE_COMBINE_CONTEXT": False,
            },
        )

    def test_enabled_flags_are_collectively_emitted(self):
        env = {
            accuracy.GIN_DISPATCH_WARP_SCAN_ENV: "1",
            accuracy.GIN_COOP_DIRECT_PACK_ENV: "1",
            accuracy.GIN_PRECONSENSUS_PACK_ENV: "1",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            evidence = accuracy._collect_gin_experiment_flags(
                _args(direct_dispatch=True), 0, 16, _FakeDist()
            )
        self.assertTrue(evidence[accuracy.GIN_DISPATCH_WARP_SCAN_ENV])
        self.assertTrue(evidence[accuracy.GIN_COOP_DIRECT_PACK_ENV])
        self.assertTrue(evidence[accuracy.GIN_PRECONSENSUS_PACK_ENV])

    def test_invalid_value_is_rejected_collectively(self):
        records = _records(direct=True)
        records[7]["flags"][accuracy.GIN_COOP_DIRECT_PACK_ENV] = "yes"
        with self.assertRaisesRegex(
            RuntimeError,
            "DG_MEGAMOE_GIN_COOP_DIRECT_PACK must be 0 or 1.*rank 7='yes'",
        ):
            accuracy._collect_gin_experiment_flags(
                _args(direct_dispatch=True), 0, 16, _FakeDist(records)
            )

    def test_nonuniform_flag_is_rejected_collectively(self):
        records = _records(warp="1", direct=True)
        records[9]["flags"][accuracy.GIN_DISPATCH_WARP_SCAN_ENV] = "0"
        with self.assertRaisesRegex(
            RuntimeError,
            "DG_MEGAMOE_GIN_DISPATCH_WARP_SCAN is not uniform across ranks",
        ):
            accuracy._collect_gin_experiment_flags(
                _args(direct_dispatch=True), 0, 16, _FakeDist(records)
            )

    def test_each_enabled_experiment_requires_direct_dispatch(self):
        for name, values in (
            (accuracy.GIN_DISPATCH_WARP_SCAN_ENV, {"warp": "1"}),
            (accuracy.GIN_COOP_DIRECT_PACK_ENV, {"coop": "1"}),
            (accuracy.GIN_PRECONSENSUS_PACK_ENV, {"prepack": "1"}),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(
                RuntimeError, f"{name}=1 requires --gin-direct-dispatch"
            ):
                accuracy._collect_gin_experiment_flags(
                    _args(), 0, 16, _FakeDist(_records(**values))
                )

    def test_preconsensus_pack_requires_cooperative_pack_collectively(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "DG_MEGAMOE_GIN_PRECONSENSUS_PACK=1 requires "
            "DG_MEGAMOE_GIN_COOP_DIRECT_PACK=1",
        ):
            accuracy._collect_gin_experiment_flags(
                _args(direct_dispatch=True), 0, 16,
                _FakeDist(_records(prepack="1", direct=True)),
            )

    def test_direct_dispatch_must_also_be_uniform(self):
        records = _records(direct=True)
        records[15]["direct_dispatch"] = False
        with self.assertRaisesRegex(
            RuntimeError, "--gin-direct-dispatch is not uniform across ranks"
        ):
            accuracy._collect_gin_experiment_flags(
                _args(direct_dispatch=True), 0, 16, _FakeDist(records)
            )

    def test_accuracy_and_perf_records_retain_exact_feature_mapping(self):
        source = Path(accuracy.__file__).read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count('"jit_experiment_flags"'), 3)
        self.assertIn(
            '"jit_experiment_flags": transport_evidence[\n'
            '                            "jit_experiment_flags"',
            source,
        )


if __name__ == "__main__":
    unittest.main()
