"""CPU compatibility contracts; fixture metadata is not installed-GPU proof."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gb200_mxfp4_compat as compat
import bench_gb200_transport_matched as runner


SILU, BACKEND = object(), object()


def legacy(mxfp4_backend, layer, w13_weight, w2_weight, w13_weight_scale,
           w2_weight_scale, w13_bias=None, w2_bias=None, _cache_permute_indices=None):
    return (mxfp4_backend, layer, w13_weight, w2_weight, w13_weight_scale,
            w2_weight_scale, w13_bias, w2_bias, _cache_permute_indices)


def modern(mxfp4_backend, *args, activation=None, **kwargs):
    return mxfp4_backend, args, activation, kwargs


def known_identity():
    return {"module": compat.MODULE, "function": compat.FUNCTION,
            "module_path": "/cpu-fixture/not-installed/mxfp4.py",
            "vllm_distribution_version": compat.LEGACY_VERSION,
            "module_sha256": compat.LEGACY_MODULE_SHA256,
            "function_source_sha256": compat.LEGACY_FUNCTION_SHA256,
            "signature": "CPU fixture; real signature inspected independently"}


class ConverterContracts(unittest.TestCase):
    def adapt(self, converter=legacy, identity=None):
        return compat._adapt_converter(converter, silu=SILU, backend=BACKEND,
                                       identity=identity or known_identity())

    def test_known_legacy_silu_preserves_all_tensor_argument_identities(self):
        adapted, evidence = self.adapt()
        values = [object() for _ in range(8)]
        result = adapted(BACKEND, *values[:5], w13_bias=values[5], w2_bias=values[6],
                         _cache_permute_indices=values[7], activation=SILU)
        self.assertEqual(result, (BACKEND, *values))
        self.assertTrue(evidence["activation_keyword_omitted"])
        self.assertEqual(evidence["original_converter"], known_identity())

    def test_new_explicit_activation_api_delegates_without_omission(self):
        adapted, evidence = self.adapt(modern, {"new_api_fixture": True})
        tensor, cache = object(), {}
        result = adapted(BACKEND, tensor, activation=SILU, cache=cache)
        self.assertEqual(result, (BACKEND, (tensor,), SILU, {"cache": cache}))
        self.assertIs(result[3]["cache"], cache)
        self.assertFalse(evidence["activation_keyword_omitted"])
        self.assertEqual(evidence["selection"], "explicit_activation_api_unchanged")

    def test_unknown_legacy_source_version_or_identity_rejected(self):
        for field in ("module", "function", "vllm_distribution_version", "module_sha256",
                      "function_source_sha256"):
            identity = known_identity()
            identity[field] = "unknown"
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "unsupported legacy"):
                self.adapt(identity=identity)

    def test_unknown_signature_and_kwargs_only_activation_are_rejected(self):
        def unknown(*args, **kwargs):
            raise AssertionError("must not be called")
        def changed(mxfp4_backend, layer, w13_weight, w2_weight, w13_weight_scale,
                    w2_weight_scale, w13_bias=None, w2_bias=None, _cache_permute_indices=1):
            raise AssertionError("must not be called")
        for converter in (unknown, changed):
            with self.assertRaisesRegex(RuntimeError, "unsupported legacy"):
                self.adapt(converter)

    def test_non_silu_missing_activation_and_wrong_backend_fail_before_delegate(self):
        for converter in (legacy, modern):
            adapted, _ = self.adapt(converter)
            for activation in (None, "silu", object()):
                with self.assertRaisesRegex(ValueError, "only explicit"):
                    adapted(BACKEND, *[None] * 5, activation=activation)
            with self.assertRaisesRegex(ValueError, "only explicit"):
                adapted(BACKEND, *[None] * 5)
            with self.assertRaisesRegex(ValueError, "only FLASHINFER"):
                adapted(object(), *[None] * 5, activation=SILU)

    def test_identity_hashes_actual_function_and_module_source(self):
        module = SimpleNamespace(__file__=__file__, __name__=compat.MODULE)
        with patch.object(legacy, "__module__", compat.MODULE), patch.object(
                compat.importlib.metadata, "version", return_value="CPU fixture"):
            identity = compat._identity(module, legacy)
        self.assertEqual(identity["module_path"], str(Path(__file__).resolve()))
        self.assertEqual(len(identity["module_sha256"]), 64)
        self.assertEqual(len(identity["function_source_sha256"]), 64)
        self.assertEqual(identity["vllm_distribution_version"], "CPU fixture")
        self.assertNotEqual(identity["module_sha256"], compat.LEGACY_MODULE_SHA256)

    def test_prepare_restores_original_converter_even_on_error(self):
        module = SimpleNamespace(**{compat.FUNCTION: legacy},
                                 Mxfp4MoeBackend=SimpleNamespace(FLASHINFER_TRTLLM_MXFP4_MXFP8=BACKEND))
        activation_module = SimpleNamespace(MoEActivation=SimpleNamespace(SILU=SILU))
        with patch.object(compat.importlib, "import_module", side_effect=[module, activation_module]), patch.object(
                compat, "_identity", return_value=known_identity()):
            adapter = compat.prepare_converter_adapter()
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            with adapter.install():
                self.assertIsNot(getattr(module, compat.FUNCTION), legacy)
                raise RuntimeError("fixture failure")
        self.assertIs(getattr(module, compat.FUNCTION), legacy)


class CollectiveIntegrationContracts(unittest.TestCase):
    def harness(self, altered_peer=False):
        class Dist:
            def get_world_size(self):
                return 16
            def all_gather_object(self, output, value):
                output[:] = [value] * 16
                if altered_peer and isinstance(value, dict):
                    output[11] = {**value, "selection": "different"}
        return SimpleNamespace(dist=Dist(), torch=SimpleNamespace(cuda=Mock()))

    def test_prepare_error_or_peer_identity_mismatch_prevents_constructor(self):
        for peer in (False, True):
            adapter = SimpleNamespace(evidence={"selection": "cpu_fixture"}, install=nullcontext)
            with patch.object(runner, "prepare_converter_adapter", return_value=adapter,
                              side_effect=None if peer else RuntimeError("unknown legacy")), patch.object(
                    runner.matched, "DeepEPTRTLLM") as constructor, self.assertRaises(RuntimeError):
                runner._compare_gin(self.harness(peer), {}, None)
            constructor.assert_not_called()

    def test_converter_patch_is_restored_before_timing_and_metadata_is_retained(self):
        module = SimpleNamespace(converter=legacy)
        adapter = SimpleNamespace(evidence={"selection": "cpu_fixture"},
                                  install=lambda: patch.object(module, "converter", modern))
        comparator = SimpleNamespace(buffer=SimpleNamespace(destroy=Mock()), last_receipt={},
                                     api_evidence={}, weight_sha256="fixture", recipe_profile=None)
        def construct(*args):
            self.assertIs(module.converter, modern)
            return comparator
        def pair(*args):
            self.assertIs(module.converter, legacy)
            return {}, {}
        with patch.object(runner, "prepare_converter_adapter", return_value=adapter), patch.object(
                runner.matched, "DeepEPTRTLLM", side_effect=construct), patch.object(
                runner, "_validate_deepep_domains", return_value={}), patch.object(
                runner.matched, "_benchmark_pair", side_effect=pair):
            result = runner._compare_gin(self.harness(), {}, None)
        evidence = result["gb200_mxfp4_converter_compatibility"]
        self.assertEqual(evidence["per_rank"], [adapter.evidence] * 16)
        self.assertTrue(evidence["installed_only_during_construction"])
        comparator.buffer.destroy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
