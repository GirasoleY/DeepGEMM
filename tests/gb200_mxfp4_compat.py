"""GB200-only converter compatibility; no import-time Torch/vLLM dependency.

The captured K3 vLLM converter predates the explicit activation keyword, but its
TRTLLM branch already performs the fixed SwiGLU gate/up permutation. Only that
source/version/signature may omit our SILU keyword. The original converter and
all tensor arguments are otherwise unchanged. This is initialization, not a
timed graph operation or a general vLLM compatibility layer.
"""

import functools
import hashlib
import importlib
import importlib.metadata
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


MODULE = "vllm.model_executor.layers.fused_moe.oracle.mxfp4"
FUNCTION = "convert_weight_to_mxfp4_moe_kernel_format"
LEGACY_VERSION = "0.1.dev19262+gb6bbf29dd.d20260727"
LEGACY_MODULE_SHA256 = "4b85e450bf8e07e29dd035ff3301fc4651e172d3b2f8228b35f41982dd9c18cf"
LEGACY_FUNCTION_SHA256 = "aa4ec17a54ac5d80ffccf6502bc9de3aadc29a4d3d4e41e918e3bfa44bfb74e2"
LEGACY_PARAMETERS = (
    "mxfp4_backend", "layer", "w13_weight", "w2_weight", "w13_weight_scale",
    "w2_weight_scale", "w13_bias", "w2_bias", "_cache_permute_indices",
)


def _identity(module, converter):
    module_file = Path(module.__file__).resolve()
    converter_file = Path(inspect.getsourcefile(converter)).resolve()
    if converter_file != module_file or converter.__module__ != MODULE:
        raise RuntimeError("converter source does not belong to the loaded mxfp4 module")
    return {
        "module": module.__name__, "function": converter.__qualname__,
        "module_path": str(module_file),
        "module_sha256": hashlib.sha256(module_file.read_bytes()).hexdigest(),
        "function_source_sha256": hashlib.sha256(inspect.getsource(converter).encode()).hexdigest(),
        "signature": str(inspect.signature(converter)),
        "vllm_distribution_version": importlib.metadata.version("vllm"),
    }


def _adapt_converter(converter, *, silu, backend, identity):
    """Pure selection/call policy; identity must come from _identity at runtime."""
    signature = inspect.signature(converter)
    activation = signature.parameters.get("activation")
    modern = activation is not None and activation.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    if not modern:
        exact_signature = tuple(signature.parameters) == LEGACY_PARAMETERS and all(
            parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and
            (parameter.default is inspect.Parameter.empty if index < 6 else parameter.default is None)
            for index, parameter in enumerate(signature.parameters.values()))
        known_source = (
            identity["module"] == MODULE and identity["function"] == FUNCTION and
            identity["vllm_distribution_version"] == LEGACY_VERSION and
            identity["module_sha256"] == LEGACY_MODULE_SHA256 and
            identity["function_source_sha256"] == LEGACY_FUNCTION_SHA256)
        if not exact_signature or not known_source:
            raise RuntimeError("unsupported legacy MXFP4 converter source/version/signature; refusing activation omission")

    @functools.wraps(converter)
    def adapted(*args, **kwargs):
        # This benchmark requests exactly this enum member and backend, not an
        # arbitrary activation/backend that the old converter might also accept.
        if kwargs.get("activation") is not silu:
            raise ValueError("GB200 converter adapter accepts only explicit MoEActivation.SILU")
        requested_backend = args[0] if args else kwargs.get("mxfp4_backend")
        if requested_backend is not backend:
            raise ValueError("GB200 converter adapter accepts only FLASHINFER_TRTLLM_MXFP4_MXFP8")
        forwarded = kwargs if modern else {key: value for key, value in kwargs.items() if key != "activation"}
        signature.bind(*args, **forwarded)
        return converter(*args, **forwarded)

    evidence = {
        "selection": "explicit_activation_api_unchanged" if modern else "pinned_legacy_fixed_swiglu_silu_keyword_omitted",
        "original_converter": dict(identity),
        "activation_request": "MoEActivation.SILU",
        "backend": "FLASHINFER_TRTLLM_MXFP4_MXFP8",
        "activation_keyword_omitted": not modern,
        "tensor_arguments_and_converter_body_unchanged": True,
        "scope": "GB200 comparator construction only; no timing/math/layout change",
        "legacy_recipe_evidence": None if modern else {
            "module_sha256": LEGACY_MODULE_SHA256,
            "function_lines": [1249, 1571], "swiglu_gate_up_shuffle_lines": [1341, 1359],
            "claim": "fixed TRTLLM SwiGLU gate/up layout conversion; not new GPU accuracy evidence",
        },
    }
    return adapted, evidence


def prepare_converter_adapter():
    """Inspect installed source before constructing any DeepEP comparator."""
    module = importlib.import_module(MODULE)
    activation_module = importlib.import_module("vllm.model_executor.layers.fused_moe.activation")
    original = getattr(module, FUNCTION)
    adapted, evidence = _adapt_converter(
        original, silu=activation_module.MoEActivation.SILU,
        backend=module.Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
        identity=_identity(module, original))
    return SimpleNamespace(evidence=evidence, install=lambda: patch.object(module, FUNCTION, adapted))
