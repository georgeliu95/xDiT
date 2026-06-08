from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger

try:
    import nvtx
except ImportError:
    class _NoopNvtx:
        @staticmethod
        def annotate(*args, **kwargs):
            del args, kwargs

            def decorator(func):
                return func

            return decorator

    nvtx = _NoopNvtx()


_TLLM_LINEAR_LITE_ROOT = Path(
    os.environ.get(
        "TLLM_LINEAR_LITE_ROOT",
        Path(__file__).resolve().parents[1] / "third_party" / "tllm_linear_lite",
    )
)


def _ensure_tllm_linear_lite_on_path() -> None:
    if _TLLM_LINEAR_LITE_ROOT.is_dir():
        root = str(_TLLM_LINEAR_LITE_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)


@lru_cache(maxsize=1)
def _load_tllm_linear_classes():
    _ensure_tllm_linear_lite_on_path()
    try:
        from tllm_linear_lite.nvfp4_linear import NVFP4DynamicLinear  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Unable to import tllm_linear_lite. Install it, or point "
            "TLLM_LINEAR_LITE_ROOT at a built checkout:\n"
            "  export TLLM_LINEAR_LITE_ROOT=/path/to/tllm_linear_lite\n"
            "  CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST='10.0a' "
            "pip install -e \"$TLLM_LINEAR_LITE_ROOT\" --no-build-isolation"
        ) from exc

    try:
        from tllm_linear_lite.fp8_blockscale_linear import (  # noqa: PLC0415
            FP8BlockScaleDynamicLinear,
        )
    except ImportError:
        FP8BlockScaleDynamicLinear = None
    return NVFP4DynamicLinear, FP8BlockScaleDynamicLinear


def _get_parent_module(model: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    tokens = qualified_name.split(".")
    parent = model
    for token in tokens[:-1]:
        parent = parent[int(token)] if token.isdigit() else getattr(parent, token)
    return parent, tokens[-1]


def _is_wan_major_linear(qualified_name: str) -> bool:
    tokens = qualified_name.split(".")
    return any(token in {"to_q", "to_k", "to_v"} for token in tokens) or any(
        token == "ffn" for token in tokens
    )


def _is_fp8_blockscale_eligible(module: nn.Linear) -> bool:
    return module.in_features % 128 == 0 and module.out_features % 128 == 0


def _needs_small_batch_support(qualified_name: str) -> bool:
    return qualified_name.startswith("condition_embedder.time_embedder.") or (
        qualified_name == "condition_embedder.time_proj"
    )


def _is_nvfp4_eligible(module: nn.Linear) -> bool:
    return module.in_features % 16 == 0


def _ensure_tllm_op(op_name: str, quant_gemm_type: str) -> None:
    if not hasattr(torch.ops.tllm_linear_lite, op_name):
        raise RuntimeError(
            f"tllm_linear_lite op '{op_name}' is not available for "
            f"quant_gemm_type='{quant_gemm_type}'. Rebuild tllm_linear_lite "
            "for the target GPU and CUDA toolkit, then make sure the submodule "
            "root is on PYTHONPATH."
        )


def _copy_linear_to_quant_device(module: nn.Linear) -> nn.Linear:
    if module.weight.device.type == "cuda":
        return module
    if not torch.cuda.is_available():
        raise RuntimeError(
            "tllm_linear_lite quantization requires CUDA, but the source "
            "nn.Linear is on CPU and CUDA is not available."
        )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    quant_module = nn.Linear(
        module.in_features,
        module.out_features,
        bias=module.bias is not None,
        device=device,
        dtype=module.weight.dtype,
    )
    with torch.no_grad():
        quant_module.weight.copy_(module.weight.to(device, non_blocking=True))
        if module.bias is not None:
            quant_module.bias.copy_(module.bias.to(device, non_blocking=True))
    return quant_module


def _build_nvfp4_linear(
    module: nn.Linear,
    gemm_backend: str = "cublaslt",
    scale_rule: str = "static_6",
) -> nn.Module:
    NVFP4DynamicLinear, _ = _load_tllm_linear_classes()
    _ensure_tllm_op("fp4_quantize", "nvfp4")
    quant_module = _copy_linear_to_quant_device(module)
    return NVFP4DynamicLinear.from_linear(
        quant_module,
        gemm_backend=gemm_backend,
        scale_rule=scale_rule,
    )


def _build_fp8_blockscale_linear(module: nn.Linear) -> nn.Module:
    _, FP8BlockScaleDynamicLinear = _load_tllm_linear_classes()
    if FP8BlockScaleDynamicLinear is None:
        raise RuntimeError(
            "The current tllm_linear_lite submodule does not provide "
            "FP8BlockScaleDynamicLinear. Use quant_gemm_type='nvfp4' on B200, "
            "or switch the submodule to an FP8-capable tllm_linear_lite commit."
        )
    _ensure_tllm_op("fp8_blockscale_quantize_128x128", "trtllm-fp8-blockwise")
    quant_module = _copy_linear_to_quant_device(module)
    return FP8BlockScaleDynamicLinear.from_linear(quant_module, gemm_backend="auto")


class VflyLinear:
    """Compatibility shim for older xDiT experiments.

    The previous implementation returned a custom ``VflyLinear`` wrapper.
    New code should use ``replace_linear_layer`` directly; this class keeps
    old scripts working while delegating quantized linears to tllm_linear_lite.
    """

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        linear_type: str = "default",
        nvfp4_gemm_backend: str = "cublaslt",
        nvfp4_scale_rule: str = "static_6",
    ) -> nn.Module:
        if linear_type == "default":
            return linear
        if linear_type == "trtllm-fp8-blockwise":
            if not _is_fp8_blockscale_eligible(linear):
                return linear
            return _build_fp8_blockscale_linear(linear)
        if linear_type in {"trtllm-nvfp4-blockwise", "nvfp4"}:
            if not _is_nvfp4_eligible(linear):
                return linear
            return _build_nvfp4_linear(
                linear,
                gemm_backend=nvfp4_gemm_backend,
                scale_rule=nvfp4_scale_rule,
            )
        raise ValueError(
            f"Unsupported linear_type '{linear_type}' in tllm_linear_lite adapter"
        )


def _select_replacement(
    qualified_name: str,
    module: nn.Linear,
    quant_gemm_type: str,
    nvfp4_gemm_backend: str,
    nvfp4_scale_rule: str,
) -> tuple[nn.Module, str]:
    is_major = _is_wan_major_linear(qualified_name)

    if quant_gemm_type == "nvfp4":
        if is_major and _is_nvfp4_eligible(module):
            return _build_nvfp4_linear(
                module,
                gemm_backend=nvfp4_gemm_backend,
                scale_rule=nvfp4_scale_rule,
            ), "nvfp4"
        return module, "default"

    if quant_gemm_type == "trtllm-fp8-blockwise":
        if (
            not _needs_small_batch_support(qualified_name)
            and _is_fp8_blockscale_eligible(module)
        ):
            return _build_fp8_blockscale_linear(module), "trtllm-fp8-blockwise"
        return module, "default"

    if quant_gemm_type == "nvfp4+trtllm-fp8-blockwise":
        if is_major and _is_nvfp4_eligible(module):
            return _build_nvfp4_linear(
                module,
                gemm_backend=nvfp4_gemm_backend,
                scale_rule=nvfp4_scale_rule,
            ), "nvfp4"
        if (
            not _needs_small_batch_support(qualified_name)
            and _is_fp8_blockscale_eligible(module)
        ):
            return _build_fp8_blockscale_linear(module), "trtllm-fp8-blockwise"
        return module, "default"

    raise ValueError(
        f"Unsupported quant_gemm_type '{quant_gemm_type}' in tllm_linear_lite adapter"
    )


@nvtx.annotate(message="replace_linear_layer", color="red")
def replace_linear_layer(
    model: nn.Module,
    quant_gemm_type: str = "nvfp4",
    enable_hadamard: bool = False,
    quant_recipe: object | None = None,
    nvfp4_gemm_backend: str = "cublaslt",
    nvfp4_scale_rule: str = "static_6",
) -> nn.Module:
    del enable_hadamard, quant_recipe

    if quant_gemm_type in (None, "bf16"):
        return model
    if quant_gemm_type == "svdquant.nvfp4":
        raise ValueError(
            "svdquant.nvfp4 is not implemented by tllm_linear_lite. "
            "Use 'nvfp4' or 'nvfp4+trtllm-fp8-blockwise'."
        )

    replacements: list[tuple[str, nn.Module, str]] = []
    counts = {
        "nvfp4": 0,
        "trtllm-fp8-blockwise": 0,
        "default": 0,
    }

    linear_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    for name, module in linear_modules:
        wrapped_module, replacement_type = _select_replacement(
            name,
            module,
            quant_gemm_type,
            nvfp4_gemm_backend,
            nvfp4_scale_rule,
        )
        counts[replacement_type] += 1
        if wrapped_module is not module:
            replacements.append((name, wrapped_module, replacement_type))

    for name, wrapped_module, _ in replacements:
        parent, child_name = _get_parent_module(model, name)
        setattr(parent, child_name, wrapped_module)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank == 0:
        logger.info(
            "Replaced {} linear layers with tllm_linear_lite NVFP4",
            counts["nvfp4"],
        )
        if counts["nvfp4"]:
            logger.info(
                "tllm_linear_lite NVFP4 config: gemm_backend={}, scale_rule={}",
                nvfp4_gemm_backend,
                nvfp4_scale_rule,
            )
        logger.info(
            "Replaced {} linear layers with tllm_linear_lite FP8 blockscale",
            counts["trtllm-fp8-blockwise"],
        )
        logger.info("Remained {} default linear layers", counts["default"])

    return model
