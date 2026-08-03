"""Construct tllm_linear_lite modules from torch linear layers."""

from __future__ import annotations

import torch
import torch.nn as nn

from .tllm_config import TllmLinearOptions
from .tllm_loader import load_tllm_linear_classes


def _copy_linear_to_device(module: nn.Linear, device: str) -> nn.Linear:
    if module.weight.device.type == "cuda":
        return module
    if not torch.cuda.is_available():
        raise RuntimeError(
            "tllm_linear_lite quantization requires CUDA, but CUDA is unavailable."
        )
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


def _require_class(class_name: str, module_class, import_error: Exception | None):
    if module_class is not None:
        return module_class
    message = (
        f"The current tllm_linear_lite revision does not provide {class_name}. "
        "Update and rebuild the tracked tllm_linear_lite submodule."
    )
    if import_error is not None:
        raise RuntimeError(message) from import_error
    raise RuntimeError(message)


def build_nvfp4_linear(
    module: nn.Linear,
    options: TllmLinearOptions,
    device: str,
) -> nn.Module:
    """Build NVFP4 while delegating option validation to the dependency."""
    classes = load_tllm_linear_classes()
    quant_module = _copy_linear_to_device(module, device)
    return classes.nvfp4.from_linear(
        quant_module,
        gemm_backend=options.nvfp4_gemm_backend,
        quant_backend=options.nvfp4_quant_backend,
        scale_rule=options.nvfp4_scale_rule,
    )


def build_fp8_blockscale_linear(
    module: nn.Linear,
    options: TllmLinearOptions,
    device: str,
) -> nn.Module:
    """Build a dynamic block-scaled FP8 linear."""
    classes = load_tllm_linear_classes()
    module_class = _require_class(
        "FP8BlockScaleDynamicLinear", classes.fp8_blockscale, None
    )
    quant_module = _copy_linear_to_device(module, device)
    return module_class.from_linear(
        quant_module,
        gemm_backend=options.fp8_gemm_backend,
    )


def build_svdquant_fp8_linear(
    module: nn.Linear,
    options: TllmLinearOptions,
    device: str,
) -> nn.Module:
    """Build an SVDQuant residual plus block-scaled FP8 linear."""
    classes = load_tllm_linear_classes()
    module_class = _require_class(
        "SVDQuantFP8BlockScaleLinear",
        classes.svdquant_fp8,
        classes.fp8_import_error,
    )
    quant_module = _copy_linear_to_device(module, device)
    return module_class.from_linear(
        quant_module,
        rank=options.svdquant_rank,
        alpha=options.svdquant_alpha,
        method=options.svdquant_method,
        gemm_backend=options.fp8_gemm_backend,
        use_ue8m0=options.svdquant_use_ue8m0,
    )


def build_svdquant_nvfp4_linear(
    module: nn.Linear,
    options: TllmLinearOptions,
    device: str,
) -> nn.Module:
    """Build a fused SVDQuant plus NVFP4 linear."""
    classes = load_tllm_linear_classes()
    module_class = _require_class(
        "SVDQuantLinear", classes.svdquant, classes.svdquant_import_error
    )
    quant_module = _copy_linear_to_device(module, device)
    return module_class.from_linear(
        quant_module,
        implementation="nvfp4_fused",
        rank=options.svdquant_rank,
        alpha=options.svdquant_alpha,
        method=options.svdquant_method,
        activation_amax=options.svdquant_activation_amax,
        gscale_x=options.svdquant_gscale_x,
        clone_output=options.svdquant_clone_output,
        max_cached_states=options.svdquant_max_cached_states,
    )
