"""In-place model conversion to tllm_linear_lite implementations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch.nn as nn

from xfuser.core.utils.runner_utils import log

from .linear_backend import LinearBackendType, TLLM_LINEAR_BACKENDS
from .tllm_builders import (
    build_fp8_blockscale_linear,
    build_nvfp4_linear,
    build_svdquant_fp8_linear,
    build_svdquant_nvfp4_linear,
)
from .tllm_config import TllmLinearOptions
from .tllm_policy import (
    estimate_svdquant_nvfp4_state_bytes,
    get_parent_module,
    is_fp8_blockscale_eligible,
    is_nvfp4_eligible,
    is_svdquant_fp8_eligible,
    is_svdquant_nvfp4_eligible,
    is_wan_major_linear,
    matches_fp8_override,
    needs_small_batch_support,
)
from .tllm_wrappers import wrap_svdquant_clear_cache, wrap_svdquant_debug


@dataclass
class TllmConversionState:
    """Mutable budget accounting shared across FSDP blocks and subtrees."""

    nvfp4_state_bytes_used: int = 0


def _build_fp8(
    module: nn.Linear,
    options: TllmLinearOptions,
    device: str,
) -> tuple[nn.Module, str]:
    if not is_fp8_blockscale_eligible(module):
        return module, "bf16"
    return build_fp8_blockscale_linear(module, options, device), "fp8-blockwise"


def _build_svdquant_fp8(
    module: nn.Linear,
    options: TllmLinearOptions,
    device: str,
) -> tuple[nn.Module, str]:
    if not is_svdquant_fp8_eligible(module):
        return module, "bf16"
    return build_svdquant_fp8_linear(module, options, device), "svdquant-fp8"


def _select_replacement(
    qualified_name: str,
    module: nn.Linear,
    backend: LinearBackendType,
    options: TllmLinearOptions,
    device: str,
    force_fp8: bool,
) -> tuple[nn.Module, str]:
    if needs_small_batch_support(qualified_name):
        return module, "bf16"

    if backend == LinearBackendType.TLLM_FP8_BLOCKWISE:
        return _build_fp8(module, options, device)
    if backend == LinearBackendType.TLLM_SVDQUANT_FP8_BLOCKWISE:
        return _build_svdquant_fp8(module, options, device)

    if backend == LinearBackendType.TLLM_SVDQUANT_NVFP4_FUSED:
        if force_fp8:
            return _build_svdquant_fp8(module, options, device)
        if is_svdquant_nvfp4_eligible(module):
            return (
                build_svdquant_nvfp4_linear(module, options, device),
                "svdquant-nvfp4",
            )
        return module, "bf16"

    if force_fp8:
        return _build_fp8(module, options, device)
    if is_wan_major_linear(qualified_name) and is_nvfp4_eligible(module):
        return build_nvfp4_linear(module, options, device), "nvfp4"
    if backend == LinearBackendType.TLLM_NVFP4_FP8_BLOCKWISE:
        return _build_fp8(module, options, device)
    return module, "bf16"


def _validate_options(
    backend: LinearBackendType, options: TllmLinearOptions
) -> None:
    if (
        backend == LinearBackendType.TLLM_SVDQUANT_NVFP4_FUSED
        and options.svdquant_activation_amax is None
        and options.svdquant_gscale_x is None
    ):
        raise ValueError(
            "tllm-svdquant-nvfp4-fused requires "
            "--tllm_svdquant_activation_amax or --tllm_svdquant_gscale_x."
        )
    if (
        options.svdquant_clear_cache_after_forward
        and not options.svdquant_clone_output
    ):
        raise ValueError(
            "--tllm_svdquant_clear_cache_after_forward requires "
            "--tllm_svdquant_clone_output."
        )


def replace_tllm_linear_layers(
    model: nn.Module,
    backend: LinearBackendType,
    options: TllmLinearOptions,
    *,
    device: str,
    fp8_override_prefixes: tuple[str, ...] | None = None,
    fp8_override_suffixes: tuple[str, ...] | None = None,
    conversion_state: TllmConversionState | None = None,
) -> nn.Module:
    """Replace eligible linears in one model subtree in place."""
    if backend not in TLLM_LINEAR_BACKENDS:
        raise ValueError(f"{backend.value} is not a tllm_linear_lite backend.")
    _validate_options(backend, options)

    replacements: list[tuple[str, nn.Module, str]] = []
    counts: Counter[str] = Counter()
    budget = options.svdquant_nvfp4_state_cache_budget_gb
    budget_bytes = None if budget is None else int(budget * 1024**3)
    if conversion_state is None:
        conversion_state = TllmConversionState()

    linear_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    for name, module in linear_modules:
        force_fp8 = matches_fp8_override(
            name, fp8_override_prefixes, fp8_override_suffixes
        )
        if (
            backend == LinearBackendType.TLLM_SVDQUANT_NVFP4_FUSED
            and budget_bytes is not None
            and not force_fp8
            and not needs_small_batch_support(name)
            and is_svdquant_nvfp4_eligible(module)
        ):
            estimated = estimate_svdquant_nvfp4_state_bytes(
                module,
                state_m=options.svdquant_nvfp4_state_m,
                rank=options.svdquant_rank,
            )
            if (
                conversion_state.nvfp4_state_bytes_used + estimated
                <= budget_bytes
            ):
                replacement = build_svdquant_nvfp4_linear(
                    module, options, device
                )
                kind = "svdquant-nvfp4"
                conversion_state.nvfp4_state_bytes_used += estimated
            else:
                replacement, kind = _build_svdquant_fp8(
                    module, options, device
                )
                kind = "budget-fallback-" + kind
        else:
            replacement, kind = _select_replacement(
                name, module, backend, options, device, force_fp8
            )
        counts[kind] += 1
        if replacement is not module:
            replacements.append((name, replacement, kind))

    for name, replacement, kind in replacements:
        if kind == "svdquant-nvfp4":
            replacement = wrap_svdquant_debug(name, replacement)
            replacement = wrap_svdquant_clear_cache(
                name,
                replacement,
                options.svdquant_clear_cache_after_forward,
            )
        parent, child_name = get_parent_module(model, name)
        setattr(parent, child_name, replacement)

    log(
        f"tllm_linear_lite {backend.value}: replaced {len(replacements)} "
        f"linear layers; routing={dict(counts)}"
    )
    if budget_bytes is not None:
        log(
            "tllm_linear_lite SVDQuant NVFP4 state cache estimate: "
            f"{conversion_state.nvfp4_state_bytes_used / 1024**3:.2f}/"
            f"{budget_bytes / 1024**3:.2f} GiB"
        )
    return model
