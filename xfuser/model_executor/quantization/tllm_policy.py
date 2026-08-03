"""Eligibility and routing policy for tllm_linear_lite conversion."""

from __future__ import annotations

import torch
import torch.nn as nn


def get_parent_module(
    model: nn.Module, qualified_name: str
) -> tuple[nn.Module, str]:
    """Resolve a dotted module name to its parent and child token."""
    tokens = qualified_name.split(".")
    parent = model
    for token in tokens[:-1]:
        parent = parent[int(token)] if token.isdigit() else getattr(parent, token)
    return parent, tokens[-1]


def is_wan_major_linear(qualified_name: str) -> bool:
    """Return whether a Wan linear is a Q/K/V or FFN projection."""
    tokens = qualified_name.split(".")
    return any(token in {"to_q", "to_k", "to_v"} for token in tokens) or (
        "ffn" in tokens
    )


def needs_small_batch_support(qualified_name: str) -> bool:
    """Identify linears whose batch shapes are unsupported by block kernels."""
    return qualified_name.startswith("condition_embedder.time_embedder.") or (
        qualified_name == "condition_embedder.time_proj"
    )


def is_nvfp4_eligible(module: nn.Linear) -> bool:
    return module.in_features % 16 == 0


def is_fp8_blockscale_eligible(module: nn.Linear) -> bool:
    return module.in_features % 128 == 0 and module.out_features % 128 == 0


def is_svdquant_fp8_eligible(module: nn.Linear) -> bool:
    return (
        module.in_features % 128 == 0
        and module.out_features % 16 == 0
        and module.weight.dtype in (torch.bfloat16, torch.float16)
    )


def is_svdquant_nvfp4_eligible(module: nn.Linear) -> bool:
    return module.in_features % 16 == 0 and module.weight.dtype == torch.bfloat16


def matches_fp8_override(
    qualified_name: str,
    prefixes: tuple[str, ...] | None,
    suffixes: tuple[str, ...] | None,
) -> bool:
    """Match the same per-block FP8 overrides as native FP4 conversion."""
    return bool(
        (prefixes and qualified_name.startswith(prefixes))
        or (suffixes and qualified_name.endswith(suffixes))
    )


def estimate_svdquant_nvfp4_state_bytes(
    module: nn.Linear,
    *,
    state_m: int,
    rank: int,
) -> int:
    """Approximate persistent state cached by fused SVDQuant NVFP4."""
    m = max(int(state_m), 1)
    in_features = int(module.in_features)
    out_features = int(module.out_features)
    residual_rank = max(min(int(rank), in_features, out_features), 0)
    scale_features = in_features // 16
    return int(
        m * in_features * 4
        + m * in_features * 2
        + m * (in_features // 2)
        + m * scale_features
        + m * residual_rank * 2
        + m * (in_features // 2)
        + 2 * m * scale_features
        + m * out_features * 2
    )
