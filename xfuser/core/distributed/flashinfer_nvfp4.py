"""FlashInfer SM120/SM121 NVFP4 attention adapter."""

from __future__ import annotations

import torch


class FlashInferNvfp4InputError(ValueError):
    """Raised when an xDiT attention call violates the FlashInfer API contract."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"FlashInfer SM12x NVFP4 attention: {reason}")


class FlashInferNvfp4UnavailableError(RuntimeError):
    """Raised when the requested backend is unavailable in the current runtime."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"FlashInfer SM12x NVFP4 attention is unavailable: {reason}")


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float,
) -> None:
    if dropout_p != 0.0:
        raise FlashInferNvfp4InputError(
            f"dropout is unsupported; expected 0.0, received {dropout_p}"
        )
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise FlashInferNvfp4InputError("Q, K, and V must be rank-4 [B, H, S, D]")
    if query.shape != key.shape or query.shape != value.shape:
        raise FlashInferNvfp4InputError(
            "Q, K, and V must have the same shape; use a separate backend for "
            "cross-attention"
        )
    if query.shape[-1] not in (64, 128):
        raise FlashInferNvfp4InputError(
            f"head dimension must be 64 or 128, received {query.shape[-1]}"
        )
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise FlashInferNvfp4InputError("Q, K, and V must have the same dtype")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise FlashInferNvfp4InputError(
            f"Q, K, and V must use FP16 or BF16, received {query.dtype}"
        )


def flashinfer_nvfp4_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    dropout_p: float,
    is_causal: bool,
    per_block_mean: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize QKV, run FlashInfer's SM12x kernel, and remove internal padding."""
    _validate_inputs(query, key, value, dropout_p)

    from flashinfer import (
        nvfp4_attention_sm120_fwd,
        nvfp4_attention_sm120_quantize_qkv,
    )

    sequence_length = query.shape[-2]
    packed_qkv = nvfp4_attention_sm120_quantize_qkv(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        per_block_mean=per_block_mean,
    )
    output, softmax_lse = nvfp4_attention_sm120_fwd(
        *packed_qkv,
        causal=is_causal,
        per_block_mean=per_block_mean,
        out_dtype=query.dtype,
    )
    return (
        output[..., :sequence_length, :],
        softmax_lse[..., :sequence_length],
    )
