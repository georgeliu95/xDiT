"""Optional NVIDIA attention adapters used by the unified backend registry."""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Callable

import torch


class SageImplementation(str, Enum):
    """Explicit SageAttention kernel choices retained from the Wan runners."""

    FP16 = "fp16"
    FP8 = "fp8"
    FP8_SM90 = "fp8_sm90"
    FP16_TRITON = "fp16_triton"


def flashinfer_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    dropout_p: float,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run generic FlashInfer prefill attention for tensors in BHSD layout."""
    if dropout_p != 0.0:
        raise ValueError("FlashInfer prefill attention does not support dropout.")

    from flashinfer.prefill import single_prefill_with_kv_cache

    query_bshd = query.transpose(1, 2).contiguous()
    key_bshd = key.transpose(1, 2).contiguous()
    value_bshd = value.transpose(1, 2).contiguous()
    outputs: list[torch.Tensor] = []
    logsumexp: list[torch.Tensor] = []
    for query_item, key_item, value_item in zip(
        query_bshd, key_bshd, value_bshd, strict=True
    ):
        output, lse = single_prefill_with_kv_cache(
            query_item,
            key_item,
            value_item,
            sm_scale=query.shape[-1] ** -0.5,
            causal=is_causal,
            return_lse=True,
        )
        outputs.append(output)
        logsumexp.append(lse.transpose(0, 1) / math.log2(math.e))
    return (
        torch.stack(outputs).transpose(1, 2),
        torch.stack(logsumexp),
    )


def _sage_function(
    implementation: SageImplementation,
    *,
    device: torch.device,
) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    import sageattention

    match implementation:
        case SageImplementation.FP16:
            return sageattention.sageattn_qk_int8_pv_fp16_cuda
        case SageImplementation.FP8:
            if torch.cuda.get_device_capability(device) == (9, 0):
                return sageattention.sageattn_qk_int8_pv_fp8_cuda_sm90
            return sageattention.sageattn_qk_int8_pv_fp8_cuda
        case SageImplementation.FP8_SM90:
            return sageattention.sageattn_qk_int8_pv_fp8_cuda_sm90
        case SageImplementation.FP16_TRITON:
            return sageattention.sageattn_qk_int8_pv_fp16_triton
    raise ValueError(f"Unsupported SageAttention implementation: {implementation}")


def sage_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    is_causal: bool,
    implementation: SageImplementation,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run an explicit SageAttention implementation for BHSD inputs."""
    function = _sage_function(implementation, device=query.device)
    kwargs = {
        "is_causal": is_causal,
        "tensor_layout": "NHD",
        "return_lse": True,
    }
    if implementation in (SageImplementation.FP8, SageImplementation.FP8_SM90):
        kwargs["pv_accum_dtype"] = "fp32+fp32"
    elif implementation is SageImplementation.FP16:
        kwargs["pv_accum_dtype"] = "fp32"

    output, lse = function(
        query.transpose(1, 2).contiguous(),
        key.transpose(1, 2).contiguous(),
        value.transpose(1, 2).contiguous(),
        **kwargs,
    )
    return output.transpose(1, 2), lse


def create_sparse_sage_processor() -> Any:
    """Construct one stateful Sparse Sage tuner for a Wan attention layer."""
    from spas_sage_attn.autotune import SparseAttentionMeansim

    return SparseAttentionMeansim()


def sparse_sage_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    is_causal: bool,
    attention_kwargs: dict[str, Any] | None,
) -> tuple[torch.Tensor, None]:
    """Run Sparse Sage, tuning its per-layer state on the first invocation."""
    processor = (attention_kwargs or {}).get("sparse_sage_processor")
    if processor is None:
        raise RuntimeError(
            "Sparse Sage requires a per-layer processor from the Wan attention adapter."
        )
    output = processor(
        query,
        key,
        value,
        is_causal=is_causal,
        tensor_layout="HND",
        tune_mode=getattr(processor, "is_sparse", None) is None,
    )
    return output, None
