"""Public linear backend names and compatibility aliases."""

from __future__ import annotations

from enum import Enum


class LinearBackendType(str, Enum):
    """Linear implementations supported by the unified model runner."""

    BF16 = "bf16"
    TORCHAO_FP8 = "torchao-fp8"
    TORCHAO_NVFP4 = "torchao-nvfp4"
    AITER_FP8_BLOCKWISE = "aiter-fp8-blockwise"
    AITER_MXFP4 = "aiter-mxfp4"
    TLLM_NVFP4 = "tllm-nvfp4"
    TLLM_FP8_BLOCKWISE = "tllm-fp8-blockwise"
    TLLM_NVFP4_FP8_BLOCKWISE = "tllm-nvfp4-fp8-blockwise"
    TLLM_SVDQUANT_FP8_BLOCKWISE = "tllm-svdquant-fp8-blockwise"
    TLLM_SVDQUANT_NVFP4_FUSED = "tllm-svdquant-nvfp4-fused"


_ALIASES = {
    "nvfp4": LinearBackendType.TLLM_NVFP4,
    "trtllm-fp8-blockwise": LinearBackendType.TLLM_FP8_BLOCKWISE,
    "nvfp4+trtllm-fp8-blockwise": (
        LinearBackendType.TLLM_NVFP4_FP8_BLOCKWISE
    ),
    "svdquant-fp8-blockwise": LinearBackendType.TLLM_SVDQUANT_FP8_BLOCKWISE,
    "svdquant-nvfp4-fused": LinearBackendType.TLLM_SVDQUANT_NVFP4_FUSED,
}

TLLM_LINEAR_BACKENDS = frozenset(
    backend for backend in LinearBackendType if backend.name.startswith("TLLM_")
)

FP4_LINEAR_BACKENDS = frozenset(
    {
        LinearBackendType.TORCHAO_NVFP4,
        LinearBackendType.AITER_MXFP4,
        LinearBackendType.TLLM_NVFP4,
        LinearBackendType.TLLM_NVFP4_FP8_BLOCKWISE,
        LinearBackendType.TLLM_SVDQUANT_NVFP4_FUSED,
    }
)

FP8_LINEAR_BACKENDS = frozenset(
    {
        LinearBackendType.TORCHAO_FP8,
        LinearBackendType.AITER_FP8_BLOCKWISE,
        LinearBackendType.TLLM_FP8_BLOCKWISE,
        LinearBackendType.TLLM_SVDQUANT_FP8_BLOCKWISE,
    }
)


def parse_linear_backend(name: str | None) -> LinearBackendType | None:
    """Parse a canonical backend or a retired standalone-Wan backend name."""
    if name is None:
        return None
    normalized = name.strip().lower().replace("_", "-")
    if normalized in _ALIASES:
        return _ALIASES[normalized]
    try:
        return LinearBackendType(normalized)
    except ValueError as error:
        supported = ", ".join(backend.value for backend in LinearBackendType)
        raise ValueError(
            f"Invalid linear backend '{name}'. Supported backends: {supported}."
        ) from error


def uses_tllm_linear_lite(backend: LinearBackendType | None) -> bool:
    """Return whether the selected backend is provided by tllm_linear_lite."""
    return backend in TLLM_LINEAR_BACKENDS
