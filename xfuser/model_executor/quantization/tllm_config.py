"""Configuration passed through to tllm_linear_lite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TllmLinearOptions:
    """Construction options shared by tllm_linear_lite linear adapters.

    Kernel, quantization, and scale backend strings are intentionally not
    validated here. The installed tllm_linear_lite version owns that contract.
    """

    nvfp4_gemm_backend: str = "heuristic"
    nvfp4_quant_backend: str = "tllm"
    nvfp4_scale_rule: str = "static_6"
    fp8_gemm_backend: str = "auto"
    svdquant_rank: int = 32
    svdquant_alpha: float = 0.5
    svdquant_method: str = "svd"
    svdquant_use_ue8m0: bool = False
    svdquant_activation_amax: float | None = None
    svdquant_gscale_x: float | None = None
    svdquant_clone_output: bool = True
    svdquant_max_cached_states: int = 4
    svdquant_clear_cache_after_forward: bool = False
    svdquant_nvfp4_state_cache_budget_gb: float | None = None
    svdquant_nvfp4_state_m: int = 7800

    @classmethod
    def from_config(cls, config: Any) -> "TllmLinearOptions":
        """Create options from xFuserArgs without importing the CLI module."""
        return cls(
            nvfp4_gemm_backend=config.tllm_nvfp4_gemm_backend,
            nvfp4_quant_backend=config.tllm_nvfp4_quant_backend,
            nvfp4_scale_rule=config.tllm_nvfp4_scale_rule,
            fp8_gemm_backend=config.tllm_fp8_gemm_backend,
            svdquant_rank=config.tllm_svdquant_rank,
            svdquant_alpha=config.tllm_svdquant_alpha,
            svdquant_method=config.tllm_svdquant_method,
            svdquant_use_ue8m0=config.tllm_svdquant_use_ue8m0,
            svdquant_activation_amax=config.tllm_svdquant_activation_amax,
            svdquant_gscale_x=config.tllm_svdquant_gscale_x,
            svdquant_clone_output=config.tllm_svdquant_clone_output,
            svdquant_max_cached_states=config.tllm_svdquant_max_cached_states,
            svdquant_clear_cache_after_forward=(
                config.tllm_svdquant_clear_cache_after_forward
            ),
            svdquant_nvfp4_state_cache_budget_gb=(
                config.tllm_svdquant_nvfp4_state_cache_budget_gb
            ),
            svdquant_nvfp4_state_m=config.tllm_svdquant_nvfp4_state_m,
        )
