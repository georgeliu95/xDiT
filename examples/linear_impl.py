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

    try:
        from tllm_linear_lite.svdquant_fp8_linear import (  # noqa: PLC0415
            SVDQuantFP8BlockScaleLinear,
        )
    except ImportError as exc:
        SVDQuantFP8BlockScaleLinear = None
        svdquant_fp8_import_error = exc
    else:
        svdquant_fp8_import_error = None

    try:
        from tllm_linear_lite.svdquant_linear import SVDQuantLinear  # noqa: PLC0415
    except ImportError as exc:
        SVDQuantLinear = None
        svdquant_factory_import_error = exc
    else:
        svdquant_factory_import_error = None

    return (
        NVFP4DynamicLinear,
        FP8BlockScaleDynamicLinear,
        SVDQuantFP8BlockScaleLinear,
        SVDQuantLinear,
        svdquant_fp8_import_error,
        svdquant_factory_import_error,
    )


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


def _is_svdquant_fp8_eligible(module: nn.Linear) -> bool:
    return (
        module.in_features % 128 == 0
        and module.out_features % 16 == 0
        and module.weight.dtype in (torch.bfloat16, torch.float16)
    )


def _is_svdquant_nvfp4_eligible(module: nn.Linear) -> bool:
    return module.in_features % 16 == 0 and module.weight.dtype == torch.bfloat16


def _is_svdquant_nvfp4_fallback_shape(
    qualified_name: str,
    module: nn.Linear,
) -> bool:
    # The Wan2.2 FFN up-proj (5120 -> 13824) illegal-address crash is FIXED in the svdquant-nvfp4
    # fused kernel: activation row counts M that are not a multiple of the 128 MMA M-tile (e.g.
    # M=7800) are now predicated in the epilogue stores. No shape needs the bf16 fallback anymore.
    del qualified_name, module
    return False


def _estimate_svdquant_nvfp4_state_bytes(
    module: nn.Linear,
    *,
    state_m: int,
    rank: int,
) -> int:
    """Approximate persistent CUDA bytes in tllm_linear_lite's fused state.

    The current fused runtime caches activation staging, packed activation,
    SFA gather, down-projection, and output buffers per layer. Wan2.2 has
    hundreds of linear layers at the same latent M, so unbounded caching can
    exhaust B200 memory even though the quantized weights are small.
    """
    m = max(int(state_m), 1)
    i = int(module.in_features)
    o = int(module.out_features)
    r = max(min(int(rank), i, o), 0)
    sf_k = i // 16

    # Dynamic buffers created by prepare_svdquant_state:
    # xf fp32, xb bf16, xq uint8, sf uint8, D bf16, A raw buffer,
    # SFA + gather buffers, and the state-owned output C bf16.
    bytes_ = 0
    bytes_ += m * i * 4  # xf_t backing
    bytes_ += m * i * 2  # xb_t backing
    bytes_ += m * (i // 2)  # xq_torch
    bytes_ += m * sf_k  # sf_torch
    bytes_ += m * r * 2  # d_tensor
    bytes_ += m * (i // 2)  # a_buf
    bytes_ += 2 * m * sf_k  # sfa_torch + g_buf
    bytes_ += m * o * 2  # c_torch
    return int(bytes_)


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


def _svdquant_debug_enabled() -> bool:
    return os.environ.get("WAN22_SVDQUANT_DEBUG", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _maybe_wrap_svdquant_nvfp4_debug(
    qualified_name: str,
    module: nn.Module,
) -> nn.Module:
    if not _svdquant_debug_enabled() or getattr(module, "_xdit_debug_wrapped", False):
        return module
    if not hasattr(module, "_state_cache"):
        return module

    original_forward = module.forward

    def debug_forward(x, *args, **kwargs):
        try:
            return original_forward(x, *args, **kwargs)
        except Exception:
            try:
                states = list(getattr(module, "_state_cache", {}).values())
                state = states[-1] if states else None
                if state is None:
                    logger.error(
                        (
                            "SVDQuant NVFP4 failure: name={} input_shape={} "
                            "in_features={} out_features={} state=missing"
                        ),
                        qualified_name,
                        tuple(x.shape),
                        getattr(module, "in_features", None),
                        getattr(module, "out_features", None),
                    )
                else:
                    logger.error(
                        (
                            "SVDQuant NVFP4 failure: name={} input_shape={} "
                            "M={} I={} O={} r={} enable_lora={} use_2cta={} "
                            "down_split={} down_m_tile={} down_n_tile={} "
                            "num_ab_stage={} weight_fp4_shape={} weight_sf_shape={}"
                        ),
                        qualified_name,
                        tuple(x.shape),
                        state.get("M"),
                        state.get("I"),
                        state.get("O"),
                        state.get("r"),
                        state.get("enable_lora"),
                        state.get("use_2cta"),
                        state.get("down_split"),
                        state.get("down_m_tile"),
                        state.get("down_n_tile"),
                        state.get("num_ab_stage"),
                        tuple(getattr(module, "weight_fp4").shape),
                        tuple(getattr(module, "weight_sf").shape),
                    )
            except Exception as debug_exc:  # pragma: no cover - best-effort logging
                logger.error(
                    "SVDQuant NVFP4 debug logging failed for {}: {}",
                    qualified_name,
                    repr(debug_exc),
                )
            raise

    module.forward = debug_forward
    module._xdit_debug_wrapped = True
    return module


def _maybe_wrap_svdquant_nvfp4_clear_cache(
    qualified_name: str,
    module: nn.Module,
    enabled: bool,
) -> nn.Module:
    if not enabled or getattr(module, "_xdit_clear_cache_wrapped", False):
        return module
    if not hasattr(module, "clear_state_cache"):
        return module
    if not getattr(module, "clone_output", True):
        raise RuntimeError(
            "svdquant_clear_cache_after_forward requires clone_output=True for "
            f"{qualified_name}; otherwise the returned tensor may alias a "
            "state-owned buffer that is freed after forward."
        )

    original_forward = module.forward

    def clear_cache_forward(*args, **kwargs):
        try:
            return original_forward(*args, **kwargs)
        finally:
            module.clear_state_cache()

    module.forward = clear_cache_forward
    module._xdit_clear_cache_wrapped = True
    return module


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
    NVFP4DynamicLinear, _, _, _, _, _ = _load_tllm_linear_classes()
    _ensure_tllm_op("fp4_quantize", "nvfp4")
    quant_module = _copy_linear_to_quant_device(module)
    return NVFP4DynamicLinear.from_linear(
        quant_module,
        gemm_backend=gemm_backend,
        scale_rule=scale_rule,
    )


def _build_fp8_blockscale_linear(module: nn.Linear) -> nn.Module:
    _, FP8BlockScaleDynamicLinear, _, _, _, _ = _load_tllm_linear_classes()
    if FP8BlockScaleDynamicLinear is None:
        raise RuntimeError(
            "The current tllm_linear_lite submodule does not provide "
            "FP8BlockScaleDynamicLinear. Use quant_gemm_type='nvfp4' on B200, "
            "or switch the submodule to an FP8-capable tllm_linear_lite commit."
        )
    _ensure_tllm_op("fp8_blockscale_quantize_128x128", "trtllm-fp8-blockwise")
    quant_module = _copy_linear_to_quant_device(module)
    return FP8BlockScaleDynamicLinear.from_linear(quant_module, gemm_backend="auto")


def _build_svdquant_fp8_linear(
    module: nn.Linear,
    rank: int = 32,
    alpha: float = 0.5,
    method: str = "svd",
    gemm_backend: str = "auto",
    use_ue8m0: bool = False,
) -> nn.Module:
    _, _, SVDQuantFP8BlockScaleLinear, _, import_error, _ = _load_tllm_linear_classes()
    if SVDQuantFP8BlockScaleLinear is None:
        message = (
            "The current tllm_linear_lite root does not provide "
            "SVDQuantFP8BlockScaleLinear. Point TLLM_LINEAR_LITE_ROOT at a "
            "checkout containing tllm_linear_lite/svdquant_fp8_linear.py. "
            f"Current root: {_TLLM_LINEAR_LITE_ROOT}"
        )
        if import_error is not None:
            raise RuntimeError(message) from import_error
        raise RuntimeError(message)

    _ensure_tllm_op("fp8_blockscale_quantize_128x128", "svdquant-fp8-blockwise")
    _ensure_tllm_op("fp8_blockscale_quantize_1x128", "svdquant-fp8-blockwise")
    quant_module = _copy_linear_to_quant_device(module)
    return SVDQuantFP8BlockScaleLinear.from_linear(
        quant_module,
        rank=rank,
        alpha=alpha,
        method=method,
        gemm_backend=gemm_backend,
        use_ue8m0=use_ue8m0,
    )


def _build_svdquant_nvfp4_linear(
    module: nn.Linear,
    rank: int = 32,
    alpha: float = 0.5,
    method: str = "svd",
    activation_amax: float | None = None,
    gscale_x: float | None = None,
    clone_output: bool = True,
    max_cached_states: int = 4,
) -> nn.Module:
    _, _, _, SVDQuantLinear, _, import_error = _load_tllm_linear_classes()
    if SVDQuantLinear is None:
        message = (
            "The current tllm_linear_lite root does not provide "
            "SVDQuantLinear. Point TLLM_LINEAR_LITE_ROOT at a checkout "
            "containing tllm_linear_lite/svdquant_linear.py. "
            f"Current root: {_TLLM_LINEAR_LITE_ROOT}"
        )
        if import_error is not None:
            raise RuntimeError(message) from import_error
        raise RuntimeError(message)
    if activation_amax is None and gscale_x is None:
        raise RuntimeError(
            "quant_gemm_type='svdquant-nvfp4-fused' requires a static "
            "activation scale. Pass --svdquant_activation_amax, or pass "
            "--svdquant_gscale_x directly."
        )

    _ensure_tllm_op("fp4_quantize", "svdquant-nvfp4-fused")
    quant_module = _copy_linear_to_quant_device(module)
    return SVDQuantLinear.from_linear(
        quant_module,
        backend="nvfp4_fused",
        rank=rank,
        alpha=alpha,
        method=method,
        activation_amax=activation_amax,
        gscale_x=gscale_x,
        clone_output=clone_output,
        max_cached_states=max_cached_states,
    )


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
        svdquant_rank: int = 32,
        svdquant_alpha: float = 0.5,
        svdquant_method: str = "svd",
        svdquant_gemm_backend: str = "auto",
        svdquant_use_ue8m0: bool = False,
        svdquant_activation_amax: float | None = None,
        svdquant_gscale_x: float | None = None,
        svdquant_clone_output: bool = True,
        svdquant_max_cached_states: int = 4,
        svdquant_clear_cache_after_forward: bool = False,
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
        if linear_type == "svdquant-fp8-blockwise":
            if not _is_svdquant_fp8_eligible(linear):
                return linear
            return _build_svdquant_fp8_linear(
                linear,
                rank=svdquant_rank,
                alpha=svdquant_alpha,
                method=svdquant_method,
                gemm_backend=svdquant_gemm_backend,
                use_ue8m0=svdquant_use_ue8m0,
            )
        if linear_type == "svdquant-nvfp4-fused":
            if not _is_svdquant_nvfp4_eligible(linear):
                return linear
            built = _build_svdquant_nvfp4_linear(
                linear,
                rank=svdquant_rank,
                alpha=svdquant_alpha,
                method=svdquant_method,
                activation_amax=svdquant_activation_amax,
                gscale_x=svdquant_gscale_x,
                clone_output=svdquant_clone_output,
                max_cached_states=svdquant_max_cached_states,
            )
            return _maybe_wrap_svdquant_nvfp4_clear_cache(
                "VflyLinear",
                built,
                svdquant_clear_cache_after_forward,
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
    svdquant_rank: int,
    svdquant_alpha: float,
    svdquant_method: str,
    svdquant_gemm_backend: str,
    svdquant_use_ue8m0: bool,
    svdquant_activation_amax: float | None,
    svdquant_gscale_x: float | None,
    svdquant_clone_output: bool,
    svdquant_max_cached_states: int,
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

    if quant_gemm_type == "svdquant-fp8-blockwise":
        if (
            not _needs_small_batch_support(qualified_name)
            and _is_svdquant_fp8_eligible(module)
        ):
            return _build_svdquant_fp8_linear(
                module,
                rank=svdquant_rank,
                alpha=svdquant_alpha,
                method=svdquant_method,
                gemm_backend=svdquant_gemm_backend,
                use_ue8m0=svdquant_use_ue8m0,
            ), "svdquant-fp8-blockwise"
        return module, "default"

    if quant_gemm_type == "svdquant-nvfp4-fused":
        if _is_svdquant_nvfp4_fallback_shape(qualified_name, module):
            return module, "svdquant-nvfp4-fallback-bf16"
        if (
            not _needs_small_batch_support(qualified_name)
            and _is_svdquant_nvfp4_eligible(module)
        ):
            return _build_svdquant_nvfp4_linear(
                module,
                rank=svdquant_rank,
                alpha=svdquant_alpha,
                method=svdquant_method,
                activation_amax=svdquant_activation_amax,
                gscale_x=svdquant_gscale_x,
                clone_output=svdquant_clone_output,
                max_cached_states=svdquant_max_cached_states,
            ), "svdquant-nvfp4-fused"
        return module, "default"

    raise ValueError(
        f"Unsupported quant_gemm_type '{quant_gemm_type}' in tllm_linear_lite adapter"
    )


def _build_svdquant_nvfp4_budget_fallback(
    module: nn.Linear,
    *,
    svdquant_rank: int,
    svdquant_alpha: float,
    svdquant_method: str,
    svdquant_gemm_backend: str,
    svdquant_use_ue8m0: bool,
) -> tuple[nn.Module, str]:
    if _is_svdquant_fp8_eligible(module):
        return _build_svdquant_fp8_linear(
            module,
            rank=svdquant_rank,
            alpha=svdquant_alpha,
            method=svdquant_method,
            gemm_backend=svdquant_gemm_backend,
            use_ue8m0=svdquant_use_ue8m0,
        ), "svdquant-nvfp4-budget-fallback-fp8"
    return module, "default"


@nvtx.annotate(message="replace_linear_layer", color="red")
def replace_linear_layer(
    model: nn.Module,
    quant_gemm_type: str = "nvfp4",
    enable_hadamard: bool = False,
    quant_recipe: object | None = None,
    nvfp4_gemm_backend: str = "cublaslt",
    nvfp4_scale_rule: str = "static_6",
    svdquant_rank: int = 32,
    svdquant_alpha: float = 0.5,
    svdquant_method: str = "svd",
    svdquant_gemm_backend: str = "auto",
    svdquant_use_ue8m0: bool = False,
    svdquant_activation_amax: float | None = None,
    svdquant_gscale_x: float | None = None,
    svdquant_clone_output: bool = True,
    svdquant_max_cached_states: int = 4,
    svdquant_clear_cache_after_forward: bool = False,
    svdquant_nvfp4_state_cache_budget_gb: float | None = None,
    svdquant_nvfp4_state_m: int = 7800,
) -> nn.Module:
    del enable_hadamard, quant_recipe

    if quant_gemm_type in (None, "bf16"):
        return model
    if quant_gemm_type == "svdquant.nvfp4":
        raise ValueError(
            "Use 'svdquant-nvfp4-fused' for the fused SVDQuant+NVFP4 "
            "backend, or 'svdquant-fp8-blockwise' for the FP8 backend."
        )
    if (
        quant_gemm_type == "svdquant-nvfp4-fused"
        and svdquant_clear_cache_after_forward
        and not svdquant_clone_output
    ):
        raise RuntimeError(
            "svdquant_clear_cache_after_forward requires "
            "svdquant_clone_output=True because the fused runtime output may "
            "alias state-owned storage."
        )

    replacements: list[tuple[str, nn.Module, str]] = []
    counts = {
        "nvfp4": 0,
        "trtllm-fp8-blockwise": 0,
        "svdquant-fp8-blockwise": 0,
        "svdquant-nvfp4-fused": 0,
        "svdquant-nvfp4-budget-fallback-fp8": 0,
        "svdquant-nvfp4-fallback-bf16": 0,
        "default": 0,
    }
    nvfp4_budget_bytes = None
    if (
        quant_gemm_type == "svdquant-nvfp4-fused"
        and svdquant_nvfp4_state_cache_budget_gb is not None
    ):
        nvfp4_budget_bytes = int(svdquant_nvfp4_state_cache_budget_gb * 1024**3)
    nvfp4_state_bytes_used = 0

    linear_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    for name, module in linear_modules:
        if (
            quant_gemm_type == "svdquant-nvfp4-fused"
            and nvfp4_budget_bytes is not None
            and not _is_svdquant_nvfp4_fallback_shape(name, module)
            and not _needs_small_batch_support(name)
            and _is_svdquant_nvfp4_eligible(module)
        ):
            estimated_bytes = _estimate_svdquant_nvfp4_state_bytes(
                module,
                state_m=svdquant_nvfp4_state_m,
                rank=svdquant_rank,
            )
            if nvfp4_state_bytes_used + estimated_bytes <= nvfp4_budget_bytes:
                wrapped_module = _build_svdquant_nvfp4_linear(
                    module,
                    rank=svdquant_rank,
                    alpha=svdquant_alpha,
                    method=svdquant_method,
                    activation_amax=svdquant_activation_amax,
                    gscale_x=svdquant_gscale_x,
                    clone_output=svdquant_clone_output,
                    max_cached_states=svdquant_max_cached_states,
                )
                replacement_type = "svdquant-nvfp4-fused"
                nvfp4_state_bytes_used += estimated_bytes
            else:
                wrapped_module, replacement_type = _build_svdquant_nvfp4_budget_fallback(
                    module,
                    svdquant_rank=svdquant_rank,
                    svdquant_alpha=svdquant_alpha,
                    svdquant_method=svdquant_method,
                    svdquant_gemm_backend=svdquant_gemm_backend,
                    svdquant_use_ue8m0=svdquant_use_ue8m0,
                )
        else:
            wrapped_module, replacement_type = _select_replacement(
                name,
                module,
                quant_gemm_type,
                nvfp4_gemm_backend,
                nvfp4_scale_rule,
                svdquant_rank,
                svdquant_alpha,
                svdquant_method,
                svdquant_gemm_backend,
                svdquant_use_ue8m0,
                svdquant_activation_amax,
                svdquant_gscale_x,
                svdquant_clone_output,
                svdquant_max_cached_states,
            )
        counts[replacement_type] += 1
        if wrapped_module is not module:
            replacements.append((name, wrapped_module, replacement_type))

    for name, wrapped_module, _ in replacements:
        parent, child_name = _get_parent_module(model, name)
        if quant_gemm_type == "svdquant-nvfp4-fused":
            wrapped_module = _maybe_wrap_svdquant_nvfp4_debug(name, wrapped_module)
            wrapped_module = _maybe_wrap_svdquant_nvfp4_clear_cache(
                name,
                wrapped_module,
                svdquant_clear_cache_after_forward,
            )
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
        logger.info(
            "Replaced {} linear layers with tllm_linear_lite SVDQuant FP8 blockscale",
            counts["svdquant-fp8-blockwise"],
        )
        if counts["svdquant-fp8-blockwise"]:
            logger.info(
                (
                    "tllm_linear_lite SVDQuant FP8 config: rank={}, alpha={}, "
                    "method={}, gemm_backend={}, use_ue8m0={}"
                ),
                svdquant_rank,
                svdquant_alpha,
                svdquant_method,
                svdquant_gemm_backend,
                svdquant_use_ue8m0,
            )
        logger.info(
            "Replaced {} linear layers with tllm_linear_lite SVDQuant fused NVFP4",
            counts["svdquant-nvfp4-fused"],
        )
        if counts["svdquant-nvfp4-fused"]:
            logger.info(
                (
                    "tllm_linear_lite SVDQuant NVFP4 config: rank={}, alpha={}, "
                    "method={}, activation_amax={}, gscale_x={}, clone_output={}, "
                    "max_cached_states={}, clear_cache_after_forward={}"
                ),
                svdquant_rank,
                svdquant_alpha,
                svdquant_method,
                svdquant_activation_amax,
                svdquant_gscale_x,
                svdquant_clone_output,
                svdquant_max_cached_states,
                svdquant_clear_cache_after_forward,
            )
            if nvfp4_budget_bytes is not None:
                logger.info(
                    (
                        "tllm_linear_lite SVDQuant NVFP4 state-cache budget: "
                        "{:.2f}/{:.2f} GiB at estimated M={}"
                    ),
                    nvfp4_state_bytes_used / 1024**3,
                    nvfp4_budget_bytes / 1024**3,
                    svdquant_nvfp4_state_m,
                )
        if counts["svdquant-nvfp4-budget-fallback-fp8"]:
            logger.info(
                (
                    "Fell back {} linear layers to SVDQuant FP8 blockscale "
                    "after SVDQuant NVFP4 state-cache budget"
                ),
                counts["svdquant-nvfp4-budget-fallback-fp8"],
            )
        if counts["svdquant-nvfp4-fallback-bf16"]:
            logger.info(
                (
                    "Kept {} Wan FFN up-proj layers in bf16 for "
                    "SVDQuant NVFP4 fused fallback"
                ),
                counts["svdquant-nvfp4-fallback-bf16"],
            )
        logger.info("Remained {} default linear layers", counts["default"])

    return model
