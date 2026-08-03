"""Optional diagnostic and cache-lifetime wrappers for SVDQuant."""

from __future__ import annotations

import os

import torch.nn as nn

from xfuser.logger import init_logger


logger = init_logger(__name__)


def _debug_enabled() -> bool:
    values = (
        os.environ.get("XDIT_SVDQUANT_DEBUG", ""),
        os.environ.get("WAN22_SVDQUANT_DEBUG", ""),
    )
    return any(value.lower() in {"1", "true", "yes", "on"} for value in values)


def wrap_svdquant_debug(qualified_name: str, module: nn.Module) -> nn.Module:
    """Log the latest fused state if an opted-in SVDQuant forward fails."""
    if not _debug_enabled() or getattr(module, "_xdit_debug_wrapped", False):
        return module
    if not hasattr(module, "_state_cache"):
        return module
    original_forward = module.forward

    def debug_forward(inputs, *args, **kwargs):
        try:
            return original_forward(inputs, *args, **kwargs)
        except Exception:
            try:
                states = list(getattr(module, "_state_cache", {}).values())
                state = states[-1] if states else None
                logger.error(
                    "SVDQuant NVFP4 failure: name=%s input_shape=%s state=%s",
                    qualified_name,
                    tuple(inputs.shape),
                    state,
                )
            except Exception as error:  # pragma: no cover - best-effort logging
                logger.error(
                    "SVDQuant NVFP4 debug logging failed for %s: %r",
                    qualified_name,
                    error,
                )
            raise

    module.forward = debug_forward
    module._xdit_debug_wrapped = True
    return module


def wrap_svdquant_clear_cache(
    qualified_name: str,
    module: nn.Module,
    enabled: bool,
) -> nn.Module:
    """Clear a fused layer's shape-state cache after every forward."""
    if not enabled or getattr(module, "_xdit_clear_cache_wrapped", False):
        return module
    if not hasattr(module, "clear_state_cache"):
        return module
    if not getattr(module, "clone_output", True):
        raise RuntimeError(
            "tllm_svdquant_clear_cache_after_forward requires clone_output=True "
            f"for {qualified_name}."
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
