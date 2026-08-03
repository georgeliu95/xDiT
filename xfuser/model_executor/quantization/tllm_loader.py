"""Lazy loading for the optional tllm_linear_lite dependency."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TllmLinearClasses:
    """Linear classes available in the installed dependency revision."""

    nvfp4: type
    fp8_blockscale: type | None
    svdquant_fp8: type | None
    svdquant: type | None
    fp8_import_error: ImportError | None = None
    svdquant_import_error: ImportError | None = None


def _dependency_root() -> Path:
    configured = os.environ.get("TLLM_LINEAR_LITE_ROOT")
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[3] / "third_party" / "tllm_linear_lite"


def _add_dependency_to_path() -> None:
    root = _dependency_root()
    if root.is_dir() and str(root) not in sys.path:
        sys.path.insert(0, str(root))


@lru_cache(maxsize=1)
def load_tllm_linear_classes() -> TllmLinearClasses:
    """Load dependency classes only when a tllm backend is selected."""
    _add_dependency_to_path()
    try:
        from tllm_linear_lite.layers.linear import NVFP4DynamicLinear
    except ImportError as error:
        root = _dependency_root()
        raise ImportError(
            "Unable to import tllm_linear_lite. Initialize/build the tracked "
            f"submodule at {root}, install the package, or set "
            "TLLM_LINEAR_LITE_ROOT to a built checkout."
        ) from error

    try:
        from tllm_linear_lite.layers.linear import (
            FP8BlockScaleDynamicLinear,
        )
    except ImportError:
        FP8BlockScaleDynamicLinear = None

    try:
        from tllm_linear_lite.layers.linear import (
            SVDQuantFP8BlockScaleLinear,
        )
    except ImportError as error:
        SVDQuantFP8BlockScaleLinear = None
        fp8_import_error = error
    else:
        fp8_import_error = None

    try:
        from tllm_linear_lite.layers.linear import SVDQuantLinear
    except ImportError as error:
        SVDQuantLinear = None
        svdquant_import_error = error
    else:
        svdquant_import_error = None

    return TllmLinearClasses(
        nvfp4=NVFP4DynamicLinear,
        fp8_blockscale=FP8BlockScaleDynamicLinear,
        svdquant_fp8=SVDQuantFP8BlockScaleLinear,
        svdquant=SVDQuantLinear,
        fp8_import_error=fp8_import_error,
        svdquant_import_error=svdquant_import_error,
    )
