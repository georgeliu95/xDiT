"""Fail-closed preflight for the xDiT FA4 CUDA 13 runtime image."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import os


DEFAULT_FLASH_ATTN_4_VERSION = "4.0.0b22"
DEFAULT_CUTLASS_DSL_VERSION = "4.6.0.dev0"


class Fa4RuntimeProbeError(RuntimeError):
    """The image cannot safely run the requested FA4 workload."""


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-device-check",
        action="store_true",
        help="Verify packages and imports only; intended for docker build.",
    )
    return parser.parse_args()


def _require_version(distribution: str, expected: str) -> str:
    installed = metadata.version(distribution)
    if installed != expected:
        raise Fa4RuntimeProbeError(
            f"unexpected_version:{distribution}:{installed}:expected:{expected}"
        )
    return installed


def _device_capability(skip_device_check: bool) -> list[int] | None:
    if skip_device_check:
        return None
    import torch

    if not torch.cuda.is_available():
        raise Fa4RuntimeProbeError("torch_cuda_unavailable")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 0):
        raise Fa4RuntimeProbeError(
            f"gpu_capability_not_sm120:{capability[0]}.{capability[1]}"
        )
    return list(capability)


def main() -> None:
    args = _arguments()
    expected_fa4 = os.environ.get(
        "FLASH_ATTN_4_VERSION",
        DEFAULT_FLASH_ATTN_4_VERSION,
    )
    expected_cutlass = os.environ.get(
        "CUTLASS_DSL_VERSION",
        DEFAULT_CUTLASS_DSL_VERSION,
    )

    import cutlass
    import cutlass.cute
    import torch
    from flash_attn import flash_attn_func as fa2
    from flash_attn.cute.interface import flash_attn_func as fa4

    payload = {
        "status": "passed",
        "flash_attn_version": metadata.version("flash-attn"),
        "flash_attn_4_version": _require_version("flash-attn-4", expected_fa4),
        "nvidia_cutlass_dsl_version": _require_version(
            "nvidia-cutlass-dsl",
            expected_cutlass,
        ),
        "fa2_module": fa2.__module__,
        "fa4_module": fa4.__module__,
        "cutlass_module": cutlass.__name__,
        "gpu_capability": _device_capability(args.skip_device_check),
        "torch_version": torch.__version__,
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
