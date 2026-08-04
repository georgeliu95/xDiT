"""Regression coverage for architecture-aware SageAttention dispatch."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "xfuser/core/distributed/nvidia_attention.py"
)


class _Tensor:
    """Minimal tensor surface exercised by the xDiT Sage adapter."""

    device = "cuda:0"

    def transpose(self, _dim0: int, _dim1: int) -> _Tensor:
        return self

    def contiguous(self) -> _Tensor:
        return self


def _load_adapter() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "xdit_nvidia_attention_dispatch_test",
        MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_sage_fp8(
    *,
    compute_capability: tuple[int, int],
    explicit_sm90: bool = False,
) -> list[str]:
    calls: list[str] = []
    torch_module = ModuleType("torch")

    def get_device_capability(_device: str) -> tuple[int, int]:
        return compute_capability

    setattr(
        torch_module,
        "cuda",
        SimpleNamespace(get_device_capability=get_device_capability),
    )
    sage_module = ModuleType("sageattention")

    def generic_fp8(
        query: _Tensor,
        _key: _Tensor,
        _value: _Tensor,
        **_kwargs: bool | str,
    ) -> tuple[_Tensor, _Tensor]:
        calls.append("generic")
        return query, query

    def sm90_fp8(
        query: _Tensor,
        _key: _Tensor,
        _value: _Tensor,
        **_kwargs: bool | str,
    ) -> tuple[_Tensor, _Tensor]:
        calls.append("sm90")
        return query, query

    setattr(sage_module, "sageattn_qk_int8_pv_fp8_cuda", generic_fp8)
    setattr(sage_module, "sageattn_qk_int8_pv_fp8_cuda_sm90", sm90_fp8)

    with patch.dict(
        sys.modules,
        {"torch": torch_module, "sageattention": sage_module},
    ):
        adapter = _load_adapter()
        tensor = _Tensor()
        implementation = (
            adapter.SageImplementation.FP8_SM90
            if explicit_sm90
            else adapter.SageImplementation.FP8
        )
        adapter.sage_attention(
            tensor,
            tensor,
            tensor,
            is_causal=False,
            implementation=implementation,
        )

    return calls


class SageAttentionArchitectureDispatchTest(unittest.TestCase):
    def test_sage_fp8_uses_sm90_kernel_on_sm90(self) -> None:
        # Given/When: the generic FP8 backend runs on an SM90 CUDA device.
        calls = _run_sage_fp8(compute_capability=(9, 0))

        # Then: SM90 must not silently execute the generic SM89 kernel family.
        self.assertEqual(calls, ["sm90"])

    def test_sage_fp8_keeps_generic_kernel_on_non_sm90_devices(self) -> None:
        # Given/When: the generic backend runs on its existing non-SM90 targets.
        calls = {
            capability: _run_sage_fp8(compute_capability=capability)
            for capability in ((8, 9), (12, 0), (12, 1))
        }

        # Then: the SM90 fix does not change those established routes.
        self.assertEqual(
            calls,
            {(8, 9): ["generic"], (12, 0): ["generic"], (12, 1): ["generic"]},
        )

    def test_explicit_sm90_backend_remains_pinned(self) -> None:
        # Given/When: the explicit SM90 backend is selected directly.
        calls = _run_sage_fp8(
            compute_capability=(9, 0),
            explicit_sm90=True,
        )

        # Then: it remains pinned to the specialized function.
        self.assertEqual(calls, ["sm90"])


if __name__ == "__main__":
    unittest.main()
