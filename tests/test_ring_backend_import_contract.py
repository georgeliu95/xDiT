"""Import contracts for selecting the Ring attention backend."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
RING_PACKAGE = ROOT / "xfuser" / "core" / "long_ctx_attention" / "ring"


def _load_ring_package(*, is_npu: bool) -> ModuleType:
    package_name = "ring_backend_contract_target"

    envs = ModuleType("xfuser.envs")
    envs._is_npu = lambda: is_npu
    xfuser = ModuleType("xfuser")
    xfuser.__path__ = []
    xfuser.envs = envs

    ring_flash = ModuleType(f"{package_name}.ring_flash_attn")
    ring_flash.xdit_ring_flash_attn_func = object()
    ring_flash.xdit_sana_ring_flash_attn_func = object()

    ring_npu = ModuleType(f"{package_name}.ring_npu_flash_attn")
    if is_npu:
        ring_npu.xdit_ring_npu_flash_attn_func = object()

    modules = {
        "xfuser": xfuser,
        "xfuser.envs": envs,
        f"{package_name}.ring_flash_attn": ring_flash,
        f"{package_name}.ring_npu_flash_attn": ring_npu,
    }
    spec = importlib.util.spec_from_file_location(
        package_name,
        RING_PACKAGE / "__init__.py",
        submodule_search_locations=[str(RING_PACKAGE)],
    )
    assert spec is not None and spec.loader is not None

    package = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(package)
    return package


class RingBackendImportContractTest(unittest.TestCase):
    def test_cuda_path_does_not_require_npu_backend(self) -> None:
        package = _load_ring_package(is_npu=False)

        self.assertEqual(
            package.__all__,
            (
                "xdit_ring_flash_attn_func",
                "xdit_sana_ring_flash_attn_func",
            ),
        )
        self.assertFalse(hasattr(package, "xdit_ring_npu_flash_attn_func"))

    def test_npu_path_exports_npu_backend(self) -> None:
        package = _load_ring_package(is_npu=True)

        self.assertIn("xdit_ring_npu_flash_attn_func", package.__all__)
        self.assertTrue(hasattr(package, "xdit_ring_npu_flash_attn_func"))


if __name__ == "__main__":
    unittest.main()
