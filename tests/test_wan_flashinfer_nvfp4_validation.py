"""Executable validation tests for the legacy Wan FlashInfer NVFP4 route."""

from __future__ import annotations

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.test_wan_fa4_validation import _load_module


class WanFlashInferNvfp4ValidationTest(unittest.TestCase):
    def test_rejects_parallel_launch_with_typed_error(self) -> None:
        # Given: an available SM120 kernel requested with multiple processes.
        module, _, _ = _load_module(
            package_available=True,
            flashinfer_nvfp4_available=True,
            cuda_available=True,
            capability=(12, 0),
        )

        # When/Then: the route fails before distributed initialization.
        with patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            with self.assertRaises(module.WanFlashInferNvfp4ParallelismError):
                module.validate_wan_fa4_request("flashinfer_nvfp4", 1, 1)

    def test_rejects_missing_api_fa4_cross_or_non_sm120_with_typed_error(self) -> None:
        # Given: the NVFP4 API, FA4 cross backend, or exact SM120 device is absent.
        cases = (
            (False, True, True, (12, 0)),
            (True, False, True, (12, 0)),
            (True, True, True, (12, 1)),
            (True, True, False, None),
        )

        # When/Then: every unsupported host fails closed at validation time.
        for flashinfer_available, fa4_available, cuda_available, capability in cases:
            with self.subTest(
                flashinfer_available=flashinfer_available,
                fa4_available=fa4_available,
                capability=capability,
            ):
                module, _, _ = _load_module(
                    package_available=fa4_available,
                    flashinfer_nvfp4_available=flashinfer_available,
                    cuda_available=cuda_available,
                    capability=capability,
                )
                with self.assertRaises(module.WanFlashInferNvfp4RuntimeError):
                    module.validate_wan_fa4_request("flashinfer_nvfp4", 1, 1)

    def test_configures_nvfp4_self_and_bf16_fa4_cross_attention(self) -> None:
        # Given: a single-device Wan transformer on a supported host.
        module, runtime, state = _load_module(
            package_available=True,
            flashinfer_nvfp4_available=True,
            cuda_available=True,
            capability=(12, 0),
        )
        block = SimpleNamespace(
            attn1=SimpleNamespace(processor=None),
            attn2=SimpleNamespace(processor=None),
        )
        transformer = SimpleNamespace(blocks=[block])

        # When: the explicit legacy route is configured.
        module.configure_wan_flashinfer_nvfp4_single_device(transformer, 1)

        # Then: only same-shape self-attention uses the NVFP4 kernel.
        self.assertTrue(state["initialized"])
        self.assertEqual(runtime.main_backend, "flashinfer_nvfp4")
        self.assertEqual(runtime.cross_backend, "flash_4")
        self.assertEqual(
            block.attn1.processor.kwargs,
            {"use_ulysses_parallel_attention": False},
        )
        self.assertEqual(
            block.attn2.processor.kwargs,
            {
                "use_ulysses_parallel_attention": False,
                "is_cross_attention": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
