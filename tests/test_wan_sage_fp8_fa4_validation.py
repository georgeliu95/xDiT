"""Executable validation tests for Sage FP8 self-attention plus FA4 cross-attention."""

from __future__ import annotations

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.test_wan_fa4_validation import _load_module


class WanSageFp8Fa4ValidationTest(unittest.TestCase):
    def test_rejects_parallel_launch_with_typed_error(self) -> None:
        module, _, _ = _load_module(
            package_available=True,
            sage_available=True,
            cuda_available=True,
            capability=(12, 0),
        )

        with patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            with self.assertRaises(module.WanSageFp8ParallelismError):
                module.validate_wan_fa4_request("sage_fp8", 1, 1)

    def test_rejects_missing_sage_fa4_or_non_sm120_with_typed_error(self) -> None:
        cases = (
            (False, True, True, (12, 0)),
            (True, False, True, (12, 0)),
            (True, True, True, (10, 0)),
            (True, True, False, None),
        )
        for sage_available, fa4_available, cuda_available, capability in cases:
            with self.subTest(
                sage_available=sage_available,
                fa4_available=fa4_available,
                capability=capability,
            ):
                module, _, _ = _load_module(
                    package_available=fa4_available,
                    sage_available=sage_available,
                    cuda_available=cuda_available,
                    capability=capability,
                )
                with self.assertRaises(module.WanSageFp8RuntimeError):
                    module.validate_wan_fa4_request("sage_fp8", 1, 1)

    def test_configures_sage_fp8_self_and_fa4_cross_attention(self) -> None:
        module, runtime, state = _load_module(
            package_available=True,
            sage_available=True,
            cuda_available=True,
            capability=(12, 0),
        )
        block = SimpleNamespace(
            attn1=SimpleNamespace(processor=None),
            attn2=SimpleNamespace(processor=None),
        )
        transformer = SimpleNamespace(blocks=[block])

        module.configure_wan_sage_fp8_fa4_single_device(transformer, 1)

        self.assertTrue(state["initialized"])
        self.assertEqual(runtime.main_backend, "sage")
        self.assertEqual(runtime.cross_backend, "flash_4")
        self.assertEqual(block.attn1.processor.args, ("sage_fp8",))
        self.assertEqual(block.attn1.processor.kwargs, {"single_device": True})
        self.assertEqual(block.attn2.processor.args, ())
        self.assertEqual(
            block.attn2.processor.kwargs,
            {
                "use_ulysses_parallel_attention": False,
                "is_cross_attention": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
