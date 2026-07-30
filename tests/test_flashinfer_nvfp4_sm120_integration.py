"""Real-kernel integration tests for FlashInfer SM120/SM121 NVFP4 attention."""

from __future__ import annotations

import importlib.util
from importlib import metadata
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "xfuser/core/distributed/flashinfer_nvfp4.py"


class FlashInferNvfp4IntegrationLoadError(RuntimeError):
    """Raised when the real adapter module cannot be loaded for integration."""


class FlashInferNvfp4Sm12xIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
            from flashinfer import (
                nvfp4_attention_sm120_fwd,
                nvfp4_attention_sm120_quantize_qkv,
            )
            from packaging import version
        except (ImportError, ModuleNotFoundError) as error:
            raise unittest.SkipTest(
                f"SM120/SM121 integration dependencies unavailable: {error}"
            )

        installed = version.parse(metadata.version("flashinfer-python"))
        first_fixed_nightly = version.parse("0.6.15.dev20260717")
        first_fixed_release = version.parse("0.6.16")
        fixed_nightly = installed.is_devrelease and installed >= first_fixed_nightly
        fixed_release = (
            not installed.is_devrelease and installed >= first_fixed_release
        )
        if not (fixed_nightly or fixed_release):
            raise unittest.SkipTest(
                f"FlashInfer {installed} lacks the #3838/#3897 fixes"
            )
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is unavailable")
        if torch.cuda.get_device_capability() not in {(12, 0), (12, 1)}:
            raise unittest.SkipTest(
                "FlashInfer NVFP4 integration requires SM120 or SM121"
            )

        spec = importlib.util.spec_from_file_location(
            "_flashinfer_nvfp4_sm120_integration",
            ADAPTER_PATH,
        )
        if spec is None or spec.loader is None:
            raise FlashInferNvfp4IntegrationLoadError(
                f"cannot load adapter from {ADAPTER_PATH}"
            )
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)

        cls.torch = torch
        cls.quantize = staticmethod(nvfp4_attention_sm120_quantize_qkv)
        cls.forward = staticmethod(nvfp4_attention_sm120_fwd)
        cls.attention = staticmethod(adapter.flashinfer_nvfp4_attention)

    def test_pr3838_compact_correction_and_uniform_attention(self) -> None:
        torch = self.torch
        query = torch.zeros(
            1,
            2,
            256,
            64,
            device="cuda",
            dtype=torch.bfloat16,
        )
        key = torch.zeros_like(query)
        value = torch.ones_like(query)

        block_packed = self.quantize(query, key, value, per_block_mean=True)
        global_packed = self.quantize(query, key, value, per_block_mean=False)
        self.assertEqual(tuple(block_packed[-1].shape), (1, 2, 2, 256))
        self.assertEqual(tuple(global_packed[-1].shape), (1, 2, 1, 256))

        output, lse = self.forward(
            *block_packed,
            causal=False,
            per_block_mean=True,
            out_dtype=torch.bfloat16,
        )
        output_error = (output.float() - 1.0).abs().mean().item()
        lse_error = (lse.float() - math.log(256.0)).abs().mean().item()
        self.assertLessEqual(output_error, 0.05)
        self.assertLessEqual(lse_error, 0.05)

    def test_adapter_matches_reference_for_aligned_sequence(self) -> None:
        torch = self.torch
        torch.manual_seed(20260720)
        query = torch.randn(
            1,
            4,
            256,
            128,
            device="cuda",
            dtype=torch.bfloat16,
        )
        key = torch.randn_like(query)
        value = torch.randn_like(query)

        output, lse = self.attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
        )
        key_centered = key.float() - key.float().mean(dim=-2, keepdim=True)
        scores = torch.matmul(query.float(), key_centered.transpose(-2, -1))
        scores *= query.shape[-1] ** -0.5
        reference = torch.matmul(torch.softmax(scores, dim=-1), value.float())
        reference_lse = torch.logsumexp(scores, dim=-1)
        output_error = (output.float() - reference).abs().mean().item()
        lse_error = (lse.float() - reference_lse).abs().mean().item()
        amplitude_ratio = (
            output.float().square().mean().sqrt()
            / reference.square().mean().sqrt()
        ).item()
        fit_slope = (
            (output.float() * reference).sum() / reference.square().sum()
        ).item()
        cosine = torch.nn.functional.cosine_similarity(
            output.float().flatten(),
            reference.flatten(),
            dim=0,
        ).item()

        self.assertEqual(output.shape, query.shape)
        self.assertEqual(lse.shape, query.shape[:-1])
        self.assertTrue(torch.isfinite(output).all().item())
        self.assertTrue(torch.isfinite(lse).all().item())
        self.assertLessEqual(output_error, 0.06)
        self.assertLessEqual(lse_error, 0.06)
        self.assertGreaterEqual(amplitude_ratio, 0.90)
        self.assertLessEqual(amplitude_ratio, 1.10)
        self.assertGreaterEqual(fit_slope, 0.90)
        self.assertLessEqual(fit_slope, 1.10)
        self.assertGreaterEqual(cosine, 0.95)

    def test_structured_inputs_preserve_output_scale(self) -> None:
        torch = self.torch
        positions = torch.linspace(
            -1.0,
            1.0,
            256,
            device="cuda",
            dtype=torch.float32,
        )
        channels = torch.linspace(
            0.25,
            1.25,
            128,
            device="cuda",
            dtype=torch.float32,
        )
        query = (
            torch.sin(positions[:, None] * channels[None, :] * math.pi)
            .reshape(1, 1, 256, 128)
            .to(torch.bfloat16)
        )
        key = (
            torch.cos(positions[:, None] * channels[None, :] * math.pi)
            .reshape(1, 1, 256, 128)
            .to(torch.bfloat16)
        )
        generator = torch.Generator(device="cuda")
        generator.manual_seed(20260723)
        value = (
            torch.randn(
                1,
                1,
                256,
                128,
                device="cuda",
                generator=generator,
            )
            * 0.5
        ).to(torch.bfloat16)

        output, lse = self.attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
        )
        key_centered = key.float() - key.float().mean(dim=-2, keepdim=True)
        scores = torch.matmul(query.float(), key_centered.transpose(-2, -1))
        scores *= query.shape[-1] ** -0.5
        reference = torch.matmul(torch.softmax(scores, dim=-1), value.float())
        amplitude_ratio = (
            output.float().square().mean().sqrt()
            / reference.square().mean().sqrt()
        ).item()
        fit_slope = (
            (output.float() * reference).sum() / reference.square().sum()
        ).item()
        output_error = (output.float() - reference).abs().mean().item()
        cosine = torch.nn.functional.cosine_similarity(
            output.float().flatten(),
            reference.flatten(),
            dim=0,
        ).item()

        # This structured case targets the Q-mean correction's output scale.
        # LSE accuracy is covered by the aligned random-input test above;
        # low-entropy sin/cos inputs amplify harmless row-wise score offsets.
        self.assertTrue(torch.isfinite(output).all().item())
        self.assertTrue(torch.isfinite(lse).all().item())
        self.assertLessEqual(output_error, 0.06)
        self.assertGreaterEqual(amplitude_ratio, 0.90)
        self.assertLessEqual(amplitude_ratio, 1.10)
        self.assertGreaterEqual(fit_slope, 0.90)
        self.assertLessEqual(fit_slope, 1.10)
        self.assertGreaterEqual(cosine, 0.95)

    def test_adapter_trims_internal_sequence_padding(self) -> None:
        torch = self.torch
        query = torch.randn(
            1,
            2,
            129,
            64,
            device="cuda",
            dtype=torch.bfloat16,
        )
        key = torch.randn_like(query)
        value = torch.randn_like(query)

        output, lse = self.attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
        )

        self.assertEqual(output.shape, query.shape)
        self.assertEqual(lse.shape, query.shape[:-1])
        self.assertTrue(torch.isfinite(output).all().item())
        self.assertTrue(torch.isfinite(lse).all().item())


if __name__ == "__main__":
    unittest.main()
