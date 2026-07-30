"""Unit contracts for the FlashInfer SM120 NVFP4 adapter."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "xfuser/core/distributed/flashinfer_nvfp4.py"


class _FakeTensor:
    def __init__(self, shape, dtype, *, contiguous=True, fill=None) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype
        self.ndim = len(self.shape)
        self._contiguous = contiguous
        self.fill = fill

    def contiguous(self):
        return _FakeTensor(self.shape, self.dtype, contiguous=True, fill=self.fill)

    def is_contiguous(self) -> bool:
        return self._contiguous

    def __getitem__(self, key):
        shape = list(self.shape)
        sequence_slice = key[-2] if len(key) == 3 else key[-1]
        shape[-2 if len(key) == 3 else -1] = sequence_slice.stop
        return _FakeTensor(shape, self.dtype, fill=self.fill)


def _load_adapter():
    torch = ModuleType("torch")
    torch.Tensor = _FakeTensor
    torch.float16 = "float16"
    torch.bfloat16 = "bfloat16"
    torch.float32 = "float32"
    spec = importlib.util.spec_from_file_location(
        "_flashinfer_nvfp4_under_test",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"torch": torch}):
        spec.loader.exec_module(module)
    return module


class FlashInferNvfp4AdapterTest(unittest.TestCase):
    def test_quantizes_runs_and_trims_flashinfer_padding(self) -> None:
        # Given: non-contiguous BF16 QKV with a sequence length not divisible by 128.
        module = _load_adapter()
        query = _FakeTensor((1, 2, 129, 64), module.torch.bfloat16, contiguous=False)
        key = _FakeTensor((1, 2, 129, 64), module.torch.bfloat16, contiguous=False)
        value = _FakeTensor((1, 2, 129, 64), module.torch.bfloat16, contiguous=False)
        packed = tuple(object() for _ in range(7))
        calls = {}
        flashinfer = ModuleType("flashinfer")

        def quantize(q, k, v, *, per_block_mean):
            calls["quantize"] = (q, k, v, per_block_mean)
            return packed

        def forward(*args, **kwargs):
            calls["forward"] = (args, kwargs)
            output = _FakeTensor((1, 2, 256, 64), query.dtype, fill=7)
            lse = _FakeTensor((1, 2, 256), "float32", fill=11)
            return output, lse

        flashinfer.nvfp4_attention_sm120_quantize_qkv = quantize
        flashinfer.nvfp4_attention_sm120_fwd = forward

        # When: the adapter invokes FlashInfer's quantize-and-forward path.
        with patch.dict(sys.modules, {"flashinfer": flashinfer}):
            output, lse = module.flashinfer_nvfp4_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=True,
            )

        # Then: inputs are contiguous, flags/dtype are forwarded, and padding is removed.
        quantized_qkv = calls["quantize"]
        self.assertTrue(all(tensor.is_contiguous() for tensor in quantized_qkv[:3]))
        self.assertFalse(quantized_qkv[3])
        forward_args, forward_kwargs = calls["forward"]
        self.assertEqual(forward_args, packed)
        self.assertEqual(
            forward_kwargs,
            {
                "causal": True,
                "per_block_mean": False,
                "out_dtype": module.torch.bfloat16,
            },
        )
        self.assertEqual(output.shape, query.shape)
        self.assertEqual(lse.shape, query.shape[:-1])
        self.assertEqual(output.fill, 7)
        self.assertEqual(lse.fill, 11)

    def test_rejects_dropout_before_importing_flashinfer(self) -> None:
        # Given: valid same-shape QKV but unsupported dropout.
        module = _load_adapter()
        qkv = _FakeTensor((1, 1, 128, 64), module.torch.float16)

        # When/Then: the adapter fails at the xDiT boundary with a typed error.
        with self.assertRaisesRegex(module.FlashInferNvfp4InputError, "dropout"):
            module.flashinfer_nvfp4_attention(
                qkv,
                qkv,
                qkv,
                dropout_p=0.1,
                is_causal=False,
            )

    def test_rejects_cross_attention_shapes(self) -> None:
        # Given: cross-attention Q and KV sequence lengths differ.
        module = _load_adapter()
        query = _FakeTensor((1, 1, 128, 64), module.torch.float16)
        key_value = _FakeTensor((1, 1, 64, 64), module.torch.float16)

        # When/Then: the dense SM120 API is not silently used for cross-attention.
        with self.assertRaisesRegex(module.FlashInferNvfp4InputError, "same shape"):
            module.flashinfer_nvfp4_attention(
                query,
                key_value,
                key_value,
                dropout_p=0.0,
                is_causal=False,
            )

    def test_rejects_unsupported_head_dimension(self) -> None:
        # Given: QKV use a head dimension outside FlashInfer's 64/128 contract.
        module = _load_adapter()
        qkv = _FakeTensor((1, 1, 128, 96), module.torch.float16)

        # When/Then: the adapter reports the unsupported dimension explicitly.
        with self.assertRaisesRegex(module.FlashInferNvfp4InputError, "64 or 128"):
            module.flashinfer_nvfp4_attention(
                qkv,
                qkv,
                qkv,
                dropout_p=0.0,
                is_causal=False,
            )


if __name__ == "__main__":
    unittest.main()
