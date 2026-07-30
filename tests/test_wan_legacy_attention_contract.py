"""Dependency-free contract for the legacy Wan attention processor base class."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class WanLegacyAttentionContractTest(unittest.TestCase):
    def test_processor_uses_current_diffusers_base_class(self) -> None:
        tree = ast.parse(
            (ROOT / "examples/wan_legacy_attention.py").read_text(encoding="utf-8")
        )
        processor = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "xDiTWanAttnProcessor"
        )
        imported_names = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module == "diffusers.models.transformers.transformer_wan"
            for alias in node.names
        }

        self.assertIn("WanAttnProcessor", imported_names)
        self.assertNotIn("WanAttnProcessor2_0", imported_names)
        self.assertEqual(
            tuple(ast.unparse(base) for base in processor.bases),
            ("WanAttnProcessor",),
        )

    def test_single_device_sage_uses_direct_forward_implementation(self) -> None:
        tree = ast.parse(
            (ROOT / "examples/wan_legacy_attention.py").read_text(encoding="utf-8")
        )
        processor = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "xDiTWanAttnProcessor"
        )
        init = next(
            node
            for node in processor.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        selected_functions = [
            node
            for node in ast.walk(init)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "select_flash_attn_impl"
        ]

        self.assertEqual(len(selected_functions), 1)
        self.assertIn("single_device", ast.unparse(init.args))

        run_attention = next(
            node
            for node in processor.body
            if isinstance(node, ast.FunctionDef) and node.name == "_run_attention"
        )
        direct_source = ast.unparse(run_attention)
        self.assertIn("qk_quant_gran='per_warp'", direct_source)
        self.assertIn("pv_accum_dtype='fp32+fp16'", direct_source)
        self.assertIn("return_lse=False", direct_source)


if __name__ == "__main__":
    unittest.main()
