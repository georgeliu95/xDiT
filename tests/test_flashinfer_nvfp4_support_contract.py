"""User-facing support contract for FlashInfer SM12x NVFP4."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _tree(relative_path: str) -> ast.Module:
    path = ROOT / relative_path
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


class FlashInferNvfp4SupportContractTest(unittest.TestCase):
    def test_user_facing_support_contract_covers_sm120_and_sm121(self) -> None:
        # Given: the backend label and its fail-closed diagnostic.
        backend_tree = _tree("xfuser/core/distributed/attention_backend.py")
        backend_enum = next(
            node
            for node in backend_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "AttentionBackendType"
        )
        backend_label = next(
            ast.literal_eval(node.value)
            for node in backend_enum.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "FLASHINFER_NVFP4"
                for target in node.targets
            )
        )
        runtime_tree = _tree("xfuser/core/distributed/runtime_state.py")
        unavailable_call = next(
            node
            for node in ast.walk(runtime_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "FlashInferNvfp4UnavailableError"
        )

        # When: the public support contract is inspected.
        unavailable_reason = ast.unparse(unavailable_call.args[0])

        # Then: diagnostics describe both supported SM12x capabilities and fixes.
        self.assertEqual(backend_label, "FlashInfer SM12x NVFP4")
        for required_detail in ("SM120", "SM121", "#3838", "#3897"):
            with self.subTest(required_detail=required_detail):
                self.assertIn(required_detail, unavailable_reason)


if __name__ == "__main__":
    unittest.main()
