import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class QwenRequestContractTests(unittest.TestCase):
    def test_runner_args_preserve_weights_locator(self) -> None:
        source = (ROOT / "xfuser/config/args.py").read_text(encoding="utf-8")
        module = ast.parse(source)
        xfuser_args = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "xFuserArgs"
        )
        fields = {
            node.target.id
            for node in xfuser_args.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        self.assertIn("weights_locator", fields)

    def test_qwen_loads_bound_weights_and_forwards_sequence_length(self) -> None:
        source = (
            ROOT
            / "xfuser/model_executor/models/runner_models/qwen.py"
        ).read_text(encoding="utf-8")
        module = ast.parse(source)
        classes = [
            node
            for node in module.body
            if isinstance(node, ast.ClassDef)
            and node.name in {"xFuserQwenImageEditModel", "xFuserQwenImageModel"}
        ]
        self.assertEqual(2, len(classes))
        for model in classes:
            rendered = ast.unparse(model)
            self.assertIn(
                "weights_source = self.config.weights_locator or self.settings.model_name",
                rendered,
            )
            self.assertGreaterEqual(rendered.count("weights_source"), 3)
            self.assertIn(
                "'max_sequence_length': input_args['max_sequence_length']",
                rendered,
            )


if __name__ == "__main__":
    unittest.main()
