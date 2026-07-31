"""Packaging contracts for the supported Wan implementations."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _install_requirements() -> tuple[str, ...]:
    tree = ast.parse(
        (ROOT / "setup.py").read_text(encoding="utf-8"),
        filename="setup.py",
    )
    setup_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    )
    install_requires = next(
        keyword.value
        for keyword in setup_call.keywords
        if keyword.arg == "install_requires"
    )
    assert isinstance(install_requires, (ast.List, ast.Tuple))
    return tuple(
        element.value
        for element in install_requires.elts
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    )


class WanDependencyContractTest(unittest.TestCase):
    def test_diffusers_minimum_supports_current_wan_processor(self) -> None:
        requirements = _install_requirements()

        self.assertIn("diffusers>=0.35.2", requirements)
        self.assertNotIn("diffusers>=0.33.0", requirements)


if __name__ == "__main__":
    unittest.main()
