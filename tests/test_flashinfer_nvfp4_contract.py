"""Dependency-free integration contracts for SM12x FlashInfer NVFP4."""

from __future__ import annotations

import ast
import copy
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _tree(relative_path: str) -> ast.Module:
    path = ROOT / relative_path
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, f"expected one function: {name}"
    return matches[0]


def _calls_named(node: ast.AST, name: str) -> tuple[ast.Call, ...]:
    return tuple(
        candidate
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Call)
        and (
            isinstance(candidate.func, ast.Name) and candidate.func.id == name
            or isinstance(candidate.func, ast.Attribute) and candidate.func.attr == name
        )
    )


def _setup_extras() -> dict[str, tuple[str, ...]]:
    tree = _tree("setup.py")
    setup_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    )
    extras = next(
        keyword.value for keyword in setup_call.keywords if keyword.arg == "extras_require"
    )
    assert isinstance(extras, ast.Dict)
    result = {}
    for key, value in zip(extras.keys, extras.values):
        assert isinstance(key, ast.Constant) and isinstance(value, ast.List)
        result[key.value] = tuple(
            item.value for item in value.elts if isinstance(item, ast.Constant)
        )
    return result


def _runtime_cross_backend_setter():
    method = copy.deepcopy(
        _function(
            _tree("xfuser/core/distributed/runtime_state.py"),
            "set_cross_attention_backend",
        )
    )
    method.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            method,
        ],
        type_ignores=[],
    )

    class Backend(Enum):
        FLASHINFER_NVFP4 = "flashinfer_nvfp4"
        SPARSE_SAGE = "sparse_sage"
        SDPA_FLASH = "sdpa_flash"

    def parse_attention_backend(name: str):
        return Backend[name.upper()]

    namespace = {
        "AttentionBackendType": Backend,
        "parse_attention_backend": parse_attention_backend,
        "logger": SimpleNamespace(warning=lambda *_args, **_kwargs: None),
    }
    exec(
        compile(
            ast.fix_missing_locations(module),
            filename="runtime_state.set_cross_attention_backend",
            mode="exec",
        ),
        namespace,
    )
    return namespace["set_cross_attention_backend"], Backend


def _flashinfer_version_predicate():
    try:
        from packaging import version
    except ModuleNotFoundError:
        from pip._vendor.packaging import version

    tree = _tree("xfuser/envs.py")
    predicate = copy.deepcopy(
        _function(tree, "_has_fixed_flashinfer_nvfp4_version")
    )
    predicate.decorator_list = []
    constants = [
        copy.deepcopy(node)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id.startswith("FLASHINFER_NVFP4_")
            for target in node.targets
        )
    ]
    module = ast.Module(
        body=[
            *constants,
            predicate,
        ],
        type_ignores=[],
    )
    namespace = {"version": version}
    exec(
        compile(
            ast.fix_missing_locations(module),
            filename="envs._has_fixed_flashinfer_nvfp4_version",
            mode="exec",
        ),
        namespace,
    )
    return namespace["_has_fixed_flashinfer_nvfp4_version"]


class FlashInferNvfp4ContractTest(unittest.TestCase):
    def test_core_backend_is_distinct_and_registered(self) -> None:
        # Given: xDiT's central attention registry.
        tree = _tree("xfuser/core/distributed/attention_backend.py")
        enum = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "AttentionBackendType"
        )

        # When: its enum and registered handlers are inspected.
        enum_names = {
            target.id
            for node in enum.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        handler = _function(tree, "_flashinfer_nvfp4_attn_call")
        decorators = tuple(ast.unparse(item) for item in handler.decorator_list)

        # Then: FlashInfer NVFP4 does not alias the existing FA4 FP4 backend.
        self.assertIn("FLASHINFER_NVFP4", enum_names)
        self.assertIn("FLASH_4_FP4", enum_names)
        self.assertIn(
            "register_attention_function(AttentionBackendType.FLASHINFER_NVFP4)",
            decorators,
        )
        self.assertEqual(len(_calls_named(handler, "flashinfer_nvfp4_attention")), 1)
        kwargs_get = _calls_named(handler, "get")
        self.assertEqual(len(kwargs_get), 1)
        self.assertEqual(
            tuple(ast.literal_eval(argument) for argument in kwargs_get[0].args),
            ("flashinfer_nvfp4_per_block_mean", False),
        )

    def test_environment_and_runtime_fail_closed(self) -> None:
        # Given: environment discovery and runtime compatibility checks.
        env_source = (ROOT / "xfuser/envs.py").read_text(encoding="utf-8")
        runtime_source = (
            ROOT / "xfuser/core/distributed/runtime_state.py"
        ).read_text(encoding="utf-8")

        # When/Then: public APIs, native SM12x capabilities, and fail-closed gates exist.
        self.assertIn('packages_info["has_flashinfer_nvfp4"]', env_source)
        self.assertIn("def _check_flashinfer_nvfp4", env_source)
        self.assertIn("nvfp4_attention_sm120_quantize_qkv", env_source)
        self.assertIn("nvfp4_attention_sm120_fwd", env_source)
        self.assertIn("(12, 0)", env_source)
        self.assertIn("(12, 1)", env_source)
        self.assertIn("0.6.16", env_source)
        self.assertIn("_has_fixed_flashinfer_nvfp4_version", env_source)
        self.assertIn("AttentionBackendType.FLASHINFER_NVFP4", runtime_source)
        self.assertIn('env_info.get("has_flashinfer_nvfp4")', runtime_source)
        self.assertIn("does not support ring parallelism", runtime_source)

    def test_version_gate_rejects_diverged_stable_post_release(self) -> None:
        # Given: stable 0.6.15.post1 predates both #3838 and #3897.
        has_fixed_version = _flashinfer_version_predicate()

        # When/Then: only a fixed nightly or a future fixed release is accepted.
        self.assertFalse(has_fixed_version("0.6.15.dev20260716"))
        self.assertFalse(has_fixed_version("0.6.15"))
        self.assertFalse(has_fixed_version("0.6.15.post1"))
        self.assertTrue(has_fixed_version("0.6.15.dev20260717"))
        self.assertTrue(has_fixed_version("0.6.15.dev20260722"))
        self.assertTrue(has_fixed_version("0.6.16.dev20260701"))
        self.assertTrue(has_fixed_version("0.6.16"))

    def test_runtime_rejects_nvfp4_cross_attention_at_configuration(self) -> None:
        # Given: the generic RuntimeState cross-backend setter.
        setter, backend = _runtime_cross_backend_setter()
        compatibility_checks = []
        state = SimpleNamespace(
            attention_backend=backend.SDPA_FLASH,
            cross_attention_backend=None,
            _check_if_backend_compatible_with_current_configuration=(
                compatibility_checks.append
            ),
        )

        # When/Then: explicit NVFP4 cross-attention fails before compatibility.
        for value in (backend.FLASHINFER_NVFP4, "flashinfer_nvfp4"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "identical shapes"):
                    setter(state, value)
        self.assertEqual(compatibility_checks, [])

        # When/Then: an NVFP4 main backend may not implicitly handle cross-attention.
        state.attention_backend = backend.FLASHINFER_NVFP4
        with self.assertRaisesRegex(ValueError, "explicit non-NVFP4"):
            setter(state, None)

        # When/Then: a supported BF16 cross backend is accepted normally.
        setter(state, backend.SDPA_FLASH)
        self.assertEqual(state.cross_attention_backend, backend.SDPA_FLASH)
        self.assertEqual(compatibility_checks, [backend.SDPA_FLASH])

    def test_package_extra_pins_latest_validated_nightly(self) -> None:
        # Given/When: xDiT's optional dependency metadata is inspected.
        extras = _setup_extras()

        # Then: users receive the latest validated nightly with #3838 and #3897.
        self.assertEqual(
            extras["flashinfer-nvfp4"],
            ("flashinfer-python==0.6.15.dev20260722",),
        )

if __name__ == "__main__":
    unittest.main()
