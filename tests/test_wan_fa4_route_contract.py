"""Dependency-free contracts for the legacy Wan FA4 route."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WAN_RUNNERS = (
    "examples/reproduce_wan2.1_t2v.py",
    "examples/reproduce_wan2.2_t2v.py",
)


def _tree(relative_path: str) -> ast.Module:
    path = ROOT / relative_path
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imported_modules(tree: ast.Module) -> tuple[str, ...]:
    return tuple(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )


def _argument_call(tree: ast.Module, flag: str) -> ast.Call:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == flag
    ]
    assert len(matches) == 1, f"expected one parser flag: {flag}"
    return matches[0]


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, f"expected one function: {name}"
    return matches[0]


def _calls_named(node: ast.AST, name: str) -> tuple[ast.Call, ...]:
    return tuple(
        candidate
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Name)
        and candidate.func.id == name
    )


def _attribute_calls_named(node: ast.AST, name: str) -> tuple[ast.Call, ...]:
    return tuple(
        candidate
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Attribute)
        and candidate.func.attr == name
    )


def _setup_extras(tree: ast.Module) -> dict[str, tuple[str, ...]]:
    setup_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    )
    extras = next(
        keyword.value
        for keyword in setup_call.keywords
        if keyword.arg == "extras_require"
    )
    assert isinstance(extras, ast.Dict)
    result: dict[str, tuple[str, ...]] = {}
    for key, value in zip(extras.keys, extras.values):
        assert isinstance(key, ast.Constant) and isinstance(key.value, str)
        assert isinstance(value, ast.List)
        result[key.value] = tuple(
            item.value
            for item in value.elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        )
    return result


def _choice_strings(call: ast.Call) -> tuple[str, ...]:
    choices = next(
        (keyword.value for keyword in call.keywords if keyword.arg == "choices"),
        None,
    )
    assert choices is not None, "attention backend choices must be explicit"
    if isinstance(choices, (ast.List, ast.Tuple)):
        values = choices.elts
    else:
        msg = "attention backend choices must be a literal list or tuple"
        raise AssertionError(msg)
    return tuple(
        element.value
        for element in values
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    )


class WanFa4RouteContractTest(unittest.TestCase):
    def test_legacy_wan_runners_expose_fa4(self) -> None:
        # Given: the Wan2.1 and Wan2.2 T2V command-line surfaces.
        trees = tuple(_tree(relative_path) for relative_path in WAN_RUNNERS)

        # When: the accepted attention implementations are read.
        choices = tuple(
            _choice_strings(_argument_call(tree, "--attn_type")) for tree in trees
        )

        # Then: both runners accept an explicit FA4 route.
        for runner_choices in choices:
            self.assertIn("fa4", runner_choices)

    def test_wan21_parallelizer_configures_fa4_before_inference(self) -> None:
        # Given: the shared parallelizer used by Wan2.1 and both Wan2.2 DiTs.
        parallelizer = _function(
            _tree("examples/wan_t2v_parallel.py"),
            "parallelize_transformer",
        )

        # When: its single-device FA4 configuration calls are inspected.
        calls = _calls_named(parallelizer, "configure_wan_fa4_single_device")

        # Then: the transformer and SP size cross the dedicated route once.
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            tuple(ast.unparse(argument) for argument in calls[0].args),
            ("transformer", "sp_size"),
        )

    def test_wan22_parallelizes_both_denoisers_with_selected_backend(self) -> None:
        # Given: Wan2.2's high-noise and low-noise DiT execution path.
        main = _function(_tree("examples/reproduce_wan2.2_t2v.py"), "main")

        # When: calls into the shared Wan parallelizer are inspected.
        calls = _calls_named(main, "parallelize_transformer")

        # Then: both DiTs receive the exact backend selected by the user.
        self.assertEqual(
            tuple(
                tuple(ast.unparse(argument) for argument in call.args)
                for call in calls
            ),
            (
                ("transformer_high", "sp_size", "sp_rank", "args.attn_type"),
                ("transformer_low", "sp_size", "sp_rank", "args.attn_type"),
            ),
        )

    def test_legacy_non_fa4_attention_route_remains_available(self) -> None:
        # Given: the shared parallelizer still serves pre-existing attention types.
        parallelizer = _function(
            _tree("examples/wan_t2v_parallel.py"),
            "parallelize_transformer",
        )

        # When: its legacy processor setup is inspected.
        processor_calls = _calls_named(parallelizer, "xDiTWanAttnProcessor")
        attribute_writes = {
            ast.unparse(target)
            for node in ast.walk(parallelizer)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Attribute)
        }

        # Then: multi-device and single-device legacy routes are both preserved.
        self.assertEqual(len(processor_calls), 1)
        self.assertEqual(ast.unparse(processor_calls[0].args[0]), "attn_type")
        self.assertIn("block.attn1.processor.attn_type", attribute_writes)

    def test_fa4_configurator_replaces_self_and_cross_processors(self) -> None:
        # Given: the FA4 configurator colocated with the shared parallelizer.
        configurator = _function(
            _tree("examples/wan_t2v_parallel.py"),
            "configure_wan_fa4_single_device",
        )

        # When: its runtime backend and processor assignments are inspected.
        main_backend = _attribute_calls_named(
            configurator,
            "set_attention_backend",
        )
        cross_backend = _attribute_calls_named(
            configurator,
            "set_cross_attention_backend",
        )
        assignments = {
            ast.unparse(target)
            for node in ast.walk(configurator)
            if isinstance(node, ast.Assign)
            for target in node.targets
        }

        # Then: both attention kinds use the same explicit core FA4 backend.
        self.assertEqual(len(main_backend), 1)
        self.assertEqual(len(cross_backend), 1)
        self.assertEqual(
            ast.unparse(main_backend[0].args[0]),
            "AttentionBackendType.FLASH_4",
        )
        self.assertEqual(
            ast.unparse(cross_backend[0].args[0]),
            "AttentionBackendType.FLASH_4",
        )
        self.assertIn("block.attn1.processor", assignments)
        self.assertIn("block.attn2.processor", assignments)

    def test_fa4_route_has_no_standalone_glue_module(self) -> None:
        # Given: every legacy runner-side module that consumes the FA4 helpers.
        consumers = (
            "examples/wan21_t2v_runner.py",
            "examples/reproduce_wan2.1_t2v.py",
            "examples/reproduce_wan2.2_t2v.py",
            "examples/wan_t2v_parallel.py",
        )

        # When: the source layout and imports are inspected.
        standalone_exists = (ROOT / "examples/wan_fa4.py").exists()
        imported_modules = tuple(_imported_modules(_tree(path)) for path in consumers)

        # Then: the shared parallelizer owns the route without a second module.
        self.assertFalse(standalone_exists)
        for modules in imported_modules:
            self.assertNotIn("wan_fa4", modules)

    def test_legacy_runners_validate_fa4_before_distributed_start(self) -> None:
        # Given: both legacy T2V entry points.
        mains = (
            _function(_tree("examples/wan21_t2v_runner.py"), "main"),
            _function(_tree("examples/reproduce_wan2.2_t2v.py"), "main"),
        )

        # When: validation and process-group initialization are located.
        for main in mains:
            validations = _calls_named(main, "validate_wan_fa4_request")
            distributed_starts = _attribute_calls_named(
                main,
                "init_process_group",
            )

            # Then: unsupported FA4 exits before distributed/model setup begins.
            self.assertEqual(len(validations), 1)
            self.assertEqual(len(distributed_starts), 1)
            self.assertLess(validations[0].lineno, distributed_starts[0].lineno)
            self.assertEqual(
                tuple(ast.unparse(argument) for argument in validations[0].args),
                (
                    "args.attn_type",
                    "args.ulysses_degree",
                    "args.ring_degree",
                ),
            )

    def test_xfuser_declares_pinned_fa4_install_extras(self) -> None:
        # Given: xDiT's optional dependency metadata.
        extras = _setup_extras(_tree("setup.py"))

        # When/Then: CUDA 12 and CUDA 13 FA4 wheels are separately selectable.
        self.assertEqual(
            extras["flash-attn-4"],
            ("flash-attn-4==4.0.0b22",),
        )
        self.assertEqual(
            extras["flash-attn-4-cu13"],
            ("flash-attn-4[cu13]==4.0.0b22",),
        )


if __name__ == "__main__":
    unittest.main()
