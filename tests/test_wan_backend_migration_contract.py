"""Dependency-free contracts for the unified Wan backend surface."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]

ATTENTION_BACKENDS = {
    "FLASHINFER",
    "SAGE_FP16",
    "SAGE_FP8",
    "SAGE_FP8_SM90",
    "SAGE_FP16_TRITON",
    "SPARSE_SAGE",
}

LINEAR_BACKENDS = {
    "BF16",
    "TORCHAO_FP8",
    "TORCHAO_NVFP4",
    "AITER_FP8_BLOCKWISE",
    "AITER_MXFP4",
    "TLLM_NVFP4",
    "TLLM_FP8_BLOCKWISE",
    "TLLM_NVFP4_FP8_BLOCKWISE",
    "TLLM_SVDQUANT_FP8_BLOCKWISE",
    "TLLM_SVDQUANT_NVFP4_FUSED",
}

LEGACY_WAN_FILES = {
    "linear_impl.py",
    "reproduce_wan2.1.py",
    "reproduce_wan2.1_t2v.py",
    "reproduce_wan2.2_t2v.py",
    "wan21_t2v_runner.py",
    "wan22_runtime.py",
    "wan_legacy_attention.py",
    "wan_runtime.py",
    "wan_t2v_parallel.py",
}


def _class_members(relative_path: str, class_name: str) -> set[str]:
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return {
        target.id
        for statement in class_node.body
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name)
    }


def _runner_argument(flag: str) -> ast.Call:
    tree = ast.parse((ROOT / "xfuser/config/args.py").read_text(encoding="utf-8"))
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
    assert len(matches) == 1, f"expected one runner argument for {flag}"
    return matches[0]


def _registered_attention_backends() -> set[str]:
    tree = ast.parse(
        (ROOT / "xfuser/core/distributed/attention_backend.py").read_text(
            encoding="utf-8"
        )
    )
    return {
        decorator.args[0].attr
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Name)
        and decorator.func.id == "register_attention_function"
        and decorator.args
        and isinstance(decorator.args[0], ast.Attribute)
    }


def _function(relative_path: str, name: str) -> ast.FunctionDef:
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, f"expected one function named {name}"
    return matches[0]


class WanBackendMigrationContractTest(unittest.TestCase):
    def test_attention_registry_contains_every_migrated_backend(self) -> None:
        # Given: the unified attention backend enum.
        members = _class_members(
            "xfuser/core/distributed/attention_backend.py",
            "AttentionBackendType",
        )

        # When/Then: every backend previously exposed by Wan remains selectable.
        self.assertTrue(ATTENTION_BACKENDS.issubset(members))

    def test_attention_registry_implements_every_migrated_backend(self) -> None:
        # Given: concrete functions registered by the unified attention runtime.
        registrations = _registered_attention_backends()

        # When/Then: no enum-only backend can reach a missing implementation.
        self.assertTrue(ATTENTION_BACKENDS.issubset(registrations))

    def test_every_wan_model_capability_accepts_sparse_attention(self) -> None:
        # Given: every Wan runner model that declares its own capabilities.
        tree = ast.parse(
            (
                ROOT / "xfuser/model_executor/models/runner_models/wan.py"
            ).read_text(encoding="utf-8")
        )
        capability_calls = [
            statement.value
            for class_node in tree.body
            if isinstance(class_node, ast.ClassDef)
            and class_node.name.startswith("xFuserWan")
            for statement in class_node.body
            if isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "capabilities"
                for target in statement.targets
            )
            and isinstance(statement.value, ast.Call)
        ]

        # When: the sparse capability flag is inspected.
        sparse_values = [
            keyword.value.value
            for call in capability_calls
            for keyword in call.keywords
            if keyword.arg == "supports_sparse_attention_backends"
            and isinstance(keyword.value, ast.Constant)
        ]

        # Then: inherited and explicit Wan variants cannot silently reject it.
        self.assertEqual(sparse_values, [True] * len(capability_calls))

    def test_linear_registry_contains_every_supported_backend(self) -> None:
        # Given: the unified linear backend enum.
        members = _class_members(
            "xfuser/model_executor/quantization/linear_backend.py",
            "LinearBackendType",
        )

        # When/Then: native and tllm_linear_lite routes share one public surface.
        self.assertTrue(LINEAR_BACKENDS.issubset(members))

    def test_tllm_nvfp4_gemm_backend_is_not_locally_whitelisted(self) -> None:
        # Given: the runner argument delegated to tllm_linear_lite.
        argument = _runner_argument("--tllm_nvfp4_gemm_backend")

        # When: argparse keywords are inspected.
        keywords = {keyword.arg for keyword in argument.keywords}

        # Then: xDiT accepts arbitrary names and lets the dependency validate them.
        self.assertNotIn("choices", keywords)

    def test_tllm_nvfp4_options_are_delegated_to_dependency(self) -> None:
        # Given: the NVFP4 construction adapter.
        builder = _function(
            "xfuser/model_executor/quantization/tllm_builders.py",
            "build_nvfp4_linear",
        )

        # When: the dependency factory call is inspected.
        factory = next(
            call
            for call in ast.walk(builder)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "from_linear"
        )
        keyword_values = {
            keyword.arg: ast.unparse(keyword.value) for keyword in factory.keywords
        }

        # Then: xDiT forwards all strings without translating their values.
        self.assertEqual(
            keyword_values,
            {
                "gemm_backend": "options.nvfp4_gemm_backend",
                "quant_backend": "options.nvfp4_quant_backend",
                "scale_rule": "options.nvfp4_scale_rule",
            },
        )

    def test_wan_image_context_uses_cross_attention_backend(self) -> None:
        # Given: Wan's added image-context attention path.
        call = _function(
            "xfuser/model_executor/models/transformers/transformer_wan.py",
            "__call__",
        )

        # When: the special image backend assignment is inspected.
        assignments = {
            target.id: ast.unparse(statement.value)
            for statement in ast.walk(call)
            if isinstance(statement, ast.Assign)
            for target in statement.targets
            if isinstance(target, ast.Name) and target.id == "image_backend"
        }

        # Then: self-attention-only kernels cannot leak into image cross-attention.
        self.assertEqual(
            assignments,
            {"image_backend": "runtime_state.get_cross_attention_backend()"},
        )

    def test_wan_attention_kwargs_include_flex_ssta_contract(self) -> None:
        # Given: the kwargs builder shared by every Wan model variant.
        builder = _function(
            "xfuser/model_executor/models/runner_models/wan.py",
            "_build_attention_kwargs",
        )
        expected_ssta_defaults = {
            "attn_mask_share_within_head": False,
            "attn_pad_type": "zero",
            "attn_sparse_type": "ssta",
            "attn_use_text_mask": False,
            "encoder_sequence_length": 0,
            "sparse_text_to_image": False,
            "ssta_adaptive_pool": None,
            "ssta_lambda": 0.7,
            "ssta_sampling_type": "importance",
            "ssta_threshold": 0.0,
            "ssta_topk": 64,
            "text_mask": None,
            "tile_size": [6, 8, 8],
            "win_size": [[3, 3, 3]],
        }

        # When: the returned mapping literals are inspected.
        returned_mapping = next(
            node.value
            for node in ast.walk(builder)
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
        )
        literal_values = {
            key.value: ast.literal_eval(value)
            for key, value in zip(returned_mapping.keys, returned_mapping.values)
            if isinstance(key, ast.Constant)
            and isinstance(key.value, str)
            and key.value in expected_ssta_defaults
        }

        # Then: Flex Block Attention receives a complete Wan SSTA configuration.
        self.assertEqual(literal_values, expected_ssta_defaults)

    def test_legacy_wan_modules_are_removed_after_migration(self) -> None:
        # Given: the old standalone Wan implementation files.
        remaining = {
            path.name
            for path in (ROOT / "examples").iterdir()
            if path.name in LEGACY_WAN_FILES
        }

        # When/Then: the unified runner is the only supported implementation.
        self.assertEqual(remaining, set())


if __name__ == "__main__":
    unittest.main()
