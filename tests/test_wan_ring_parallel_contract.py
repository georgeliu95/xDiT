"""Contracts for legacy Wan Ring and Ulysses sequence parallelism."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
WAN_RUNTIME = ROOT / "examples" / "wan_runtime.py"
LEGACY_RUNNERS = (
    ROOT / "examples" / "wan21_t2v_runner.py",
    ROOT / "examples" / "reproduce_wan2.2_t2v.py",
    ROOT / "examples" / "reproduce_wan2.1.py",
)


def _load_wan_runtime(
    sequence_parallel_world_size: int,
    sequence_parallel_rank: int,
) -> tuple[ModuleType, list[dict[str, int]]]:
    initialize_calls: list[dict[str, int]] = []

    distributed = ModuleType("xfuser.core.distributed")

    def initialize_model_parallel(**degrees: int) -> None:
        initialize_calls.append(degrees)

    distributed.initialize_model_parallel = initialize_model_parallel
    distributed.get_sequence_parallel_world_size = lambda: sequence_parallel_world_size
    distributed.get_sequence_parallel_rank = lambda: sequence_parallel_rank

    torch = ModuleType("torch")
    torch.distributed = ModuleType("torch.distributed")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.float16 = "float16"
    torch.bfloat16 = "bfloat16"

    diffusers = ModuleType("diffusers")
    diffusers.AutoencoderKLWan = type("AutoencoderKLWan", (), {})
    diffusers.WanPipeline = type("WanPipeline", (), {})
    diffusers_models = ModuleType("diffusers.models")
    diffusers_transformers = ModuleType("diffusers.models.transformers")
    diffusers_wan = ModuleType("diffusers.models.transformers.transformer_wan")
    diffusers_wan.WanTransformer3DModel = type("WanTransformer3DModel", (), {})

    xfuser = ModuleType("xfuser")
    xfuser_core = ModuleType("xfuser.core")
    xfuser_logger = ModuleType("xfuser.logger")
    xfuser_logger.init_logger = lambda _name: SimpleNamespace(info=lambda *_args: None)
    psutil = ModuleType("psutil")

    modules = {
        "psutil": psutil,
        "torch": torch,
        "torch.distributed": torch.distributed,
        "diffusers": diffusers,
        "diffusers.models": diffusers_models,
        "diffusers.models.transformers": diffusers_transformers,
        "diffusers.models.transformers.transformer_wan": diffusers_wan,
        "xfuser": xfuser,
        "xfuser.core": xfuser_core,
        "xfuser.core.distributed": distributed,
        "xfuser.logger": xfuser_logger,
    }
    spec = importlib.util.spec_from_file_location("wan_runtime_contract_target", WAN_RUNTIME)
    assert spec is not None and spec.loader is not None

    runtime = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(runtime)
    return runtime, initialize_calls


class WanRingParallelRuntimeContractTest(unittest.TestCase):
    def test_ring_only_initializes_ring_group(self) -> None:
        # Given: two ranks assigned entirely to Ring parallelism.
        runtime, calls = _load_wan_runtime(2, 1)

        # When: the shared Wan sequence-parallel initializer runs.
        result = runtime.initialize_wan_sequence_parallel(
            ulysses_degree=1,
            ring_degree=2,
        )

        # Then: both degrees reach xFuser and the actual group position is returned.
        self.assertEqual(
            calls,
            [{"ulysses_degree": 1, "ring_degree": 2}],
        )
        self.assertEqual(result, (2, 1))

    def test_hybrid_parallelism_initializes_both_groups(self) -> None:
        # Given: four ranks split across Ulysses and Ring dimensions.
        runtime, calls = _load_wan_runtime(4, 3)

        # When: the shared Wan sequence-parallel initializer runs.
        result = runtime.initialize_wan_sequence_parallel(
            ulysses_degree=2,
            ring_degree=2,
        )

        # Then: xFuser receives both dimensions and reports their combined group.
        self.assertEqual(
            calls,
            [{"ulysses_degree": 2, "ring_degree": 2}],
        )
        self.assertEqual(result, (4, 3))

    def test_single_device_skips_model_parallel_initialization(self) -> None:
        # Given: both sequence-parallel dimensions are disabled.
        runtime, calls = _load_wan_runtime(1, 0)

        # When: the shared Wan sequence-parallel initializer runs.
        result = runtime.initialize_wan_sequence_parallel(
            ulysses_degree=1,
            ring_degree=1,
        )

        # Then: legacy single-device behavior is preserved.
        self.assertEqual(calls, [])
        self.assertEqual(result, (1, 0))


class WanRingParallelRunnerContractTest(unittest.TestCase):
    def test_all_legacy_runners_delegate_both_degrees(self) -> None:
        for path in LEGACY_RUNNERS:
            with self.subTest(path=path.name):
                # Given: a legacy Wan runner's main function.
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                main = next(
                    node
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "main"
                )

                # When: sequence-parallel initialization calls are inspected.
                shared_calls = [
                    node
                    for node in ast.walk(main)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "initialize_wan_sequence_parallel"
                ]
                direct_calls = [
                    node
                    for node in ast.walk(main)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "initialize_model_parallel"
                ]

                # Then: one shared call forwards Ring and Ulysses without a stale path.
                self.assertEqual(len(shared_calls), 1)
                self.assertEqual(
                    {
                        keyword.arg: ast.unparse(keyword.value)
                        for keyword in shared_calls[0].keywords
                    },
                    {
                        "ulysses_degree": "args.ulysses_degree",
                        "ring_degree": "args.ring_degree",
                    },
                )
                self.assertEqual(direct_calls, [])


if __name__ == "__main__":
    unittest.main()
