"""Executable unit tests for the legacy Wan FA4 validation and wiring."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class _FakeRuntime:
    def __init__(self) -> None:
        self.main_backend = None
        self.cross_backend = None

    def set_attention_backend(self, backend) -> None:
        self.main_backend = backend

    def set_cross_attention_backend(self, backend) -> None:
        self.cross_backend = backend


class _FakeProcessor:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


def _empty_module(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    return module


def _load_module(
    *,
    package_available: bool,
    cuda_available: bool,
    capability,
    flashinfer_nvfp4_available: bool = False,
    sage_available: bool = False,
):
    runtime = _FakeRuntime()
    state = {"initialized": False}

    torch = ModuleType("torch")
    torch.__path__ = []
    torch.cuda = SimpleNamespace(
        is_available=lambda: cuda_available,
        get_device_capability=lambda: capability,
    )
    torch_distributed = ModuleType("torch.distributed")
    torch_distributed.is_initialized = lambda: False
    torch.distributed = torch_distributed

    nvtx = ModuleType("nvtx")

    modeling_outputs = ModuleType("diffusers.models.modeling_outputs")
    modeling_outputs.Transformer2DModelOutput = type(
        "Transformer2DModelOutput",
        (),
        {},
    )

    transformer_wan = ModuleType("diffusers.models.transformers.transformer_wan")
    transformer_wan.WanTransformer3DModel = type("WanTransformer3DModel", (), {})

    diffusers_utils = ModuleType("diffusers.utils")
    diffusers_utils.USE_PEFT_BACKEND = False
    diffusers_utils.scale_lora_layers = lambda *args, **kwargs: None
    diffusers_utils.unscale_lora_layers = lambda *args, **kwargs: None

    distributed = ModuleType("xfuser.core.distributed")
    distributed.runtime_state_is_initialized = lambda: state["initialized"]
    distributed.get_sp_group = lambda: None

    def initialize_runtime_state() -> None:
        state["initialized"] = True

    distributed.initialize_runtime_state = initialize_runtime_state
    distributed.get_runtime_state = lambda: runtime

    backend_module = ModuleType("xfuser.core.distributed.attention_backend")
    backend_module.AttentionBackendType = SimpleNamespace(
        FLASH_4="flash_4",
        FLASHINFER_NVFP4="flashinfer_nvfp4",
        SAGE="sage",
        SDPA_FLASH="sdpa_flash",
    )

    envs = ModuleType("xfuser.envs")
    envs.PACKAGES_CHECKER = SimpleNamespace(
        get_packages_info=lambda: {
            "has_flash_attn_4": package_available,
            "has_flashinfer_nvfp4": flashinfer_nvfp4_available,
            "has_sage": sage_available,
        }
    )

    xfuser_wan = ModuleType("xfuser.model_executor.models.transformers.transformer_wan")
    xfuser_wan.xFuserWanAttnProcessor = _FakeProcessor

    logger_module = ModuleType("xfuser.logger")
    logger_module.init_logger = lambda name: SimpleNamespace(
        warning=lambda *args, **kwargs: None
    )

    legacy_attention = ModuleType("wan_legacy_attention")
    legacy_attention.xDiTWanAttnProcessor = _FakeProcessor

    modules = {
        "torch": torch,
        "torch.distributed": torch_distributed,
        "nvtx": nvtx,
        "diffusers": _empty_module("diffusers"),
        "diffusers.models": _empty_module("diffusers.models"),
        "diffusers.models.modeling_outputs": modeling_outputs,
        "diffusers.models.transformers": _empty_module("diffusers.models.transformers"),
        "diffusers.models.transformers.transformer_wan": transformer_wan,
        "diffusers.utils": diffusers_utils,
        "xfuser": _empty_module("xfuser"),
        "xfuser.core": _empty_module("xfuser.core"),
        "xfuser.core.distributed": distributed,
        "xfuser.core.distributed.attention_backend": backend_module,
        "xfuser.envs": envs,
        "xfuser.logger": logger_module,
        "xfuser.model_executor": _empty_module("xfuser.model_executor"),
        "xfuser.model_executor.models": _empty_module("xfuser.model_executor.models"),
        "xfuser.model_executor.models.transformers": _empty_module(
            "xfuser.model_executor.models.transformers"
        ),
        "xfuser.model_executor.models.transformers.transformer_wan": xfuser_wan,
        "wan_legacy_attention": legacy_attention,
    }
    module_name = "_wan_t2v_parallel_under_test"
    spec = importlib.util.spec_from_file_location(
        module_name,
        ROOT / "examples/wan_t2v_parallel.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
    return module, runtime, state


class WanFa4ValidationTest(unittest.TestCase):
    def test_non_fa4_route_skips_fa4_runtime_checks(self) -> None:
        module, _, _ = _load_module(
            package_available=False,
            cuda_available=False,
            capability=None,
        )

        module.validate_wan_fa4_request("fa", 2, 2)

    def test_rejects_sequence_parallelism_before_runtime_checks(self) -> None:
        module, _, _ = _load_module(
            package_available=False,
            cuda_available=False,
            capability=None,
        )
        with self.assertRaises(module.WanFa4ParallelismError):
            module.validate_wan_fa4_request("fa4", 2, 1)

    def test_rejects_multi_process_launch(self) -> None:
        module, _, _ = _load_module(
            package_available=True,
            cuda_available=True,
            capability=(12, 0),
        )
        with patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            with self.assertRaises(module.WanFa4ParallelismError):
                module.validate_wan_fa4_request("fa4", 1, 1)

    def test_rejects_missing_sm120_or_package(self) -> None:
        cases = (
            (False, True, (12, 0)),
            (True, True, (10, 0)),
            (True, False, None),
        )
        for package_available, cuda_available, capability in cases:
            with self.subTest(
                package_available=package_available,
                capability=capability,
            ):
                module, _, _ = _load_module(
                    package_available=package_available,
                    cuda_available=cuda_available,
                    capability=capability,
                )
                with self.assertRaises(module.WanFa4RuntimeError):
                    module.validate_wan_fa4_request("fa4", 1, 1)

    def test_configures_self_and_cross_attention_processors(self) -> None:
        module, runtime, state = _load_module(
            package_available=True,
            cuda_available=True,
            capability=(12, 0),
        )
        block = SimpleNamespace(
            attn1=SimpleNamespace(processor=None),
            attn2=SimpleNamespace(processor=None),
        )
        transformer = SimpleNamespace(blocks=[block])

        module.configure_wan_fa4_single_device(transformer, 1)

        self.assertTrue(state["initialized"])
        self.assertEqual(runtime.main_backend, "flash_4")
        self.assertEqual(runtime.cross_backend, "flash_4")
        self.assertEqual(
            block.attn1.processor.kwargs,
            {"use_ulysses_parallel_attention": False},
        )
        self.assertEqual(
            block.attn2.processor.kwargs,
            {
                "use_ulysses_parallel_attention": False,
                "is_cross_attention": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
