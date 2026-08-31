import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
WAN_RUNNER = ROOT / "xfuser/model_executor/models/runner_models/wan.py"


def module(name: str, **attributes: object) -> ModuleType:
    fake = ModuleType(name)
    fake.__dict__.update(attributes)
    if "." not in name:
        fake.__path__ = []
    return fake


class ConfigValue:
    def __init__(self, **values: object) -> None:
        self.__dict__.update(values)


class Scheduler:
    pass


class Model:
    pass


class Tensor:
    pass


class RecordingTransformer:
    calls: list[dict[str, object]] = []

    @classmethod
    def from_pretrained(cls, **kwargs: object) -> object:
        cls.calls.append(kwargs)
        return object()


class RecordingPipeline:
    calls: list[dict[str, object]] = []

    @classmethod
    def from_pretrained(cls, **kwargs: object) -> object:
        cls.calls.append(kwargs)
        return object()


def register_model(_name: str):
    def decorate(model_class: type[object]) -> type[object]:
        return model_class

    return decorate


def noop(*_args: object, **_kwargs: object) -> None:
    pass


def fake_modules() -> dict[str, ModuleType]:
    torch = module(
        "torch",
        Tensor=Tensor,
        bfloat16=object(),
        nn=SimpleNamespace(Module=Model),
    )
    pil_image = module("PIL.Image", Image=type("Image", (), {}))
    diffusers = module(
        "diffusers",
        FlowMatchEulerDiscreteScheduler=Scheduler,
        WanPipeline=RecordingPipeline,
        AutoencoderKLWan=Model,
        WanVACEPipeline=RecordingPipeline,
    )
    base_model = module(
        "xfuser.model_executor.models.runner_models.base_model",
        ModelSettings=ConfigValue,
        xFuserModel=Model,
        register_model=register_model,
        ModelCapabilities=ConfigValue,
        DefaultInputValues=ConfigValue,
        DiffusionOutput=ConfigValue,
    )
    modules = {
        "torch": torch,
        "PIL": module("PIL"),
        "PIL.Image": pil_image,
        "diffusers": diffusers,
        "diffusers.pipelines": module("diffusers.pipelines"),
        "diffusers.pipelines.pipeline_utils": module(
            "diffusers.pipelines.pipeline_utils", DiffusionPipeline=Model
        ),
        "diffusers.utils": module("diffusers.utils", load_image=noop),
        "safetensors": module("safetensors"),
        "safetensors.torch": module("safetensors.torch", load_file=noop),
        "xfuser": module("xfuser", xFuserArgs=ConfigValue),
        "xfuser.model_executor": module("xfuser.model_executor"),
        "xfuser.model_executor.models": module("xfuser.model_executor.models"),
        "xfuser.model_executor.models.transformers": module(
            "xfuser.model_executor.models.transformers"
        ),
        "xfuser.model_executor.models.transformers.transformer_wan": module(
            "xfuser.model_executor.models.transformers.transformer_wan",
            xFuserWanTransformer3DWrapper=RecordingTransformer,
        ),
        "xfuser.model_executor.models.transformers.transformer_wan_vace": module(
            "xfuser.model_executor.models.transformers.transformer_wan_vace",
            xFuserWanVACETransformer3DWrapper=RecordingTransformer,
        ),
        "xfuser.model_executor.pipelines": module("xfuser.model_executor.pipelines"),
        "xfuser.model_executor.pipelines.pipeline_wan_i2v": module(
            "xfuser.model_executor.pipelines.pipeline_wan_i2v",
            xFuserWanImageToVideoPipeline=RecordingPipeline,
        ),
        "xfuser.model_executor.models.runner_models": module(
            "xfuser.model_executor.models.runner_models"
        ),
        "xfuser.model_executor.models.runner_models.base_model": base_model,
        "xfuser.core": module("xfuser.core"),
        "xfuser.core.distributed": module("xfuser.core.distributed"),
        "xfuser.core.distributed.runtime_state": module(
            "xfuser.core.distributed.runtime_state", get_runtime_state=noop
        ),
        "xfuser.core.distributed.parallel_state": module(
            "xfuser.core.distributed.parallel_state", get_vae_parallel_group=noop
        ),
        "xfuser.core.utils": module("xfuser.core.utils"),
        "xfuser.core.utils.runner_utils": module(
            "xfuser.core.utils.runner_utils",
            log=noop,
            resize_and_crop_image=noop,
            resize_image_to_max_area=noop,
        ),
        "xfuser.envs": module("xfuser.envs", PACKAGES_CHECKER=object()),
    }
    return modules


def load_wan_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_wan_contract_under_test", WAN_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Wan runner has no importable module specification")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


class Wan22RequestContractTests(unittest.TestCase):
    def test_t2v_loader_uses_the_recorded_weights_locator(self) -> None:
        # Given
        RecordingTransformer.calls = []
        RecordingPipeline.calls = []
        with patch.dict(sys.modules, fake_modules()):
            loaded = load_wan_runner()
        runner = loaded.xFuserWan22T2VModel.__new__(loaded.xFuserWan22T2VModel)
        runner.settings = SimpleNamespace(model_name="Wan-AI/registry-id")
        runner.config = SimpleNamespace(
            weights_locator="/bound/wan-checkpoint",
            spargeattn_simthreshold=0.0,
            spargeattn_cdfthreshold=0.0,
            spargeattn_reorder_sequence=False,
            use_spargeattn_static_block_mask=False,
        )

        # When
        runner._load_model()

        # Then
        self.assertEqual(2, len(RecordingTransformer.calls))
        self.assertEqual(1, len(RecordingPipeline.calls))
        self.assertEqual(
            ["transformer", "transformer_2"],
            [call["subfolder"] for call in RecordingTransformer.calls],
        )
        for call in [*RecordingTransformer.calls, *RecordingPipeline.calls]:
            self.assertEqual(
                "/bound/wan-checkpoint",
                call["pretrained_model_name_or_path"],
            )


if __name__ == "__main__":
    unittest.main()
