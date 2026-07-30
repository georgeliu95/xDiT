"""Model and pipeline helpers for the legacy Wan2.2 T2V runner."""

from __future__ import annotations

import gc
import inspect

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

from xfuser.logger import init_logger

from linear_impl import replace_linear_layer
from wan_runtime import print_memory_usage


logger = init_logger(__name__)


class Wan22PipelineSupportError(RuntimeError):
    """Raised when the installed Diffusers build cannot load Wan2.2 MoE DiTs."""


def check_wan22_pipeline_support() -> None:
    signature = inspect.signature(WanPipeline.__init__)
    if "transformer_2" not in signature.parameters:
        raise Wan22PipelineSupportError(
            "Wan2.2 Diffusers checkpoints require WanPipeline transformer_2 "
            "support. Install a newer diffusers build before running this example."
        )


def load_transformer(model_id: str, subfolder: str) -> WanTransformer3DModel:
    return WanTransformer3DModel.from_pretrained(
        model_id,
        subfolder=subfolder,
        torch_dtype=torch.bfloat16,
    ).to(torch.bfloat16)


def apply_quantization(
    transformer: WanTransformer3DModel,
    label: str,
    args,
) -> WanTransformer3DModel:
    if args.quant_gemm_type in (None, "bf16"):
        return transformer

    logger.info(
        f"Applying {args.quant_gemm_type} quantization to Wan2.2 {label} transformer"
    )
    return replace_linear_layer(
        transformer,
        quant_gemm_type=args.quant_gemm_type,
        nvfp4_gemm_backend=args.nvfp4_gemm_backend,
        nvfp4_scale_rule=args.nvfp4_scale_rule,
    )


def predownload_components(model_id: str, global_rank: int) -> None:
    if global_rank != 0:
        return

    logger.info("Rank 0: Pre-downloading Wan2.2 components to cache...")
    temp_model = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float16,
    )
    del temp_model
    gc.collect()

    for subfolder in ("transformer", "transformer_2"):
        temp_model = load_transformer(model_id, subfolder)
        del temp_model
        gc.collect()

    temp_pipe = WanPipeline.from_pretrained(
        model_id,
        transformer=None,
        transformer_2=None,
        vae=None,
        torch_dtype=torch.bfloat16,
    )
    del temp_pipe
    gc.collect()
    logger.info("Rank 0: Wan2.2 components downloaded and released")
    print_memory_usage("After pre-download", global_rank)


def load_models(model_id: str, global_rank: int):
    logger.info(f"Rank {global_rank}: Loading Wan2.2 VAE and transformers...")
    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float16,
    )
    transformer_high = load_transformer(model_id, "transformer")
    transformer_low = load_transformer(model_id, "transformer_2")
    logger.info(f"Rank {global_rank}: Wan2.2 models loaded successfully")
    print_memory_usage(f"After loading models (Rank {global_rank})", global_rank)
    return vae, transformer_high, transformer_low


def set_boundary_ratio(pipe: WanPipeline, boundary_ratio: float | None) -> None:
    current = getattr(pipe.config, "boundary_ratio", None)
    if boundary_ratio is None and current is not None:
        return

    resolved = 0.875 if boundary_ratio is None else boundary_ratio
    pipe.register_to_config(boundary_ratio=resolved)
    logger.info(f"Wan2.2 boundary_ratio={resolved}")


def build_pipeline(
    model_id: str,
    vae: AutoencoderKLWan,
    transformer_high: WanTransformer3DModel,
    transformer_low: WanTransformer3DModel,
    device: str,
    boundary_ratio: float | None,
) -> WanPipeline:
    # The pipeline is constructed without either expert, so pipe.to(device)
    # cannot move them.  Complete resident placement before attaching the
    # experts; otherwise skipped BF16 modules such as patch_embedding remain
    # on CPU while the latent input is on CUDA.
    transformer_high = transformer_high.to(device)
    transformer_low = transformer_low.to(device)
    pipe = WanPipeline.from_pretrained(
        pretrained_model_name_or_path=model_id,
        vae=vae.to(device),
        transformer=None,
        transformer_2=None,
        torch_dtype=torch.bfloat16,
    )
    pipe.transformer = None
    pipe.transformer_2 = None
    pipe.to(device)
    pipe.transformer = transformer_high
    pipe.transformer_2 = transformer_low
    set_boundary_ratio(pipe, boundary_ratio)
    return pipe
