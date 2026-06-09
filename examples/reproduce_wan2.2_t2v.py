import argparse
import gc
import importlib.util
import inspect
import os
import sys
import time
from datetime import timedelta
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.utils import export_to_video
from torch.distributed.elastic.multiprocessing.errors import record
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    init_distributed_environment,
    initialize_model_parallel,
)

from fsdp import shard_model
from linear_impl import replace_linear_layer


_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))


def _load_wan21_t2v_module():
    """Reuse the Wan2.1 T2V xDiT attention and RoPE compatibility helpers."""
    module_path = _EXAMPLES_DIR / "reproduce_wan2.1_t2v.py"
    spec = importlib.util.spec_from_file_location("reproduce_wan21_t2v", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


wan21_t2v = _load_wan21_t2v_module()
logger = wan21_t2v.logger
get_system_memory_info = wan21_t2v.get_system_memory_info
print_memory_usage = wan21_t2v.print_memory_usage
parallelize_transformer = wan21_t2v.parallelize_transformer


def _check_wan22_pipeline_support() -> None:
    signature = inspect.signature(WanPipeline.__init__)
    if "transformer_2" not in signature.parameters:
        raise RuntimeError(
            "Wan2.2 Diffusers checkpoints require WanPipeline transformer_2 "
            "support. Install a newer diffusers build before running this "
            "example."
        )


def _load_transformer(model_id: str, subfolder: str) -> WanTransformer3DModel:
    return WanTransformer3DModel.from_pretrained(
        model_id,
        subfolder=subfolder,
        torch_dtype=torch.bfloat16,
    ).to(torch.bfloat16)


def _apply_quantization(
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


def _predownload_components(model_id: str, global_rank: int) -> None:
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
        temp_model = _load_transformer(model_id, subfolder)
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


def _load_models(model_id: str, global_rank: int):
    logger.info(f"Rank {global_rank}: Loading Wan2.2 VAE and transformers...")
    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float16,
    )
    transformer_high = _load_transformer(model_id, "transformer")
    transformer_low = _load_transformer(model_id, "transformer_2")
    logger.info(f"Rank {global_rank}: Wan2.2 models loaded successfully")
    print_memory_usage(f"After loading models (Rank {global_rank})", global_rank)
    return vae, transformer_high, transformer_low


def _set_boundary_ratio(pipe: WanPipeline, boundary_ratio: float | None) -> None:
    current = getattr(pipe.config, "boundary_ratio", None)
    if boundary_ratio is None and current is not None:
        return

    resolved = 0.875 if boundary_ratio is None else boundary_ratio
    pipe.register_to_config(boundary_ratio=resolved)
    logger.info(f"Wan2.2 boundary_ratio={resolved}")


def _build_pipeline(
    model_id: str,
    vae: AutoencoderKLWan,
    transformer_high: WanTransformer3DModel,
    transformer_low: WanTransformer3DModel,
    device: str,
    boundary_ratio: float | None,
) -> WanPipeline:
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
    _set_boundary_ratio(pipe, boundary_ratio)
    return pipe


@record
def main(args):
    _check_wan22_pipeline_support()
    dist.init_process_group("nccl", timeout=timedelta(seconds=3600))
    init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
    global_rank = dist.get_rank()
    local_rank = global_rank % torch.cuda.device_count()
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)

    if args.ulysses_degree > 1:
        initialize_model_parallel(
            sequence_parallel_degree=args.ulysses_degree,
            ulysses_degree=args.ulysses_degree,
        )
        sp_size = get_sequence_parallel_world_size()
        sp_rank = get_sequence_parallel_rank()
    else:
        sp_size = 1
        sp_rank = 0

    model_id = args.model_id
    shard_fn = partial(shard_model, device_id=local_rank)

    _predownload_components(model_id, global_rank)
    dist.barrier()

    if get_system_memory_info("available") < args.stagger_load_threshold_gb:
        vae = transformer_high = transformer_low = None
        for rank in range(dist.get_world_size()):
            if rank == global_rank:
                vae, transformer_high, transformer_low = _load_models(model_id, global_rank)
            dist.barrier()
    else:
        vae, transformer_high, transformer_low = _load_models(model_id, global_rank)

    assert args.quant_gemm_type in [
        None,
        "bf16",
        "nvfp4",
        "trtllm-fp8-blockwise",
        "nvfp4+trtllm-fp8-blockwise",
    ], "Invalid quant_gemm_type"

    transformer_high = _apply_quantization(transformer_high, "high-noise", args)
    torch.cuda.empty_cache()
    transformer_low = _apply_quantization(transformer_low, "low-noise", args)
    torch.cuda.empty_cache()

    parallelize_transformer(transformer_high, sp_size, sp_rank, args.attn_type)
    parallelize_transformer(transformer_low, sp_size, sp_rank, args.attn_type)

    if args.ulysses_degree > 1 or args.ring_degree > 1:
        transformer_high = shard_fn(transformer_high.to(device))
        transformer_low = shard_fn(transformer_low.to(device))
    else:
        transformer_high = transformer_high.to(device)
        transformer_low = transformer_low.to(device)

    torch.cuda.empty_cache()
    dist.barrier()

    pipe = _build_pipeline(
        model_id,
        vae,
        transformer_high,
        transformer_low,
        device,
        args.boundary_ratio,
    )

    default_prompt = (
        "Two anthropomorphic cats in comfy boxing gear and bright gloves fight "
        "intensely on a spotlighted stage."
    )
    prompt = args.prompt or default_prompt
    negative_prompt = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, "
        "works, paintings, images, static, overall gray, worst quality, low "
        "quality, JPEG compression residue, ugly, incomplete, extra fingers, "
        "poorly drawn hands, poorly drawn faces, deformed, disfigured, "
        "misshapen limbs, fused fingers, still picture, messy background, "
        "three legs, many people in the background, walking backwards"
    )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()

    if local_rank == 0 and args.print_transformers:
        print("=== high-noise transformer ===")
        print(pipe.transformer)
        print("=== low-noise transformer ===")
        print(pipe.transformer_2)

    pipe_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "generator": torch.Generator(device="cuda").manual_seed(args.seed),
    }
    if args.guidance_scale_2 is not None:
        pipe_kwargs["guidance_scale_2"] = args.guidance_scale_2

    with torch.no_grad():
        output = pipe(**pipe_kwargs).frames[0]

    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    memory_peak = torch.cuda.max_memory_allocated(device)
    print(f"Memory peak: {memory_peak / 1024**3:.2f} GB")

    if local_rank == 0 and not args.skip_saving_output:
        output_filename = f"xDiT.wan2.2.output.sp{sp_size}.{args.attn_type}"
        if args.quant_gemm_type is not None:
            output_filename += f".{args.quant_gemm_type}"
        if args.quant_gemm_type == "nvfp4":
            output_filename += f".{args.nvfp4_scale_rule}.{args.nvfp4_gemm_backend}"
        output_path = args.output_path or f"{output_filename}.mp4"
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        export_to_video(output, output_path, fps=args.fps)
        print(f"epoch time: {elapsed_time:.2f} sec; export video to {output_path}")

    torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    parser.add_argument("--ulysses_degree", type=int, default=1)
    parser.add_argument("--ring_degree", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument(
        "--quant_gemm_type",
        type=str,
        default=None,
        choices=[
            "bf16",
            "nvfp4",
            "trtllm-fp8-blockwise",
            "nvfp4+trtllm-fp8-blockwise",
        ],
        help="Available choices: bf16, nvfp4, trtllm-fp8-blockwise, nvfp4+trtllm-fp8-blockwise.",
    )
    parser.add_argument(
        "--nvfp4_gemm_backend",
        type=str,
        default="cublaslt",
        choices=["auto", "cutlass", "cublaslt"],
        help="GEMM backend for quant_gemm_type=nvfp4.",
    )
    parser.add_argument(
        "--nvfp4_scale_rule",
        type=str,
        default="static_6",
        choices=["static_6", "mse", "mae", "abs_max"],
        help="'static_6' is standard NVFP4; mse/mae/abs_max enable adaptive 4/6.",
    )
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--guidance_scale_2", type=float, default=None)
    parser.add_argument("--boundary_ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument(
        "--attn_type",
        type=str,
        default="fa",
        help="Available choices: fa, fa3, flashinfer, sage_fp16, sage_fp8, sage_fp8_sm90, sage_fp16_triton, sage_auto, sparse_sage.",
    )
    parser.add_argument("--skip_saving_output", action="store_true")
    parser.add_argument("--print_transformers", action="store_true")
    parser.add_argument("--stagger_load_threshold_gb", type=float, default=500.0)
    args = parser.parse_args()

    main(args)
