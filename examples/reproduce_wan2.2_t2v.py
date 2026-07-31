"""Legacy Wan2.2 T2V reproduction command."""

from __future__ import annotations

import argparse
import os
import time
from datetime import timedelta
from functools import partial

import torch
import torch.distributed as dist
from diffusers.utils import export_to_video
from torch.distributed.elastic.multiprocessing.errors import record

from xfuser.core.distributed import init_distributed_environment
from xfuser.logger import init_logger

from fsdp import shard_model
from wan_runtime import get_system_memory_info, initialize_wan_sequence_parallel
from wan_t2v_parallel import parallelize_transformer, validate_wan_fa4_request
from wan22_runtime import (
    apply_quantization,
    build_pipeline,
    check_wan22_pipeline_support,
    load_models,
    predownload_components,
)


logger = init_logger(__name__)


@record
def main(args) -> None:
    check_wan22_pipeline_support()
    validate_wan_fa4_request(
        args.attn_type,
        args.ulysses_degree,
        args.ring_degree,
    )
    dist.init_process_group("nccl", timeout=timedelta(seconds=3600))
    init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
    global_rank = dist.get_rank()
    local_rank = global_rank % torch.cuda.device_count()
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)

    sp_size, sp_rank = initialize_wan_sequence_parallel(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
    )

    model_id = args.model_id
    shard_fn = partial(shard_model, device_id=local_rank)
    predownload_components(model_id, global_rank)
    dist.barrier()

    if get_system_memory_info("available") < args.stagger_load_threshold_gb:
        vae = transformer_high = transformer_low = None
        for rank in range(dist.get_world_size()):
            if rank == global_rank:
                vae, transformer_high, transformer_low = load_models(
                    model_id,
                    global_rank,
                )
            dist.barrier()
    else:
        vae, transformer_high, transformer_low = load_models(model_id, global_rank)

    assert args.quant_gemm_type in [
        None,
        "bf16",
        "nvfp4",
        "trtllm-fp8-blockwise",
        "nvfp4+trtllm-fp8-blockwise",
    ], "Invalid quant_gemm_type"

    transformer_high = apply_quantization(transformer_high, "high-noise", args)
    torch.cuda.empty_cache()
    transformer_low = apply_quantization(transformer_low, "low-noise", args)
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
    pipe = build_pipeline(
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
            output_filename += (
                f".{args.nvfp4_scale_rule}.{args.nvfp4_gemm_backend}"
            )
        output_path = args.output_path or f"{output_filename}.mp4"
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        export_to_video(output, output_path, fps=args.fps)
        print(f"epoch time: {elapsed_time:.2f} sec; export video to {output_path}")

    torch.cuda.empty_cache()
    dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    )
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
        help="Available choices: bf16, nvfp4, trtllm-fp8-blockwise, "
        "nvfp4+trtllm-fp8-blockwise.",
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
        choices=[
            "fa",
            "fa3",
            "fa4",
            "flashinfer_nvfp4",
            "flashinfer",
            "sage_fp16",
            "sage_fp8",
            "sage_fp8_sm90",
            "sage_fp16_triton",
            "sage_auto",
            "sparse_sage",
        ],
    )
    parser.add_argument("--skip_saving_output", action="store_true")
    parser.add_argument("--print_transformers", action="store_true")
    parser.add_argument("--stagger_load_threshold_gb", type=float, default=500.0)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
