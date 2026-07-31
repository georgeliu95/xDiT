"""Runtime orchestration for the legacy Wan2.1 T2V reproduction command."""

from __future__ import annotations

import os
import time
from datetime import timedelta
from functools import partial

import nvtx
import torch
import torch.distributed as dist
from diffusers import WanPipeline
from diffusers.utils import export_to_video
from torch.distributed.elastic.multiprocessing.errors import record

from xfuser.core.distributed import init_distributed_environment
from xfuser.logger import init_logger

from fsdp import shard_model
from linear_impl import replace_linear_layer
from wan_runtime import (
    get_system_memory_info,
    initialize_wan_sequence_parallel,
    load_wan21_models,
    predownload_wan21_components,
    print_memory_usage,
)
from wan_t2v_parallel import parallelize_transformer, validate_wan_fa4_request


logger = init_logger(__name__)


@record
def main(args) -> None:
    validate_wan_fa4_request(
        args.attn_type,
        args.ulysses_degree,
        args.ring_degree,
    )
    dist.init_process_group("nccl", timeout=timedelta(seconds=3600))
    init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
    global_rank = dist.get_rank()
    local_rank = global_rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    sp_size, sp_rank = initialize_wan_sequence_parallel(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
    )

    shard_fn = partial(shard_model, device_id=local_rank)
    model_id = args.model_id
    predownload_wan21_components(model_id, global_rank)
    vae, transformer = load_wan21_models(model_id, global_rank)

    if args.quant_gemm_type not in (None, "bf16"):
        if dist.is_initialized() and get_system_memory_info("available") < 500:
            for rank in range(dist.get_world_size()):
                if rank == global_rank:
                    logger.info(
                        f"Rank {global_rank}: Moving VAE to GPU before quantization..."
                    )
                    vae = vae.to(f"cuda:{local_rank}")
                    torch.cuda.empty_cache()
                    print_memory_usage(
                        f"Before quantization (Rank {global_rank})",
                        global_rank,
                    )
                dist.barrier()

    if args.quant_gemm_type is not None:
        assert args.quant_gemm_type in [
            "bf16",
            "nvfp4",
            "trtllm-fp8-blockwise",
            "nvfp4+trtllm-fp8-blockwise",
        ], "Invalid quant_gemm_type"
        if args.quant_gemm_type != "bf16":
            if dist.is_initialized() and get_system_memory_info("available") < 500:
                transformer_on_gpu = False
                for rank in range(dist.get_world_size()):
                    if rank == global_rank:
                        if transformer_on_gpu:
                            transformer = transformer.cpu()
                            torch.cuda.empty_cache()
                        print_memory_usage(
                            f"Before quantization (Rank {global_rank})",
                            global_rank,
                        )
                        transformer = replace_linear_layer(
                            transformer,
                            quant_gemm_type=args.quant_gemm_type,
                            nvfp4_gemm_backend=args.nvfp4_gemm_backend,
                            nvfp4_scale_rule=args.nvfp4_scale_rule,
                        )
                        transformer = transformer.to(f"cuda:{local_rank}")
                        torch.cuda.empty_cache()
                        transformer_on_gpu = True
                        print_memory_usage(
                            f"After quantization (Rank {global_rank})",
                            global_rank,
                        )
                    elif not transformer_on_gpu and rank == 0:
                        transformer = transformer.to(f"cuda:{local_rank}")
                        torch.cuda.synchronize()
                        transformer_on_gpu = True
                    dist.barrier()
            else:
                transformer = replace_linear_layer(
                    transformer,
                    quant_gemm_type=args.quant_gemm_type,
                    nvfp4_gemm_backend=args.nvfp4_gemm_backend,
                    nvfp4_scale_rule=args.nvfp4_scale_rule,
                )

    parallelize_transformer(transformer, sp_size, sp_rank, args.attn_type)

    if dist.is_initialized() and get_system_memory_info("available") < 500:
        if args.quant_gemm_type in (None, "bf16"):
            for rank in range(dist.get_world_size()):
                if rank == global_rank:
                    vae = vae.to(f"cuda:{local_rank}")
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    print_memory_usage(
                        f"After moving VAE to GPU (Rank {global_rank})",
                        global_rank,
                    )
                dist.barrier()

        transformer = transformer.to(f"cuda:{local_rank}")
        torch.cuda.synchronize()
        if args.ulysses_degree > 1 or args.ring_degree > 1:
            transformer = shard_fn(transformer)
        torch.cuda.empty_cache()
        print_memory_usage(
            f"After transformer to GPU (Rank {global_rank})",
            global_rank,
        )
        dist.barrier()

        for rank in range(dist.get_world_size()):
            if rank == global_rank:
                pipe = WanPipeline.from_pretrained(
                    pretrained_model_name_or_path=model_id,
                    vae=vae,
                    torch_dtype=torch.bfloat16,
                )
                pipe.transformer = transformer
                pipe.to(f"cuda:{local_rank}")
                torch.cuda.empty_cache()
                print_memory_usage(
                    f"After creating pipeline (Rank {global_rank})",
                    global_rank,
                )
            dist.barrier()
    else:
        pipe = WanPipeline.from_pretrained(
            pretrained_model_name_or_path=model_id,
            vae=vae.to(f"cuda:{local_rank}"),
            torch_dtype=torch.bfloat16,
        )
        pipe.transformer = None
        pipe.to(f"cuda:{local_rank}")
        if args.ulysses_degree > 1 or args.ring_degree > 1:
            pipe.transformer = shard_fn(transformer)
        else:
            pipe.transformer = transformer.to(f"cuda:{local_rank}")
        torch.cuda.empty_cache()

    default_prompt = (
        "Summer beach vacation style, a white cat wearing sunglasses sits on a "
        "surfboard. The fluffy-furred feline gazes directly at the camera with a "
        "relaxed expression. Blurred beach scenery forms the background featuring "
        "crystal-clear waters, distant green hills, and a blue sky dotted with white "
        "clouds. The cat assumes a naturally relaxed posture, as if savoring the sea "
        "breeze and warm sunlight. A close-up shot highlights the feline's intricate "
        "details and the soft texture of its fur. The cat's expression conveys a "
        "sense of relaxation and contentment, as it enjoys the warm sun and the gentle "
        "sea breeze."
    )
    prompt = args.prompt or default_prompt
    negative_prompt = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
        "paintings, images, static, overall gray, worst quality, low quality, JPEG "
        "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
        "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
        "still picture, messy background, three legs, many people in the background, "
        "walking backwards"
    )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()
    if local_rank == 0:
        print(pipe.transformer)

    with torch.no_grad():
        pipeline_rng = nvtx.start_range("pipeline", color="blue")
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            guidance_scale=5.0,
            num_inference_steps=args.num_inference_steps,
            generator=torch.Generator(device="cuda").manual_seed(999),
        ).frames[0]
        nvtx.end_range(pipeline_rng)

    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    memory_peak = torch.cuda.max_memory_allocated(f"cuda:{local_rank}")
    print(f"Memory peak: {memory_peak / 1024**3:.2f} GB")
    if local_rank == 0 and not args.skip_saving_output:
        output_filename = f"xDiT.wan.output.sp{sp_size}.{args.attn_type}"
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
