"""Shared host-memory and Wan2.1 model-loading helpers for legacy runners."""

from __future__ import annotations

import gc
import os

import psutil
import torch
import torch.distributed as dist
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

from xfuser.logger import init_logger


logger = init_logger(__name__)


class SystemMemoryInfoKeyError(KeyError):
    """Raised when a caller requests an unsupported process-memory field."""

    def __init__(self, key: str) -> None:
        super().__init__(f"Invalid key for memory info: {key}")


def get_system_memory_info(key: str | None = None):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    if key is None:
        return mem_info
    if key == "available":
        with open("/proc/meminfo", encoding="utf-8") as meminfo_file:
            for line in meminfo_file:
                if "MemAvailable" in line:
                    mem_available_kb = int(line.split()[1])
                    return mem_available_kb / 1024**2
        raise LookupError("MemAvailable is missing from /proc/meminfo")

    info = getattr(mem_info, key.lower(), None)
    if info is None:
        raise SystemMemoryInfoKeyError(key)
    return info / 1024**3


def print_memory_usage(stage_name: str, rank: int) -> None:
    """Print process, system, and device memory from rank zero."""
    if rank != 0:
        return

    rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    available_gb = get_system_memory_info("available")
    if not torch.cuda.is_available():
        logger.info(
            f"[{stage_name}] Rank 0 Memory - Process RSS: {rss_gb:.2f}GB, "
            f"System Available: {available_gb:.2f}GB"
        )
        return

    allocated_gb = torch.cuda.memory_allocated() / 1024**3
    reserved_gb = torch.cuda.memory_reserved() / 1024**3
    logger.info(
        f"[{stage_name}] Rank 0 Memory - Process RSS: {rss_gb:.2f}GB, "
        f"System Available: {available_gb:.2f}GB, "
        f"GPU Allocated: {allocated_gb:.2f}GB, GPU Reserved: {reserved_gb:.2f}GB"
    )


def predownload_wan21_components(model_id: str, global_rank: int) -> None:
    """Populate the shared model cache on rank zero before all ranks load."""
    if not dist.is_initialized():
        return

    if global_rank == 0:
        logger.info("Rank 0: Pre-downloading all components to cache...")
        temp_model = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.float16,
        )
        del temp_model
        gc.collect()

        temp_model = WanTransformer3DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        del temp_model
        gc.collect()

        logger.info("Rank 0: Pre-downloading pipeline components...")
        temp_pipe = WanPipeline.from_pretrained(
            model_id,
            transformer=None,
            vae=None,
            torch_dtype=torch.bfloat16,
        )
        del temp_pipe
        gc.collect()
        logger.info("Rank 0: All components downloaded and released")
        print_memory_usage("After pre-download", global_rank)

    dist.barrier()
    logger.info(f"Rank {global_rank}: Starting to load models from cache...")


def load_wan21_models(
    model_id: str,
    global_rank: int,
) -> tuple[AutoencoderKLWan, WanTransformer3DModel]:
    """Load the VAE and DiT, staggering ranks when host memory is constrained."""
    if dist.is_initialized() and get_system_memory_info("available") < 500:
        vae = None
        transformer = None
        for rank in range(dist.get_world_size()):
            if rank == global_rank:
                logger.info(f"Rank {global_rank}: Loading models...")
                vae = AutoencoderKLWan.from_pretrained(
                    model_id,
                    subfolder="vae",
                    torch_dtype=torch.float16,
                )
                transformer = WanTransformer3DModel.from_pretrained(
                    model_id,
                    subfolder="transformer",
                    torch_dtype=torch.bfloat16,
                ).to(torch.bfloat16)
                logger.info(f"Rank {global_rank}: Models loaded successfully")
                print_memory_usage(f"After loading models (Rank {global_rank})", global_rank)
            dist.barrier()
        assert vae is not None and transformer is not None
        return vae, transformer

    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float16,
    )
    transformer = WanTransformer3DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    ).to(torch.bfloat16)
    return vae, transformer
