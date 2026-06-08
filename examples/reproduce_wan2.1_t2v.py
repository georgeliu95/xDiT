import sys
import os
import argparse
import functools
import types
import math
from functools import partial
from typing import List, Optional, Tuple, Union, Dict, Any
import nvtx
import time
import torch
import numpy as np
import psutil
import gc

from PIL import Image
from fsdp import shard_model, free_model
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from datetime import timedelta

import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from xfuser.core.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
    get_world_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
)
from xfuser.logger import init_logger
from yunchang.kernels import AttnType

from diffusers.utils import export_to_video, load_image

from diffusers.models.attention import Attention
from diffusers.models.transformers.transformer_wan import WanAttnProcessor2_0, WanTransformer3DModel
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from xfuser.core.long_ctx_attention import xFuserLongContextAttention
from linear_impl import replace_linear_layer

logger = init_logger(__name__)

# Map argument to AttnType enum
attn_impl_map = {
    # "torch": AttnType.TORCH,
    "fa": AttnType.FA,
    "fa3": AttnType.FA3,
    "flashinfer": AttnType.FLASHINFER,
    "sage_fp16": AttnType.SAGE_FP16,
    "sage_fp8": AttnType.SAGE_FP8,
    "sage_fp8_sm90": AttnType.SAGE_FP8_SM90,
    "sage_fp16_triton": AttnType.SAGE_FP16_TRITON,
    "sage_auto": AttnType.SAGE_AUTO,
    "sparse_sage": AttnType.SPARSE_SAGE,
}


def get_system_memory_info(key: str = None):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    if key is None:
        return mem_info
    elif key == "available":
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
            for line in meminfo.split('\n'):
                if 'MemAvailable' in line:
                    mem_available_kb = int(line.split()[1])
                    mem_available_gb = mem_available_kb / 1024**2
                    break
        return mem_available_gb
    else:
        info = getattr(mem_info, key.lower(), None)
        assert info is not None, f"Invalid key for memory info: {key}"
        return info / 1024**3


def print_memory_usage(stage_name, rank):
    """Print the memory usage of the current process"""
    if rank != 0:
        return
    
    # Get the current process
    process = psutil.Process(os.getpid())
    
    # Process RSS memory
    mem_info = process.memory_info()
    rss_gb = mem_info.rss / 1024**3  # RSS in GB
    
    # System available memory
    mem_available_gb = get_system_memory_info("available")
    
    # GPU allocated memory and reserved memory
    if torch.cuda.is_available():
        gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"[{stage_name}] Rank 0 Memory - Process RSS: {rss_gb:.2f}GB, System Available: {mem_available_gb:.2f}GB, GPU Allocated: {gpu_mem_allocated:.2f}GB, GPU Reserved: {gpu_mem_reserved:.2f}GB")
    else:
        logger.info(f"[{stage_name}] Rank 0 Memory - Process RSS: {rss_gb:.2f}GB, System Available: {mem_available_gb:.2f}GB")


def pad_freqs(original_tensor, target_len, seq_dim_idx=-2, pad_value=1):
    seq_len = original_tensor.shape[seq_dim_idx]
    pad_size = target_len - seq_len
    if pad_size <= 0:
        return original_tensor
    padding_shape = (
        *original_tensor.shape[:seq_dim_idx],
        pad_size,
        *original_tensor.shape[seq_dim_idx+1:],
    )
    padding_tensor = original_tensor.new_full(padding_shape, pad_value)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=seq_dim_idx)
    return padded_tensor


def pad_rotary_emb(rotary_emb, target_len, seq_dim_idx=-2):
    if isinstance(rotary_emb, tuple):
        freqs_cos, freqs_sin = rotary_emb
        return (
            pad_freqs(freqs_cos, target_len, seq_dim_idx=1, pad_value=1),
            pad_freqs(freqs_sin, target_len, seq_dim_idx=1, pad_value=0),
        )
    return pad_freqs(rotary_emb, target_len, seq_dim_idx=seq_dim_idx)


def chunk_rotary_emb(rotary_emb, chunks, rank, seq_dim_idx=-2):
    if isinstance(rotary_emb, tuple):
        return tuple(torch.chunk(freqs, chunks, dim=1)[rank] for freqs in rotary_emb)
    return torch.chunk(rotary_emb, chunks, dim=seq_dim_idx)[rank]


class xDiTWanAttnProcessor(WanAttnProcessor2_0):
    r"""
    Processor for implementing scaled dot-product attention for the Wan2.1 model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self, attn_type: str = "fa"):
        super().__init__()
        attn_impl = attn_impl_map[attn_type]
        self.hybrid_seq_parallel_attn = xFuserLongContextAttention(attn_type=attn_impl)

    @nvtx.annotate(message="xDiTWanAttnProcessor.__call__", color="red")
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        assert attention_mask is None, "attention_mask is not supported for xDiT"

        attn_proc_rng = nvtx.start_range("attn_proc", color="red")
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        with nvtx.annotate(message="qkv", color="red"):
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
            ):
                if isinstance(freqs, tuple):
                    freqs_cos, freqs_sin = freqs
                    hidden_states = hidden_states.transpose(1, 2)
                    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                    cos = freqs_cos[..., 0::2]
                    sin = freqs_sin[..., 1::2]
                    out = torch.empty_like(hidden_states)
                    out[..., 0::2] = x1 * cos - x2 * sin
                    out[..., 1::2] = x1 * sin + x2 * cos
                    return out.type_as(hidden_states).transpose(1, 2)

                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        query = query.transpose(1, 2)
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            ## xDiT Attention
            key_img = key_img.transpose(1, 2)
            value_img = value_img.transpose(1, 2)

            with nvtx.annotate(message="hybrid_seq_parallel_attn_img", color="red"):
                hidden_states_img = self.hybrid_seq_parallel_attn(
                    None,
                    query,
                    key_img,
                    value_img,
                    dropout_p=0.0,
                    causal=False,
                )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        with nvtx.annotate(message="hybrid_seq_parallel_attn", color="red"):
            hidden_states = self.hybrid_seq_parallel_attn(
                None,
                query,
                key,
                value,
                dropout_p=0.0,
                causal=False,
            )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        nvtx.end_range(attn_proc_rng)
        return hidden_states


def parallelize_transformer(transformer: WanTransformer3DModel,
                            sp_size: int,
                            sp_rank: int,
                            attn_type: str = "fa"):
    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        seq_dim_idx = -2
        forward_rng = nvtx.start_range("dit.forward", color="green")

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        max_seq_len = int(math.ceil(hidden_states.shape[seq_dim_idx] / sp_size)) * sp_size
        original_seq_len = hidden_states.shape[seq_dim_idx]

        ## Padding hidden_states to the max sequence length
        padding_shape = list(hidden_states.shape)
        padding_shape[seq_dim_idx] = max_seq_len - hidden_states.shape[seq_dim_idx]
        hidden_states = torch.cat([hidden_states, 
                                   hidden_states.new_zeros(*padding_shape)], 
                                  dim=seq_dim_idx)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        hidden_states = torch.chunk(
            hidden_states, sp_size, dim=seq_dim_idx)[sp_rank]
        max_seq_len = (original_seq_len + sp_size - 1) // sp_size

        rotary_emb = pad_rotary_emb(rotary_emb, max_seq_len * sp_size, seq_dim_idx=seq_dim_idx)
        rotary_emb = chunk_rotary_emb(rotary_emb, sp_size, sp_rank, seq_dim_idx=seq_dim_idx)

        if sp_size > 1:
            for block in transformer.blocks:
                block.attn1.processor = xDiTWanAttnProcessor(attn_type)
                # [NOTE] USP is not verified yet with Cross-Attention
                # block.attn2.processor = xDiTWanAttnProcessor(attn_type)
        else:
            for block in transformer.blocks:
                block.attn1.processor.attn_type = attn_type

        # Directly call the WanTransformer3DModel.blocks.forward
        block_rng = nvtx.start_range("dit.block", color="yellow")
        for block in transformer.blocks:
            hidden_states = block.forward(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
            )
        nvtx.end_range(block_rng)

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        if sp_size > 1:
            hidden_states = get_sp_group().all_gather(hidden_states.contiguous(), dim=seq_dim_idx)

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        p_t, p_h, p_w = self.config.patch_size
        hidden_states = hidden_states[:, :original_seq_len, :]
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            output = (output,)
        else:
            output = Transformer2DModelOutput(sample=output)

        nvtx.end_range(forward_rng)

        if dist.is_initialized():
            dist.barrier()
        return output

    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward


@record
def main(args):
    # Increase timeout because rank 0 needs to pre-download all model files
    dist.init_process_group("nccl", timeout=timedelta(seconds=3600))
    init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
    global_rank = dist.get_rank()
    local_rank = global_rank % torch.cuda.device_count()
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
    
    shard_fn = partial(shard_model, device_id=local_rank)

    model_id = args.model_id
    
    # Let rank 0 download all model files to HuggingFace cache to avoid timeout due to multiple ranks downloading simultaneously
    if dist.is_initialized():
        if global_rank == 0:
            logger.info(f"Rank 0: Pre-downloading all components to cache...")
            # 1. Download vae
            temp_model = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
            del temp_model
            gc.collect()
            
            # 2. Download transformer
            temp_model = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
            del temp_model
            gc.collect()
            
            # 3. Download pipeline other components (text_encoder, etc.)
            logger.info(f"Rank 0: Pre-downloading pipeline components...")
            temp_pipe = WanPipeline.from_pretrained(
                model_id,
                transformer=None,  # Do not load transformer to save memory
                vae=None,
                torch_dtype=torch.bfloat16,
            )
            del temp_pipe
            gc.collect()
            
            logger.info(f"Rank 0: All components downloaded to cache and memory released")
            print_memory_usage("After pre-download", global_rank)
        # Wait for rank 0 to complete download
        dist.barrier()
        logger.info(f"Rank {global_rank}: Starting to load models from cache...")

    # Load models in a staggered manner: each rank loads sequentially to avoid OOM
    # After each rank loads, wait for the next rank to start loading
    if dist.is_initialized() and get_system_memory_info("available") < 500:
        for i in range(dist.get_world_size()):
            if i == global_rank:
                logger.info(f"Rank {global_rank}: Loading models...")
                vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
                transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16).to(torch.bfloat16)
                logger.info(f"Rank {global_rank}: Models loaded successfully")
                print_memory_usage(f"After loading models (Rank {global_rank})", global_rank)
            dist.barrier()
    else:
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
        transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16).to(torch.bfloat16)

    # Before quantization, move VAE to GPU to free CPU memory
    if args.quant_gemm_type is not None and args.quant_gemm_type != "bf16":
        if dist.is_initialized() and get_system_memory_info("available") < 500:
            # First stagger the movement of the VAE to GPU
            for i in range(dist.get_world_size()):
                if i == global_rank:
                    logger.info(f"Rank {global_rank}: Moving VAE to GPU before quantization...")
                    vae = vae.to(f"cuda:{local_rank}")
                    torch.cuda.empty_cache()
                    logger.info(f"Rank {global_rank}: VAE moved to GPU, CPU memory freed for quantization")
                    print_memory_usage(f"Before quantization (Rank {global_rank})", global_rank)
                dist.barrier()
    
    # Stagger the quantization conversion (if needed)
    # Strategy: when one rank quantizes, other ranks move the transformer to GPU to free CPU memory
    if args.quant_gemm_type is not None:
        assert args.quant_gemm_type in ["bf16", "nvfp4", "trtllm-fp8-blockwise", "nvfp4+trtllm-fp8-blockwise"], "Invalid quant_gemm_type"
        if args.quant_gemm_type != "bf16":
            if dist.is_initialized() and get_system_memory_info("available") < 500:
                transformer_on_gpu = False
                for i in range(dist.get_world_size()):
                    if i == global_rank:
                        # If on GPU, move back to CPU
                        if transformer_on_gpu:
                            logger.info(f"Rank {global_rank}: Moving transformer back to CPU for quantization...")
                            transformer = transformer.cpu()
                            torch.cuda.empty_cache()
                        
                        print_memory_usage(f"Before quantization (Rank {global_rank})", global_rank)
                        # Current rank performs quantization
                        logger.info(f"Rank {global_rank}: Applying quantization ({args.quant_gemm_type})...")
                        transformer = replace_linear_layer(
                            transformer,
                            quant_gemm_type=args.quant_gemm_type,
                            nvfp4_gemm_backend=args.nvfp4_gemm_backend,
                            nvfp4_scale_rule=args.nvfp4_scale_rule,
                        )
                        logger.info(f"Rank {global_rank}: Quantization completed, moving to GPU...")
                        transformer = transformer.to(f"cuda:{local_rank}")
                        torch.cuda.empty_cache()
                        transformer_on_gpu = True
                        logger.info(f"Rank {global_rank}: Quantized transformer on GPU")
                        print_memory_usage(f"After quantization (Rank {global_rank})", global_rank)
                    elif not transformer_on_gpu and i == 0:
                        # On the first round, other ranks move the unquantized transformer to GPU to free CPU memory
                        logger.info(f"Rank {global_rank}: Temporarily moving transformer to GPU to free CPU memory...")
                        transformer = transformer.to(f"cuda:{local_rank}")
                        torch.cuda.synchronize()
                        transformer_on_gpu = True
                        logger.info(f"Rank {global_rank}: Transformer temporarily on GPU")
                    
                    dist.barrier()
            else:
                transformer = replace_linear_layer(
                    transformer,
                    quant_gemm_type=args.quant_gemm_type,
                    nvfp4_gemm_backend=args.nvfp4_gemm_backend,
                    nvfp4_scale_rule=args.nvfp4_scale_rule,
                )

    parallelize_transformer(transformer, sp_size, sp_rank, args.attn_type)

    # Stagger the movement of loaded models to GPU to free CPU memory, then create pipeline
    if dist.is_initialized() and get_system_memory_info("available") < 500:
        # If there is no quantization, or bf16, move VAE to GPU
        if args.quant_gemm_type is None or args.quant_gemm_type == "bf16":
            for i in range(dist.get_world_size()):
                if i == global_rank:
                    logger.info(f"Rank {global_rank}: Moving VAE to GPU...")
                    vae = vae.to(f"cuda:{local_rank}")
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    logger.info(f"Rank {global_rank}: VAE moved to GPU")
                    print_memory_usage(f"After moving VAE to GPU (Rank {global_rank})", global_rank)
                dist.barrier()
        else:
            logger.info(f"Rank {global_rank}: VAE already on GPU (moved before quantization)")
        
        # All ranks process transformer simultaneously (FSDP needs synchronization)
        logger.info(f"Rank {global_rank}: Moving transformer to GPU and applying FSDP...")
        transformer = transformer.to(f"cuda:{local_rank}")
        torch.cuda.synchronize()
        logger.info(f"Rank {global_rank}: Transformer moved to GPU")
        
        if args.ulysses_degree > 1 or args.ring_degree > 1:
            logger.info(f"Rank {global_rank}: Applying FSDP sharding (all ranks synchronized)...")
            transformer = shard_fn(transformer)
            logger.info(f"Rank {global_rank}: FSDP sharding completed")
        
        torch.cuda.empty_cache()
        logger.info(f"Rank {global_rank}: Transformer processing done")
        print_memory_usage(f"After transformer to GPU (Rank {global_rank})", global_rank)
        dist.barrier()
        
        # Now CPU memory is released, stagger the creation of pipeline (will load text_encoder, etc.)
        for i in range(dist.get_world_size()):
            if i == global_rank:
                logger.info(f"Rank {global_rank}: Creating pipeline...")
                pipe = WanPipeline.from_pretrained(
                    pretrained_model_name_or_path=model_id,
                    vae=vae,  # Already on GPU
                    torch_dtype=torch.bfloat16,
                )
                pipe.transformer = transformer  # Already on GPU
                pipe.to(f"cuda:{local_rank}")
                torch.cuda.empty_cache()
                logger.info(f"Rank {global_rank}: Pipeline created successfully")
                print_memory_usage(f"After creating pipeline (Rank {global_rank})", global_rank)
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

    # image = load_image(
    #     # "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/resolve/main/examples/i2v_input.JPG"
    #     "./cat_robot.png"
    # )
    # # image = image.resize((960, 1280), Image.LANCZOS)
    # # image = image.resize((1280, 720), resample=Image.Resampling.LANCZOS)
    # # image = image.resize((480, 854), resample=Image.Resampling.LANCZOS)
    # image = image.resize((720, 1280), resample=Image.Resampling.LANCZOS)
    # max_area = np.prod(image.size)
    # aspect_ratio = image.height / image.width
    # mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    # height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    # width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    # image = image.resize((width, height))

    default_prompt = (
        "Summer beach vacation style, a white cat wearing sunglasses \
        sits on a surfboard. The fluffy-furred feline gazes directly at the camera \
        with a relaxed expression. Blurred beach scenery forms the background featuring \
        crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. \
        The cat assumes a naturally relaxed posture, as if savoring the sea breeze and \
        warm sunlight. A close-up shot highlights the feline's intricate details and the \
        soft texture of its fur. The cat's expression conveys a sense of relaxation and \
        contentment, as it enjoys the warm sun and the gentle sea breeze."
    )
    prompt = args.prompt or default_prompt
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()

    if local_rank == 0:
        print(pipe.transformer)

    with torch.no_grad():
        pipeline_rng = nvtx.start_range("pipeline", color="blue")
        output = pipe(
            # image=image,
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
    end_time = time.time()
    elapsed_time = end_time - start_time
    memory_peak = torch.cuda.max_memory_allocated(f"cuda:{local_rank}")
    print(f"Memory peak: {memory_peak / 1024**3:.2f} GB")
    if local_rank == 0 and not args.skip_saving_output:
        output_filename = f"xDiT.wan.output.sp{sp_size}.{args.attn_type}"
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
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-14B-Diffusers")
    parser.add_argument("--ulysses_degree", type=int, default=1)
    parser.add_argument("--ring_degree", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--height", type=int, default=832)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--quant_gemm_type", 
                        type=str, default=None, 
                        help="Available choices: [bf16 (default, no quantization), nvfp4, trtllm-fp8-blockwise, nvfp4+trtllm-fp8-blockwise]")
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
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--attn_type", type=str, default="fa", help="Available choices: [torch, fa, fa3, flashinfer, sage_fp16, sage_fp8, sage_fp8_sm90, sage_fp16_triton, sage_auto, sparse_sage]")
    parser.add_argument("--skip_saving_output", action="store_true")
    args = parser.parse_args()

    main(args)
