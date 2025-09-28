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
from PIL import Image
from fsdp import shard_model, free_model
from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from transformers import CLIPVisionModel
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

logger = init_logger(__name__)

# Map argument to AttnType enum
attn_impl_map = {
    "torch": AttnType.TORCH,
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


def pad_freqs(original_tensor, target_len, seq_dim_idx=-2):
    seq_len = original_tensor.shape[seq_dim_idx]
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        *original_tensor.shape[:seq_dim_idx],
        pad_size,
        *original_tensor.shape[seq_dim_idx+1:],
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=seq_dim_idx)
    return padded_tensor


class xDiTWanAttnProcessor(WanAttnProcessor2_0):
    r"""
    Processor for implementing scaled dot-product attention for the Wan2.1 model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self, attn_type: str = "fa"):
        super().__init__()
        attn_impl = attn_impl_map[attn_type]
        self.hybrid_seq_parallel_attn = xFuserLongContextAttention(attn_type=attn_impl)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert attention_mask is None, "attention_mask is not supported for xDiT"

        attn_proc_rng = nvtx.start_range("attn_proc", color="red")
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

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

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
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

            attn_rng = nvtx.start_range("hybrid_seq_parallel_attn", color="red")
            hidden_states_img = self.hybrid_seq_parallel_attn(
                None,
                query,
                key_img,
                value_img,
                dropout_p=0.0,
                causal=False,
            )
            nvtx.end_range(attn_rng)
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attn_rng = nvtx.start_range("hybrid_seq_parallel_attn", color="red")
        hidden_states = self.hybrid_seq_parallel_attn(
            None,
            query,
            key,
            value,
            dropout_p=0.0,
            causal=False,
        )
        nvtx.end_range(attn_rng)
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

        rotary_emb = pad_freqs(rotary_emb, max_seq_len * sp_size, seq_dim_idx=seq_dim_idx)
        rotary_emb = torch.chunk(rotary_emb, sp_size, dim=seq_dim_idx)[sp_rank]

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


def replace_blockwise_gemm(transformer: WanTransformer3DModel):
    from linear_impl import VflyLinear
    # Only replace the linear layers in Wan2.1TransformerBlock
    model = transformer.blocks
    replace_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            tokens = name.strip().split('.')
            layer = model
            for t in tokens[:-1]:
                if not t.isnumeric():
                    layer = getattr(layer, t)
                else:
                    layer = layer[int(t)]
            replace_layers.append([layer, tokens[-1], module])

    for layer, name, module in replace_layers:
        setattr(layer, name, VflyLinear.from_linear(module, linear_type="trtllm-fp8-blockwise"))
    return transformer


@record
def main(args):
    dist.init_process_group("nccl", timeout=timedelta(seconds=1000))
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
    image_encoder = CLIPVisionModel.from_pretrained(
        model_id, subfolder="image_encoder", torch_dtype=torch.float16
    )
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
    transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16).to(torch.bfloat16)

    if args.blockwise_gemm:
        transformer = replace_blockwise_gemm(transformer)

    parallelize_transformer(transformer, sp_size, sp_rank, args.attn_type)

    pipe = WanImageToVideoPipeline.from_pretrained(
        pretrained_model_name_or_path=model_id,
        vae=vae.to(f"cuda:{local_rank}"),
        image_encoder=image_encoder.to(f"cuda:{local_rank}"),
        torch_dtype=torch.bfloat16,
    )
    pipe.transformer = None
    pipe.to(f"cuda:{local_rank}")
    if args.ulysses_degree > 1 or args.ring_degree > 1:
        pipe.transformer = shard_fn(transformer)
    else:
        pipe.transformer = transformer.to(f"cuda:{local_rank}")

    torch.cuda.empty_cache()

    image = load_image(
        "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/resolve/main/examples/i2v_input.JPG"
    )
    # image = image.resize((960, 1280), Image.LANCZOS)
    image = image.resize((1280, 720), resample=Image.Resampling.LANCZOS)
    # image = image.resize((480, 854), resample=Image.Resampling.LANCZOS)
    max_area = np.prod(image.size)
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((width, height))

    prompt = (
        "Summer beach vacation style, a white cat wearing sunglasses \
        sits on a surfboard. The fluffy-furred feline gazes directly at the camera \
        with a relaxed expression. Blurred beach scenery forms the background featuring \
        crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. \
        The cat assumes a naturally relaxed posture, as if savoring the sea breeze and \
        warm sunlight. A close-up shot highlights the feline's intricate details and the \
        soft texture of its fur. The cat's expression conveys a sense of relaxation and \
        contentment, as it enjoys the warm sun and the gentle sea breeze."
    )
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()

    # print(f"cuda:{local_rank} memory summary:\n{torch.cuda.memory_summary(device=f'cuda:{local_rank}', abbreviated=False)}")
    if local_rank == 0:
        print(pipe.transformer)

    with torch.no_grad():
        pipeline_rng = nvtx.start_range("pipeline", color="blue")
        output = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=args.num_frames,
            guidance_scale=5.0,
            num_inference_steps=args.num_inference_steps,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).frames[0]
        nvtx.end_range(pipeline_rng)

    torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time
    memory_peak = torch.cuda.max_memory_allocated(f"cuda:{local_rank}")
    print(f"Memory peak: {memory_peak / 1024**3:.2f} GB")
    if local_rank == 0 and not args.skip_saving_output:
        output_filename = f"xDiT.wan.output.sp{sp_size}.{args.attn_type}"
        if args.blockwise_gemm:
            output_filename += ".blockwise_gemm"
        export_to_video(output, f"{output_filename}.mp4", fps=args.fps)
        print(f"epoch time: {elapsed_time:.2f} sec; export video to {output_filename}.mp4")
    torch.cuda.empty_cache()

    dist.destroy_process_group()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers")
    parser.add_argument("--ulysses_degree", type=int, default=1)
    parser.add_argument("--ring_degree", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--blockwise_gemm", action="store_true")
    parser.add_argument("--attn_type", type=str, default="fa", help="Available choices: [torch, fa, fa3, flashinfer, sage_fp16, sage_fp8, sage_fp8_sm90, sage_fp16_triton, sage_auto, sparse_sage]")
    parser.add_argument("--skip_saving_output", action="store_true")
    args = parser.parse_args()

    main(args)
