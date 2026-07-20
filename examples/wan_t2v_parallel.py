"""Wan T2V sequence sharding and legacy forward instrumentation."""

from __future__ import annotations

import functools
import math
import os
from typing import Any, Dict, Final, Optional

import nvtx
import torch
import torch.distributed as dist
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

from xfuser.core.distributed import (
    get_runtime_state,
    get_sp_group,
    initialize_runtime_state,
    runtime_state_is_initialized,
)
from xfuser.core.distributed.attention_backend import AttentionBackendType
from xfuser.envs import PACKAGES_CHECKER
from xfuser.logger import init_logger
from xfuser.model_executor.models.transformers.transformer_wan import (
    xFuserWanAttnProcessor,
)

from wan_legacy_attention import xDiTWanAttnProcessor


logger = init_logger(__name__)
FA4_ATTENTION_TYPE: Final = "fa4"


class WanFa4ParallelismError(RuntimeError):
    """Raised when the legacy runner requests FA4 with sequence parallelism."""


class WanFa4RuntimeError(RuntimeError):
    """Raised when the legacy FA4 route is not runnable on this host."""


def validate_wan_fa4_request(
    attention_type: str,
    ulysses_degree: int,
    ring_degree: int,
) -> None:
    """Fail before model loading when a requested legacy FA4 route cannot run."""
    if attention_type != FA4_ATTENTION_TYPE:
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if ulysses_degree != 1 or ring_degree != 1 or world_size != 1:
        raise WanFa4ParallelismError(
            "The legacy Wan FA4 route requires ulysses_degree=1 and "
            "ring_degree=1 in a single-process launch; received "
            f"ulysses_degree={ulysses_degree}, ring_degree={ring_degree}, "
            f"and WORLD_SIZE={world_size}."
        )

    package_available = PACKAGES_CHECKER.get_packages_info()["has_flash_attn_4"]
    device_capability = (
        torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    )
    if device_capability != (12, 0) or not package_available:
        raise WanFa4RuntimeError(
            "The legacy Wan FA4 route requires an SM120 CUDA device and the "
            "matching flash-attn-4 package; received "
            f"device_capability={device_capability} and "
            f"package_available={package_available}."
        )


def configure_wan_fa4_single_device(
    transformer: WanTransformer3DModel,
    sequence_parallel_size: int,
) -> None:
    """Route both Wan self- and cross-attention through core BF16 FA4."""
    validate_wan_fa4_request(FA4_ATTENTION_TYPE, sequence_parallel_size, 1)

    if not runtime_state_is_initialized():
        initialize_runtime_state()
    runtime_state = get_runtime_state()
    runtime_state.set_attention_backend(AttentionBackendType.FLASH_4)
    runtime_state.set_cross_attention_backend(AttentionBackendType.FLASH_4)

    for block in transformer.blocks:
        block.attn1.processor = xFuserWanAttnProcessor(
            use_ulysses_parallel_attention=False,
        )
        block.attn2.processor = xFuserWanAttnProcessor(
            use_ulysses_parallel_attention=False,
            is_cross_attention=True,
        )


def pad_freqs(
    original_tensor: torch.Tensor,
    target_len: int,
    seq_dim_idx: int = -2,
    pad_value: int = 1,
) -> torch.Tensor:
    seq_len = original_tensor.shape[seq_dim_idx]
    pad_size = target_len - seq_len
    if pad_size <= 0:
        return original_tensor
    padding_shape = (
        *original_tensor.shape[:seq_dim_idx],
        pad_size,
        *original_tensor.shape[seq_dim_idx + 1 :],
    )
    padding_tensor = original_tensor.new_full(padding_shape, pad_value)
    return torch.cat([original_tensor, padding_tensor], dim=seq_dim_idx)


def pad_rotary_emb(rotary_emb, target_len: int, seq_dim_idx: int = -2):
    if isinstance(rotary_emb, tuple):
        freqs_cos, freqs_sin = rotary_emb
        return (
            pad_freqs(freqs_cos, target_len, seq_dim_idx=1, pad_value=1),
            pad_freqs(freqs_sin, target_len, seq_dim_idx=1, pad_value=0),
        )
    return pad_freqs(rotary_emb, target_len, seq_dim_idx=seq_dim_idx)


def chunk_rotary_emb(rotary_emb, chunks: int, rank: int, seq_dim_idx: int = -2):
    if isinstance(rotary_emb, tuple):
        return tuple(torch.chunk(freqs, chunks, dim=1)[rank] for freqs in rotary_emb)
    return torch.chunk(rotary_emb, chunks, dim=seq_dim_idx)[rank]


def parallelize_transformer(
    transformer: WanTransformer3DModel,
    sp_size: int,
    sp_rank: int,
    attn_type: str = "fa",
) -> None:
    """Install the profiled Wan forward and selected attention processors."""
    uses_fa4 = attn_type == FA4_ATTENTION_TYPE
    if uses_fa4:
        configure_wan_fa4_single_device(transformer, sp_size)

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
            scale_lora_layers(self, lora_scale)
        elif attention_kwargs is not None and attention_kwargs.get("scale") is not None:
            logger.warning(
                "Passing `scale` via `attention_kwargs` without PEFT has no effect."
            )

        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        max_seq_len = int(math.ceil(hidden_states.shape[seq_dim_idx] / sp_size)) * sp_size
        original_seq_len = hidden_states.shape[seq_dim_idx]

        padding_shape = list(hidden_states.shape)
        padding_shape[seq_dim_idx] = max_seq_len - hidden_states.shape[seq_dim_idx]
        hidden_states = torch.cat(
            [hidden_states, hidden_states.new_zeros(*padding_shape)],
            dim=seq_dim_idx,
        )

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(
                timestep,
                encoder_hidden_states,
                encoder_hidden_states_image,
            )
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states],
                dim=1,
            )

        hidden_states = torch.chunk(hidden_states, sp_size, dim=seq_dim_idx)[sp_rank]
        max_seq_len = (original_seq_len + sp_size - 1) // sp_size
        rotary_emb = pad_rotary_emb(
            rotary_emb,
            max_seq_len * sp_size,
            seq_dim_idx=seq_dim_idx,
        )
        rotary_emb = chunk_rotary_emb(
            rotary_emb,
            sp_size,
            sp_rank,
            seq_dim_idx=seq_dim_idx,
        )

        if not uses_fa4:
            if sp_size > 1:
                for block in transformer.blocks:
                    block.attn1.processor = xDiTWanAttnProcessor(attn_type)
            else:
                for block in transformer.blocks:
                    block.attn1.processor.attn_type = attn_type

        block_rng = nvtx.start_range("dit.block", color="yellow")
        for block in transformer.blocks:
            hidden_states = block.forward(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
            )
        nvtx.end_range(block_rng)

        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        hidden_states = (
            self.norm_out(hidden_states.float()) * (1 + scale) + shift
        ).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        if sp_size > 1:
            hidden_states = get_sp_group().all_gather(
                hidden_states.contiguous(),
                dim=seq_dim_idx,
            )

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        p_t, p_h, p_w = self.config.patch_size
        hidden_states = hidden_states[:, :original_seq_len, :]
        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)
        if not return_dict:
            output = (output,)
        else:
            output = Transformer2DModelOutput(sample=output)

        nvtx.end_range(forward_rng)
        if dist.is_initialized():
            dist.barrier()
        return output

    transformer.forward = new_forward.__get__(transformer)
