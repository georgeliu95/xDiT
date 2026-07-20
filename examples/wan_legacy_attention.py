"""Legacy yunchang attention processor retained by Wan reproduction runners."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import nvtx
import torch
from diffusers.models.attention import Attention
from diffusers.models.transformers.transformer_wan import WanAttnProcessor2_0
from xfuser.core.long_ctx_attention import xFuserLongContextAttention
from yunchang.kernels import AttnType


ATTENTION_IMPLEMENTATIONS = {
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


class xDiTWanAttnProcessor(WanAttnProcessor2_0):
    """Wan rotary attention backed by the legacy yunchang implementations."""

    def __init__(self, attn_type: str = "fa"):
        super().__init__()
        self.hybrid_seq_parallel_attn = xFuserLongContextAttention(
            attn_type=ATTENTION_IMPLEMENTATIONS[attn_type]
        )

    @nvtx.annotate(message="xDiTWanAttnProcessor.__call__", color="red")
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
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

            def apply_wan_rotary_emb(
                states: torch.Tensor,
                freqs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
            ) -> torch.Tensor:
                if isinstance(freqs, tuple):
                    freqs_cos, freqs_sin = freqs
                    states = states.transpose(1, 2)
                    x1, x2 = states.unflatten(-1, (-1, 2)).unbind(-1)
                    cos = freqs_cos[..., 0::2]
                    sin = freqs_sin[..., 1::2]
                    out = torch.empty_like(states)
                    out[..., 0::2] = x1 * cos - x2 * sin
                    out[..., 1::2] = x1 * sin + x2 * cos
                    return out.type_as(states).transpose(1, 2)

                rotated = torch.view_as_complex(
                    states.to(torch.float64).unflatten(3, (-1, 2))
                )
                return torch.view_as_real(rotated * freqs).flatten(3, 4).type_as(states)

            query = apply_wan_rotary_emb(query, rotary_emb)
            key = apply_wan_rotary_emb(key, rotary_emb)

        hidden_states_img = None
        query = query.transpose(1, 2)
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)
            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            with nvtx.annotate(
                message="hybrid_seq_parallel_attn_img",
                color="red",
            ):
                hidden_states_img = self.hybrid_seq_parallel_attn(
                    None,
                    query,
                    key_img.transpose(1, 2),
                    value_img.transpose(1, 2),
                    dropout_p=0.0,
                    causal=False,
                )
            hidden_states_img = hidden_states_img.flatten(2, 3).type_as(query)

        with nvtx.annotate(message="hybrid_seq_parallel_attn", color="red"):
            hidden_states = self.hybrid_seq_parallel_attn(
                None,
                query,
                key.transpose(1, 2),
                value.transpose(1, 2),
                dropout_p=0.0,
                causal=False,
            )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        nvtx.end_range(attn_proc_rng)
        return hidden_states
