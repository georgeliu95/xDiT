import xfuser.envs as envs

from .ring_flash_attn import (
    xdit_ring_flash_attn_func,
    xdit_sana_ring_flash_attn_func,
)

__all__ = (
    "xdit_ring_flash_attn_func",
    "xdit_sana_ring_flash_attn_func",
)

if envs._is_npu():
    from .ring_npu_flash_attn import xdit_ring_npu_flash_attn_func

    __all__ += ("xdit_ring_npu_flash_attn_func",)
