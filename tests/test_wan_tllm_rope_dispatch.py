"""Source contracts for xDiT's thin tllm_linear_lite Wan RoPE dispatch."""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WAN_TRANSFORMER = (
    ROOT
    / "xfuser"
    / "model_executor"
    / "models"
    / "transformers"
    / "transformer_wan.py"
)
OLD_IMPLEMENTATION = (
    ROOT
    / "xfuser"
    / "model_executor"
    / "models"
    / "transformers"
    / "wan_flashinfer_rope.py"
)


class WanTllmRopeDispatchContractTest(unittest.TestCase):
    def test_xdit_only_calls_the_tllm_public_api(self) -> None:
        source = WAN_TRANSFORMER.read_text(encoding="utf-8")

        self.assertIn("AttentionBackendType.FLASHINFER_NVFP4", source)
        self.assertIn(
            "from tllm_linear_lite.wan_qk_rope import "
            "apply_wan_qk_rotary_embedding",
            source,
        )
        self.assertIn("apply_wan_qk_rotary_embedding(", source)
        self.assertNotIn("prepare_flashinfer_wan_rotary_embedding", source)
        self.assertNotIn("get_cached_flashinfer_wan_rotary_embedding", source)
        self.assertNotIn("apply_flashinfer_wan_rotary_embedding", source)

    def test_xdit_no_longer_owns_a_rope_kernel_adapter(self) -> None:
        self.assertFalse(OLD_IMPLEMENTATION.exists())


if __name__ == "__main__":
    unittest.main()
