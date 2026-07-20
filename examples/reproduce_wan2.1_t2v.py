"""Legacy Wan2.1 T2V reproduction command."""

from __future__ import annotations

import argparse

from wan21_t2v_runner import logger, main
from wan_runtime import get_system_memory_info, print_memory_usage
from wan_t2v_parallel import parallelize_transformer, validate_wan_fa4_request


__all__ = [
    "get_system_memory_info",
    "logger",
    "main",
    "parallelize_transformer",
    "print_memory_usage",
    "validate_wan_fa4_request",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="Wan-AI/Wan2.1-T2V-14B-Diffusers",
    )
    parser.add_argument("--ulysses_degree", type=int, default=1)
    parser.add_argument("--ring_degree", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--height", type=int, default=832)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument(
        "--quant_gemm_type",
        type=str,
        default=None,
        help="Available choices: [bf16 (default, no quantization), nvfp4, "
        "trtllm-fp8-blockwise, nvfp4+trtllm-fp8-blockwise]",
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
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument(
        "--attn_type",
        type=str,
        default="fa",
        choices=[
            "fa",
            "fa3",
            "fa4",
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
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
