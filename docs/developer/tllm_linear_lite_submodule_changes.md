# tllm_linear_lite Submodule Change Summary

Date: 2026-06-08

This note summarizes the current dirty worktree under
`third_party/tllm_linear_lite`. These changes belong to the
`tllm_linear_lite` repository, not to the xDiT repository. A main-repo commit
can only record a submodule pointer; it cannot include these internal file
changes.

## Current State

- Submodule path: `third_party/tllm_linear_lite`
- Clean HEAD before dirty changes:
  `b2a9cce932a085c913575cfa7dc468655f873090`
- HEAD subject: `docs: document FP8 blockscale backend policy`
- Dirty tracked files:
  - `README.md`
  - `docs/p1_fp8_blockscale_handoff.md`
  - `gemm/cublaslt_fp8_blockscale_gemm.cpp`
  - `setup.py`
  - `tests/README.md`
  - `tests/test_fp8_blockscale_gemm.py`
  - `tllm_linear_lite/__init__.py`
  - `tllm_linear_lite/fp8_blockscale_gemm.py`
  - `tllm_linear_lite/nvfp4_linear.py`
- Dirty untracked files:
  - `gemm/trtllm_gen/`
  - `gemm/trtllm_gen_fp8_blockscale_gemm_op.cpp`

## Functional Changes

### FP8 Blockscale on Blackwell

The dirty submodule adds a Blackwell `trtllm_gen` FP8 blockscale GEMM backend:

- `fp8_blockscale_gemm(..., backend="trtllm_gen")` now dispatches to
  `torch.ops.tllm_linear_lite.trtllm_gen_fp8_blockscale_gemm`.
- `resolve_backend("auto", sm=100/103, ...)` now returns `trtllm_gen` for
  supported BF16 output shapes.
- `required_sf_layout("trtllm_gen")` uses the `deepgemm` activation scale
  layout: `[K / 128, pad4(M)]`.
- `setup.py` builds the new wrapper and vendored `gemm/trtllm_gen/` sources.
- `cublaslt_fp8_blockscale_gemm.cpp` rejects SM100/103 and tells callers to use
  `auto` or `trtllm_gen`.

This matters for xDiT because `examples/linear_impl.py` constructs
`FP8BlockScaleDynamicLinear.from_linear(..., gemm_backend="auto")`. On B200,
the successful FP8 T2V run relied on this dirty `auto -> trtllm_gen` behavior.

### NVFP4 Activation Quantization on B200

The dirty submodule changes `NVFP4DynamicLinear` to always use
`cuda_prologue(...)` for activation amax/global-scale calculation when
`quant_backend == "tllm"`.

The removed path used a Triton amax threshold. The local note in the diff says
that Triton amax can hit illegal memory accesses on B200/SM100 for large
Wan2.1 activation tensors, while `cuda_prologue` computes the same amax/scale
pair in one native kernel.

This matters for xDiT because `examples/linear_impl.py` constructs
`NVFP4DynamicLinear.from_linear(..., gemm_backend="cublaslt",
scale_rule="static_6")`. The successful NVFP4 T2V run used the dirty
submodule implementation.

### Tests and Documentation

The dirty submodule updates README, handoff docs, and FP8 tests so they reflect:

- `trtllm_gen` as implemented instead of reserved.
- SM100/103 FP8 blockscale policy: use `trtllm_gen`, do not fall back to
  cuBLASLt.
- FP16 output on SM100/103 is rejected for this path.
- A new SM100/103 test checks `trtllm_gen` against an FP32 reference.

## Verified Locally

With these dirty submodule changes copied into a temporary build directory and
installed inside the B200 container, Wan2.1 T2V ran successfully with normal
settings:

- BF16: pass
- FP8 blockwise: pass
- NVFP4: pass

The generated videos were visually sane in first/middle/end frame checks. This
does not prove that the clean submodule HEAD is sufficient; the runs depended
on the dirty submodule changes above.

## Recommended Handling

1. Review, clean up, and commit the `tllm_linear_lite` changes in the
   `tllm_linear_lite` repository first.
2. Push that submodule commit.
3. In this xDiT repo, update `third_party/tllm_linear_lite` to the clean
   committed submodule SHA and stage only the pointer update.
4. If xDiT should not own this dependency as a submodule, drop the
   `.gitmodules` and submodule pointer changes and require users to pass an
   external `TLLM_LINEAR_LITE_ROOT` instead.
