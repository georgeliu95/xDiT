# `tllm_linear_lite` Submodule Policy

Updated: 2026-08-31

## Current State

- Path: `third_party/tllm_linear_lite`
- Upstream: `https://github.com/georgeliu95/tllm_linear_lite.git`
- Update branch: `main`
- Recorded commit: `7dccdc0846b5953cec00d94d0b336eac6d2749c4`
- Recorded commit subject: `Remove internal artifacts from docs`
- Nested CUTLASS commit: `e64a9136dd929639e5f7c969fe5af3bf7415cd4f`
- Submodule worktree: clean

The parent repository records a fixed submodule commit so that a normal
checkout is reproducible. The `branch = main` setting in `.gitmodules` tells
Git which upstream branch to follow when an explicit remote update is
requested; it does not make a normal checkout move automatically.

## Checkout

Initialize the recorded dependency and its nested submodules:

```bash
git submodule update --init --recursive
```

## Blackwell Build and Validation Environment

The full NVFP4, FP8 block-scale, and fused SVDQuant paths require CUDA Toolkit
13.1 or newer. On B200/GB200 (SM100), materialize the TRTLLMGen Git-LFS cubins
before building; pointer files are rejected rather than silently embedded:

```bash
git lfs install
git -C third_party/tllm_linear_lite lfs pull

CUDA_HOME=/usr/local/cuda \
TLLM_LINEAR_LITE_BUILD_MODE=full \
TLLM_LINEAR_LITE_ENABLE_TRTLLM_GEN=1 \
TORCH_CUDA_ARCH_LIST="10.0a" \
  pip install -e "third_party/tllm_linear_lite[cutedsl,trtllm_gen]" \
    --no-build-isolation
```

The `cutedsl` extra installs `nvidia-cutlass-dsl[cu13]>=4.6.1`. Do not rely on
an older CuTe DSL already present in the container: version 4.4.2 does not
export `OperandMajorMode` from `cutlass.cute.nvgpu`, so the fused SVDQuant
NVFP4 implementation fails during import. When TRTLLMGen is compiled, the
FP8 block-scale `auto` backend selects it on SM100/SM103.

## Refresh to the Latest `main`

Update the dependency and nested submodules from the configured branch:

```bash
git submodule update --remote --recursive third_party/tllm_linear_lite
git -C third_party/tllm_linear_lite status --short --branch
git diff --submodule=log -- third_party/tllm_linear_lite
```

After the relevant xDiT tests pass, record the new dependency commit in the
parent repository:

```bash
git add third_party/tllm_linear_lite
git commit
```

Do not edit files inside the detached submodule checkout and expect a parent
repository commit to preserve them. Changes to `tllm_linear_lite` must first
be committed and pushed in that repository; xDiT then records the resulting
commit pointer.
