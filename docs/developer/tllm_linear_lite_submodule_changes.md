# `tllm_linear_lite` Submodule Policy

Updated: 2026-07-31

## Current State

- Path: `third_party/tllm_linear_lite`
- Upstream: `https://github.com/georgeliu95/tllm_linear_lite.git`
- Update branch: `main`
- Recorded commit: `39b646162f101ea0884cb895a9a0dab103c1b886`
- Recorded commit subject: `Refactor SVDQuant NVFP4 M-tail contracts`
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
