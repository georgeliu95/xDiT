# Running Wan2.2 T2V and Qwen-Image

Wan2.2 T2V and Qwen-Image use the unified xDiT runner. This guide covers the
model variants that support a separately bound weights directory:

- `Wan2.2-T2V`
- `Qwen-Image` and `Qwen-Image-2512`

The commands below assume xDiT is installed from this checkout so that the
`xdit` command is available. Create the output directory before running a
pipeline:

```bash
mkdir -p outputs
```

## Wan2.2 T2V

Run the default Hugging Face model on one GPU:

```bash
xdit --model Wan2.2-T2V \
  --prompt "Two cats wearing boxing gloves spar on a brightly lit stage" \
  --height 720 \
  --width 1280 \
  --num_frames 81 \
  --num_inference_steps 40 \
  --guidance_scale 3.5 \
  --output_directory outputs
```

For sequence parallelism across eight GPUs, add:

```bash
--ulysses_degree 8
```

The `xdit` launcher derives the process count from the parallel degrees and
starts `torchrun`; do not wrap the command in another `torchrun` invocation.

Attention and linear implementations can be selected through the unified
backend arguments. For example, the following single-GPU command uses
FlashInfer NVFP4 for self-attention and FlashAttention 4 for cross-attention:

```bash
xdit --model Wan2.2-T2V \
  --prompt "Two cats wearing boxing gloves spar on a brightly lit stage" \
  --attention_backend flashinfer_nvfp4 \
  --cross_attention_backend flash_4 \
  --ulysses_degree 1 \
  --ring_degree 1 \
  --output_directory outputs
```

These optional backends require their corresponding packages and supported GPU
architectures. Omit the backend arguments for the default implementation.

## Qwen-Image

Run Qwen-Image with its model-specific default resolution of 928x1664:

```bash
xdit --model Qwen-Image \
  --prompt "A cinematic photograph of a red panda reading beside a window" \
  --negative_prompt "low quality, blurry" \
  --num_inference_steps 50 \
  --max_sequence_length 256 \
  --seed 42 \
  --output_directory outputs
```

Use `--model Qwen-Image-2512` to select the 2512 checkpoint. Qwen-Image also
supports Ulysses and ring sequence parallelism; for example, add
`--ulysses_degree 4` to launch on four GPUs.

`max_sequence_length` is forwarded to the Qwen pipeline. Increase it when a
prompt needs more text tokens, subject to the model and available memory.

## Loading from a bound weights directory

The CLI model name selects a registered pipeline and its default Hugging Face
repository. When an embedding application or benchmark binds a different
weights location, use the programmatic runner and keep the registered model
name separate from `weights_locator`.

The weights directory must use Diffusers layout and contain the complete
pipeline assets, including `model_index.json` and the `transformer/` subfolder.
For Wan2.2 T2V it must also contain `transformer_2/`.

Save the following as `run_bound_pipeline.py` outside the `examples/`
directory, or adapt it in the calling application:

```python
from xfuser.runner import xFuserModelRunner


wan22_config = {
    "model": "Wan2.2-T2V",
    "weights_locator": "/path/to/wan2.2-diffusers-checkpoint",
    "prompt": "Two cats wearing boxing gloves spar on a brightly lit stage",
    "height": 720,
    "width": 1280,
    "num_frames": 81,
    "num_inference_steps": 40,
    "guidance_scale": 3.5,
    "max_sequence_length": 256,
    "seed": 42,
    "ulysses_degree": 1,
    "ring_degree": 1,
    "output_directory": "outputs",
}

qwen_image_config = {
    "model": "Qwen-Image",
    "weights_locator": "/path/to/qwen-image-diffusers-checkpoint",
    "prompt": "A cinematic photograph of a red panda reading beside a window",
    "negative_prompt": "low quality, blurry",
    "height": 928,
    "width": 1664,
    "num_inference_steps": 50,
    "guidance_scale": 0.0,
    "max_sequence_length": 256,
    "seed": 42,
    "ulysses_degree": 1,
    "ring_degree": 1,
    "output_directory": "outputs",
}

# Select one pipeline configuration.
config = wan22_config

runner = xFuserModelRunner(config)
try:
    input_args = runner.preprocess_args(config)
    runner.initialize(input_args)
    output, timings = runner.run(input_args)
    runner.save(output=output, timings=timings)
finally:
    runner.cleanup()
```

Run it on one GPU:

```bash
torchrun --nproc_per_node=1 run_bound_pipeline.py
```

For a four-GPU Qwen-Image run, select `qwen_image_config`, set its
`ulysses_degree` to `4`, and launch:

```bash
torchrun --nproc_per_node=4 run_bound_pipeline.py
```

`weights_locator` is currently a programmatic runner argument; it is not an
`xdit` CLI flag. If it is omitted or empty, both pipelines load from their
registered Hugging Face repository.

Generated images or videos and `timings.json` are written to the selected
`output_directory`.
