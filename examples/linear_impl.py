import gc
import torch
import torch.nn.functional as F


class DefaultLinear:
    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        return F.linear(input, weight, bias)


class TrtllmFp8BlockLinear:
    def __init__(self):
        try:
            import tensorrt_llm  # noqa

            pass  # noqa
        except ImportError:
            raise ImportError("TensorRT-LLM is not installed.")

    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:

        # input
        origin_shape = input.shape
        origin_dtype = input.dtype
        input = input.to(torch.bfloat16)

        if input.dim() > 2:
            input = input.reshape(-1, input.shape[-1])

        act_input_fp8, input_scale = torch.ops.trtllm.fp8_quantize_1x128(input)
        output = torch.ops.trtllm.fp8_block_scaling_gemm(act_input_fp8, weight, input_scale, weight_scale)
        output = output.to(origin_dtype)

        if bias is not None:
            output = output + bias
        if output.dim() == 2:
            output = output.reshape(origin_shape[0], origin_shape[1], -1)
        return output


class TrtllmFp8PerTensorLinear:
    def __init__(self):
        try:
            import tensorrt_llm  # noqa

            pass  # noqa
        except ImportError:
            raise ImportError("TensorRT-LLM is not installed.")

    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        origin_shape = input.shape
        origin_dtype = input.dtype
        input = input.to(torch.bfloat16)

        # Dynamic quantization
        qinput, cur_input_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(input)
        cur_input_scale = cur_input_scale.to(torch.float32)
        # This op does not support bias now.
        if qinput.dim() == 3:
            qinput = qinput.reshape(-1, qinput.shape[-1])

        output = torch.ops.trtllm.cublas_scaled_mm(
            qinput,
            weight,
            scale_a=cur_input_scale,
            scale_b=weight_scale,
            bias=None,
            out_dtype=input.dtype,
        )
        output = output.to(origin_dtype)
        if bias is not None:
            output = output + bias
        if output.dim() == 2:
            output = output.reshape(origin_shape[0], origin_shape[1], -1)
        return output


class VflyLinear(torch.nn.Linear):
    def __init__(self, linear_type: str = "default", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = None  # module name in the model
        self.linear_impl = None
        self.input_scale = None
        self.weight_scale = None
        self.linear_type = linear_type

    def create_blockwise_quantized_weight(
        self,
        param_value: torch.Tensor,
        block_size: int = 128,
    ):
        # refer to transfromers fp8 128*128 block quantization
        # (https://github.com/huggingface/transformers/blob/main/src/transformers/quantizers/quantizer_finegrained_fp8.py)

        param_value = param_value.to(torch.float32)

        # Get FP8 min/max values
        fp8_min = torch.finfo(torch.float8_e4m3fn).min
        fp8_max = torch.finfo(torch.float8_e4m3fn).max

        block_size_m, block_size_n = block_size, block_size
        rows, cols = param_value.shape[-2:]
        if rows % block_size_m != 0 or cols % block_size_n != 0:
            raise ValueError(
                f"Matrix dimensions ({rows}, {cols}) must be divisible by block sizes ({block_size_m}, {block_size_n})"
            )
        param_value_orig_shape = param_value.shape
        param_value = param_value.reshape(
            -1, rows // block_size_m, block_size_m, cols // block_size_n, block_size_n
        ).permute(0, 1, 3, 2, 4)

        # Calculate scaling factor for each block
        max_abs = torch.amax(torch.abs(param_value), dim=(-1, -2))
        scale = fp8_max / max_abs
        scale_orig_shape = scale.shape
        scale = scale.unsqueeze(-1).unsqueeze(-1)

        @torch.compiler.disable()
        def _quantize(param_value, scale, fp8_min, fp8_max):
            # Quantize the weights
            quantized_param = torch.clamp(param_value * scale, min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

            quantized_param = quantized_param.permute(0, 1, 3, 2, 4)
            # Reshape back to matrix shape
            quantized_param = quantized_param.reshape(param_value_orig_shape)

            # Reshape scale to match the number of blocks
            scale = scale.reshape(scale_orig_shape).squeeze().reciprocal()

            return quantized_param, scale

        quantized_param, scale = _quantize(param_value, scale, fp8_min, fp8_max)
        return quantized_param, scale

    def create_per_tensor_quantized_weight(self, param_value: torch.Tensor):
        param_value = param_value.to(torch.float32)

        # Get FP8 min/max values
        fp8_min = torch.finfo(torch.float8_e4m3fn).min
        fp8_max = torch.finfo(torch.float8_e4m3fn).max

        max_abs = torch.amax(torch.abs(param_value))
        scale = fp8_max / max_abs

        @torch.compiler.disable()
        def _quantize(param_value, scale, fp8_min, fp8_max):
            quantized_param = torch.clamp(param_value * scale, min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
            quantized_param = quantized_param.t()
            scale = scale.reshape(1, 1).reciprocal()
            return quantized_param, scale

        quantized_param, scale = _quantize(param_value, scale, fp8_min, fp8_max)
        return quantized_param, scale

    def select_linear_impl(self):
        # select linear implementation
        if self.linear_type == "trtllm-fp8-blockwise":
            self.linear_impl = TrtllmFp8BlockLinear()
        elif self.linear_type == "trtllm-fp8-per-tensor":
            self.linear_impl = TrtllmFp8PerTensorLinear()
        else:
            self.linear_impl = DefaultLinear()

        weight_name = self.linear_type + "_weight"
        weight_scale_name = self.linear_type + "_weight_scale"
        # compute quantized weight and weight scale if needed
        if self.linear_type == "trtllm-fp8-blockwise":
            if not hasattr(self, weight_name) or not hasattr(self, weight_scale_name):
                weight, weight_scale = self.create_blockwise_quantized_weight(self.weight)
                self.register_parameter(weight_name, torch.nn.Parameter(weight))
                self.register_buffer(weight_scale_name, weight_scale)
        elif self.linear_type == "trtllm-fp8-per-tensor":
            if not hasattr(self, weight_name) or not hasattr(self, weight_scale_name):
                weight, weight_scale = self.create_per_tensor_quantized_weight(self.weight)
                self.register_parameter(weight_name, torch.nn.Parameter(weight))
                self.register_buffer(weight_scale_name, weight_scale)

        if self.linear_type != "auto":
            # Free default weight to save memory
            keys_to_delete = []
            for key, _ in self.named_parameters():
                if key == "weight" and self.linear_type != "default":
                    keys_to_delete.append(key)
                elif key.endswith("_weight") and key != weight_name:
                    keys_to_delete.append(key)
            for key, _ in self.named_buffers():
                if key.endswith("_weight_scale") and key != weight_scale_name:
                    keys_to_delete.append(key)
            for key in keys_to_delete:
                delattr(self, key)
            if keys_to_delete:  # Only clear cache if we actually deleted something
                gc.collect()
                torch.cuda.empty_cache()
        else:
            # TODO: we cached all kinds of weights, weight_scales for "auto"
            pass

        if self.linear_type == "default":
            weight = self.weight
            weight_scale = None
        else:
            weight = getattr(self, weight_name)
            weight_scale = getattr(self, weight_scale_name)

        return weight, weight_scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight, weight_scale = self.select_linear_impl()
        return self.linear_impl(input, weight, self.bias, self.input_scale, weight_scale)

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear, linear_type: str = "default") -> "VflyLinear":
        device = linear.weight.device
        dtype = linear.weight.dtype
        vfly_linear = cls(
            in_features=linear.in_features, 
            out_features=linear.out_features, 
            device=device, 
            dtype=dtype, 
            linear_type=linear_type)
        with torch.no_grad():
            vfly_linear.weight.copy_(linear.weight)
            vfly_linear.bias.copy_(linear.bias)
        return vfly_linear

