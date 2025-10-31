from functools import partial
import gc
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import nvtx
from loguru import logger


default_wan22_recipe = {
    "blocks": [() for _ in range(40)],
}


########################################################
# FAKE SVDQuant INT4 Linear
########################################################

def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.
    Args:
        x: the dividend.
        y: the divisor.
    Returns:
        The result of the ceiling division.
    """
    return (x + y - 1) // y

def fake_quant_per_channel(x):
    # 逐通道计算scale (卷积权重的形状: [out_c, in_c, kH, kW])
    max_vals = torch.amax(torch.abs(x.detach()), dim=(-1), keepdim=True)
    scales = max_vals / 7.0
    scales = torch.clamp(scales, min=1e-8)
    
    # 量化 & 反量化
    q_x = torch.clamp(torch.round(x / scales), -7, 7)
    x_dequant = q_x * scales
    
    return x_dequant

def fake_svdquant_weight(weight, rank=64):
    dtype = weight.dtype
    device = weight.device
    #ori_weight = weight
    #u, s, vh = torch.linalg.svd(weight.cpu().float())
    u, s, vh = torch.linalg.svd(weight.float())
    us = u[:, : rank] * s[: rank]
    vh = vh[: rank]
    lora = torch.mm(us, vh)
    #residual_ori = weight.cpu().float() - lora
    residual_ori = weight.float() - lora
    residual = fake_quant_per_channel(residual_ori)
    quant_residual = residual_ori - residual
    quant_w = torch.mm(us, vh)
    quant = torch.norm(quant_residual) < 1000
    quant = quant.item()
    if quant:
        return True, us.to(dtype).to(device), vh.to(dtype).to(device), residual.to(dtype).to(device)
    else:
        return False, None, None, None

def int4_quantize_deepgemm_with_padding(x, int4_max=7.0, eps=1e-3):
    dtype = x.dtype
    ndim = x.ndim
    if ndim == 3:
        x = x.squeeze(0)
    m, n = x.shape
    x_padded = torch.zeros((m, ceil_div(n, 256) * 256), dtype=x.dtype, device=x.device)
    x_padded[:m,:n] = x
    x_padded_view = x_padded.view(m, -1, 128)
    x_amax = x_padded_view.abs().float().amax(dim=2).view(m, -1).clamp(eps).unsqueeze(2)
    x_padded_view = torch.clamp(torch.round(x_padded_view * (int4_max / x_amax)), -7, 7)
    x_dequant = (x_padded_view * (x_amax / int4_max)).view(m, -1)
    x_dequant = x_dequant[:m, :n]
    if ndim == 3:
        x_dequant = x_dequant.unsqueeze(0)
    return x_dequant.to(dtype)
    #return x_padded_view, (x_amax / int4_max).view(m, -1)

class INT4Linear_svdquant(nn.Module):
    def __init__(self, ori_linear: nn.Linear):
        super().__init__()
        assert isinstance(ori_linear, nn.Linear)
        self.in_features = ori_linear.in_features
        self.out_features = ori_linear.out_features
        self.bias = ori_linear.bias

        self.n = ori_linear.weight.shape[0]
        self.k = ori_linear.weight.shape[1]

        ### svdquant weight
        use_quant, us, vh, residual = fake_svdquant_weight(ori_linear.weight)
        self.use_quant = use_quant
        if self.use_quant:
            self.us = nn.Parameter(us, requires_grad=False)
            self.vh = nn.Parameter(vh, requires_grad=False)
            self.residual = nn.Parameter(residual, requires_grad=False)
            del ori_linear.weight
        else:
            self.weight = ori_linear.weight
        print('convert int4 weight, use_quant: ', self.use_quant)

    def forward(self, x):
        if self.use_quant:
            weight = torch.mm(self.us, self.vh)
            l_out = torch.nn.functional.linear(x, weight, self.bias)
            fq_x = int4_quantize_deepgemm_with_padding(x)
            r_out = torch.nn.functional.linear(fq_x, self.residual)
            l_out = l_out + r_out
        else:
            l_out = torch.nn.functional.linear(x, self.weight, self.bias)
        return l_out

def set_int4_layer(model):
    for block in model.single_blocks:
        for name, module in block.named_children():
            if name == "linear1" or name == "linear2":
                wrapped_module = INT4Linear_svdquant(module)
                setattr(block, name, wrapped_module)
    for block in model.double_blocks:
        for name, module in block.named_children():
            if name == "img_mlp":
                for subname, submodule in module.named_children():
                    if subname == "fc1" or subname == "fc2":
                        wrapped_submodule = INT4Linear_svdquant(submodule)
                        setattr(module, subname, wrapped_submodule)
            elif name == "img_attn_qkv":
                wrapped_module = INT4Linear_svdquant(module)
                setattr(block, name, wrapped_module)
            elif name == "img_attn_proj":
                wrapped_module = INT4Linear_svdquant(module)
                setattr(block, name, wrapped_module)

########################################################
# TensorRT-LLM Linear
########################################################

class DefaultLinear:
    @nvtx.annotate(message="DefaultLinear.__call__", color="yellow")
    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        input_scale: torch.Tensor = None,
        weight_scale: torch.Tensor = None,
        weight_global_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        return F.linear(input, weight, bias)


class TrtllmNVFp4BlockLinear:
    def __init__(self):
        try:
            import tensorrt_llm  # noqa
        except ImportError:
            raise ImportError("TensorRT-LLM is not installed.")
        
        self.scaling_vector_size = 16
        self.alpha = 0.5
        self.online_quantize = True
        self.trtllm_tuned = False

    @nvtx.annotate(message="TrtllmNVFp4BlockLinear.__call__", color="yellow")
    # nvFP4 GEMM only accepts bfloat16 inputs
    @torch.amp.custom_fwd(cast_inputs=torch.bfloat16, device_type="cuda")
    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_global_scale: torch.Tensor,
    ) -> torch.Tensor:
        import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils

        origin_dtype = input.dtype
        act_global_scale = 448.0 * 6.0 / input.abs().amax().float()
        alpha = 1.0 / (weight_global_scale * act_global_scale)
        if input.dim() == 3:
            act_fp4, act_sf = torch.ops.trtllm.fp4_batched_quantize(input, act_global_scale, self.scaling_vector_size, False)
            if not self.trtllm_tuned:
                with torch.inference_mode():
                    output = torch.ops.trtllm.fp4_bmm(
                        act_fp4,
                        weight.unsqueeze(0),
                        act_sf,
                        weight_scale.unsqueeze(0),
                        alpha,
                        fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4,
                        out_dtype=origin_dtype,
                    )
                self.trtllm_tuned = True
            else:
                output = torch.ops.trtllm.fp4_bmm(
                    act_fp4,
                    weight.unsqueeze(0),
                    act_sf,
                    weight_scale.unsqueeze(0),
                    alpha,
                    fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4,
                    out_dtype=origin_dtype,
                )
        else:
            act_fp4, act_sf = torch.ops.trtllm.fp4_quantize(input, act_global_scale, self.scaling_vector_size, False)
            output = torch.ops.trtllm.nvfp4_gemm(
                act_fp4, weight, act_sf, weight_scale, alpha, output_dtype=origin_dtype
            )

        if bias is not None:
            output = output + bias

        return output


class TrtllmFp4SvdquantLinear:
    def __init__(self):
        try:
            import tensorrt_llm  # noqa
        except ImportError:
            raise ImportError("TensorRT-LLM is not installed.")

        self.scaling_vector_size = 16
        self.alpha = 0.5
        self.online_quantize = True
        self.trtllm_tuned = False
        self.us = None
        self.vh = None
    
    @nvtx.annotate(message="TrtllmFp4SvdquantLinear.__call__", color="yellow")
    # nvFP4 GEMM only accepts bfloat16 inputs
    @torch.amp.custom_fwd(cast_inputs=torch.bfloat16, device_type="cuda")
    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_global_scale: torch.Tensor,
    ) -> torch.Tensor:
        import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils

        # Low-rank branch GEMM
        assert self.us is not None and self.vh is not None, "parameters us and vh are not initialized"
        input_shape = input.shape
        input = input.reshape(-1, input.shape[-1])
        lr_out = torch.matmul(input, self.vh.to(input.dtype).T)
        lr_out = torch.matmul(lr_out, self.us.to(input.dtype).T)
        lr_out = lr_out.reshape(*input_shape[:-1], -1)
        if bias is not None:
            lr_out = lr_out + bias

        # nvFP4 GEMM
        origin_dtype = input.dtype
        act_global_scale = 448.0 * 6.0 / input.abs().amax().float()
        alpha = 1.0 / (weight_global_scale * act_global_scale)
        if input.dim() == 3:
            act_fp4, act_sf = torch.ops.trtllm.fp4_batched_quantize(input, act_global_scale, self.scaling_vector_size, False)
            if not self.trtllm_tuned:
                with torch.inference_mode():
                    residual_out = torch.ops.trtllm.fp4_bmm(
                        act_fp4,
                        weight.unsqueeze(0),
                        act_sf,
                        weight_scale.unsqueeze(0),
                        alpha,
                        fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4,
                        out_dtype=origin_dtype,
                    )
                self.trtllm_tuned = True
            else:
                residual_out = torch.ops.trtllm.fp4_bmm(
                    act_fp4,
                    weight.unsqueeze(0),
                    act_sf,
                    weight_scale.unsqueeze(0),
                    alpha,
                    fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4,
                    out_dtype=origin_dtype,
                )
        else:
            act_fp4, act_sf = torch.ops.trtllm.fp4_quantize(input, act_global_scale, self.scaling_vector_size, False)
            residual_out = torch.ops.trtllm.nvfp4_gemm(
                act_fp4, weight, act_sf, weight_scale, alpha, output_dtype=origin_dtype
            )

        output = lr_out + residual_out

        return output


class TrtllmFp8BlockLinear:
    def __init__(self):
        try:
            import tensorrt_llm  # noqa

            pass  # noqa
        except ImportError:
            raise ImportError("TensorRT-LLM is not installed.")

    @nvtx.annotate(message="TrtllmFp8BlockLinear.__call__", color="yellow")
    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_global_scale: torch.Tensor = None,
    ) -> torch.Tensor:

        # input
        origin_shape = input.shape
        origin_dim = input.dim()
        origin_dtype = input.dtype
        if origin_dtype != torch.bfloat16:
            logger.warning(f"Input dtype is {origin_dtype}, forcing the input & output to bfloat16")
        input = input.to(torch.bfloat16)

        if input.dim() > 2:
            input = input.reshape(-1, input.shape[-1])

        act_input_fp8, input_scale = torch.ops.trtllm.fp8_quantize_1x128(input)
        output = torch.ops.trtllm.fp8_block_scaling_gemm(act_input_fp8, weight, input_scale, weight_scale)

        if bias is not None:
            output = output + bias
        if output.dim() == 2 and origin_dim == 3:
            output = output.reshape(origin_shape[0], origin_shape[1], -1)
        return output


class TrtllmFp8PerTensorLinear:
    def __init__(self):
        try:
            import tensorrt_llm  # noqa

            pass  # noqa
        except ImportError:
            raise ImportError("TensorRT-LLM is not installed.")

    @nvtx.annotate(message="TrtllmFp8PerTensorLinear.__call__", color="yellow")
    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_global_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        origin_dim = input.dim()
        origin_shape = input.shape
        origin_dtype = input.dtype
        input = input.to(torch.bfloat16)

        # Dynamic quantization
        qinput, cur_input_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(input)
        cur_input_scale = cur_input_scale.to(torch.float32)
        # This op does not support bias now.
        if qinput.dim() == 3:
            qinput = qinput.reshape(-1, qinput.shape[-1])

        # This op does not support bias now.
        if qinput.shape[0] <= 8:
            # use cuda core for small m dimension
            output = torch.ops.trtllm.cuda_scaled_mm(
                qinput,
                weight,
                scale_a=cur_input_scale,
                scale_b=weight_scale,
                bias=None,
                out_dtype=input.dtype,
            )
        else:
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
        if output.dim() == 2 and origin_dim == 3:
            output = output.reshape(origin_shape[0], origin_shape[1], -1)
        return output


class VflyLinear(torch.nn.Linear):
    def __init__(self, linear_type: str = "default", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = None  # module name in the model
        self.linear_impl = None
        self.input_scale = None
        self.weight_scale = None
        self.weight_global_scale = None
        self.linear_type = linear_type
        self.low_rank_weights = None

    @nvtx.annotate(message="VflyLinear.create_nvfp4_blockwise_quantized_weight", color="green")
    def create_nvfp4_blockwise_quantized_weight(
        self,
        param_value: torch.Tensor,
        block_size: int = 128,
    ):
        try:
            import tensorrt_llm  # noqa
        except ImportError:
            raise ImportError("TensorRT-LLM is not installed.")
        import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
        vec_size = 16

        global_max_abs = torch.amax(torch.abs(param_value))
        weight_global_scale = 448.0 * 6.0 / global_max_abs.float()
        weight_fp4, weight_scale = torch.ops.trtllm.fp4_quantize(param_value, weight_global_scale, vec_size, False)
        return weight_fp4.to(fp4_utils.float4_e2m1x2), weight_scale.to(fp4_utils.float4_sf_dtype), weight_global_scale

    @nvtx.annotate(message="VflyLinear.create_nvfp4_svdquant_quantized_weight", color="green")
    def create_nvfp4_svdquant_quantized_weight(
            self, 
            param_value: torch.Tensor,
            rank: int = 64):
        u, s, vh = torch.linalg.svd(param_value.float())
        us = (u[:, : rank] * s[: rank]).to(param_value.dtype)
        vh = vh[: rank].to(param_value.dtype)
        lora = torch.mm(us, vh)
        residual_fp32 = param_value.float() - lora
        residual_bf16 = residual_fp32.to(torch.bfloat16)
        residual_fp4, residual_scale, residual_global_scale = self.create_nvfp4_blockwise_quantized_weight(residual_bf16)
        return us, vh, residual_fp4, residual_scale, residual_global_scale

    @nvtx.annotate(message="VflyLinear.create_fp8_blockwise_quantized_weight", color="green")
    def create_fp8_blockwise_quantized_weight(
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

    @nvtx.annotate(message="VflyLinear.create_fp8_per_tensor_quantized_weight", color="green")
    def create_fp8_per_tensor_quantized_weight(self, param_value: torch.Tensor):
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

    @nvtx.annotate(message="VflyLinear.select_linear_impl", color="green")
    def select_linear_impl(self):
        # select linear implementation
        if self.linear_type == "trtllm-fp8-blockwise":
            self.linear_impl = TrtllmFp8BlockLinear()
        elif self.linear_type == "trtllm-fp8-per-tensor":
            self.linear_impl = TrtllmFp8PerTensorLinear()
        elif self.linear_type == "trtllm-nvfp4-blockwise":
            self.linear_impl = TrtllmNVFp4BlockLinear()
        elif self.linear_type == "trtllm-nvfp4-svdquant":
            self.linear_impl = TrtllmFp4SvdquantLinear()
        else:
            self.linear_impl = DefaultLinear()

        weight_name = self.linear_type + "_weight"
        weight_scale_name = self.linear_type + "_weight_scale"
        weight_global_scale_name = self.linear_type + "_weight_global_scale"
        # compute quantized weight and weight scale if needed
        if self.linear_type == "trtllm-fp8-blockwise":
            if not hasattr(self, weight_name) or not hasattr(self, weight_scale_name):
                if self.weight.shape[-1] % 128 != 0 or self.weight.shape[-2] % 128 != 0:
                    self.linear_type = "default"
                    self.linear_impl = DefaultLinear()
                    return self.weight, None, None
                weight, weight_scale = self.create_fp8_blockwise_quantized_weight(self.weight)
                self.register_parameter(weight_name, torch.nn.Parameter(weight))
                self.register_buffer(weight_scale_name, weight_scale)
        elif self.linear_type == "trtllm-fp8-per-tensor":
            if not hasattr(self, weight_name) or not hasattr(self, weight_scale_name):
                weight, weight_scale = self.create_fp8_per_tensor_quantized_weight(self.weight)
                self.register_parameter(weight_name, torch.nn.Parameter(weight))
                self.register_buffer(weight_scale_name, weight_scale)
        elif self.linear_type == "trtllm-nvfp4-blockwise":
            self.scaling_vector_size = self.linear_impl.scaling_vector_size
            assert self.in_features % self.scaling_vector_size == 0, f"in_features {self.in_features} must be divisible by scaling_vector_size {self.scaling_vector_size}"
            if not hasattr(self, weight_name) or not hasattr(self, weight_scale_name):
                weight, weight_scale, weight_global_scale = self.create_nvfp4_blockwise_quantized_weight(self.weight)
                self.register_parameter(weight_name, torch.nn.Parameter(weight, requires_grad=False))
                self.register_buffer(weight_scale_name, weight_scale)
                self.register_buffer(weight_global_scale_name, weight_global_scale)
        elif self.linear_type == "trtllm-nvfp4-svdquant":
            self.scaling_vector_size = self.linear_impl.scaling_vector_size
            assert self.in_features % self.scaling_vector_size == 0, f"in_features {self.in_features} must be divisible by scaling_vector_size {self.scaling_vector_size}"
            if not hasattr(self, weight_name) or not hasattr(self, weight_scale_name):
                us, vh, residual_fp4, residual_scale, residual_global_scale = self.create_nvfp4_svdquant_quantized_weight(self.weight)
                self.register_parameter(weight_name + "_us", torch.nn.Parameter(us.to(torch.float16), requires_grad=False))
                self.register_parameter(weight_name + "_vh", torch.nn.Parameter(vh.to(torch.float16), requires_grad=False))
                self.register_parameter(weight_name, torch.nn.Parameter(residual_fp4, requires_grad=False))
                self.register_buffer(weight_scale_name, residual_scale)
                self.register_buffer(weight_global_scale_name, residual_global_scale)

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
                elif key.endswith("_weight_global_scale") and key != weight_global_scale_name:
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
            weight_global_scale = None
        elif self.linear_type == "trtllm-nvfp4-blockwise":
            weight = getattr(self, weight_name)
            weight_scale = getattr(self, weight_scale_name)
            weight_global_scale = getattr(self, weight_global_scale_name)
        elif self.linear_type == "trtllm-nvfp4-svdquant":
            weight = getattr(self, weight_name)
            weight_scale = getattr(self, weight_scale_name)
            weight_global_scale = getattr(self, weight_global_scale_name)
            self.us = getattr(self, weight_name + "_us")
            self.vh = getattr(self, weight_name + "_vh")
            self.linear_impl.us = self.us
            self.linear_impl.vh = self.vh
        else:
            weight = getattr(self, weight_name)
            weight_scale = getattr(self, weight_scale_name)
            weight_global_scale = None

        return weight, weight_scale, weight_global_scale

    @nvtx.annotate(message="VflyLinear.forward", color="red")
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight, weight_scale, weight_global_scale = self.select_linear_impl()
        return self.linear_impl(input, weight, self.bias, self.input_scale, weight_scale, weight_global_scale)

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


@nvtx.annotate(message="replace_linear_layer", color="red")
def replace_linear_layer(model, quant_gemm_type="svdquant.int4", enable_hadamard=False, quant_recipe=default_wan22_recipe):
    # Replace all the linear layers in the model with VflyLinear layers
    major_quantized_linear_count = 0
    minor_quantized_linear_count = 0
    default_linear_count = 0
    default_linear_fn = partial(VflyLinear.from_linear, linear_type="default")
    if quant_gemm_type == "svdquant.int4":
        quant_linear_fn = INT4Linear_svdquant
    elif quant_gemm_type == "nvfp4":
        quant_linear_fn = partial(VflyLinear.from_linear, linear_type="trtllm-nvfp4-blockwise")
    elif quant_gemm_type == "svdquant.nvfp4":
        quant_linear_fn = partial(VflyLinear.from_linear, linear_type="trtllm-nvfp4-svdquant")
    elif quant_gemm_type == "trtllm-fp8-blockwise":
        quant_linear_fn = partial(VflyLinear.from_linear, linear_type="trtllm-fp8-blockwise")
    elif quant_gemm_type == "trtllm-fp8-per-tensor":
        quant_linear_fn = partial(VflyLinear.from_linear, linear_type="trtllm-fp8-per-tensor")
    elif quant_gemm_type == "nvfp4+trtllm-fp8-blockwise":
        quant_linear_fn = partial(VflyLinear.from_linear, linear_type="trtllm-nvfp4-blockwise")
        quant_linear2_fn = partial(VflyLinear.from_linear, linear_type="trtllm-fp8-blockwise")
    else:
        raise ValueError(f"Invalid quant_gemm_type: {quant_gemm_type}")

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            subnames = name.split(".")
            should_major_quantize = should_minor_quantize = False
            if quant_gemm_type in ["nvfp4", "svdquant.nvfp4", "svdquant.int4", "nvfp4+trtllm-fp8-blockwise"]:
                should_major_quantize = (
                    any(x in subnames for x in ['to_q', 'to_k', 'to_v']) or
                    any(x == "ffn" for x in subnames)
                )
                if quant_gemm_type == "nvfp4+trtllm-fp8-blockwise" and not should_major_quantize:
                    should_minor_quantize = module.weight.shape[-1] % 128 == 0 and module.weight.shape[-2] % 128 == 0
            elif quant_gemm_type in ["trtllm-fp8-blockwise"]:
                should_major_quantize = module.weight.shape[-1] % 128 == 0 and module.weight.shape[-2] % 128 == 0
            elif quant_gemm_type in ["trtllm-fp8-per-tensor"]:
                should_major_quantize = True
            else:
                pass

            if should_major_quantize:
                wrapped_module = quant_linear_fn(module)
                major_quantized_linear_count += 1
            elif should_minor_quantize:
                wrapped_module = quant_linear2_fn(module)
                minor_quantized_linear_count += 1
            else:
                wrapped_module = default_linear_fn(module)
                default_linear_count += 1
            # To get parent, walk subnames[:-1] carefully
            parent = model
            for p in subnames[:-1]:
                parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
            setattr(parent, subnames[-1], wrapped_module)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank == 0:
        if '+' in quant_gemm_type:
            logger.info(f"Have replaced {major_quantized_linear_count} quantized layers with gemm type {quant_gemm_type.split('+')[0]}")
            logger.info(f"Have replaced {minor_quantized_linear_count} quantized layers with gemm type {quant_gemm_type.split('+')[1]}")
        else:
            logger.info(f"Have replaced {major_quantized_linear_count} quantized layers with gemm type {quant_gemm_type}")
        logger.info(f"Remained {default_linear_count} default layers")
    return model