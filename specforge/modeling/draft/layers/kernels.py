"""Opt-in Liger RMSNorm and SwiGLU without changing checkpoint parameters."""

import logging

LIGER_REQUIREMENT = "liger-kernel>=0.8.3,<0.9"

import torch

from .conv.grouped_causal import GroupedCausalConv
from .mlp.dense import DenseFFN
from .norm.rms import RMSNorm


def load_liger_kernels():
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction

    return LigerRMSNormFunction.apply, LigerSiLUMulFunction.apply


def configure_kernels(model, *, use_liger=False, use_fused_conv=False):
    convolutions = [m for m in model.modules() if isinstance(m, GroupedCausalConv)]
    if use_fused_conv:
        if not convolutions:
            raise ValueError("use_fused_conv requires a grouped convolution")
        for module in convolutions:
            if not module.base_kernel.is_cuda or module.base_kernel.dtype not in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
            ):
                raise ValueError(
                    "use_fused_conv requires CUDA/ROCm FP16/BF16/FP32 weights"
                )
        from .conv.triton import supports_group_size

        for module in convolutions:
            if not supports_group_size(module.config.group_size):
                raise ValueError(
                    "use_fused_conv requires a power-of-two group_size <= 1024"
                )
    for module in convolutions:
        module.use_fused = use_fused_conv
    if convolutions:
        logging.getLogger(__name__).info(
            "Grouped convolution uses %s", "Triton" if use_fused_conv else "eager"
        )

    if not use_liger:
        for module in model.modules():
            if isinstance(module, RMSNorm):
                module.liger_norm = None
            elif isinstance(module, DenseFFN):
                module.liger_swiglu = None
        return
    if any(p.device.type != "cuda" for p in model.parameters()):
        raise ValueError("Liger kernels require a CUDA/ROCm model")
    rms_norm, swiglu = load_liger_kernels()
    for module in model.modules():
        if isinstance(module, RMSNorm):
            # the Liger kernel scales by a weight; weightless norms stay on PyTorch
            module.liger_norm = rms_norm if module.weight is not None else None
        elif isinstance(module, DenseFFN) and module.config.hidden_act in {
            "silu",
            "swish",
        }:
            module.liger_swiglu = swiglu
