"""Optional Liger RMSNorm/SwiGLU and Triton convolution kernels; KDA's FLA check.

None of them changes checkpoint parameters.
"""

import logging

LIGER_REQUIREMENT = "liger-kernel>=0.8.3,<0.9"

import torch

from .attention.kda import KDAAttention
from .attention.kda.kernels import load_fla_chunk_kda
from .conv.grouped_causal import GroupedCausalConv
from .mlp.dense import DenseFFN
from .norm.rms import RMSNorm


def liger_hidden_act(model):
    """Activation Liger's SwiGLU has to implement for this draft's dense FFNs."""
    activations = sorted(
        {m.config.hidden_act for m in model.modules() if isinstance(m, DenseFFN)}
    )
    unsupported = [name for name in activations if name not in ("silu", "swish")]
    return unsupported[0] if unsupported else (activations or ["none"])[0]


def load_liger_kernels():
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction

    return LigerRMSNormFunction.apply, LigerSiLUMulFunction.apply


def fused_conv_unsupported(convolutions):
    """Why the Triton grouped convolution cannot run on these modules, or None."""
    for module in convolutions:
        if not module.base_kernel.is_cuda or module.base_kernel.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            return "requires CUDA/ROCm FP16/BF16/FP32 weights"
    try:
        from .conv.triton import supports_group_size
    except ModuleNotFoundError as error:
        if (error.name or "").split(".")[0] != "triton":
            raise
        return "requires Triton"
    if not all(supports_group_size(m.config.group_size) for m in convolutions):
        return "requires a power-of-two group_size <= 1024"
    return None


def configure_kernels(model, *, use_liger=False, use_fused_conv=None):
    """Select the optional kernels of a built draft.

    Args:
        use_liger: Resolved Liger choice; True fails where Liger cannot run.
        use_fused_conv: True requires the Triton grouped convolution and fails
            where it cannot run; None uses it when the draft has grouped
            convolutions that all support it; False keeps the eager one.

    KDA layers have no kernel choice: CUDA runs FLA, so a missing FLA fails
    here rather than at the first forward.

    Returns:
        Whether the grouped convolutions use the Triton kernel.
    """
    if any(
        parameter.is_cuda
        for module in model.modules()
        if isinstance(module, KDAAttention)
        for parameter in module.parameters()
    ):
        load_fla_chunk_kda()
    convolutions = [m for m in model.modules() if isinstance(m, GroupedCausalConv)]
    if use_fused_conv is True and not convolutions:
        raise ValueError("use_fused_conv requires a grouped convolution")
    reason = (
        fused_conv_unsupported(convolutions)
        if convolutions and use_fused_conv is not False
        else None
    )
    if use_fused_conv is True and reason:
        raise ValueError(f"use_fused_conv {reason}")
    fused_conv = bool(convolutions) and use_fused_conv is not False and reason is None
    for module in convolutions:
        module.use_fused = fused_conv
    if convolutions:
        logging.getLogger(__name__).info(
            "Grouped convolution uses %s%s",
            "Triton" if fused_conv else "eager",
            f" (auto: {reason})" if reason and use_fused_conv is None else "",
        )

    if not use_liger:
        for module in model.modules():
            if isinstance(module, RMSNorm):
                module.liger_norm = None
            elif isinstance(module, DenseFFN):
                module.liger_swiglu = None
        return fused_conv
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
    return fused_conv
