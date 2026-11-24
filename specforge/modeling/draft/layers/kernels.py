"""Optional Liger RMSNorm/SwiGLU and Triton convolution kernels, and the routed-expert kernel.

None of them changes checkpoint parameters.
"""

import logging

LIGER_REQUIREMENT = "liger-kernel>=0.8.3,<0.9"

import torch

from .conv.grouped_causal import GroupedCausalConv
from .mlp.dense import DenseFFN
from .mlp.moe import MoEFFN, SharedExpert
from .norm.rms import RMSNorm


def liger_hidden_act(model):
    """Activation Liger's SwiGLU has to implement for this draft's dense FFNs and shared experts."""
    activations = sorted(
        {
            m.config.hidden_act
            for m in model.modules()
            if isinstance(m, (DenseFFN, SharedExpert))
        }
    )
    unsupported = [name for name in activations if name not in ("silu", "swish")]
    # A draft whose FFNs are all routed experts leaves Liger only its RMSNorm.
    return unsupported[0] if unsupported else (activations or ["silu"])[0]


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

    MoE layers always run F.grouped_mm; every one is probed here, so widths it
    rejects (not multiples of 8 for bf16/fp16, 4 for fp32) fail at startup
    rather than fall back.

    Args:
        use_liger: Resolved Liger choice; True fails where Liger cannot run.
        use_fused_conv: True requires the Triton grouped convolution and fails
            where it cannot run; None uses it when the draft has grouped
            convolutions that all support it; False keeps the eager one.

    Returns:
        Whether the grouped convolutions use the Triton kernel.
    """
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
    for module in model.modules():
        if not isinstance(module, MoEFFN):
            continue
        try:
            module.experts.probe()
        except RuntimeError as error:
            weight = module.experts.gate_up_proj
            raise ValueError(
                f"F.grouped_mm cannot run a MoE layer (hidden_size {weight.shape[-1]}, "
                f"intermediate_size {module.config.experts.intermediate_size}, "
                f"{weight.dtype} on {weight.device.type}): {error}; hidden_size and "
                "intermediate_size must be multiples of 8 (bf16/fp16) or 4 (fp32)"
            ) from error

    if not use_liger:
        for module in model.modules():
            if isinstance(module, RMSNorm):
                module.liger_norm = None
            elif isinstance(module, (DenseFFN, SharedExpert)):
                module.liger_swiglu = None
        return fused_conv
    if any(p.device.type != "cuda" for p in model.parameters()):
        raise ValueError("Liger kernels require a CUDA/ROCm model")
    rms_norm, swiglu = load_liger_kernels()
    for module in model.modules():
        if isinstance(module, RMSNorm):
            # the Liger kernel scales by a weight; weightless norms stay on PyTorch
            module.liger_norm = rms_norm if module.weight is not None else None
        elif isinstance(
            module, (DenseFFN, SharedExpert)
        ) and module.config.hidden_act in {
            "silu",
            "swish",
        }:
            module.liger_swiglu = swiglu
    return fused_conv
