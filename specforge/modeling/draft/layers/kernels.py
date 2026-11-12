"""Opt-in Liger RMSNorm and SwiGLU, and the routed-expert kernel; checkpoints unchanged."""

from .mlp.dense import DenseFFN
from .mlp.moe import EXPERT_KERNELS, MoEFFN, SharedExpert
from .norm.rms import RMSNorm


def load_liger_kernels():
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction

    return LigerRMSNormFunction.apply, LigerSiLUMulFunction.apply


def configure_kernels(model, *, use_liger=False, moe_experts_kernel="grouped_mm"):
    """Resolve each kernel choice once at startup; an unusable choice fails, never falls back."""
    if moe_experts_kernel not in EXPERT_KERNELS:
        raise ValueError(
            f"unknown moe_experts_kernel {moe_experts_kernel!r}; "
            f"expected one of {sorted(EXPERT_KERNELS)}"
        )
    for module in model.modules():
        if not isinstance(module, MoEFFN):
            continue
        try:
            module.experts.probe_kernel(moe_experts_kernel)
        except RuntimeError as error:
            weight = module.experts.gate_up_proj
            hint = (
                ""
                if moe_experts_kernel == "loop"
                else "; set model.moe_experts_kernel: loop"
            )
            raise ValueError(
                f"moe_experts_kernel {moe_experts_kernel!r} cannot run a MoE layer "
                f"(hidden_size {weight.shape[-1]}, intermediate_size "
                f"{module.config.intermediate_size}, {weight.dtype} on "
                f"{weight.device.type}): {error}{hint}"
            ) from error
        module.experts_kernel = moe_experts_kernel
    if not use_liger:
        for module in model.modules():
            if isinstance(module, RMSNorm):
                module.liger_norm = None
            elif isinstance(module, (DenseFFN, SharedExpert)):
                module.liger_swiglu = None
        return
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
