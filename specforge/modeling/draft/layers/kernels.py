"""Opt-in Liger RMSNorm and SwiGLU without changing checkpoint parameters."""

from .mlp.dense import DenseFFN
from .norm.rms import RMSNorm


def load_liger_kernels():
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction

    return LigerRMSNormFunction.apply, LigerSiLUMulFunction.apply


def configure_kernels(model, *, use_liger=False):
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
