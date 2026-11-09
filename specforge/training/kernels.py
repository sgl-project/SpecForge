"""Select runtime kernels without replacing modules or changing checkpoint keys."""

from specforge.modeling.draft.layers.mlp import DenseFFN
from specforge.modeling.draft.layers.norm import RMSNorm


def resolve_liger(requested, *, activations, device, default_enabled):
    if requested is not None and type(requested) is not bool:
        raise ValueError("use_liger_kernel must be boolean or null")
    supported_activations = all(value in ("silu", "swish") for value in activations)
    enabled = (
        requested
        if requested is not None
        else (default_enabled and device == "cuda" and supported_activations)
    )
    if enabled:
        if device not in ("cuda", "npu", "xpu") or not supported_activations:
            raise ValueError("Liger requires a Triton accelerator and silu/swish FFNs")
        try:
            from liger_kernel.ops.rms_norm import LigerRMSNormFunction  # noqa: F401
            from liger_kernel.ops.swiglu import LigerSiLUMulFunction  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "use_liger_kernel resolved to true but Liger is unavailable; "
                "install the project's liger-kernel dependency or set it to false"
            ) from exc
    return enabled


def configure_liger(model, requested, *, default_enabled):
    ffn = [module for module in model.modules() if isinstance(module, DenseFFN)]
    enabled = resolve_liger(
        requested,
        activations=[m.config.hidden_act for m in ffn],
        device=model.embed_tokens.weight.device.type,
        default_enabled=default_enabled,
    )
    for module in model.modules():
        if isinstance(module, (DenseFFN, RMSNorm)):
            module.use_liger_kernel = enabled
    return enabled
