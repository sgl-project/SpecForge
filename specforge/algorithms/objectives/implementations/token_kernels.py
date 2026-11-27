"""Optional token kernels; mathematical loss registration stays independent."""

import torch

from specforge.losses import fused as fused_loss

from ..losses import (
    LossOutput,
    ce_l1_output_options,
    ce_l1_outputs,
    ce_output_options,
    ce_outputs,
    distillation_loss,
    lk_loss,
)
from .reference import result_group
from .registry import register_implementation

_DEVICE_LABELS = {"cuda": "CUDA/ROCm", "npu": "NPU"}


def _ce(logits, labels, teacher=None, **requested_outputs):
    return ce_outputs(
        fused_loss.fused_ce_l1_loss,
        logits,
        labels,
        teacher,
        **ce_output_options(requested_outputs),
    )


def _ce_l1(
    logits,
    labels,
    teacher,
    *,
    context=None,
    ce_weight=0.1,
    l1_weight=0.9,
    **requested_outputs,
):
    return ce_l1_outputs(
        fused_loss.fused_ce_l1_loss,
        logits,
        labels,
        teacher,
        context=context,
        ce_weight=ce_weight,
        l1_weight=l1_weight,
        **ce_l1_output_options(requested_outputs),
    )


def _require_triton(kind):
    def check(tensor):
        if not fused_loss.triton_available(tensor.device):
            raise ImportError(f"fused {kind} requires Triton")

    return check


def register_token_kernel(kind, compute, *, devices=("cuda",), check_dependency=None):
    """Adapt a per-position kernel to the group execution registry.

    Args:
        kind: Mathematical loss name matched by this implementation.
        compute: Callable(logits, labels, teacher, **options) returning losses
            shaped like labels, or a LossOutput with reusable statistics.
            Contextual losses additionally receive context.
        devices: Supported PyTorch device types for this kernel.
        check_dependency: Optional callable(tensor) verifying kernel-specific
            optional runtime dependencies (returning None or raising ImportError).

    Returns:
        The registered group executor under the name "token_" + kind. It lazily
        projects outputs and applies position weights, but not coefficients or
        denominator normalization.

    Raises:
        ValueError: The execution implementation name is already registered.
    """
    device_label = "/".join(_DEVICE_LABELS.get(d, d.upper()) for d in devices)

    def supports(request, group):
        tensor = request.input_tensor
        if len(group) != 1 or request.terms[group[0]].loss.kind != kind:
            return "requires one matching token loss"
        if request.requested_outputs:
            if kind == "ce":
                ce_output_options(request.requested_outputs)
            elif kind == "ce_l1":
                ce_l1_output_options(request.requested_outputs)
            else:
                return "token kernel does not provide requested extra outputs"
        if (
            kind in ("lk_alpha", "lk_tv", "lk_lambda")
            and request.terms[group[0]].loss.options.get("target") == "teacher"
        ):
            return "teacher LK uses the reference formula"
        if tensor.device.type not in devices:
            return f"requires {device_label} inputs"
        if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return "requires FP16/BF16/FP32"
        if check_dependency is not None:
            return check_dependency(tensor)
        return None

    @register_implementation("token_" + kind, supports=supports, losses=(kind,))
    def execute(request, group):
        index = group[0]
        inputs = request.terms[index]
        logits = request.tensor(index)
        loss = inputs.loss
        if loss.requires_teacher and (
            request.teacher is None or request.teacher.shape != logits.shape
        ):
            raise ValueError("distillation requires aligned teacher and draft logits")
        if logits.device.type not in devices or logits.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError(
                f"fused token loss requires {device_label} FP16/BF16/FP32 logits"
            )
        kwargs = {"context": inputs.context} if loss.registration.contextual else {}
        result = compute(
            logits,
            request.labels,
            request.teacher,
            **kwargs,
            **loss.options,
            **request.requested_outputs,
        )
        if not isinstance(result, LossOutput):
            result = LossOutput(result)
        return result_group(request, index, result)

    return execute


register_token_kernel("ce", _ce, check_dependency=_require_triton("ce"))
register_token_kernel("ce_l1", _ce_l1, check_dependency=_require_triton("ce_l1"))
for _kind, _compute in {
    "soft_ce": fused_loss.fused_soft_ce_loss,
    "kl": fused_loss.fused_kl_div_loss,
    "reverse_kl": fused_loss.fused_reverse_kl_div_loss,
    "js": fused_loss.fused_js_div_loss,
    "tv": fused_loss.fused_tv_loss,
    "nla": fused_loss.fused_nla_loss,
    "lk": fused_loss.fused_lk_hybrid_loss,
}.items():
    register_token_kernel(
        _kind,
        distillation_loss(_compute),
        devices=("cuda", "npu"),
        check_dependency=_require_triton(_kind),
    )
for _mode in ("alpha", "tv", "lambda"):
    register_token_kernel(
        "lk_" + _mode,
        lk_loss(_mode, ce_compute=lambda logits, labels: _ce(logits, labels).values),
        check_dependency=_require_triton("lk_" + _mode),
    )
