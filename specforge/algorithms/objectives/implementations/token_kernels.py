"""Optional token kernels; mathematical loss registration stays independent."""

import torch

from specforge.losses import fused as fused_loss

from ..losses import (
    LossOutput,
    ce_output_options,
    ce_outputs,
    combine_ce_l1,
    distillation_loss,
    lk_loss,
)
from .reference import result_group
from .registry import register_implementation


def _ce(logits, labels, teacher=None, **requested_outputs):
    return ce_outputs(
        fused_loss.fused_ce_l1_loss,
        logits,
        labels,
        teacher,
        **ce_output_options(requested_outputs),
    )


def _ce_l1(logits, labels, teacher, *, context=None, ce_weight=0.1, l1_weight=0.9):
    terms = fused_loss.fused_ce_l1_loss(logits, labels, teacher_logits=teacher)
    return combine_ce_l1(
        terms.cross_entropy, terms.teacher_l1, context, ce_weight, l1_weight
    )


def register_token_kernel(kind, compute):
    """Adapt a per-position kernel to the group execution registry.

    Args:
        kind: Mathematical loss name matched by this implementation.
        compute: Callable(logits, labels, teacher, **options) returning losses
            shaped like labels, or a LossOutput with reusable statistics.
            Contextual losses additionally receive context.

    Returns:
        The registered group executor under the name "token_" + kind. It lazily
        projects outputs and applies position weights, but not coefficients or
        denominator normalization.

    Raises:
        ValueError: The execution implementation name is already registered.
    """

    def supports(request, group):
        tensor = request.input_tensor
        if len(group) != 1 or request.terms[group[0]].loss.kind != kind:
            return "requires one matching token loss"
        if request.requested_outputs:
            if kind != "ce":
                return "token kernel does not provide requested extra outputs"
            ce_output_options(request.requested_outputs)
        if (
            kind in ("lk_alpha", "lk_tv", "lk_lambda")
            and request.terms[group[0]].loss.options.get("target") == "teacher"
        ):
            return "teacher LK uses the reference formula"
        if not tensor.is_cuda:
            return "requires CUDA inputs"
        if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return "requires FP16/BF16/FP32"
        if kind in ("ce", "ce_l1") and not fused_loss.supports_fused_ce_l1_loss(tensor):
            raise ValueError("fused CE/L1 requires Triton")
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
        if not logits.is_cuda or logits.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError(
                "fused token loss requires CUDA/ROCm FP16/BF16/FP32 logits"
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


register_token_kernel("ce", _ce)
register_token_kernel("ce_l1", _ce_l1)
register_token_kernel("soft_ce", distillation_loss(fused_loss.fused_soft_ce_loss))
for _kind, _compute in {
    "kl": fused_loss.fused_kl_div_loss,
    "reverse_kl": fused_loss.fused_reverse_kl_div_loss,
    "js": fused_loss.fused_js_div_loss,
    "tv": fused_loss.fused_tv_loss,
    "nla": fused_loss.fused_nla_loss,
    "lk": fused_loss.fused_lk_hybrid_loss,
}.items():
    register_token_kernel(_kind, distillation_loss(_compute))
for _mode in ("alpha", "tv", "lambda"):
    register_token_kernel(
        "lk_" + _mode,
        lk_loss(_mode, ce_compute=lambda logits, labels: _ce(logits, labels).values),
    )
