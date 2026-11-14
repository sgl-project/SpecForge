"""Optional token kernels; mathematical loss registration stays independent."""

import torch

from specforge.losses import fused as fused_loss

from ..losses import LossOutput, distillation_loss, lk_loss
from .registry import GroupResult, register_implementation


def ce_l1_terms(logits, labels, *, fused=False, selection=None, key=None, **options):
    """Execute the CE/L1 primitive supplied to auxiliary callbacks.

    Args:
        logits: Scores with vocabulary in the last dimension.
        labels: Token IDs aligned with the leading dimensions.
        fused: Executor policy; protocols do not choose the kernel.
        selection: Optional run-local implementation recorder.
        key: Stable identity of this auxiliary primitive in the prediction plan.
        **options: CE/L1 options, including teacher_logits and top_k.

    Returns:
        Unreduced CEL1Loss values. Empty inputs retain the zero-gradient path.
        Unsupported fused environments raise instead of switching implementations.
    """
    if not fused:
        return fused_loss.ce_l1_loss(logits, labels, **options)
    error = None
    if not fused_loss.supports_fused_ce_l1_loss(logits):
        error = "fused CE/L1 requires CUDA/ROCm FP16/BF16/FP32 logits and Triton"
    if selection is not None:
        selection.record(key, "fused_ce_l1", error=error)
    elif error is not None:
        raise ValueError(error)
    compute = fused_loss.fused_ce_l1_loss if logits.numel() else fused_loss.ce_l1_loss
    return compute(logits, labels, **options)


def _ce(logits, labels, teacher=None):
    return ce_l1_terms(logits, labels, fused=True).cross_entropy


def _ce_l1(logits, labels, teacher, *, ce_weight=0.1, l1_weight=0.9):
    terms = ce_l1_terms(logits, labels, teacher_logits=teacher, fused=True)
    return LossOutput(
        ce_weight * terms.cross_entropy + l1_weight * terms.teacher_l1,
        {"teacher_l1": terms.teacher_l1},
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
        p = request.predictions
        if len(group) != 1 or request.terms[group[0]].loss.kind != kind:
            return "requires one matching token loss"
        if not p.hidden.is_cuda:
            return "requires CUDA inputs"
        if p.hidden.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return "requires FP16/BF16/FP32"
        return None

    @register_implementation("token_" + kind, supports=supports)
    def execute(request, group):
        index = group[0]
        term, inputs = request.predictions.plan.terms[index], request.terms[index]
        logits = request.outputs()[term.output]
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
            logits, request.labels, request.teacher, **kwargs, **loss.options
        )
        if not isinstance(result, LossOutput):
            result = LossOutput(result)
        return GroupResult(
            {index: (result.values * inputs.weights).sum()},
            statistics={index: result.statistics},
        )

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
    register_token_kernel("lk_" + _mode, lk_loss(_mode, ce_compute=_ce))
