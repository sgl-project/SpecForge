"""Optional token kernels; mathematical loss registration stays independent."""

import torch

from specforge.core import fused_loss

from .implementations import GroupResult, register_implementation
from .losses import _distillation, _lk


def ce_l1_terms(logits, labels, *, fused=False, **options):
    optimized = (
        fused
        and logits.is_cuda
        and logits.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and logits.numel() > 0
    )
    compute = fused_loss.fused_ce_l1_loss if optimized else fused_loss.ce_l1_loss
    return compute(logits, labels, **options)


def _ce(logits, labels, teacher=None):
    return ce_l1_terms(logits, labels, fused=True).cross_entropy


def _ce_l1(logits, labels, teacher, *, ce_weight=0.1, l1_weight=0.9):
    terms = ce_l1_terms(logits, labels, teacher_logits=teacher, fused=True)
    return ce_weight * terms.cross_entropy + l1_weight * terms.teacher_l1


def register_token_kernel(kind, compute):
    def supports(request, group):
        p = request.predictions
        if len(group) != 1 or request.terms[group[0]].loss.kind != kind:
            return "requires one matching token loss"
        if not p.hidden.is_cuda or p.hidden.numel() == 0:
            return "requires nonempty CUDA inputs"
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
        # A projection may promote dtype. Preserve the loss reference outside
        # the kernel's supported dtype domain.
        if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            values = loss(
                logits, request.labels, teacher=request.teacher, context=inputs.context
            )
        else:
            kwargs = {"context": inputs.context} if loss.registration.contextual else {}
            values = compute(
                logits, request.labels, request.teacher, **kwargs, **loss.options
            )
        return GroupResult({index: (values * inputs.weights).sum()})

    return execute


register_token_kernel("ce", _ce)
register_token_kernel("ce_l1", _ce_l1)
register_token_kernel("soft_ce", _distillation(fused_loss.fused_soft_ce_loss))
for _kind, _compute in {
    "kl": fused_loss.fused_kl_div_loss,
    "reverse_kl": fused_loss.fused_reverse_kl_div_loss,
    "js": fused_loss.fused_js_div_loss,
    "tv": fused_loss.fused_tv_loss,
    "nla": fused_loss.fused_nla_loss,
    "lk": fused_loss.fused_lk_hybrid_loss,
}.items():
    register_token_kernel(_kind, _distillation(_compute))
for _mode in ("alpha", "tv", "lambda"):
    register_token_kernel("lk_" + _mode, _lk(_mode, ce_compute=_ce))
