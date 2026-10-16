"""Fused DSpark probability-distance loss and detached telemetry."""

from functools import lru_cache

import torch
from torch.autograd.function import once_differentiable


def _token_terms(draft_logits, target_logits):
    draft_probabilities = torch.softmax(draft_logits.float(), dim=-1)
    with torch.no_grad():
        target_probabilities = torch.softmax(target_logits.float(), dim=-1)
        teacher_ids = target_logits.argmax(dim=-1)
        teacher_top1 = target_probabilities.max(dim=-1).values
        predicted_ids = draft_logits.argmax(dim=-1)
        draft_top1 = draft_probabilities.max(dim=-1).values
    l1 = (draft_probabilities - target_probabilities).abs().sum(dim=-1)
    return l1, predicted_ids, teacher_ids, draft_top1, teacher_top1


@lru_cache(maxsize=1)
def _compiled_token_terms():
    return torch.compile(_token_terms, fullgraph=True)


def _l1_backward(draft_logits, target_logits, gradient):
    draft_probabilities = torch.softmax(draft_logits.float(), dim=-1)
    target_probabilities = torch.softmax(target_logits.float(), dim=-1)
    probability_gradient = (draft_probabilities - target_probabilities).sign()
    probability_gradient = probability_gradient * gradient.unsqueeze(-1)
    centered_gradient = probability_gradient - (
        probability_gradient * draft_probabilities
    ).sum(dim=-1, keepdim=True)
    return (centered_gradient * draft_probabilities).to(draft_logits.dtype)


@lru_cache(maxsize=1)
def _compiled_l1_backward():
    return torch.compile(_l1_backward, fullgraph=True)


@torch.library.custom_op("specforge::dspark_token_terms", mutates_args=())
def _fused_token_terms(
    draft_logits: torch.Tensor, target_logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _compiled_token_terms()(draft_logits, target_logits)


@_fused_token_terms.register_fake
def _fake_token_terms(draft_logits, target_logits):
    shape = draft_logits.shape[:-1]
    return tuple(
        torch.empty(shape, device=draft_logits.device, dtype=dtype)
        for dtype in (
            torch.float32,
            torch.int64,
            torch.int64,
            torch.float32,
            torch.float32,
        )
    )


@torch.library.custom_op("specforge::dspark_l1_backward", mutates_args=())
def _fused_l1_backward(
    draft_logits: torch.Tensor, target_logits: torch.Tensor, gradient: torch.Tensor
) -> torch.Tensor:
    return _compiled_l1_backward()(draft_logits, target_logits, gradient)


@_fused_l1_backward.register_fake
def _fake_l1_backward(draft_logits, target_logits, gradient):
    return torch.empty_like(draft_logits)


class _CompiledTokenTerms(torch.autograd.Function):
    """Recompute probabilities behind an opaque boundary for selective checkpointing."""

    @staticmethod
    def forward(ctx, draft_logits, target_logits):
        terms = _fused_token_terms(draft_logits, target_logits)
        ctx.save_for_backward(draft_logits, target_logits)
        ctx.mark_non_differentiable(*terms[1:])
        return terms

    @staticmethod
    @once_differentiable
    def backward(ctx, l1_gradient, *unused_gradients):
        draft_logits, target_logits = ctx.saved_tensors
        return _fused_l1_backward(draft_logits, target_logits, l1_gradient), None


def compiled_dspark_token_terms(draft_logits, target_logits):
    return _CompiledTokenTerms.apply(draft_logits, target_logits)
