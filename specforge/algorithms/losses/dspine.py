"""Chunked backbone, refinement and inter-layer alignment terms."""

import torch
from torch.nn import functional as F

from specforge.core.fused_loss import ce_l1_loss
from specforge.modeling.draft.bindings import BoundLMHead

from .blocks import logits_loss_terms


def dspine_chunk_terms(
    model,
    options,
    teacher_head,
    hidden: torch.Tensor,
    pre_injection: torch.Tensor,
    layer_features: torch.Tensor,
    labels: torch.Tensor,
    predecessors: torch.Tensor,
    teacher_hidden: torch.Tensor,
    weights: torch.Tensor,
    supervised: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    config = options
    with torch.no_grad():
        teacher_logits = teacher_head(teacher_hidden)
    terms = logits_loss_terms(
        options,
        model.lm_head(hidden),
        labels,
        teacher_logits=teacher_logits,
        top_k=config.top_k,
    )
    candidates = terms.topk_ids
    refined = model.refine(pre_injection, predecessors)
    head = model.lm_head
    candidate_weights = (
        head.candidate_weights(candidates)
        if isinstance(head, BoundLMHead)
        else F.embedding(candidates, head.weight)
    )
    scores = torch.einsum("bmh,bmkh->bmk", refined, candidate_weights).float()
    target_candidates = teacher_logits.softmax(dim=-1, dtype=torch.float32).gather(
        -1, candidates
    )
    # The gate uses full-vocabulary mass, before candidate normalization.
    mass = target_candidates.sum(dim=-1)
    covered = candidates == labels.unsqueeze(-1)
    # Normalize both candidate distributions through the same softmax path.
    refinement = ce_l1_loss(
        scores,
        covered.long().argmax(dim=-1),
        teacher_logits=teacher_logits.gather(-1, candidates),
    )
    # Missing labels use index zero above only as a placeholder for CE.
    refined_ce = refinement.cross_entropy * covered.any(dim=-1)
    refined_l1 = refinement.teacher_l1 * (mass > config.candidate_mass_threshold)
    reference_codes = F.embedding(labels, model.transfer_codes).float()
    alignment = 1 - F.cosine_similarity(
        layer_features.float(), reference_codes.unsqueeze(1), dim=-1
    )
    with torch.no_grad():
        correct = ((terms.argmax_ids == labels) * supervised).sum()
        coverage = (covered.any(dim=-1) * supervised).sum()
    return (
        (terms.cross_entropy * weights).sum(),
        (terms.teacher_l1 * weights).sum(),
        (
            (config.ce_weight * refined_ce + (1 - config.ce_weight) * refined_l1)
            * weights
        ).sum(),
        (alignment * weights.unsqueeze(1)).sum(),
        correct.float(),
        coverage.float(),
    )
