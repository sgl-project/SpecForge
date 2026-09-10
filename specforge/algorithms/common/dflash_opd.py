"""DFlash2 draft-policy replay losses, independent of rollout transport.

The block objective follows the Draft-OPD TV variant. DFlash2 has a sparse
selector distribution; target mass outside its natural candidates is retained
implicitly by TV = 1 - overlap. The prefix product is a replay-path surrogate,
not an estimate of unconditional acceptance for a Markov selector.
"""
from __future__ import annotations

from typing import NamedTuple

import torch

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel


class OPDBlockTerms(NamedTuple):
    loss_sum: torch.Tensor
    block_count: torch.Tensor
    overlap_sum: torch.Tensor
    position_count: torch.Tensor
    prefix_survival_sum: torch.Tensor
    target_candidate_mass_sum: torch.Tensor


def first_rejection_mask(
    accepted_lengths: torch.Tensor,
    exposed_lengths: torch.Tensor,
    *,
    proposal_width: int = 7,
) -> torch.Tensor:
    """Select accepted proposals plus the first reject, respecting censoring.

    Lengths count draft proposals only, excluding the anchor and bonus token.
    exposed_lengths excludes positions beyond the request's EOS/length limit.
    accepted_lengths may exceed exposure when serving reports pre-censor counts.
    """
    if type(proposal_width) is not int or proposal_width < 1:
        raise ValueError("proposal_width must be a positive integer")
    if accepted_lengths.shape != exposed_lengths.shape:
        raise ValueError("Accepted and exposed lengths must have the same shape")
    for value in (accepted_lengths, exposed_lengths):
        if value.dtype not in (torch.int32, torch.int64):
            raise ValueError("Replay lengths must be integer tensors")
        if bool(((value < 0) | (value > proposal_width)).any()):
            raise ValueError("Replay length outside the proposal block")
    length = torch.minimum(accepted_lengths + 1, exposed_lengths)
    positions = torch.arange(proposal_width, device=length.device)
    return positions < length.unsqueeze(-1)


def sparse_overlap(
    selector_logits: torch.Tensor,
    target_candidate_probs: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact overlap against a target distribution with arbitrary outside mass.

    Inputs contain unique candidate IDs in matching order, checked by the replay
    adapter. DFlash2 serving applies temperature to selector scores and samples
    from all natural K candidates; target filtering is done by the verifier.
    """
    if selector_logits.shape != target_candidate_probs.shape or selector_logits.ndim < 2:
        raise ValueError("Matching [..., K] scores and teacher probabilities required")
    if not isinstance(temperature, (float, int)) or not 0 < temperature < float("inf"):
        raise ValueError("Temperature must be finite and positive")
    p = target_candidate_probs.detach().float()
    logits = selector_logits.float()
    if not bool(torch.isfinite(p).all()) or not bool(torch.isfinite(logits).all()):
        raise ValueError("Nonfinite replay probabilities or logits")
    mass = p.sum(dim=-1)
    if bool((p < 0).any()) or bool((mass > 1.00001).any()):
        raise ValueError("Teacher candidate probabilities must retain their original mass")
    q = torch.softmax(logits / temperature, dim=-1)
    return torch.minimum(p, q).sum(dim=-1), mass


def opd_block_terms(
    selector_logits: torch.Tensor,
    target_candidate_probs: torch.Tensor,
    position_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> OPDBlockTerms:
    """Return additive block statistics for a globally normalized update."""
    if selector_logits.ndim != 3 or position_mask.shape != selector_logits.shape[:-1]:
        raise ValueError("Expected [blocks, positions, candidates] and [blocks, positions]")
    if position_mask.dtype != torch.bool:
        raise ValueError("Replay position mask must be boolean")
    if bool((position_mask[:, 1:] & ~position_mask[:, :-1]).any()):
        raise ValueError("Replay positions must form a contiguous prefix")
    # Mask padding before evaluating it; arbitrary padding must have zero
    # contribution and zero gradient, including NaN padding from empty records.
    mask = position_mask.unsqueeze(-1)
    logits = torch.where(mask, selector_logits, torch.zeros_like(selector_logits))
    p = torch.where(mask, target_candidate_probs, torch.zeros_like(target_candidate_probs))
    overlap, mass = sparse_overlap(logits, p, temperature=temperature)
    survival = torch.cumprod(torch.where(position_mask, overlap, torch.ones_like(overlap)), dim=-1)
    survival = torch.where(position_mask, survival, torch.zeros_like(survival))
    lengths = position_mask.sum(dim=-1)
    active = lengths > 0
    block_loss = 1 - survival.sum(dim=-1) / lengths.clamp_min(1)
    loss_sum = torch.where(active, block_loss, torch.zeros_like(block_loss)).sum()
    return OPDBlockTerms(
        loss_sum, active.sum().float(), (overlap * position_mask).sum().detach(),
        lengths.sum().float(), survival.sum().detach(), (mass * position_mask).sum().detach(),
    )


def global_block_loss(
    terms: OPDBlockTerms, *, global_block_count: int, gradient_average_world_size: int
) -> torch.Tensor:
    """Scale a local SUM before a DDP/FSDP averaged gradient reduction.

    global_block_count includes the complete accumulation window, not just the
    current microbatch. The caller must not divide by accumulation again.
    """
    if type(global_block_count) is not int or global_block_count < 0:
        raise ValueError("Global block count must be a nonnegative integer")
    if type(gradient_average_world_size) is not int or gradient_average_world_size < 1:
        raise ValueError("Gradient average world size must be positive")
    if global_block_count < int(terms.block_count):
        raise ValueError("Global block count is smaller than local block count")
    return terms.loss_sum * (gradient_average_world_size / max(global_block_count, 1))


def probabilities_on_candidates(candidate_ids, target_ids, target_probs):
    """Join the verifier's complete sparse support onto fresh draft candidates."""
    if target_ids.shape != target_probs.shape or candidate_ids.shape[:-1] != target_ids.shape[:-1]:
        raise ValueError("Sparse teacher and candidate shape mismatch")
    if not bool(torch.isfinite(target_probs).all()) or bool((target_probs < 0).any()):
        raise ValueError("Invalid sparse teacher probabilities")
    for ids in (candidate_ids, target_ids):
        if ids.dtype not in (torch.int32, torch.int64) or bool((ids < 0).any()):
            raise ValueError("Sparse support requires nonnegative integer token IDs")
        ordered = ids.sort(dim=-1).values
        if bool((ordered[..., 1:] == ordered[..., :-1]).any()):
            raise ValueError("Sparse support contains duplicate token IDs")
    if not bool(torch.allclose(target_probs.sum(-1), torch.ones_like(target_probs[..., 0]),
                               atol=2e-5, rtol=0)):
        raise ValueError("Teacher sparse support is incomplete")
    matches = candidate_ids.unsqueeze(-1) == target_ids.unsqueeze(-2)
    return (matches * target_probs.detach().unsqueeze(-2)).sum(-1)


class OPDDFlash2Model(OnlineDFlashModel):
    """Replay actual verified blocks with coupled selector/backbone gradients.

    The adapter passes a single, audited variable-length rollout per rank. This
    implementation returns an unnormalized block sum; normalize accumulated
    gradients by the globally reduced valid-block count BEFORE gradient clip.
    """

    def forward(self, *, input_ids, hidden_states, loss_mask, replay,
                auxiliary_coefficient: float = 0.1):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or self.block_size < 2:
            raise ValueError("OPD replay requires one rollout and at least one proposal per block")
        if self.selector_stop_gradient:
            raise ValueError("OPD replay requires coupled selector and backbone gradients")
        if not 0 <= auxiliary_coefficient <= 1:
            raise ValueError("Invalid auxiliary coefficient")
        anchors = replay['anchor_positions']
        n = anchors.numel()
        if anchors.ndim != 1 or n == 0 or torch.unique(anchors).numel() != n:
            raise ValueError("Need nonempty unique actual rollout anchors")
        captured_length = hidden_states.shape[1]
        if bool((anchors < 0).any()) or bool((anchors > captured_length).any()):
            raise ValueError("An OPD anchor refers to uncaptured target context")
        if captured_length > input_ids.shape[1] or input_ids.shape[1] - captured_length > 1:
            raise ValueError("Unexpected target tap/returned-token alignment")
        if captured_length < input_ids.shape[1]:
            hidden_states = torch.cat([hidden_states, hidden_states.new_zeros(
                1, input_ids.shape[1] - captured_length, hidden_states.shape[-1])], dim=1)
        _, _, hidden = self._forward_draft_blocks(input_ids, hidden_states, loss_mask,
            anchor_positions=anchors[None], block_keep_mask=torch.ones_like(anchors[None], dtype=torch.bool))
        hidden = hidden.reshape(1, n, self.block_size, -1)[0, :, 1:]
        logits = self.draft_model.transform_unary_logits(self.lm_head(hidden))
        unary, ids = logits.topk(self.draft_model.candidate_selector.top_k, dim=-1)
        predecessors = torch.cat([input_ids[0, anchors, None], replay['proposed_ids'][:, :-1]], dim=-1)
        scores = self.draft_model.candidate_selector.score_candidates(candidate_ids=ids,
            unary_logits=unary, hidden_states=hidden, predecessor_ids=predecessors)
        mask = first_rejection_mask(replay['accepted_lengths'], replay['exposed_lengths'],
                                    proposal_width=self.block_size - 1)
        # Reusing stale candidates would train a different policy from the rollout.
        if bool(((ids != replay['candidate_ids']).any(-1) & mask).any()):
            raise ValueError("Current-policy replay candidate identity differs from serving")
        recorded_q = replay['q']
        if recorded_q.shape != scores.shape or not bool(torch.isfinite(recorded_q[mask]).all()):
            raise ValueError("Invalid recorded replay probabilities")
        if bool((recorded_q[mask] < 0).any()) or not torch.allclose(
                recorded_q[mask].sum(-1), torch.ones_like(recorded_q[mask][:, 0]), atol=2e-5, rtol=0):
            raise ValueError("Recorded replay probability mass differs from one")
        q = scores.float().softmax(-1)
        max_error = (torch.where(mask[..., None], (q.detach() - replay['q']).abs(), 0.)).max()
        if float(max_error) > 0.01:
            raise ValueError("Current-policy replay probability drift exceeds 0.01")
        p = probabilities_on_candidates(ids, replay['target_ids'], replay['target_probs'])
        terms = opd_block_terms(scores, p, mask)
        auxiliary_loss = terms.loss_sum.new_zeros(())
        if auxiliary_coefficient:
            auxiliary_loss, _, _ = super().forward(input_ids=input_ids, hidden_states=hidden_states,
                loss_mask=loss_mask, collect_detailed_metrics=False)
        # Equal block weighting of each record's prior auxiliary objective.
        total = terms.loss_sum + auxiliary_coefficient * terms.block_count * auxiliary_loss
        return total, {'opd_terms': terms, 'auxiliary_loss': auxiliary_loss.detach(),
                       'replay_probability_max_error': max_error.detach()}
