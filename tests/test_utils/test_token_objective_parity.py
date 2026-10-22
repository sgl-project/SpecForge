# coding=utf-8
"""Bit-exact parity for the shared token objective extraction.

``OnlineDFlashModel`` and ``OnlineDSparkModel`` now bind
:mod:`specforge.core.token_objective` instead of computing CE / hard TV / LK /
teacher L1 inline. This module carries verbatim copies of the pre-refactor
methods (``origin/main`` at ``03f3682e``), binds them onto an otherwise
identical model, and asserts that loss, every parameter gradient, accuracy and
every logged metric are bit-identical to the refactored code on fixed-seed
inputs, across chunk sizes, masks, objective variants and both wrappers.

Two trivial adapters bridge the pre-refactor method signatures to the new
``forward``: the legacy DFlash chunk function gains a trailing zero
``teacher_l1_num`` term (the new ``DFlashObjectiveTerms`` field, always zero
for DFlash) and the legacy composer ignores that term. Neither touches the
legacy arithmetic.
"""

import os
import unittest
from functools import partial
from types import MethodType
from typing import Dict, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import (
    _DPACE_LOSS_TYPES,
    OnlineDFlashModel,
    OnlineDSparkModel,
    SelectorTerms,
    _scatter_accepted_prefix,
    compute_walk_accepted_length_terms,
)
from specforge.core import fused_loss
from specforge.core.chunking import checkpointed_chunk_reduce
from specforge.core.token_objective import (
    TokenObjectiveConfig,
    compute_token_objective,
    token_objective_chunk_terms,
)
from specforge.modeling.draft.dflash2 import DFlash2DraftModel


class DFlashObjectiveTerms(NamedTuple):
    """Pre-refactor ``DFlashObjectiveTerms`` (no ``teacher_l1_num`` field)."""

    ce_loss_num: torch.Tensor
    tv_loss_num: torch.Tensor
    loss_den: torch.Tensor
    target_probability_num: torch.Tensor
    correct_num: torch.Tensor
    accuracy_den: torch.Tensor
    selector_ce_num: torch.Tensor
    selector_probability_num: torch.Tensor
    selector_correct_num: torch.Tensor
    selector_weight_den: torch.Tensor
    selector_covered_num: torch.Tensor


# ---------------------------------------------------------------------------
# Verbatim pre-refactor methods. Do not edit: they are the parity reference.
# ---------------------------------------------------------------------------


# --- origin/main 03f3682e lines 418-448 (verbatim, dedented) ---
def _dpace_weight(
    self,
    prob: torch.Tensor,
    binary_mask: torch.Tensor,
    binary_mask_b: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    """Compute detached D-PACE position weights.

    ``prob`` is the draft probability on the target token at each draft
    position. Invalid positions are treated as multiplicative no-ops inside
    prefix products and excluded from suffix sums; the caller still
    multiplies the returned weights by ``binary_mask`` before reduction.
    """
    smooth = (1.0 - self.dpace_alpha) * prob + self.dpace_alpha
    smooth = torch.where(binary_mask_b, smooth, torch.ones_like(smooth))
    prefix = torch.cumprod(smooth, dim=-1)

    if loss_type == "dpace-cumulative-confidence-only":
        return prefix

    suffix = torch.flip(
        torch.cumsum(torch.flip(prefix * binary_mask, dims=[-1]), dim=-1),
        dims=[-1],
    )

    if loss_type == "dpace":
        return suffix
    if loss_type == "dpace-continuation-value-only":
        return suffix / prefix.clamp_min(torch.finfo(prefix.dtype).tiny)
    raise ValueError(f"unknown D-PACE loss_type {loss_type!r}")


# --- origin/main 03f3682e lines 815-944 (verbatim, dedented) ---
def _dflash_objective_chunk_terms(
    self,
    hidden: torch.Tensor,
    target_ids: torch.Tensor,
    weight_mask: torch.Tensor,
    predecessor_ids: torch.Tensor,
    sequence_anchor_scale: Optional[torch.Tensor] = None,
    *,
    fused_head: bool = False,
) -> DFlashObjectiveTerms:
    """Return a flat tuple of additive objective and metric tensors.

    ``fused_head=True`` (see ``_use_fused_unary_head``) takes the unary CE,
    strict top-k, and argmax from the fused Triton head. Its inputs omit
    block slot 0, so decay positions start at 1.
    """

    batch_size, num_blocks, block_size, hidden_size = hidden.shape
    candidate_selector = getattr(self.draft_model, "candidate_selector", None)
    objective_logits = None
    topk_values = topk_ids = fused_predicted_ids = None
    if fused_head:
        from specforge.core.dflash_head_triton import dflash_unary_head_fused

        neg_log_q, topk_values, topk_ids, fused_predicted_ids = dflash_unary_head_fused(
            hidden,
            self.lm_head.weight,
            target_ids,
            candidate_selector.top_k if self._selector_objective_enabled else 0,
        )
    else:
        logits = self.lm_head(
            hidden.reshape(batch_size, num_blocks * block_size, hidden_size)
        ).reshape(batch_size, num_blocks, block_size, -1)
        objective_logits = (
            self.draft_model.transform_unary_logits(logits)
            if candidate_selector is not None
            else logits
        )
        neg_log_q = F.cross_entropy(
            objective_logits.reshape(-1, objective_logits.shape[-1]),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape_as(target_ids)

    target_probability = torch.exp(-neg_log_q)
    loss_weights = weight_mask
    if self.loss_type == "dflash":
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            positions = torch.arange(
                1 if fused_head else 0,
                self.block_size,
                device=hidden.device,
            ).view(1, 1, -1)
            decay_weights = torch.exp(
                -(positions - 1).clamp(min=0).float() / self.loss_decay_gamma
            )
            loss_weights = loss_weights * decay_weights
        loss_den = loss_weights.sum()
    elif self.loss_type in _DPACE_LOSS_TYPES:
        if sequence_anchor_scale is None:
            raise ValueError(
                "precomputed full-sequence sequence_anchor_scale is required "
                "for D-PACE chunk reduction"
            )
        with torch.no_grad():
            dpace_weights = self._dpace_weight(
                target_probability.detach(),
                weight_mask,
                weight_mask > 0,
                self.loss_type,
            )
        loss_weights = weight_mask * dpace_weights
        valid_anchors = (weight_mask > 0).any(dim=-1)
        loss_weights = loss_weights * sequence_anchor_scale
        # Each valid anchor contributes 1 / A_b to the denominator, so
        # reducing all chunks yields the number of valid sequences. This
        # preserves D-PACE's total credit mass while balancing sequences
        # with different numbers of sampled anchors.
        loss_den = (
            valid_anchors.to(weight_mask.dtype) * sequence_anchor_scale.squeeze(-1)
        ).sum()
    else:  # defensive: __init__ validates the configured loss type.
        raise ValueError(f"unknown loss_type {self.loss_type!r}")

    ce_loss_num = (neg_log_q * loss_weights).sum()
    if self.lk_loss_type in {"lambda", "tv"}:
        tv_loss_num = ((1.0 - target_probability) * loss_weights).sum()
    else:
        tv_loss_num = ce_loss_num.new_zeros(())
    target_probability_num = (target_probability.detach() * weight_mask).sum()

    selector_terms = SelectorTerms.zeros(ce_loss_num)
    if self._selector_objective_enabled:
        selector_terms = self._selector_chunk_terms(
            candidate_selector=candidate_selector,
            objective_logits=objective_logits,
            hidden=hidden,
            target_ids=target_ids,
            predecessor_ids=predecessor_ids,
            loss_weights=loss_weights,
            weight_mask=weight_mask,
            topk_values=topk_values,
            topk_ids=topk_ids,
        )

    with torch.no_grad():
        predicted_ids = (
            fused_predicted_ids if fused_head else objective_logits.argmax(dim=-1)
        )
        correct_num = (
            ((predicted_ids == target_ids) & (weight_mask > 0.5)).sum().float()
        )
        accuracy_den = weight_mask.sum()
    return DFlashObjectiveTerms(
        ce_loss_num=ce_loss_num,
        tv_loss_num=tv_loss_num,
        loss_den=loss_den,
        target_probability_num=target_probability_num,
        correct_num=correct_num,
        accuracy_den=accuracy_den,
        selector_ce_num=selector_terms.ce_num,
        selector_probability_num=selector_terms.probability_num,
        selector_correct_num=selector_terms.correct_num,
        selector_weight_den=selector_terms.weight_den,
        selector_covered_num=selector_terms.covered_num,
    )


# --- origin/main 03f3682e lines 1360-1370 (verbatim, dedented) ---
def _lk_kl_weight(
    self,
    probability_num: torch.Tensor,
    probability_den: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Return the LK-lambda CE weight, or ``None`` for other objectives."""

    if self.lk_loss_type != "lambda":
        return None
    acceptance = probability_num / probability_den.clamp_min(1.0)
    return self.kl_scale * torch.exp(-self.kl_decay * acceptance.detach())


# --- origin/main 03f3682e lines 1372-1389 (verbatim, dedented) ---
def _compose_token_objective(
    self,
    ce_num: torch.Tensor,
    tv_num: torch.Tensor,
    probability_num: torch.Tensor,
    probability_den: torch.Tensor,
) -> torch.Tensor:
    """Compose a hard-target CE/TV/LK numerator after chunk reduction."""

    if self.lk_loss_type is None or self.lk_loss_type == "alpha":
        # With a one-hot target distribution, LK-alpha is exactly NLL/CE.
        return ce_num
    if self.lk_loss_type == "tv":
        return tv_num
    if self.lk_loss_type == "lambda":
        kl_weight = self._lk_kl_weight(probability_num, probability_den)
        return kl_weight * ce_num + (1.0 - kl_weight) * tv_num
    raise ValueError(f"unknown lk_loss_type {self.lk_loss_type!r}")


# --- origin/main 03f3682e lines 2087-2098 (verbatim, dedented) ---
def _dspark_loss_weight_mask(
    self,
    eval_mask: torch.Tensor,
) -> torch.Tensor:
    loss_weight_mask = eval_mask.to(torch.float32)
    if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
        positions = torch.arange(self.block_size, device=eval_mask.device).view(
            1, 1, -1
        )
        decay_weights = torch.exp(-positions.float() / float(self.loss_decay_gamma))
        loss_weight_mask = loss_weight_mask * decay_weights
    return loss_weight_mask


# --- origin/main 03f3682e lines 2119-2279 (verbatim, dedented) ---
def _dspark_objective_chunk_terms(
    self,
    hidden: torch.Tensor,
    prev_token_ids: torch.Tensor,
    target_ids: torch.Tensor,
    loss_weights: torch.Tensor,
    eval_mask: torch.Tensor,
    aligned_target_hidden: Optional[torch.Tensor],
    block_index: Optional[torch.Tensor] = None,
    *,
    total_blocks: int = 0,
) -> Tuple[torch.Tensor, ...]:
    """Return additive loss and telemetry numerators for one block slice."""

    batch_size, num_blocks, block_size, hidden_size = hidden.shape
    base_logits = self.lm_head(
        hidden.reshape(batch_size, num_blocks * block_size, hidden_size)
    ).reshape(batch_size, num_blocks, block_size, -1)
    draft_logits = self.draft_model.apply_logits_head(
        base_logits,
        prev_token_ids=prev_token_ids,
        hidden_states=hidden,
    )
    vocab_size = draft_logits.shape[-1]
    cross_entropy = F.cross_entropy(
        draft_logits.reshape(-1, vocab_size),
        target_ids.reshape(-1),
        reduction="none",
    ).reshape_as(target_ids)
    ce_num = (cross_entropy * loss_weights).sum()

    zero = ce_num.new_zeros(())
    l1_num = zero
    confidence_num = zero
    confidence_error_num = zero
    teacher_agreement_num = zero
    teacher_top1_num = zero
    draft_top1_num = zero
    tau_num = zero
    tau_den = zero
    accept_probability = None

    teacher_ids = None
    target_logits = None
    if aligned_target_hidden is not None:
        with torch.no_grad():
            target_logits = self.lm_head(
                aligned_target_hidden.reshape(
                    batch_size,
                    num_blocks * block_size,
                    hidden_size,
                )
            ).reshape_as(draft_logits)
            teacher_ids = target_logits.argmax(dim=-1)
        # BF16 benefits from fusion; FP32 stays eager unless explicitly enabled.
        if (
            os.environ.get(
                "SPECFORGE_DSPARK_FUSED_LOSS",
                "1" if draft_logits.dtype == torch.bfloat16 else "0",
            )
            == "1"
        ):
            # Fused TV: l1 = 2 * tv and accept = 1 - tv for normalized
            # softmax pairs; no [B, T, V] probability tensor materialized.
            tv_per_token = fused_loss.fused_tv_loss(
                draft_logits.reshape(batch_size, num_blocks * block_size, vocab_size),
                target_logits.reshape(batch_size, num_blocks * block_size, vocab_size),
            ).reshape_as(target_ids)
            l1_per_token = 2.0 * tv_per_token
            accept_probability = (1.0 - tv_per_token).clamp(0.0, 1.0)
        else:
            with torch.no_grad():
                target_probabilities = torch.softmax(target_logits.float(), dim=-1)
            draft_probabilities = torch.softmax(draft_logits.float(), dim=-1)
            l1_per_token = (
                (draft_probabilities - target_probabilities).abs().sum(dim=-1)
            )
            accept_probability = (1.0 - 0.5 * l1_per_token).clamp(0.0, 1.0)
        if self.dspark_l1_loss_alpha > 0:
            l1_num = (l1_per_token * loss_weights).sum()

    confidence_pred = self.draft_model.predict_confidence(
        hidden,
        prev_token_ids=prev_token_ids,
    )
    if confidence_pred is not None and self.dspark_confidence_head_alpha > 0:
        if accept_probability is None:
            raise ValueError(
                "DSpark confidence loss requires target_last_hidden_states"
            )
        confidence_per_token = F.binary_cross_entropy_with_logits(
            confidence_pred.float(),
            accept_probability.detach(),
            reduction="none",
        )
        confidence_num = (confidence_per_token * loss_weights).sum()
        confidence_error_num = (
            (confidence_pred.float().sigmoid() - accept_probability).abs()
            * loss_weights
        ).sum()

    with torch.no_grad():
        predicted_ids = draft_logits.argmax(dim=-1)
        correct = ((predicted_ids == target_ids) & eval_mask).float()
        correct_num = correct.sum()
        eval_den = eval_mask.float().sum()
        ce_position_num = (cross_entropy.detach() * eval_mask).sum(dim=(0, 1))
        correct_position_num = correct.sum(dim=(0, 1))
        position_den = eval_mask.float().sum(dim=(0, 1))
        walk_accepted = _scatter_accepted_prefix(
            predicted_ids, target_ids, eval_mask, block_index, total_blocks
        )
        if aligned_target_hidden is not None:
            assert teacher_ids is not None and target_logits is not None
            teacher_agreement_num = (
                (predicted_ids == teacher_ids).float() * eval_mask
            ).sum()
            # Top-1 softmax probability without materializing probabilities:
            # p_max = exp(max_logit - logsumexp(logits)), in fp32.
            teacher_top1_num = (
                torch.exp(
                    target_logits.max(dim=-1).values
                    - target_logits.float().logsumexp(dim=-1)
                )
                * eval_mask
            ).sum()
            draft_top1_num = (
                torch.exp(
                    draft_logits.max(dim=-1).values
                    - draft_logits.float().logsumexp(dim=-1)
                )
                * eval_mask
            ).sum()
            valid_blocks = eval_mask.any(dim=-1).float()
            accepted_expectation = (accept_probability.detach() * eval_mask).cumprod(
                dim=-1
            ).sum(dim=-1) + 1.0
            tau_num = (accepted_expectation * valid_blocks).sum()
            tau_den = valid_blocks.sum()

    return (
        ce_num,
        l1_num,
        confidence_num,
        confidence_error_num,
        correct_num,
        eval_den,
        ce_position_num,
        correct_position_num,
        position_den,
        teacher_agreement_num,
        teacher_top1_num,
        draft_top1_num,
        tau_num,
        tau_den,
        walk_accepted,
    )


# --- origin/main 03f3682e lines 2281-2416 (verbatim, dedented) ---
def _compute_dspark_loss(
    self,
    *,
    output_hidden: torch.Tensor,
    target_ids: torch.Tensor,
    eval_mask: torch.Tensor,
    prev_token_ids: torch.Tensor,
    safe_label_indices: torch.Tensor,
    target_last_hidden_states: Optional[torch.Tensor],
    anchor_positions: torch.Tensor,
    collect_detailed_metrics: bool,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Token-pooled DSpark objective with bounded vocab-logit memory."""

    batch_size, num_blocks, block_size = target_ids.shape
    hidden_4d = output_hidden.reshape(
        batch_size,
        num_blocks,
        block_size,
        -1,
    )
    loss_weights = self._dspark_loss_weight_mask(eval_mask)
    local_loss_den = loss_weights.sum()
    need_target = self.dspark_l1_loss_alpha > 0 or (
        self.dspark_confidence_head_alpha > 0
        and getattr(self.draft_model, "confidence_head", None) is not None
    )
    aligned_target_hidden = None
    if need_target:
        if target_last_hidden_states is None:
            raise ValueError(
                "DSpark L1/confidence loss requires target_last_hidden_states"
            )
        aligned_target_hidden = self._aligned_target_hidden(
            target_last_hidden_states,
            safe_label_indices,
        )

    totals = checkpointed_chunk_reduce(
        partial(self._dspark_objective_chunk_terms, total_blocks=num_blocks),
        hidden_4d,
        prev_token_ids,
        target_ids,
        loss_weights,
        eval_mask,
        aligned_target_hidden,
        (
            torch.arange(num_blocks, device=target_ids.device).unsqueeze(0)
            if collect_detailed_metrics
            else None
        ),
        chunk_size=self.objective_chunk_blocks,
        dim=1,
    )

    (
        ce_num,
        l1_num,
        confidence_num,
        confidence_error_num,
        correct_num,
        eval_den,
        ce_position_num,
        correct_position_num,
        position_den,
        teacher_agreement_num,
        teacher_top1_num,
        draft_top1_num,
        tau_num,
        tau_den,
        walk_accepted,
    ) = totals

    global_loss_den = local_loss_den.detach().clone()
    world_size = 1
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        if world_size > 1:
            dist.all_reduce(global_loss_den, op=dist.ReduceOp.SUM)
    # Device-side assert: a host-side float() here drains the stream per
    # microbatch right after a collective, serializing all ranks.
    torch._assert_async(
        (global_loss_den > 0).any(),
        "DSpark objective has no supervised target tokens",
    )
    loss = (
        world_size
        * (
            self.dspark_ce_loss_alpha * ce_num
            + self.dspark_l1_loss_alpha * l1_num
            + self.dspark_confidence_head_alpha * confidence_num
        )
        / global_loss_den
    )

    ratio_metrics = {
        "acc": (correct_num, eval_den),
        "ce_loss": (ce_num.detach(), local_loss_den.detach()),
        "l1_loss": (l1_num.detach(), local_loss_den.detach()),
        "confidence_loss": (
            confidence_num.detach(),
            local_loss_den.detach(),
        ),
        "confidence_abs_error": (
            confidence_error_num.detach(),
            local_loss_den.detach(),
        ),
        "ce_position": (ce_position_num, position_den),
        "accuracy_position": (correct_position_num, position_den),
    }
    if aligned_target_hidden is not None:
        ratio_metrics.update(
            {
                "teacher_agreement": (teacher_agreement_num, eval_den),
                "teacher_top1_prob": (teacher_top1_num, eval_den),
                "draft_top1_prob": (draft_top1_num, eval_den),
                "tau_probabilistic": (tau_num, tau_den),
            }
        )
    if collect_detailed_metrics:
        ratio_metrics["dspark/hard_label/walk_accepted_length"] = (
            compute_walk_accepted_length_terms(
                walk_accepted, anchor_positions, eval_mask.any(dim=-1)
            )
        )
    metrics: Dict[str, object] = {
        "ratio_metrics": {
            name: (numerator.detach(), denominator.detach())
            for name, (numerator, denominator) in ratio_metrics.items()
        },
        "accuracy_denom": eval_den.detach(),
    }
    accuracy = correct_num / eval_den.clamp_min(1.0)
    return loss, {"accuracy": accuracy.detach(), **metrics}


# ---------------------------------------------------------------------------
# Binding and adapters.
# ---------------------------------------------------------------------------


def _legacy_dflash_chunk_terms(self, *args, **kwargs):
    terms = _dflash_objective_chunk_terms(self, *args, **kwargs)
    # The refactored tuple carries one more (always zero) DFlash term.
    return (*terms, terms.ce_loss_num.new_zeros(()))


def _legacy_compose(
    self, ce_num, tv_num, probability_num, probability_den, teacher_l1_num=None
):
    del teacher_l1_num
    return _compose_token_objective(
        self, ce_num, tv_num, probability_num, probability_den
    )


def _bind_legacy(model: OnlineDFlashModel) -> OnlineDFlashModel:
    model._dpace_weight = MethodType(_dpace_weight, model)
    model._lk_kl_weight = MethodType(_lk_kl_weight, model)
    model._compose_token_objective = MethodType(_legacy_compose, model)
    model._dflash_objective_chunk_terms = MethodType(_legacy_dflash_chunk_terms, model)
    if isinstance(model, OnlineDSparkModel):
        model._dspark_loss_weight_mask = MethodType(_dspark_loss_weight_mask, model)
        model._dspark_objective_chunk_terms = MethodType(
            _dspark_objective_chunk_terms, model
        )
        model._compute_dspark_loss = MethodType(_compute_dspark_loss, model)
    return model


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------

HIDDEN = 16
VOCAB = 32
BLOCK = 4
BATCH, SEQ = 2, 20
NUM_ANCHORS = 6
CHUNK_SIZES = (0, 1, 4)


class _LearnableDraft(nn.Module):
    """Draft stub whose block output, corrected logits and confidence all
    depend on trainable parameters, so gradient parity is meaningful."""

    def __init__(self, *, corrected_logits: bool = False, confidence: bool = False):
        super().__init__()
        self.sliding_window = None
        self.noise_proj = nn.Linear(HIDDEN, HIDDEN)
        self.context_proj = nn.Linear(HIDDEN, HIDDEN)
        self.correction = (
            nn.Parameter(torch.randn(HIDDEN, VOCAB) * 0.1) if corrected_logits else None
        )
        self.confidence_head = nn.Linear(HIDDEN, 1) if confidence else None

    def forward(
        self, position_ids, noise_embedding, target_hidden, attention_mask, **kw
    ):
        del position_ids, attention_mask, kw
        context = torch.tanh(self.context_proj(target_hidden)).mean(dim=1, keepdim=True)
        return torch.tanh(self.noise_proj(noise_embedding)) + context

    def apply_logits_head(self, base_logits, prev_token_ids, hidden_states):
        del prev_token_ids
        if self.correction is None:
            return base_logits
        return base_logits + hidden_states @ self.correction

    def predict_confidence(self, hidden_states, prev_token_ids=None):
        del prev_token_ids
        if self.confidence_head is None:
            return None
        return self.confidence_head(hidden_states).squeeze(-1)


def _tiny_dflash2_config():
    return Qwen3Config(
        architectures=["DFlash2DraftModel"],
        hidden_size=HIDDEN,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        num_target_layers=4,
        head_dim=4,
        max_position_embeddings=64,
        vocab_size=VOCAB,
        layer_types=["full_attention"],
        dflash_config={
            "block_size": BLOCK,
            "conv_group_size": 4,
            "conv_kernel_size": 2,
            "mask_token_id": VOCAB - 1,
            "selector_rank": 4,
            "selector_top_k": 3,
            "target_layer_ids": [1],
        },
    )


def _inputs(dtype: torch.dtype, *, feed_teacher: bool) -> Dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    loss_mask = torch.ones(BATCH, SEQ, dtype=torch.long)
    loss_mask[1, 14:] = 0
    loss_mask[0, 5] = 0
    loss_mask[0, 11:13] = 0
    inputs = {
        "input_ids": torch.randint(0, VOCAB - 2, (BATCH, SEQ), generator=generator),
        "hidden_states": torch.randn(BATCH, SEQ, HIDDEN, generator=generator).to(dtype),
        "loss_mask": loss_mask,
    }
    if feed_teacher:
        inputs["target_last_hidden_states"] = torch.randn(
            BATCH, SEQ, HIDDEN, generator=generator
        ).to(dtype)
    return inputs


def _build(kind: str, loss_kwargs: dict, *, chunk: int, dtype: torch.dtype):
    torch.manual_seed(0)
    if kind == "dflash":
        draft = _LearnableDraft()
        cls = OnlineDFlashModel
    elif kind == "dflash2":
        draft = DFlash2DraftModel(_tiny_dflash2_config())
        cls = OnlineDFlashModel
    elif kind == "dspark":
        draft = _LearnableDraft(corrected_logits=True)
        cls = OnlineDSparkModel
    elif kind == "dspark-confidence":
        draft = _LearnableDraft(corrected_logits=True, confidence=True)
        cls = OnlineDSparkModel
    else:
        raise ValueError(kind)
    model = cls(
        draft_model=draft,
        target_lm_head=nn.Linear(HIDDEN, VOCAB, bias=False).requires_grad_(False),
        target_embed_tokens=nn.Embedding(VOCAB, HIDDEN).requires_grad_(False),
        mask_token_id=VOCAB - 1,
        block_size=BLOCK,
        attention_backend="sdpa",
        num_anchors=NUM_ANCHORS,
        objective_chunk_blocks=chunk,
        **loss_kwargs,
    )
    return model.to(dtype)


def _run(
    kind: str,
    loss_kwargs: dict,
    *,
    chunk: int,
    legacy: bool,
    feed_teacher: bool,
    dtype: torch.dtype = torch.float32,
    collect_detailed_metrics: bool = True,
):
    model = _build(kind, loss_kwargs, chunk=chunk, dtype=dtype)
    if legacy:
        _bind_legacy(model)
    inputs = _inputs(dtype, feed_teacher=feed_teacher)
    torch.manual_seed(2)  # anchor sampling
    loss, accuracy, metrics = model(
        **inputs, collect_detailed_metrics=collect_detailed_metrics
    )
    loss.backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return loss.detach(), accuracy.detach(), metrics, gradients


def _flatten_metrics(metrics: object, prefix: str = "") -> Dict[str, torch.Tensor]:
    """Flatten nested dict/tuple metric containers into ``path -> tensor``."""

    flat: Dict[str, torch.Tensor] = {}
    if isinstance(metrics, torch.Tensor):
        flat[prefix] = metrics
    elif isinstance(metrics, dict):
        for key, value in metrics.items():
            flat.update(_flatten_metrics(value, f"{prefix}/{key}"))
    elif isinstance(metrics, (tuple, list)):
        for index, value in enumerate(metrics):
            flat.update(_flatten_metrics(value, f"{prefix}/{index}"))
    else:
        flat[prefix] = torch.as_tensor(metrics)
    return flat


DFLASH_CASES = (
    {},
    {"loss_decay_gamma": 3.0},
    {"lk_loss_type": "alpha"},
    {"lk_loss_type": "tv", "loss_decay_gamma": 2.0},
    {"lk_loss_type": "lambda", "kl_scale": 0.7, "kl_decay": 1.3},
    {"loss_type": "dpace", "dpace_alpha": 0.3},
    {"loss_type": "dpace-cumulative-confidence-only"},
    {"loss_type": "dpace-continuation-value-only", "lk_loss_type": "lambda"},
    {"loss_type": "dpace", "lk_loss_type": "tv"},
)
DFLASH2_CASES = (
    {},
    {"loss_type": "dpace"},
    {"lk_loss_type": "lambda"},
    {"lk_loss_type": "tv", "loss_decay_gamma": 2.0},
    {"selector_stop_gradient": True},
)
DSPARK_CASES = (
    # (kind, kwargs, feed_teacher)
    ("dspark", {"dspark_ce_loss_alpha": 0.1, "dspark_l1_loss_alpha": 0.9}, True),
    (
        "dspark",
        {
            "dspark_ce_loss_alpha": 0.1,
            "dspark_l1_loss_alpha": 0.9,
            "loss_decay_gamma": 4.0,
        },
        True,
    ),
    ("dspark", {"dspark_ce_loss_alpha": 1.0, "dspark_l1_loss_alpha": 0.0}, False),
    ("dspark", {"dspark_ce_loss_alpha": 1.0, "dspark_l1_loss_alpha": 0.0}, True),
    ("dspark", {"dspark_ce_loss_alpha": 0.4, "dspark_l1_loss_alpha": 0.0}, True),
    (
        "dspark-confidence",
        {
            "dspark_ce_loss_alpha": 0.1,
            "dspark_l1_loss_alpha": 0.9,
            "dspark_confidence_head_alpha": 0.5,
        },
        True,
    ),
    (
        "dspark-confidence",
        {
            "dspark_ce_loss_alpha": 0.3,
            "dspark_l1_loss_alpha": 0.0,
            "loss_decay_gamma": 2.5,
        },
        True,
    ),
)


class TokenObjectiveParityTest(unittest.TestCase):
    def _assert_bit_exact(self, reference, candidate, label: str):
        ref_loss, ref_acc, ref_metrics, ref_grads = reference
        loss, acc, metrics, grads = candidate
        self.assertTrue(torch.isfinite(loss), label)
        self.assertTrue(torch.equal(loss, ref_loss), f"{label}: loss")
        self.assertTrue(torch.equal(acc, ref_acc), f"{label}: accuracy")
        self.assertEqual(set(ref_grads), set(grads), f"{label}: gradient keys")
        self.assertTrue(ref_grads, f"{label}: no gradients flowed")
        for name, gradient in grads.items():
            self.assertTrue(
                torch.equal(gradient, ref_grads[name]), f"{label}: grad {name}"
            )
        ref_flat = _flatten_metrics(ref_metrics)
        flat = _flatten_metrics(metrics)
        self.assertEqual(set(ref_flat), set(flat), f"{label}: metric keys")
        for key, tensor in flat.items():
            self.assertTrue(torch.equal(tensor, ref_flat[key]), f"{label}: {key}")

    def _check(self, kind, loss_kwargs, *, feed_teacher, **run_kwargs):
        for chunk in CHUNK_SIZES:
            label = f"{kind} {loss_kwargs} chunk={chunk} {run_kwargs}"
            with self.subTest(kind=kind, chunk=chunk, **loss_kwargs, **run_kwargs):
                reference = _run(
                    kind,
                    loss_kwargs,
                    chunk=chunk,
                    legacy=True,
                    feed_teacher=feed_teacher,
                    **run_kwargs,
                )
                candidate = _run(
                    kind,
                    loss_kwargs,
                    chunk=chunk,
                    legacy=False,
                    feed_teacher=feed_teacher,
                    **run_kwargs,
                )
                self._assert_bit_exact(reference, candidate, label)

    def test_dflash_objectives_match_legacy_bit_exactly(self):
        for loss_kwargs in DFLASH_CASES:
            self._check("dflash", loss_kwargs, feed_teacher=True)

    def test_dflash_without_detailed_metrics_matches_legacy(self):
        for loss_kwargs in ({}, {"loss_type": "dpace"}, {"lk_loss_type": "lambda"}):
            self._check(
                "dflash",
                loss_kwargs,
                feed_teacher=False,
                collect_detailed_metrics=False,
            )

    def test_dflash_double_precision_matches_legacy(self):
        for loss_kwargs in ({"loss_decay_gamma": 3.0}, {"lk_loss_type": "lambda"}):
            self._check("dflash", loss_kwargs, feed_teacher=True, dtype=torch.float64)

    def test_dflash2_selector_objectives_match_legacy_bit_exactly(self):
        for loss_kwargs in DFLASH2_CASES:
            self._check("dflash2", loss_kwargs, feed_teacher=True)

    def test_dspark_objectives_match_legacy_bit_exactly(self):
        for kind, loss_kwargs, feed_teacher in DSPARK_CASES:
            self._check(kind, loss_kwargs, feed_teacher=feed_teacher)

    def test_dspark_double_precision_matches_legacy(self):
        self._check(
            "dspark-confidence",
            {"dspark_ce_loss_alpha": 0.1, "dspark_l1_loss_alpha": 0.9},
            feed_teacher=True,
            dtype=torch.float64,
        )

    def test_dspark_fused_tv_call_site_matches_legacy(self):
        """Route the fused TV kernel to an eager stand-in on both paths and
        compare bit-exactly; proves the shared function wires the kernel with
        the same shapes, dtypes and weighting as the legacy call site."""

        def eager_tv(logits, targets):
            draft_p = torch.softmax(logits.float(), dim=-1)
            target_p = torch.softmax(targets.float(), dim=-1)
            return 0.5 * (draft_p - target_p).abs().sum(dim=-1)

        from unittest.mock import patch

        with (
            patch.dict(os.environ, {"SPECFORGE_DSPARK_FUSED_LOSS": "1"}),
            patch.object(fused_loss, "fused_tv_loss", side_effect=eager_tv) as call,
        ):
            self._check(
                "dspark-confidence",
                {"dspark_ce_loss_alpha": 0.1, "dspark_l1_loss_alpha": 0.9},
                feed_teacher=True,
            )
        self.assertTrue(call.called)


class TokenObjectiveModuleTest(unittest.TestCase):
    def test_requires_teacher_distribution_follows_l1_weight(self):
        self.assertFalse(TokenObjectiveConfig().requires_teacher_distribution)
        self.assertTrue(
            TokenObjectiveConfig(teacher_l1_weight=0.9).requires_teacher_distribution
        )
        dflash = _build("dflash", {}, chunk=0, dtype=torch.float32)
        self.assertFalse(dflash.requires_teacher_distribution)
        dspark = _build(
            "dspark",
            {"dspark_ce_loss_alpha": 0.1, "dspark_l1_loss_alpha": 0.9},
            chunk=0,
            dtype=torch.float32,
        )
        self.assertTrue(dspark.requires_teacher_distribution)
        self.assertTrue(dspark.token_objective.requires_teacher_distribution)
        dspark.dspark_l1_loss_alpha = 0.0
        self.assertFalse(dspark.requires_teacher_distribution)

    def test_legacy_field_names_bind_the_shared_config(self):
        model = _build(
            "dflash",
            {"loss_type": "dpace", "lk_loss_type": "lambda", "kl_scale": 0.5},
            chunk=0,
            dtype=torch.float32,
        )
        config = model.token_objective
        self.assertEqual(config.position_weight_mode, "dpace")
        self.assertEqual(config.hard_label_mode, "lambda")
        self.assertEqual(config.kl_scale, 0.5)
        self.assertTrue(config.uses_default_weights)
        # The config tracks later attribute changes (tests mutate loss_type).
        model.loss_type = "dflash"
        self.assertEqual(model.token_objective.position_weight_mode, "static")

    def test_distillation_without_teacher_logits_is_refused(self):
        config = TokenObjectiveConfig(hard_ce_weight=0.1, teacher_l1_weight=0.9)
        logits = torch.randn(1, 2, 3, 5)
        labels = torch.randint(0, 5, (1, 2, 3))
        with self.assertRaisesRegex(ValueError, "teacher_logits"):
            token_objective_chunk_terms(
                config, labels=labels, loss_mask=torch.ones(1, 2, 3), logits=logits
            )

    def test_standalone_objective_matches_wrapper_chunk_terms(self):
        torch.manual_seed(5)
        model = _build(
            "dflash", {"lk_loss_type": "lambda"}, chunk=0, dtype=torch.float32
        )
        hidden = torch.randn(2, 3, BLOCK, HIDDEN)
        targets = torch.randint(0, VOCAB, (2, 3, BLOCK))
        weights = (torch.rand(2, 3, BLOCK) > 0.3).float()
        weights[..., 0] = 0.0
        terms = model._dflash_objective_chunk_terms(hidden, targets, weights, targets)
        loss, result, metrics = compute_token_objective(
            model.token_objective,
            logits=model.lm_head(hidden),
            labels=targets,
            loss_mask=weights,
            chunk_size=2,
        )
        torch.testing.assert_close(result.ce_num, terms.ce_loss_num)
        torch.testing.assert_close(result.hard_tv_num, terms.tv_loss_num)
        torch.testing.assert_close(
            result.loss_num,
            model._compose_token_objective(
                terms.ce_loss_num,
                terms.tv_loss_num,
                terms.target_probability_num,
                terms.accuracy_den,
            ),
        )
        self.assertIn("lk_loss", metrics)
        self.assertIn("objective/lk_kl_weight", metrics)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
