# coding=utf-8
"""DSpark Training Wrapper."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.core.dflash import (
    FLEX_ATTENTION_AVAILABLE,
    OnlineDFlashModel,
)
from specforge.modeling.draft.dflash import DFlashDraftModel


class OnlineDSparkModel(OnlineDFlashModel):
    """DSpark online training wrapper over DFlash block-parallel components."""

    def __init__(
        self,
        draft_model: DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        dspark_ce_loss_alpha: float = 0.1,
        dspark_l1_loss_alpha: float = 0.9,
        dspark_confidence_head_alpha: float = 1.0,
    ):
        super().__init__(
            draft_model=draft_model,
            target_lm_head=target_lm_head,
            target_embed_tokens=target_embed_tokens,
            mask_token_id=mask_token_id,
            block_size=block_size,
            attention_backend=attention_backend,
            num_anchors=num_anchors,
            loss_decay_gamma=loss_decay_gamma,
            loss_type="dflash",
        )
        if dspark_ce_loss_alpha < 0:
            raise ValueError("dspark_ce_loss_alpha must be >= 0")
        if dspark_l1_loss_alpha < 0:
            raise ValueError("dspark_l1_loss_alpha must be >= 0")
        if dspark_confidence_head_alpha < 0:
            raise ValueError("dspark_confidence_head_alpha must be >= 0")

        self.loss_type = "dspark"
        self.dspark_ce_loss_alpha = float(dspark_ce_loss_alpha)
        self.dspark_l1_loss_alpha = float(dspark_l1_loss_alpha)
        self.dspark_confidence_head_alpha = float(dspark_confidence_head_alpha)

    def _build_anchor_candidate_mask(
        self,
        seq_len: int,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        num_candidates = max(seq_len - 1, 0)
        if num_candidates == 0:
            return loss_mask[:, :0].bool()
        anchor_valid = loss_mask[:, :num_candidates] > 0.5
        first_target_valid = loss_mask[:, 1 : num_candidates + 1] > 0.5
        return anchor_valid & first_target_valid

    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample fixed-width DSpark anchors; invalid slots are masked out."""
        valid = self._build_anchor_candidate_mask(seq_len, loss_mask)
        bsz = loss_mask.shape[0]
        num_candidates = valid.shape[1]
        max_n = int(self.num_anchors)
        if num_candidates == 0:
            anchors = torch.zeros(bsz, max_n, dtype=torch.long, device=device)
            keep_mask = torch.zeros(bsz, max_n, dtype=torch.bool, device=device)
            return anchors, keep_mask

        valid_counts = valid.sum(dim=1)
        indices = torch.arange(num_candidates, device=device).unsqueeze(0).expand(
            bsz, -1
        )
        masked_indices = torch.where(
            valid,
            indices,
            torch.full_like(indices, seq_len + 1),
        )
        random_vals = torch.rand(bsz, num_candidates, device=device)
        random_vals = torch.where(valid, random_vals, torch.full_like(random_vals, 2.0))
        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        if num_candidates < max_n:
            pad = torch.full(
                (bsz, max_n - num_candidates),
                seq_len + 1,
                dtype=gathered.dtype,
                device=device,
            )
            gathered = torch.cat([gathered, pad], dim=1)
        anchors = gathered[:, :max_n].sort(dim=1).values
        keep_mask = torch.arange(max_n, device=device).unsqueeze(0) < (
            valid_counts.unsqueeze(1).clamp(max=max_n)
        )
        anchors = torch.where(keep_mask, anchors, torch.zeros_like(anchors))
        return anchors, keep_mask

    def _build_labels_and_mask(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        del bsz
        device = input_ids.device
        label_offsets = torch.arange(1, self.block_size + 1, device=device).view(
            1, 1, -1
        )
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1),
            safe_label_indices,
            torch.zeros_like(safe_label_indices),
        )
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )

        target_valid = label_indices < seq_len
        target_loss_mask = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, label_indices.size(1), -1),
            2,
            safe_label_indices,
        )
        eval_mask = target_valid & (target_loss_mask > 0.5)
        eval_mask = eval_mask & block_keep_mask.unsqueeze(-1)
        eval_mask = eval_mask.to(torch.int32).cumprod(dim=-1).bool()
        return target_ids, eval_mask, label_indices, safe_label_indices

    def _loss_weight_mask(
        self,
        eval_mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        loss_weight_mask = eval_mask.to(torch.float32)
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            positions = torch.arange(self.block_size, device=device).view(1, 1, -1)
            decay_weights = torch.exp(
                -positions.float() / float(self.loss_decay_gamma)
            )
            loss_weight_mask = loss_weight_mask * decay_weights
        return loss_weight_mask

    def _aligned_target_logits(
        self,
        target_last_hidden_states: Optional[torch.Tensor],
        safe_label_indices: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if target_last_hidden_states is None:
            return None
        target_pred_indices = (safe_label_indices - 1).clamp(min=0)
        aligned_target_hidden = torch.gather(
            target_last_hidden_states.unsqueeze(1).expand(
                -1,
                safe_label_indices.size(1),
                -1,
                -1,
            ),
            2,
            target_pred_indices.unsqueeze(-1).expand(
                -1,
                -1,
                -1,
                target_last_hidden_states.size(-1),
            ),
        )
        return self.lm_head(aligned_target_hidden)

    def _compute_loss(
        self,
        *,
        draft_logits: torch.Tensor,
        target_ids: torch.Tensor,
        eval_mask: torch.Tensor,
        confidence_pred: Optional[torch.Tensor],
        aligned_target_logits: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = draft_logits.device
        vocab_size = draft_logits.size(-1)
        loss_weight_mask = self._loss_weight_mask(eval_mask, device)
        flat_logits = draft_logits.reshape(-1, vocab_size)
        flat_targets = target_ids.reshape(-1)
        flat_weights = loss_weight_mask.reshape(-1)

        loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        ce_loss_num = (loss_per_token * flat_weights).sum()
        ce_loss_den = flat_weights.sum()
        ce_loss = ce_loss_num / (ce_loss_den + 1e-6)

        l1_loss = ce_loss.new_zeros(())
        accept_rate_3d = None
        if aligned_target_logits is not None:
            draft_probs = torch.softmax(draft_logits.float(), dim=-1)
            target_probs = torch.softmax(aligned_target_logits.float(), dim=-1)
            accept_rate_3d = 1.0 - 0.5 * (draft_probs - target_probs).abs().sum(dim=-1)
            accept_rate_3d = accept_rate_3d.clamp_(0.0, 1.0)
            if self.dspark_l1_loss_alpha > 0:
                l1_dist = (draft_probs - target_probs).abs().sum(dim=-1)
                l1_loss = (l1_dist * loss_weight_mask).sum() / (ce_loss_den + 1e-6)
        elif self.dspark_l1_loss_alpha > 0 or self.dspark_confidence_head_alpha > 0:
            raise ValueError(
                "DSpark L1/confidence loss requires target_last_hidden_states. "
                "Use the disaggregated DSpark server-capture path so the "
                "consumer receives target_last_hidden_states."
            )

        confidence_loss = ce_loss.new_zeros(())
        confidence_abs_error = ce_loss.new_zeros(())
        if confidence_pred is not None:
            if accept_rate_3d is None:
                raise ValueError("DSpark confidence head requires aligned target logits.")
            confidence_errors = F.binary_cross_entropy_with_logits(
                confidence_pred.float(),
                accept_rate_3d.detach(),
                reduction="none",
            )
            confidence_loss = (confidence_errors * loss_weight_mask).sum() / (
                ce_loss_den + 1e-6
            )
            with torch.no_grad():
                confidence_abs_error = (
                    (confidence_pred.float().sigmoid() - accept_rate_3d).abs()
                    * loss_weight_mask
                ).sum() / (ce_loss_den + 1e-6)

        loss = (
            self.dspark_ce_loss_alpha * ce_loss
            + self.dspark_l1_loss_alpha * l1_loss
            + self.dspark_confidence_head_alpha * confidence_loss
        )
        metrics = {
            "ce_loss": ce_loss.detach(),
            "l1_loss": l1_loss.detach(),
            "confidence_loss": confidence_loss.detach(),
            "confidence_abs_error": confidence_abs_error.detach(),
        }
        return loss, metrics

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_last_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Parallel DSpark training forward pass."""
        if self.attention_backend == "flex_attention" and not FLEX_ATTENTION_AVAILABLE:
            raise ValueError(
                "flex_attention is not available on this device; use sdpa/eager."
            )
        bsz = input_ids.shape[0]
        anchor_positions, block_keep_mask, output_hidden = self._forward_draft_blocks(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
        )

        logits = self.lm_head(output_hidden)
        num_blocks = anchor_positions.size(1)
        output_hidden_4d = output_hidden.reshape(
            bsz, num_blocks, self.block_size, -1
        )
        (
            target_ids,
            eval_mask,
            _label_indices,
            safe_label_indices,
        ) = self._build_labels_and_mask(
            input_ids=input_ids,
            loss_mask=loss_mask,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
        )
        anchor_token_ids = torch.gather(input_ids, 1, anchor_positions)
        prev_token_ids = torch.cat(
            [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]],
            dim=-1,
        )
        draft_logits = logits.reshape(bsz, num_blocks, self.block_size, -1)
        if hasattr(self.draft_model, "apply_markov_logits"):
            draft_logits = self.draft_model.apply_markov_logits(
                draft_logits,
                prev_token_ids=prev_token_ids,
                hidden_states=output_hidden_4d,
            )
        confidence_pred = None
        if hasattr(self.draft_model, "predict_confidence"):
            confidence_pred = self.draft_model.predict_confidence(
                output_hidden_4d,
                prev_token_ids=prev_token_ids,
            )
        aligned_target_logits = self._aligned_target_logits(
            target_last_hidden_states,
            safe_label_indices,
        )
        loss, metrics = self._compute_loss(
            draft_logits=draft_logits,
            target_ids=target_ids,
            eval_mask=eval_mask,
            confidence_pred=confidence_pred,
            aligned_target_logits=aligned_target_logits,
        )
        flat_logits = draft_logits.reshape(-1, draft_logits.size(-1))
        flat_targets = target_ids.reshape(-1)
        binary_eval_mask = eval_mask.reshape(-1)
        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & binary_eval_mask
            accuracy_denom = binary_eval_mask.to(torch.float32).sum()
            accuracy = correct.sum().float() / (accuracy_denom + 1e-6)
        metrics["accuracy_denom"] = accuracy_denom
        return loss, accuracy, metrics
