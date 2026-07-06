# coding=utf-8
"""EDSD Training Wrapper.

EDSD (Enhanced Draft Speculative Decoding) extends EAGLE3 with:
1. Curriculum learning: entropy-based position masking drops confusing tokens
   early in training, progressively including harder positions (learn from easy
   to hard).
2. Variable TTT length: the number of test-time-training unroll steps grows
   with epoch, starting from the full length at epoch 0.
3. EDFuse: a gated fusion module replacing concatenation for embedding-hidden
   state fusion in the draft model (see specforge/modeling/draft/edsd.py).
"""

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from specforge.core.eagle3 import (
    OnlineEagle3Model,
    _compute_loss_and_acceptance_rate,
    _compute_metric_counts,
)
from specforge.core.eagle3_adapters import BackendAdapter, SdpaLikeAdapter, UspAdapter
from specforge.core.loss import LogSoftmaxLoss
from specforge.modeling.draft import Eagle3DraftModel
from specforge.utils import padding


# ---------------------------------------------------------------------------
# Target distribution computation with curriculum learning mask
# ---------------------------------------------------------------------------


@torch.compile(dynamic=None)
def _edsd_compute_target_p_core(target, t2d, loss_mask, compute_on_draft=False):
    """Core target distribution computation (kept inside torch.compile).

    Same as EAGLE3's ``_compute_target_p`` but with an optional
    ``compute_on_draft`` flag to skip the expensive logsumexp when LK loss
    is not used.
    """
    target_head = target.float()
    target_max_token = target_head.argmax(-1)
    target_mask = t2d[target_max_token]
    target_mask = target_mask[..., None].int()
    position_mask = target_mask * loss_mask
    draft_target_head = target_head[..., t2d]
    target_p = nn.Softmax(dim=2)(draft_target_head)
    target_p = target_p.detach()
    target_token_ids = target_max_token.detach()

    # Target probabilities on the full vocabulary restricted to draft tokens.
    # Expensive: full-vocab logsumexp. Only computed when LK loss is enabled.
    target_p_on_draft = None
    if compute_on_draft:
        target_logsumexp = torch.logsumexp(target_head, dim=-1, keepdim=True)
        target_p_on_draft = torch.exp(draft_target_head - target_logsumexp)
        target_p_on_draft = target_p_on_draft.detach()

    return target_p, target_p_on_draft, target_token_ids, position_mask


def _edsd_apply_curriculum_mask(
    target_p, position_mask, epoch_idx, drop_ratio_scale, total_epochs
):
    """Apply entropy-based curriculum learning mask (outside torch.compile).

    Drops the top-``drop_ratio`` fraction of valid positions ranked by entropy
    (confusing tokens).  ``drop_ratio = min(max((total_epochs - epoch_idx) *
    drop_ratio_scale, 0.0), 0.4)``: it is largest at epoch 0 and decays to ~0 by
    the final epoch, so higher epochs retain more positions.  Placed outside
    ``torch.compile`` because ``Tensor.item()`` and dynamic-shape ``topk``
    cause graph breaks.
    """

    eps = 1e-12
    entropy = -(target_p * (target_p + eps).log()).sum(dim=-1)
    valid_mask_flat = position_mask.squeeze(-1).view(-1).bool()
    entropy_flat = entropy.view(-1)

    drop_ratio = min(
        max((total_epochs - epoch_idx) * drop_ratio_scale, 0.0), 0.4
    )
    num_valid = valid_mask_flat.sum()
    k = int((drop_ratio * num_valid).item())
    confusion_mask_flat = torch.ones_like(
        entropy_flat, dtype=position_mask.dtype
    )

    if k > 0 and num_valid > 0:
        valid_indices = valid_mask_flat.nonzero(as_tuple=False).squeeze(-1)
        valid_entropy = entropy_flat[valid_indices]
        _, topk_rel_indices = torch.topk(valid_entropy, k)
        drop_indices = valid_indices[topk_rel_indices]
        confusion_mask_flat[drop_indices] = 0

    confusion_mask = confusion_mask_flat.view_as(entropy)[..., None]
    return position_mask * confusion_mask


def _edsd_compute_target_p_padded(
    target, t2d, loss_mask, length, epoch_idx=0, compute_on_draft=False,
    drop_ratio_scale=0.02, total_epochs=1,
):
    """Pad target distributions for TTT unrolling (EDSD variant).

    Same as EAGLE3's ``_compute_target_p_padded`` but applies EDSD's
    curriculum learning mask after the compiled core computation.
    """
    with torch.no_grad():
        (
            target_p,
            target_p_on_draft,
            target_token_ids,
            position_mask,
        ) = _edsd_compute_target_p_core(
            target=target,
            t2d=t2d,
            loss_mask=loss_mask,
            compute_on_draft=compute_on_draft,
        )

        # Curriculum mask is applied outside torch.compile to avoid
        # graph breaks from Tensor.item() and dynamic-shape topk.
        position_mask = _edsd_apply_curriculum_mask(
            target_p, position_mask, epoch_idx, drop_ratio_scale, total_epochs
        )

        assert len(target_p.shape) == 3
        target_p_padded = F.pad(
            target_p,
            pad=(0, 0, 0, length),
            mode="constant",
            value=1 / target_p.shape[-1],
        )

        if target_p_on_draft is not None:
            target_p_on_draft_padded = F.pad(
                target_p_on_draft,
                pad=(0, 0, 0, length),
                mode="constant",
                value=0.0,
            )
        else:
            target_p_on_draft_padded = None

        target_token_ids_padded = F.pad(
            target_token_ids,
            pad=(0, length),
            mode="constant",
            value=0,
        )

        return (
            target_p_padded,
            target_p_on_draft_padded,
            target_token_ids_padded,
            position_mask,
        )


# ---------------------------------------------------------------------------
# Online EDSD Model
# ---------------------------------------------------------------------------


class OnlineEdsdModel(OnlineEagle3Model):
    """EDSD online training wrapper.

    Inherits from ``OnlineEagle3Model`` and overrides:

    1. ``forward`` — TTT length grows with ``epoch_idx``; uses EDSD's
       curriculum-learning-aware ``_compute_target_p_padded``.
    2. ``_acc_and_loss`` — always computes ``acceptance_rate`` as a
       monitoring metric (matching EAGLE3 behaviour).

    All other logic (adapter, position_ids, attention mask, padding, etc.)
    is inherited from EAGLE3.
    """

    def __init__(
        self,
        draft_model: Eagle3DraftModel,
        length: int = 7,
        attention_backend: str = "sdpa",
        target_model: Optional[nn.Module] = None,
        lk_loss_type: Optional[str] = None,
        kl_scale: float = 1.0,
        kl_decay: float = 1.0,
        drop_ratio_scale: float = 0.02,
        step_n_schedule: Optional[List[int]] = None,
    ):
        super().__init__(
            draft_model=draft_model,
            length=length,
            attention_backend=attention_backend,
            target_model=target_model,
            lk_loss_type=lk_loss_type,
            kl_scale=kl_scale,
            kl_decay=kl_decay,
        )
        # Always compute target_p_on_draft so acceptance_rate can be
        # reported as a monitoring metric even when LK loss is off.
        self._compute_on_draft = True
        self.drop_ratio_scale = drop_ratio_scale
        self.step_n_schedule = step_n_schedule

    @staticmethod
    def compute_step_n(
        current_epoch: int,
        total_epochs: int,
        s_max: int = 7,
    ) -> int:
        """Compute the Step-n TTT length for the current epoch.

        From the EDSD paper (Eq. 6):
            n_t = ceil(1 + (S_max - 1) * t / (T - 1))

        At epoch 0: n_0 = 1 (minimum TTT length)
        At epoch T-1: n_{T-1} = S_max (maximum TTT length)

        Args:
            current_epoch: Current training epoch (0-indexed).
            total_epochs: Total number of training epochs T.
            s_max: Maximum simulation steps (``self.length``).

        Returns:
            TTT length n_t for the current epoch.
        """
        import math
        if total_epochs <= 1:
            return s_max
        return math.ceil(1 + (s_max - 1) * current_epoch / (total_epochs - 1))

    def _compute_actual_length(self, epoch_idx: int, total_epochs: int) -> int:
        """Compute the actual TTT unroll length for the given epoch.

        If ``step_n_schedule`` is provided, it takes priority: the value at
        ``epoch_idx`` is used directly (clamped to the last value if out of range).
        Otherwise, delegates to ``compute_step_n`` (EDSD paper Eq. 6):
            n_t = ceil(1 + (S_max - 1) * t / (T - 1))

        At epoch 0: n_0 = 1 (minimum TTT length)
        At epoch T-1: n_{T-1} = S_max (maximum TTT length)
        """
        if self.step_n_schedule is not None:
            if epoch_idx < len(self.step_n_schedule):
                return self.step_n_schedule[epoch_idx]
            return self.step_n_schedule[-1]
        return self.compute_step_n(
            current_epoch=epoch_idx,
            total_epochs=total_epochs,
            s_max=self.length,
        )

    def _acc_and_loss(
        self,
        *,
        logits: torch.Tensor,
        target_p: torch.Tensor,
        target_p_on_draft: Optional[torch.Tensor],
        target_token_ids: torch.Tensor,
        position_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        adapter: BackendAdapter,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Compute accuracy metric, acceptance_rate, and loss."""
        with torch.no_grad():
            pred_draft_token_ids = logits.argmax(-1)
            pred_target_token_ids = (
                pred_draft_token_ids + self.draft_model.d2t[pred_draft_token_ids]
            )
            local_correct = (
                (pred_target_token_ids == target_token_ids) * loss_mask.squeeze(-1)
            ).sum()
            local_denom = loss_mask.sum().clamp_min(1e-6)
            local_correct, local_denom = adapter.reduce_metrics(
                local_correct=local_correct, local_denom=local_denom
            )
            acc = local_correct / local_denom

        # Always compute acceptance_rate (as a monitoring metric) and loss.
        # When lk_loss_type is None, acceptance_rate is computed with
        # gradients disabled inside _compute_loss_and_acceptance_rate.
        acceptance_rate, loss = _compute_loss_and_acceptance_rate(
            logits=logits,
            target_p=target_p,
            target_p_on_draft=target_p_on_draft,
            position_mask=position_mask,
            lk_loss_type=self.lk_loss_type,
            kl_scale=self.kl_scale,
            kl_decay=self.kl_decay,
            reduce_metrics_fn=adapter.reduce_metrics,
            reduce_loss_fn=adapter.reduce_loss,
        )

        loss_denom = torch.tensor(
            logits.shape[0] * logits.shape[1],
            device=logits.device,
            dtype=torch.float32,
        )
        return (
            acc,
            acceptance_rate,
            loss,
            local_correct,
            local_denom,
            loss.detach(),
            loss_denom,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        is_vlm: bool = False,
        epoch_idx: int = 0,
        total_epochs: int = 1,
        **kwargs,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        """EDSD forward with variable TTT length and curriculum learning.

        Args:
            ... (same as OnlineEagle3Model.forward)
            epoch_idx: Training epoch index (0-indexed). Controls TTT unroll
                length via Step-n (Eq. 6) and curriculum masking. At epoch 0,
                n=1 with no masking; at epoch T-1, n=S_max.
            total_epochs: Total number of training epochs. Used by Step-n
                scheduling when ``step_n_schedule`` is not provided.
        """
        # Step 1: precompute the EDSD variable TTT length (Step-n). This must
        # happen before Step 2 because target_p padding depends on it.
        actual_length = self._compute_actual_length(epoch_idx, total_epochs)

        # Step 2: handle vocab size (with EDSD curriculum-aware target_p)
        (
            target_p_padded,
            target_p_on_draft_padded,
            target_token_ids_padded,
            position_mask,
        ) = _edsd_compute_target_p_padded(
            target=target,
            t2d=self.draft_model.t2d,
            loss_mask=loss_mask,
            length=actual_length,
            epoch_idx=epoch_idx,
            compute_on_draft=self._compute_on_draft,
            drop_ratio_scale=self.drop_ratio_scale,
            total_epochs=total_epochs,
        )
        del target
        torch.cuda.empty_cache()

        # basic info
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        # Step 3: project the concatenated hidden states to the target hidden size
        hidden_states = self.draft_model.project_hidden_states(hidden_states)

        # Step 4: process kv cache, position ids
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        position_ids = self._prepare_position_ids(
            position_ids=position_ids,
            seq_length=seq_length,
            past_key_values_length=past_key_values_length,
            device=hidden_states.device,
            is_vlm=is_vlm,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
        )

        # Step 5: handle attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=hidden_states.device,
            )
        if self.attention_backend == "sdpa":
            attention_mask = self.draft_model.prepare_decoder_attention_mask(
                attention_mask=attention_mask,
                hidden_states=hidden_states,
                batch_size=batch_size,
                seq_length=seq_length,
                past_key_values_length=past_key_values_length,
            )

        # Step 6: run TTT with variable length
        # (actual_length computed above in Step 1)

        plosses = []
        acceptance_rates = []
        acces = []
        metric_corrects = []
        metric_denoms = []
        metric_losses = []
        metric_loss_denoms = []
        adapter = self._make_adapter()
        global_input_ids = input_ids
        if self.attention_backend in ["sdpa", "fa", "usp"]:
            cache_hidden = [[], []]
            past_key_values = None
        elif self.attention_backend == "flex_attention":
            cache_hidden = None
            past_key_values = DynamicCache()
        else:
            raise ValueError(f"Unknown attention backend: {self.attention_backend}")

        for idx in range(actual_length):
            state = adapter.step_view(
                idx=idx,
                ttt_length=actual_length,
                global_input_ids=global_input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
                position_ids=position_ids,
                hidden_states=hidden_states,
                target_p_padded=target_p_padded,
                target_p_on_draft_padded=target_p_on_draft_padded,
                target_token_ids_padded=target_token_ids_padded,
                position_mask=position_mask,
                seq_length=seq_length,
            )
            is_last = idx == actual_length - 1

            # Step 6.1: embed the input ids
            inputs_embeds = self.draft_model.embed_input_ids(state.input_ids)
            inputs_embeds = inputs_embeds.to(hidden_states.dtype)

            # Step 6.2: run the draft model backbone
            hidden_states_out = self.draft_model.backbone(
                input_embeds=inputs_embeds,
                hidden_states=state.hidden_states,
                cache_hidden=cache_hidden,
                attention_mask=state.attention_mask,
                position_ids=state.position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

            # update hidden states for next step
            hidden_states = hidden_states_out

            # Step 6.4: get logits
            logits = self.draft_model.compute_logits(hidden_states)

            # Step 6.5 + 6.6: metric and loss
            (
                acc,
                acceptance_rate,
                loss,
                correct,
                denom,
                metric_loss,
                loss_denom,
            ) = self._acc_and_loss(
                logits=logits,
                target_p=state.target_p,
                target_p_on_draft=state.target_p_on_draft,
                target_token_ids=state.target_token_ids,
                position_mask=state.position_mask,
                loss_mask=state.loss_mask,
                adapter=adapter,
            )
            acces.append(acc)
            acceptance_rates.append(acceptance_rate)
            plosses.append(loss)
            metric_corrects.append(correct)
            metric_denoms.append(denom)
            metric_losses.append(metric_loss)
            metric_loss_denoms.append(loss_denom)

            if not is_last:
                # Step 6.7: we need to update the loss mask
                global_input_ids = padding(global_input_ids, left=False)
                position_mask = padding(position_mask, left=False)
                loss_mask = padding(loss_mask, left=False)
                # Flex attention mask shrinking is handled inside attention module

        return (
            plosses,
            acceptance_rates,
            acces,
            metric_corrects,
            metric_denoms,
            metric_losses,
            metric_loss_denoms,
        )
