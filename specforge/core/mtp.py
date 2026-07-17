# coding=utf-8
"""Online training wrapper for single-layer MTP (Qwen3.5 style).

MTP predicts the next token from the current token's embedding plus the target
model's last hidden state.  Shift is performed inside this wrapper; the target
backend is expected to return *raw* input_ids and last_hidden_states (DFlash
style), not the pre-shifted output of generate_eagle3_data.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.modeling.draft.mtp import Qwen3_5MTPDraftModel


class OnlineMTPModel(nn.Module):
    """
    Online MTP training wrapper.

    Args:
        draft_model: The MTP draft model (e.g. Qwen3_5MTPDraftModel).
        ploss_decay: Per-layer loss decay.  For a single MTP layer this is
            unused, but kept for multi-layer extension.
    """

    def __init__(
        self,
        draft_model: Qwen3_5MTPDraftModel,
        ploss_decay: float = 1.0,
    ) -> None:
        super().__init__()
        self.draft_model = draft_model
        self.ploss_decay = ploss_decay

    def _shift_for_next_token(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shift logits/labels/mask to match vLLM speculative decoding.

        In serving, the draft model's input_ids are the target input_ids shifted
        right by one (draft position t sees token x_{t+1}) and it predicts the
        token after that (x_{t+2}) using the target's post-norm hidden state at
        position t.  We therefore train with:
          - draft input: input_ids[:, 1:]  (x_1..x_T, padded)
          - label:       x_2..x_T followed by a pad (length matches logits)
        """
        shift_logits = logits[:, :-1, :].contiguous()
        # x_2..x_T has length seq_len-2; pad one position so its length equals
        # seq_len-1 (same as shift_logits). The padded position is ignored.
        shift_labels = F.pad(input_ids[:, 2:], (0, 1), value=-100).contiguous()
        shift_mask = F.pad(loss_mask[:, 2:], (0, 1), value=0).contiguous()
        return shift_logits, shift_labels, shift_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            input_ids: raw token ids, [batch, seq_len].
            hidden_states: target model last hidden states, [batch, seq_len, hidden].
            loss_mask: [batch, seq_len].
            attention_mask: optional padding mask, [batch, seq_len].
            position_ids: optional position ids, [batch, seq_len].

        Returns:
            loss: scalar weighted loss.
            acc_corrects: per-layer per-position correct tensors.
            acc_denoms: per-layer per-position denominator tensors.
        """
        # Draft input is the target sequence shifted right by one.  The last
        # position is padded because there is no x_{T+1}; the corresponding
        # logit is dropped in _shift_for_next_token.
        pad_token_id = getattr(self.draft_model.config, "pad_token_id", 0)
        shifted_input_ids = F.pad(input_ids[:, 1:], (0, 1), value=pad_token_id)

        # The padding mask must follow the same shift so the synthetic pad token
        # at the last position is not attended to.
        if attention_mask is not None:
            shifted_attention_mask = F.pad(attention_mask[:, 1:], (0, 1), value=0).to(
                attention_mask.dtype
            )
        else:
            shifted_attention_mask = None

        outputs = self.draft_model(
            input_ids=shifted_input_ids,
            hidden_states=hidden_states,
            attention_mask=shifted_attention_mask,
            position_ids=position_ids,
        )
        logits = outputs.logits

        shift_logits, shift_labels, shift_mask = self._shift_for_next_token(
            logits, input_ids, loss_mask
        )

        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        losses = F.cross_entropy(flat_logits, flat_labels, reduction="none")
        losses = losses * shift_mask.view(-1).float()
        loss = losses.sum() / shift_mask.sum().clamp_min(1)

        with torch.no_grad():
            preds = shift_logits.argmax(dim=-1)
            corrects = (preds == shift_labels).float() * shift_mask.float()
            denoms = shift_mask.float()

        # Single-layer MTP: wrap in length-1 lists for E1 evaluator compatibility.
        return loss, [corrects], [denoms]
