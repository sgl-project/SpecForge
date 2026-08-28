# coding=utf-8
"""Online training wrapper for single-layer MTP (architecture-independent).

MTP predicts the next token from the current token's embedding plus the target
model's last hidden state.  Shift is performed inside this wrapper; the target
backend is expected to return *raw* input_ids and last_hidden_states (DFlash
style), not the pre-shifted output of generate_eagle3_data.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_step_weights(beta: float = 0.6, num_steps: int = 3) -> List[float]:
    """Compute normalized exponential-decay step weights.

    alpha_k = beta^(k-1) / sum(beta^(j-1) for j=1..K)

    See FastMTP (arXiv:2509.18362), Equation 2.
    """
    raw = [beta**k for k in range(num_steps)]
    total = sum(raw)
    return [w / total for w in raw]


class OnlineMTPModel(nn.Module):
    """
    Online MTP training wrapper.

    Architecture-agnostic: any registered MTP draft module exposing
    ``forward(input_ids, hidden_states, attention_mask, position_ids)`` and a
    ``config`` with ``pad_token_id`` can be plugged in (see
    ``specforge/modeling/draft/mtp/``).

    Args:
        draft_model: The MTP draft model (e.g. ``modeling/draft/mtp/qwen3_5.py``).
        ploss_decay: Per-layer loss decay.  For a single MTP layer this is
            unused, but kept for multi-layer extension.
        num_speculative_steps: Number of teacher-forced draft steps per
            position (1 = single-step native fine-tune, the default).
        step_weight_beta: FastMTP exponential-decay base for per-step loss
            weights (only used when num_speculative_steps > 1).
        step_weights: Explicit per-step loss weights; overrides
            ``step_weight_beta`` when given.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        ploss_decay: float = 1.0,
        num_speculative_steps: int = 1,
        step_weight_beta: float = 0.6,
        step_weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.draft_model = draft_model
        self.ploss_decay = ploss_decay
        self.num_speculative_steps = num_speculative_steps
        self.step_weight_beta = step_weight_beta
        if step_weights is None and num_speculative_steps > 1:
            step_weights = compute_step_weights(step_weight_beta, num_speculative_steps)
        if step_weights is not None and len(step_weights) != num_speculative_steps:
            raise ValueError(
                f"step_weights has {len(step_weights)} entries but "
                f"num_speculative_steps={num_speculative_steps}"
            )
        self.step_weights = step_weights

    def _shift_for_next_token(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shift logits/labels/mask to match vLLM speculative decoding.

        In serving, the draft model's input_ids are the target input_ids shifted
        right by one (the draft fuses token x_{t+1} with target hidden state
        h_t) and it predicts the token after that (x_{t+2}). Training therefore
        uses:
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
        if self.num_speculative_steps > 1:
            return self._forward_multi_step(
                input_ids, hidden_states, loss_mask, attention_mask, position_ids
            )

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

        # Serving evaluates the shifted draft token x[t+1] at its own position
        # p[t+1], even though it is fused with the target hidden state h[t].
        # Preserve caller-supplied offsets (for packed/non-zero-based sequences)
        # and give the synthetic final token the next monotonic position.
        batch_size, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
        elif position_ids.shape != input_ids.shape:
            raise ValueError(
                "position_ids must have the same [batch, seq_len] shape as "
                f"input_ids; got {tuple(position_ids.shape)} and "
                f"{tuple(input_ids.shape)}"
            )
        shifted_position_ids = torch.cat(
            (position_ids[:, 1:], position_ids[:, -1:] + 1), dim=1
        )

        outputs = self.draft_model(
            input_ids=shifted_input_ids,
            hidden_states=hidden_states,
            attention_mask=shifted_attention_mask,
            position_ids=shifted_position_ids,
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

    def _forward_multi_step(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Teacher-forced multi-step MTP training (FastMTP-style).

        At step k the draft consumes the ground-truth token x[t+k+1] (embedded)
        fused with the previous step's MTP output (the target's last hidden
        state at step 0), and predicts x[t+k+2].  Each step's loss is weighted
        by the normalized exponential-decay schedule from ``step_weights``.

        Follows the reference implementation in vllm-project/speculators
        (models/mtp/core.py): hidden states are fed recursively and every step
        reuses the base window positions.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        effective_steps = min(self.num_speculative_steps, max(0, seq_len - 2))
        valid_len = seq_len - effective_steps - 1
        if valid_len <= 0 or effective_steps == 0:
            zero = torch.zeros((), device=device, requires_grad=True)
            empty = torch.zeros(batch_size, 0, device=device)
            return zero, [empty], [empty]

        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, dtype=torch.long, device=device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
        elif position_ids.shape != input_ids.shape:
            raise ValueError(
                "position_ids must have the same [batch, seq_len] shape as "
                f"input_ids; got {tuple(position_ids.shape)} and "
                f"{tuple(input_ids.shape)}"
            )
        step_pos_ids = position_ids[:, :valid_len]
        step_attn_mask = (
            attention_mask[:, :valid_len] if attention_mask is not None else None
        )

        total_loss = torch.zeros((), device=device)
        step0_corrects = step0_denoms = None
        current_hidden = hidden_states
        for step in range(effective_steps):
            step_input_ids = input_ids[:, step + 1 : step + 1 + valid_len]
            outputs = self.draft_model(
                input_ids=step_input_ids,
                hidden_states=current_hidden[:, :valid_len],
                attention_mask=step_attn_mask,
                position_ids=step_pos_ids,
                return_hidden=True,
            )
            logits = outputs.logits

            step_targets = input_ids[:, step + 2 : step + 2 + valid_len]
            step_mask = loss_mask[:, step + 2 : step + 2 + valid_len]
            unreduced = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                step_targets.reshape(-1),
                reduction="none",
            )
            flat_mask = step_mask.reshape(-1).float()
            weight = self.step_weights[step]
            total_loss = total_loss + weight * (unreduced * flat_mask).sum() / (
                flat_mask.sum().clamp_min(1)
            )

            if step == 0:
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    step0_corrects = (preds == step_targets).float() * step_mask
                    step0_denoms = step_mask.float()

            # The next step fuses its token embeddings with this step's output.
            current_hidden = outputs.hidden_states[-1]

        return total_loss, [step0_corrects], [step0_denoms]
