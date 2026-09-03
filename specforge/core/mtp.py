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
        objective_chunk_size: Token positions per lm_head+CE chunk.  Bounds the
            full-vocab logits to one chunk at a time (with activation
            checkpointing), instead of materializing [batch*seq, vocab] twice.
            0 disables chunking.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        ploss_decay: float = 1.0,
        objective_chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        if objective_chunk_size < 0:
            raise ValueError(
                f"objective_chunk_size must be >= 0, got {objective_chunk_size}"
            )
        self.draft_model = draft_model
        self.ploss_decay = ploss_decay
        self.objective_chunk_size = objective_chunk_size

    def _shift_for_next_token(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Shift labels/mask to match vLLM speculative decoding.

        In serving, the draft model's input_ids are the target input_ids shifted
        right by one (the draft fuses token x_{t+1} with target hidden state
        h_t) and it predicts the token after that (x_{t+2}). Training therefore
        uses:
          - draft input: input_ids[:, 1:]  (x_1..x_T, padded)
          - label:       x_2..x_T followed by a pad (length matches logits)
        """
        # x_2..x_T has length seq_len-2; pad one position so its length equals
        # seq_len-1 (same as the shifted hidden states). The padded position is
        # ignored.
        shift_labels = F.pad(input_ids[:, 2:], (0, 1), value=-100)
        shift_mask = F.pad(loss_mask[:, 2:], (0, 1), value=0)
        return shift_labels, shift_mask

    def _chunked_objective(
        self,
        shift_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply lm_head + cross-entropy in bounded chunks over positions.

        ``shift_hidden`` drops the final position up front (lm_head is
        position-wise, so this equals shifting the logits), and each chunk's
        logits are freed (or recomputed under checkpointing) before the next
        chunk runs.
        """
        shift_labels, shift_mask = self._shift_for_next_token(input_ids, loss_mask)
        batch, positions, hidden_size = shift_hidden.shape
        flat_hidden = shift_hidden.reshape(batch * positions, hidden_size)
        flat_labels = shift_labels.reshape(-1)
        flat_mask = shift_mask.reshape(-1)
        rows = flat_hidden.shape[0]
        chunk_size = self.objective_chunk_size if self.objective_chunk_size else rows
        lm_head = self.draft_model.mtp.lm_head

        def _chunk_terms(hidden_chunk, labels_chunk, mask_chunk):
            logits = lm_head(hidden_chunk)
            losses = F.cross_entropy(logits, labels_chunk, reduction="none")
            with torch.no_grad():
                corrects = (logits.argmax(dim=-1) == labels_chunk).float()
                corrects = corrects * mask_chunk
            return (losses * mask_chunk.float()).sum(), corrects, mask_chunk.sum()

        loss_num = None
        denom = None
        corrects_chunks = []
        for start in range(0, rows, chunk_size):
            end = min(start + chunk_size, rows)
            chunk_args = (
                flat_hidden[start:end],
                flat_labels[start:end],
                flat_mask[start:end],
            )
            if (
                chunk_size < rows
                and torch.is_grad_enabled()
                and flat_hidden.requires_grad
            ):
                chunk_terms = torch.utils.checkpoint.checkpoint(
                    _chunk_terms, *chunk_args, use_reentrant=False
                )
            else:
                chunk_terms = _chunk_terms(*chunk_args)
            chunk_loss_num, chunk_corrects, chunk_denom = chunk_terms
            loss_num = chunk_loss_num if loss_num is None else loss_num + chunk_loss_num
            denom = chunk_denom if denom is None else denom + chunk_denom
            corrects_chunks.append(chunk_corrects)

        loss = loss_num / denom.clamp_min(1)
        corrects = torch.cat(corrects_chunks).view(batch, positions)
        return loss, corrects, shift_mask.float()

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
        # position is padded because there is no x_{T+1}; its hidden state is
        # dropped before the objective in _chunked_objective.
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

        draft_hidden = self.draft_model.forward_hidden(
            input_ids=shifted_input_ids,
            hidden_states=hidden_states,
            attention_mask=shifted_attention_mask,
            position_ids=shifted_position_ids,
        )

        # The synthetic pad position has no label; drop it before the objective.
        loss, corrects, denoms = self._chunked_objective(
            draft_hidden[:, :-1], input_ids, loss_mask
        )

        # Single-layer MTP: wrap in length-1 lists for E1 evaluator compatibility.
        return loss, [corrects], [denoms]
