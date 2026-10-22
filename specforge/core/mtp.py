# coding=utf-8
"""Online training wrapper for single-layer MTP (architecture-independent).

MTP predicts the next token from the current token's embedding plus the target
model's last hidden state.  Shift is performed inside this wrapper; the target
backend is expected to return *raw* input_ids and last_hidden_states (DFlash
style), not the pre-shifted output of generate_eagle3_data.

With ``num_steps > 1`` the wrapper trains the same module as a chain
(EAGLE3-style TTT): step j feeds the previous step's draft hidden state back
in, shifts the teacher-forced input/labels j tokens further, and supervises
x_{t+2+j}.  Each later step can attend to the preceding steps at its row.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlineMTPModel(nn.Module):
    """
    Online MTP training wrapper.

    Architecture-agnostic: any registered MTP draft module exposing
    ``forward_hidden(input_ids, hidden_states, attention_mask, position_ids)``,
    a position-wise ``mtp.lm_head``, and a ``config`` with ``pad_token_id`` can
    be plugged in (see
    ``specforge/modeling/draft/mtp/``).

    Args:
        draft_model: The MTP draft model (e.g. ``modeling/draft/mtp/qwen3_5.py``).
        ploss_decay: Loss decay across the chain: step i is weighted
            ``ploss_decay ** i`` (step 0 keeps weight 1.0).
        num_steps: Chain length.  1 (default) keeps the original single-step
            teacher-forced objective; 3 mirrors NEXTN's 3-step chain.
        objective_chunk_size: Token positions per lm_head+CE chunk.  Bounds the
            full-vocab logits to one chunk at a time (with activation
            checkpointing), instead of materializing [batch*seq, vocab] twice.
            0 disables chunking. Configured by ``training.mtp_objective_chunk_size``
            in YAML. This bounds objective intermediates only; backbone
            activations, attention, weights, and optimizer state are unchanged.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        ploss_decay: float = 1.0,
        objective_chunk_size: int = 4096,
        num_steps: int = 1,
    ) -> None:
        super().__init__()
        if objective_chunk_size < 0:
            raise ValueError(
                f"objective_chunk_size must be >= 0, got {objective_chunk_size}"
            )
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        self.draft_model = draft_model
        self.ploss_decay = ploss_decay
        self.objective_chunk_size = objective_chunk_size
        self.num_steps = num_steps

    def _shift_for_next_token(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Shift labels/mask to match vLLM speculative decoding.

        In serving, the draft model's input_ids are the target input_ids shifted
        right by one (the draft fuses token x_{t+1} with target hidden state
        h_t) and it predicts the token after that (x_{t+2}). Training therefore
        uses:
          - draft input: input_ids[:, 1:]  (x_1..x_T, padded)
          - label:       x_2..x_T followed by a pad (length matches logits)

        Chain step ``offset`` keeps the same convention one token further per
        step: label row p is x_{p + 2 + offset}.
        """
        # x_2..x_T has length seq_len-2; pad one position so its length equals
        # seq_len-1 (same as the shifted hidden states). The padded position is
        # ignored.
        shift_labels = F.pad(input_ids[:, 2 + offset :], (0, 1 + offset), value=-100)
        shift_mask = F.pad(loss_mask[:, 2 + offset :], (0, 1 + offset), value=0)
        return shift_labels, shift_mask

    def _chunked_objective(
        self,
        shift_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply lm_head + cross-entropy in bounded chunks over positions.

        Only rows with nonzero supervision weights enter the position-wise
        lm_head. The backbone still processes the full context. Accuracy is
        scattered back to the original row layout for chain metrics.

        ``shift_hidden`` drops the final position up front (lm_head is
        position-wise, so this equals shifting the logits), and each chunk's
        logits are freed (or recomputed under checkpointing) before the next
        chunk runs.  ``offset`` selects the chain step's label shift.
        """
        shift_labels, shift_mask = self._shift_for_next_token(
            input_ids, loss_mask, offset
        )
        batch, positions, hidden_size = shift_hidden.shape
        flat_hidden = shift_hidden.reshape(batch * positions, hidden_size)
        flat_labels = shift_labels.reshape(-1)
        flat_mask = shift_mask.reshape(-1)
        full_rows = flat_hidden.shape[0]
        chunk_size = self.objective_chunk_size or full_rows
        lm_head = self.draft_model.mtp.lm_head

        supervised_rows = flat_mask.nonzero(as_tuple=True)[0]
        if supervised_rows.numel() == 0:
            # Keep a zero-weight row connected to the hidden states and head.
            # This preserves zero (rather than absent) gradients on empty ranks.
            supervised_rows = torch.zeros(1, dtype=torch.long, device=flat_mask.device)
        flat_hidden = flat_hidden.index_select(0, supervised_rows)
        flat_labels = flat_labels.index_select(0, supervised_rows)
        flat_mask = flat_mask.index_select(0, supervised_rows)
        rows = flat_hidden.shape[0]

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
                # Preserve the original checkpoint policy for this batch.
                chunk_size < full_rows
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
        selected_corrects = torch.cat(corrects_chunks)
        corrects = selected_corrects.new_zeros(full_rows)
        corrects.index_copy_(0, supervised_rows, selected_corrects)
        corrects = corrects.view(batch, positions)
        return loss, corrects, shift_mask.float()

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        """
        Args:
            input_ids: raw token ids, [batch, seq_len].
            hidden_states: target model last hidden states, [batch, seq_len, hidden].
            loss_mask: [batch, seq_len].
            attention_mask: optional padding mask, [batch, seq_len].
            position_ids: optional position ids, [batch, seq_len].

        Returns:
            loss: scalar chain loss ``sum_i ploss_decay**i * ploss_i``.
            acc_corrects: per-step per-position correct tensors (step i is the
                i-th chain step, input shifted by 1+i tokens).
            acc_denoms: per-step per-position denominator tensors.
            plosses: per-step detached scalar losses.
            loss_denoms: per-step detached scalar loss denominators.
        """
        pad_token_id = getattr(self.draft_model.config, "pad_token_id", 0)

        # Serving evaluates the shifted draft token x[t+1+step] at its own
        # position p[t+1+step], even though it is fused with the previous
        # step's hidden state.  Preserve caller-supplied offsets (for
        # packed/non-zero-based sequences) and give the synthetic final token
        # the next monotonic position.
        batch_size, seq_len = input_ids.shape
        if seq_len <= self.num_steps:
            raise ValueError("MTP sequence length must exceed num_steps")
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

        # Chain mode (num_steps > 1) runs an EAGLE3-style TTT loop: step j
        # feeds the previous step's output hidden states back in, shifts the
        # teacher-forced input one token further, and attends across steps
        # through the shared cache.  Single-step mode passes no cache so the
        # module keeps the plain (numerically unchanged) attention path.
        cache_hidden = [[], []] if self.num_steps > 1 else None

        acc_corrects: List[torch.Tensor] = []
        acc_denoms: List[torch.Tensor] = []
        plosses: List[torch.Tensor] = []
        loss_denoms: List[torch.Tensor] = []
        step_hidden = hidden_states
        for step in range(self.num_steps):
            # Draft input at chain step ``step`` is the target sequence shifted
            # right by 1+step: row p fuses token x_{p+1+step} with the hidden
            # state the previous step produced for that row.  The tail is
            # padded because there is no x_{T+1+step}.
            shifted_input_ids = F.pad(
                input_ids[:, 1 + step :], (0, 1 + step), value=pad_token_id
            )

            # The padding mask must follow the same shift so the synthetic pad
            # tokens at the tail are not attended to.
            if attention_mask is not None:
                shifted_attention_mask = F.pad(
                    attention_mask[:, 1 + step :], (0, 1 + step), value=0
                ).to(attention_mask.dtype)
            else:
                shifted_attention_mask = None

            forward_kwargs = (
                {"cache_hidden": cache_hidden} if cache_hidden is not None else {}
            )
            draft_hidden = self.draft_model.forward_hidden(
                input_ids=shifted_input_ids,
                hidden_states=step_hidden,
                attention_mask=shifted_attention_mask,
                position_ids=shifted_position_ids + step,
                **forward_kwargs,
            )

            # The synthetic pad positions have no label; drop them before the
            # objective (retained row p predicts x_{p+2+step}).
            ploss, corrects, denoms = self._chunked_objective(
                draft_hidden[:, :-1], input_ids, loss_mask, offset=step
            )
            acc_corrects.append(corrects)
            acc_denoms.append(denoms)
            plosses.append(ploss)
            loss_denoms.append(denoms.sum())
            step_hidden = draft_hidden

        weights = [self.ploss_decay**step for step in range(self.num_steps)]
        loss = sum(weight * ploss for weight, ploss in zip(weights, plosses))

        # Per-step lists indexed by chain depth (EAGLE3 telemetry contract).
        return (
            loss,
            acc_corrects,
            acc_denoms,
            [ploss.detach() for ploss in plosses],
            [denom.detach() for denom in loss_denoms],
        )
