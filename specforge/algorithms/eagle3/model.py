# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in HuggingFace Transformers.
# Portions of this code are adapted from:
#   - https://github.com/EleutherAI/gpt-neox (Apache License 2.0)
#   - https://github.com/huggingface/transformers (Apache License 2.0)
#   - https://github.com/SafeAILab/EAGLE (Apache License 2.0)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""EAGLE3 training model implementation."""

import logging
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from specforge.core.compact_teacher import (
    DEFAULT_VOCAB_CHUNK_SIZE,
    compute_target_p_padded_from_hidden,
)
from specforge.core.eagle3_adapters import BackendAdapter, SdpaLikeAdapter, UspAdapter
from specforge.core.lk_loss import compute_acceptance_rate, compute_lk_loss
from specforge.core.loss import LogSoftmaxLoss
from specforge.modeling.draft import Eagle3DraftModel
from specforge.utils import padding

logger = logging.getLogger(__name__)


class Eagle3Model(nn.Module):
    pass


def _compute_loss_and_acceptance_rate(
    *,
    logits: torch.Tensor,
    target_p: torch.Tensor,
    target_p_on_draft: torch.Tensor,
    position_mask: torch.Tensor,
    lk_loss_type: Optional[str],
    kl_scale: float,
    kl_decay: float,
    reduce_metrics_fn: Optional[
        Callable[..., Tuple[torch.Tensor, torch.Tensor]]
    ] = None,
    reduce_loss_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute step loss and acceptance rate for KL/LK objectives.

    Args:
        logits: Draft model logits for current step.
        target_p: Renormalized target distribution over draft-vocab tokens (for KL).
        target_p_on_draft: Original target probabilities restricted to draft-vocab tokens (for acceptance terms).
        position_mask: Mask indicating valid tokens for loss/metric aggregation.
        lk_loss_type: LK objective mode (`None`, `"alpha"`, or `"lambda"`).
        kl_scale: Scale factor for lambda LK mixing weight.
        kl_decay: Decay factor for lambda LK mixing weight.
        reduce_metrics_fn: Optional distributed reducer for metric numer/denom.
        reduce_loss_fn: Optional distributed reducer for KL loss.
    """
    kl_loss = LogSoftmaxLoss.apply(logits, target_p, position_mask)
    if reduce_loss_fn is not None:
        kl_loss = reduce_loss_fn(kl_loss)

    with torch.set_grad_enabled(lk_loss_type is not None):
        acceptance_rate, log_acceptance_rate = compute_acceptance_rate(
            logits=logits,
            target_probs=target_p_on_draft,
            position_mask=position_mask,
            reduce_fn=reduce_metrics_fn,
        )

    if lk_loss_type is None:
        loss = kl_loss
    else:
        loss = compute_lk_loss(
            kl_loss=kl_loss,
            acceptance_rate=acceptance_rate,
            log_acceptance_rate=log_acceptance_rate,
            lk_loss_type=lk_loss_type,
            kl_scale=kl_scale,
            kl_decay=kl_decay,
        )
    return acceptance_rate.detach(), loss


class OnlineEagle3Model(Eagle3Model):
    """
    In sgl-spec, we implement offline/online training.
    Online training means we have the target hidden_states available during training.
    Eagle3 using test time training technique (TTT) to train the draft model.
    1. We first extract the hidden states from the target model.
    2. Then concatenate the hidden states from 3 aux layers (layer 1, layer num_layers//2, layer num_layers-4).
    3. We project the concatenated hidden states to the target hidden size. from (batch, seq_len, 3*hidden_size) to (batch, seq_len, hidden_size)
    4. We concat the projected hidden states and embedding output as the input for the draft model.
    5. finally, we run TTT to train the draft model. input size is (batch, seq_len, hidden_size * 2)
    """

    def __init__(
        self,
        draft_model: Eagle3DraftModel,
        length: int = 7,
        attention_backend="sdpa",
        lk_loss_type: Optional[str] = None,
        kl_scale: float = 1.0,
        kl_decay: float = 1.0,
    ):
        """
        Args:
            draft_model: the draft model to be trained.
            length: TTT length, it means how many turns to unroll during TTT.
            lk_loss_type: LK loss objective type. One of {"lambda", "alpha"}.
            kl_scale: Initial KL weight scale for lambda LK loss.
            kl_decay: Decay factor for adaptive KL weight in lambda LK loss.
        """
        super().__init__()
        self.draft_model = draft_model
        self.length = length
        self.attention_backend = attention_backend
        self.lk_loss_type = lk_loss_type
        self.kl_scale = kl_scale
        self.kl_decay = kl_decay

    def _make_adapter(self) -> BackendAdapter:
        if self.attention_backend == "usp":
            return UspAdapter(self)
        return SdpaLikeAdapter(self)

    def _acc_and_loss(
        self,
        *,
        logits: torch.Tensor,
        target_p: torch.Tensor,
        target_p_on_draft: torch.Tensor,
        target_token_ids: torch.Tensor,
        position_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        adapter: BackendAdapter,
        loss_scale: float = 1.0,
        full_positions: Optional[int] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
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
        if loss_scale != 1.0:
            # The trimmed loss kernel averages over n_sup supervised positions, but
            # the full-length semantics average over L; rescale to recover it. Only
            # valid when lk_loss_type is None (a plain KL loss is linearly scalable).
            loss = loss * loss_scale
        loss_denom = torch.tensor(
            logits.shape[0]
            * (full_positions if full_positions is not None else logits.shape[1]),
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

    def _prepare_position_ids(
        self,
        position_ids: Optional[torch.Tensor],
        *,
        seq_length: int,
        past_key_values_length: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        # USP owns its position-ID layout: the offline collator keeps the
        # all-to-all-expanded Ulysses dimension for the backend adapter to slice.
        # It therefore must not be reshaped or validated against this rank's
        # local hidden-state sequence length.
        if self.attention_backend == "usp":
            return position_ids

        if position_ids is None:
            return (
                torch.arange(
                    past_key_values_length,
                    seq_length + past_key_values_length,
                    dtype=torch.long,
                    device=device,
                )
                .unsqueeze(0)
                .view(-1, seq_length)
            )

        position_ids = position_ids.long()
        if position_ids.ndim not in (2, 3):
            raise ValueError(
                "position_ids must have shape [batch, seq_length] or "
                "[axes, batch, seq_length], got "
                f"{tuple(position_ids.shape)}"
            )
        if position_ids.shape[-1] != seq_length:
            raise ValueError(
                "position_ids final dimension must equal seq_length="
                f"{seq_length}, got {tuple(position_ids.shape)}"
            )
        if position_ids.ndim == 3:
            return position_ids
        return position_ids.reshape(-1, seq_length)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        target_hidden_for_compact: Optional[torch.Tensor] = None,
        target_head_weight: Optional[torch.Tensor] = None,
        compact_teacher_chunk_size: int = DEFAULT_VOCAB_CHUNK_SIZE,
        trim_loss_positions: bool = False,
        trim_backbone_rows: bool = False,
        trim_backbone_rows_max_density: float = 0.35,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        """
        Online eagle model trainer, modified from: https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/cnets.py#L711

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            loss_mask: (batch, seq_len)
            past_key_values: We dont use this past_key_values in eagle3, but keep it for compatibility. We control kvcache by cache_hidden.
            position_ids: ``[batch, seq_len]`` or ``[axes, batch, seq_len]``.
            target_hidden_for_compact, target_head_weight, compact_teacher_chunk_size:
                when the first two are given, the padded teacher is built from hidden
                states in draft-vocab space and ``target`` is ignored.
            trim_loss_positions: compute the teacher, draft logits and loss only at
                supervised positions when the batch/objective supports it.
            trim_backbone_rows: additionally run the draft backbone at TTT steps
                >= 1 only on the rows that can still emit loss at that or a later
                step (requires trim_loss_positions; step 0 stays full-length as it
                provides the cross-position K/V context).
            trim_backbone_rows_max_density: steps whose kept-row set exceeds this
                fraction of the chunk run the full-length path instead -- trimming
                costs more per row, so it stops paying once few rows are dropped.
        """
        adapter = self._make_adapter()
        # Step 1: handle vocab size
        if target_hidden_for_compact is not None:
            (
                target_p_padded,
                target_p_on_draft_padded,
                target_token_ids_padded,
                position_mask,
            ) = compute_target_p_padded_from_hidden(
                hidden=target_hidden_for_compact,
                lm_head_weight=target_head_weight,
                t2d=self.draft_model.t2d,
                loss_mask=loss_mask,
                length=self.length,
                chunk_size=compact_teacher_chunk_size,
            )
            del target_hidden_for_compact
            trim_pack = None
        else:
            # A-level trim: with batch==1 and no lk_loss, compute the teacher only at
            # supervised positions; fall back to the full path otherwise. Under USP
            # the backbone keeps running on this rank's own chunk (usp_chunk_size =
            # local_len - ttt_length); the local buffer's ttt_length overlap tail may
            # only act as teacher positions for own-chunk rows, never emit loss rows
            # itself (those rows belong to the next rank), so the per-step row sets
            # are bounded by chunk_len.
            # chunk_len must come from the SAME source the full path uses for its
            # slicing/normalization: the hidden-state sequence length (the loss
            # kernel means over backbone rows). loss_mask can carry an extra
            # zero-padded slot in the offline pipeline, so deriving from it would
            # be off by one (wrong rows, wrong denominator, and under USP a
            # backbone length that disagrees with full-path ranks).
            trim_chunk_len = adapter.backbone_row_count(
                seq_length=hidden_states.shape[1], ttt_length=self.length
            )
            _trim_ok = (
                trim_loss_positions
                and self.lk_loss_type is None
                and loss_mask.shape[0] == 1
                and trim_chunk_len > 0
            )
            trim_pack = None
            if _trim_ok:
                # Returns None when no supervised position can reach any row
                # (e.g. supervision only beyond the reachable window); then we
                # fall through to the full path below.
                trim_pack = _build_trim_pack(
                    target,
                    self.draft_model.t2d,
                    loss_mask,
                    self.length,
                    chunk_len=trim_chunk_len,
                    backbone_rows=trim_backbone_rows,
                )
            if trim_pack is not None:
                target_p_padded = None
                target_p_on_draft_padded = None
                target_token_ids_padded = None
                position_mask = trim_pack["position_mask_sup"]
            else:
                (
                    target_p_padded,
                    target_p_on_draft_padded,
                    target_token_ids_padded,
                    position_mask,
                ) = _compute_target_p_padded(
                    target=target,
                    t2d=self.draft_model.t2d,
                    loss_mask=loss_mask,
                    length=self.length,
                )
            del target
        torch.cuda.empty_cache()

        # basic info
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        # Step 2: project the concatenated hidden states to the target hidden size
        hidden_states = self.draft_model.project_hidden_states(hidden_states)

        # Step 3: process the KV cache and position IDs
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        position_ids = self._prepare_position_ids(
            position_ids=position_ids,
            seq_length=seq_length,
            past_key_values_length=past_key_values_length,
            device=hidden_states.device,
        )

        # Step 4: handle attention mask
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

        # Step 5: run TTT
        plosses = []
        acceptance_rates = []
        acces = []
        metric_corrects = []
        metric_denoms = []
        metric_losses = []
        metric_loss_denoms = []
        # for sequence paralle, position mask and input ids will split by sequence dim, need to keep origin for ttt shift
        global_input_ids = input_ids
        if self.attention_backend in ["sdpa", "fa", "usp"]:
            cache_hidden = [[], []]
            past_key_values = None
        elif self.attention_backend == "flex_attention":
            cache_hidden = None
            past_key_values = DynamicCache()
        else:
            raise ValueError(f"Unknown attention backend: {self.attention_backend}")

        b_enabled = trim_pack is not None and "b_rows_steps" in trim_pack
        if trim_backbone_rows and self.attention_backend == "usp":
            # Backbone-row trimming changes the backbone's collective pattern
            # (one K/V all-gather instead of per-step ring attention), so the
            # decision must be rank-uniform: a rank whose pack fell back to
            # None (no reachable supervision) would otherwise keep issuing
            # ring collectives that trimmed ranks never join -> deadlock.
            # Agree by MIN across the sequence-parallel group; on disagreement
            # every rank drops to the A-level path, whose backbone collectives
            # are identical to the full path's.
            flag = torch.tensor(1 if b_enabled else 0, device=hidden_states.device)
            torch.distributed.all_reduce(
                flag, op=torch.distributed.ReduceOp.MIN, group=adapter.sp_group
            )
            if int(flag.item()) == 0:
                # b_enabled alone gates every B-level read below; the pack's
                # b_* entries are simply never consulted again.
                b_enabled = False
        if b_enabled and position_ids is not None and position_ids.dim() == 3:
            # Multimodal RoPE carries [axes, batch, seq] position ids; the
            # row-selection below would slice the batch axis. Fail fast here
            # (the attention-level guard cannot be reached for this layout).
            raise NotImplementedError(
                "trim_backbone_rows does not support multimodal RoPE (mrope) "
                "drafts (3D position_ids)"
            )
        if b_enabled:
            # The trimmed path builds its causal mask from position VALUES
            # (row at position p attends step-0 keys 0..p), which matches the
            # full path's index-based mask only when positions are the row
            # indices themselves (plain arange locally, or the collator's
            # contiguous global arange under USP). Packed sequences with
            # per-segment position resets would silently diverge -- reject.
            _pos = position_ids[:, : trim_pack["full_len"]]
            if not bool((_pos.diff(dim=-1) == 1).all()):
                raise NotImplementedError(
                    "trim_backbone_rows requires contiguous ascending "
                    "position_ids (packed/segment-reset positions are not "
                    "supported)"
                )
        # Persistent across steps: the trimmed-attention path stashes the
        # (possibly all-gathered) step-0 K/V here at the first trimmed step.
        # k0_len is the step-0 K/V row count (global length under USP).
        trim_ctx = (
            {
                "k0": None,
                "v0": None,
                "k0_len": trim_pack["full_len"] * adapter.sp_world_size,
            }
            if b_enabled
            else None
        )
        # Cost guard. A trimmed step swaps flash attention for an explicit
        # masked-matmul kernel that costs more per row, so trimming only pays
        # while the kept-row sets are small. Measured on this repo (8192-token
        # chunk, TTT=7, RTX 6000 Ada): flash attention costs ~16.7 us/row and
        # the trimmed kernel 28-62 us/row, so the step time breaks even near
        # 40% density and degrades to 2.4x slower at full supervision, where
        # no row can be dropped at all. Compare the two sides directly:
        #   trimmed rows  sum_i |R_i|   vs   full rows  (k-1) * chunk
        # and keep the B path only while the former stays under the budget.
        # The decision is per sample, not per step: |R_i| barely varies across
        # steps for contiguous supervision (it shrinks by one row per
        # supervised run per step), and a mixed full/trimmed unroll would need
        # per-backend K/V layout conversion plus an extra head-dimension
        # gather under USP -- new collectives for a safety guard is a bad
        # trade.
        if b_enabled:
            chunk = trim_pack["full_len"]
            rows_sum = sum(
                int(trim_pack["b_rows_steps"][i].numel()) for i in range(1, self.length)
            )
            budget = trim_backbone_rows_max_density * (self.length - 1) * chunk
            trim_pays = rows_sum <= budget
            if self.attention_backend == "usp":
                # Same reason as the b_enabled agreement above: per-rank row
                # counts differ, so ranks can land on opposite sides of the
                # budget, and a mixed trimmed/full unroll across ranks would
                # mismatch collectives. Agree by MIN -- running the full path
                # is always correct, it only forgoes savings.
                flag = torch.tensor(1 if trim_pays else 0, device=hidden_states.device)
                torch.distributed.all_reduce(
                    flag, op=torch.distributed.ReduceOp.MIN, group=adapter.sp_group
                )
                trim_pays = bool(flag.item())
            if not trim_pays:
                # Surface it: the user asked for row trimming and did not get
                # it, and the reason (dense supervision) is a property of their
                # data, not a bug.
                logger.info(
                    "trim_backbone_rows skipped for this sample: kept rows "
                    "%d exceed the budget %d (%.2f x %d steps x %d chunk); "
                    "the trimmed attention kernel costs more per row than "
                    "flash attention, so trimming this sample would be slower",
                    rows_sum,
                    int(budget),
                    trim_backbone_rows_max_density,
                    self.length - 1,
                    chunk,
                )
                b_enabled = False
        for idx in range(self.length):
            b_active = b_enabled and idx >= 1
            if b_active:
                # B-level: this step's backbone runs only R_i = the union of all
                # rows that can still emit loss at this or a later step. hidden
                # chains through per-step row maps (R_i is nested in R_{i-1});
                # input_ids/positions follow the absolute rows so RoPE and the
                # causal mask stay position-correct inside the trimmed path.
                rows_b = trim_pack["b_rows_steps"][idx]
                # R_i nested in R_{i-1}: positions of R_i inside the previous
                # step's compact output are exactly the last diagonal map.
                prev_sel = (
                    rows_b if idx == 1 else trim_pack["b_diag_sels"][idx][idx - 1]
                )
                step_hidden = hidden_states.index_select(1, prev_sel)
                step_input_ids = global_input_ids.index_select(1, rows_b)
                step_pos = position_ids[:, : trim_pack["full_len"]].index_select(
                    1, rows_b
                )
                step_attn = None  # the trimmed path builds its own causal mask
                trim_ctx["positions"] = step_pos
                trim_ctx["diag_sels"] = trim_pack["b_diag_sels"][idx]
            elif trim_pack is not None:
                # A-level: the teacher tables are already compacted to supervised
                # positions; the backbone runs exactly the same inputs as the full
                # path (per-rank chunk under USP, full length otherwise) and only
                # supervised rows go through logits/loss below.
                backbone = adapter.backbone_view(
                    row_count=trim_pack["full_len"],
                    global_input_ids=global_input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    hidden_states=hidden_states,
                )
                step_input_ids = backbone.input_ids
                step_hidden = backbone.hidden_states
                step_attn = backbone.attention_mask
                step_pos = backbone.position_ids
            else:
                state = adapter.step_view(
                    idx=idx,
                    ttt_length=self.length,
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
                step_input_ids = state.input_ids
                step_hidden = state.hidden_states
                step_attn = state.attention_mask
                step_pos = state.position_ids
            is_last = idx == self.length - 1

            # Step 5.1: embed the input ids
            inputs_embeds = self.draft_model.embed_input_ids(step_input_ids)
            inputs_embeds = inputs_embeds.to(hidden_states.dtype)

            # Step 5.2: run the draft model backbone
            hidden_states_out = self.draft_model.backbone(
                input_embeds=inputs_embeds,
                hidden_states=step_hidden,
                cache_hidden=cache_hidden,
                attention_mask=step_attn,
                position_ids=step_pos,
                past_key_values=past_key_values,
                use_cache=True,
                **({"trim_rows_ctx": trim_ctx} if b_active else {}),
            )

            # update hidden states for next step
            hidden_states = hidden_states_out

            # Step 5.4 + 5.5 + 5.6: logits, metric and loss
            if trim_pack is not None:
                # A-level: only the rows that can carry loss at this step go through
                # norm + lm_head. Rows shift down by one per step (rows = sup - idx)
                # while the teacher/mask stay pinned at the supervised positions.
                rows_j = trim_pack["rows_steps"][idx]
                keep_j = trim_pack["keep_steps"][idx]
                nrows_j = trim_pack["nrows_steps"][idx]
                if b_active:
                    # backbone output is compacted to R_i rows; select loss rows
                    # by their position inside R_i rather than absolute index.
                    loss_sel = trim_pack["b_loss_sel"][idx]
                else:
                    loss_sel = rows_j
                logits = self.draft_model.compute_logits(
                    hidden_states.index_select(1, loss_sel)
                )
                pm_j = trim_pack["position_mask_sup"].index_select(1, keep_j)
                lm_j = trim_pack["loss_mask_sup"].index_select(1, keep_j)
                if nrows_j == 0:
                    # Dead step: no own-chunk row carries loss here (e.g. all local
                    # supervised positions sit in the USP overlap tail at this
                    # depth). The pack padded one dummy entry; zeroing its masks
                    # makes the contribution exactly zero while every rank still
                    # runs the same kernels and collective calls.
                    pm_j = torch.zeros_like(pm_j)
                    lm_j = torch.zeros_like(lm_j)
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
                    target_p=trim_pack["target_p_c"].index_select(1, keep_j),
                    target_p_on_draft=trim_pack["on_draft_c"].index_select(1, keep_j),
                    target_token_ids=trim_pack["token_ids_c"].index_select(1, keep_j),
                    position_mask=pm_j,
                    loss_mask=lm_j,
                    adapter=adapter,
                    loss_scale=nrows_j / trim_pack["full_len"],
                    full_positions=trim_pack["full_len"],
                )
            else:
                logits = self.draft_model.compute_logits(hidden_states)
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
                # Step 5.7: we need to update the loss mask
                global_input_ids = padding(global_input_ids, left=False)
                position_mask = padding(position_mask, left=False)
                loss_mask = padding(loss_mask, left=False)
                # Flex attention mask shirnking is handled inside attention module
        return (
            plosses,
            acceptance_rates,
            acces,
            metric_corrects,
            metric_denoms,
            metric_losses,
            metric_loss_denoms,
        )


def _compute_target_p_padded(target, t2d, loss_mask, length):
    with torch.no_grad():
        (
            target_p,
            target_p_on_draft,
            target_token_ids,
            position_mask,
        ) = _compute_target_p(
            target=target,
            t2d=t2d,
            loss_mask=loss_mask,
        )

        assert len(target_p.shape) == 3
        target_p_padded = F.pad(
            target_p,
            pad=(0, 0, 0, length),
            mode="constant",
            # For bitwise equality with previous code
            value=1 / target_p.shape[-1],
        )
        target_p_on_draft_padded = F.pad(
            target_p_on_draft,
            pad=(0, 0, 0, length),
            mode="constant",
            value=0.0,
        )
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


@torch.compile(dynamic=None)
def _compute_target_p(target, t2d, loss_mask):
    target_head = target.float()
    target_token_ids = target_head.argmax(-1)
    target_mask = t2d[target_token_ids]
    target_mask = target_mask[..., None].int()
    position_mask = target_mask * loss_mask
    draft_target_head = target_head[..., t2d]
    target_p = nn.Softmax(dim=2)(draft_target_head)
    target_logsumexp = torch.logsumexp(target_head, dim=-1, keepdim=True)
    target_p_on_draft = torch.exp(draft_target_head - target_logsumexp)
    target_p = target_p.detach()
    target_p_on_draft = target_p_on_draft.detach()
    target_token_ids = target_token_ids.detach()
    return target_p, target_p_on_draft, target_token_ids, position_mask


@torch.compile(dynamic=None)
def _compute_metric_acc(logits, target_token_ids, loss_mask, d2t):
    correct, denom = _compute_metric_counts(logits, target_token_ids, loss_mask, d2t)
    return correct / denom


@torch.compile(dynamic=None)
def _compute_metric_counts(logits, target_token_ids, loss_mask, d2t):
    pred_draft_token_ids = logits.argmax(-1)
    pred_target_token_ids = pred_draft_token_ids + d2t[pred_draft_token_ids]
    correct = (
        (pred_target_token_ids == target_token_ids) * loss_mask.squeeze(-1)
    ).sum()
    denom = loss_mask.sum().clamp_min(1e-6)
    return correct, denom


def _compute_target_p_eager(target, t2d, loss_mask, row_chunk=256):
    """Uncompiled variant of the teacher target_p computation.

    Kept uncompiled because the supervised-row count varies per batch, which would
    trigger repeated torch.compile recompilation. Mathematically identical to the
    compiled path; chunks over rows to bound the transient full-vocab fp32
    activation (which can reach several GB when the row count is large).
    """
    tps, tpds, toks, pms = [], [], [], []
    n = target.shape[1]
    for s in range(0, n, row_chunk):
        t = target[:, s : s + row_chunk].float()
        ids = t.argmax(-1)
        tm = t2d[ids][..., None].int()
        pms.append(tm * loss_mask[:, s : s + row_chunk])
        dth = t[..., t2d]
        tps.append(F.softmax(dth, dim=2).detach())
        lse = torch.logsumexp(t, dim=-1, keepdim=True)
        tpds.append(torch.exp(dth - lse).detach())
        toks.append(ids.detach())
    return (
        torch.cat(tps, 1),
        torch.cat(tpds, 1),
        torch.cat(toks, 1),
        torch.cat(pms, 1),
    )


def _build_trim_pack(
    target, t2d, loss_mask, length, chunk_len=None, backbone_rows=False
):
    """A-level trim (--trim-loss-positions): keep only the rows that can carry loss.

    Derivation of the per-step row set. On the full-length path the loop applies
    ``padding(..., left=False)`` to ``position_mask`` / ``loss_mask`` once per TTT
    step, so at step j the mask seen at row p is ``mask[p + j]``; meanwhile
    ``step_view`` slices the padded teacher so row p is supervised by the teacher at
    absolute position ``p + j``. A row therefore contributes at step j iff
    ``p + j`` is supervised, i.e. ``p = s - j`` for some supervised position s.

    So the rows shift *down* by one per step while the teacher/mask positions stay
    pinned at the supervised set:

        step j:  rows = {s - j : s in sup, s >= j}   teacher/mask at those s

    That makes the teacher cheap: it only ever has to be evaluated at ``sup``
    (no sliding window), and each step just drops the entries whose row would fall
    off the front of the sequence.

    ``chunk_len`` is the number of rows the backbone actually produces per step.
    On single-rank backends it equals the full local length L (default). Under USP
    the backbone runs on this rank's own chunk (``usp_chunk_size = L -
    ttt_length``) while the local buffer keeps a ``ttt_length`` overlap tail:
    tail positions may act as teachers for own-chunk rows at deeper steps, but
    tail rows belong to the next rank and must never emit loss here — hence the
    additional ``s - j < chunk_len`` bound on the row set, and ``full_len`` (the
    loss denominator) becomes ``chunk_len`` to match the full path's
    mean-over-chunk semantics.

    A step whose row set comes out empty (all supervised positions out of reach
    at that depth) is padded with one dummy entry and reported with
    ``nrows_steps[j] == 0``; the caller zeroes its masks so the step contributes
    exactly zero loss while kernel launches and collective calls stay aligned
    across ranks.

    batch == 1 only (online training uses batch == 1 per rank); the caller falls
    back to the full-length path otherwise.

    Returns a dict with, per step j: ``rows_steps[j]`` (row indices into the draft
    hidden states), ``keep_steps[j]`` (which supervised entries survive),
    ``nrows_steps[j]`` (real row count, 0 for dead steps), plus the teacher
    tables evaluated once at ``sup`` and the mask values at ``sup``.
    """
    with torch.no_grad():
        B, L = loss_mask.shape[0], loss_mask.shape[1]
        assert B == 1, "trim path requires batch==1"
        if chunk_len is None:
            chunk_len = L
        sup = loss_mask.view(-1).nonzero(as_tuple=False).squeeze(-1)  # [n_sup]
        # Positions beyond chunk_len + length - 2 can never supervise any row at
        # any step (would need j >= length); dropping them keeps every later
        # index within the hidden/target sequence range even when loss_mask is
        # longer than the hidden states (offline pipelines pad it by one).
        sup = sup[sup < chunk_len + length - 1]
        if sup.numel() == 0:
            # Nothing reachable at any step; tell the caller to use the full path.
            return None
        # Teacher is only ever needed at the supervised positions themselves.
        target_sel = target[:, sup]  # [1, n_sup, V_target]
        lm_sel = loss_mask[:, sup]
        target_p_c, on_draft_c, token_ids_c, pm_sup = _compute_target_p_eager(
            target_sel, t2d, lm_sel
        )
        lm_sup = loss_mask.view(-1)[sup].view(1, -1, 1)

        rows_steps, keep_steps, nrows_steps, raw_rows = [], [], [], []
        pad_idx = torch.zeros(1, dtype=sup.dtype, device=sup.device)
        for j in range(length):
            keep = (
                ((sup >= j) & (sup - j < chunk_len)).nonzero(as_tuple=False).squeeze(-1)
            )
            n = int(keep.numel())
            if backbone_rows:
                raw_rows.append(sup[keep] - j)
            if n == 0:
                # Dead step: pad with one dummy entry; the caller zeroes its
                # masks so it contributes nothing.
                keep = pad_idx
                rows = pad_idx
            else:
                rows = sup[keep] - j
            rows_steps.append(rows)
            keep_steps.append(keep)
            nrows_steps.append(n)

        extra = {}
        if backbone_rows and length > 1:
            # B-level (--trim-backbone-rows): the rows step i must forward are
            #   R_i = union_{j >= i} { s - j : s in sup, 0 <= s - j < chunk_len }
            # -- every row that can still emit loss at this or a later step
            # (hidden chains between steps, so a row needed at step j must be
            # forwarded at every step i <= j). The sets are nested
            # (R_i \subseteq R_{i-1}), which makes the per-step row maps plain
            # searchsorted lookups. Step 0 always runs full-length: it produces
            # the K/V context every later row attends to.
            b_rows_steps, b_diag_sels, b_loss_sel = {}, {}, {}
            prev = None
            for i in range(1, length):
                nonempty = [r for r in raw_rows[i:] if r.numel()]
                if nonempty:
                    uni = torch.unique(torch.cat(nonempty))
                else:
                    # Dead tail: keep one row from the previous set so the
                    # nested chain (and kernel/collective alignment across
                    # ranks) survives; its loss is zero-masked by nrows == 0.
                    uni = (prev[:1] if prev is not None else pad_idx).clone()
                b_rows_steps[i] = uni
                b_diag_sels[i] = {
                    e: torch.searchsorted(b_rows_steps[e], uni) for e in range(1, i)
                }
                b_loss_sel[i] = (
                    torch.searchsorted(uni, rows_steps[i])
                    if nrows_steps[i] > 0
                    else pad_idx
                )
                prev = uni
            extra = dict(
                b_rows_steps=b_rows_steps,
                b_diag_sels=b_diag_sels,
                b_loss_sel=b_loss_sel,
            )
        return dict(
            **extra,
            sup=sup,
            rows_steps=rows_steps,
            keep_steps=keep_steps,
            nrows_steps=nrows_steps,
            target_p_c=target_p_c,
            on_draft_c=on_draft_c,
            token_ids_c=token_ids_c,
            position_mask_sup=pm_sup,
            loss_mask_sup=lm_sup,
            full_len=chunk_len,
        )
