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

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from specforge.core.lk_loss import compute_acceptance_rate, compute_lk_loss
from specforge.core.loss import LogSoftmaxLoss
from specforge.modeling.draft import Eagle3DraftModel
from specforge.utils import padding


def _compute_loss_and_acceptance_rate(
    *,
    logits: torch.Tensor,
    target_p: torch.Tensor,
    target_p_on_draft: torch.Tensor,
    position_mask: torch.Tensor,
    lk_loss_type: Optional[str],
    kl_scale: float,
    kl_decay: float,
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
    """
    kl_loss = LogSoftmaxLoss.apply(logits, target_p, position_mask)

    with torch.set_grad_enabled(lk_loss_type is not None):
        acceptance_rate, log_acceptance_rate = compute_acceptance_rate(
            logits=logits,
            target_probs=target_p_on_draft,
            position_mask=position_mask,
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


class OnlineEagle3Model(nn.Module):
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

    def _acc_and_loss(
        self,
        *,
        logits: torch.Tensor,
        target_p: torch.Tensor,
        target_p_on_draft: torch.Tensor,
        target_token_ids: torch.Tensor,
        position_mask: torch.Tensor,
        loss_mask: torch.Tensor,
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
            acc = local_correct / local_denom

        acceptance_rate, loss = _compute_loss_and_acceptance_rate(
            logits=logits,
            target_p=target_p,
            target_p_on_draft=target_p_on_draft,
            position_mask=position_mask,
            lk_loss_type=self.lk_loss_type,
            kl_scale=self.kl_scale,
            kl_decay=self.kl_decay,
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

    def _prepare_position_ids(
        self,
        position_ids: Optional[torch.Tensor],
        *,
        seq_length: int,
        past_key_values_length: int,
        device: torch.device,
    ) -> torch.Tensor:
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
        return position_ids.view(-1, seq_length)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
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
            past_key_values: Optional autoregressive cache used to derive the
                past sequence length and continue draft decoding.
            position_ids: (batch, seq_len)
        """
        # Step 1: handle vocab size
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

        # Step 3: process kv cache, position ids and position ids
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
        global_input_ids = input_ids
        if self.attention_backend in ["sdpa", "fa"]:
            cache_hidden = [[], []]
            past_key_values = None
        elif self.attention_backend == "flex_attention":
            cache_hidden = None
            past_key_values = DynamicCache()
        else:
            raise ValueError(f"Unknown attention backend: {self.attention_backend}")

        for idx in range(self.length):
            step_slice = slice(idx, idx + seq_length)
            target_p = target_p_padded[:, step_slice, :].contiguous()
            target_p_on_draft = target_p_on_draft_padded[:, step_slice, :].contiguous()
            target_token_ids = target_token_ids_padded[:, step_slice].contiguous()
            is_last = idx == self.length - 1

            # Step 5.1: embed the input ids
            inputs_embeds = self.draft_model.embed_input_ids(global_input_ids)
            inputs_embeds = inputs_embeds.to(hidden_states.dtype)

            # Step 5.2: run the draft model backbone
            hidden_states_out = self.draft_model.backbone(
                input_embeds=inputs_embeds,
                hidden_states=hidden_states,
                cache_hidden=cache_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

            # update hidden states for next step
            hidden_states = hidden_states_out

            # Step 5.4: get logits
            logits = self.draft_model.compute_logits(hidden_states)

            # Step 5.5 + 5.6: metric and loss
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
                target_p=target_p,
                target_p_on_draft=target_p_on_draft,
                target_token_ids=target_token_ids,
                position_mask=position_mask,
                loss_mask=loss_mask,
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
def _compute_metric_counts(logits, target_token_ids, loss_mask, d2t):
    pred_draft_token_ids = logits.argmax(-1)
    pred_target_token_ids = pred_draft_token_ids + d2t[pred_draft_token_ids]
    correct = (
        (pred_target_token_ids == target_token_ids) * loss_mask.squeeze(-1)
    ).sum()
    denom = loss_mask.sum().clamp_min(1e-6)
    return correct, denom
