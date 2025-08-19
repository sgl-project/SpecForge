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
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers.cache_utils import DynamicCache

from specforge.modeling.draft import Eagle3DraftModel
from specforge.utils import padding


class Eagle3Model(nn.Module):
    pass


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
        target_model,
        draft_model: Eagle3DraftModel,
        length: int = 7,
        attention_backend="sdpa",
    ):
        """
        Args:
            target_model: the target model to extract hidden states.
            draft_model: the draft model to be trained.
            length: TTT length, it means how many turns to unroll during TTT.
        """
        super().__init__()
        self.target_model = target_model
        self.draft_model = draft_model
        self.length = length
        self.attention_backend = attention_backend

    @torch.no_grad()
    def _prepare_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        modified from: https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/cnets.py#L692
        Extract the hidden states from the target model outputs.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            loss_mask: (batch, seq_len)
            device: the device to run the target model, if None, use the input_ids device

        Returns:
            hidden_states: (batch, seq_len, 3*hidden_size)
            target: (batch, seq_len, vocab_size)
            loss_mask: (batch, seq_len)
            input_ids: (batch, seq_len)
        """

        if device is None:
            device = input_ids.device

        outputs = self.target_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # extract the aux hidden states
        # output_hidden_states = True will return the embedding output as well
        # so we have an offset of 1
        num_hidden_states = len(outputs.hidden_states)
        offset = 1
        num_layers = num_hidden_states - 1

        # Eagle3 uses 3 aux layers from layer 1, num_layers//2, num_layers-4
        low_aux_layer = 1 + offset
        mid_aux_layer = num_layers // 2 - 1 + offset
        last_aux_layer = num_layers - 4 + offset

        hidden_states0 = outputs.hidden_states[low_aux_layer]
        hidden_states1 = outputs.hidden_states[mid_aux_layer]
        hidden_states2 = outputs.hidden_states[last_aux_layer]

        hidden_states = torch.cat(
            (hidden_states0, hidden_states1, hidden_states2), dim=-1
        )

        # apply pading
        target = outputs.logits
        target = padding(target, left=False)
        input_ids = padding(input_ids, left=False)

        if target is not None:
            target = target.to(device)
            loss_mask = loss_mask[..., None]
            loss_mask = loss_mask.to(device)

        return hidden_states, target, loss_mask, input_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Online eagle model trainer, modified from: https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/cnets.py#L711

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            loss_mask: (batch, seq_len)
            past_key_values: We dont use this past_key_values in eagle3, but keep it for compatibility. We control kvcache by cache_hidden.
            position_ids: (batch, seq_len)
        """
        # Step 1: prepare data with the target model
        hidden_states, target, loss_mask, input_ids = self._prepare_data(
            input_ids, attention_mask, loss_mask
        )

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
        if position_ids is None:
            device = hidden_states.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

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
        vlosses = []
        acces = []
        if self.attention_backend == "sdpa":
            cache_hidden = [[], []]
            past_key_values = None
        elif self.attention_backend == "flex_attention":
            cache_hidden = None
            past_key_values = DynamicCache()

        for idx in range(self.length):
            is_last = idx == self.length - 1

            # Step 5.1: embed the input ids
            inputs_embeds = self.draft_model.embed_input_ids(input_ids)
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

            # Step 5.3: handle vocab size
            with torch.no_grad():
                target_head = target
                target_max_token = target_head.argmax(-1)
                target_mask = self.draft_model.t2d[target_max_token]
                target_mask = target_mask[..., None].int()
                position_mask = target_mask * loss_mask
                target_head = target_head[..., self.draft_model.t2d]
                target_head = target_head.float()
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()

            # update hidden states for next step
            hidden_states = hidden_states_out

            # Step 5.4: get logits
            logits = self.draft_model.compute_logits(hidden_states)
            logits = logits.float()

            # Step 5.5: calculate loss
            out_logp = nn.LogSoftmax(dim=2)(logits)
            plogp = target_p * out_logp
            loss = -torch.sum(position_mask * plogp, 2).mean()

            # Step 5.6: record metrics
            plosses.append(loss)
            with torch.no_grad():
                acces.append(
                    (
                        (logits.argmax(-1) == target_p.argmax(-1))
                        * position_mask.squeeze(-1)
                    )
                    .sum()
                    .item()
                    / (loss_mask.sum().item() + 1e-6)
                )

            if not is_last:
                # Step 5.7: we need to update the loss mask
                input_ids = padding(input_ids, left=False)
                target = padding(target, left=False)
                loss_mask = padding(loss_mask, left=False)
                if self.attention_backend == "sdpa":
                    ind = torch.arange(seq_length, device=attention_mask.device)
                    ind0 = ind[idx:]
                    ind1 = ind[: seq_length - idx]
                    attention_mask[:, :, ind0, ind1] = torch.finfo(
                        attention_mask.dtype
                    ).min
                # Flex attention mask shirnking is handled inside attention module
        return plosses, vlosses, acces


class OfflineEagle3Model(Eagle3Model):
    """
    In sgl-spec, we implement offline/online training.
    Offline training means we have the target hidden_states available before training.
    """

    def __init__(
        self, target_head, draft_model, length: int = 7, attention_backend="sdpa"
    ):
        """
        Args:
            target_head: the target head to process the target hidden states.
            draft_model: the draft model to be trained.
            length: TTT length, it means how many turns to unroll during TTT.
        """
        super().__init__()
        self.draft_model = draft_model
        self.target_head = target_head
        self.length = length
        self.attention_backend = attention_backend

    def forward(
        self,
        input_ids,
        attention_mask,
        loss_mask,
        target,
        hidden_states,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass for the offline Eagle3 model.
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            loss_mask: (batch, seq_len, 1)
            target: (batch, seq_len, vocab_size)
            hidden_states: (batch, seq_len, 3*hidden_size)
            past_key_values: We dont use this past_key_values in eagle3, but keep it
                for compatibility. We control kvcache by cache_hidden.
            position_ids: (batch, seq_len)

        Returns:
            plosses: List of losses for each TTT step.
            vlosses: List of validation losses (not used in this implementation).
            acces: List of accuracies for each TTT step.
        """
        # basic info
        target = self.target_head(target)
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        # Step 1: project the concatenated hidden states to the target hidden size
        hidden_states = self.draft_model.project_hidden_states(hidden_states)

        # Step 3: process kv cache, position ids and position ids
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = hidden_states.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

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
        vlosses = []
        acces = []
        if self.attention_backend == "sdpa":
            cache_hidden = [[], []]
            past_key_values = None
        elif self.attention_backend == "flex_attention":
            cache_hidden = None
            past_key_values = DynamicCache()

        for idx in range(self.length):
            is_last = idx == self.length - 1

            # Step 5.1: embed the input ids
            inputs_embeds = self.draft_model.embed_input_ids(input_ids)
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

            # Step 5.3: handle vocab size
            with torch.no_grad():
                target_head = target
                target_max_token = target_head.argmax(-1)
                target_mask = self.draft_model.t2d[target_max_token]
                target_mask = target_mask[..., None].int()
                position_mask = target_mask * loss_mask
                target_head = target_head[..., self.draft_model.t2d]
                target_head = target_head.float()
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()

            # update hidden states for next step
            hidden_states = hidden_states_out

            # Step 5.4: get logits
            logits = self.draft_model.compute_logits(hidden_states)
            logits = logits.float()

            # Step 5.5: calculate loss
            out_logp = nn.LogSoftmax(dim=2)(logits)
            plogp = target_p * out_logp
            loss = -torch.sum(position_mask * plogp, 2).mean()

            # Step 5.6: record metrics
            plosses.append(loss)
            with torch.no_grad():
                acces.append(
                    (
                        (logits.argmax(-1) == target_p.argmax(-1))
                        * position_mask.squeeze(-1)
                    )
                    .sum()
                    .item()
                    / (loss_mask.sum().item() + 1e-6)
                )

            if not is_last:
                # Step 5.7: we need to update the loss mask
                input_ids = padding(input_ids, left=False)
                target = padding(target, left=False)
                loss_mask = padding(loss_mask, left=False)
                if self.attention_backend == "sdpa":
                    ind = torch.arange(seq_length, device=attention_mask.device)
                    ind0 = ind[idx:]
                    ind1 = ind[: seq_length - idx]
                    attention_mask[:, :, ind0, ind1] = torch.finfo(
                        attention_mask.dtype
                    ).min
                # Flex attention mask shirnking is handled inside attention module
        return plosses, vlosses, acces
