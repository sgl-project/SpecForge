# modified from

import json
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from sgl_spec.modeling.draft import Eagle3DraftModel
from sgl_spec.utils import padding


class Eagle3Model(nn.Module):
    pass


class OnlineEagle3Model(Eagle3Model):

    def __init__(self, target_model, draft_model: Eagle3DraftModel, length: int = 7):
        super().__init__()
        self.target_model = target_model
        self.draft_model = draft_model
        self.length = length

    @torch.no_grad()
    def _prepare_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract the hidden states from the target model outputs.
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
        input_ids,
        attention_mask,
        loss_mask,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
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
        cache_hidden = [[], []]

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
                ind = torch.arange(seq_length, device=attention_mask.device)
                ind0 = ind[idx:]
                ind1 = ind[: seq_length - idx]
                attention_mask[:, :, ind0, ind1] = torch.finfo(attention_mask.dtype).min
        return plosses, vlosses, acces


class OfflineEagle3Model(Eagle3Model):

    def __init__(self, draft_model, length: int = 7):
        super().__init__()
        self.draft_model = draft_model
        self.length = length

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
        # basic info
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
        cache_hidden = [[], []]

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
                ind = torch.arange(seq_length, device=attention_mask.device)
                ind0 = ind[idx:]
                ind1 = ind[: seq_length - idx]
                attention_mask[:, :, ind0, ind1] = torch.finfo(attention_mask.dtype).min
        return plosses, vlosses, acces
