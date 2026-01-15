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
from yunchang import EXTRACT_FUNC_DICT

from specforge.core.loss import LogSoftmaxLoss
from specforge.distributed import (
    gather_outputs_and_unpad,
    get_sp_ring_group,
    get_sp_ulysses_group,
)
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
        self.draft_model = draft_model
        self.length = length
        self.attention_backend = attention_backend

        if self.attention_backend == "usp":
            self.extract_func = EXTRACT_FUNC_DICT["basic"]
            self.sp_ring_degree = torch.distributed.get_world_size(get_sp_ring_group())
            self.sp_ulysses_degree = torch.distributed.get_world_size(
                get_sp_ulysses_group()
            )
            self.sp_world_size = self.sp_ring_degree * self.sp_ulysses_degree
            self.sp_rank = torch.distributed.get_rank() % self.sp_world_size

    @torch.compile()
    def prepare_usp_input(self, full_input):
        shared_input = self.extract_func(
            full_input,
            rank=self.sp_rank,
            world_size=self.sp_world_size,
        ).clone()
        return shared_input

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
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
        # Step 1: handle vocab size
        target_p_padded, position_mask = _compute_target_p_padded(
            target=target,
            t2d=self.draft_model.t2d,
            loss_mask=loss_mask,
            length=self.length,
        )
        del target

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
        if self.attention_backend in ("sdpa", "usp"):
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
        # for sequence paralle, position mask and input ids will split by sequence dim, need to keep origin for ttt shift
        global_input_ids = input_ids
        if self.attention_backend in ["sdpa", "fa"]:
            cache_hidden = [[], []]
            past_key_values = None
        elif self.attention_backend == "flex_attention":
            cache_hidden = None
            past_key_values = DynamicCache()
        elif self.attention_backend == "usp":
            cache_hidden = [[], []]
            past_key_values = None
            hidden_states = self.prepare_usp_input(hidden_states)
        else:
            raise ValueError(f"Unknown attention backend: {self.attention_backend}")

        for idx in range(self.length):
            target_p = target_p_padded[:, idx : idx + seq_length, :]
            if self.attention_backend == "usp":
                input_ids = self.prepare_usp_input(global_input_ids)
            else:
                input_ids = global_input_ids

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

            # update hidden states for next step
            hidden_states = hidden_states_out

            # Step 5.4: get logits
            logits = self.draft_model.compute_logits(hidden_states)
            logits = gather_outputs_and_unpad(logits, gather_dim=1)
            # Step 5.5: record metrics first as we in-place modify logits
            with torch.no_grad():
                acces.append(
                    _compute_metric_acc(
                        logits=logits,
                        target_p=target_p,
                        position_mask=position_mask,
                        loss_mask=loss_mask,
                    )
                )

            # Step 5.6: calculate loss, in-place modifies logits!
            loss = LogSoftmaxLoss.apply(logits, target_p, position_mask)
            plosses.append(loss)

            if not is_last:
                # Step 5.7: we need to update the loss mask
                global_input_ids = padding(global_input_ids, left=False)
                position_mask = padding(position_mask, left=False)
                loss_mask = padding(loss_mask, left=False)
                # Flex attention mask shirnking is handled inside attention module
        return plosses, vlosses, acces


class QwenVLOnlineEagle3Model(Eagle3Model):
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
        processor,
        length: int = 7,
        attention_backend: str = "sdpa",
        target_model_type: Optional[str] = None,
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
        self.processor = processor
        self.length = length
        self.attention_backend = attention_backend
        if target_model_type is not None:
            model_type = target_model_type
        else:
            target_config = getattr(target_model, "config", None)
            model_type = getattr(target_config, "model_type", None)
        self.target_model_type = model_type
        self.rope_deltas: Optional[torch.Tensor] = None

    @torch.no_grad()
    def _prepare_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
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
            pixel_values: image pixel values, used for VLM models
            pixel_values_videos: video pixel values, used for VLM models supporting videos
            image_grid_thw: image grid thw, used for VLM models
            video_grid_thw: video grid thw, used for VLM models supporting videos
            second_per_grid_ts: temporal interval per grid, required by qwen2_5_vl video inputs

        Returns:
            hidden_states: (batch, seq_len, 3*hidden_size)
            target: (batch, seq_len, vocab_size)
            loss_mask: (batch, seq_len)
            input_ids: (batch, seq_len)
        """

        if device is None:
            device = input_ids.device

        # run the target model to get the hidden states
        target_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "pixel_values_videos": pixel_values_videos,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "output_hidden_states": True,
            "use_cache": False,
        }
        if self.target_model_type not in {"qwen3_vl", "qwen3_vl_moe"}:
            target_kwargs["second_per_grid_ts"] = second_per_grid_ts
        # remove optional None entries to avoid unexpected kwargs errors
        filtered_target_kwargs = {}
        for key, value in target_kwargs.items():
            if key in {"input_ids", "attention_mask", "output_hidden_states", "use_cache"} or value is not None:
                filtered_target_kwargs[key] = value

        outputs = self.target_model(**filtered_target_kwargs)
        rope_deltas = getattr(outputs, "rope_deltas", None)
        if rope_deltas is not None:
            self.rope_deltas = rope_deltas

        # extract the aux hidden states
        # output_hidden_states = True will return the embedding output as well
        # so we have an offset of 1
        num_hidden_states = len(outputs.hidden_states)
        offset = 1
        num_layers = num_hidden_states - 1

        # Eagle3 uses 3 aux layers from layer 1, num_layers//2, num_layers-4
        eagle3_config_dict = self.draft_model.config.to_dict()
        eagle_config = eagle3_config_dict.get("eagle_config", None)
        if (
            eagle_config is not None
            and "eagle_aux_hidden_state_layer_ids" in eagle_config
        ):
            aux_layer_ids = eagle_config["eagle_aux_hidden_state_layer_ids"]
            assert len(aux_layer_ids) == 3, "EAGLE3 requires 3 aux layers"
        else:
            # Qwen3VL uses deepstack at decoder layers 0, 1, 2
            # Start at layer 3 to avoid capture timing mismatch with sglang
            first_layer = 3 if self.target_model_type in ("qwen3_vl", "qwen3_vl_moe") else 1
            aux_layer_ids = [first_layer, num_layers // 2 - 1, num_layers - 4]

        low_aux_layer = aux_layer_ids[0] + offset
        mid_aux_layer = aux_layer_ids[1] + offset
        last_aux_layer = aux_layer_ids[2] + offset

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

    @torch.no_grad()
    def _get_input_embeds(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        # get input embeding with image
        # inputs_embeds = self.target_model.model.get_input_embeddings()(input_ids)
        inputs_embeds = self.draft_model.embed_input_ids(input_ids)
        image_features = self.target_model.model.get_image_features(
            pixel_values, image_grid_thw
        )
        if self.target_model_type in {"qwen3_vl", "qwen3_vl_moe"}:
            image_embeds, *_ = image_features
        else:
            image_embeds = image_features
        image_embeds = torch.cat(image_embeds, dim=0)
        n_image_tokens = (
            input_ids == self.target_model.model.config.image_token_id
        ).sum()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        mask = input_ids == self.target_model.model.config.image_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(inputs_embeds.device)

        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Online eagle model trainer, modified from: https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/cnets.py#L711

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            loss_mask: (batch, seq_len)
            past_key_values: We dont use this past_key_values in eagle3, but keep it for compatibility. We control kvcache by cache_hidden.
            position_ids: (batch, seq_len)
            pixel_values: batch image pixel values, used for VLM models
            image_grid_thw: (batch, 3), image grid thw, used for VLM models
            pixel_values_videos: batch video pixel values, optional for models supporting videos
            video_grid_thw: (batch, 3), video grid thw, optional
            second_per_grid_ts: per-grid temporal interval for qwen2_5_vl video support
        """
        # Step 0: prepare data with the target model
        hidden_states, target, loss_mask, input_ids = self._prepare_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
        )

        # Step 1: handle vocab size
        target_p_padded, position_mask = _compute_target_p_padded(
            target=target,
            t2d=self.draft_model.t2d,
            loss_mask=loss_mask,
            length=self.length,
        )
        del target

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

        base_attention_mask = (
            attention_mask
            if not isinstance(attention_mask, dict)
            else attention_mask["full_attention"]
        )
        # Cache the raw mask so that SDPA and RoPE refresh both see the same window-aligned view.
        if base_attention_mask is not None and base_attention_mask.ndim == 4:
            base_attention_mask = torch.diagonal(
                base_attention_mask[:, 0], dim1=1, dim2=2
            )
            base_attention_mask = (
                base_attention_mask / torch.finfo(base_attention_mask.dtype).min
            )
            base_attention_mask = (1.0 - base_attention_mask).int()

        if position_ids is None:
            get_rope_kwargs = {
                "input_ids": input_ids,
                "image_grid_thw": image_grid_thw,
                "attention_mask": base_attention_mask,
            }
            if self.target_model_type in {"qwen3_vl", "qwen3_vl_moe"}:
                get_rope_kwargs["video_grid_thw"] = video_grid_thw
            else:
                get_rope_kwargs["video_grid_thw"] = video_grid_thw
                get_rope_kwargs["second_per_grid_ts"] = second_per_grid_ts
            position_ids, rope_deltas = self.target_model.model.get_rope_index(
                **get_rope_kwargs
            )
            if rope_deltas is not None:
                self.rope_deltas = rope_deltas
            full_attention_mask = (
                base_attention_mask.clone()
                if base_attention_mask is not None
                else None
            )
        else:
            position_ids = position_ids
            full_attention_mask = (
                base_attention_mask.clone()
                if base_attention_mask is not None
                else None
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
                attention_mask=full_attention_mask,
                hidden_states=hidden_states,
                batch_size=batch_size,
                seq_length=seq_length,
                past_key_values_length=past_key_values_length,
            )

        # Step 5: run TTT
        plosses = []
        vlosses = []
        acces = []
        if self.attention_backend in ["sdpa", "fa"]:
            cache_hidden = [[], []]
            past_key_values = None
        elif self.attention_backend == "flex_attention":
            cache_hidden = None
            past_key_values = DynamicCache()
        else:
            raise ValueError(f"Unknown attention backend: {self.attention_backend}")

        for idx in range(self.length):
            target_p = target_p_padded[:, idx : idx + seq_length, :].contiguous()
            is_last = idx == self.length - 1

            # Step 5.1: embed the input ids
            # inputs_embeds = self._get_input_embeds(input_ids, pixel_values, image_grid_thw)
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

            # update hidden states for next step
            hidden_states = hidden_states_out

            # Step 5.4: get logits
            logits = self.draft_model.compute_logits(hidden_states)

            # Step 5.5: record metrics first as we in-place modify logits
            with torch.no_grad():
                acces.append(
                    _compute_metric_acc(
                        logits=logits,
                        target_p=target_p,
                        position_mask=position_mask,
                        loss_mask=loss_mask,
                    )
                )

            # Step 5.6: calculate loss, in-place modifies logits!
            loss = LogSoftmaxLoss.apply(logits, target_p, position_mask)
            plosses.append(loss)

            if not is_last:
                # Step 5.7: we need to update the loss mask
                input_ids = padding(input_ids, left=False)
                position_mask = padding(position_mask, left=False)
                loss_mask = padding(loss_mask, left=False)
                # Shrink the cached mask so SDPA keeps the same view after padding.
                # Roll the cached SDPA mask so it matches the new left-shifted window.
                if full_attention_mask is not None:
                    full_attention_mask = padding(full_attention_mask, left=False)

                if self.attention_backend == "sdpa":
                    attention_mask = self.draft_model.prepare_decoder_attention_mask(
                        attention_mask=full_attention_mask,
                        hidden_states=hidden_states,
                        batch_size=batch_size,
                        seq_length=seq_length,
                        past_key_values_length=past_key_values_length,
                    )
                elif (
                    attention_mask is not None
                    and self.target_model_type in {"qwen3_vl", "qwen3_vl_moe"}
                ):
                    # qwen3 path carries the un-expanded 2D causal mask directly.
                    attention_mask = padding(attention_mask, left=False)

                next_attention_tensor = (
                    full_attention_mask
                    if full_attention_mask is not None
                    else (
                        attention_mask
                        if self.target_model_type in {"qwen3_vl", "qwen3_vl_moe"}
                        else None
                    )
                )
                if (
                    next_attention_tensor is not None
                    and self.target_model_type not in {"qwen3_vl", "qwen3_vl_moe"}
                    and next_attention_tensor.ndim == 4
                ):
                    # qwen2.5 still produces inverted 4D masks; collapse and flip them before RoPE.
                    next_attention_tensor = torch.diagonal(
                        next_attention_tensor[:, 0], dim1=1, dim2=2
                    )
                    if next_attention_tensor.dtype.is_floating_point:
                        next_attention_tensor = (
                            next_attention_tensor
                            / torch.finfo(next_attention_tensor.dtype).min
                        )
                        next_attention_tensor = (1.0 - next_attention_tensor).int()

                # qwen3_vl expects video grid kwargs rather than second_per_grid_ts, qwen2.5_vl still needs both.
                rope_kwargs = {
                    "input_ids": input_ids,
                    "image_grid_thw": image_grid_thw,
                    "attention_mask": next_attention_tensor,
                }
                if self.target_model_type in {"qwen3_vl", "qwen3_vl_moe"}:
                    rope_kwargs["video_grid_thw"] = video_grid_thw
                else:
                    rope_kwargs["video_grid_thw"] = video_grid_thw
                    rope_kwargs["second_per_grid_ts"] = second_per_grid_ts
                position_ids, rope_deltas = self.target_model.model.get_rope_index(
                    **rope_kwargs
                )
                if rope_deltas is not None:
                    self.rope_deltas = rope_deltas
                # Flex attention mask shirnking is handled inside attention module
        return plosses, vlosses, acces


def _compute_target_p_padded(target, t2d, loss_mask, length):
    with torch.no_grad():
        target_p, position_mask = _compute_target_p(
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

        return target_p_padded, position_mask


@torch.compile(dynamic=None)
def _compute_target_p(target, t2d, loss_mask):
    target_head = target
    target_max_token = target_head.argmax(-1)
    target_mask = t2d[target_max_token]
    target_mask = target_mask[..., None].int()
    position_mask = target_mask * loss_mask
    target_head = target_head[..., t2d]
    target_head = target_head.float()
    target_p = nn.Softmax(dim=2)(target_head)
    target_p = target_p.detach()
    return target_p, position_mask


@torch.compile(dynamic=None)
def _compute_metric_acc(logits, target_p, position_mask, loss_mask):
    return (
        (logits.argmax(-1) == target_p.argmax(-1)) * position_mask.squeeze(-1)
    ).sum() / loss_mask.sum().clamp_min(1e-6)
