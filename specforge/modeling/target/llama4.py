# coding=utf-8
# Copyright 2025 The LLAMA4 and HuggingFace Inc. team. All rights reserved.
#
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

from typing import Callable, Dict, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations.hub_kernels import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask, create_chunked_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.llama4.configuration_llama4 import (
    Llama4Config,
    Llama4TextConfig,
)
from transformers.models.llama4.modeling_llama4 import (
    Llama4TextL2Norm,
    Llama4TextRMSNorm,
    Llama4TextRotaryEmbedding,
    Llama4VisionModel,
    apply_rotary_emb,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack
from transformers.utils import auto_docstring, can_return_tuple, logging

# [MODIFIED] Import from transformers library
from specforge.distributed import get_tp_group
from specforge.layers.linear import ColumnParallelLinear, RowParallelLinear

from .base import DistributedTargetModel

logger = logging.get_logger(__name__)


class Llama4TextExperts(nn.Module):
    def __init__(self, config: Llama4TextConfig):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size

        # self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))
        # self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.expert_dim, self.hidden_size)))

        self.tp_group = get_tp_group()
        self.tp_size = dist.get_world_size(self.tp_group)
        self.expert_dim_per_shard = self.expert_dim // self.tp_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                self.num_experts, self.hidden_size, 2 * self.expert_dim_per_shard
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty((self.num_experts, self.expert_dim_per_shard, self.hidden_size))
        )

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        This should really not be run on a single machine, as we are reaching compute bound:
        - the inputs are expected to be "sorted" per expert already.
        - the weights are viewed with another dim, to match num_expert, 1, shape * num_tokens, shape

        Args:
            hidden_states (torch.Tensor): (batch_size * token_num, hidden_size)
            selected_experts (torch.Tensor): (batch_size * token_num, top_k)
            routing_weights (torch.Tensor): (batch_size * token_num, top_k)
        Returns:
            torch.Tensor
        """
        hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
        gate_up = torch.bmm(hidden_states, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)  # not supported for DTensors
        next_states = torch.bmm((up * self.act_fn(gate)), self.down_proj)
        dist.all_reduce(next_states, op=dist.ReduceOp.SUM, group=self.tp_group)
        next_states = next_states.view(-1, self.hidden_size)
        return next_states


# Phi3MLP
class Llama4TextMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()

        if intermediate_size is None:
            intermediate_size = config.intermediate_size
        self.intermediate_size = intermediate_size
        self.config = config

        self.tp_group = get_tp_group()
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size, self.intermediate_size, bias=False
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size, self.intermediate_size, bias=False
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size, config.hidden_size, bias=False
        )

        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.activation_fn(self.gate_proj(x)) * self.up_proj(x)
        out = self.down_proj(down_proj)
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.tp_group)
        return out


class Llama4TextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Llama4TextConfig, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attn_scale = config.attn_scale
        self.floor_scale = config.floor_scale
        self.attn_temperature_tuning = config.attn_temperature_tuning
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.use_rope = config.no_rope_layers[layer_idx]

        # distributed linear layers
        self.tp_group = get_tp_group()
        self.q_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        if self.config.use_qk_norm and self.use_rope:
            self.qk_norm = Llama4TextL2Norm(config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(*input_shape, -1, self.head_dim)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if self.use_rope:  # the 16E model skips rope for long context on certain layers
            query_states, key_states = apply_rotary_emb(
                query_states, key_states, position_embeddings.to(query_states.device)
            )

        if hasattr(self, "qk_norm"):  # the 128E model does not use qk_norm
            query_states = self.qk_norm(query_states)
            key_states = self.qk_norm(key_states)

        # Use temperature tuning from https://huggingface.co/papers/2501.19399) to NoROPE layers
        if self.attn_temperature_tuning and not self.use_rope:
            attn_scales = (
                torch.log(
                    torch.floor((cache_position.float() + 1.0) / self.floor_scale) + 1.0
                )
                * self.attn_scale
                + 1.0
            )
            attn_scales = attn_scales.view((1, input_shape[-1], 1, 1)).expand(
                (*input_shape, 1, 1)
            )  # batch size > 1
            query_states = (query_states * attn_scales).to(query_states.dtype)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        dist.all_reduce(attn_output, op=dist.ReduceOp.SUM, group=self.tp_group)
        return attn_output, attn_weights


@use_kernel_forward_from_hub("Llama4TextMoe")
class Llama4TextMoe(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.experts = Llama4TextExperts(config)
        self.router = nn.Linear(
            config.hidden_size, config.num_local_experts, bias=False
        )
        self.shared_expert = Llama4TextMLP(config)

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = self.router(hidden_states)

        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=1)

        router_scores = (
            torch.full_like(router_logits, float("-inf"))
            .scatter_(1, router_indices, router_top_value)
            .transpose(0, 1)
        )
        router_scores = torch.sigmoid(router_scores.float()).to(hidden_states.dtype)

        routed_in = hidden_states.repeat(self.num_experts, 1)
        routed_in = routed_in * router_scores.reshape(-1, 1)
        routed_out = self.experts(routed_in)

        out = self.shared_expert(hidden_states)
        out.add_(routed_out.reshape(self.num_experts, -1, self.hidden_dim).sum(dim=0))

        return out, router_scores


class Llama4TextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = Llama4TextAttention(config, layer_idx)
        self.is_moe_layer = layer_idx in config.moe_layers
        if self.is_moe_layer:  # the 128E model interleaves dense / sparse
            self.feed_forward = Llama4TextMoe(config)
        else:
            self.feed_forward = Llama4TextMLP(
                config, intermediate_size=config.intermediate_size_mlp
            )

        self.input_layernorm = Llama4TextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Llama4TextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[
        torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attention_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + attention_states

        # Fully Connected
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        if self.is_moe_layer:
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None
        hidden_states = residual + hidden_states.view(residual.shape)
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


@auto_docstring
class Llama4PreTrainedModel(PreTrainedModel):
    config_class = Llama4Config
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = False
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, Llama4TextRMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, Llama4TextExperts):
            module.gate_up_proj.data.normal_(mean=0.0, std=std)
            module.down_proj.data.normal_(mean=0.0, std=std)
        elif isinstance(module, Llama4VisionModel):
            module.class_embedding.data.normal_(std=module.scale)
            module.positional_embedding_vlm.data.normal_(std=module.scale)


@auto_docstring
class Llama4TextModel(Llama4PreTrainedModel):
    _no_split_modules = ["Llama4TextDecoderLayer"]
    base_model_prefix = "model"
    config_class = Llama4TextConfig

    def __init__(self, config: Llama4TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                Llama4TextDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Llama4TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Llama4TextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(
                input_ids.to(self.embed_tokens.weight.device)
            )

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "chunked_attention": create_chunked_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        freq_cis = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=freq_cis,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Llama4ForCausalLM(Llama4PreTrainedModel, GenerationMixin, DistributedTargetModel):
    _no_split_modules = ["Llama4TextDecoderLayer"]
    base_model_prefix = "language_model"
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    config_class = Llama4TextConfig

    def __init__(self, config: Llama4TextConfig):
        super().__init__(config)
        self.model = Llama4TextModel(config)
        self.vocab_size = config.vocab_size

        # distributed the lm head
        self.lm_head = ColumnParallelLinear(
            config.hidden_size, config.vocab_size, bias=False
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Llama4ForCausalLM

        >>> model = Llama4ForCausalLM.from_pretrained("meta-llama4/Llama4-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama4/Llama4-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        logits = self._gather_tensor(logits, get_tp_group())

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def load_weights(self, state_dict: Dict[str, torch.Tensor]):
        tp_group = get_tp_group()

        updated_state_dict = {}
        for key, value in state_dict.items():
            # Ensure that the state dict is a flat dict of keys and tensors. Breaking this assumption
            # will break recipe code
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"Expected all values in the state dict to be torch.Tensor. "
                    f"Found {type(value)} instead."
                )

            # update key

            if key.startswith("vision_model.") or key.startswith(
                "multi_modal_projector."
            ):
                continue
            elif key.startswith("language_model."):
                key = key[len("language_model.") :]

            module_key = ".".join(key.split(".")[:-1])
            module = self.get_submodule(module_key)

            # get the module type based on key
            if "experts.gate_up_proj" in key:
                gate, up = value.chunk(2, dim=-1)

                # shard the gate and up based on tp
                gate = self._shard_tensor(gate, tp_group, -1)
                up = self._shard_tensor(up, tp_group, -1)
                value = torch.cat((gate, up), dim=-1)
            elif "experts.down_proj" in key:
                value = self._shard_tensor(value, tp_group, 1)
            elif isinstance(module, RowParallelLinear) and key.endswith(".weight"):
                value = self._shard_tensor(value, tp_group, -1)
            elif isinstance(module, ColumnParallelLinear) and key.endswith(".weight"):
                value = self._shard_tensor(value, tp_group, 0)
            elif isinstance(module, ColumnParallelLinear) and key.endswith(".bias"):
                value = self._shard_tensor(value, tp_group, 0)

            updated_state_dict[key] = value

        # load state dict
        self.load_state_dict(updated_state_dict, strict=False)
