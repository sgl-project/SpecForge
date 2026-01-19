"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from sglang.srt.utils import add_prefix

# Adapted from
# https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/cnets.py
"""Inference-only LLaMA-EAGLE model compatible with HuggingFace weights."""

from typing import Iterable, Optional, Tuple

import torch
from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.communicator import LayerCommunicator
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeForCausalLM
from torch import nn
from transformers import Qwen3MoeConfig


class Qwen3MoeDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(
        self,
        config: Qwen3MoeConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, layer_id, quant_config, prefix)

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        # override qkv
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.self_attn.qkv_proj = QKVParallelLinear(
            2 * config.hidden_size,
            head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )
        # del self.self_attn.q_norm
        # setattr(self.self_attn, "q_norm", lambda x: x)
        # del self.self_attn.k_norm
        # setattr(self.self_attn, "k_norm", lambda x: x)

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        embeds, residual = self.layer_communicator.prepare_attn(
            embeds, residual, forward_batch
        )
        # Copy and Paste from Qwen3MoeDecoderLayer, This line is the only difference
        hidden_states = torch.cat([embeds, hidden_states], dim=-1)

        # print("[yikai eagle debug]hidden_states.shape ", hidden_states.shape)
        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        # For DP with padding, reduce scatter can be used instead of all-reduce.
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        hidden_states = self.mlp(hidden_states, forward_batch, use_reduce_scatter)

        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )

        return hidden_states, residual


class Qwen3MoeModel(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        if hasattr(config, "target_hidden_size"):
            self.hidden_size_in = config.target_hidden_size
        else:
            self.hidden_size_in = config.hidden_size

        self.fc = torch.nn.Linear(
            self.hidden_size_in * 3,
            config.hidden_size,
            bias=getattr(config, "bias", False),
        )
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.midlayer = Qwen3MoeDecoderLayer(config, 0, quant_config, prefix)
        self.hidden_layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.midlayer.layer_scatter_modes,
            input_layernorm=self.hidden_norm,
            post_attention_layernorm=self.midlayer.post_attention_layernorm,
            allow_reduce_scatter=True,
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            embeds = self.embed_tokens(input_ids)
        else:
            embeds = input_embeds

        hidden_states = forward_batch.spec_info.hidden_states
        if hidden_states.shape[-1] != embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)

        hidden_states, _ = self.hidden_layer_communicator.prepare_attn(
            hidden_states, None, forward_batch
        )

        residual = None
        hidden_states, residual = self.midlayer(
            positions,
            embeds,
            hidden_states,
            forward_batch,
            residual,
        )
        # TODO: this is a temporary fix for the bug in Qwen3MoeForCausalLM
        # return hidden_states, [hidden_states]

        hidden_states_to_logits, hidden_states_to_aux = self.norm(
            hidden_states, residual
        )

        # For draft decode, we capture the hidden state before norm
        return hidden_states_to_logits, [hidden_states_to_aux]


class Qwen3MoEForCausalLMEagle3(Qwen3MoeForCausalLM):
    def __init__(
        self,
        config: Qwen3MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        if self.config.num_hidden_layers != 1:
            raise ValueError("EAGLE3 currently only supports 1 layer")

        self.model = Qwen3MoeModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.draft_vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = True
        self.hot_token_id = None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        print("Loading weights for Qwen3MoEForCausalLMEagle3")
        params_dict = dict(self.named_parameters())
        print("params_dict.keys ", list(params_dict.keys()))
        # Define the parameter mapping for stacked parameters
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )
        # print("expert_params_mapping ", expert_params_mapping)

        # Cache params_dict to avoid repeated expensive traversal of model parameters
        if not hasattr(self, "_cached_params_dict"):
            self._cached_params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            print("load weight from ", name, loaded_weight.shape)
            if "d2t" in name:
                # d2t stores diffs between draft id and target id
                self.hot_token_id = loaded_weight + torch.arange(loaded_weight.shape[0])
                continue

            if "t2d" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                # print(f"found {weight_name} in {name}, try to replace it with {param_name}")
                name = name.replace(weight_name, param_name)
                param_name = f"model.{name}" if name not in params_dict else name
                if param_name in params_dict:
                    param = params_dict[param_name]
                    print("load weight to ", param_name, param.shape)
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, shard_id)
                else:
                    raise ValueError(
                        f"Parameter {param_name}(from name {name}) not found in params_dict"
                    )
                break
            else:
                # Track if this is an expert weight to enable early skipping
                is_expert_weight = False

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    # print("for expert weight name", name, "param_name ", param_name, "weight_name ", weight_name, "expert_id ", expert_id, "shard_id ", shard_id)

                    # Mark as expert weight regardless of whether we can process it
                    is_expert_weight = True

                    name = name.replace(weight_name, param_name)
                    name = f"model.{name}" if name not in params_dict else name
                    if name not in params_dict:
                        # Expert weight not on this rank, will be skipped below
                        raise ValueError(
                            f"skipping expert weight {name} since it is not on this rank"
                        )
                        continue

                    param = params_dict[name]
                    print("load weight to ", name, param.shape)
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    if is_expert_weight:
                        # This is an expert weight but not mapped to this rank, skip all remaining processing
                        raise ValueError(
                            f"skipping expert weight {name} since it is not on this rank"
                        )
                        continue

                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        print("skipping bias ", name, "since it has bias")
                        continue
                    param_name = f"model.{name}" if name not in params_dict else name
                    if param_name not in params_dict:
                        raise ValueError(
                            f"Parameter {param_name}(from name {name}) not found in params_dict"
                        )

                    if param_name in params_dict.keys():
                        param = params_dict[param_name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        print("load weight to ", param_name, param.shape)
                        weight_loader(param, loaded_weight)
                    else:
                        raise ValueError(
                            f"Parameter {param_name}(from name {name}) not found in params_dict"
                        )

    def get_hot_token_id(self):
        return self.hot_token_id

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def get_embed(self):
        return self.model.embed_tokens.weight

    def set_embed(self, embed):
        # NOTE: If draft hidden size != target hidden size, the embed weight cannot be shared for EAGLE3
        if (
            hasattr(self.config, "target_hidden_size")
            and self.config.target_hidden_size != self.config.hidden_size
        ):
            return
        del self.model.embed_tokens.weight
        self.model.embed_tokens.weight = embed
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


EntryClass = [Qwen3MoEForCausalLMEagle3]
