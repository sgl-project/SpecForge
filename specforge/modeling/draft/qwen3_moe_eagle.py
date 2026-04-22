# coding=utf-8
# Copyright 2025 SpecForge Team. All rights reserved.
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

"""
Qwen3 MoE Eagle3 Draft Model.

This module implements an Eagle3 draft model with MoE (Mixture of Experts) layers,
designed for Qwen3.5-122B-A10B and similar MoE target models. The draft model uses
the same MoE structure as the target model's MTP heads, enabling weight initialization
from native MTP and better alignment with the target model's behavior.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from specforge.utils import print_with_rank

from .base import Eagle3DraftModel
from .llama3_eagle import (
    LlamaAttention,
    LlamaFlashAttention,
    LlamaFlexAttention,
    LlamaRMSNorm,
    LlamaUSPFlashAttention,
    prepare_decoder_attention_mask,
)

class Qwen3MoeDraftMLP(nn.Module):
    """Single expert MLP for the MoE draft model (no TP, used in training)."""

    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            intermediate_size
            if intermediate_size is not None
            else config.intermediate_size
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3MoeDraftSparseMoeBlock(nn.Module):
    """
    Sparse MoE block for the Eagle3 draft model.

    This implements the same MoE routing logic as the target model's MoE layers,
    but without tensor parallelism (since the draft model is small and runs on
    a single device during training).
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # gating / router
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                Qwen3MoeDraftMLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hitted:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                expert_layer(current_state) * routing_weights[top_x, idx, None]
            )

            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states


class Qwen3MoeDecoderLayerEagle3(nn.Module):
    """
    Eagle3 decoder layer with MoE MLP.

    This layer follows the same structure as LlamaDecoderLayer in the existing
    Eagle3 implementation, but replaces the dense MLP with a MoE block.
    The attention mechanism is reused from the LLaMA Eagle3 implementation.
    """

    def __init__(self, config, attention_backend: str = "sdpa"):
        super().__init__()
        self.hidden_size = config.hidden_size

        if attention_backend == "sdpa":
            self.self_attn = LlamaAttention(config=config)
        elif attention_backend == "flex_attention":
            print_with_rank("Using flex attention on draft model training!")
            self.self_attn = LlamaFlexAttention(config=config)
        elif attention_backend == "fa":
            self.self_attn = LlamaFlashAttention(config=config)
        elif attention_backend == "usp":
            self.self_attn = LlamaUSPFlashAttention(config=config)
        else:
            raise ValueError(f"Unknown attention backend {attention_backend}")

        self.attention_backend = attention_backend

        # Use MoE block instead of dense MLP
        self.mlp = Qwen3MoeDraftSparseMoeBlock(config)

        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: List[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)

        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)
        # Self Attention
        hidden_states = self.self_attn(
            cache_hidden=cache_hidden,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # MoE MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3MoeForCausalLMEagle3(Eagle3DraftModel):
    """
    Eagle3 draft model with MoE layers for Qwen3.5 MoE target models.

    This model uses the same architecture as LlamaForCausalLMEagle3 but replaces
    the dense MLP in the decoder layer with a Sparse MoE block. This allows:
    1. Weight initialization from the target model's native MTP heads
    2. Better alignment with the target model's MoE routing behavior
    3. Same active parameter count as native MTP during inference
    """

    config_class = Qwen3MoeConfig

    def __init__(self, config, quant_config=None, attention_backend="sdpa") -> None:
        super().__init__(config)
        self.config = config
        self.quant_config = quant_config

        self.vocab_size = config.vocab_size
        self.draft_vocab_size = config.draft_vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.midlayer = Qwen3MoeDecoderLayerEagle3(
            config, attention_backend=attention_backend
        )

        if hasattr(config, "target_hidden_size"):
            self.fc = torch.nn.Linear(
                config.target_hidden_size * 3, config.hidden_size, bias=False
            )
        else:
            self.fc = torch.nn.Linear(
                config.hidden_size * 3, config.hidden_size, bias=False
            )

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(
            config.hidden_size, config.draft_vocab_size, bias=False
        )

        # create vocab buffers
        t2d = torch.ones(self.vocab_size, dtype=torch.bool)
        d2t = torch.zeros(self.draft_vocab_size, dtype=torch.int64)
        self.register_buffer("t2d", t2d)
        self.register_buffer("d2t", d2t)

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        ttt_length: int = 1,
    ):
        if ttt_length == 1:
            print_with_rank("using ttt_length 1, no need to cache hidden states")
            cache_hidden = None
        else:
            print_with_rank(f"using ttt_length {ttt_length}, caching hidden states")
            cache_hidden = [[], []]

        batch_size, seq_length, _ = hidden_states.size()

        # make position ids
        device = hidden_states.device
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        # make attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, 0
        )

        # fc
        hidden_states = self.fc(hidden_states)
        hidden_states = self.midlayer(
            input_emb=inputs_embeds,
            hidden_states=hidden_states,
            cache_hidden=cache_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            output_attentions=False,
            use_cache=False,
        )

        # norm
        hidden_states = self.norm(hidden_states)

        return hidden_states

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # eagle 3 requires hidden states from 3 layers
        assert hidden_states.size(-1) == self.config.hidden_size * 3
        return self.fc(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        norm_hidden_states = self.norm(hidden_states)
        return self.lm_head(norm_hidden_states)

    def backbone(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        return self.midlayer(
            input_emb=input_embeds,
            hidden_states=hidden_states,
            cache_hidden=cache_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=False,
            use_cache=False,
        )

    @torch.no_grad()
    def load_mtp_weights(
        self,
        model_path: str,
        mtp_layer_idx: int = 0,
    ) -> None:
        """
        Load weights from the target model's native MTP (Multi-Token Prediction) head
        into this draft model's MoE layer.

        The MTP head in Qwen3.5 MoE models typically has the structure:
            model.mtp_block_{idx}.mtp_mlp.gate_proj / up_proj / down_proj  (for dense MLP layers)
            model.mtp_block_{idx}.mtp_moe.gate  (router)
            model.mtp_block_{idx}.mtp_moe.experts.{i}.gate_proj / up_proj / down_proj

        This method loads the MoE weights (router + all experts) from the target model's
        MTP block into the draft model's midlayer.mlp.

        Args:
            model_path: Path to the target model directory containing safetensors files.
            mtp_layer_idx: Index of the MTP block to load from (default: 0).
        """
        import glob
        import json
        import os

        from safetensors import safe_open

        if not os.path.exists(model_path):
            print_with_rank(
                f"Warning: model_path {model_path} does not exist, skipping MTP weight loading"
            )
            return

        # Find the weight index file
        glob_path = os.path.join(model_path, "*.index.json")
        index_json_paths = glob.glob(glob_path)

        weight_files = {}
        if len(index_json_paths) > 0:
            index_json_path = index_json_paths[0]
            with open(index_json_path, "r") as f:
                index_json = json.load(f)
            weight_map = index_json["weight_map"]
        else:
            # Single file model
            safetensors_path = os.path.join(model_path, "model.safetensors")
            if not os.path.exists(safetensors_path):
                print_with_rank(
                    f"Warning: No model files found in {model_path}, skipping MTP weight loading"
                )
                return
            weight_map = None

        # Define the MTP weight key patterns
        # Common patterns for Qwen3.5 MoE MTP blocks
        mtp_prefix = f"model.mtp_block_{mtp_layer_idx}"
        moe_prefix = f"{mtp_prefix}.mtp_moe"

        # Map from MTP weight keys to draft model keys
        weight_mapping = {}

        # Router weights
        weight_mapping[f"{moe_prefix}.gate.weight"] = "midlayer.mlp.gate.weight"

        # Expert weights
        for expert_idx in range(self.config.num_experts):
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                src_key = f"{moe_prefix}.experts.{expert_idx}.{proj}.weight"
                dst_key = f"midlayer.mlp.experts.{expert_idx}.{proj}.weight"
                weight_mapping[src_key] = dst_key

        # Load and copy weights
        loaded_count = 0
        for src_key, dst_key in weight_mapping.items():
            try:
                if weight_map is not None:
                    if src_key not in weight_map:
                        print_with_rank(
                            f"Warning: {src_key} not found in weight map, skipping"
                        )
                        continue
                    ckpt_file = weight_map[src_key]
                    file_path = os.path.join(model_path, ckpt_file)
                else:
                    file_path = safetensors_path

                with safe_open(file_path, framework="pt") as f:
                    if src_key in f.keys():
                        src_tensor = f.get_tensor(src_key)
                    else:
                        print_with_rank(
                            f"Warning: {src_key} not found in {file_path}, skipping"
                        )
                        continue

                # Navigate to the target parameter
                parts = dst_key.split(".")
                target = self
                for part in parts[:-1]:
                    if part.isdigit():
                        target = target[int(part)]
                    else:
                        target = getattr(target, part)
                param_name = parts[-1]
                param = getattr(target, param_name)

                if param.shape != src_tensor.shape:
                    print_with_rank(
                        f"Warning: Shape mismatch for {src_key}: "
                        f"src={src_tensor.shape}, dst={param.shape}, skipping"
                    )
                    continue

                param.copy_(src_tensor)
                loaded_count += 1

            except Exception as e:
                print_with_rank(
                    f"Warning: Failed to load {src_key} -> {dst_key}: {e}"
                )
                continue

        print_with_rank(
            f"Loaded {loaded_count}/{len(weight_mapping)} MTP weights from {model_path} "
            f"(mtp_block_{mtp_layer_idx})"
        )
