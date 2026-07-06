from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.models.llama.configuration_llama import LlamaConfig

from .llama3_eagle import LlamaDecoderLayer, LlamaForCausalLMEagle3, LlamaMLP, LlamaRMSNorm


class EDFuse(LlamaMLP):

    def forward(self, x: torch.Tensor, embed_tokens: torch.Tensor) -> torch.Tensor:
        if self.config.pretraining_tp > 1:
            slice_ = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice_, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice_, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice_, dim=1)

            gate_proj = torch.cat(
                [torch.nn.functional.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)],
                dim=-1,
            )
            up_proj = torch.cat(
                [torch.nn.functional.linear(embed_tokens, up_proj_slices[i]) for i in range(self.config.pretraining_tp)],
                dim=-1,
            )
            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice_, dim=2)
            down_proj = sum(
                torch.nn.functional.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            )
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(embed_tokens))

        return down_proj


class EdsdDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, attention_backend: str = "sdpa"):
        super().__init__(config, attention_backend=attention_backend)
        self.input_layernorm = nn.Identity()
        for name in ("q_proj", "k_proj", "v_proj"):
            old = getattr(self.self_attn, name)
            setattr(self.self_attn, name, nn.Linear(config.hidden_size, old.out_features, bias=False))

    def forward(
        self,
        input_emb,
        hidden_states: torch.Tensor,
        cache_hidden: Optional[List[List[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.hidden_norm(hidden_states)

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

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class EdsdDraftModel(LlamaForCausalLMEagle3):

    config_class = LlamaConfig

    def _build_midlayer(self, config, attention_backend):
        return EdsdDecoderLayer(config, attention_backend=attention_backend)

    def _build_fc(self, config):
        if hasattr(config, "target_hidden_size"):
            return torch.nn.Linear(
                config.target_hidden_size * 2, config.hidden_size, bias=False
            )
        return torch.nn.Linear(
            config.hidden_size * 2, config.hidden_size, bias=False
        )

    def __init__(self, config, quant_config=None, attention_backend="sdpa") -> None:
        assert hasattr(config, "target_layer_ids") and len(config.target_layer_ids) == 2, \
            f"EdsdDraftModel requires exactly 2 target layers, got {getattr(config, 'target_layer_ids', None)}"

        super().__init__(config, quant_config=quant_config, attention_backend=attention_backend)

        self.edfuse = EDFuse(config)
        self.embnorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.size(-1) == self.config.hidden_size * 2
        return self.fc(hidden_states)

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
        input_embeds = self.embnorm(input_embeds)
        hidden_states = self.edfuse(hidden_states, input_embeds)
        return self.midlayer(
            input_emb=None,
            hidden_states=hidden_states,
            cache_hidden=cache_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=False,
            use_cache=False,
        )

