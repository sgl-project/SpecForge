import torch
import torch.nn as nn

from typing import List, Optional, Union

from transformers.models.llama.modeling_llama import LlamaModel
from transformers.models.llama.configuration_llama import LlamaConfig

from transformers.cache_utils import Cache


class LlamaForCausalLMEagle(LlamaModel):
    def __init__(self, config: LlamaConfig):

        config._attn_implementation = "flash_attention_2"

        # Monkey patch post init to no op - Hugging Face weight initialization makes model worse
        def noop_post_init():
            print("Running no op post init")
            pass

        self.post_init = noop_post_init
        super().__init__(config)

        # Follow EAGLE removal of norm layers
        del self.norm
        setattr(self, "norm", lambda x: x)
        del self.layers[0].input_layernorm
        setattr(self.layers[0], "input_layernorm", lambda x: x)

        # This projection layer maps the concatenated [token_emb; full_hidden] (of size 2*hidden_size)
        # down to hidden_size.
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=True)

    def load_embedding_weights(self, weights):
        self.embed_tokens.weight = nn.Parameter(weights)

    def forward(
        self,
        hidden_state: torch.Tensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ):
        token_emb = self.embed_tokens(input_ids)
        concat = torch.cat([token_emb, hidden_state], dim=-1)

        proj = self.fc(concat)

        outputs = super().forward(
            input_ids=None,
            inputs_embeds=proj,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        return hidden_states
