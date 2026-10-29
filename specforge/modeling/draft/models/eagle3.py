from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.models.llama.configuration_llama import LlamaConfig

from specforge.modeling.draft.config.eagle3 import Eagle3Config, EaglePretrainedConfig
from specforge.modeling.draft.models.base import Eagle3DraftModel
from specforge.modeling.draft.registry import register_draft
from specforge.utils import print_with_rank

from .layers.attention.eagle import (
    LlamaAttention,
    LlamaFlashAttention,
    LlamaFlexAttention,
    LlamaUSPFlashAttention,
)
from .layers.ffn import build_ffn
from .layers.ffn.dense import LlamaMLP
from .layers.normalization import LlamaRMSNorm
from .layers.residual import build_residual


class LlamaDecoderLayer(nn.Module):
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
        self.mlp = build_ffn(config, dense_factory=LlamaMLP)
        self.residual = build_residual(config)
        # self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # if self.index!=0:

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
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_values (`Cache`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.hidden_norm(self.residual.branch_input(hidden_states))
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
        hidden_states = self.residual.combine(residual, hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(
            self.residual.branch_input(hidden_states)
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.residual.combine(residual, hidden_states)

        # outputs = (hidden_states, return_hidden)
        return hidden_states


@register_draft
class LlamaForCausalLMEagle3(Eagle3DraftModel):

    config_class = EaglePretrainedConfig
    draft_options_class = Eagle3Config

    @classmethod
    def prepare_config(cls, config: LlamaConfig) -> LlamaConfig:
        return cls.draft_options_class.from_model_config(config).to_model_config(config)

    def __init__(self, config, quant_config=None, attention_backend="sdpa") -> None:
        options = self.draft_options_class.from_model_config(config)
        config = options.to_model_config(config)
        super().__init__(config)
        self.draft_options = options
        self.config = config
        self.quant_config = quant_config

        self.vocab_size = config.vocab_size
        self.draft_vocab_size = config.draft_vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.midlayer = LlamaDecoderLayer(config, attention_backend=attention_backend)

        self.target_hidden_size = options.target_hidden_size

        self.fc = torch.nn.Linear(
            self.target_hidden_size * 3,
            config.hidden_size,
            bias=False,
        )
        if options.fc_norm:
            self.fc_norm = nn.ModuleList(
                [
                    LlamaRMSNorm(self.target_hidden_size, eps=config.rms_norm_eps)
                    for _ in range(3)
                ]
            )
        else:
            self.fc_norm = None

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_output = options.norm_output
        self.lm_head = nn.Linear(
            config.hidden_size, config.draft_vocab_size, bias=False
        )

        # create vocab buffers
        t2d = torch.ones(self.vocab_size, dtype=torch.bool)
        d2t = torch.zeros(self.draft_vocab_size, dtype=torch.int64)
        self.register_buffer("t2d", t2d)
        self.register_buffer("d2t", d2t)

        self.post_init()

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        ttt_length: int = 1,
    ):
        """
        Arguments:
            hidden_states (`torch.FloatTensor`): input to the layer, cat low, mid high hidden_states of shape `(batch, seq_len, hidden_states * 3)`
            input_ids (`torch.LongTensor`): input ids of shape `(batch, seq_len)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (`torch.LongTensor`, *optional*): position ids of shape `(batch, seq_len)`
        """
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
        attention_mask = self.prepare_decoder_attention_mask(
            attention_mask=attention_mask,
            hidden_states=hidden_states,
            batch_size=batch_size,
            seq_length=seq_length,
            past_key_values_length=0,
        )

        # fc
        hidden_states = self.project_hidden_states(hidden_states)

        return self.backbone(
            input_embeds=inputs_embeds,
            hidden_states=hidden_states,
            cache_hidden=cache_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.size(-1) == self.target_hidden_size * 3
        if self.fc_norm is not None:
            chunks = hidden_states.chunk(3, dim=-1)
            hidden_states = torch.cat(
                [norm(chunk) for norm, chunk in zip(self.fc_norm, chunks)],
                dim=-1,
            )
        return self.fc(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(
            hidden_states if self.norm_output else self.norm(hidden_states)
        )

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
        hidden_states = self.midlayer(
            input_emb=input_embeds,
            hidden_states=hidden_states,
            cache_hidden=cache_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=False,
            use_cache=False,
        )
        return self.norm(hidden_states) if self.norm_output else hidden_states
