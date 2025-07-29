import logging
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLTextConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

from ..utils import padding
from .base import Eagle3DraftModel
from .llama3_eagle import LlamaDecoderLayer, LlamaMLP, LlamaRMSNorm,LlamaAttention,LlamaForCausalLMEagle3
logger = logging.getLogger(__name__)



class Qwen2_5_VLRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2_5_VLTextConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        # In contrast to other models, Qwen2_5_VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen2_5VLAttention(LlamaAttention):
    def __init__(self, config):
        super().__init__(config)
    def _init_rope(self):
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=self.config)


class Qwen2_5VLDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, last=True):
        super().__init__(config, last)
        self.self_attn = Qwen2_5VLAttention(config=config)

class Qwen2_5_VLForCausalLMEagle3(LlamaForCausalLMEagle3):

    config_class = Qwen2_5_VLConfig

    def __init__(self, config, quant_config=None) -> None:
        super().__init__(config,quant_config)
        self.midlayer = Qwen2_5VLDecoderLayer(config)


