import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import GptOssConfig
from transformers.cache_utils import Cache
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP

from .base import Eagle3DraftModel
from .llama3_eagle import LlamaForCausalLMEagle3

logger = logging.getLogger(__name__)


class GptOssForCausalLMEagle3(LlamaForCausalLMEagle3):

    config_class = GptOssConfig

    def __init__(self, config, quant_config=None, attention_backend="sdpa") -> None:
        super().__init__(config)
        self.midlayer.mlp = GptOssMLP(config)


__all__ = ["GptOssForCausalLMEagle3"]
