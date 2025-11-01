import json
import os
import warnings
from typing import Optional, Union

import torch
from transformers import AutoConfig
from transformers import AutoModelForCausalLM as AutoModelForCausalLMBase
from transformers import (
    GptOssConfig,
    Llama4Config,
    Llama4TextConfig,
    LlamaConfig,
    Phi3Config,
    Qwen2Config,
    Qwen3Config,
    Qwen3MoeConfig,
)

from .gpt_oss import GptOssForCausalLM
from .llama import LlamaForCausalLM
from .llama4 import Llama4ForCausalLM
from .phi3 import Phi3ForCausalLM
from .qwen2 import Qwen2ForCausalLM
from .qwen3 import Qwen3ForCausalLM
from .qwen3_moe import Qwen3MoeForCausalLM


class AutoDistributedTargetModel(AutoModelForCausalLMBase):
    # the model mapping is currently hardcoded, we should support lazy model mapping via registry
    _model_mapping = {
        Llama4TextConfig: [Llama4ForCausalLM],
        Qwen3MoeConfig: [Qwen3MoeForCausalLM],
        Qwen2Config: [Qwen2ForCausalLM],
        LlamaConfig: [LlamaForCausalLM],
        Qwen3Config: [Qwen3ForCausalLM],
        Phi3Config: [Phi3ForCausalLM],
        GptOssConfig: [GptOssForCausalLM],
    }

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike[str]],
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **config_kwargs,
    ):
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
        )

        if isinstance(config, Llama4Config):
            config = config.text_config

        assert (
            type(config) in cls._model_mapping
        ), f"Unsupported config type: {type(config)}"
        model_cls = cls._model_mapping[type(config)][0]
        model = model_cls.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            **config_kwargs,
        ).cuda()
        return model
