import json
import os
from typing import Union

import torch
from transformers import AutoConfig
from transformers import AutoModelForCausalLM as AutoModelForCausalLMBase
from transformers import (
    Llama4Config,
    Llama4TextConfig,
    LlamaConfig,
    PretrainedConfig,
    Qwen3Config,
    Qwen3MoeConfig,
)

from specforge.utils import default_torch_dtype

from .draft.llama3_eagle import LlamaForCausalLMEagle3
from .target.llama4 import Llama4ForCausalLM
from .target.qwen3 import Qwen3ForCausalLM
from .target.qwen3_moe import Qwen3MoeForCausalLM


class AutoEagle3DraftModel(AutoModelForCausalLMBase):
    # the model mapping is currently hardcoded, we should support lazy model mapping via registry
    _model_mapping = {
        LlamaConfig: LlamaForCausalLMEagle3,
    }

    @classmethod
    def from_config(cls, config: PretrainedConfig):
        """
        This class method takes a configuration object and create its model based on the
        _model_mapping class variable.

        Args:
            config (PretrainedConfig): A configuration object.

        Returns:
            A model instance.
        """
        # get the model class from the
        _model_cls = cls._model_mapping[type(config)]
        return _model_cls(config)


class AutoDistributedTargetModel(AutoModelForCausalLMBase):
    # the model mapping is currently hardcoded, we should support lazy model mapping via registry
    _model_mapping = {
        Llama4TextConfig: [Llama4ForCausalLM],
        Qwen3MoeConfig: [Qwen3MoeForCausalLM],
        Qwen3Config: [Qwen3ForCausalLM],
    }

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike[str]],
        torch_dtype: torch.dtype = None,
        device: str = None,
        **config_kwargs,
    ):
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, **config_kwargs
        )

        if isinstance(config, Llama4Config):
            config = config.text_config

        assert (
            type(config) in cls._model_mapping
        ), f"Unsupported config type: {type(config)}"
        model_cls = cls._model_mapping[type(config)][0]

        if device is None:
            device = torch.device("cpu")
        else:
            device = torch.device(device)

        if torch_dtype is None:
            torch_dtype = torch.get_default_dtype()

        # load model
        with default_torch_dtype(torch_dtype), torch.device(device):
            model = model_cls(config)
        model.load_checkpoint(pretrained_model_name_or_path)

        # just ensure that all the parameters follow the same dtype and device
        # model = model.to(torch_dtype)
        # model = model.to(device)

        return model


class AutoDraftModelConfig:

    _config_mapping = {
        "LlamaForCausalLMEagle3": LlamaConfig,
    }

    @classmethod
    def from_file(cls, config_path: str):
        """
        This class method takes a configuration file path and create its configuration object based on the
        _config_mapping class variable.

        Args:
            config_path (str): A path to a configuration file.

        Returns:
            A configuration object.
        """
        with open(config_path, "r") as f:
            config = json.load(f)

        # check for architectures
        architectures = config.get("architectures", None)

        if architectures is None:
            raise ValueError("No architectures found in the config file")

        if len(architectures) != 1:
            raise ValueError("Only one architecture is supported")

        architecture = architectures[0]

        if architecture not in cls._config_mapping:
            raise ValueError(f"Architecture {architecture} not supported")

        return cls._config_mapping[architecture].from_dict(config)
