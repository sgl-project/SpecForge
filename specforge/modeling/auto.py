import json
import os
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
    PretrainedConfig,
    Qwen2Config,
    Qwen3Config,
    Qwen3MoeConfig,
    modeling_utils,
)

from .draft.llama3_eagle import LlamaForCausalLMEagle3
from .draft.qwen3_moe_eagle import Qwen3MoeForCausalLMEagle3
from .draft.qwen3_moe_mtp import Qwen3MoeForCausalLMMTP
from .target.custom_backend import (
    GptOssForCausalLM,
    Llama4ForCausalLM,
    LlamaForCausalLM,
    Phi3ForCausalLM,
    Qwen2ForCausalLM,
    Qwen3ForCausalLM,
    Qwen3MoeForCausalLM,
)


class AutoEagle3DraftModel(AutoModelForCausalLMBase):
    # the model mapping is currently hardcoded, we should support lazy model mapping via registry
    _model_mapping = {
        LlamaConfig: LlamaForCausalLMEagle3,
        Qwen3MoeConfig: Qwen3MoeForCausalLMEagle3,
    }

    # Architecture-level mapping for disambiguating configs that map to multiple model classes
    # (e.g., Qwen3MoeConfig can be either Eagle3 or MTP)
    _arch_model_mapping = {
        "LlamaForCausalLMEagle3": LlamaForCausalLMEagle3,
        "Qwen3MoeForCausalLMEagle3": Qwen3MoeForCausalLMEagle3,
        "Qwen3MoeForCausalLMMTP": Qwen3MoeForCausalLMMTP,
    }

    @classmethod
    def from_config(cls, config: PretrainedConfig, torch_dtype=None, **config_kwargs):
        """
        This class method takes a configuration object and create its model based on the
        _model_mapping class variable.

        When multiple model classes share the same config type (e.g., Qwen3MoeConfig
        for both Eagle3 and MTP), the architectures field in the config is used to
        disambiguate.

        Args:
            config (PretrainedConfig): A configuration object.

        Returns:
            A model instance.
        """
        # First try architecture-level disambiguation
        architectures = getattr(config, "architectures", None)
        if architectures and len(architectures) == 1 and architectures[0] in cls._arch_model_mapping:
            _model_cls = cls._arch_model_mapping[architectures[0]]
        else:
            # Fallback to config-type mapping
            _model_cls = cls._model_mapping[type(config)]
        model = _model_cls(config, **config_kwargs)

        # Convert model to specified dtype if provided
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)
        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike[str]],
        *model_args,
        **kwargs,
    ):
        original_warn = modeling_utils.logger.warning

        def filtered_warning(msg):
            if "embed_tokens.weight" in str(msg) and "initialized" in str(msg):
                return
            original_warn(msg)

        modeling_utils.logger.warning = filtered_warning

        try:
            # Check if we need architecture-level disambiguation
            config_path = os.path.join(str(pretrained_model_name_or_path), "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    raw_config = json.load(f)
                architectures = raw_config.get("architectures", [])
                if len(architectures) == 1 and architectures[0] in cls._arch_model_mapping:
                    _model_cls = cls._arch_model_mapping[architectures[0]]
                    config = AutoDraftModelConfig.from_file(config_path)
                    model = _model_cls(config, **kwargs)
                    # Load weights manually
                    import glob
                    from safetensors.torch import load_file
                    ckpt_files = sorted(glob.glob(os.path.join(str(pretrained_model_name_or_path), "*.safetensors")))
                    if ckpt_files:
                        state_dict = {}
                        for ckpt_file in ckpt_files:
                            state_dict.update(load_file(ckpt_file))
                        missing, unexpected = model.load_state_dict(state_dict, strict=False)
                        if missing:
                            print(f"Missing keys when loading checkpoint: {missing[:10]}{'...' if len(missing) > 10 else ''}")
                    else:
                        # Try pytorch_model.bin
                        pt_path = os.path.join(str(pretrained_model_name_or_path), "pytorch_model.bin")
                        if os.path.exists(pt_path):
                            state_dict = torch.load(pt_path, map_location="cpu", weights_only=False)
                            model.load_state_dict(state_dict, strict=False)
                    # Apply dtype
                    torch_dtype = kwargs.get("torch_dtype", None)
                    if torch_dtype is not None:
                        model = model.to(dtype=torch_dtype)
                    return model

            model = super().from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
        finally:
            modeling_utils.logger.warning = original_warn

        return model


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
        )

        if device is not None:
            model = model.to(device)
        else:
            model = model.cuda()
        return model


class AutoDraftModelConfig:

    _config_mapping = {
        "LlamaForCausalLMEagle3": LlamaConfig,
        "Qwen3MoeForCausalLMEagle3": Qwen3MoeConfig,
        "Qwen3MoeForCausalLMMTP": Qwen3MoeConfig,
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

        if "tie_word_embeddings" in config:
            print("Set draft model tie_word_embeddings to False")
            config["tie_word_embeddings"] = False

        # check for architectures
        architectures = config.get("architectures", None)

        if architectures is None:
            raise ValueError("No architectures found in the config file")

        if len(architectures) != 1:
            raise ValueError("Only one architecture is supported")

        architecture = architectures[0]

        if architecture not in cls._config_mapping:
            raise ValueError(f"Architecture {architecture} not supported")

        # If draft_vocab_size is not in config or is None, set draft_vocab_size to vocab_size
        if "draft_vocab_size" not in config or config["draft_vocab_size"] is None:
            config["draft_vocab_size"] = config.get("vocab_size", None)

        return cls._config_mapping[architecture].from_dict(config)
