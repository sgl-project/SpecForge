import json
import os
from typing import Union

from transformers import AutoConfig
from transformers import AutoModelForCausalLM as AutoModelForCausalLMBase
from transformers import PretrainedConfig, modeling_utils

from .draft.registry import DRAFT_REGISTRY, available_drafts


class AutoDraftModel(AutoModelForCausalLMBase):
    @classmethod
    def _model_cls_from_config(cls, config: PretrainedConfig):
        archs = getattr(config, "architectures", None) or []
        if len(archs) != 1 or archs[0] not in DRAFT_REGISTRY:
            raise ValueError(
                "draft config must name exactly one registered architecture; "
                f"got {archs!r}, available: {available_drafts()}"
            )
        return DRAFT_REGISTRY[archs[0]]

    @classmethod
    def from_config(cls, config: PretrainedConfig, torch_dtype=None, **config_kwargs):
        """
        This class method takes a configuration object and creates its model,
        resolving the class from DRAFT_REGISTRY via ``config.architectures``.

        Args:
            config (PretrainedConfig): A configuration object.

        Returns:
            A model instance.
        """
        _model_cls = cls._model_cls_from_config(config)
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
            config = kwargs.get("config")
            if config is None:
                config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            model_cls = cls._model_cls_from_config(config)
            kwargs = {**kwargs, "config": config}
            model = model_cls.from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
        finally:
            modeling_utils.logger.warning = original_warn

        return model


class AutoDraftModelConfig:
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

        if architecture not in DRAFT_REGISTRY:
            raise ValueError(
                f"Architecture {architecture} not registered; "
                f"available: {available_drafts()}"
            )
        config_cls = DRAFT_REGISTRY[architecture].config_class

        # If draft_vocab_size is not in config or is None, set draft_vocab_size to vocab_size
        if "draft_vocab_size" not in config or config["draft_vocab_size"] is None:
            config["draft_vocab_size"] = config.get("vocab_size", None)

        return config_cls.from_dict(config)
