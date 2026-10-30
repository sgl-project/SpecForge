import os
from typing import Union

from transformers import AutoModelForCausalLM as AutoModelForCausalLMBase
from transformers import PretrainedConfig, modeling_utils

from specforge.config.draft import AutoDraftConfig

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
                config = AutoDraftConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    **{
                        key: kwargs[key]
                        for key in (
                            "cache_dir",
                            "trust_remote_code",
                            "revision",
                            "token",
                            "local_files_only",
                            "force_download",
                            "subfolder",
                        )
                        if key in kwargs
                    },
                )
            model_cls = cls._model_cls_from_config(config)
            kwargs = {**kwargs, "config": config}
            model = model_cls.from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
        finally:
            modeling_utils.logger.warning = original_warn

        return model


class AutoDraftModelConfig(AutoDraftConfig):
    """Backward-compatible name for the shared draft config loader."""
