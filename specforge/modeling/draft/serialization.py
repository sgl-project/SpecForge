"""HF adapters for models whose runtime settings are immutable."""

from copy import copy, deepcopy

from specforge.config.draft.auto import AutoDraftConfig
from specforge.config.draft.hf import DraftSaveConfig
from specforge.config.draft.model import DraftModelConfig


class DraftSerializationMixin:
    def __setattr__(self, name, value):
        if name == "config" and isinstance(
            self.__dict__.get("config"), DraftModelConfig
        ):
            raise AttributeError("draft model.config is fixed after construction")
        super().__setattr__(name, value)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and args:
            config = AutoDraftConfig.from_pretrained(
                args[0],
                **{
                    key: kwargs[key]
                    for key in (
                        "cache_dir",
                        "revision",
                        "token",
                        "local_files_only",
                        "force_download",
                        "subfolder",
                    )
                    if key in kwargs
                },
            )
        if isinstance(config, DraftModelConfig):
            kwargs["config"] = config.to_hf_config(for_export=False)
        elif config is not None:
            kwargs["config"] = config
        return super().from_pretrained(*args, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        # Share weights; only the HF adapter may receive serialization metadata.
        export_model = copy(self)
        export_model.__dict__["config"] = DraftSaveConfig(self.config)
        if hasattr(self, "generation_config"):
            export_model.generation_config = deepcopy(self.generation_config)
        return super(DraftSerializationMixin, export_model).save_pretrained(
            *args, **kwargs
        )
