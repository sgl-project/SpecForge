"""Transformers metadata around the single immutable draft configuration."""

import copy
from collections.abc import Mapping
from dataclasses import fields

from transformers import PreTrainedConfig

from specforge.config.draft import DraftConfig
from specforge.config.draft.codec import decode_config, encode_config

# HF and artifact metadata are separated from the canonical payload. Unknown saved keys
# still reach the strict codec rather than becoming arbitrary HF attributes.
_METADATA = frozenset(
    {
        "specforge_artifact",
        "auto_map",
        "transformers_version",
        "dtype",
        "torch_dtype",
        "output_hidden_states",
        "output_attentions",
        "return_dict",
        "chunk_size_feed_forward",
        "is_encoder_decoder",
        "id2label",
        "label2id",
        "num_labels",
        "problem_type",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "use_cache",
        "_name_or_path",
        "name_or_path",
        "_commit_hash",
        "attn_implementation",
        "experts_implementation",
    }
)
_STRUCTURE = frozenset(field.name for field in fields(DraftConfig)) | {
    "draft",
    "_draft",
    "model_type",
    "schema_version",
    "architectures",
    "extensions",
    "num_hidden_layers",
    "head_dim",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "block_size",
}


class SpecForgeDraftConfig(PreTrainedConfig):
    model_type = "specforge_draft"
    has_no_defaults_at_init = True

    def __init__(self, draft: DraftConfig, **metadata):
        if type(draft) is not DraftConfig:
            raise TypeError("draft must be a resolved DraftConfig")
        unknown = metadata.keys() - _METADATA
        if unknown:
            raise ValueError(f"unknown HF metadata: {sorted(unknown)}")
        object.__setattr__(self, "_draft", draft)
        super().__init__(
            architectures=[draft.architecture],
            tie_word_embeddings=draft.tie_word_embeddings,
            **copy.deepcopy(metadata),
        )

    @property
    def draft(self) -> DraftConfig:
        return self._draft

    @property
    def architectures(self) -> list[str]:
        # Returning a fresh list also prevents append/item assignment from
        # creating a second architecture selector.
        return [self.draft.architecture]

    @architectures.setter
    def architectures(self, value):
        if value != [self.draft.architecture]:
            raise ValueError("architectures must match draft.architecture")

    @property
    def tie_word_embeddings(self) -> bool:
        return self.draft.tie_word_embeddings

    @tie_word_embeddings.setter
    def tie_word_embeddings(self, value):
        if type(value) is not bool or value != self.draft.tie_word_embeddings:
            raise ValueError("tie_word_embeddings must match draft")

    def __setattr__(self, name, value):
        if name in {"draft", "_draft"}:
            raise AttributeError("draft is read-only")
        if name == "model_type":
            if value != type(self).model_type:
                raise ValueError("model_type must be specforge_draft")
            return
        if not name.startswith("_") and name not in _METADATA | {
            "architectures",
            "tie_word_embeddings",
        }:
            raise AttributeError(f"{name} is not HF metadata; structure lives in draft")
        super().__setattr__(name, value)

    def to_dict(self):
        metadata = {
            key: value for key, value in super().to_dict().items() if key in _METADATA
        }
        return {**metadata, **encode_config(self.draft)}

    def to_diff_dict(self):
        # Canonical structural fields are required, even if they resemble HF
        # defaults. There is deliberately no empty/default DraftConfig.
        return self.to_dict()

    @classmethod
    def from_dict(cls, config_dict: Mapping, **kwargs):
        if not isinstance(config_dict, Mapping):
            raise ValueError("config must be an object")
        values = copy.deepcopy(dict(config_dict))
        metadata = {key: values.pop(key) for key in list(values) if key in _METADATA}
        return_unused = kwargs.pop("return_unused_kwargs", False)
        for key in ("_from_auto", "_from_pipeline"):
            kwargs.pop(key, None)
        forbidden = kwargs.keys() & _STRUCTURE
        if forbidden:
            raise ValueError(f"cannot override draft structure: {sorted(forbidden)}")
        for key in list(kwargs):
            if key in _METADATA:
                if key in {"dtype", "torch_dtype"} and kwargs[key] == "auto":
                    continue
                metadata[key] = kwargs.pop(key)
        if kwargs and not return_unused:
            raise ValueError(f"unknown configuration overrides: {sorted(kwargs)}")
        config = cls(decode_config(values), **metadata)
        return (config, kwargs) if return_unused else config
