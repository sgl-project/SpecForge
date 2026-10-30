"""Load and normalize draft configs independently of training-run policy.

The configuration registry owns the HF adapter and typed architecture options.
This loader does not infer the draft architecture from the target or construct
model parameters.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping
from typing import Any

from transformers import PretrainedConfig

from .base import integer
from .format import decode_format
from .layer_decoder import decode_layers, strip_legacy_layer_fields
from .model import DraftModelConfig, inline_values
from .registry import DraftConfigRegistration, resolve_draft_config_type


class AutoDraftConfig:
    """Resolve inline settings or read HF checkpoint metadata.

    Training uses from_inline; legacy readers return HF adapters. Models resolve
    either representation to immutable settings before allocating parameters.
    """

    @classmethod
    def from_inline(cls, payload: Mapping[str, Any]) -> DraftModelConfig:
        """Resolve the YAML draft section without parsing it through HF."""
        if not isinstance(payload, Mapping):
            raise ValueError("draft config must be an object")
        payload = decode_format(payload)
        architectures = payload.get("architectures")
        if (
            not isinstance(architectures, (list, tuple))
            or len(architectures) != 1
            or not isinstance(architectures[0], str)
        ):
            raise ValueError("draft config must name exactly one architecture")
        registration = resolve_draft_config_type(architectures[0])
        values = inline_values(payload, model_type=registration.config_class.model_type)
        integer(values.draft_vocab_size, "draft_vocab_size")
        layers = decode_layers(values, registration.options_class)
        if registration.options_class is not None:
            options = registration.options_class.from_model_config(values)
            values = options.to_model_config(values)
        data = strip_legacy_layer_fields(values.to_dict())
        data["layers"] = layers
        return DraftModelConfig(data)

    @classmethod
    def resolve(cls, config, *, architecture: str | None = None) -> DraftModelConfig:
        """Accept a resolved config, or adapt an existing HF checkpoint config."""
        if isinstance(config, DraftModelConfig):
            return config
        if isinstance(config, Mapping):
            return cls.from_inline(config)
        payload = config.to_dict()
        # An explicitly selected model class also accepts legacy base-model
        # metadata (e.g. LlamaForCausalLM). Never rewrite resolved inline settings.
        if architecture is not None:
            payload["architectures"] = [architecture]
        payload["_attn_implementation"] = getattr(config, "_attn_implementation", None)
        return cls.from_inline(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PretrainedConfig:
        """Read legacy HF checkpoint metadata; training uses from_inline."""
        if not isinstance(payload, Mapping):
            raise ValueError("draft config must be a JSON object")
        if "schema_version" in payload:
            return cls.from_inline(payload)
        payload = copy.deepcopy(dict(payload))
        architectures = payload.get("architectures")
        if (
            not isinstance(architectures, (list, tuple))
            or len(architectures) != 1
            or not isinstance(architectures[0], str)
        ):
            raise ValueError(
                "draft config must name exactly one architecture; "
                f"got {architectures!r}"
            )
        architecture = architectures[0]
        registration = resolve_draft_config_type(architecture)

        payload["architectures"] = [architecture]
        payload["tie_word_embeddings"] = False
        payload.setdefault("use_cache", True)
        config = registration.config_class.from_dict(payload)
        if payload.get("draft_vocab_size") is None:
            config.draft_vocab_size = config.vocab_size
        if type(config.draft_vocab_size) is not int or config.draft_vocab_size <= 0:
            raise ValueError("draft config draft_vocab_size must be a positive integer")
        return config

    @classmethod
    def finalize(cls, config: PretrainedConfig) -> PretrainedConfig:
        """Normalize and validate model-specific settings before allocation.

        Typed options own validation and normalization. Resolving configuration
        never imports the model registry or calls model implementation hooks.
        """
        if isinstance(config, DraftModelConfig):
            return config
        result = cls.resolve(config).to_hf_config()
        if hasattr(result, "dflash_config"):
            result.dflash_config.pop("block_size", None)
        return result

    @classmethod
    def validate_training_config(cls, config: PretrainedConfig) -> None:
        """Check training-only requirements on an already finalized config."""
        registration = cls._registration(config)
        if registration.options_class is not None:
            registration.options_class.validate_training_config(config)

    @staticmethod
    def _registration(config: PretrainedConfig) -> DraftConfigRegistration:
        architectures = getattr(config, "architectures", None) or []
        if (
            not isinstance(architectures, (list, tuple))
            or len(architectures) != 1
            or not isinstance(architectures[0], str)
        ):
            raise ValueError(
                "draft config must name exactly one registered architecture"
            )
        return resolve_draft_config_type(architectures[0])

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> PretrainedConfig:
        try:
            with open(path, encoding="utf-8") as stream:
                payload = json.load(stream)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid draft config JSON {path}: {exc}") from exc
        return cls.from_dict(payload)

    @classmethod
    def from_pretrained(
        cls,
        source: str | os.PathLike[str],
        *,
        cache_dir: str | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        force_download: bool = False,
        subfolder: str = "",
    ) -> PretrainedConfig:
        """Read JSON, a local directory, or an HF repository.

        ``trust_remote_code`` is accepted for caller compatibility. Draft
        classes are explicitly registered locally; remote Python is not needed
        to read their configuration, even when config.json contains auto_map.
        """
        if not source:
            raise ValueError("draft config source must be non-empty")
        raw_source = os.fspath(source)
        path = os.path.abspath(os.path.expanduser(raw_source.removeprefix("file://")))
        if os.path.isfile(path):
            if subfolder:
                raise ValueError("subfolder cannot be used with a draft config file")
            return cls.from_file(path)
        if os.path.isdir(path):
            config_path = os.path.join(path, subfolder, "config.json")
            if not os.path.isfile(config_path):
                raise FileNotFoundError(
                    f"draft model directory has no config.json: {os.path.dirname(config_path)}"
                )
            return cls.from_file(config_path)
        if raw_source.startswith((".", "/", "~", "file://")) or raw_source.endswith(
            ".json"
        ):
            raise FileNotFoundError(f"draft model config does not exist: {source}")

        # Read the raw payload so model_type does not select a second config
        # class (or inject its defaults) before the draft registry gets a say.
        payload, _ = PretrainedConfig.get_config_dict(
            raw_source,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            subfolder=subfolder,
        )
        config = cls.from_dict(payload)
        config._name_or_path = raw_source
        return config
