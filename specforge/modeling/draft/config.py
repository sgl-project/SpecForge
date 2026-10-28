"""Load and normalize draft configs independently of training-run policy.

The architecture registry owns the HF config class. This loader does not infer
the draft architecture from the target or construct model parameters.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping
from typing import Any

from transformers import PretrainedConfig

from .registry import DRAFT_REGISTRY, available_drafts


class AutoDraftConfig:
    """One parsing contract for training, export, and pretrained loading.

    Existing Llama/Qwen config classes and custom fields are preserved. An
    omitted draft vocabulary inherits the resolved HF vocabulary; explicit
    null is rejected. Draft embedding and output weights are always untied.
    Algorithm-specific overrides remain owned by DraftConfigProvider.
    """

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PretrainedConfig:
        if not isinstance(payload, Mapping):
            raise ValueError("draft config must be a JSON object")
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
        if architecture not in DRAFT_REGISTRY:
            raise ValueError(
                f"draft architecture {architecture!r} is not registered; "
                f"available: {available_drafts()}"
            )
        if "draft_vocab_size" in payload and payload["draft_vocab_size"] is None:
            raise ValueError("draft config draft_vocab_size cannot be null")

        payload["architectures"] = [architecture]
        payload["tie_word_embeddings"] = False
        payload.setdefault("use_cache", True)
        config = DRAFT_REGISTRY[architecture].config_class.from_dict(payload)
        if "draft_vocab_size" not in payload:
            config.draft_vocab_size = config.vocab_size
        if type(config.draft_vocab_size) is not int or config.draft_vocab_size <= 0:
            raise ValueError("draft config draft_vocab_size must be a positive integer")
        return config

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
        return cls.from_dict(payload)
