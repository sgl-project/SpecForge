"""Read saved settings only to check compatibility with the user's config."""

from __future__ import annotations

import json
import os
from pathlib import Path

from transformers import PretrainedConfig

from ..auto import AutoDraftConfig
from ..model import DraftModelConfig
from .contract import draft_config_contract, validate_draft_config_contract


def read_checkpoint_contract(source, *, subfolder="", **hub_options) -> dict:
    """Return comparison metadata, never a config for model construction."""
    raw_source = os.fspath(source)
    path = Path(raw_source.removeprefix("file://")).expanduser()
    if path.is_file():
        if subfolder:
            raise ValueError("subfolder cannot be used with a config file")
        config_path = path
    elif path.is_dir():
        config_path = path / subfolder / "config.json"
    elif raw_source.startswith((".", "/", "~", "file://")) or raw_source.endswith(
        ".json"
    ):
        raise FileNotFoundError(f"checkpoint config does not exist: {source}")
    else:
        payload, _ = PretrainedConfig.get_config_dict(
            raw_source, subfolder=subfolder, **hub_options
        )
        return draft_config_contract(AutoDraftConfig.from_dict(payload))
    with config_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    return draft_config_contract(AutoDraftConfig.from_dict(payload))


def validate_checkpoint_config(config: DraftModelConfig, source, **hub_options) -> None:
    """Reject incompatible weights without applying saved values or defaults."""
    validate_draft_config_contract(
        draft_config_contract(config),
        read_checkpoint_contract(source, **hub_options),
        source=os.fspath(source),
    )
