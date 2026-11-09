"""Select a reader by configuration/format, never by directory names."""

import json
from dataclasses import dataclass
from pathlib import Path

from specforge.config.draft import DraftConfig
from specforge.config.draft.codec.target import validate_target
from specforge.modeling.draft.hf_config import _METADATA, SpecForgeDraftConfig

from .draft_dir import read_artifact_metadata
from .formats.legacy import ARCHITECTURES, read_legacy
from .tensors import TensorStore

_ADAPTERS = {}


def register_adapter(name, matches, read):
    """Register config detection and a (config, weights, HF metadata) reader."""
    if not isinstance(name, str) or not name or name in _ADAPTERS:
        raise ValueError(f"invalid or duplicate checkpoint adapter: {name}")
    if not callable(matches) or not callable(read):
        raise TypeError("adapter detection and reader must be callable")
    _ADAPTERS[name] = (matches, read)


def resolve_directory(source, *, cache_dir=None):
    path = Path(str(source).removeprefix("file://")).expanduser()
    if path.exists():
        return path
    if path.is_absolute() or str(source).startswith((".", "~", "file://")):
        raise FileNotFoundError(path)
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            str(source),
            cache_dir=cache_dir,
            allow_patterns=["*.json", "*.safetensors", "*.bin"],
        )
    )


def read_json(source):
    if isinstance(source, dict):
        return dict(source)
    return json.loads(Path(source).read_text())


def target_configuration(directory):
    config = read_json(Path(directory) / "config.json")
    config = dict(config.get("text_config", config))
    if (
        "LongcatFlashForCausalLM" in config.get("architectures", ())
        and "num_hidden_layers" not in config
    ):
        config["num_hidden_layers"] = config["num_layers"]
    return config


@dataclass(frozen=True)
class DraftCheckpoint:
    config: DraftConfig
    weights: dict
    hf_metadata: dict
    source: str


def _canonical(payload, store, target):
    shell = SpecForgeDraftConfig.from_dict(payload)
    if target is not None:
        validate_target(shell.draft, target)
    metadata = {k: v for k, v in payload.items() if k in _METADATA}
    return shell.draft, store.refs(), metadata


register_adapter(
    "canonical", lambda p: p.get("model_type") == "specforge_draft", _canonical
)
register_adapter(
    "legacy",
    lambda p: p.get("model_type") != "specforge_draft"
    and isinstance(p.get("architectures"), list)
    and len(p["architectures"]) == 1
    and p["architectures"][0] in ARCHITECTURES,
    read_legacy,
)


def read_checkpoint(
    source, *, target_config=None, legacy_config=None, adapter=None, cache_dir=None
):
    path = resolve_directory(source, cache_dir=cache_dir)
    directory = path.parent if path.is_file() else path
    config_path = directory / "config.json"
    if config_path.exists():
        if legacy_config is not None:
            raise ValueError(
                "legacy_config is only for checkpoints without config.json"
            )
        payload = read_json(config_path)
    elif legacy_config is not None:
        payload = read_json(legacy_config)
    else:
        raise ValueError(
            "checkpoint has no config.json; supply legacy_config explicitly"
        )
    if not isinstance(payload, dict):
        raise ValueError("checkpoint config must be an object")
    artifact = read_artifact_metadata(payload)
    if artifact and payload.get("model_type") != "specforge_draft":
        raise ValueError("canonical artifact metadata contradicts its configuration")
    matches = [name for name, (detect, _) in _ADAPTERS.items() if detect(payload)]
    if adapter is not None:
        if adapter not in matches:
            raise ValueError(f"adapter {adapter!r} does not match checkpoint")
        matches = [adapter]
    if len(matches) != 1:
        raise ValueError(
            f"expected one checkpoint adapter, found {matches}; select adapter explicitly if ambiguous"
        )
    config, weights, metadata = _ADAPTERS[matches[0]][1](
        payload, TensorStore(path), target_config
    )
    return DraftCheckpoint(config, weights, metadata, str(path))
