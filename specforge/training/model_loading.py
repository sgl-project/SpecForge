# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Draft configuration and weights-only initialization for unified training.

This module deliberately keeps model initialization separate from training
resume.  A warm start can read draft weights from a Hugging Face checkpoint or
from SpecForge's shared ``training_state.pt`` payload, but it has no access to
an optimizer, scheduler, controller counters, or RNG state.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from specforge.algorithms.common.providers import DraftConfigProvider
    from specforge.config import Config

logger = logging.getLogger(__name__)

_CONFIG_FILE = "config.json"
_STATE_FILE = "training_state.pt"

# These are the architecture fields the legacy EAGLE generator copied from the
# target, plus RoPE/attention fields needed by newer Qwen-family targets.  A
# whitelist avoids feeding a target-specific nested config into the registered
# Llama/Qwen3 draft classes.
_TARGET_ARCHITECTURE_FIELDS = (
    "vocab_size",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "max_position_embeddings",
    "rms_norm_eps",
    "hidden_act",
    "initializer_range",
    "attention_bias",
    "attention_dropout",
    "head_dim",
    "rope_theta",
    "rope_scaling",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "sliding_window",
    "use_sliding_window",
    "max_window_layers",
)


@dataclass(frozen=True)
class WarmStartReport:
    """Observable result of loading draft weights without training state."""

    source: str
    checkpoint_format: Literal["specforge", "pretrained"]
    loaded_keys: int
    missing_keys: Tuple[str, ...]
    loaded_embedding: bool


def _without_file_uri(source: str) -> str:
    return source[len("file://") :] if source.startswith("file://") else source


def _looks_like_local_path(source: str) -> bool:
    return (
        source.startswith((".", "/", "~", "file://"))
        or source.endswith(".json")
        or os.path.exists(os.path.expanduser(_without_file_uri(source)))
    )


def _draft_config_from_dict(payload: Dict[str, Any]) -> "PretrainedConfig":
    # Importing the draft package registers every built-in architecture before
    # consulting the registry. Model modules keep accelerator-only dependencies
    # behind execution boundaries, so this remains safe in a CPU producer.
    import specforge.modeling.draft  # noqa: F401
    from specforge.modeling.draft.registry import DRAFT_REGISTRY, available_drafts

    payload = dict(payload)
    architectures = payload.get("architectures") or []
    if not isinstance(architectures, (list, tuple)) or len(architectures) != 1:
        raise ValueError(
            "draft config must name exactly one architecture; " f"got {architectures!r}"
        )
    architecture = architectures[0]
    if architecture not in DRAFT_REGISTRY:
        raise ValueError(
            f"draft architecture {architecture!r} is not registered; "
            f"available: {available_drafts()}"
        )

    payload["architectures"] = [architecture]
    payload["tie_word_embeddings"] = False
    payload.setdefault("use_cache", True)
    if payload.get("draft_vocab_size") is None:
        payload["draft_vocab_size"] = payload.get("vocab_size")
    return DRAFT_REGISTRY[architecture].config_class.from_dict(payload)


def load_draft_config_source(
    source: str,
    *,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = False,
) -> "PretrainedConfig":
    """Load a registered draft config from JSON, a directory, or an HF repo."""

    if not source:
        raise ValueError("draft config source must be non-empty")
    local_source = os.path.abspath(os.path.expanduser(_without_file_uri(str(source))))
    if os.path.isfile(local_source):
        config_path = local_source
    elif os.path.isdir(local_source):
        config_path = os.path.join(local_source, _CONFIG_FILE)
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"draft model directory has no {_CONFIG_FILE}: {local_source}"
            )
    else:
        if _looks_like_local_path(str(source)):
            raise FileNotFoundError(f"draft model config does not exist: {source}")
        from transformers import AutoConfig

        loaded = AutoConfig.from_pretrained(
            source,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        return _draft_config_from_dict(loaded.to_dict())

    try:
        with open(config_path, encoding="utf-8") as stream:
            payload = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid draft config JSON {config_path}: {exc}") from exc
    return _draft_config_from_dict(payload)


def _checkpoint_config_source(checkpoint_source: Optional[str]) -> Optional[str]:
    """Return a config-bearing warm-start source, if it can be identified."""

    if not checkpoint_source:
        return None
    raw_source = str(checkpoint_source)
    local_source = os.path.abspath(os.path.expanduser(_without_file_uri(raw_source)))
    if os.path.isfile(local_source):
        if os.path.basename(local_source) == _CONFIG_FILE:
            return local_source
        sibling = os.path.join(os.path.dirname(local_source), _CONFIG_FILE)
        return sibling if os.path.isfile(sibling) else None
    if os.path.isdir(local_source):
        direct = os.path.join(local_source, _CONFIG_FILE)
        if os.path.isfile(direct):
            return direct
        # A run root may point at a future checkpoint layout that also persists
        # config.json.  Do not resolve or read training state just to find it.
        candidates = []
        for pointer in glob.glob(os.path.join(local_source, "*-latest")):
            candidate = os.path.join(os.path.realpath(pointer), _CONFIG_FILE)
            if os.path.isfile(candidate):
                candidates.append(candidate)
        if len(set(candidates)) == 1:
            return candidates[0]
        return None
    if _looks_like_local_path(raw_source):
        return None
    # A non-local warm-start source is an HF repository. Its config is the best
    # architecture contract unless the run supplies an explicit config source.
    return raw_source


def _target_text_config(config: "PretrainedConfig") -> "PretrainedConfig":
    return getattr(config, "text_config", config)


def _serializable_config_value(value: Any) -> Any:
    try:
        import torch

        if isinstance(value, torch.dtype):
            return str(value).replace("torch.", "")
    except ImportError:
        pass
    return value


def _generate_draft_config(
    cfg: "Config", provider: "DraftConfigProvider"
) -> "PretrainedConfig":
    defaults = provider.target_defaults
    if defaults is None:
        raise ValueError(
            f"training.strategy={cfg.training.strategy!r} requires "
            "model.draft_model_config "
            "or a pretrained model.draft_checkpoint_path containing config.json"
        )

    from specforge.modeling.target.target_utils import (
        load_target_config,
        target_text_config,
        target_vocab_size,
    )

    target_config = load_target_config(
        cfg.model.target_model_path,
        cache_dir=cfg.model.cache_dir,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    target_config = target_text_config(target_config)
    payload: Dict[str, Any] = {}
    for field in _TARGET_ARCHITECTURE_FIELDS:
        value = getattr(target_config, field, None)
        if value is not None:
            payload[field] = _serializable_config_value(value)
    payload["vocab_size"] = target_vocab_size(target_config)

    required = [name for name in ("vocab_size", "hidden_size") if name not in payload]
    if required:
        raise ValueError(
            f"target config {cfg.model.target_model_path!r} cannot generate a "
            f"draft config; missing {required}"
        )
    payload["architectures"] = [provider.architecture]
    payload["model_type"] = defaults.model_type
    payload["num_hidden_layers"] = defaults.num_hidden_layers
    payload["tie_word_embeddings"] = False
    payload["use_cache"] = True
    if payload.get("pad_token_id") is None:
        payload["pad_token_id"] = 0
    if defaults.draft_vocab_size is not None:
        payload["draft_vocab_size"] = defaults.draft_vocab_size
    if defaults.populate is not None:
        defaults.populate(payload, target_config, cfg)
    return _draft_config_from_dict(payload)


def _apply_draft_overrides(
    cfg: "Config",
    draft_config: "PretrainedConfig",
    provider: "DraftConfigProvider",
) -> None:
    num_layers = cfg.model.draft_num_hidden_layers
    if num_layers is not None:
        draft_config.num_hidden_layers = num_layers
    if cfg.model.draft_block_size is not None:
        draft_config.block_size = cfg.model.draft_block_size
    if provider.apply_overrides is not None:
        provider.apply_overrides(cfg, draft_config)


def resolve_draft_config(
    cfg: "Config", *, provider: "DraftConfigProvider"
) -> "PretrainedConfig":
    """Resolve and validate the draft architecture for one typed run config."""

    source = cfg.model.draft_model_config
    if not source:
        source = _checkpoint_config_source(cfg.model.draft_checkpoint_path)
    if source:
        draft_config = load_draft_config_source(
            source,
            cache_dir=cfg.model.cache_dir,
            trust_remote_code=cfg.model.trust_remote_code,
        )
    else:
        draft_config = _generate_draft_config(cfg, provider)

    expected = provider.architecture
    architectures = list(getattr(draft_config, "architectures", None) or [])
    if architectures != [expected]:
        raise ValueError(
            f"training.strategy={cfg.training.strategy!r} requires draft "
            f"architecture {expected}, got {architectures!r}"
        )
    _apply_draft_overrides(cfg, draft_config, provider)
    return draft_config


def draft_config_dict(
    cfg: "Config", *, provider: "DraftConfigProvider"
) -> Dict[str, Any]:
    """JSON-compatible resolved draft config for capture-only producer roles."""

    return resolve_draft_config(cfg, provider=provider).to_dict()


def _has_pretrained_weights(path: str) -> bool:
    names = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    return any(os.path.isfile(os.path.join(path, name)) for name in names)


def _runtime_state_file(source: str) -> Optional[str]:
    """Resolve SpecForge checkpoint storage without loading resume state."""

    path = os.path.abspath(os.path.expanduser(_without_file_uri(str(source))))
    if os.path.isfile(path):
        return path if os.path.basename(path) == _STATE_FILE else None
    if not os.path.isdir(path):
        return None

    direct = os.path.join(path, _STATE_FILE)
    if os.path.isfile(direct) and not _has_pretrained_weights(path):
        return direct
    has_run_pointers = bool(
        glob.glob(os.path.join(path, "*-latest"))
        or glob.glob(os.path.join(path, "*-step*"))
    )
    if not has_run_pointers:
        return None

    from specforge.training.checkpoint import CheckpointManager

    checkpoint_dir = CheckpointManager.resolve_resume_dir(path)
    return os.path.join(checkpoint_dir, _STATE_FILE)


def _load_specforge_draft_state(
    state_path: str, *, expected_strategy: str
) -> Dict[str, Any]:
    import torch

    try:
        # The restricted unpickler accepts tensors and primitive metadata while
        # refusing arbitrary checkpoint objects. Only draft_state_dict is used.
        state = torch.load(state_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"cannot read weights-only SpecForge checkpoint {state_path}: {exc}"
        ) from exc
    if not isinstance(state, dict) or not isinstance(
        state.get("draft_state_dict"), dict
    ):
        raise ValueError(
            f"SpecForge warm-start checkpoint has no draft_state_dict: {state_path}"
        )
    saved_strategy = state.get("strategy")
    if saved_strategy is not None and saved_strategy != expected_strategy:
        raise ValueError(
            f"warm-start checkpoint {state_path} was written by strategy "
            f"{saved_strategy!r}; this run trains {expected_strategy!r}"
        )
    return state["draft_state_dict"]


def _load_pretrained_draft_state(
    source: str,
    *,
    draft_config: "PretrainedConfig",
    cache_dir: Optional[str],
    trust_remote_code: bool,
) -> Dict[str, Any]:
    from specforge.modeling.auto import AutoDraftModel

    loaded, loading_info = AutoDraftModel.from_pretrained(
        source,
        config=draft_config,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        output_loading_info=True,
    )
    missing_from_source = set(loading_info.get("missing_keys") or [])
    state = {
        key: value.detach().cpu()
        for key, value in loaded.state_dict().items()
        if key not in missing_from_source
    }
    del loaded
    return state


def warm_start_draft_model(
    model: Any,
    source: str,
    *,
    draft_config: "PretrainedConfig",
    strategy: str,
    allow_missing_embedding: bool = False,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = False,
) -> WarmStartReport:
    """Load only draft weights, never optimizer/counters/RNG training state."""

    runtime_state = _runtime_state_file(source)
    if runtime_state is not None:
        checkpoint_format: Literal["specforge", "pretrained"] = "specforge"
        state = _load_specforge_draft_state(runtime_state, expected_strategy=strategy)
    else:
        checkpoint_format = "pretrained"
        state = _load_pretrained_draft_state(
            source,
            draft_config=draft_config,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )

    if not state:
        raise ValueError(f"warm-start checkpoint {source!r} contains no draft weights")
    try:
        result = model.load_state_dict(state, strict=False)
    except RuntimeError as exc:
        raise ValueError(
            f"warm-start checkpoint {source!r} has incompatible draft tensor "
            f"shapes: {exc}"
        ) from exc
    loaded_keys = len(state) - len(result.unexpected_keys)
    if result.unexpected_keys or loaded_keys == 0:
        raise ValueError(
            f"warm-start checkpoint {source!r} does not match this draft model: "
            f"loaded={loaded_keys}/{len(state)}, "
            f"unexpected={sorted(result.unexpected_keys)}"
        )
    allowed_missing = set()
    if allow_missing_embedding:
        allowed_missing = {key for key in result.missing_keys if "embed" in key.lower()}
    required_missing = sorted(set(result.missing_keys) - allowed_missing)
    if required_missing:
        raise ValueError(
            f"warm-start checkpoint {source!r} is missing draft weights required "
            f"by this architecture: {required_missing}"
        )

    report = WarmStartReport(
        source=str(source),
        checkpoint_format=checkpoint_format,
        loaded_keys=loaded_keys,
        missing_keys=tuple(sorted(result.missing_keys)),
        loaded_embedding=any("embed" in key.lower() for key in state),
    )
    logger.info(
        "Warm-started %d draft tensors from %s (%s); missing=%s",
        report.loaded_keys,
        report.source,
        report.checkpoint_format,
        list(report.missing_keys),
    )
    return report


__all__ = [
    "WarmStartReport",
    "draft_config_dict",
    "load_draft_config_source",
    "resolve_draft_config",
    "warm_start_draft_model",
]
