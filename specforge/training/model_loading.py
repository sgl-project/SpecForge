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
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple

if TYPE_CHECKING:
    from specforge.algorithms.common.providers import DraftConfigProvider
    from specforge.algorithms.registry import AlgorithmRegistration
    from specforge.config import Config
    from specforge.config.draft import DraftModelConfig

logger = logging.getLogger(__name__)

_CONFIG_FILE = "config.json"
_STATE_FILE = "training_state.pt"


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


def resolve_draft_config(
    cfg: "Config", *, provider: "DraftConfigProvider"
) -> "DraftModelConfig":
    """Resolve and validate the draft architecture for one typed run config."""

    from specforge.config.draft import AutoDraftConfig

    payload = cfg.model.draft_model_config
    if not isinstance(payload, dict) or not payload:
        raise ValueError(
            "model.draft_model_config must be a non-empty inline mapping; "
            "checkpoint and target configs never supply draft settings"
        )
    draft_config = AutoDraftConfig.from_inline(
        {**payload, "_attn_implementation": cfg.training.attention_backend}
    )

    expected = provider.architecture
    compatible = provider.compatible_architectures or frozenset({expected})
    architectures = list(getattr(draft_config, "architectures", None) or [])
    if len(architectures) != 1 or architectures[0] not in compatible:
        raise ValueError(
            f"training.strategy={cfg.training.strategy!r} requires draft "
            f"architecture in {sorted(compatible)!r}, got {architectures!r}"
        )
    AutoDraftConfig.validate_training_config(draft_config)
    return draft_config


def preflight_draft_kernels(
    cfg: "Config", *, algorithm: "AlgorithmRegistration"
) -> None:
    """Import the Liger kernels this host's trainer draft build will resolve.

    A supervisor about to start services or workers, and a disaggregated
    consumer about to wait on its producer, call this so a missing install
    fails at startup. The trainer's ``build_draft`` still owns the resolution.
    """

    if (
        cfg.model.use_liger_kernel is False
        or (
            cfg.model.use_liger_kernel is None
            and cfg.training.strategy in ("domino", "dspark", "dspine")
        )
        or not algorithm.spec.capabilities.supports_liger_kernel
    ):
        return
    from specforge.modeling.draft.models.layers.kernels import (
        load_liger_dflash_kernels,
        resolve_draft_liger_kernel_choice,
    )

    draft_config = resolve_draft_config(
        cfg, provider=algorithm.providers.model.draft_config
    )
    choice = resolve_draft_liger_kernel_choice(cfg.model.use_liger_kernel, draft_config)
    if choice.enabled:
        load_liger_dflash_kernels(choice)


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
    state_path: str, *, expected_strategy: str, draft_config: "DraftModelConfig"
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
    from specforge.algorithms.common.providers import MODEL_PROVENANCE_CONTRACT_KEY
    from specforge.config.draft import AutoDraftConfig
    from specforge.config.draft.contract import (
        draft_config_contract,
        validate_draft_config_contract,
    )

    provenance = dict(state.get(MODEL_PROVENANCE_CONTRACT_KEY) or {})
    saved = provenance.get("draft_settings")
    if saved is None:
        # Older checkpoints may keep a sibling config.json. Read it only for
        # validation, never as an initialization source.
        source = os.path.join(os.path.dirname(state_path), _CONFIG_FILE)
        if not os.path.isfile(source):
            raise ValueError(
                f"checkpoint {state_path!r} has no saved draft settings; "
                "provide its original config.json next to training_state.pt"
            )
        saved = draft_config_contract(AutoDraftConfig.from_pretrained(source))
    validate_draft_config_contract(
        draft_config_contract(draft_config), saved, source=state_path
    )
    return state["draft_state_dict"]


def _load_pretrained_draft_state(
    source: str,
    *,
    draft_config: "DraftModelConfig",
    cache_dir: Optional[str],
    trust_remote_code: bool,
) -> Dict[str, Any]:
    from specforge.config.draft import AutoDraftConfig
    from specforge.config.draft.contract import (
        draft_config_contract,
        validate_draft_config_contract,
    )
    from specforge.modeling.auto import AutoDraftModel

    saved = AutoDraftConfig.from_pretrained(
        source, cache_dir=cache_dir, trust_remote_code=trust_remote_code
    )
    validate_draft_config_contract(
        draft_config_contract(draft_config), draft_config_contract(saved), source=source
    )
    loaded, loading_info = AutoDraftModel.from_pretrained(
        source,
        config=draft_config,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        output_loading_info=True,
    )
    unexpected = loading_info.get("unexpected_keys") or []
    if unexpected:
        raise ValueError(
            f"warm-start checkpoint {source!r} has unexpected draft weights: {unexpected}"
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
    draft_config: "DraftModelConfig",
    strategy: str,
    allow_missing_embedding: bool = False,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = False,
) -> WarmStartReport:
    """Load only draft weights, never optimizer/counters/RNG training state."""

    runtime_state = _runtime_state_file(source)
    if runtime_state is not None:
        checkpoint_format: Literal["specforge", "pretrained"] = "specforge"
        state = _load_specforge_draft_state(
            runtime_state, expected_strategy=strategy, draft_config=draft_config
        )
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
    "preflight_draft_kernels",
    "resolve_draft_config",
    "warm_start_draft_model",
]
