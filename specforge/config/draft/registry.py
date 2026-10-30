"""Configuration types keyed by draft architecture, independent of models."""

from __future__ import annotations

from dataclasses import dataclass

from transformers import PretrainedConfig, Qwen3Config

from .base import DraftConfig
from .dflash import DFlashConfig
from .dflash2 import DFlash2Config
from .domino import DominoConfig
from .dspark import DSparkConfig
from .dspine import DSpineDraftConfig
from .eagle3 import Eagle3Config, EaglePretrainedConfig
from .mtp import Qwen3_5MTPConfig
from .peagle import PEagleConfig


@dataclass(frozen=True)
class DraftConfigRegistration:
    """HF serialization type and optional typed architecture options."""

    config_class: type[PretrainedConfig]
    options_class: type[DraftConfig] | None = None


DRAFT_CONFIG_REGISTRY: dict[str, DraftConfigRegistration] = {}


def register_draft_config(
    name: str,
    config_class: type[PretrainedConfig],
    options_class: type[DraftConfig] | None = None,
) -> None:
    """Register settings without importing or allocating their model.

    Repeating the same registration is harmless; conflicting declarations fail
    before changing the registry. Model registration uses this same check.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("draft architecture name must be a non-empty string")
    if not isinstance(config_class, type) or not issubclass(
        config_class, PretrainedConfig
    ):
        raise TypeError("config_class must be a PretrainedConfig subclass")
    if options_class is not None and (
        not isinstance(options_class, type)
        or not issubclass(options_class, DraftConfig)
    ):
        raise TypeError("options_class must be a DraftConfig subclass or None")
    registration = DraftConfigRegistration(config_class, options_class)
    previous = DRAFT_CONFIG_REGISTRY.get(name)
    if previous is not None and previous != registration:
        raise ValueError(
            f"draft configuration {name!r} already has a different registration"
        )
    DRAFT_CONFIG_REGISTRY[name] = registration


def resolve_draft_config_type(name: str) -> DraftConfigRegistration:
    try:
        return DRAFT_CONFIG_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"draft architecture {name!r} is not registered; "
            f"available: {sorted(DRAFT_CONFIG_REGISTRY)}"
        ) from None


# This catalog is deliberately independent of the model registry. Configuration
# validation is useful on hosts that do not load the training/model stack.
for _name, _config, _options in (
    ("LlamaForCausalLMEagle3", EaglePretrainedConfig, Eagle3Config),
    ("PEagleDraftModel", EaglePretrainedConfig, PEagleConfig),
    ("DFlashDraftModel", Qwen3Config, DFlashConfig),
    ("DFlash2DraftModel", Qwen3Config, DFlash2Config),
    ("DominoDraftModel", Qwen3Config, DominoConfig),
    ("DSparkDraftModel", Qwen3Config, DSparkConfig),
    ("DSpineDraftModel", Qwen3Config, DSpineDraftConfig),
    ("Qwen3_5MTPDraftModel", Qwen3Config, Qwen3_5MTPConfig),
):
    register_draft_config(_name, _config, _options)
del _name, _config, _options
