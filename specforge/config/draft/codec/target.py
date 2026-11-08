"""Resolve target-derived values from an explicit, already-unwrapped mapping."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from .._validation import finite_number, positive_int
from ..architecture.eagle_fusion import AuxiliaryFeatureSpec
from ..architecture.features import TargetFeatureSpec
from ..logits import MuPLogitTransformSpec, NoLogitTransformSpec, TargetLogitSpec

if TYPE_CHECKING:
    from ..model import DraftConfig


def _text_config(target: Mapping) -> Mapping:
    if not isinstance(target, Mapping):
        raise ValueError("target_config must be an explicit text configuration mapping")
    if "text_config" in target:
        raise ValueError("target_config must be unwrapped by the checkpoint adapter")
    return target


def resolve_target_logits(target: Mapping) -> TargetLogitSpec:
    """Closed table of known logit settings; no target-family construction.

    muP means folding division into the loaded head weight in training dtype.
    Other known transforms are accepted only when they are neutral. Unknown
    target fields are not interpreted; bundled targets need a logits oracle.
    """
    target = _text_config(target)
    neutral_values = {
        "final_logit_softcapping": None,
        "logit_scale": 1,
        "logits_scale": 1,
        "logits_scaling": 1,
    }
    for name, neutral in neutral_values.items():
        value = target.get(name)
        if value is None:
            continue
        finite_number(f"target_config.{name}", value)
        if value != neutral:
            raise ValueError(f"unsupported target logit setting {name}={value!r}")
    multiplier = target.get("logits_mup_width_multiplier")
    if multiplier is None:
        return NoLogitTransformSpec()
    return MuPLogitTransformSpec(multiplier=multiplier)


def default_target_layer_ids(
    num_target_layers: int, num_draft_layers: int
) -> tuple[int, ...]:
    """Historical DFlash feature selection, resolved before node construction."""
    positive_int("num_target_layers", num_target_layers)
    positive_int("num_draft_layers", num_draft_layers)
    if num_draft_layers == 1:
        return (num_target_layers // 2,)
    layers = tuple(
        int(round(1 + index * (num_target_layers - 4) / (num_draft_layers - 1)))
        for index in range(num_draft_layers)
    )
    if any(layer < 0 or layer >= num_target_layers for layer in layers):
        raise ValueError(
            "default target_layer_ids are out of range; supply explicit layers"
        )
    return layers


def validate_target(config: DraftConfig, target: Mapping) -> None:
    """Recheck saved semantics and binding dimensions against the supplied target."""
    target = _text_config(target)
    if config.target_bindings:
        for name in ("hidden_size", "vocab_size"):
            value = target.get(name)
            positive_int(f"target_config.{name}", value)
            if value != getattr(config, name):
                raise ValueError(f"bound target {name} must match draft {name}")
    if "lm_head" in config.target_bindings:
        if config.target_logit_transform != resolve_target_logits(target):
            raise ValueError("target_logit_transform does not match target_config")
    features = getattr(config.spec_config, "target_features", None)
    if isinstance(features, (TargetFeatureSpec, AuxiliaryFeatureSpec)):
        count = target.get("num_hidden_layers")
        positive_int("target_config.num_hidden_layers", count)
        if any(layer >= count for layer in features.target_layer_ids):
            raise ValueError("target_layer_ids exceed target num_hidden_layers")
    if isinstance(features, AuxiliaryFeatureSpec):
        width = target.get("hidden_size")
        positive_int("target_config.hidden_size", width)
        if features.hidden_size != width:
            raise ValueError(
                "target_features.hidden_size must match target hidden_size"
            )
