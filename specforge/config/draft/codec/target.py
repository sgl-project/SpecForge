"""Resolve target-derived values from an explicit, already-unwrapped mapping."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from .._validation import finite_number, positive_int
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
    config.spec_config.validate_target(target)
