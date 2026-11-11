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
        "lm_head_multiplier": 1,  # Falcon-H1 multiplies the head output
    }
    for name, neutral in neutral_values.items():
        value = target.get(name)
        if value is None:
            continue
        finite_number(f"target_config.{name}", value)
        if value != neutral:
            raise ValueError(f"unsupported target logit setting {name}={value!r}")
    if target.get("norm_head") not in (None, False):
        # SGLang normalizes a Bailing head's weight at load
        raise ValueError(
            f"unsupported target logit setting norm_head={target['norm_head']!r}"
        )
    base = target.get("dim_model_base")
    if base is not None and base != target.get("hidden_size"):
        # MiniCPM divides hidden states by hidden_size / dim_model_base before the head
        raise ValueError(f"unsupported target logit setting dim_model_base={base!r}")
    multiplier = target.get("logits_mup_width_multiplier")
    if multiplier is None:
        return NoLogitTransformSpec()
    return MuPLogitTransformSpec(multiplier=multiplier)


def bound_vocab_size(target: Mapping) -> int:
    """Rows of the target's embedding and LM head: padded when the target pads."""
    value = target.get("padded_vocab_size") or target.get("vocab_size")
    positive_int("target_config.vocab_size", value)
    return value


def validate_target(config: DraftConfig, target: Mapping) -> None:
    """Recheck saved semantics and binding dimensions against the supplied target."""
    target = _text_config(target)
    if config.target_bindings:
        hidden_size = target.get("hidden_size")
        positive_int("target_config.hidden_size", hidden_size)
        if hidden_size != config.hidden_size:
            raise ValueError("bound target hidden_size must match draft hidden_size")
        if bound_vocab_size(target) != config.vocab_size:
            raise ValueError("bound target vocab_size must match draft vocab_size")
    else:
        target_vocab_size = target.get("vocab_size")
        positive_int("target_config.vocab_size", target_vocab_size)
        if config.vocab_size < target_vocab_size:
            raise ValueError("draft vocab_size must cover target vocab_size")
    if "lm_head" in config.target_bindings:
        if config.target_logit_transform != resolve_target_logits(target):
            raise ValueError("target_logit_transform does not match target_config")
    config.spec_config.validate_target(target)
