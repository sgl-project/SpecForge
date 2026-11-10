"""Decode P-EAGLE's effective Llama settings without importing Transformers."""

from collections.abc import Mapping

from ...architecture import PEagleSpec
from ...model import DraftConfig
from ...rope import DefaultRoPESpec
from ..target import validate_target
from .auxiliary import _FIELDS, read_auxiliary
from .context import _check_fields
from .rules import resolve_alias

_ROPE_FIELDS = {
    "type",
    "rope_type",
    "rope_theta",
    "factor",
    "partial_rotary_factor",
    "original_max_position_embeddings",
    "low_freq_factor",
    "high_freq_factor",
    "attention_factor",
    "beta_fast",
    "beta_slow",
    "mscale",
    "mscale_all_dim",
    "truncate",
    "mrope_section",
    "mrope_interleaved",
    "short_factor",
    "long_factor",
}


def _rope(payload):
    sources = []
    for name in ("rope_parameters", "rope_scaling"):
        value = payload.get(name)
        if value is not None:
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} must be an object or null")
            _check_fields(value, _ROPE_FIELDS, name)
            sources.append(value)
    # PEagleDraftModel reads theta but constructs unscaled, full-dimensional
    # LlamaRotaryEmbedding. Scaling fields were accepted but never executed.
    theta = resolve_alias("rope_theta", *sources, payload, default=10000.0)
    return DefaultRoPESpec(
        interleaved=False, rope_theta=theta, partial_rotary_factor=1.0
    )


def decode_peagle(
    payload,
    *,
    target_config,
    mask_token_id,
    capture_layers=None,
    norm_before_residual=None,
):
    _check_fields(payload, _FIELDS | {"norm_before_residual"}, "legacy P-EAGLE")
    if mask_token_id is None:
        raise ValueError(
            "legacy P-EAGLE requires mask_token_id from peagle_mask_token_id "
            "in the checkpoint resume contract"
        )
    fields, features = read_auxiliary(
        payload, target_config, capture_layers=capture_layers
    )
    recorded = (
        {}
        if norm_before_residual is None
        else {"norm_before_residual": norm_before_residual}
    )
    norm_before = resolve_alias(
        "norm_before_residual", payload, recorded, default=False, ignore_none=True
    )
    config = DraftConfig(
        architecture="PEagleDraftModel",
        rope_parameters=_rope(payload),
        spec_config=PEagleSpec(
            target_features=features,
            norm_before_residual=norm_before,
            mask_token_id=mask_token_id,
            pad_token_id=payload.get("pad_token_id"),
        ),
        **fields,
    )
    validate_target(config, target_config)
    return config
