"""Separate legacy DSpine injection structure from saved training options."""

from collections.abc import Mapping

from ..._validation import boolean, finite_number, positive_int
from ...architecture import DSpineSpec
from ...architecture.block import AnchoredBlockSpec
from ...architecture.dspine import AdjacentInjectionSpec
from ...model import DraftConfig
from .context import _COMMON_FIELDS, _TOP_FIELDS, _check_fields, read_context

_INJECTION_DEFAULTS = {"transfer_rank": 1024, "message_size": 512}
_TRAINING_DEFAULTS = {
    "top_k": 16,
    "ce_weight": 0.1,
    "alignment_weight": 0.5,
    "alignment_warmup_steps": 146,
    "replacement_layers": 3,
    "replacement_probability": 0.5,
    "replacement_hold_ratio": 1 / 6,
    "replacement_end_ratio": 1 / 3,
    "candidate_mass_threshold": 1e-4,
}


def _options(payload):
    if not isinstance(payload, Mapping) or payload.get("architectures") not in (
        ["DSpineDraftModel"],
        ("DSpineDraftModel",),
    ):
        raise ValueError("expected a legacy DSpineDraftModel configuration")
    _check_fields(payload, _TOP_FIELDS | {"dspine_config"}, "legacy DSpine")
    options = payload.get("dspine_config")
    if options is None:
        options = {}
    if not isinstance(options, Mapping):
        raise ValueError("dspine_config must be an object or null")
    _check_fields(
        options, set(_INJECTION_DEFAULTS) | set(_TRAINING_DEFAULTS), "dspine_config"
    )
    return options


def extract_dspine_training_settings(payload: Mapping) -> dict:
    """Return a fresh mapping for the future training/checkpoint migration path.

    These values are intentionally excluded from DraftConfig and canonical
    structure. This helper performs no training orchestration or state loading.
    """
    options = _options(payload)
    values = {
        name: options.get(name, default) for name, default in _TRAINING_DEFAULTS.items()
    }
    vocab_size = payload.get("vocab_size")
    positive_int("vocab_size", vocab_size)
    for name in ("top_k", "alignment_warmup_steps", "replacement_layers"):
        if type(values[name]) is not int:
            raise ValueError(f"{name} must be an integer")
    for name, value in values.items():
        finite_number(name, value)
    if not 1 <= values["top_k"] <= vocab_size:
        raise ValueError("top_k must be in [1, vocab_size]")
    for name in ("alignment_warmup_steps", "replacement_layers", "alignment_weight"):
        if values[name] < 0:
            raise ValueError(f"{name} must be nonnegative")
    for name in ("ce_weight", "replacement_probability"):
        if not 0 <= values[name] <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if not 0 <= values["replacement_hold_ratio"] < values["replacement_end_ratio"] <= 1:
        raise ValueError("replacement ratios must satisfy 0 <= hold < end <= 1")
    if not 0 <= values["candidate_mass_threshold"] < 1:
        raise ValueError("candidate_mass_threshold must be in [0, 1)")
    return values


def decode_dspine(payload, *, target_config):
    options = _options(payload)
    extract_dspine_training_settings(payload)  # Validate without mixing ownership.
    causal = payload.get("is_causal")
    if causal is not None:
        boolean("is_causal", causal)
    if causal is False:
        raise ValueError("DSpine requires causal attention")
    method = payload.get("dflash_config")
    if method is None:
        method = {}
    if not isinstance(method, Mapping):
        raise ValueError("dflash_config must be an object or null")
    _check_fields(method, _COMMON_FIELDS, "dflash_config")
    if method.get("projector_type") is not None:
        raise ValueError("DSpine does not support a projector_type")
    fields, features, block_size = read_context(
        {**payload, "is_causal": True}, method, target_config
    )
    config = DraftConfig(
        architecture="DSpineDraftModel",
        spec_config=DSpineSpec(
            block=AnchoredBlockSpec(block_size=block_size),
            target_features=features,
            injection=AdjacentInjectionSpec(
                **{
                    name: options.get(name, default)
                    for name, default in _INJECTION_DEFAULTS.items()
                }
            ),
        ),
        **fields,
    )
    return config
