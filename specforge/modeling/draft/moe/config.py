# coding=utf-8
"""MoE config: the draft-JSON keys, ``MoEConfig`` and the preset registry."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping, Optional

from .base import Registry

_MISSING = object()

#: Draft-JSON top-level key -> ``MoEConfig`` field (architecture keys).
ARCHITECTURE_KEYS: Dict[str, str] = {
    key: key
    for key in (
        "n_routed_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "n_shared_experts",
        "shared_expert_intermediate_size",
        "scoring_func",
        "norm_topk_prob",
        "routed_scaling_factor",
        "router",
        "balance",
        "experts_backend",
        "shared_expert",
    )
}
ARCHITECTURE_KEYS["moe_layout"] = "layout"

#: ``dflash_config`` key (training-only) -> ``MoEConfig`` field.
TRAINING_KEYS: Dict[str, str] = {
    "moe_aux_loss_coeff": "aux_loss_coeff",
    "moe_dispatch": "dispatch",
}

#: Fields a draft JSON must state explicitly (a preset may not default them).
PER_RUN_FIELDS = ("n_routed_experts", "num_experts_per_tok", "moe_intermediate_size")


def _positive_int(name: str, value: Any) -> int:
    """``value`` as an ``int`` > 0; integral floats pass, bools and strings do not."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class MoEConfig:
    """Resolved MoE configuration of one draft model (all layers share it)."""

    preset: str
    n_routed_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    n_shared_experts: int = 1
    shared_expert_intermediate_size: Optional[int] = None
    # Routing recipe; scoring_func, router and balance name registered ones.
    scoring_func: str = "softmax"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0
    router: str = "topk"
    balance: str = "none"
    #: Registered implementations of the routed and the shared experts.
    experts_backend: str = "grouped"
    shared_expert: str = "swiglu"
    #: Checkpoint file naming (a registered :mod:`.checkpoint` layout).
    layout: str = "native"
    # Training-only fields, never part of the checkpoint's architecture.
    aux_loss_coeff: float = 0.0
    dispatch: str = "sorted_loop"

    def __post_init__(self) -> None:
        for name in PER_RUN_FIELDS:
            object.__setattr__(self, name, _positive_int(name, getattr(self, name)))
        if not self.num_experts_per_tok <= self.n_routed_experts:
            raise ValueError(
                "num_experts_per_tok must be in [1, n_routed_experts], got "
                f"{self.num_experts_per_tok} with {self.n_routed_experts} experts"
            )
        if self.n_shared_experts not in (0, 1):
            raise ValueError(
                "n_shared_experts must be 0 or 1 (one shared expert of "
                "shared_expert_intermediate_size width), got "
                f"{self.n_shared_experts}"
            )
        if self.shared_expert_intermediate_size is None:
            object.__setattr__(
                self, "shared_expert_intermediate_size", self.moe_intermediate_size
            )
        object.__setattr__(
            self,
            "shared_expert_intermediate_size",
            _positive_int(
                "shared_expert_intermediate_size", self.shared_expert_intermediate_size
            ),
        )
        if self.aux_loss_coeff < 0:
            raise ValueError("aux_loss_coeff must be >= 0")


#: Preset name -> one target family's architecture defaults.
MOE_PRESETS: Registry[Dict[str, Any]] = Registry("MoE preset")

_FIELD_NAMES = {f.name for f in fields(MoEConfig)}


def register_moe_preset(name: str, **defaults: Any) -> None:
    """Register a family's architecture defaults (no per-run or training fields)."""
    forbidden = set(TRAINING_KEYS.values()) | set(PER_RUN_FIELDS) | {"preset"}
    bad = sorted(set(defaults) - _FIELD_NAMES)
    if bad:
        raise ValueError(f"preset {name!r} sets unknown MoEConfig fields: {bad}")
    bad = sorted(set(defaults) & forbidden)
    if bad:
        raise ValueError(f"preset {name!r} may not set per-run/training fields: {bad}")
    MOE_PRESETS.register(name, dict(defaults))


def _get(config: Any, key: str, default: Any = _MISSING) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def is_moe_config(config: Any) -> bool:
    """True when a draft config names a ``moe_preset`` (DESIGN.md: detection)."""
    return bool(_get(config, "moe_preset", None))


def requests_routed_experts(config: Any) -> bool:
    """True when ``n_routed_experts`` is set to anything but ``0``/``None``."""
    value = _get(config, "n_routed_experts", None)
    if value is None:
        return False
    return isinstance(value, bool) or not (
        isinstance(value, (int, float)) and value == 0
    )


def resolve_moe_config(config: Any) -> Optional[MoEConfig]:
    """Resolve a draft config (HF config or dict) to ``MoEConfig``; ``None`` if dense."""
    if not is_moe_config(config):
        return None
    name = _get(config, "moe_preset")
    values: Dict[str, Any] = dict(MOE_PRESETS.get(name))
    for key, field_name in ARCHITECTURE_KEYS.items():
        explicit = _get(config, key, _MISSING)
        if explicit is not _MISSING and explicit is not None:
            values[field_name] = explicit
    missing = [f for f in PER_RUN_FIELDS if f not in values]
    if missing:
        raise ValueError(f"MoE draft config (preset {name!r}) must set {missing}")
    dflash_config = _get(config, "dflash_config", None) or {}
    unknown = sorted(
        key
        for key in dflash_config
        if key.startswith("moe_") and key not in TRAINING_KEYS
    )
    if unknown:
        raise ValueError(
            f"unknown MoE training keys in dflash_config: {unknown}; "
            f"known: {sorted(TRAINING_KEYS)}"
        )
    for json_key, field_name in TRAINING_KEYS.items():
        if json_key in dflash_config:
            values[field_name] = dflash_config[json_key]
    cfg = MoEConfig(preset=name, **values)
    from .checkpoint import LAYOUTS  # local: checkpoint imports this module

    LAYOUTS.get(cfg.layout)  # unknown layouts fail at model build, not at save
    return cfg
