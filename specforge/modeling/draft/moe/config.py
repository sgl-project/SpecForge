# coding=utf-8
"""MoE architecture config: the knobs a draft JSON can set, and their presets.

Two kinds of keys, deliberately kept apart:

- **Architecture keys** live at the top level of the draft JSON under the
  target checkpoints' native HF names (``n_routed_experts``,
  ``moe_intermediate_size``, ``scoring_func``, ...). They determine which
  weights a checkpoint carries and how serving must route, so a draft JSON can
  be assembled by copying them from the target's ``config.json``. A
  ``moe_preset`` supplies the defaults for one target family; explicit keys
  override the preset for ablations.
- **Training-only keys** live under ``dflash_config`` with an ``moe_`` prefix
  (``moe_bias_update_rate``, ``moe_aux_loss_coeff``, ``moe_dispatch``). They
  never change the checkpoint and are invisible to serving.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping, Optional

from ._registry import Registry

_MISSING = object()

#: Draft-JSON top-level keys that describe the MoE architecture.
ARCHITECTURE_KEYS = (
    "n_routed_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "n_shared_experts",
    "shared_expert_intermediate_size",
    "scoring_func",
    "norm_topk_prob",
    "routed_scaling_factor",
    "n_group",
    "topk_group",
    "swiglu_limit",
    "router",
    "balance",
    "shared_expert",
    "shared_expert_gate",
    "experts_backend",
)

#: ``dflash_config`` keys (training-only) -> MoEConfig field.
TRAINING_KEYS = {
    "moe_bias_update_rate": "bias_update_rate",
    "moe_aux_loss_coeff": "aux_loss_coeff",
    "moe_dispatch": "dispatch",
    "moe_freeze_experts": "freeze_experts",
}


@dataclass(frozen=True)
class MoEConfig:
    """Resolved MoE configuration for one draft model (all layers share it)."""

    preset: str
    n_routed_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    n_shared_experts: int = 1
    shared_expert_intermediate_size: Optional[int] = None
    # Routing recipe. ``scoring_func`` names a registered score function;
    # ``router``/``balance`` name registered component implementations.
    scoring_func: str = "softmax"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0
    n_group: int = 1
    topk_group: int = 1
    router: str = "topk"
    balance: str = "none"
    # Expert MLPs.
    swiglu_limit: float = 0.0
    experts_backend: str = "grouped"
    shared_expert: str = "swiglu"
    shared_expert_gate: str = "none"
    # Training-only (never part of the checkpoint).
    bias_update_rate: float = 0.0
    aux_loss_coeff: float = 0.0
    dispatch: str = "sorted_loop"
    #: Keep the routed experts fixed (e.g. warm-started from the target) and
    #: train only the router, shared expert and the rest of the draft. Frozen
    #: experts are replicated by the FSDP backend instead of sharded, which
    #: removes the per-micro-batch weight all-gathers that dominate large MoEs.
    freeze_experts: bool = False

    def __post_init__(self) -> None:
        if self.n_routed_experts <= 0:
            raise ValueError("n_routed_experts must be positive for an MoE FFN")
        if not 0 < self.num_experts_per_tok <= self.n_routed_experts:
            raise ValueError(
                "num_experts_per_tok must be in [1, n_routed_experts], got "
                f"{self.num_experts_per_tok} with {self.n_routed_experts} experts"
            )
        if self.moe_intermediate_size <= 0:
            raise ValueError("moe_intermediate_size must be positive")
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
        if self.shared_expert_intermediate_size <= 0:
            raise ValueError("shared_expert_intermediate_size must be positive")
        if self.n_group <= 0 or self.n_routed_experts % self.n_group:
            raise ValueError(
                f"n_group={self.n_group} must divide n_routed_experts="
                f"{self.n_routed_experts}"
            )
        if not 0 < self.topk_group <= self.n_group:
            raise ValueError(
                f"topk_group={self.topk_group} must be in [1, n_group={self.n_group}]"
            )
        if self.swiglu_limit < 0:
            raise ValueError("swiglu_limit must be >= 0 (0 disables the clamp)")
        if self.bias_update_rate < 0 or self.aux_loss_coeff < 0:
            raise ValueError("bias_update_rate and aux_loss_coeff must be >= 0")

    @property
    def group_limited(self) -> bool:
        """Whether routing restricts top-k to ``topk_group`` of ``n_group``."""
        return self.topk_group < self.n_group

    def as_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def serving_fields(self) -> Dict[str, Any]:
        """The resolved recipe in the DeepSeek HF config vocabulary.

        Exports write these to ``config.json`` so a serving engine reads the
        complete routing recipe without knowing SpecForge presets. Only the
        keys a DeepSeek-style MoE reads; ``swiglu_limit`` is omitted when the
        clamp is off (a serving engine treats 0 as a clamp at 0).
        """
        out: Dict[str, Any] = {
            "n_routed_experts": self.n_routed_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "moe_intermediate_size": self.moe_intermediate_size,
            "n_shared_experts": self.n_shared_experts,
            "scoring_func": self.scoring_func,
            "norm_topk_prob": self.norm_topk_prob,
            "routed_scaling_factor": self.routed_scaling_factor,
            "n_group": self.n_group,
            "topk_group": self.topk_group,
            "topk_method": "noaux_tc" if self.balance == "noaux_tc" else "greedy",
        }
        if self.swiglu_limit > 0:
            out["swiglu_limit"] = self.swiglu_limit
        return out


#: preset name -> architecture defaults (a subset of ARCHITECTURE_KEYS).
MOE_PRESETS: Registry[Dict[str, Any]] = Registry("MoE preset")

_FIELD_NAMES = {f.name for f in fields(MoEConfig)}


def register_moe_preset(name: str, **defaults: Any) -> Dict[str, Any]:
    """Register the architecture defaults of one target family.

    A preset may set any ``MoEConfig`` field except the training-only ones and
    the per-run sizes (``n_routed_experts``, ``num_experts_per_tok``,
    ``moe_intermediate_size``), which the draft JSON must state explicitly.
    """
    forbidden = set(TRAINING_KEYS.values()) | {
        "preset",
        "n_routed_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
    }
    bad = sorted(set(defaults) - _FIELD_NAMES)
    if bad:
        raise ValueError(f"preset {name!r} sets unknown MoEConfig fields: {bad}")
    bad = sorted(set(defaults) & forbidden)
    if bad:
        raise ValueError(f"preset {name!r} may not set per-run/training fields: {bad}")
    MOE_PRESETS.register(name, dict(defaults))
    return defaults


def available_moe_presets() -> list[str]:
    return MOE_PRESETS.names()


def _get(config: Any, key: str, default: Any = _MISSING) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def is_moe_config(config: Any) -> bool:
    """True when a draft config asks for an MoE FFN (``n_routed_experts > 0``)."""
    value = _get(config, "n_routed_experts", 0)
    return int(value or 0) > 0


def resolve_moe_config(config: Any) -> Optional[MoEConfig]:
    """Resolve a draft config (HF ``PretrainedConfig`` or dict) to ``MoEConfig``.

    Returns ``None`` for dense drafts. For MoE drafts, ``moe_preset`` is
    required: it names the target family's routing recipe and is the only way
    the architecture keys get validated defaults.
    """
    if not is_moe_config(config):
        return None
    preset = _get(config, "moe_preset", None)
    if not preset:
        raise ValueError(
            "n_routed_experts > 0 requires moe_preset in the draft config; "
            f"available presets: {available_moe_presets() or '<none registered>'}"
        )
    values: Dict[str, Any] = dict(MOE_PRESETS.get(preset))
    for key in ARCHITECTURE_KEYS:
        explicit = _get(config, key, _MISSING)
        if explicit is not _MISSING and explicit is not None:
            values[key] = explicit
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
    return MoEConfig(preset=preset, **values)
