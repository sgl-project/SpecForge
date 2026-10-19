# coding=utf-8
"""The checkpoint-file boundary of MoE drafts.

Checkpoint files (trainer checkpoints, warm-start sources, exports) name an
MoE layer's tensors in each target family's own way, while modules keep a
native layout (stacked ``[E, out, in]`` experts). A layout, registered by name
and chosen by the preset (``moe_layout`` overrides it), maps one MoE layer at
a time in both directions. Exports also fold router centering into the gate.
See ``DESIGN.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from torch import nn

from .base import Registry
from .config import MoEConfig, resolve_moe_config
from .layer import iter_moe_layers
from .router import fold_router_centering, fold_router_centering_config

StateDict = Dict[str, Any]
#: One MoE layer's tensors (layer-relative keys) -> the same layer, other naming.
LayerTransform = Callable[[StateDict], StateDict]


# -- layouts: module naming <-> file naming -------------------------------------


@dataclass(frozen=True)
class CheckpointLayout:
    """How one family names an MoE layer's tensors in checkpoint files.

    ``to_file`` and ``from_file`` map one layer between module naming and file
    naming; ``from_file(to_file(state))`` must give back ``state``. Both keep
    the layer's ``experts`` component (the boundary finds layers by it).
    """

    to_file: LayerTransform
    from_file: LayerTransform


LAYOUTS: Registry[CheckpointLayout] = Registry("MoE checkpoint layout")


def register_layout(
    name: str, *, to_file: LayerTransform, from_file: LayerTransform
) -> None:
    LAYOUTS.register(name, CheckpointLayout(to_file, from_file))


def _same_names(state: StateDict) -> StateDict:
    return dict(state)


# Files keep the module's own names: the default until a preset names a family.
register_layout("native", to_file=_same_names, from_file=_same_names)


def moe_config_for(owner: Any) -> Optional[MoEConfig]:
    """The MoE config of a draft model (any module tree) or a draft config."""
    if isinstance(owner, nn.Module):
        layer = next(iter_moe_layers(owner), None)
        return None if layer is None else layer.cfg
    return resolve_moe_config(owner)


def layout_for(owner: Any) -> Optional[CheckpointLayout]:
    """The layout ``owner`` selects; ``None`` for drafts without MoE layers,
    which every boundary function passes through unchanged."""
    cfg = moe_config_for(owner)
    return None if cfg is None else LAYOUTS.get(cfg.layout)


#: An MoE layer's prefix is everything before its ``experts`` component.
_LAYER_PREFIX = re.compile(r"^(?P<prefix>(?:.*\.)?)experts\.")


def _split_moe_layers(
    state: Mapping[str, Any],
) -> Tuple[Dict[str, StateDict], StateDict]:
    """``({layer prefix: layer-relative tensors}, every other tensor)``."""
    prefixes = sorted(
        {m["prefix"] for key in state if (m := _LAYER_PREFIX.match(key))},
        key=len,
        reverse=True,
    )
    layers: Dict[str, StateDict] = {prefix: {} for prefix in prefixes}
    rest: StateDict = {}
    for key, value in state.items():
        prefix = next((p for p in prefixes if key.startswith(p)), None)
        if prefix is None:
            rest[key] = value
        else:
            layers[prefix][key[len(prefix) :]] = value
    return layers, rest


def _per_layer(state: Mapping[str, Any], transform: LayerTransform) -> StateDict:
    layers, out = _split_moe_layers(state)
    for prefix, layer_state in layers.items():
        out.update({prefix + k: v for k, v in transform(layer_state).items()})
    return out


def to_checkpoint_state_dict(state: Mapping[str, Any], owner: Any) -> StateDict:
    """Module naming -> the file naming of ``owner``'s layout (identity if dense)."""
    layout = layout_for(owner)
    return dict(state) if layout is None else _per_layer(state, layout.to_file)


def from_checkpoint_state_dict(state: Mapping[str, Any], owner: Any) -> StateDict:
    """The file naming of ``owner``'s layout -> module naming (identity if dense)."""
    layout = layout_for(owner)
    return dict(state) if layout is None else _per_layer(state, layout.from_file)


# -- exports: fold router centering, then name for serving ----------------------


def to_serving_state_dict(state: Mapping[str, Any], owner: Any) -> StateDict:
    """Module naming -> a servable file: router centering folded into the
    gate, then the layout's naming."""
    layout = layout_for(owner)
    if layout is None:
        return dict(state)
    return _per_layer(state, lambda layer: layout.to_file(fold_router_centering(layer)))


def exported_config_updates(owner: Any) -> Dict[str, Any]:
    """``config.json`` changes for an export of ``owner``'s draft: the folded
    router centering (training keys under ``"dflash_config"``)."""
    cfg = moe_config_for(owner)
    if cfg is None:
        return {}
    return fold_router_centering_config(cfg)


# -- warm start ------------------------------------------------------------------


def select_target_experts(
    n_target: int, n_draft: int, strategy: str = "strided"
) -> Tuple[int, ...]:
    """Pick ``n_draft`` distinct target experts.

    ``"strided"`` spreads picks evenly over the target's expert ids (no
    assumption about which experts matter for the draft's data); ``"first"``
    takes the leading ids.
    """
    if n_draft <= 0 or n_target <= 0:
        raise ValueError("expert counts must be positive")
    if n_draft > n_target:
        raise ValueError(
            f"cannot seed {n_draft} draft experts from {n_target} target experts"
        )
    if strategy == "first":
        return tuple(range(n_draft))
    if strategy == "strided":
        return tuple((i * n_target) // n_draft for i in range(n_draft))
    raise ValueError(f"unknown warm-start selection strategy {strategy!r}")
