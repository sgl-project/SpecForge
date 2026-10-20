# coding=utf-8
"""The checkpoint-file boundary: each family's file naming of MoE tensors."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from torch import nn

from .base import Registry
from .config import MoEConfig, resolve_moe_config
from .layer import iter_moe_layers

StateDict = Dict[str, Any]
#: One MoE layer's tensors (layer-relative keys) -> the same layer, other naming.
LayerTransform = Callable[[StateDict], StateDict]


@dataclass(frozen=True)
class CheckpointLayout:
    """One family's file naming of an MoE layer; ``from_file(to_file(s)) == s``."""

    # Both directions keep the layer's ``experts.*`` keys: layers are found by them.
    to_file: LayerTransform
    from_file: LayerTransform


LAYOUTS: Registry[CheckpointLayout] = Registry("MoE checkpoint layout")


def register_layout(
    name: str, *, to_file: LayerTransform, from_file: LayerTransform
) -> None:
    LAYOUTS.register(name, CheckpointLayout(to_file, from_file))


def _same_names(state: StateDict) -> StateDict:
    return dict(state)


# Files keep the module's own names until a preset names a family.
register_layout("native", to_file=_same_names, from_file=_same_names)


def moe_config_for(owner: Any) -> Optional[MoEConfig]:
    """The MoE config of a draft model (any module tree) or of a draft config."""
    if isinstance(owner, nn.Module):
        layer = next(iter_moe_layers(owner), None)
        return None if layer is None else layer.cfg
    return resolve_moe_config(owner)


def layout_for(owner: Any) -> Optional[CheckpointLayout]:
    """The layout ``owner`` selects; ``None`` for dense drafts, passed through."""
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
