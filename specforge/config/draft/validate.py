"""Cross-node constraints; no model construction or algorithm dispatch."""

from collections.abc import Mapping

from ._validation import nonempty_string, positive_int
from .attention import AttentionSpec
from .conv import ConvSpec
from .layer import LayerSpec
from .mlp import FFNSpec
from .registry import component_schema, validate_component


def validate_layers(
    *,
    hidden_size: int,
    attention_configs: Mapping[str, AttentionSpec],
    mlp_configs: Mapping[str, FFNSpec],
    conv_configs: Mapping[str, ConvSpec],
    layer_configs: Mapping[str, LayerSpec],
    layer_types: tuple[str, ...],
    num_model_slots: int | None,
) -> None:
    """Validate named references and shared dimensions before allocating weights.

    Repeated layer names select the same structure, not shared parameters. The
    architecture supplies its slot count; this module never interprets block_size.
    """
    positive_int("hidden_size", hidden_size)
    if num_model_slots is not None:
        positive_int("num_model_slots", num_model_slots)
    if type(layer_types) is not tuple or not layer_types:
        raise ValueError(
            "layer_types must be a nonempty tuple of layer definition names"
        )
    tables = {
        "attention": attention_configs,
        "mlp": mlp_configs,
        "conv": conv_configs,
        "layer": layer_configs,
    }
    for kind, table in tables.items():
        if not isinstance(table, Mapping):
            raise ValueError(f"{kind}_configs must be a mapping")
        for name, node in table.items():
            nonempty_string(f"{kind}_configs definition name", name)
            schema = (
                LayerSpec
                if kind == "layer"
                else component_schema(kind, getattr(node, "type", None))
            )
            if type(node) is not schema:
                raise ValueError(f"{kind}_configs.{name}: expected {schema.__name__}")

    used = {kind: set() for kind in tables}

    def require(kind: str, name: str, path: str):
        nonempty_string(path, name)
        if name not in tables[kind]:
            raise ValueError(f"{path}: unknown {kind} definition {name!r}")
        used[kind].add(name)
        return tables[kind][name]

    for index, name in enumerate(layer_types):
        layer = require("layer", name, f"layer_types[{index}]")
        path = f"layer_configs.{name}"
        require("attention", layer.attention, f"{path}.attention")
        require("mlp", layer.mlp, f"{path}.mlp")
        if layer.sublayer_conv is not None:
            conv = require("conv", layer.sublayer_conv, f"{path}.sublayer_conv")
            try:
                validate_component(
                    "conv",
                    conv,
                    hidden_size=hidden_size,
                    num_model_slots=num_model_slots,
                )
            except ValueError as exc:
                raise ValueError(f"{path}.sublayer_conv: {exc}") from exc

    for kind, table in tables.items():
        unused = table.keys() - used[kind]
        if unused:
            raise ValueError(f"{kind}_configs: unused definitions {sorted(unused)}")
