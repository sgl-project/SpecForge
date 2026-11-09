"""Cross-node constraints; no model construction or algorithm dispatch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from ._validation import nonempty_string, positive_int
from .attention import AttentionSpec
from .conv import ConvSpec
from .layer import KIND_BY_REFERENCE_FIELD, LayerSpec
from .logits import TargetLogitSpec
from .mlp import FFNSpec
from .registry import (
    architecture_spec,
    component_schema,
    component_type,
    validate_component,
)

if TYPE_CHECKING:
    from .model import DraftConfig


def validate(config: DraftConfig) -> None:
    """Validate the whole structure without importing or allocating model code."""
    registration = architecture_spec(config.architecture)
    if type(config.spec_config) is not registration.schema:
        raise ValueError(
            f"spec_config for {config.architecture} must be {registration.schema.__name__}"
        )
    protocol = config.spec_config.protocol
    validate_layers(
        hidden_size=config.hidden_size,
        attention_configs=config.attention_configs,
        mlp_configs=config.mlp_configs,
        conv_configs=config.conv_configs,
        layer_configs=config.layer_configs,
        layer_types=config.layer_types,
        num_model_slots=None if protocol is None else protocol.num_model_slots,
    )
    for kind, node in (
        ("norm", config.norm_config),
        ("residual", config.residual_config),
    ):
        if type(node) is not component_schema(kind, getattr(node, "type", None)):
            raise ValueError(f"{kind}_config must use its registered schema")
    for path, node in (
        ("rope_parameters", config.rope_parameters),
        *(
            (f"attention_configs.{name}.rope_parameters", attention.rope_parameters)
            for name, attention in config.attention_configs.items()
            if attention.rope_parameters is not None
        ),
    ):
        try:
            schema = component_schema("rope", component_type("rope", node))
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
        if type(node) is not schema:
            raise ValueError(f"{path} must use its registered schema")
    for name, attention in config.attention_configs.items():
        if attention.rotary_input_dim is None:
            if attention.rope_parameters is not None:
                raise ValueError(
                    f"attention_configs.{name}: rope_parameters has no effect on "
                    "an attention kind without a rotary subspace"
                )
        else:
            try:
                attention.effective_rope(config.rope_parameters).rotary_dim(
                    attention.rotary_input_dim
                )
            except ValueError as exc:
                raise ValueError(f"attention_configs.{name}: {exc}") from exc
    if config.draft_vocab_size > config.vocab_size:
        raise ValueError("draft_vocab_size must not exceed vocab_size")
    if config.tie_word_embeddings:
        if config.target_bindings:
            raise ValueError(
                "tie_word_embeddings cannot be combined with target_bindings"
            )
        if config.draft_vocab_size != config.vocab_size:
            raise ValueError("tie_word_embeddings requires equal vocabulary sizes")
    if "lm_head" in config.target_bindings:
        node = config.target_logit_transform
        if not isinstance(node, TargetLogitSpec):
            raise ValueError(
                "bound lm_head requires an explicit supported target_logit_transform"
            )
        if type(node) is not component_schema("logits", getattr(node, "type", None)):
            raise ValueError("target_logit_transform must use its registered schema")
    elif config.target_logit_transform is not None:
        raise ValueError("target_logit_transform requires lm_head in target_bindings")
    registration.validate(config)


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
    # layers first: an unused layer explains the components only it references
    tables = {
        "layer": layer_configs,
        "attention": attention_configs,
        "mlp": mlp_configs,
        "conv": conv_configs,
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
        for field, kind in KIND_BY_REFERENCE_FIELD.items():
            reference = getattr(layer, field)
            if reference is None:
                continue
            node = require(kind, reference, f"{path}.{field}")
            try:
                validate_component(
                    kind,
                    node,
                    hidden_size=hidden_size,
                    num_model_slots=num_model_slots,
                )
            except ValueError as exc:
                raise ValueError(f"{path}.{field}: {exc}") from exc

    unused = [
        f"{kind}_configs: unused definitions {sorted(table.keys() - used[kind])}"
        for kind, table in tables.items()
        if table.keys() - used[kind]
    ]
    if unused:
        raise ValueError("; ".join(unused))
