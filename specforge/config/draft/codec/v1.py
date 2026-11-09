"""Canonical structure v1 and homogeneous YAML shorthand; no file I/O.

HF/runtime and artifact metadata are handled by their adapters, not stored in
DraftConfig. This codec accepts resolved structural settings only.
"""

from collections.abc import Mapping
from typing import get_type_hints

from .._validation import positive_int
from ..architecture.features import TargetFeatureSpec
from ..model import DraftConfig
from ..registry import architecture_spec
from ._schema import _construct
from .components import decode_component, decode_layer, encode_component, encode_layer
from .nodes import decode_node, encode_node
from .target import (
    _text_config,
    default_target_layer_ids,
    resolve_target_logits,
    validate_target,
)

SCHEMA_VERSION = 1
MODEL_TYPE = "specforge_draft"
_GROUPS = {
    "attention_configs": "attention",
    "mlp_configs": "mlp",
    "conv_configs": "conv",
}
_DIMENSIONS = (
    "hidden_size",
    "vocab_size",
    "draft_vocab_size",
    "max_position_embeddings",
    "initializer_range",
    "tie_word_embeddings",
)


def _array(value, path):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path}: expected an array")
    return tuple(value)


def _map(value, path, decode):
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected an object of named definitions")
    result = {}
    for name, node in value.items():
        try:
            result[name] = decode(node)
        except ValueError as exc:
            raise ValueError(f"{path}.{name}: {exc}") from exc
    return result


def decode_config(
    payload: Mapping, *, target_config: Mapping | None = None
) -> DraftConfig:
    """Decode canonical structure, checking redundant depth against layer_types."""
    if not isinstance(payload, Mapping):
        raise ValueError("draft: expected a configuration object")
    values = dict(payload)
    version = values.pop("schema_version", None)
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {version!r}")
    if values.pop("model_type", None) != MODEL_TYPE:
        raise ValueError(f"model_type must be {MODEL_TYPE!r}")
    architectures = values.pop("architectures", None)
    if (
        not isinstance(architectures, (list, tuple))
        or len(architectures) != 1
        or type(architectures[0]) is not str
    ):
        raise ValueError("architectures must contain exactly one architecture name")
    if "architecture" in values:
        raise ValueError("unknown field 'architecture'; use 'architectures'")
    values["architecture"] = architectures[0]
    registration = architecture_spec(architectures[0])
    extensions = values.pop("extensions", [])
    if not isinstance(extensions, (list, tuple)) or extensions:
        raise ValueError(
            "extensions: external extension loading is not implemented by this codec"
        )
    declared_depth = values.pop("num_hidden_layers", None)
    if "num_hidden_layers" in payload:
        positive_int("num_hidden_layers", declared_depth)
    for field, kind in _GROUPS.items():
        if field in values:
            values[field] = _map(
                values[field], field, lambda node: decode_component(kind, node)
            )
    if "layer_configs" in values:
        values["layer_configs"] = _map(
            values["layer_configs"], "layer_configs", decode_layer
        )
    for name in ("layer_types", "target_bindings"):
        if name in values:
            values[name] = _array(values[name], name)
    for field, kind in (("norm_config", "norm"), ("residual_config", "residual")):
        if field in values:
            values[field] = decode_component(kind, values[field])
    if "rope_parameters" in values:
        values["rope_parameters"] = decode_component(
            "rope", values["rope_parameters"], path="rope_parameters"
        )
    if "spec_config" in values:
        values["spec_config"] = decode_node(
            registration.schema, values["spec_config"], "spec_config"
        )
    if "lm_head" in values.get("target_bindings", ()):
        if "target_logit_transform" not in values:
            raise ValueError("bound lm_head requires target_logit_transform")
        values["target_logit_transform"] = decode_component(
            "logits", values["target_logit_transform"], path="target_logit_transform"
        )
    else:
        if "target_logit_transform" in values:
            raise ValueError(
                "target_logit_transform requires lm_head in target_bindings"
            )
        values["target_logit_transform"] = None
    config = _construct(DraftConfig, values, "draft")
    if declared_depth is not None and declared_depth != config.num_hidden_layers:
        raise ValueError("num_hidden_layers must equal len(layer_types)")
    if target_config is not None:
        validate_target(config, target_config)
    return config


def encode_config(config: DraftConfig) -> dict:
    """Produce complete canonical structure without changing the source config."""
    if type(config) is not DraftConfig:
        raise ValueError("expected DraftConfig")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_type": MODEL_TYPE,
        "architectures": [config.architecture],
        "extensions": [],
        **{name: getattr(config, name) for name in _DIMENSIONS},
        **{
            field: {
                name: encode_component(kind, node)
                for name, node in getattr(config, field).items()
            }
            for field, kind in _GROUPS.items()
        },
        "layer_configs": {
            name: encode_layer(node) for name, node in config.layer_configs.items()
        },
        "layer_types": list(config.layer_types),
        "num_hidden_layers": config.num_hidden_layers,
        "norm_config": encode_component("norm", config.norm_config),
        "residual_config": encode_component("residual", config.residual_config),
        "rope_parameters": encode_component("rope", config.rope_parameters),
        "spec_config": encode_node(config.spec_config),
        "target_bindings": list(config.target_bindings),
    }
    if config.target_logit_transform is not None:
        payload["target_logit_transform"] = encode_component(
            "logits", config.target_logit_transform
        )
    return payload


def parse_draft(
    payload: Mapping, *, target_config: Mapping | None = None
) -> DraftConfig:
    """Parse a resolved YAML draft mapping, expanding only documented shorthand.

    Vocabulary size derives from vocab_size when omitted. initializer_range is
    non-structural and defaults to 0.02. Omitted TargetFeatureSpec layer IDs
    derive from target_config and draft depth. Canonical decoding stays strict.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("draft: expected a configuration object")
    values = dict(payload)
    shorthand = {"attention", "mlp", "sublayer_conv"} & values.keys()
    if shorthand:
        named = {*_GROUPS, "layer_configs", "layer_types"} & values.keys()
        if named:
            raise ValueError(
                f"cannot mix shorthand with named definitions: {sorted(named)}"
            )
        missing = {"attention", "mlp", "num_hidden_layers"} - values.keys()
        if missing:
            raise ValueError(f"draft shorthand: missing fields {sorted(missing)}")
        depth = values["num_hidden_layers"]
        positive_int("num_hidden_layers", depth)
        layer = {"attention": "default", "mlp": "default"}
        values["attention_configs"] = {"default": values.pop("attention")}
        values["mlp_configs"] = {"default": values.pop("mlp")}
        values["conv_configs"] = {}
        if "sublayer_conv" in values:
            values["conv_configs"]["default"] = values.pop("sublayer_conv")
            layer["sublayer_conv"] = "default"
        values["layer_configs"] = {"default": layer}
        values["layer_types"] = ["default"] * depth
    values.setdefault("schema_version", SCHEMA_VERSION)
    values.setdefault("model_type", MODEL_TYPE)
    values.setdefault("conv_configs", {})
    # YAML omission means inherit the explicitly supplied model-level RoPE.
    # Canonical decoding remains strict and requires the resolved choice.
    if isinstance(values.get("attention_configs"), Mapping):
        values["attention_configs"] = {
            name: (
                {"rope_parameters": None, **node} if isinstance(node, Mapping) else node
            )
            for name, node in values["attention_configs"].items()
        }
    if "draft_vocab_size" not in values and "vocab_size" in values:
        values["draft_vocab_size"] = values["vocab_size"]
    values.setdefault("initializer_range", 0.02)
    architectures = values.get("architectures")
    spec = values.get("spec_config")
    if (
        target_config is not None
        and isinstance(architectures, (list, tuple))
        and len(architectures) == 1
        and type(architectures[0]) is str
        and isinstance(spec, Mapping)
    ):
        schema = architecture_spec(architectures[0]).schema
        features = spec.get("target_features")
        if (
            get_type_hints(schema).get("target_features") is TargetFeatureSpec
            and isinstance(features, Mapping)
            and "target_layer_ids" not in features
        ):
            target = _text_config(target_config)
            depth = len(_array(values.get("layer_types"), "layer_types"))
            layers = default_target_layer_ids(target.get("num_hidden_layers"), depth)
            values["spec_config"] = {
                **spec,
                "target_features": {**features, "target_layer_ids": layers},
            }
    bindings = values.get("target_bindings")
    if (
        target_config is not None
        and isinstance(bindings, (list, tuple))
        and "lm_head" in bindings
        and "target_logit_transform" not in values
    ):
        values["target_logit_transform"] = encode_component(
            "logits", resolve_target_logits(target_config)
        )
    return decode_config(values, target_config=target_config)
