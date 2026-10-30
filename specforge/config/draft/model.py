"""Read-only draft settings, independent of HF's mutable configuration objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Any


class _FrozenMapping(Mapping):
    """Immutable nested settings that remain safe to deepcopy and pickle."""

    __slots__ = ("_data",)

    def __init__(self, values):
        object.__setattr__(self, "_data", MappingProxyType(dict(values)))

    def __setattr__(self, name, value):
        raise AttributeError("draft settings are immutable")

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return type(self), (dict(self._data),)


def _freeze(value):
    if isinstance(value, Mapping):
        return _FrozenMapping({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


class _ConfigValues(SimpleNamespace):
    """Mutable parsing workspace; never retained by a model."""

    def to_dict(self):
        return dict(vars(self))


@dataclass(frozen=True)
class DraftModelConfig:
    """Resolved settings shared by the model and its layers.

    Nested mappings and sequences are read-only too. The conversion methods
    return independent, mutable representations.
    """

    _values: Mapping[str, Any]

    def __post_init__(self):
        object.__setattr__(self, "_values", _freeze(self._values))

    def __getattr__(self, name):
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(name) from None

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return type(self), (self.to_dict(),)

    def to_dict(self) -> dict[str, Any]:
        return _plain(self._values)

    def get_text_config(self, **kwargs):
        return self

    def to_hf_config(self):
        """Create an HF adapter without changing any runtime settings."""
        from .registry import resolve_draft_config_type

        registration = resolve_draft_config_type(self.architectures[0])
        payload = self.to_dict()
        backend = payload.pop("_attn_implementation", None)
        if "dflash_config" in payload:
            payload["dflash_config"]["block_size"] = self.block_size
        config = registration.config_class.from_dict(payload)
        config._attn_implementation = backend
        return config

    def save_pretrained(self, *args, **kwargs):
        return self.to_hf_config().save_pretrained(*args, **kwargs)


# Architecture defaults belong to SpecForge; HF adapters do not choose settings
# for inline training configs. Method defaults remain in their own parsers.
_COMMON_DEFAULTS = {
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "hidden_act": "silu",
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-6,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "pad_token_id": None,
    "bos_token_id": None,
    "eos_token_id": None,
    "use_cache": True,
    "tie_word_embeddings": False,
    "output_attentions": False,
    "output_hidden_states": False,
    "return_dict": True,
    "is_encoder_decoder": False,
    "dtype": None,
}


def inline_values(payload: Mapping, *, model_type: str) -> _ConfigValues:
    """Fill common defaults directly from the inline mapping, before validation."""
    from .base import integer

    family_defaults = {
        "llama": {
            "mlp_bias": False,
            "vocab_size": 32000,
            "intermediate_size": 11008,
            "max_position_embeddings": 2048,
            "num_key_value_heads": None,
            "head_dim": None,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
        "qwen3": {
            "vocab_size": 151936,
            "intermediate_size": 22016,
            "max_position_embeddings": 32768,
            "num_key_value_heads": 32,
            "head_dim": 128,
        },
    }
    values = {
        **_COMMON_DEFAULTS,
        **family_defaults.get(model_type, {}),
        **payload,
        "model_type": model_type,
    }
    values = _plain(values)
    values["tie_word_embeddings"] = False
    if values.get("draft_vocab_size") is None:
        values["draft_vocab_size"] = values["vocab_size"]
    if values.get("num_key_value_heads") is None:
        values["num_key_value_heads"] = values["num_attention_heads"]
    if values.get("head_dim") is None:
        heads = integer(values["num_attention_heads"], "num_attention_heads")
        values["head_dim"] = integer(values["hidden_size"], "hidden_size") // heads
    values["_attn_implementation"] = (
        values.get("_attn_implementation")
        or values.pop("attn_implementation", None)
        or "sdpa"
    )
    integer(values["num_hidden_layers"], "num_hidden_layers")
    if model_type == "qwen3":
        window = (
            values.get("sliding_window") if values.get("use_sliding_window") else None
        )
        values.setdefault("sliding_window", window)
        if values.get("layer_types") is None:
            values["layer_types"] = [
                (
                    "sliding_attention"
                    if window is not None and i >= values.get("max_window_layers", 28)
                    else "full_attention"
                )
                for i in range(values["num_hidden_layers"])
            ]
    rope = values.get("rope_parameters") or values.get("rope_scaling") or {}
    if "rope_type" in rope or "type" in rope or not rope:
        rope = dict(rope)
        rope.setdefault("rope_type", rope.pop("type", "default"))
        rope.setdefault("rope_theta", values.get("rope_theta", 10000.0))
    values["rope_parameters"] = rope
    return _ConfigValues(**values)
