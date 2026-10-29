"""HF configuration serialization with the public SpecForge JSON layout."""

from __future__ import annotations

import json

from huggingface_hub.dataclasses import strict
from transformers import PretrainedConfig, Qwen3Config

from .layout import decode_layout, encode_layout


class DraftSerializationMixin:
    """Keep legacy runtime attributes private to Python; save canonical JSON.

    ``to_dict`` retains HF runtime compatibility for existing providers. Use
    AutoDraftConfig.to_dict for the portable model setting. Full JSON is saved
    so loading never depends on defaults from a newer Transformers version.
    """

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        return super().from_dict(decode_layout(config_dict), **kwargs)

    def to_json_string(self, use_diff: bool = True) -> str:
        return (
            json.dumps(encode_layout(self.to_dict()), indent=2, sort_keys=True) + "\n"
        )


@strict
class DraftQwen3Config(DraftSerializationMixin, Qwen3Config):
    """Qwen3 HF fields with SpecForge draft JSON serialization."""


def prepare_hf_config(
    config: PretrainedConfig,
    config_class: type[PretrainedConfig],
    architecture: str,
) -> PretrainedConfig:
    """Give direct HF callers the same parsing/serialization as AutoDraftConfig.

    Preserve runtime backend selection, which HF deliberately omits from its
    serialized dictionary. The caller still owns algorithm validation.
    """
    if isinstance(config, config_class) and not hasattr(config, "spec_config"):
        return config
    payload = config.to_dict()
    payload.setdefault("architectures", [architecture])
    if payload["architectures"] is None:
        payload["architectures"] = [architecture]
    # Qwen's post-init clears sliding_window when the legacy enable flag is
    # false. A caller may have edited layer_types after constructing its HF
    # config; preserve that explicit layout when adapting the config class.
    if "sliding_attention" in (payload.get("layer_types") or []):
        payload["use_sliding_window"] = True
    result = config_class.from_dict(payload)
    result._attn_implementation = config._attn_implementation
    return result
