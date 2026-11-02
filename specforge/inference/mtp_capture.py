"""Frozen, text-only native Qwen MTP features for the capture plugin.

This module uses checkpoint tensors and PyTorch, never SGLang runner internals.
Row a consumes token a and target state a-1, predicting token a+1. Row zero
is padding. MTP positions follow SGLang's shifted-prefill convention (a-1).
"""

import torch
from torch import nn
from transformers import AutoConfig, Qwen3Config

from specforge.modeling.draft.mtp.qwen3_5 import (
    PartialRotaryEmbedding,
    Qwen3_5MTPDraftModel,
)
from specforge.modeling.target.checkpoint import load_checkpoint_tensors


class FrozenMTPCapture:
    def __init__(self, model):
        self.model = model.eval().requires_grad_(False)

    @classmethod
    def from_pretrained(cls, path, *, device, dtype):
        native = AutoConfig.from_pretrained(path, trust_remote_code=False)
        text = getattr(native, "text_config", native)
        if text.model_type != "qwen3_5_text" or getattr(text, "num_experts", 0):
            raise ValueError(
                "mtp_hidden capture currently supports dense Qwen3.5/3.6 text MTP"
            )
        if getattr(text, "mtp_num_hidden_layers", 1) != 1:
            raise ValueError("mtp_hidden capture requires a single native MTP layer")
        if getattr(text, "mtp_use_dedicated_embeddings", False):
            raise ValueError("dedicated MTP embeddings are not supported")
        rope = getattr(text, "rope_parameters", None) or {}
        if rope.get("rope_type", "default") != "default":
            raise ValueError("mtp_hidden capture requires default native RoPE")
        values = text.to_dict()
        values.update(
            num_hidden_layers=1,
            layer_types=["full_attention"],
            rope_theta=rope.get("rope_theta", getattr(text, "rope_theta", 10000.0)),
            partial_rotary_factor=rope.get(
                "partial_rotary_factor", getattr(text, "partial_rotary_factor", 1.0)
            ),
        )
        values.pop("model_type", None)
        values["rope_parameters"] = {
            k: v
            for k, v in rope.items()
            if k not in ("mrope_section", "mrope_interleaved")
        }
        config = Qwen3Config(**values)
        config._attn_implementation = "sdpa"
        with torch.device("meta"):
            model = Qwen3_5MTPDraftModel(config)
        # The capture operation never projects to the vocabulary.
        model.mtp.lm_head = nn.Identity()
        embedding_key = "model.language_model.embed_tokens.weight"
        wanted = set(model.state_dict()) - {"embed_tokens.weight"}
        weights = load_checkpoint_tensors(path, keys=wanted | {embedding_key})
        weights["embed_tokens.weight"] = weights.pop(embedding_key)
        model.load_state_dict(weights, strict=True, assign=True)
        # inv_freq is non-persistent and therefore was not loaded from weights.
        model.mtp.rotary_emb = PartialRotaryEmbedding(config, config.head_dim)
        model.to(device=device, dtype=dtype)
        # Keep frequency construction in float32, as in the native rotary kernel.
        model.mtp.rotary_emb = PartialRotaryEmbedding(config, config.head_dim).to(
            device=device
        )
        return cls(model)

    @torch.no_grad()
    def __call__(self, input_ids, target_hidden):
        if input_ids.ndim != 1 or target_hidden.ndim != 2:
            raise ValueError("MTP capture expects one unpadded text sequence")
        if len(input_ids) != len(target_hidden):
            raise ValueError("MTP input_ids and target hidden lengths differ")
        output = target_hidden.new_zeros(target_hidden.shape)
        if len(input_ids) > 1:
            output[1:] = self.model.forward_hidden(
                input_ids[None, 1:],
                target_hidden[None, :-1],
                position_ids=torch.arange(len(input_ids) - 1, device=input_ids.device)[
                    None
                ],
            )[0]
        return output
