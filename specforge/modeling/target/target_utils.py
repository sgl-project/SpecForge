import json
import os
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoConfig

from specforge.modeling.target.checkpoint import load_checkpoint_tensors


class _RawConfigShim:
    """Attribute view for released checkpoints with an unregistered model type."""

    def __init__(self, data: dict):
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name):
        try:
            value = self._data[name]
        except KeyError:
            raise AttributeError(name) from None
        return _RawConfigShim(value) if isinstance(value, dict) else value

    def to_dict(self) -> dict:
        return dict(self._data)


def load_target_config(
    model_path: str,
    *,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = False,
):
    """Load a target config, falling back to its public raw ``config.json``."""

    try:
        return AutoConfig.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
    except (ValueError, KeyError, OSError) as auto_error:
        if os.path.isdir(model_path):
            config_path = os.path.join(model_path, "config.json")
        elif os.path.isfile(model_path):
            config_path = model_path
        else:
            try:
                config_path = hf_hub_download(
                    repo_id=model_path,
                    filename="config.json",
                    cache_dir=cache_dir,
                )
            except Exception:
                raise auto_error
        try:
            with open(config_path, encoding="utf-8") as config_file:
                return _RawConfigShim(json.load(config_file))
        except (OSError, ValueError):
            raise auto_error


def target_text_config(config):
    return getattr(config, "text_config", config)


def target_vocab_size(config) -> int:
    text_config = target_text_config(config)
    return int(
        getattr(text_config, "padded_vocab_size", None) or text_config.vocab_size
    )


def target_hidden_size(config) -> int:
    text_config = target_text_config(config)
    return int(text_config.hidden_size)


class TargetEmbeddingsAndHead(nn.Module):
    """
    Efficiently loads only the embedding layer and lm_head from a pretrained model.
    Handles safetensors slicing and Weight Tying correctly.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        text_config = target_text_config(config)
        vocab_size = target_vocab_size(text_config)
        hidden_size = int(text_config.hidden_size)
        self.embed_tokens = nn.Embedding(
            vocab_size,
            hidden_size,
            padding_idx=getattr(text_config, "pad_token_id", None),
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        embed_key: Optional[str] = None,
        lm_head_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = False,
    ) -> "TargetEmbeddingsAndHead":

        # 1. Load Config
        config = load_target_config(
            model_path,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        instance = cls(config)

        if embed_key is None:
            embed_key = "model.embed_tokens.weight"
        if lm_head_key is None:
            lm_head_key = "lm_head.weight"

        # 2. Handle Weight Tying
        tie_weights = getattr(config, "tie_word_embeddings", False)

        # 3. Load Weights
        instance._load_weights(
            model_path,
            embed_key,
            lm_head_key,
            tie_weights,
            cache_dir=cache_dir,
        )

        text_config = target_text_config(config)
        mup_multiplier = getattr(
            text_config,
            "logits_mup_width_multiplier",
            getattr(config, "logits_mup_width_multiplier", None),
        )
        if mup_multiplier:
            if tie_weights:
                raise RuntimeError(
                    "cannot fold logits_mup_width_multiplier into a tied "
                    "embedding/LM head"
                )
            instance.lm_head.weight.data.div_(float(mup_multiplier))
            instance.lm_head_mup_folded = float(mup_multiplier)

        # 4. Move to Device & Freeze
        instance.to(device=device, dtype=dtype)
        instance.eval()
        instance.requires_grad_(False)

        return instance

    @torch.no_grad()
    def _load_weights(
        self,
        model_path: str,
        embed_key: str,
        lm_head_key: str,
        tie_weights: bool,
        cache_dir: Optional[str] = None,
    ) -> set[str]:
        destinations = [(embed_key, self.embed_tokens.weight)]
        if not tie_weights:
            destinations.append((lm_head_key, self.lm_head.weight))
        required_keys = [key for key, _destination in destinations]

        try:
            tensors = load_checkpoint_tensors(
                model_path,
                keys=required_keys,
                cache_dir=cache_dir,
            )
        except KeyError as exc:
            raise RuntimeError(
                f"Required target weight tensors were not loaded: {exc}"
            ) from exc

        for key, destination in destinations:
            tensor = tensors[key]
            if tensor.shape != destination.shape:
                raise RuntimeError(
                    f"Shape mismatch for {key}. Expected {destination.shape}, "
                    f"got {tensor.shape}"
                )
            destination.copy_(tensor)

        if tie_weights:
            print(
                "Weight tying detected: Sharing weights between Embeddings and LM Head."
            )
            self.lm_head.weight = self.embed_tokens.weight

        return set(required_keys)


__all__ = [
    "TargetEmbeddingsAndHead",
    "load_target_config",
    "target_text_config",
    "target_vocab_size",
]
