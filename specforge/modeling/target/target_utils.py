import json
import logging
import os
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoConfig

from specforge.modeling.target.checkpoint import load_checkpoint_tensors

logger = logging.getLogger(__name__)


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


def target_ties_word_embeddings(config) -> bool:
    return any(
        getattr(candidate, "tie_word_embeddings", False)
        for candidate in (config, target_text_config(config))
    )


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
        instance = cls(config).to(dtype=dtype)

        if embed_key is None:
            embed_key = "model.embed_tokens.weight"
        if lm_head_key is None:
            lm_head_key = "lm_head.weight"

        # 2. Handle Weight Tying
        tie_weights = target_ties_word_embeddings(config)

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
            if instance.lm_head.weight is instance.embed_tokens.weight:
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
        keys = {embed_key, lm_head_key}
        try:
            if tie_weights:
                tensors = load_checkpoint_tensors(
                    model_path, key_filter=keys.__contains__, cache_dir=cache_dir
                )
                if not tensors:
                    raise KeyError(
                        f"Checkpoint is missing tied tensors: {sorted(keys)}"
                    )
            else:
                tensors = load_checkpoint_tensors(
                    model_path, keys=keys, cache_dir=cache_dir
                )
        except KeyError as exc:
            raise RuntimeError(
                f"Required target weight tensors were not loaded: {exc}"
            ) from exc

        for key, tensor in tensors.items():
            if tensor.shape != self.embed_tokens.weight.shape:
                raise RuntimeError(
                    f"Shape mismatch for {key}. Expected {self.embed_tokens.weight.shape}, "
                    f"got {tensor.shape}"
                )
            tensors[key] = tensor.to(dtype=self.embed_tokens.weight.dtype)

        if tie_weights and len(tensors) > 1:
            tie_weights = torch.equal(tensors[embed_key], tensors[lm_head_key])
            if not tie_weights:
                logger.warning(
                    "The config requests tying %s and %s, but the checkpoint contains "
                    "different values. Keeping the weights separate.",
                    embed_key,
                    lm_head_key,
                )
        embedding = tensors[embed_key] if embed_key in tensors else tensors[lm_head_key]
        lm_head = tensors[lm_head_key] if lm_head_key in tensors else embedding
        if not tie_weights and self.lm_head.weight is self.embed_tokens.weight:
            self.lm_head.weight = nn.Parameter(
                self.lm_head.weight.clone(),
                requires_grad=self.lm_head.weight.requires_grad,
            )
        self.embed_tokens.weight.copy_(embedding)
        if tie_weights:
            self.lm_head.weight = self.embed_tokens.weight
        else:
            self.lm_head.weight.copy_(lm_head)

        return set(tensors)


__all__ = [
    "TargetEmbeddingsAndHead",
    "load_target_config",
    "target_text_config",
    "target_ties_word_embeddings",
    "target_vocab_size",
]
