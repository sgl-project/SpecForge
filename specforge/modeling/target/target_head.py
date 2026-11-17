import dataclasses
import logging
from typing import Optional

import torch
import torch.nn as nn

from specforge.modeling.target.checkpoint import load_checkpoint_tensors
from specforge.modeling.target.logit_transform import (
    TargetLogitTransform,
    resolve_target_logit_transform,
)
from specforge.modeling.target.target_utils import (
    load_target_config,
    target_hidden_size,
    target_ties_word_embeddings,
    target_vocab_size,
)
from specforge.utils import get_local_device, padding

logger = logging.getLogger(__name__)


class TargetHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = target_hidden_size(self.config)
        self.vocab_size = target_vocab_size(self.config)

        self.fc = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.logit_softcap: Optional[float] = None

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        lm_head_key: str = "lm_head.weight",
        embedding_key: str = "model.embed_tokens.weight",
        cache_dir: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = False,
    ) -> "TargetHead":

        config = load_target_config(
            model_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )

        instance = cls(config).to(dtype=dtype)

        tie_weights = target_ties_word_embeddings(config)

        instance._load_weights(
            model_path=model_path,
            lm_head_key=lm_head_key,
            embedding_key=embedding_key,
            tie_weights=tie_weights,
            cache_dir=cache_dir,
        )
        instance.apply_logit_transform(resolve_target_logit_transform(config))
        instance.freeze_weights()
        instance = instance.eval().to(device=get_local_device(), dtype=dtype)
        return instance

    @torch.no_grad()
    def apply_logit_transform(self, transform: TargetLogitTransform) -> None:
        """Make this head return the target's served logits.

        The head only produces teacher logits, so every linear factor folds
        into ``fc`` (the compact teacher streams ``fc.weight`` directly); the
        softcap is applied after the projection.
        """
        folded = dataclasses.replace(
            transform,
            head_weight_scale=transform.head_weight_scale * transform.logit_scale,
        )
        if not folded.head_is_identity:
            self.fc.weight.copy_(folded.fold_into_head(self.fc.weight))
        self.logit_softcap = transform.softcap

    @torch.no_grad()
    def _load_weights(
        self,
        model_path,
        lm_head_key: str = "lm_head.weight",
        embedding_key: str = "model.embed_tokens.weight",
        tie_weights: bool = False,
        cache_dir: Optional[str] = None,
    ):
        keys = {lm_head_key}
        if tie_weights:
            keys.add(embedding_key)
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
            if tensor.shape != self.fc.weight.shape:
                raise RuntimeError(
                    f"Shape mismatch for {key}. Expected {self.fc.weight.shape}, "
                    f"got {tensor.shape}"
                )
            tensors[key] = tensor.to(dtype=self.fc.weight.dtype)

        if tie_weights and len(tensors) > 1:
            tie_weights = torch.equal(tensors[embedding_key], tensors[lm_head_key])
            if not tie_weights:
                logger.warning(
                    "The config requests tying %s and %s, but the checkpoint contains "
                    "different values. Keeping the weights separate.",
                    embedding_key,
                    lm_head_key,
                )
        head_key = (
            embedding_key if tie_weights and embedding_key in tensors else lm_head_key
        )
        self.fc.weight.copy_(tensors[head_key])

    def freeze_weights(self):
        for param in self.fc.parameters():
            param.requires_grad = False

    def forward(self, hidden_states):
        logits = self.fc(hidden_states)
        if self.logit_softcap is not None:
            served = torch.tanh(logits.float() / self.logit_softcap)
            logits = (served * self.logit_softcap).to(logits.dtype)
        return logits

    def preprocess(self, input_ids, target, loss_mask):
        # Shift loss_mask with input_ids and final target states; auxiliary hidden
        # states remain unshifted because the draft model consumes them at row p.
        target = padding(target, left=False)
        input_ids = padding(input_ids, left=False)
        loss_mask = padding(loss_mask, left=False)[..., None]
        return input_ids, target, loss_mask
