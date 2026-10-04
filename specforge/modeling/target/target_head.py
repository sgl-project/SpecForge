from typing import Optional

import torch
import torch.nn as nn

from specforge.modeling.target.checkpoint import load_checkpoint_tensors
from specforge.modeling.target.target_utils import (
    load_target_config,
    target_hidden_size,
    target_text_config,
    target_vocab_size,
)
from specforge.utils import get_local_device, padding


class TargetHead(nn.Module):
    def __init__(
        self,
        model_path,
        trust_remote_code: bool = False,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.config = load_target_config(
            model_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        self.hidden_size = target_hidden_size(self.config)
        self.vocab_size = target_vocab_size(self.config)

        self.fc = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        lm_head_key: str = "lm_head.weight",
        embedding_key: str = "model.embed_tokens.weight",
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> "TargetHead":
        target_head = cls(
            model_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        target_head.load_weights(
            model_path=model_path,
            lm_head_key=lm_head_key,
            embedding_key=embedding_key,
            cache_dir=cache_dir,
        )
        target_head.freeze_weights()
        target_head = target_head.eval().to(
            device=get_local_device(), dtype=torch.bfloat16
        )
        return target_head

    def _ties_word_embeddings(self) -> bool:
        # Multimodal configs may carry the flag at the top level, on
        # text_config, or both.
        return any(
            getattr(config, "tie_word_embeddings", False)
            for config in (self.config, target_text_config(self.config))
        )

    @torch.no_grad()
    def load_weights(
        self,
        model_path,
        lm_head_key: str = "lm_head.weight",
        embedding_key: str = "model.embed_tokens.weight",
        cache_dir: Optional[str] = None,
    ):
        head_key = lm_head_key
        try:
            tensors = load_checkpoint_tensors(
                model_path,
                keys=[lm_head_key],
                cache_dir=cache_dir,
            )
        except KeyError as exc:
            if not self._ties_word_embeddings():
                raise RuntimeError(
                    f"Target head key {lm_head_key!r} is missing from {model_path}"
                ) from exc
            # Tied checkpoints usually store only the input embedding, which
            # is also the output head.
            head_key = embedding_key
            try:
                tensors = load_checkpoint_tensors(
                    model_path,
                    keys=[embedding_key],
                    cache_dir=cache_dir,
                )
            except KeyError as tied_exc:
                raise RuntimeError(
                    f"Target head key {lm_head_key!r} and tied embedding key "
                    f"{embedding_key!r} are both missing from {model_path}"
                ) from tied_exc
        lm_head = tensors[head_key]
        if tuple(lm_head.shape) != tuple(self.fc.weight.shape):
            raise RuntimeError(
                f"Target head {head_key!r} has shape {tuple(lm_head.shape)}, "
                f"expected {tuple(self.fc.weight.shape)}"
            )
        self.fc.weight.copy_(lm_head)

    def freeze_weights(self):
        for param in self.fc.parameters():
            param.requires_grad = False

    def forward(self, hidden_states):
        return self.fc(hidden_states)

    def preprocess(self, input_ids, target, loss_mask):
        # Shift loss_mask with input_ids and final target states; auxiliary hidden
        # states remain unshifted because the draft model consumes them at row p.
        target = padding(target, left=False)
        input_ids = padding(input_ids, left=False)
        loss_mask = padding(loss_mask, left=False)[..., None]
        return input_ids, target, loss_mask
