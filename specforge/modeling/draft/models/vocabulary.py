"""Checkpoint-owned EAGLE vocabulary tensors, independent of dataset/file loading."""

import torch
from torch import Tensor, nn
from transformers import initialization as init

from specforge.config.draft._validation import positive_int


class DraftVocabulary(nn.Module):
    """t2d selects ascending target IDs; d2t stores offsets from draft IDs.

    Equal vocabularies start with identity mapping. Reduced vocabularies require
    an explicit mapping before their logits can be interpreted as target tokens.
    """

    def __init__(self, *, vocab_size: int, draft_vocab_size: int, device=None):
        super().__init__()
        positive_int("vocab_size", vocab_size)
        positive_int("draft_vocab_size", draft_vocab_size)
        if draft_vocab_size > vocab_size:
            raise ValueError("draft vocabulary must not exceed target vocabulary")
        self.vocab_size, self.draft_vocab_size = vocab_size, draft_vocab_size
        self.register_buffer(
            "t2d", torch.empty(vocab_size, device=device, dtype=torch.bool)
        )
        self.register_buffer(
            "d2t", torch.empty(draft_vocab_size, device=device, dtype=torch.long)
        )
        self._ready = False
        self.register_load_state_dict_post_hook(self._after_load)
        self.reset_draft_parameters()

    def reset_draft_parameters(self):
        identity = self.vocab_size == self.draft_vocab_size
        init.constant_(self.t2d, identity)
        init.zeros_(self.d2t)
        self._ready = identity and not self.t2d.is_meta

    @property
    def ready(self) -> bool:
        return self._ready

    def _after_load(self, module, incompatible_keys):
        self.validate_draft_state()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        self._ready = False
        for name in ("t2d", "d2t"):
            if prefix + name not in state_dict:
                raise ValueError(f"missing vocabulary tensor: {prefix + name}")
        self._validate_mapping(state_dict[prefix + "t2d"], state_dict[prefix + "d2t"])
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def validate_draft_state(self) -> None:
        """Validate restored buffers once, before any token IDs are interpreted."""
        self._ready = False
        self._validate_mapping(self.t2d, self.d2t)
        self._ready = True

    @torch.no_grad()
    def load_mapping(self, *, t2d: Tensor, d2t: Tensor) -> None:
        """Validate the entire mapping before copying into existing buffers."""
        if self.t2d.is_meta or self.d2t.is_meta:
            raise ValueError(
                "materialize vocabulary tensors before installing a mapping"
            )
        self._validate_mapping(t2d, d2t)
        self.t2d.copy_(t2d)
        self.d2t.copy_(d2t)
        self._ready = True

    def _validate_mapping(self, t2d: Tensor, d2t: Tensor) -> None:
        if t2d.shape != self.t2d.shape or t2d.dtype != torch.bool:
            raise ValueError("t2d must be boolean [vocab_size]")
        if d2t.shape != self.d2t.shape or d2t.dtype != torch.long:
            raise ValueError("d2t must be int64 [draft_vocab_size]")
        if any(t.is_meta for t in (t2d, d2t)):
            raise ValueError(
                "materialize vocabulary tensors before installing a mapping"
            )
        selected = t2d.nonzero().flatten()
        mapped = d2t.to(selected.device) + torch.arange(
            self.draft_vocab_size, device=selected.device
        )
        if not torch.equal(selected, mapped):
            raise ValueError("t2d/d2t must describe the same ascending target IDs")

    def require_ready(self):
        if not self._ready:
            raise RuntimeError(
                "Install or restore the draft vocabulary mapping before use"
            )

    def to_target(self, draft_ids: Tensor) -> Tensor:
        self.require_ready()
        if draft_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("draft IDs must be integers")
        # gather rejects negative IDs, unlike Python-style tensor indexing.
        offsets = self.d2t.gather(0, draft_ids.long().reshape(-1)).reshape_as(draft_ids)
        return draft_ids + offsets
