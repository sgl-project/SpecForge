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
        self.reset_draft_parameters()

    def reset_draft_parameters(self):
        identity = self.vocab_size == self.draft_vocab_size
        init.constant_(self.t2d, identity)
        init.zeros_(self.d2t)
        self._ready = identity and not self.t2d.is_meta

    @property
    def ready(self) -> bool:
        return self._ready

    def _apply(self, fn, recurse=True):
        was_ready = self._ready
        self._ready = False
        result = super()._apply(fn, recurse=recurse)
        if was_ready:
            self.validate_draft_state()
        return result

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        keys = (prefix + "t2d", prefix + "d2t")
        present = [key in state_dict for key in keys]
        if any(present):
            self._ready = False
            if not all(present):
                raise ValueError(
                    f"missing vocabulary tensor: {keys[present.index(False)]}"
                )
            self._validate_mapping(
                *(state_dict[key] for key in keys), allow_unmapped=True
            )
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        if all(present):
            self.validate_draft_state()

    def validate_draft_state(self) -> None:
        """Validate restored buffers once, before any token IDs are interpreted."""
        self._ready = False
        if not (self.t2d.is_meta or self.d2t.is_meta):
            self._ready = self._validate_mapping(
                self.t2d, self.d2t, allow_unmapped=True
            )

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

    def _validate_mapping(
        self, t2d: Tensor, d2t: Tensor, *, allow_unmapped=False
    ) -> bool:
        if t2d.shape != self.t2d.shape or t2d.dtype != torch.bool:
            raise ValueError("t2d must be boolean [vocab_size]")
        if d2t.shape != self.d2t.shape or d2t.dtype != torch.long:
            raise ValueError("d2t must be int64 [draft_vocab_size]")
        if any(t.is_meta for t in (t2d, d2t)):
            raise ValueError(
                "materialize vocabulary tensors before installing a mapping"
            )
        if allow_unmapped and self.draft_vocab_size < self.vocab_size:
            if not t2d.any().item() and not d2t.any().item():
                return False
        selected = t2d.nonzero().flatten()
        mapped = d2t.to(selected.device) + torch.arange(
            self.draft_vocab_size, device=selected.device
        )
        if not torch.equal(selected, mapped):
            raise ValueError("t2d/d2t must describe the same ascending target IDs")
        return True

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
