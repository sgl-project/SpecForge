# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Target -> draft vocabulary pruning shared by draft architectures.

A draft model may predict over a frequency-pruned subset of the target
vocabulary.  Two buffers describe that subset:

``t2d``  ``bool[target_vocab_size]``  -- True where a target id is kept.
``d2t``  ``long[draft_vocab_size]``   -- offset such that ``i + d2t[i]`` is the
                                         target id of draft id ``i``.

The buffers are registered by the concrete architecture and populated later,
once the token-frequency map for the run is known -- an EAGLE3 draft is built in
``build_model_bundle`` but its mapping is installed further down in
``_ensure_offline_vocab_mapping``.  Anything derived from the mapping (a
row-sliced target head, a target->draft label lookup) must therefore be built
lazily and re-derived when a different mapping is installed, which is what
``vocab_mapping_version`` exists for.
"""

from __future__ import annotations

from typing import Optional

import torch

#: Label value for target tokens outside the draft vocabulary. Matches
#: ``F.cross_entropy``'s default ``ignore_index`` and can never equal a real
#: draft id, so an out-of-vocabulary label is both unsupervised and counted as
#: an accuracy miss -- which is exactly what serving does with it.
OUT_OF_DRAFT_VOCAB_LABEL = -100


def _reject_conflicting_vocab_mapping(
    module, state_dict, prefix, *_load_hook_args
) -> None:
    """Refuse to silently replace an installed mapping with a different one.

    This is the resume hazard: the run resolves a mapping (from
    ``model.vocab_mapping_path``, or derived from the offline features), and then
    ``load_state_dict`` overwrites it with whatever the checkpoint carried. With
    the same ``draft_vocab_size`` nothing complains, yet every row of the pruned
    head now means a different token than the run believes. It does not crash --
    it just trains against a quietly wrong label space.

    Recording a fingerprint in the resume contract would not catch this: the
    contract is bound in ``build_model_bundle``, which runs *before*
    ``_ensure_offline_vocab_mapping`` installs the derived mapping, so the
    fingerprint would describe an empty one. Comparing at load time has no such
    ordering dependency.
    """
    if not module.vocab_mapping_loaded:
        return
    incoming = state_dict.get(prefix + "t2d")
    if incoming is None:
        return
    current = module.t2d
    incoming = incoming.to(device=current.device, dtype=current.dtype)
    if incoming.shape == current.shape and torch.equal(incoming, current):
        return
    raise ValueError(
        "the checkpoint's vocabulary mapping differs from the one this run "
        f"resolved ({int(current.sum())} kept tokens vs "
        f"{int(incoming.sum())} in the checkpoint, differing selections); the "
        "pruned lm_head rows would silently change meaning. Point "
        "model.vocab_mapping_path at the mapping this checkpoint was trained "
        "with, or start a fresh run."
    )


def _invalidate_vocab_mapping_derivations(module, _incompatible_keys) -> None:
    """Bump the mapping version and drop everything derived from the buffers.

    Registered as a ``load_state_dict`` post hook and called directly by
    :meth:`DraftVocabMappingMixin.install_vocab_mapping`, so both routes into the
    buffers converge on one invalidation point.
    """
    module.vocab_mapping_version = int(getattr(module, "vocab_mapping_version", 0)) + 1
    module._draft_vocab_index = None


class DraftVocabMappingMixin:
    """``t2d``/``d2t`` ownership for drafts that prune the target vocabulary.

    Concrete architectures register the buffers themselves (EAGLE3 always,
    DFlash-family only under :meth:`register_draft_vocab_buffers`); this mixin
    owns installation, validation, and the derived lookup table.
    """

    #: Bumped whenever the buffers change so lazy consumers can invalidate.
    vocab_mapping_version: int = 0

    @property
    def vocab_mapping_loaded(self) -> bool:
        """Whether a usable mapping is present, derived from the buffer itself.

        Deliberately not a flag set by :meth:`install_vocab_mapping`: buffers
        also arrive through ``load_state_dict`` and ``from_pretrained``, which
        call no method of ours.  A flag would leave a correctly reloaded
        checkpoint reporting "not installed" and refusing to run.  The empty
        ``t2d`` that :meth:`register_draft_vocab_buffers` starts from is the
        unambiguous "nothing installed yet" state, since a real mapping always
        selects exactly ``draft_vocab_size`` tokens.
        """
        if not getattr(self, "use_draft_vocab", False):
            return True
        t2d = getattr(self, "t2d", None)
        return t2d is not None and bool(t2d.any())

    def register_draft_vocab_buffers(
        self,
        *,
        vocab_size: int,
        draft_vocab_size: Optional[int],
    ) -> None:
        """Record the vocabulary sizes and register buffers only when pruning.

        Registering unconditionally would add ``t2d``/``d2t`` to the state dict
        of every existing full-vocabulary checkpoint, and warm start treats any
        missing key as fatal (``training/model_loading.py``).  Keeping the
        buffers behind ``use_draft_vocab`` leaves those checkpoints loadable
        byte-for-byte.
        """
        vocab_size = int(vocab_size)
        # Only None means "unset". Coercing with ``or`` would turn an explicit 0
        # into the full vocabulary and make the check below unreachable, while
        # build_model_bundle kept reading the literal 0 out of the config -- the
        # same setting meaning two different things in two places.
        if draft_vocab_size is None:
            draft_vocab_size = vocab_size
        if isinstance(draft_vocab_size, bool) or not isinstance(draft_vocab_size, int):
            raise ValueError(
                "draft_vocab_size must be an integer or None, got "
                f"{draft_vocab_size!r}"
            )
        if draft_vocab_size <= 0:
            raise ValueError(f"draft_vocab_size must be > 0, got {draft_vocab_size}")
        if draft_vocab_size > vocab_size:
            raise ValueError(
                "draft_vocab_size must not exceed vocab_size; got "
                f"draft_vocab_size={draft_vocab_size}, vocab_size={vocab_size}"
            )
        self.vocab_size = vocab_size
        self.draft_vocab_size = draft_vocab_size
        self.use_draft_vocab = draft_vocab_size != vocab_size
        if self.use_draft_vocab:
            self.register_buffer("t2d", torch.zeros(vocab_size, dtype=torch.bool))
            self.register_buffer("d2t", torch.zeros(draft_vocab_size, dtype=torch.long))
            # load_state_dict writes the buffers behind our back, so anything
            # derived from them (the row-sliced head, the label lookup) has to be
            # invalidated here or a resumed run keeps using the previous slice.
            self.register_load_state_dict_pre_hook(_reject_conflicting_vocab_mapping)
            self.register_load_state_dict_post_hook(
                _invalidate_vocab_mapping_derivations
            )

    def install_vocab_mapping(self, t2d: torch.Tensor, d2t: torch.Tensor) -> None:
        """Validate one mapping and copy it into the buffers.

        Every producer of a mapping funnels through here so the ascending-ids
        invariant (``nonzero(t2d) == d2t + arange``) is checked exactly once,
        rather than at each call site.
        """
        if not hasattr(self, "t2d") or not hasattr(self, "d2t"):
            raise ValueError(
                "t2d/d2t buffers are not present on this draft model; it was "
                "built without vocabulary pruning"
            )
        from specforge.core.compact_teacher import validate_vocab_mapping_consistency

        t2d = t2d.to(dtype=self.t2d.dtype)
        d2t = d2t.to(dtype=self.d2t.dtype)
        if t2d.shape != self.t2d.shape:
            raise ValueError(
                f"t2d has shape {list(t2d.shape)}, expected {list(self.t2d.shape)}"
            )
        if d2t.shape != self.d2t.shape:
            raise ValueError(
                f"d2t has shape {list(d2t.shape)}, expected {list(self.d2t.shape)}"
            )
        validate_vocab_mapping_consistency(t2d, d2t)
        self.t2d.copy_(t2d)
        self.d2t.copy_(d2t)
        _invalidate_vocab_mapping_derivations(self, None)

    def load_vocab_mapping(self, file_path: str) -> None:
        """Load and install the ``{"t2d", "d2t"}`` tensor file at ``file_path``."""
        mapping = torch.load(file_path, map_location="cpu")
        missing = [key for key in ("t2d", "d2t") if key not in mapping]
        if missing:
            raise ValueError(f"{file_path} is missing vocab-mapping keys {missing}")
        self.install_vocab_mapping(mapping["t2d"], mapping["d2t"])

    def draft_vocab_index(self) -> torch.Tensor:
        """Return ``long[vocab_size]``: draft id, or the ignore label if pruned.

        Cached against :attr:`vocab_mapping_version` rather than registered as a
        buffer -- it is fully derived from ``t2d`` and must not enter the
        checkpoint.
        """
        if not getattr(self, "use_draft_vocab", False):
            raise ValueError(
                "draft_vocab_index() is only defined when the draft prunes the "
                "target vocabulary"
            )
        self.require_vocab_mapping()
        cached = getattr(self, "_draft_vocab_index", None)
        version = int(self.vocab_mapping_version)
        if (
            cached is not None
            and cached[0] == version
            and cached[1].device == self.t2d.device
        ):
            return cached[1]
        # Built on CPU and then moved: masked assignment is a one-time setup
        # cost, and keeping it off the accelerator avoids relying on boolean
        # index-put support on backends such as Ascend NPU.
        mask = self.t2d.detach().to(device="cpu", dtype=torch.bool)
        index = torch.full(
            (int(mask.shape[0]),),
            OUT_OF_DRAFT_VOCAB_LABEL,
            dtype=torch.long,
        )
        index[mask] = torch.arange(int(mask.sum().item()), dtype=torch.long)
        index = index.to(device=self.t2d.device)
        self._draft_vocab_index = (version, index)
        return index

    def require_vocab_mapping(self) -> None:
        """Fail loudly when a pruned draft is used before its map is installed.

        The zero-initialized buffers are silently wrong rather than obviously
        wrong: an all-False ``t2d`` slices an empty head and an all-zero ``d2t``
        maps every draft id to 0.
        """
        if getattr(self, "use_draft_vocab", False) and not self.vocab_mapping_loaded:
            raise RuntimeError(
                "this draft prunes the target vocabulary "
                f"({getattr(self, 'draft_vocab_size', '?')} of "
                f"{getattr(self, 'vocab_size', '?')} tokens) but no t2d/d2t "
                "mapping has been installed; set model.vocab_mapping_path or "
                "run a topology that derives one"
            )


__all__ = ["DraftVocabMappingMixin", "OUT_OF_DRAFT_VOCAB_LABEL"]
