# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""TargetEngine: the backend-agnostic target-model abstraction (Phase B).

This is the de-EAGLE3'd boundary extracted from the former ``Eagle3TargetModel``
ABC. A ``TargetEngine`` wraps a *frozen* target model and exposes ONE generic
extraction entry point, :meth:`capture`, plus a real ``backend`` tag. The
inference/transport split (sglang in-process / hf / custom / sglang_server) is a
*backend* axis **under** each algorithm engine, and — crucially — the
sglang-version-specific glue lives behind a replaceable capture backend
(``sglang_backend``), NOT in the algorithm engine, so a sglang bump touches one
place instead of every ``*TargetModel`` subclass.

Two sibling algorithm engines subclass this ABC:

* :class:`Eagle3TargetEngine` (``eagle3_target_model``) — EAGLE3 TTT capture
  (aux hidden states + logits), keeps the EAGLE3-specific
  ``set_aux_hidden_states_layers`` / ``generate_eagle3_data``.
* :class:`DFlashTargetEngine` (``dflash_target_model``) — DFlash block capture
  (concatenated layer hidden states, no target distribution).

The runtime inference adapters (``SGLangAdapter`` / ``DFlashAdapter``) wrap a
``TargetEngine`` and remain the ``FeatureSource`` seam to the ``RolloutWorker`` —
they are unchanged by this extraction; they call the generic engine and read the
now-real ``.backend`` tag in ``health()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import torch

if TYPE_CHECKING:
    from .target_capture_policy import TargetCaptureBatch

# Known transport/backend tags. ``sglang_server`` (a live frozen-target SGLang
# server, cross-node) is introduced by the sglang-capture-backend PR; its capture
# depth is gated by the O1.3 spike. The tag set is advisory (informational, used
# by adapter health + provenance), not an enum the ABC enforces.
KNOWN_BACKENDS = ("sglang", "hf", "custom", "sglang_server")


class TargetEngine(ABC):
    """Backend-agnostic frozen-target engine.

    Subclasses are organised on two axes: the *algorithm* (EAGLE3 / DFlash —
    the intermediate ABCs :class:`Eagle3TargetEngine` / :class:`DFlashTargetEngine`)
    and the *backend/transport* (sglang / hf / custom / sglang_server — the
    concrete leaf classes). Only the leaf classes are instantiable; each sets a
    real :attr:`backend` tag.
    """

    #: Transport tag; concrete leaf engines override this class attribute
    #: ("sglang" / "hf" / "custom" / "sglang_server"). Read by the inference
    #: adapters' ``health()`` and recorded as rollout provenance.
    backend: str = "unknown"

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "TargetEngine":
        """Load a frozen target model for this backend."""

    @abstractmethod
    def capture(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        **kwargs,
    ) -> "TargetCaptureBatch":
        """Run the frozen target forward and extract training features.

        The generic extraction entry point that replaces the EAGLE3-named
        ``generate_eagle3_data``. Returns a typed ``TargetCaptureBatch``
        (``Eagle3TargetOutput`` / ``DFlashTargetOutput``). Algorithm engines keep
        their original ``generate_*_data`` method as the concrete implementation
        and as a back-compat alias; ``capture`` simply dispatches to it, so the
        extraction is byte-identical to the pre-refactor path.
        """

    def set_capture_layers(self, layer_ids: Optional[List[int]] = None) -> None:
        """Select which target layers' hidden states to capture.

        The generic hook. EAGLE3 maps this onto its 3 aux layers
        (``set_aux_hidden_states_layers``); DFlash captures an arbitrary list.
        Engines that do not capture intermediate layers may leave this
        unimplemented.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement set_capture_layers"
        )


__all__ = ["TargetEngine", "KNOWN_BACKENDS"]
