# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""TargetEngine: the backend-agnostic target-model abstraction (Phase B).

``TargetEngine`` wraps a frozen target model and exposes one generic
:meth:`capture` entry point. Algorithm-specific extraction is supplied by a
``TargetCapturePolicy``; backend-specific loading is supplied by the generic HF,
SGLang, and custom engine classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

import torch

# The live disaggregated capture server is a FeatureSource transport, not a
# second TargetEngine backend.
KNOWN_BACKENDS = ("sglang", "hf", "custom")


class TargetEngine(ABC):
    """Backend-agnostic frozen-target engine.

    Concrete subclasses implement a backend (SGLang, Hugging Face, or custom);
    their capture policy supplies the strategy-specific extraction.
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
    ) -> Any:
        """Run the frozen target forward and extract training features.

        Returns a typed ``TargetCaptureBatch`` supplied by the active capture
        policy.
        """

    def set_capture_layers(self, layer_ids: Optional[List[int]] = None) -> None:
        """Select which target layers' hidden states to capture.

        The active capture policy validates this list. EAGLE3 requires three
        auxiliary layers, while DFlash accepts an arbitrary list. Engines that
        do not capture intermediate layers may leave this unimplemented.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement set_capture_layers"
        )


__all__ = ["TargetEngine", "KNOWN_BACKENDS"]
