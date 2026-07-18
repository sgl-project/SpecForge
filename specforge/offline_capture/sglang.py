"""Local SGLang capture used exclusively by offline EAGLE3 data preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class OfflineEagle3CaptureBatch:
    """Batched target states required by offline EAGLE3 checkpoints."""

    hidden_states: torch.Tensor
    last_hidden_states: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor


class OfflineEagle3SGLangCapture:
    """Frozen local target used by ``scripts/prepare_hidden_states.py`` only."""

    def __init__(self, backend) -> None:
        self._backend = backend
        self.capture_layers: Optional[List[int]] = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "OfflineEagle3SGLangCapture":
        from .sglang_backend import OfflineSGLangCaptureBackend

        backend = OfflineSGLangCaptureBackend.build(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        return cls(backend)

    def set_capture_layers(self, layer_ids: Optional[List[int]] = None) -> None:
        self.capture_layers = layer_ids
        self._backend.set_eagle3_capture_layers(layer_ids)

    def capture(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> OfflineEagle3CaptureBatch:
        data, aux_states, last_states = self._backend.capture_eagle3(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
        )
        return OfflineEagle3CaptureBatch(
            hidden_states=torch.cat(
                [hidden.unsqueeze(0) for hidden in aux_states], dim=0
            ),
            last_hidden_states=torch.cat(
                [hidden.unsqueeze(0) for hidden in last_states], dim=0
            ),
            input_ids=torch.cat([row[0] for row in data], dim=0),
            attention_mask=torch.cat([row[1] for row in data], dim=0),
            loss_mask=torch.cat([row[2] for row in data], dim=0),
        )


def load_offline_eagle3_capture(
    pretrained_model_name_or_path: str,
    *,
    torch_dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = False,
    **kwargs,
) -> OfflineEagle3SGLangCapture:
    """Load the local SGLang target for offline hidden-state preparation."""

    return OfflineEagle3SGLangCapture.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )


__all__ = [
    "OfflineEagle3CaptureBatch",
    "OfflineEagle3SGLangCapture",
    "load_offline_eagle3_capture",
]
