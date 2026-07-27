"""Local SGLang capture for algorithm-owned offline feature preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class OfflineCaptureBatch:
    """Generic batched auxiliary and final target states."""

    hidden_states: torch.Tensor
    last_hidden_states: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor


class OfflineSGLangCapture:
    """Frozen local target used by ``scripts/prepare_hidden_states.py`` only."""

    def __init__(self, backend) -> None:
        self._backend = backend
        self.capture_layers: Optional[List[int]] = None
        self.capture_method = "eagle3"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "OfflineSGLangCapture":
        from .sglang_backend import OfflineSGLangCaptureBackend

        backend = OfflineSGLangCaptureBackend.build(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        return cls(backend)

    def set_capture_layers(
        self,
        layer_ids: Optional[List[int]] = None,
        *,
        capture_method: str = "eagle3",
    ) -> None:
        self.capture_layers = layer_ids
        self.capture_method = capture_method
        self._backend.set_capture_layers(
            layer_ids,
            capture_method=capture_method,
        )

    def capture(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> OfflineCaptureBatch:
        data, aux_states, last_states = self._backend.capture(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
        )
        return OfflineCaptureBatch(
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


def load_offline_capture(
    pretrained_model_name_or_path: str,
    *,
    torch_dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = False,
    **kwargs,
) -> OfflineSGLangCapture:
    """Load the local SGLang target for offline hidden-state preparation."""

    return OfflineSGLangCapture.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )


# Compatibility aliases for callers of the original EAGLE3-only surface.
OfflineEagle3CaptureBatch = OfflineCaptureBatch
OfflineEagle3SGLangCapture = OfflineSGLangCapture


def load_offline_eagle3_capture(
    pretrained_model_name_or_path: str,
    *,
    torch_dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = False,
    **kwargs,
) -> OfflineSGLangCapture:
    return load_offline_capture(
        pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )


__all__ = [
    "OfflineCaptureBatch",
    "OfflineEagle3CaptureBatch",
    "OfflineEagle3SGLangCapture",
    "OfflineSGLangCapture",
    "load_offline_capture",
    "load_offline_eagle3_capture",
]
