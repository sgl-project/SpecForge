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
    selected_target_k: Optional[torch.Tensor] = None
    selected_target_v: Optional[torch.Tensor] = None
    prefix_masks: Optional[torch.Tensor] = None

    def feature_rows(self):
        """Yield per-sample H-Spec records when the backend provides them."""

        if self.selected_target_k is None or self.selected_target_v is None:
            return
        for index in range(self.input_ids.shape[0]):
            yield {
                "input_ids": self.input_ids[index],
                "loss_mask": self.loss_mask[index],
                "hidden_states": self.hidden_states[index],
                "target_last_hidden_states": self.last_hidden_states[index],
                "selected_target_k": self.selected_target_k[index],
                "selected_target_v": self.selected_target_v[index],
                "prefix_masks": self.prefix_masks[index],
            }


class OfflineSGLangCapture:
    """Frozen local target used by ``scripts/prepare_hidden_states.py`` only."""

    def __init__(self, backend, capture_method: str = "eagle3") -> None:
        self._backend = backend
        self.capture_layers: Optional[List[int]] = None
        self.capture_method = capture_method

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        capture_method: str = "eagle3",
        **kwargs,
    ) -> "OfflineSGLangCapture":
        from .sglang_backend import OfflineSGLangCaptureBackend

        backend = OfflineSGLangCaptureBackend.build(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        capture = cls(backend, capture_method=capture_method)
        return capture

    def set_capture_layers(
        self,
        layer_ids: Optional[List[int]] = None,
        *,
        capture_method: str = "eagle3",
        kv_layer_ids: Optional[List[int]] = None,
    ) -> None:
        self.capture_layers = layer_ids
        self.capture_method = capture_method
        if capture_method == "hspec" and kv_layer_ids is not None:
            self._backend.set_hspec_kv_layer_ids(kv_layer_ids)
        self._backend.set_capture_layers(
            layer_ids,
            capture_method=capture_method,
        )

    def set_hspec_kv_layer_ids(self, layer_ids) -> None:
        """Explicitly select the target layers whose K/V the drafter borrows."""

        self._backend.set_hspec_kv_layer_ids(layer_ids)

    def capture(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> OfflineCaptureBatch:
        if self.capture_method == "hspec":
            features = self._backend.capture_hspec(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
            )
            return OfflineCaptureBatch(
                # Per-sample features must stack on the batch axis so that
                # OfflineCaptureBatch.feature_rows can slice one record per
                # sample: hidden rows are [seq, width] and K/V rows are
                # [seq, kv_heads * head_dim * kv_layers].
                hidden_states=torch.stack(
                    [feature["hidden_states"] for feature in features], dim=0
                ),
                last_hidden_states=torch.stack(
                    [feature["target_last_hidden_states"] for feature in features],
                    dim=0,
                ),
                input_ids=torch.cat([feature["input_ids"] for feature in features], dim=0),
                attention_mask=attention_mask,
                loss_mask=torch.cat(
                    [feature["loss_mask"] for feature in features], dim=0
                ),
                selected_target_k=torch.stack(
                    [feature["selected_target_k"] for feature in features], dim=0
                ),
                selected_target_v=torch.stack(
                    [feature["selected_target_v"] for feature in features], dim=0
                ),
                prefix_masks=torch.stack(
                    [feature["prefix_masks"] for feature in features], dim=0
                ),
            )

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

    def capture_rows(self, input_ids: List[List[int]]):
        """Capture variable-length rows without padding target compute."""
        return self._backend.capture_rows(input_ids)


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
