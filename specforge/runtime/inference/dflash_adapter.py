# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""DFlashAdapter: the DFlash counterpart of SGLangAdapter.

Wraps a DFlash ``TargetEngine`` (sglang / hf), calling its generic ``capture(...)``
(the legacy ``generate_dflash_data`` is kept as a back-compat alias), and returns
per-sample feature dicts for the DataFlow rollout. DFlash's schema is
``{input_ids, hidden_states, loss_mask}`` — note ``hidden_states`` is the
concatenated target capture layers, and there is NO ``target`` distribution / no
vocab projection (DFlash trains on hard real-token labels), so unlike
``SGLangAdapter`` there is no ``_project_target`` / ``t2d`` step.

``verify_capture`` (run by the RolloutWorker before any store write) keys its
eagle3-specific aux/target checks on the feature names ``"hidden_state"`` /
``"target"``, which DFlash does not emit, so those checks self-skip; the
recorded-aux-layer check is skipped too because the RolloutWorker reads it via
``feats.pop("__aux_layer_ids__", None)`` and DFlash simply omits the key.

Imports SpecForge model code transitively (via the target backend), so it is
imported by rollout entry points, not at package load.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from specforge.runtime.contracts import PromptTask
from specforge.runtime.inference.capture import CaptureConfig


def _as_2d_long(values, device) -> torch.Tensor:
    t = torch.as_tensor(values, dtype=torch.long, device=device)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t


class DFlashAdapter:
    """Adapter over a SpecForge DFlash ``TargetEngine`` (via its generic ``capture()``)."""

    SUPPORTED_FEATURE_NAMES = {"input_ids", "hidden_states", "loss_mask"}

    def __init__(
        self,
        target_model,
        *,
        device: str = "cuda",
        t2d: Optional[torch.Tensor] = None,  # unused (DFlash has no vocab map); kept
    ) -> None:  # for a uniform make_adapter(target_model, *, device, t2d) signature
        self.target_model = target_model
        self.device = device
        self._healthy = True

    def generate_features(
        self, tasks: List[PromptTask], *, capture: CaptureConfig
    ) -> List[Dict[str, Any]]:
        """Extract per-sample DFlash features, batching equal-length prompts.

        Mirrors SGLangAdapter's length-grouped batching, but calls the engine's
        generic ``capture(...)`` and emits the DFlash schema. The target must have
        had ``set_capture_layers`` called so ``hidden_states`` width matches the
        draft's ``len(target_layer_ids) * hidden_size``.
        """
        out: List[Optional[Dict[str, Any]]] = [None] * len(tasks)

        groups: Dict[int, List[int]] = {}
        for i, task in enumerate(tasks):
            groups.setdefault(len(task.payload["input_ids"]), []).append(i)

        for _length, idxs in groups.items():
            input_ids = torch.stack(
                [
                    _as_2d_long(tasks[i].payload["input_ids"], self.device)[0]
                    for i in idxs
                ],
                dim=0,
            )  # (G, L)
            length = input_ids.shape[1]
            loss_mask = torch.stack(
                [
                    (
                        _as_2d_long(tasks[i].payload["loss_mask"], self.device)[0]
                        if "loss_mask" in tasks[i].payload
                        else torch.ones(length, dtype=torch.long, device=self.device)
                    )
                    for i in idxs
                ],
                dim=0,
            )
            attention_mask = torch.ones_like(input_ids)
            data = self.target_model.capture(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
            )
            for j, gi in enumerate(idxs):
                out[gi] = {
                    "input_ids": data.input_ids[j : j + 1],
                    "hidden_states": data.hidden_states[j : j + 1],
                    "loss_mask": data.loss_mask[j : j + 1],
                    # DFlash emits no eagle3 aux/target features. The recorded
                    # aux-layer check in verify_capture is skipped for free: the
                    # RolloutWorker reads it via feats.pop("__aux_layer_ids__", None),
                    # so an absent key is identical to an explicit None.
                }
        return out

    def health(self) -> Dict[str, Any]:
        return {
            "healthy": self._healthy,
            "backend": getattr(self.target_model, "backend", "unknown"),
        }


__all__ = ["DFlashAdapter"]
