# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""SGLangAdapter: the clean boundary between SpecForge and the target engine.

``generate_features(tasks, *, capture)`` is the single extraction entry point.
``capture`` is the typed :class:`CaptureConfig` derived from the active strategy,
not an untyped dict. The adapter wraps the existing ``Eagle3TargetModel`` (sglang
/ hf / custom backends all expose ``generate_eagle3_data``), records the exact
aux-layer IDs it captured, applies the target→draft projection demanded by
``capture.target_repr`` (the only place pruning happens), and returns per-sample
feature dicts. The RolloutWorker then runs :func:`verify_capture` before any
store write, so a layer/name/width mismatch fails loudly at this boundary rather
than as a downstream trainer bug.

Imports the SpecForge model code, so it is imported by rollout entry points.
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


class SGLangAdapter:
    """Adapter over a SpecForge ``Eagle3TargetModel`` (or any ``generate_eagle3_data``)."""

    SUPPORTED_FEATURE_NAMES = {
        "input_ids",
        "attention_mask",
        "loss_mask",
        "hidden_state",
        "target",
    }

    def __init__(
        self,
        target_model,
        *,
        device: str = "cuda",
        t2d: Optional[torch.Tensor] = None,
        shard_returns: bool = False,
    ) -> None:
        self.target_model = target_model
        self.device = device
        # t2d (target->draft vocab mask) only needed for pruned_logits capture.
        self.t2d = t2d
        self.shard_returns = shard_returns
        self._healthy = True

    def _recorded_aux_layer_ids(self) -> tuple:
        ids = getattr(self.target_model, "aux_hidden_states_layers", None)
        return tuple(ids) if ids is not None else ()

    # target representations this online adapter actually implements
    SUPPORTED_TARGET_REPRS = ("logits", "pruned_logits")

    def _project_target(
        self, target: torch.Tensor, capture: CaptureConfig
    ) -> torch.Tensor:
        if capture.target_repr == "logits":
            return target
        if capture.target_repr == "pruned_logits":
            if self.t2d is None:
                raise ValueError("pruned_logits capture requires a t2d vocab map")
            return target[..., self.t2d.to(target.device)]
        # Only advertise what we implement. 'hidden_state' capture (storing the
        # target's last hidden state) is not wired in the online adapter yet; the
        # offline path supports it (the strategy re-runs TargetHead).
        raise NotImplementedError(
            f"SGLangAdapter does not implement online capture for target_repr="
            f"{capture.target_repr!r}; supported: {self.SUPPORTED_TARGET_REPRS}"
        )

    def generate_features(
        self, tasks: List[PromptTask], *, capture: CaptureConfig
    ) -> List[Dict[str, Any]]:
        """Extract per-sample features, batching the engine call.

        Tasks are grouped by sequence length and each group is run through
        ``generate_eagle3_data`` in ONE batched forward (the engine's native
        batching), instead of a per-sample loop that would serialize N forwards.
        Equal-length grouping avoids intra-batch padding, so per-sample features
        are sliced out cleanly. The result preserves task order.
        """
        recorded = self._recorded_aux_layer_ids()
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
            loss_mask = torch.stack(
                [
                    _as_2d_long(
                        tasks[i].payload.get("loss_mask", [1] * input_ids.shape[1]),
                        self.device,
                    )[0]
                    for i in idxs
                ],
                dim=0,
            )
            attention_mask = torch.ones_like(input_ids)
            data = self.target_model.generate_eagle3_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
                shard_returns=self.shard_returns,
            )
            target = self._project_target(data.target, capture)
            for j, gi in enumerate(idxs):
                out[gi] = {
                    "input_ids": data.input_ids[j : j + 1],
                    "attention_mask": data.attention_mask[j : j + 1],
                    "loss_mask": data.loss_mask[j : j + 1],
                    "hidden_state": data.hidden_states[j : j + 1],
                    "target": target[j : j + 1],
                    # carried out-of-band for verify_capture; popped before put
                    "__aux_layer_ids__": recorded,
                }
        return out

    # NOTE: draft-weight hot update (update_draft_weights) is not implemented yet.

    def health(self) -> Dict[str, Any]:
        return {
            "healthy": self._healthy,
            "aux_hidden_state_layer_ids": list(self._recorded_aux_layer_ids()),
            "backend": getattr(self.target_model, "backend", "unknown"),
        }


__all__ = ["SGLangAdapter"]
