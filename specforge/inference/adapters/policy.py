# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""PolicyFeatureAdapter: ONE runtime adapter, parameterized by a FeatureSchema.

The runtime conversion ``PromptTask[] -> batched tensors -> TargetEngine.capture
-> per-sample features -> projection/pruning -> store-ready dicts`` is identical
for every draft algorithm; only the *shape of the emitted feature dict* differs.
That difference is a :class:`FeatureSchema` — a frozen, per-strategy declaration
registered on the ``StrategySpec`` — so the length-grouping / stacking /
order-preserving logic lives once here instead of once per algorithm adapter.

This adapter is the exclusive owner of the runtime-side concerns the target
engine must not know about: ``PromptTask`` payloads, task order, equal-length
grouping, per-sample slicing, target->draft vocab projection (``t2d``), and the
feature-dict schema. The engine side hands back only a typed
:class:`~specforge.inference.target_engine.target_capture_policy.TargetCaptureBatch`;
anything else is rejected loudly.

Every colocated strategy uses this adapter; server-capture sources implement the
same feature-source protocol at the transport boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional

import torch

from specforge.inference.capture import CaptureConfig
from specforge.runtime.contracts import PromptTask

# NOTE: target_engine.target_capture_policy (TargetCaptureBatch) is imported
# lazily inside generate_features — it drags in transformers, and this module
# must stay light enough for the StrategySpec registry to import FeatureSchema.


@dataclass(frozen=True)
class FeatureSchema:
    """What one strategy's store-ready feature dict looks like.

    ``names`` are the emitted feature keys. ``hidden_feature`` is the key that
    receives the capture's ``hidden_states``; ``target_feature`` (None for
    algorithms without a target distribution, e.g. DFlash) receives the —
    optionally vocab-projected — ``target``. ``records_aux_layer_ids`` emits the
    out-of-band ``__aux_layer_ids__`` key that ``verify_capture`` checks
    against the requested aux layers.
    """

    names: FrozenSet[str]
    target_feature: Optional[str]
    hidden_feature: str
    needs_vocab_projection: bool
    records_aux_layer_ids: bool = False


EAGLE3_FEATURE_SCHEMA = FeatureSchema(
    names=frozenset(
        {"input_ids", "attention_mask", "loss_mask", "hidden_state", "target"}
    ),
    target_feature="target",
    hidden_feature="hidden_state",
    needs_vocab_projection=True,
    records_aux_layer_ids=True,
)

# DFlash trains on hard real-token labels: no target distribution, no vocab
# projection, no aux-layer record ('hidden_states' IS the concatenated capture).
DFLASH_FEATURE_SCHEMA = FeatureSchema(
    names=frozenset({"input_ids", "hidden_states", "loss_mask"}),
    target_feature=None,
    hidden_feature="hidden_states",
    needs_vocab_projection=False,
    records_aux_layer_ids=False,
)


def _as_2d_long(values, device) -> torch.Tensor:
    t = torch.as_tensor(values, dtype=torch.long, device=device)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t


class PolicyFeatureAdapter:
    """Schema-parameterized ``FeatureSource`` over a ``TargetEngine``."""

    # target representations the online projection actually implements
    SUPPORTED_TARGET_REPRS = ("logits", "pruned_logits")

    def __init__(
        self,
        target_model,
        *,
        schema: FeatureSchema,
        device: str = "cuda",
        t2d: Optional[torch.Tensor] = None,
        shard_returns: Optional[bool] = None,
    ) -> None:
        self.target_model = target_model
        self.schema = schema
        self.device = device
        # t2d (target->draft vocab mask) only needed for pruned_logits capture.
        self.t2d = t2d
        # None => never pass the kwarg (engines whose capture doesn't take it).
        self.shard_returns = shard_returns
        self._healthy = True

    def _recorded_aux_layer_ids(self) -> tuple:
        ids = getattr(self.target_model, "capture_layers", None)
        return tuple(ids) if ids is not None else ()

    def _project_target(
        self, target: torch.Tensor, capture: CaptureConfig
    ) -> torch.Tensor:
        """The ONLY place target->draft projection/pruning happens."""
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
            f"{type(self).__name__} does not implement online capture for "
            f"target_repr={capture.target_repr!r}; "
            f"supported: {self.SUPPORTED_TARGET_REPRS}"
        )

    def generate_features(
        self, tasks: List[PromptTask], *, capture: CaptureConfig
    ) -> List[Dict[str, Any]]:
        """Extract per-sample features, batching the engine call.

        Tasks are grouped by sequence length and each group is run through the
        engine's ``capture(...)`` in ONE batched forward (the engine's native
        batching), instead of a per-sample loop that would serialize N forwards.
        Equal-length grouping avoids intra-batch padding, so per-sample features
        are sliced out cleanly. The result preserves task order.

        The ``capture`` keyword carries the runtime :class:`CaptureConfig`.
        """
        from specforge.inference.target_engine.target_capture_policy import (
            TargetCaptureBatch,
        )

        capture_config = capture
        schema = self.schema
        recorded = (
            self._recorded_aux_layer_ids() if schema.records_aux_layer_ids else None
        )
        out: List[Optional[Dict[str, Any]]] = [None] * len(tasks)

        groups: Dict[int, List[int]] = {}
        for i, task in enumerate(tasks):
            groups.setdefault(len(task.payload["input_ids"]), []).append(i)

        capture_kwargs: Dict[str, Any] = {}
        if self.shard_returns is not None:
            capture_kwargs["shard_returns"] = self.shard_returns

        for length, idxs in groups.items():
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
                        tasks[i].payload.get("loss_mask", [1] * length), self.device
                    )[0]
                    for i in idxs
                ],
                dim=0,
            )
            attention_mask = torch.ones_like(input_ids)
            data = self.target_model.capture(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
                **capture_kwargs,
            )
            if not isinstance(data, TargetCaptureBatch):
                raise TypeError(
                    f"TargetEngine.capture must return a TargetCaptureBatch, got "
                    f"{type(data).__name__}: the target engine layer must not "
                    f"shape per-sample features"
                )
            target = None
            if schema.target_feature is not None:
                target = (
                    self._project_target(data.target, capture_config)
                    if schema.needs_vocab_projection
                    else data.target
                )
            for j, gi in enumerate(idxs):
                feats: Dict[str, Any] = {}
                for name in sorted(schema.names):
                    if name == schema.hidden_feature:
                        feats[name] = data.hidden_states[j : j + 1]
                    elif name == schema.target_feature:
                        feats[name] = target[j : j + 1]
                    else:
                        feats[name] = getattr(data, name)[j : j + 1]
                if recorded is not None:
                    # carried out-of-band for verify_capture; the RolloutWorker
                    # pops it before any store put.
                    feats["__aux_layer_ids__"] = recorded
                out[gi] = feats
        return out

    # NOTE: draft-weight hot update (update_draft_weights) is not implemented yet.

    def health(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "healthy": self._healthy,
            "backend": getattr(self.target_model, "backend", "unknown"),
        }
        if self.schema.records_aux_layer_ids:
            info["aux_hidden_state_layer_ids"] = list(self._recorded_aux_layer_ids())
        return info


__all__ = [
    "FeatureSchema",
    "PolicyFeatureAdapter",
    "EAGLE3_FEATURE_SCHEMA",
    "DFLASH_FEATURE_SCHEMA",
]
