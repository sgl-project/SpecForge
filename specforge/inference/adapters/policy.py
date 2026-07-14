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

from specforge.inference.batch_partition import TargetBatchPartition
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

EAGLE3_VLM_FEATURE_SCHEMA = FeatureSchema(
    names=frozenset(
        {
            "input_ids",
            "attention_mask",
            "loss_mask",
            "hidden_state",
            "target",
            "position_ids",
        }
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
        output_partition: Optional[TargetBatchPartition] = None,
        input_preparer=None,
    ) -> None:
        self.target_model = target_model
        self.schema = schema
        self.device = device
        # t2d (target->draft vocab mask) only needed for pruned_logits capture.
        self.t2d = t2d
        # None => never pass the kwarg (engines whose capture doesn't take it).
        self.shard_returns = shard_returns
        self.output_partition = output_partition
        if shard_returns and output_partition is None:
            raise ValueError(
                "sharded target output requires an explicit target-TP partition"
            )
        self.input_preparer = input_preparer
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

        from specforge.inference.media import MediaInputs, PreparedTargetInput

        prepared: List[PreparedTargetInput] = []
        for task in tasks:
            if self.input_preparer is not None:
                prepared.append(self.input_preparer.prepare(task, self.device))
                continue
            length = len(task.payload["input_ids"])
            input_ids = _as_2d_long(task.payload["input_ids"], self.device)[0]
            loss_mask = _as_2d_long(
                task.payload.get("loss_mask", [1] * length), self.device
            )[0]
            prepared.append(
                PreparedTargetInput(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    loss_mask=loss_mask,
                )
            )

        groups: Dict[int, List[int]] = {}
        if self.shard_returns:
            # SGLang shards a contiguous target batch by TP rank. Match the
            # legacy padded collator so one backend call sees the complete
            # ``tp_size * configured_batch_size`` batch.
            groups[-1] = list(range(len(prepared)))
        else:
            for i, item in enumerate(prepared):
                groups.setdefault(item.input_ids.numel(), []).append(i)

        capture_kwargs: Dict[str, Any] = {}
        if self.shard_returns is not None:
            capture_kwargs["shard_returns"] = self.shard_returns

        for length, idxs in groups.items():
            del length
            if self.shard_returns:
                import torch.nn.functional as F

                max_length = max(prepared[i].input_ids.numel() for i in idxs)

                def padded(name: str) -> torch.Tensor:
                    values = []
                    for i in idxs:
                        value = getattr(prepared[i], name)
                        values.append(F.pad(value, (0, max_length - value.numel())))
                    return torch.stack(values, dim=0)

                input_ids = padded("input_ids")
                loss_mask = padded("loss_mask")
                attention_mask = padded("attention_mask")
            else:
                input_ids = torch.stack(
                    [prepared[i].input_ids for i in idxs],
                    dim=0,
                )  # (G, L)
                loss_mask = torch.stack(
                    [prepared[i].loss_mask for i in idxs],
                    dim=0,
                )
                attention_mask = torch.stack(
                    [prepared[i].attention_mask for i in idxs], dim=0
                )
            group_kwargs = dict(capture_kwargs)
            media = [prepared[i].media for i in idxs]
            if self.shard_returns and any(item is not None for item in media):
                raise ValueError("VLM capture does not support sharded target output")
            if any(item is not None for item in media):
                if not all(item is not None for item in media):
                    raise ValueError("cannot mix text and VLM prompts in one batch")
                group_kwargs["media_inputs"] = MediaInputs(
                    pixel_values=torch.cat(
                        [item.pixel_values for item in media if item is not None],
                        dim=0,
                    ),
                    image_grid_thw=tuple(
                        grid
                        for item in media
                        if item is not None
                        for grid in item.image_grid_thw
                    ),
                )
            data = self.target_model.capture(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
                **group_kwargs,
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
            output_indices = (
                self.output_partition.select(idxs) if self.shard_returns else idxs
            )
            if data.input_ids.shape[0] != len(output_indices):
                raise ValueError(
                    "target capture returned the wrong local batch size: "
                    f"expected={len(output_indices)}, actual={data.input_ids.shape[0]}"
                )
            for j, gi in enumerate(output_indices):
                feats: Dict[str, Any] = {}
                for name in sorted(schema.names):
                    if name == schema.hidden_feature:
                        feats[name] = data.hidden_states[j : j + 1]
                    elif name == schema.target_feature:
                        feats[name] = target[j : j + 1]
                    elif name == "position_ids":
                        item = prepared[gi]
                        if item.media is None:
                            raise ValueError(
                                "position_ids were requested without media inputs"
                            )
                        position_ids, _ = self.target_model.get_rope_index(
                            input_ids=data.input_ids[j : j + 1],
                            image_grid_thw=item.media.image_grid_thw[0],
                            attention_mask=data.attention_mask[j : j + 1],
                        )
                        if position_ids.ndim != 3:
                            raise ValueError(
                                "VLM position_ids must have three dimensions, got "
                                f"{tuple(position_ids.shape)}"
                            )
                        if position_ids.shape[0] != 3 and position_ids.shape[1] == 3:
                            position_ids = position_ids.permute(1, 0, 2)
                        if position_ids.shape[0] != 3 or position_ids.shape[1] != 1:
                            raise ValueError(
                                "one VLM sample needs position_ids [3,1,L], got "
                                f"{tuple(position_ids.shape)}"
                            )
                        feats[name] = position_ids.contiguous()
                    else:
                        feats[name] = getattr(data, name)[j : j + 1]
                if recorded is not None:
                    # carried out-of-band for verify_capture; the RolloutWorker
                    # pops it before any store put.
                    feats["__aux_layer_ids__"] = recorded
                out[gi] = feats
        if self.shard_returns:
            return [item for item in out if item is not None]
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
    "EAGLE3_VLM_FEATURE_SCHEMA",
    "DFLASH_FEATURE_SCHEMA",
]
