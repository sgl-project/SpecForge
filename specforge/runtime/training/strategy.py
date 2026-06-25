# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""DraftTrainStrategy: per-draft-model required features + forward/loss + projection.

A strategy is the only place that knows how a particular draft model
(EAGLE3 / DFlash) turns a normalized ``TrainBatch`` into a loss. The
``TrainerCore`` stays branch-free: it never inspects ``target_repr`` or branches
on online/offline — the strategy owns the target projection (applying
``TargetHead`` / the ``t2d`` vocab map).

This module imports the SpecForge model code, so it is imported by training
entry points, not at ``specforge.runtime`` package load.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from specforge.runtime.contracts import TrainBatch


@dataclass(frozen=True)
class StepOutput:
    """Opaque per-step result a strategy hands back for metric aggregation.

    Kept generic so a TTT strategy (per-position lists) and a single-scalar
    strategy (DFlash) share one trainer loop.
    """

    loss: torch.Tensor
    metrics: Dict[str, Any]


class DraftTrainStrategy(abc.ABC):
    name: str
    required_features: set

    @abc.abstractmethod
    def trainable_module(self) -> nn.Module:
        """The module whose parameters the optimizer/backend owns."""

    def validate_batch(self, batch: TrainBatch) -> None:
        missing = {f for f in self.required_features if f not in batch.tensors}
        if missing:
            raise ValueError(
                f"{self.name} batch missing required features {sorted(missing)}; "
                f"present={sorted(batch.tensors)}"
            )

    @abc.abstractmethod
    def forward_loss(self, batch: TrainBatch) -> StepOutput: ...

    def checkpoint_state_filter(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Select the keys this strategy persists as draft weights."""
        return state_dict


class Eagle3TrainStrategy(DraftTrainStrategy):
    """EAGLE3 TTT strategy wrapping the existing ``OnlineEagle3Model``.

    Projection ownership: for ``target_repr == "hidden_state"`` the
    strategy re-runs the frozen ``TargetHead`` (lm_head) over the stored target
    last hidden state — exactly today's offline ``run_forward``. For
    ``logits`` / ``pruned_logits`` the rollout already produced the distribution
    and the strategy uses ``target`` as-is. The ``t2d`` vocab map is then applied
    inside ``OnlineEagle3Model.forward`` (unchanged math).
    """

    name = "eagle3"
    required_features = {
        "input_ids",
        "attention_mask",
        "loss_mask",
        "hidden_state",
        "target",
    }

    def __init__(
        self,
        eagle3_model: nn.Module,
        *,
        target_head: Optional[nn.Module] = None,
        ploss_decay: float = 0.8,
    ) -> None:
        self.eagle3_model = eagle3_model
        self.target_head = target_head
        self.ploss_decay = ploss_decay

    def trainable_module(self) -> nn.Module:
        return self.eagle3_model

    def _device(self) -> torch.device:
        return next(self.eagle3_model.parameters()).device

    def _prepare_target(
        self,
        target_repr: Optional[str],
        input_ids: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if target_repr == "hidden_state":
            if self.target_head is None:
                raise ValueError(
                    "target_repr='hidden_state' requires a target_head to re-run "
                    "the lm_head projection"
                )
            # mirrors offline run_forward: shift input_ids/target, add mask dim,
            # then project the target last hidden state to full-vocab logits.
            input_ids, target, loss_mask = self.target_head.preprocess(
                input_ids, target, loss_mask
            )
            target = self.target_head(target.to(device))
            return input_ids.to(device), target, loss_mask.to(device)
        # logits / pruned_logits: rollout already produced the distribution and
        # applied any shift; use the tensors as delivered.
        return input_ids.to(device), target.to(device), loss_mask.to(device)

    def forward_loss(self, batch: TrainBatch) -> StepOutput:
        self.validate_batch(batch)
        t = batch.tensors
        device = self._device()
        target_repr = batch.metadata.get("target_repr")

        input_ids, target, loss_mask = self._prepare_target(
            target_repr, t["input_ids"], t["target"], t["loss_mask"], device
        )
        position_ids = t.get("position_ids")
        (
            plosses,
            acceptance_rates,
            acces,
            acc_corrects,
            acc_denoms,
            metric_losses,
            metric_loss_denoms,
        ) = self.eagle3_model(
            input_ids=input_ids,
            attention_mask=t["attention_mask"].to(device),
            loss_mask=loss_mask,
            target=target,
            hidden_states=t["hidden_state"].to(device),
            position_ids=position_ids.to(device) if position_ids is not None else None,
        )
        weights = [self.ploss_decay**i for i in range(len(plosses))]
        loss = sum(weights[i] * plosses[i] for i in range(len(plosses)))
        return StepOutput(
            loss=loss,
            metrics={
                "plosses": [p.detach() for p in plosses],
                "acces": [a.detach() for a in acces],
                "acceptance_rates": [a.detach() for a in acceptance_rates],
                "acc_corrects": [c.detach() for c in acc_corrects],
                "acc_denoms": [d.detach() for d in acc_denoms],
                "metric_losses": [m.detach() for m in metric_losses],
                "metric_loss_denoms": [d.detach() for d in metric_loss_denoms],
            },
        )

    def checkpoint_state_filter(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        # EAGLE3 owns a frozen embedding loaded from the target; do not persist it.
        return {
            k.replace("draft_model.", ""): v
            for k, v in state_dict.items()
            if "draft_model." in k and "embed" not in k.lower()
        }


class DFlashTrainStrategy(DraftTrainStrategy):
    """DFlash block-parallel strategy wrapping the existing ``OnlineDFlashModel``.

    Shares ``TrainerController`` / ``FSDPTrainingBackend`` / ``FeatureDataLoader``
    / checkpoint with EAGLE3 — only the per-step forward/loss differs (a single
    block-wise pass returning a scalar loss, vs EAGLE3's TTT unroll). DFlash uses
    hard real-token labels + a separately-loaded target lm_head, so it needs no
    target distribution and no vocab map. Schema names are DFlash's own (not
    overloaded onto EAGLE3 names): ``hidden_states`` is the captured target
    context, distinct from EAGLE3's ``hidden_state``.
    """

    name = "dflash"
    required_features = {"input_ids", "hidden_states", "loss_mask"}

    def __init__(self, dflash_model: nn.Module) -> None:
        self.dflash_model = dflash_model

    def trainable_module(self) -> nn.Module:
        return self.dflash_model

    def _device(self) -> torch.device:
        return next(self.dflash_model.parameters()).device

    def forward_loss(self, batch: TrainBatch) -> StepOutput:
        self.validate_batch(batch)
        t = batch.tensors
        device = self._device()
        loss, accuracy = self.dflash_model(
            input_ids=t["input_ids"].to(device),
            hidden_states=t["hidden_states"].to(device),
            loss_mask=t["loss_mask"].to(device),
        )
        return StepOutput(loss=loss, metrics={"accuracy": accuracy.detach()})

    def checkpoint_state_filter(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        # DFlash keeps everything under draft_model.; the embedding/head live in
        # a separate target module that is NOT persisted as draft weights.
        return {
            k.replace("draft_model.", ""): v
            for k, v in state_dict.items()
            if "draft_model." in k
        }


__all__ = [
    "DraftTrainStrategy",
    "Eagle3TrainStrategy",
    "DFlashTrainStrategy",
    "StepOutput",
]
