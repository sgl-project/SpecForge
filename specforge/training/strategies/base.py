# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""DraftTrainStrategy: per-draft-model required features + forward/loss + projection.

A strategy is the only place that knows how a draft model (EAGLE3 / P-EAGLE /
DFlash / Domino) turns a normalized ``TrainBatch`` into a loss; ``TrainerCore``
stays branch-free and the strategy owns the target projection. Imports model
code, so it is imported by training entry points, not at package load.
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
    """Per-step result: loss + strategy-specific metrics, kept generic so
    per-position (TTT) and single-scalar strategies share one trainer loop."""

    loss: torch.Tensor
    metrics: Dict[str, Any]


@dataclass(frozen=True)
class StepContext:
    """Training-schedule state passed into ``forward_loss`` for objectives that
    depend on where in training we are (e.g. Domino's decaying ``lambda_base``);
    most strategies ignore it."""

    global_step: int = 0
    total_steps: Optional[int] = None


def linear_lambda_base(
    global_step: int,
    total_steps: int,
    lambda_start: float = 1.0,
    decay_ratio: float = 0.5,
) -> float:
    """Domino base-loss weight: linear decay from ``lambda_start`` to 0 over the
    first ``total_steps * decay_ratio`` steps, then 0, clamped to ``[0, 1]``.
    Requires a real ``total_steps`` (> 0)."""
    decay_steps = max(1, int(total_steps * decay_ratio))
    progress = min(global_step / decay_steps, 1.0)
    return max(0.0, min(1.0, lambda_start * (1.0 - progress)))


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
    def forward_loss(
        self, batch: TrainBatch, ctx: Optional["StepContext"] = None
    ) -> StepOutput: ...

    def checkpoint_state_filter(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Select the keys this strategy persists as draft weights."""
        return state_dict


def _prepare_eagle_target(
    *,
    target_head: Optional[nn.Module],
    target_repr: Optional[str],
    input_ids: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize EAGLE-family teacher features for a training forward.

    Online capture already shifts logits and input IDs. Offline capture stores
    the target model's final hidden state, so the frozen target head owns the
    equivalent shift and projection to full-vocabulary logits.
    """
    if target_repr == "hidden_state":
        if target_head is None:
            raise ValueError(
                "target_repr='hidden_state' requires a target_head to re-run "
                "the lm_head projection"
            )
        input_ids, target, loss_mask = target_head.preprocess(
            input_ids, target, loss_mask
        )
        target = target_head(target.to(device))
        return input_ids.to(device), target, loss_mask.to(device)
    return input_ids.to(device), target.to(device), loss_mask.to(device)


class Eagle3TrainStrategy(DraftTrainStrategy):
    """EAGLE3 TTT strategy wrapping the existing ``OnlineEagle3Model``.

    For ``target_repr == "hidden_state"`` the strategy re-runs the frozen
    ``TargetHead`` over the stored target hidden state; ``logits`` /
    ``pruned_logits`` are used as delivered. The ``t2d`` vocab map is applied
    inside ``OnlineEagle3Model.forward``.
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
        return _prepare_eagle_target(
            target_head=self.target_head,
            target_repr=target_repr,
            input_ids=input_ids,
            target=target,
            loss_mask=loss_mask,
            device=device,
        )

    def forward_loss(
        self, batch: TrainBatch, ctx: Optional[StepContext] = None
    ) -> StepOutput:
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
        # The target-copied embedding is skipped only when actually frozen; a
        # trainable embedding must be persisted (checked on the live module —
        # state_dict tensors are detached and carry no requires_grad).
        embed_frozen = all(
            not p.requires_grad
            for n, p in self.eagle3_model.named_parameters()
            if "embed" in n.lower()
        )
        return {
            k.replace("draft_model.", ""): v
            for k, v in state_dict.items()
            if "draft_model." in k and not (embed_frozen and "embed" in k.lower())
        }


class PEagleTrainStrategy(DraftTrainStrategy):
    """P-EAGLE COD strategy wrapping ``OnlinePEagleModel``.

    P-EAGLE consumes the same target capture as EAGLE3. The runtime schema uses
    the singular ``hidden_state`` name, while ``OnlinePEagleModel.forward`` uses
    ``hidden_states``; this strategy is the explicit boundary between them.
    """

    name = "peagle"
    required_features = {
        "input_ids",
        "attention_mask",
        "loss_mask",
        "hidden_state",
        "target",
    }

    def __init__(
        self,
        peagle_model: nn.Module,
        *,
        target_head: Optional[nn.Module] = None,
    ) -> None:
        self.peagle_model = peagle_model
        self.target_head = target_head

    def trainable_module(self) -> nn.Module:
        return self.peagle_model

    def _device(self) -> torch.device:
        return next(self.peagle_model.parameters()).device

    def forward_loss(
        self, batch: TrainBatch, ctx: Optional[StepContext] = None
    ) -> StepOutput:
        self.validate_batch(batch)
        tensors = batch.tensors
        device = self._device()
        input_ids, target, loss_mask = _prepare_eagle_target(
            target_head=self.target_head,
            target_repr=batch.metadata.get("target_repr"),
            input_ids=tensors["input_ids"],
            target=tensors["target"],
            loss_mask=tensors["loss_mask"],
            device=device,
        )

        lengths = tensors.get("lengths")
        if lengths is None:
            # P-EAGLE is currently batch-size 1. Deriving the document length
            # from the padding mask prevents COD samples from treating padded
            # positions as part of the document in the offline path.
            lengths = tensors["attention_mask"].sum(dim=-1)

        loss, model_metrics = self.peagle_model(
            input_ids=input_ids,
            attention_mask=tensors["attention_mask"].to(device),
            loss_mask=loss_mask,
            target=target,
            hidden_states=tensors["hidden_state"].to(device),
            lengths=lengths.to(device),
        )
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise ValueError(
                "peagle model must return a scalar loss tensor; "
                f"got {type(loss).__name__} with shape="
                f"{getattr(loss, 'shape', None)}"
            )

        metrics = {
            name: value.detach() if isinstance(value, torch.Tensor) else value
            for name, value in model_metrics.items()
        }
        correct = metrics.get("full_acc_sum")
        denominator = metrics.get("full_acc_total")
        if correct is not None and denominator is not None:
            correct = torch.as_tensor(correct, device=device)
            denominator = torch.as_tensor(denominator, device=device)
            metrics["accuracy"] = correct / denominator.clamp_min(1)

        return StepOutput(loss=loss.reshape(()), metrics=metrics)

    def checkpoint_state_filter(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        # Unlike the stock EAGLE3 path, P-EAGLE trains its token embeddings and
        # mask_hidden parameter. Persist the complete draft model so both survive
        # checkpoint/resume and export.
        return {
            key.replace("draft_model.", ""): value
            for key, value in state_dict.items()
            if "draft_model." in key
        }


class DFlashTrainStrategy(DraftTrainStrategy):
    """DFlash block-parallel strategy wrapping the existing ``OnlineDFlashModel``.

    Shares the trainer/backend/loader/checkpoint spine with EAGLE3; only the
    per-step forward/loss differs (single block-wise pass, scalar loss, hard
    real-token labels — no target distribution, no vocab map). ``hidden_states``
    is DFlash's own schema name, distinct from EAGLE3's ``hidden_state``.
    """

    name = "dflash"
    required_features = {"input_ids", "hidden_states", "loss_mask"}

    def __init__(self, dflash_model: nn.Module) -> None:
        self.dflash_model = dflash_model

    def trainable_module(self) -> nn.Module:
        return self.dflash_model

    def _device(self) -> torch.device:
        return next(self.dflash_model.parameters()).device

    def forward_loss(
        self, batch: TrainBatch, ctx: Optional[StepContext] = None
    ) -> StepOutput:
        self.validate_batch(batch)
        t = batch.tensors
        device = self._device()
        loss, accuracy, _ = self.dflash_model(
            input_ids=t["input_ids"].to(device),
            hidden_states=t["hidden_states"].to(device),
            loss_mask=t["loss_mask"].to(device),
        )
        return StepOutput(
            loss=loss,
            metrics={"accuracy": accuracy.detach()},
        )

    def checkpoint_state_filter(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        # Everything trainable lives under draft_model.; the target
        # embedding/head are a separate module, not persisted as draft weights.
        return {
            k.replace("draft_model.", ""): v
            for k, v in state_dict.items()
            if "draft_model." in k
        }


class DSparkTrainStrategy(DraftTrainStrategy):
    """DSpark strategy over DFlash with target hidden-state supervision."""

    name = "dspark"
    required_features = {
        "input_ids",
        "hidden_states",
        "loss_mask",
        "target_last_hidden_states",
    }

    def __init__(self, dspark_model: nn.Module) -> None:
        self.dspark_model = dspark_model

    def trainable_module(self) -> nn.Module:
        return self.dspark_model

    def _device(self) -> torch.device:
        return next(self.dspark_model.parameters()).device

    def forward_loss(
        self, batch: TrainBatch, ctx: Optional[StepContext] = None
    ) -> StepOutput:
        self.validate_batch(batch)
        t = batch.tensors
        device = self._device()
        loss, accuracy, model_metrics = self.dspark_model(
            input_ids=t["input_ids"].to(device),
            hidden_states=t["hidden_states"].to(device),
            loss_mask=t["loss_mask"].to(device),
            target_last_hidden_states=t["target_last_hidden_states"].to(device),
        )
        metrics = {
            "accuracy": accuracy.detach(),
        }
        for name in (
            "ce_loss",
            "l1_loss",
            "confidence_loss",
            "confidence_abs_error",
        ):
            if name in model_metrics:
                metrics[name] = model_metrics[name]
        return StepOutput(loss=loss, metrics=metrics)

    def checkpoint_state_filter(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k.replace("draft_model.", ""): v
            for k, v in state_dict.items()
            if "draft_model." in k
        }


class DominoTrainStrategy(DraftTrainStrategy):
    """Domino block-parallel strategy wrapping ``OnlineDominoModel``.

    Shares the trainer/backend/loader/checkpoint spine and feature schema with
    DFlash. Unlike the others, its loss blends a base loss with a weight
    ``lambda_base`` that decays over training, so it reads :class:`StepContext`.
    """

    name = "domino"
    required_features = {"input_ids", "hidden_states", "loss_mask"}

    def __init__(
        self,
        domino_model: nn.Module,
        *,
        lambda_start: float = 1.0,
        decay_ratio: float = 0.5,
    ) -> None:
        self.domino_model = domino_model
        self.lambda_start = lambda_start
        self.decay_ratio = decay_ratio

    def trainable_module(self) -> nn.Module:
        return self.domino_model

    def _device(self) -> torch.device:
        return next(self.domino_model.parameters()).device

    def _lambda_base(self, ctx: Optional[StepContext]) -> float:
        # No schedule horizon -> pure final loss (lambda_base = 0).
        if ctx is None or not ctx.total_steps:
            return 0.0
        return linear_lambda_base(
            ctx.global_step, ctx.total_steps, self.lambda_start, self.decay_ratio
        )

    def forward_loss(
        self, batch: TrainBatch, ctx: Optional[StepContext] = None
    ) -> StepOutput:
        self.validate_batch(batch)
        t = batch.tensors
        device = self._device()
        lambda_base = self._lambda_base(ctx)
        loss, accuracy, _ = self.domino_model(
            input_ids=t["input_ids"].to(device),
            hidden_states=t["hidden_states"].to(device),
            loss_mask=t["loss_mask"].to(device),
            lambda_base=lambda_base,
        )
        return StepOutput(
            loss=loss,
            metrics={
                "accuracy": accuracy.detach(),
                "lambda_base": torch.tensor(float(lambda_base)),
            },
        )

    def checkpoint_state_filter(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        # Everything trainable lives under draft_model.; the target
        # embedding/head are a separate module, not persisted as draft weights.
        return {
            k.replace("draft_model.", ""): v
            for k, v in state_dict.items()
            if "draft_model." in k
        }


__all__ = [
    "DraftTrainStrategy",
    "Eagle3TrainStrategy",
    "PEagleTrainStrategy",
    "DFlashTrainStrategy",
    "DSparkTrainStrategy",
    "DominoTrainStrategy",
    "StepOutput",
    "StepContext",
    "linear_lambda_base",
]
