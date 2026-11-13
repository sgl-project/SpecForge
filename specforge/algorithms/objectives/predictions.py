"""Deferred prediction heads and aligned targets, independent of loss execution."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace

import torch
from torch import Tensor

from .losses import TokenLoss


@dataclass(frozen=True)
class Teacher:
    values: Tensor
    head: Callable | None = None

    def logits(self):
        with torch.no_grad():
            return self.head(self.values) if self.head is not None else self.values

    def select(self, units):
        return replace(self, values=self.values[units])


@dataclass(frozen=True)
class LossTerm:
    output: str = "logits"
    coefficient: float = 1.0


@dataclass
class AuxiliaryLoss:
    values: Tensor
    ratios: dict[str, tuple[Tensor, Tensor]] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class Predictions:
    """Each row is an indivisible projection unit: a block or independent token.

    All aligned tensors have [units, positions, ...] shape. select() preserves
    whole units, including recurrent head inputs. No full-vocabulary projection
    is retained here; project() can be invoked by an eager or chunked executor.
    """

    hidden: Tensor
    projector: Callable
    mask: Tensor
    positions: Tensor
    chain_ids: Tensor | None = None
    sequence_ids: Tensor | None = None
    labels: Tensor | None = None
    teacher: Teacher | None = None
    projection_inputs: dict[str, Tensor] = field(default_factory=dict)
    auxiliary_inputs: dict[str, Tensor] = field(default_factory=dict)
    auxiliary: Callable | None = None
    requires: frozenset[str] = frozenset()
    default_loss: TokenLoss = field(default_factory=lambda: TokenLoss(kind="ce"))
    terms: tuple[LossTerm, ...] = (LossTerm(),)
    vocab_mask: Tensor | None = None
    draft_to_target: Tensor | None = None
    denominator: Tensor | None = None
    accuracy_name: str = "acc"
    report_accuracy: bool = True

    def __post_init__(self):
        shape = self.hidden.shape[:2]
        if (
            self.hidden.ndim != 3
            or self.mask.shape != shape
            or self.positions.shape != shape
        ):
            raise ValueError(
                "predictions must align [units, positions] with hidden states"
            )
        if self.labels is not None and self.labels.shape != shape:
            raise ValueError("prediction labels must align with hidden states")
        for value in (self.chain_ids, self.sequence_ids):
            if value is not None and (
                value.shape != shape or value.dtype != torch.long
            ):
                raise ValueError("chain and sequence IDs must be aligned int64 tensors")
        aligned = [*self.projection_inputs.values(), *self.auxiliary_inputs.values()]
        if self.teacher is not None:
            aligned.append(self.teacher.values)
        if any(value.shape[:2] != shape for value in aligned):
            raise ValueError("prediction context must align with projection units")

    @property
    def capabilities(self):
        flags = {"logits", "chain_positions"}
        if self.chain_ids is not None and self.sequence_ids is not None:
            flags.add("acceptance_chain")
        if self.labels is not None or self.vocab_mask is not None:
            flags.add("labels")
        if self.teacher is not None:
            flags.add("teacher_logits")
        return frozenset(flags)

    def project(self):
        return self.projector(self.hidden, **self.projection_inputs)

    def select(self, units: slice):
        return replace(
            self,
            hidden=self.hidden[units],
            mask=self.mask[units],
            positions=self.positions[units],
            chain_ids=None if self.chain_ids is None else self.chain_ids[units],
            sequence_ids=(
                None if self.sequence_ids is None else self.sequence_ids[units]
            ),
            labels=None if self.labels is None else self.labels[units],
            teacher=None if self.teacher is None else self.teacher.select(units),
            projection_inputs={
                key: value[units] for key, value in self.projection_inputs.items()
            },
            auxiliary_inputs={
                key: value[units] for key, value in self.auxiliary_inputs.items()
            },
        )


@dataclass(frozen=True)
class PredictionBatch:
    predictions: Iterable[Predictions]
    reduction: str = "tokens"
    chunk_size: int = 0
