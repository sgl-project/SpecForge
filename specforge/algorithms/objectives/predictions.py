"""Deferred prediction heads and aligned targets, independent of loss execution."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace

import torch
from torch import Tensor

from .losses import TokenLoss
from .plan import LossPlan
from .projection import Projection


@dataclass(frozen=True)
class Teacher:
    """Detached teacher input, projected only when supervision needs it.

    Attributes:
        values: Hidden states when head is set; otherwise already-computed logits.
        head: Optional callable mapping values to full-vocabulary logits.
    """

    values: Tensor
    head: Callable | None = None

    def logits(self):
        """Return teacher logits, evaluating the optional head under no_grad."""
        with torch.no_grad():
            return self.head(self.values) if self.head is not None else self.values

    def compact(self, vocab_mask, chunk_size):
        """Project a linear teacher in vocabulary tiles.

        Args:
            vocab_mask: Boolean mask of selected tokens in the full target vocabulary.
            chunk_size: Positive number of vocabulary columns per tile.

        Returns:
            (selected_logits, full_vocabulary_argmax, retained_probability_mass),
            preserving full-vocabulary normalization without retaining full logits.

        Raises:
            ValueError: The head is not a supported linear projection or the
                vocabulary mask does not match its weight rows.
        """
        from specforge.core.compact_teacher import _tiled_logsumexp_argmax
        from specforge.modeling.draft.bindings import BoundLMHead

        if isinstance(self.head, BoundLMHead):
            weight = self.head.bound_weight()
        elif isinstance(self.head, torch.nn.Linear) and self.head.bias is None:
            weight = self.head.weight
        else:
            raise ValueError(
                "compact teacher requires hidden states and a linear teacher head"
            )
        if vocab_mask is None or vocab_mask.shape != (weight.shape[0],):
            raise ValueError(
                "compact teacher requires a matching draft vocabulary mask"
            )
        ids = vocab_mask.nonzero().flatten()
        log_z, argmax, logits = _tiled_logsumexp_argmax(
            self.values, weight, chunk_size=chunk_size, selected_token_ids=ids
        )
        mass = (logits - log_z).exp().sum(-1)
        return logits, argmax, mass

    def select(self, units):
        """Return teacher input indexed by a slice/tensor of complete units."""
        return replace(self, values=self.values[units])


@dataclass
class AuxiliaryLoss:
    """Named per-position auxiliary values and additive metric pairs.

    Values match LossPlan.auxiliary_terms by name and have shape [units, positions].
    The executor applies each term's coefficient, weights and denominator.
    """

    values: dict[str, Tensor]
    ratios: dict[str, tuple[Tensor, Tensor]] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class Predictions:
    """Each row is an indivisible projection unit: a block or independent token.

    All aligned tensors have [units, positions, ...] shape. select() preserves
    whole units, including recurrent head inputs. No full-vocabulary projection
    is retained here; project() can be invoked by an eager or chunked executor.

    The optional auxiliary callback accepts (predictions, projected_outputs,
    teacher_logits, default_weights, statistics, primitives) and returns AuxiliaryLoss.
    Statistics map main-term indices to detached named tensors from this
    evaluation only. Missing optional statistics must have a reference fallback.
    Primitives maps names (currently "ce_l1") to executor-owned callables;
    callbacks invoke them without choosing kernels or reading execution policy.
    Auxiliary terms and their reduction rules are declared in plan.
    """

    hidden: Tensor
    projection: Projection
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
    plan: LossPlan = field(default_factory=LossPlan)
    vocab_mask: Tensor | None = None
    draft_to_target: Tensor | None = None
    denominator: Tensor | None = None
    accuracy_name: str = "acc"
    report_accuracy: bool = True
    chunk_safe: bool = True

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
        for term in self.plan.all_terms:
            if term.mask is not None and term.mask.shape != shape:
                raise ValueError("Loss masks must align with prediction units")
        if bool(self.plan.auxiliary_terms) != (self.auxiliary is not None):
            raise ValueError(
                "Auxiliary callback and auxiliary terms must be declared together"
            )
        aligned = [*self.projection_inputs.values(), *self.auxiliary_inputs.values()]
        if self.teacher is not None:
            aligned.append(self.teacher.values)
        if any(value.shape[:2] != shape for value in aligned):
            raise ValueError("prediction context must align with projection units")

    @property
    def capabilities(self):
        """Return the supervision/chain capabilities provided by these inputs."""
        flags = {"logits", "chain_positions"}
        if self.chain_ids is not None and self.sequence_ids is not None:
            flags.add("acceptance_chain")
        if self.labels is not None or self.vocab_mask is not None:
            flags.add("labels")
        if self.teacher is not None:
            flags.add("teacher_logits")
        return frozenset(flags)

    def project(self):
        """Return named tensors from the ordinary projection without rerunning the backbone."""
        return self.projection.project(self.hidden, **self.projection_inputs)

    def select(self, units: slice):
        """Select complete projection units and all aligned supervision tensors.

        Args:
            units: Slice or tensor index along the first dimension.

        Returns:
            A new Predictions object with the selected hidden states, term masks,
            teacher/context inputs and chain IDs. Fixed denominators are unchanged.
        """
        return replace(
            self,
            hidden=self.hidden[units],
            plan=self.plan.select(units),
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
    """Protocol output consumed by the shared executor.

    Attributes:
        predictions: Iterable of deferred prediction sets, possibly a rollout.
        reduction: "tokens", "sum_means" or "global_microbatch".
        chunk_size: Protocol's default unit chunk size; zero means unsplit.
            An explicit runtime chunk_size overrides this value.
    """

    predictions: Iterable[Predictions]
    reduction: str = "tokens"
    chunk_size: int = 0
