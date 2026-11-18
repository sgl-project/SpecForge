"""Execution implementations selected by projection and loss contracts."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from ..losses import LossContext, TokenLoss
from ..predictions import Predictions

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TermInputs:
    """Resolved inputs for one loss term.

    Attributes:
        loss: Mathematical loss and its options, including precomputed values.
        mask: Supervised positions, shaped [units, positions].
        weights: Effective per-position weights, aligned with mask.
        denominator: Scalar used after additive chunk statistics are merged.
        context: Teacher-mass and normalization inputs for contextual losses.
    """

    loss: TokenLoss
    mask: torch.Tensor
    weights: torch.Tensor
    denominator: torch.Tensor
    context: LossContext
    reduction: str = "positions"


@dataclass
class GroupResult:
    """Unnormalized output of an execution implementation.

    Attributes:
        numerators: Term indices mapped to weighted scalar loss sums. Values
            must not include the term coefficient or division by its denominator.
        predictions: Optional output names mapped to predicted token IDs,
            shaped [units, positions], for metric computation.
        values: Per-position values by term, returned for unreduced requests.
        statistics: Term indices mapped to reusable named tensors. They retain gradients;
            algorithms detach supervision explicitly.
    """

    numerators: dict[int, torch.Tensor]
    predictions: dict[str, torch.Tensor] = field(default_factory=dict)
    statistics: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)
    values: dict[int, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionRequest:
    """Data shared by applicability checks and kernel execution.

    Attributes:
        predictions: Deferred model predictions, or None for direct tensor requests.
        tensors: Supplied tensors by term index; reading them never projects the model.
        terms: Resolved inputs in the same order as LossPlan.terms.
        labels: Token targets shaped [units, positions].
        teacher: Optional teacher logits aligned with the selected vocabulary.
        outputs: Zero-argument callable that lazily projects and caches named
            outputs within this evaluation. Only compute callbacks may call it.
        requested_outputs: Extra intermediate results, separate from mathematical loss options.
        materialized: The same output cache; applicability checks may inspect
            whether it is populated without triggering projection.
    """

    predictions: Predictions | None
    terms: tuple[TermInputs, ...]
    labels: torch.Tensor
    teacher: torch.Tensor | None
    outputs: Callable | None
    materialized: dict
    # Intermediate requests require differentiable per-position results.
    unreduced: bool = False
    requested_outputs: dict = field(default_factory=dict)
    tensors: dict[int, torch.Tensor] = field(default_factory=dict)

    def tensor(self, index):
        """Get a supplied tensor or lazily project the named model output."""
        if index in self.tensors:
            return self.tensors[index]
        return self.outputs()[self.predictions.plan.terms[index].output]

    @property
    def input_tensor(self):
        """Tensor describing the actual execution device, dtype and emptiness."""
        return (
            next(iter(self.tensors.values()))
            if self.tensors
            else self.predictions.hidden
        )


@dataclass(frozen=True)
class Implementation:
    """A registered optimization and its applicability contract.

    Attributes:
        name: Unique identifier, also used to break equal-priority ties.
        priority: Selection priority; larger values are considered first.
        supports: Callable(request, group) returning None when applicable,
            otherwise a human-readable rejection reason. It must not project.
        category: Reference, generic optimization, or projection-specific optimization.
        losses: Mathematical definitions accepted by this implementation.
        compute: Callable(request, group) returning a GroupResult. Runtime
            failures propagate; they are not treated as applicability failures.
        auto: Whether the default (fused=None) may select it. An implementation
            that is not yet validated as a default registers False and runs
            only when fused is True.
    """

    name: str
    priority: int
    supports: Callable
    compute: Callable
    category: str = "generic"
    losses: tuple[str, ...] = ()
    auto: bool = True


_IMPLEMENTATIONS = {}


def register_implementation(
    name, *, supports, priority=0, category="generic", losses=(), auto=True
):
    """Register an optimized implementation for one or more loss terms.

    Args:
        name: Unique implementation identifier.
        supports: Callable(request, group) returning None if supported or a
            rejection-reason string. It checks capabilities without projection.
        priority: Selection priority; higher values take precedence.
        category: "generic" for automatic optimization, "specialized" for projection-specific
            optimization, or "reference" for portable mathematical evaluation.
        losses: Supported mathematical loss names. Without a declaration, optimizations
            receive only ordinary categorical prediction requests. The reference
            implementation may accept all request forms.
        auto: False keeps the implementation out of default selection; it then
            runs only when fused is True.

    Returns:
        A decorator that registers and returns the compute callable unchanged.
        The callable accepts an ExecutionRequest and a tuple of term indices.

    Raises:
        ValueError: The decorated implementation's name is already registered.
    """

    if category not in ("generic", "specialized", "reference"):
        raise ValueError(f"Unknown implementation category: {category!r}")

    def register(compute):
        if name in _IMPLEMENTATIONS:
            raise ValueError(f"Execution implementation {name!r} is already registered")
        _IMPLEMENTATIONS[name] = Implementation(
            name, priority, supports, compute, category, tuple(losses), auto
        )
        return compute

    return register


def implementation_registrations():
    """Return registered contracts for inspection and generated documentation."""
    from . import domino_ce, linear_ce, reference, token_kernels  # noqa: F401

    return dict(_IMPLEMENTATIONS)


def select_implementation(request, group=(0,), *, rejections=None, fused=False):
    """Choose the highest-priority compatible optimization or a reference.

    Args:
        fused: False evaluates the reference. None considers the
            implementations enabled by default and uses the reference where
            none applies; an unsupported device or dtype and a missing optional
            dependency are reasons to skip, not errors. True considers every
            implementation, propagates their dependency errors, and raises
            when a categorical loss can only be evaluated by the reference
            off CUDA/ROCm or outside FP16/BF16/FP32.

    Support checks must not materialize predictions.
    """
    candidates = sorted(
        (
            entry
            for entry in implementation_registrations().values()
            if entry.category == "reference"
            or fused is True
            or (fused is None and entry.auto)
        ),
        key=lambda entry: (entry.category == "reference", -entry.priority, entry.name),
    )
    for entry in candidates:
        if entry.losses and any(
            request.terms[i].loss.kind not in entry.losses for i in group
        ):
            reason = "mathematical loss does not match"
        elif (
            entry.category != "reference"
            and not entry.losses
            and (
                request.predictions is None
                or request.unreduced
                or any(
                    request.terms[i].loss.registration.input_kind != "categorical"
                    for i in group
                )
            )
        ):
            reason = "intermediate and non-categorical requests require declared losses"
        else:
            try:
                reason = entry.supports(request, group)
            except (ImportError, ValueError) as error:
                if fused is not None:
                    raise
                reason = str(error)
        if reason is None:
            if (
                fused is True
                and entry.category == "reference"
                and any(
                    request.terms[i].loss.registration.input_kind == "categorical"
                    for i in group
                )
                and (
                    not request.input_tensor.is_cuda
                    or request.input_tensor.dtype
                    not in (torch.float16, torch.bfloat16, torch.float32)
                )
            ):
                raise ValueError("fused loss requires CUDA/ROCm FP16/BF16/FP32 inputs")
            return entry
        if rejections is not None:
            rejections.append(f"{entry.name}: {reason}")
    return None


def compute_loss(request, group=(0,), *, fused=False, selection=None, key=0):
    """Execute declared terms through the common loss entry point.

    Args:
        request: Resolved supervision and direct or deferred prediction inputs.
        group: Term indices to compute together.
        fused: None selects a default-enabled optimization where one applies,
            True requires optimized execution, False evaluates the reference.
        selection: Optional distributed/resume selection tracker.
        key: Stable request identity within the algorithm.

    Returns:
        GroupResult with values or weighted numerators and requested statistics.
        Joint reference groups are evaluated independently without changing weights.
    """
    entry = (
        selection.resolve(request, group, key, fused=fused)
        if selection is not None
        else select_implementation(request, group, fused=fused)
    )
    if entry is not None and request.input_tensor.numel():
        return entry.compute(request, group)
    # Empty ranks still resolve the same kernel, but avoid zero-sized GPU launches.
    if len(group) == 1:
        reference = select_implementation(request, group)
        if reference is None:
            raise ValueError(
                "No reference implementation for the requested loss outputs"
            )
        return reference.compute(request, group)
    empty_fallback = entry is not None and not request.input_tensor.numel()
    result = GroupResult({})
    for index in group:
        item = compute_loss(
            request,
            (index,),
            fused=False if empty_fallback else fused,
            selection=None if empty_fallback else selection,
            key=key,
        )
        result.numerators.update(item.numerators)
        result.predictions.update(item.predictions)
        result.statistics.update(item.statistics)
        result.values.update(item.values)
    return result


class ExecutionSelection:
    """Validate group selections and retain their resume identity.

    The names dict is a live checkpoint mapping. Empty batches use the same
    selection, but the executor evaluates their zero-gradient reference path.
    """

    def __init__(self):
        self.names = {}
        self._expected = None

    def restore(self, names):
        """Set the saved selections to check when real prediction inputs arrive."""
        self._expected = dict(names)
        self.names.clear()

    def resolve(self, request, group, prediction_index, *, fused=True):
        """Select on each call; log once per key and reject rank or resume disagreement."""
        if fused is False:
            return select_implementation(request, group)
        key = f"{prediction_index}:" + ",".join(map(str, group))
        reasons, error, entry = [], None, None
        try:
            entry = select_implementation(
                request, group, fused=fused, rejections=reasons
            )
        except (ImportError, ValueError, RuntimeError) as exc:
            error = str(exc)
        name = entry.name if entry else ("split" if len(group) > 1 else "reference")
        self.record(key, name, error=error, reasons=reasons)
        return entry

    def agree(self):
        """Require every rank to have resolved the same implementations.

        A collective: the training layer calls it at a point every rank
        reaches, never from a forward. Ranks resolve independently, so a host
        missing an optional kernel package would otherwise train with a
        different implementation than the others.
        """
        import torch.distributed as dist

        if not dist.is_initialized():
            return
        resolved = [None] * dist.get_world_size()
        dist.all_gather_object(resolved, dict(self.names))
        if any(names != resolved[0] for names in resolved):
            raise ValueError(f"Loss implementations differ across ranks: {resolved}")

    def record(self, key, name, *, error=None, reasons=()):
        """Record a main or auxiliary implementation once and check consistency."""
        if key in self.names:
            if error is not None:
                raise ValueError(error)
            if self.names[key] != name:
                raise ValueError(
                    f"Loss implementation changed for {key}: {self.names[key]} -> {name}"
                )
            return
        if (
            error is None
            and self._expected is not None
            and self._expected.get(key) != name
        ):
            error = (
                f"loss implementation {key} changed on resume: "
                f"{self._expected.get(key)!r} -> {name!r}"
            )
        if error is not None:
            raise ValueError(error)
        self.names[key] = name
        logger.info(
            "Loss group %s uses %s%s",
            key,
            name,
            f"; skipped: {'; '.join(reasons)}" if reasons else "",
        )
