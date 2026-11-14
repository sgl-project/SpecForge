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
        predictions: Deferred projection, aligned inputs and the loss plan.
        terms: Resolved inputs in the same order as LossPlan.terms.
        labels: Token targets shaped [units, positions].
        teacher: Optional teacher logits aligned with the selected vocabulary.
        outputs: Zero-argument callable that lazily projects and caches named
            outputs within this evaluation. Only compute callbacks may call it.
        materialized: The same output cache; applicability checks may inspect
            whether it is populated without triggering projection.
    """

    predictions: Predictions
    terms: tuple[TermInputs, ...]
    labels: torch.Tensor
    teacher: torch.Tensor | None
    outputs: Callable
    materialized: dict
    # Intermediate requests require differentiable per-position results.
    unreduced: bool = False


@dataclass(frozen=True)
class Implementation:
    """A registered optimization and its applicability contract.

    Attributes:
        name: Unique identifier, also used to break equal-priority ties.
        priority: Selection priority; larger values are considered first.
        supports: Callable(request, group) returning None when applicable,
            otherwise a human-readable rejection reason. It must not project.
        compute: Callable(request, group) returning a GroupResult. Runtime
            failures propagate; they are not treated as applicability failures.
    """

    name: str
    priority: int
    supports: Callable
    compute: Callable
    optimized: bool = True


_IMPLEMENTATIONS = {}


def register_implementation(name, *, supports, priority=0, optimized=True):
    """Register an optimized implementation for one or more loss terms.

    Args:
        name: Unique implementation identifier.
        supports: Callable(request, group) returning None if supported or a
            rejection-reason string. It checks capabilities without projection.
        priority: Selection priority; higher values take precedence.
        optimized: False for implementations available without fused execution.

    Returns:
        A decorator that registers and returns the compute callable unchanged.
        The callable accepts an ExecutionRequest and a tuple of term indices.

    Raises:
        ValueError: The decorated implementation's name is already registered.
    """

    def register(compute):
        if name in _IMPLEMENTATIONS:
            raise ValueError(f"Execution implementation {name!r} is already registered")
        _IMPLEMENTATIONS[name] = Implementation(
            name, priority, supports, compute, optimized
        )
        return compute

    return register


def select_implementation(request, group, *, name="auto", rejections=None, fused=True):
    """Select an applicable implementation without executing its kernel.

    Args:
        request: Projection, resolved term inputs and lazy output access.
        group: Tuple of indices into request.predictions.plan.terms to execute
            together. A one-element tuple requests independent execution.
        rejections: Optional list receiving rejection reasons for diagnostics.
        fused: Whether optimized implementations are eligible.
        name: "auto" checks priority order, breaking ties by ascending name.
            Any other value requires that specific registered implementation.

    Returns:
        The first applicable Implementation, or None when auto mode finds no
        match. The caller decides whether to split the group or use reference
        execution. Applicability checks may load optional kernel dependencies.

    Raises:
        ValueError: An explicitly named implementation is unknown or rejects
            the request. Exceptions raised by applicability checks propagate.
    """
    from . import domino_ce, linear_ce, reference, token_kernels  # noqa: F401

    candidates = sorted(
        _IMPLEMENTATIONS.values(), key=lambda item: (-item.priority, item.name)
    )
    if name != "auto":
        if name not in _IMPLEMENTATIONS:
            raise ValueError(f"Unknown execution implementation: {name!r}")
        candidates = [_IMPLEMENTATIONS[name]]
    for entry in candidates:
        if entry.optimized and not fused:
            continue
        reason = entry.supports(request, group)
        if reason is None:
            logger.debug("Loss group %s uses %s", group, entry.name)
            return entry
        if rejections is not None:
            rejections.append(f"{entry.name}: {reason}")
        logger.debug("Loss group %s skips %s: %s", group, entry.name, reason)
        if name != "auto":
            raise ValueError(f"{name} cannot execute this loss group: {reason}")
    logger.debug("Loss group %s uses reference execution", group)
    return None


class ExecutionSelection:
    """Resolve kernels once per prediction/group and retain their resume identity.

    The names dict is a live checkpoint mapping. Empty batches use the same
    selection, but the executor evaluates their zero-gradient reference path.
    """

    def __init__(self):
        self.names = {}
        self._entries = {}
        self._expected = None

    def restore(self, names):
        """Set the saved selections to check when real prediction inputs arrive."""
        self._expected = dict(names)
        self.names.clear()
        self._entries.clear()

    def resolve(self, request, group, prediction_index, *, fused=True):
        """Select and log a group once; reject rank or resume disagreement."""
        if not fused:
            return select_implementation(request, group, fused=False)
        key = f"{prediction_index}:" + ",".join(map(str, group))
        if key in self._entries:
            return self._entries[key]
        reasons, error, entry = [], None, None
        try:
            entry = select_implementation(request, group, rejections=reasons)
            hidden = request.predictions.hidden
            if (
                (entry is None or entry.name == "reference")
                and len(group) == 1
                and (
                    not hidden.is_cuda
                    or hidden.dtype
                    not in (torch.float16, torch.bfloat16, torch.float32)
                )
            ):
                raise ValueError("fused loss requires CUDA/ROCm FP16/BF16/FP32 inputs")
        except (ImportError, ValueError, RuntimeError) as exc:
            error = str(exc)
        name = entry.name if entry else ("split" if len(group) > 1 else "reference")
        self.record(key, name, error=error, reasons=reasons)
        self._entries[key] = entry
        return entry

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
        # Agree before invoking a new kernel, including ranks missing a dependency.
        import torch.distributed as dist

        outcomes = [(key, name, error)]
        if dist.is_initialized():
            outcomes = [None] * dist.get_world_size()
            dist.all_gather_object(outcomes, (key, name, error))
        errors = [item[2] for item in outcomes if item[2] is not None]
        if errors:
            raise ValueError("; ".join(dict.fromkeys(errors)))
        if any(item[:2] != (key, name) for item in outcomes):
            raise ValueError(f"Loss implementations differ across ranks: {outcomes}")
        self.names[key] = name
        logger.info(
            "Loss group %s uses %s%s",
            key,
            name,
            f"; skipped: {'; '.join(reasons)}" if reasons else "",
        )
