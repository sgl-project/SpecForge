# coding=utf-8
"""Shared building blocks: :class:`Registry`, :class:`RoutingResult` and the
:class:`TrainingState` contract every MoE component implements. Imports no
other module of the package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Generic, Optional, TypeVar, Union

import torch

T = TypeVar("T")

#: A scalar diagnostic: a 0-d tensor (no host sync) or a Python float.
MetricValue = Union[torch.Tensor, float]


class Registry(Generic[T]):
    """Name -> implementation map with a decorator form and helpful errors."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: Dict[str, T] = {}

    def register(self, name: str, item: Optional[T] = None) -> Callable[[T], T] | T:
        def _register(obj: T) -> T:
            if name in self._items and self._items[name] is not obj:
                raise ValueError(f"{self.kind} {name!r} is already registered")
            self._items[name] = obj
            return obj

        return _register if item is None else _register(item)

    def unregister(self, name: str) -> None:
        self._items.pop(name, None)

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            available = ", ".join(sorted(self._items)) or "<none registered>"
            raise KeyError(
                f"unknown {self.kind} {name!r}; available: {available}"
            ) from None


@dataclass
class RoutingResult:
    """Routing decision for a flat batch of ``T`` tokens.

    ``weights`` are the final combine weights (normalized and scaled as the
    recipe dictates) in fp32; ``indices`` the chosen experts; ``counts`` the
    per-expert assignment counts on device (no host sync), which dispatch and
    balancing both consume. ``scores`` are the full pre-selection affinities
    (differentiable, fp32) for policies that need a gradient signal, such as
    an auxiliary balance loss; routers may leave it ``None``.
    """

    weights: torch.Tensor  # [T, k] fp32
    indices: torch.Tensor  # [T, k] long
    counts: torch.Tensor  # [E] long
    scores: Optional[torch.Tensor] = None  # [T, E] fp32, differentiable

    @property
    def topk(self) -> int:
        return int(self.indices.shape[-1])


class TrainingState:
    """Training-time hooks of an MoE component; every method defaults to a no-op.

    Two timing rules every implementation must respect:

    - :meth:`observe` runs inside the layer forward (training only) and may
      only *stash*, overwriting and never accumulating: an activation-checkpoint
      recompute re-runs the forward and must leave identical state behind. A
      loss built here is consumed by the trainer right after the same forward.
    - :meth:`apply_pending_update` runs from the *model* forward before any
      routing, outside every checkpoint region, and may mutate selection state
      and run collectives. Mutating selection state inside a layer forward
      would make the recompute route differently and raise ``CheckpointError``.
    """

    def observe(self, routing: RoutingResult) -> None:
        """Stash this forward's routing outcome (training only)."""

    def apply_pending_update(self) -> None:
        """Consume the stash; called by the model outside checkpoint regions."""

    def aux_loss(self) -> Optional[torch.Tensor]:
        """Already-scaled auxiliary loss of the last forward, or ``None``."""
        return None

    def metrics(self) -> Dict[str, MetricValue]:
        """Scalar diagnostics of the last training forward (rank-local)."""
        return {}

    def reset_parameters(self, std: float) -> None:
        """Initialize bare Parameters the HF ``_init_weights`` pass cannot see."""
