"""Projection adapters expose computation, never loss or training policy."""

from collections.abc import Callable
from dataclasses import dataclass, field

from torch import nn


@dataclass(frozen=True)
class Projection:
    """Ordinary projection plus explicit execution-safety constraints.

    Attributes:
        project: Callable(hidden, **inputs) returning named output tensors.
            Hidden has shape [units, positions, hidden_size].
        chunk_safe: Whether the first dimension can be split into independent
            projection units without changing computation semantics.
        recompute_safe: Whether execution may be replayed for checkpointing or
            two-pass loss evaluation, with RNG restoration when needed.
    """

    project: Callable
    chunk_safe: bool = field(default=True, kw_only=True)
    recompute_safe: bool = field(default=True, kw_only=True)


@dataclass(frozen=True)
class LinearProjection(Projection):
    """Explicit capability for a plain, untransformed linear output.

    Attributes:
        head: Existing nn.Linear or BoundLMHead referenced by the model.
            Parameters remain owned by the model; this adapter does not copy them.
        project: Ordinary projection path, inherited from Projection.
    """

    head: nn.Module

    def weight_and_bias(self):
        """Access the model's existing linear parameters.

        Returns:
            (weight, bias), with weight shaped [vocabulary, hidden_size] and bias
            shaped [vocabulary] or None.

        Raises:
            TypeError: The head is not a supported plain linear or bound head.
            RuntimeError: The bound head's target weight has not been installed.
        """
        from specforge.modeling.draft.bindings import BoundLMHead

        if type(self.head) is nn.Linear:
            return self.head.weight, self.head.bias
        if type(self.head) is BoundLMHead:
            return self.head.bound_weight(), None
        raise TypeError("Linear projection requires a plain linear or bound head")
