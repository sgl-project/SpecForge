"""Explicit, instance-owned registry for algorithm contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Tuple

from specforge.algorithms.contracts import AlgorithmSpec


@dataclass(frozen=True, init=False)
class AlgorithmRegistry:
    """Immutable catalog assembled explicitly by the composition root.

    Importing this module never registers builtins or imports algorithm
    implementations.  ``with_spec`` returns a new registry, so tests and
    independent applications cannot mutate one another through module state.
    """

    _specs: Tuple[AlgorithmSpec, ...]

    def __init__(self, specs: Iterable[AlgorithmSpec] = ()) -> None:
        resolved_specs = tuple(specs)
        invalid = [
            type(spec).__name__
            for spec in resolved_specs
            if not isinstance(spec, AlgorithmSpec)
        ]
        if invalid:
            raise TypeError(f"registry values must be AlgorithmSpec, got {invalid}")

        names = [spec.name for spec in resolved_specs]
        duplicates = sorted(name for name in set(names) if names.count(name) > 1)
        if duplicates:
            raise ValueError(f"duplicate algorithm registrations: {duplicates}")

        object.__setattr__(
            self,
            "_specs",
            tuple(sorted(resolved_specs, key=lambda spec: spec.name)),
        )

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(spec.name for spec in self._specs)

    @property
    def specs(self) -> Tuple[AlgorithmSpec, ...]:
        return self._specs

    def resolve(self, name: str) -> AlgorithmSpec:
        for spec in self._specs:
            if spec.name == name:
                return spec
        raise KeyError(
            f"unknown algorithm {name!r}; registered algorithms: {list(self.names)}"
        )

    def with_spec(self, spec: AlgorithmSpec) -> "AlgorithmRegistry":
        """Return a new registry containing ``spec``; duplicates fail closed."""

        return AlgorithmRegistry((*self._specs, spec))

    def __iter__(self) -> Iterator[AlgorithmSpec]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)


__all__ = ["AlgorithmRegistry"]
