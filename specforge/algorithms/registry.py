"""Explicit, immutable registrations for algorithm contracts and providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Tuple

from specforge.algorithms.contracts import AlgorithmSpec


@dataclass(frozen=True)
class AlgorithmRegistration:
    """One lookup result containing pure metadata and executable providers.

    Keeping the two halves in one registration avoids parallel spec/provider
    registries.  Planning reads ``spec``; the composition root alone reads
    ``providers``.
    """

    spec: AlgorithmSpec
    providers: object

    def __post_init__(self) -> None:
        if not isinstance(self.spec, AlgorithmSpec):
            raise TypeError("spec must be an AlgorithmSpec")
        if self.providers is None:
            raise TypeError("providers must not be None")
        provider_name = getattr(self.providers, "algorithm_name", self.spec.name)
        if provider_name != self.spec.name:
            raise ValueError(
                "provider algorithm_name must match spec.name: "
                f"{provider_name!r} != {self.spec.name!r}"
            )

    @property
    def name(self) -> str:
        return self.spec.name


@dataclass(frozen=True, init=False)
class AlgorithmRegistry:
    """Immutable catalog assembled explicitly by the composition root."""

    _registrations: Tuple[AlgorithmRegistration, ...]

    def __init__(
        self,
        registrations: Iterable[AlgorithmRegistration] = (),
    ) -> None:
        resolved = tuple(registrations)
        invalid = [
            type(registration).__name__
            for registration in resolved
            if not isinstance(registration, AlgorithmRegistration)
        ]
        if invalid:
            raise TypeError(
                "registry values must be AlgorithmRegistration, got " f"{invalid}"
            )
        names = [registration.name for registration in resolved]
        duplicates = sorted(name for name in set(names) if names.count(name) > 1)
        if duplicates:
            raise ValueError(f"duplicate algorithm registrations: {duplicates}")
        object.__setattr__(
            self,
            "_registrations",
            tuple(sorted(resolved, key=lambda registration: registration.name)),
        )

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(registration.name for registration in self._registrations)

    @property
    def registrations(self) -> Tuple[AlgorithmRegistration, ...]:
        return self._registrations

    def resolve(self, name: str) -> AlgorithmRegistration:
        for registration in self._registrations:
            if registration.name == name:
                return registration
        raise KeyError(
            f"unknown algorithm {name!r}; registered algorithms: {list(self.names)}"
        )

    def with_registration(
        self,
        registration: AlgorithmRegistration,
    ) -> "AlgorithmRegistry":
        return AlgorithmRegistry((*self._registrations, registration))

    def __iter__(self) -> Iterator[AlgorithmRegistration]:
        return iter(self._registrations)

    def __len__(self) -> int:
        return len(self._registrations)


__all__ = ["AlgorithmRegistration", "AlgorithmRegistry"]
