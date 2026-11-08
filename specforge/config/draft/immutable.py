"""Immutable, order-independent name tables used by the draft configuration."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

from ._validation import nonempty_string

T = TypeVar("T")


@dataclass(frozen=True, init=False)
class FrozenMap(Mapping[str, T], Generic[T]):
    """Copy a small name table into hashable storage; values must be immutable.

    Configuration nodes supply the values. Mutable collections are rejected,
    rather than retained behind a read-only view of the caller's dictionary.
    """

    _entries: tuple[tuple[str, T], ...]

    def __init__(self, values: Mapping[str, T]):
        if not isinstance(values, Mapping):
            raise ValueError("named definitions must be a mapping")
        entries = tuple(values.items())
        for name, value in entries:
            nonempty_string("definition name", name)
            try:
                hash(value)
            except TypeError as exc:
                raise ValueError(
                    f"definition {name!r} must be immutable and hashable"
                ) from exc
        object.__setattr__(self, "_entries", tuple(sorted(entries)))

    def __getitem__(self, name: str) -> T:
        for key, value in self._entries:
            if key == name:
                return value
        raise KeyError(name)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._entries)

    def __len__(self) -> int:
        return len(self._entries)
