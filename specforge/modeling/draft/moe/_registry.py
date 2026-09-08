# coding=utf-8
"""Tiny named-registry helper shared by the MoE component registries."""

from __future__ import annotations

from typing import Callable, Dict, Generic, Optional, TypeVar

T = TypeVar("T")


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

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __getitem__(self, name: str) -> T:
        return self.get(name)

    def __len__(self) -> int:
        return len(self._items)
