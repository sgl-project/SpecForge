# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Task contract and registry for ``specforge benchmark``.

A *task* pairs a dataset with a prompt format and, optionally, a scorer.  Adding
one means subclassing :class:`BenchmarkTask`, implementing :meth:`load`, and
decorating the class with ``@TASKS.register``.  The runner handles requests,
timing, concurrency, and metrics; tasks never talk to the server.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Iterable, Iterator, List, Optional, Type


@dataclass
class Sample:
    """One conversation to run: user turns plus an optional ground truth."""

    #: User messages, one per turn.  Multi-turn tasks list several.
    turns: List[str]
    #: Ground truth passed back to :meth:`BenchmarkTask.score`; ``None`` when
    #: the task is not scored.
    label: Any = None
    #: Local image path for multimodal tasks.
    image: Optional[str] = None


class BenchmarkTask(ABC):
    """A dataset, a prompt format, and an optional scorer.

    Class attributes describe the task; subclasses override the ones that
    differ from the defaults.
    """

    #: Registry name, also used on the CLI.
    name: ClassVar[str]
    #: One-line description for ``--list-tasks``.
    description: ClassVar[str] = ""
    #: Default generation cap; the config can override it per task.
    max_new_tokens: ClassVar[int] = 2048
    #: Stop strings sent with every request.
    stop: ClassVar[Optional[List[str]]] = None
    #: System prompt prepended to every conversation.
    system_prompt: ClassVar[Optional[str]] = None
    #: Whether ``subset`` is accepted.  Tasks that accept subsets validate
    #: the names themselves inside :meth:`load`.
    supports_subsets: ClassVar[bool] = False

    def __init__(
        self,
        num_samples: Optional[int] = None,
        subset: Optional[List[str]] = None,
        seed: int = 42,
    ):
        if subset and not self.supports_subsets:
            raise ValueError(f"task {self.name!r} does not support subsets")
        self.num_samples = num_samples
        self.subset = subset
        self.seed = seed

    @abstractmethod
    def load(self) -> Iterable[Sample]:
        """Yield samples in dataset order.

        Use :meth:`limit` around the source rows so tasks with expensive
        per-row preparation stop early when ``num_samples`` is set.
        """

    def score(self, output: str, label: Any) -> Optional[bool]:
        """Return whether the final-turn output is correct.

        ``None`` marks the sample as unscored; a task that never scores keeps
        the default.
        """
        return None

    def cleanup(self) -> None:
        """Release resources created by :meth:`load` (temp files, caches)."""

    def samples(self) -> List[Sample]:
        """Materialize :meth:`load`, honoring ``num_samples``."""
        return list(self.limit(self.load()))

    def limit(self, rows: Iterable[Any]) -> Iterator[Any]:
        return itertools.islice(rows, self.num_samples)

    @property
    def scored(self) -> bool:
        """True when the subclass overrides :meth:`score`."""
        return type(self).score is not BenchmarkTask.score


class TaskRegistry:
    """Name -> task class mapping populated by ``@TASKS.register``."""

    def __init__(self) -> None:
        self._tasks: Dict[str, Type[BenchmarkTask]] = {}

    def register(self, cls: Type[BenchmarkTask]) -> Type[BenchmarkTask]:
        name = getattr(cls, "name", None)
        if not name:
            raise ValueError(f"{cls.__name__} must define a non-empty `name`")
        existing = self._tasks.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"task {name!r} is already registered by {existing.__qualname__}"
            )
        self._tasks[name] = cls
        return cls

    def get(self, name: str) -> Type[BenchmarkTask]:
        try:
            return self._tasks[name]
        except KeyError:
            raise KeyError(
                f"unknown task {name!r}; available: {', '.join(self.names())}"
            ) from None

    def names(self) -> List[str]:
        return sorted(self._tasks)

    def __contains__(self, name: object) -> bool:
        return name in self._tasks

    def __iter__(self) -> Iterator[Type[BenchmarkTask]]:
        return iter(self._tasks[name] for name in self.names())


TASKS = TaskRegistry()


__all__ = ["BenchmarkTask", "Sample", "TaskRegistry", "TASKS"]
