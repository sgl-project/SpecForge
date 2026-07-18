"""Explicit immutable catalog of SpecForge's built-in algorithms."""

from __future__ import annotations

from specforge.algorithms.dflash.providers import create_registration as dflash
from specforge.algorithms.domino.providers import create_registration as domino
from specforge.algorithms.dspark.providers import create_registration as dspark
from specforge.algorithms.eagle3.providers import create_registration as eagle3
from specforge.algorithms.peagle.providers import create_registration as peagle
from specforge.algorithms.registry import AlgorithmRegistry


def builtin_algorithm_registry() -> AlgorithmRegistry:
    """Return a fresh immutable catalog without module-level mutation."""

    return AlgorithmRegistry((eagle3(), peagle(), dflash(), domino(), dspark()))


__all__ = ["builtin_algorithm_registry"]
