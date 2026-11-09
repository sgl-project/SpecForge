"""Typed rotary-position configuration, with no tensor or HF dependency."""

from .base import RopeSpec
from .default import DefaultRoPESpec
from .dynamic import DynamicNTKRoPESpec
from .linear import LinearRoPESpec
from .llama3 import Llama3RoPESpec
from .multiaxis import MultiAxisRoPESpec
from .none import NoRoPESpec
from .yarn import YaRNSpec

__all__ = [
    "RopeSpec",
    "DefaultRoPESpec",
    "DynamicNTKRoPESpec",
    "LinearRoPESpec",
    "Llama3RoPESpec",
    "MultiAxisRoPESpec",
    "NoRoPESpec",
    "YaRNSpec",
]
