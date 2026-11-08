"""Immutable draft structure, independent of model and training implementations.

Nodes validate their own fields. Codecs and cross-node validation are separate
from these types; the existing model/configuration loaders are unchanged.
"""

from .model import DraftConfig

__all__ = ["DraftConfig"]
