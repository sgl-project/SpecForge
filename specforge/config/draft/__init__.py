"""Immutable draft structure, independent of model and training implementations.

Nodes validate their own fields; `validate` checks the composed graph, and the
codecs live in `codec`.
"""

from .model import DraftConfig

__all__ = ["DraftConfig"]
