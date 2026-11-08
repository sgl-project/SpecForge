"""Compute prepared query/key/value tensors without owning learned projections."""

from .eager import eager_attention

__all__ = ["eager_attention"]
