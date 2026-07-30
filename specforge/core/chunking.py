# coding=utf-8
# Licensed under the Apache License, Version 2.0 (the "License").
"""Reusable chunked reductions for memory-bounded training objectives."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import torch

ChunkTerms = tuple[torch.Tensor, ...]


def checkpointed_chunk_reduce(
    function: Callable[..., ChunkTerms],
    *aligned_tensors: Optional[torch.Tensor],
    chunk_size: int,
    dim: int = 0,
) -> ChunkTerms:
    """Sum additive terms over aligned tensor slices.

    ``chunk_size=0`` evaluates the full dimension once without checkpointing.
    Positive chunk sizes bound intermediates and use non-reentrant activation
    checkpointing when gradients are enabled and an input requires gradients.
    ``None`` arguments are forwarded unchanged, which keeps optional objective
    inputs aligned with the tensors that are sliced.
    """

    if chunk_size < 0:
        raise ValueError(f"chunk_size must be >= 0, got {chunk_size}")

    tensors = tuple(tensor for tensor in aligned_tensors if tensor is not None)
    if not tensors:
        raise ValueError("chunked reduction requires at least one tensor")

    first = tensors[0]
    normalized_dim = dim if dim >= 0 else first.ndim + dim
    if normalized_dim < 0 or normalized_dim >= first.ndim:
        raise ValueError(f"dim {dim} is invalid for a {first.ndim}D tensor")
    length = first.shape[normalized_dim]
    if length == 0:
        raise ValueError("chunked reduction received an empty dimension")

    for tensor in tensors[1:]:
        tensor_dim = dim if dim >= 0 else tensor.ndim + dim
        if tensor_dim < 0 or tensor_dim >= tensor.ndim:
            raise ValueError(f"dim {dim} is invalid for a {tensor.ndim}D tensor")
        if tensor.shape[tensor_dim] != length:
            raise ValueError(
                "chunked reduction inputs must be aligned: "
                f"expected dimension length {length}, got {tensor.shape[tensor_dim]}"
            )

    effective_chunk_size = chunk_size or length
    totals: Optional[ChunkTerms] = None
    for start in range(0, length, effective_chunk_size):
        width = min(effective_chunk_size, length - start)
        chunk_args = tuple(
            (
                tensor.narrow(
                    dim if dim >= 0 else tensor.ndim + dim,
                    start,
                    width,
                )
                if tensor is not None
                else None
            )
            for tensor in aligned_tensors
        )
        should_checkpoint = (
            chunk_size > 0
            and torch.is_grad_enabled()
            and any(
                tensor is not None and tensor.requires_grad for tensor in chunk_args
            )
        )
        if should_checkpoint:
            from torch.utils.checkpoint import checkpoint

            chunk_terms = checkpoint(
                function,
                *chunk_args,
                use_reentrant=False,
            )
        else:
            chunk_terms = function(*chunk_args)

        if not isinstance(chunk_terms, tuple) or not all(
            isinstance(term, torch.Tensor) for term in chunk_terms
        ):
            raise TypeError("chunk function must return a tuple of tensors")
        if totals is None:
            totals = chunk_terms
            continue
        if len(totals) != len(chunk_terms):
            raise ValueError("chunk function returned a different number of terms")
        totals = tuple(left + right for left, right in zip(totals, chunk_terms))

    assert totals is not None
    return totals


__all__ = ["checkpointed_chunk_reduce"]
