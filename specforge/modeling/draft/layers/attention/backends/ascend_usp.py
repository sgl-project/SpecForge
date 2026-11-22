"""Ulysses sequence parallelism for EAGLE's recurrent attention on Ascend."""

from math import gcd

import torch
import torch.distributed as dist


def _exchange(tensor, group, scatter_dim, gather_dim):
    """Send one slice to each peer and concatenate received slices in rank order."""
    peers = dist.get_world_size(group)
    if peers == 1:
        return tensor
    # HCCL all_to_all_single sends contiguous, equally sized rank-major chunks.
    send = (
        tensor.unflatten(scatter_dim, (peers, -1)).movedim(scatter_dim, 0).contiguous()
    )
    received = torch.empty_like(send)
    dist.all_to_all_single(received, send, group=group)
    return received.movedim(0, gather_dim).flatten(gather_dim, gather_dim + 1)


class _UlyssesExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, group, scatter_dim, gather_dim):
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        return _exchange(tensor, group, scatter_dim, gather_dim)

    @staticmethod
    def backward(ctx, grad):
        # All-to-all is a permutation. Its adjoint reverses the two axes,
        # without the world-size scaling used by replicated losses.
        return (
            _exchange(grad, ctx.group, ctx.gather_dim, ctx.scatter_dim),
            None,
            None,
            None,
        )


def ascend_usp_attention(query, key, value, keys, values, *, ulysses_group, ring_group):
    """Compute cached attention for projected [B,H,S,D] sequence shards.

    Cached keys and values stay sequence-gathered and head-sharded [B,S,H,D]
    across rollout depths. Only the current depth needs a forward exchange.
    """
    from .ascend_ring import ring_cached_attention

    if ring_group is None or ulysses_group is None:
        raise ValueError("USP requires explicit ring and Ulysses process groups")
    degree = dist.get_world_size(ulysses_group)
    q_heads, kv_heads = query.shape[1], key.shape[1]
    if q_heads % degree or q_heads % kv_heads:
        raise ValueError(
            f"query heads ({q_heads}) must be divisible by the Ulysses size "
            f"({degree}) and the number of KV heads ({kv_heads})"
        )

    # Replicate only as far as the smallest KV head count divisible by the
    # Ulysses size. Each head shard then contains whole GQA groups, even for MQA.
    copies = degree // gcd(degree, kv_heads)
    if copies != 1:
        key = key.repeat_interleave(copies, dim=1)
        value = value.repeat_interleave(copies, dim=1)

    query, key, value = (
        _UlyssesExchange.apply(t.transpose(1, 2), ulysses_group, 2, 1)
        for t in (query, key, value)
    )
    output = ring_cached_attention(
        query, (*keys, key), (*values, value), ring_group, query.shape[-1] ** -0.5
    )
    output = _UlyssesExchange.apply(output, ulysses_group, 1, 2)
    return output.transpose(1, 2), key, value
