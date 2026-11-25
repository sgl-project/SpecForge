"""Ascend EAGLE3 USP attention over contiguous sequence shards.

Only the causal prefix K/V travels around the ring. Every later cache entry
is visible to the query at the same position and stays on its owning rank.
"""

from functools import lru_cache

import torch

from specforge.layers.ring.utils import RingComm


def _load_torch_npu():
    try:
        import torch_npu
    except ImportError as exc:
        raise ImportError(
            "Ascend USP attention requires a compatible torch-npu installation"
        ) from exc
    return torch_npu


@lru_cache(maxsize=8)
def _causal_mask(device):
    # CANN's left-up causal mode consumes this fixed compressed mask even for
    # sequences longer than 2048. A blocking copy makes the cached tensor
    # available to subsequent calls on other training streams.
    return torch.ones(2048, 2048, dtype=torch.bool).triu_(1).to(device)


def _circulate_kv(key, value, comm):
    """Post the next exchange before yielding the current block for computation."""
    if comm.world_size == 1:
        yield 0, key, value
        return
    block = torch.stack((key, value))
    for step in range(comm.world_size):
        another = step + 1 < comm.world_size
        if another:
            incoming = comm.send_recv(block)
            comm.commit()
        yield step, block[0], block[1]
        if another:
            comm.wait()
            block = incoming


def _merge(out, maximum, total, block_out, block_max, block_sum):
    """Merge row-wise softmax statistics and the corresponding normalized output."""
    new_max = torch.maximum(maximum, block_max)
    left = total * (maximum - new_max).exp()
    right = block_sum * (block_max - new_max).exp()
    new_sum = left + right
    return (out * (left / new_sum) + block_out * (right / new_sum), new_max, new_sum)


def _prefix_forward(q, k, v, group, scale, mask):
    torch_npu = _load_torch_npu()

    comm = RingComm(group)
    for step, block_k, block_v in _circulate_kv(k, v, comm):
        # The first block is local. After 'rank' rotations the stream wraps
        # around to future positions, so only forward those remaining blocks.
        if step > comm.rank:
            continue
        causal = step == 0
        block_out, block_max, block_sum = torch_npu.npu_fusion_attention(
            q,
            block_k,
            block_v,
            q.shape[2],
            "BSND",
            atten_mask=mask if causal else None,
            sparse_mode=2 if causal else 0,
            scale=scale,
            keep_prob=1.0,
        )[:3]
        # CANN repeats each row statistic eight times in [B,H,S,8].
        block_out = block_out.float()
        block_max = block_max[..., :1].transpose(1, 2)
        block_sum = block_sum[..., :1].transpose(1, 2)
        if causal:
            out, maximum, total = block_out, block_max, block_sum
        else:
            out, maximum, total = _merge(
                out, maximum, total, block_out, block_max, block_sum
            )
    return out, maximum, total


def _prefix_backward(q, k, v, grad, out, maximum, total, group, scale, mask):
    torch_npu = _load_torch_npu()

    kv_comm, grad_comm = RingComm(group), RingComm(group)
    dq = torch.zeros_like(q, dtype=torch.float32)
    # CANN backward takes global row statistics and the globally normalized
    # output, including the local recurrent branches. It then produces exactly
    # the gradient contribution of this K/V block to the full softmax.
    maximum = maximum.transpose(1, 2).expand(-1, -1, -1, 8).contiguous()
    total = total.transpose(1, 2).expand_as(maximum).contiguous()
    out = out.to(q.dtype).contiguous()
    for step, block_k, block_v in _circulate_kv(k, v, kv_comm):
        visible = step <= kv_comm.rank
        if visible:
            causal = step == 0
            block_dq, dk, dv = torch_npu.npu_fusion_attention_grad(
                q,
                block_k,
                block_v,
                grad,
                q.shape[2],
                "BSND",
                atten_mask=mask if causal else None,
                sparse_mode=2 if causal else 0,
                scale_value=scale,
                keep_prob=1.0,
                attention_in=out,
                softmax_max=maximum,
                softmax_sum=total,
            )[:3]
            dq.add_(block_dq)
        if step == 0:
            dkv = torch.stack((dk, dv)).float()
        else:
            # These gradients followed their K/V owner one hop behind the data.
            grad_comm.wait()
            dkv = incoming_grad
            if visible:
                dkv[0].add_(dk)
                dkv[1].add_(dv)
        if grad_comm.world_size > 1:
            incoming_grad = grad_comm.send_recv(dkv)
            grad_comm.commit()
    if grad_comm.world_size > 1:
        # The final rotation returns each accumulated gradient to its owner.
        grad_comm.wait()
        dkv = incoming_grad
    return dq, dkv[0].to(k.dtype), dkv[1].to(v.dtype)


class _RingCachedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, scale, query, *cache):
        depth = len(cache) // 2
        keys, values = cache[:depth], cache[depth:]
        mask = _causal_mask(query.device)
        out, maximum, total = _prefix_forward(
            query, keys[0], values[0], group, scale, mask
        )

        # Group adjacent query heads by KV head. Broadcasting each diagonal
        # [B,S,Hkv,1,D] avoids replicating the cache or constructing an SxS mask.
        heads = (keys[0].shape[2], -1)
        grouped_q = query.unflatten(2, heads)
        out, maximum, total = [x.unflatten(2, heads) for x in (out, maximum, total)]
        for key, value in zip(keys[1:], values[1:]):
            score = (grouped_q.float() * key.float().unsqueeze(3)).sum(
                -1, keepdim=True
            ) * scale
            out, maximum, total = _merge(
                out, maximum, total, value.float().unsqueeze(3), score, 1.0
            )
        out, maximum, total = [x.flatten(2, 3) for x in (out, maximum, total)]
        ctx.group, ctx.scale = group, scale
        ctx.save_for_backward(query, out, maximum, total, *cache)
        return out.to(query.dtype)

    @staticmethod
    def backward(ctx, grad):
        query, out, maximum, total, *cache = ctx.saved_tensors
        depth = len(cache) // 2
        keys, values = cache[:depth], cache[depth:]
        grad = grad.contiguous()
        dq, dk, dv = _prefix_backward(
            query,
            keys[0],
            values[0],
            grad,
            out,
            maximum,
            total,
            ctx.group,
            ctx.scale,
            _causal_mask(query.device),
        )
        dkeys, dvalues = [dk], [dv]
        heads = (keys[0].shape[2], -1)
        q, grad, out, maximum, total, dq = [
            x.unflatten(2, heads) for x in (query, grad, out, maximum, total, dq)
        ]
        for key, value in zip(keys[1:], values[1:]):
            k = key.float().unsqueeze(3)
            score = (q.float() * k).sum(-1, keepdim=True) * ctx.scale
            weight = (score - maximum).exp() / total
            dscore = weight * (grad.float() * (value.float().unsqueeze(3) - out)).sum(
                -1, keepdim=True
            )
            dq.add_(dscore * ctx.scale * k)
            dkeys.append((dscore * ctx.scale * q.float()).sum(3).to(key.dtype))
            dvalues.append((weight * grad.float()).sum(3).to(value.dtype))
        return None, None, dq.flatten(2, 3).to(query.dtype), *dkeys, *dvalues


def ring_cached_attention(query, keys, values, group, scale):
    """Attend to the causal base and same-anchor rollout cache in BSND layout.

    Ranks in the supplied group own consecutive, equal-length sequence blocks.
    Callers zero inactive query outputs for right-padded sequences.
    """
    if not keys or len(keys) != len(values):
        raise ValueError("expected a matching non-empty EAGLE key/value cache")
    if query.shape[2] % keys[0].shape[2]:
        raise ValueError(
            "the number of query heads must be divisible by the KV head count"
        )
    return _RingCachedAttention.apply(group, scale, query.contiguous(), *keys, *values)
