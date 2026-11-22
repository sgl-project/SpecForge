"""Exercise the HCCL ring schedule on Gloo using PyTorch for the CANN kernels."""

import os
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, skipUnless
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _scores(q, k, scale, mask, sparse_mode):
    groups = q.shape[2] // k.shape[2]
    k = k.repeat_interleave(groups, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
    if sparse_mode == 2:
        assert mask.shape == (2048, 2048) and mask.dtype == torch.bool
        scores.masked_fill_(mask[: q.shape[1], : k.shape[1]], float("-inf"))
    else:
        assert sparse_mode == 0 and mask is None
    return scores


def _fusion(q, k, v, heads, layout, *, scale, atten_mask, sparse_mode, **kwargs):
    assert layout == "BSND" and heads == q.shape[2]
    scores = _scores(q, k, scale, atten_mask, sparse_mode)
    maximum = scores.amax(dim=-1, keepdim=True)
    weights = (scores - maximum).exp()
    total = weights.sum(dim=-1, keepdim=True)
    expanded_v = v.repeat_interleave(heads // v.shape[2], dim=2)
    out = torch.einsum("bhqk,bkhd->bqhd", weights / total, expanded_v)
    return out, maximum.expand(-1, -1, -1, 8), total.expand(-1, -1, -1, 8)


def _fusion_grad(
    q,
    k,
    v,
    dy,
    heads,
    layout,
    *,
    scale_value,
    atten_mask,
    sparse_mode,
    softmax_max,
    softmax_sum,
    attention_in,
    **kwargs,
):
    assert softmax_max.shape == softmax_sum.shape == (q.shape[0], heads, q.shape[1], 8)
    assert softmax_max.is_contiguous() and softmax_sum.is_contiguous()
    scores = _scores(q, k, scale_value, atten_mask, sparse_mode)
    probability = (scores - softmax_max[..., :1]).exp() / softmax_sum[..., :1]
    groups = heads // k.shape[2]
    expanded_k = k.repeat_interleave(groups, dim=2)
    expanded_v = v.repeat_interleave(groups, dim=2)
    dprob = torch.einsum("bqhd,bkhd->bhqk", dy, expanded_v)
    dscore = probability * (
        dprob - (dy * attention_in).sum(-1).transpose(1, 2).unsqueeze(-1)
    )
    dq = torch.einsum("bhqk,bkhd->bqhd", dscore * scale_value, expanded_k)
    dk = torch.einsum("bhqk,bqhd->bkhd", dscore * scale_value, q)
    dv = torch.einsum("bhqk,bqhd->bkhd", probability, dy)
    return (
        dq,
        dk.unflatten(2, (-1, groups)).sum(3),
        dv.unflatten(2, (-1, groups)).sum(3),
    )


def fake_npu_kernels():
    return patch.dict(
        "sys.modules",
        torch_npu=SimpleNamespace(
            npu_fusion_attention=_fusion,
            npu_fusion_attention_grad=_fusion_grad,
        ),
    )


def _compare(group, *, kv_heads, depth, device="cpu"):
    from specforge.modeling.draft.layers.attention.backends.ascend_ring import (
        ring_cached_attention,
    )

    rank, size = dist.get_rank(group), dist.get_world_size(group)
    torch.manual_seed(811)
    hardware = device != "cpu"
    dtype = torch.bfloat16 if hardware else torch.float32
    chunk, width = (33, 64) if hardware else (3, 8)
    batch, length, heads = 2, size * chunk, 4

    def sample(num_heads):
        return (
            torch.randn(batch, length, num_heads, width)
            .to(dtype)
            .float()
            .requires_grad_()
        )

    q = sample(heads)
    keys = [sample(kv_heads) for _ in range(depth)]
    values = [sample(kv_heads) for _ in range(depth)]
    # Use an asymmetric upstream gradient to catch gradients returned to the wrong owner.
    upstream = torch.randn_like(q).to(dtype).float()
    upstream[1, -2:] = 0.0
    groups = heads // kv_heads
    k = torch.cat(keys, 1).repeat_interleave(groups, 2)
    v = torch.cat(values, 1).repeat_interleave(groups, 2)
    score = torch.einsum("bqhd,bkhd->bhqk", q, k) * width**-0.5
    index = torch.arange(length)
    mask = torch.cat(
        [index[:, None] >= index[None, :]]
        + [index[:, None] == index[None, :]] * (depth - 1),
        1,
    )
    full = torch.einsum(
        "bhqk,bkhd->bqhd", score.masked_fill(~mask, -torch.inf).softmax(-1), v
    )
    full.backward(upstream)
    shard = slice(rank * chunk, (rank + 1) * chunk)
    local = [
        t[:, shard].detach().to(device=device, dtype=dtype).requires_grad_()
        for t in (q, *keys, *values)
    ]
    with nullcontext() if hardware else fake_npu_kernels():
        out = ring_cached_attention(
            local[0], local[1 : depth + 1], local[depth + 1 :], group, width**-0.5
        )
        out.backward(upstream[:, shard].to(device=device, dtype=dtype))
    atol, rtol = (0.02, 0.04) if hardware else (2e-6, 2e-5)
    actual = [out, *[x.grad for x in local]]
    expected = [full[:, shard], *[x.grad[:, shard] for x in (q, *keys, *values)]]
    errors = []
    for result, reference in zip(actual, expected):
        result = result.detach().float().cpu()
        errors.append((result - reference).abs().max().item())
        torch.testing.assert_close(result, reference, atol=atol, rtol=rtol)
    if hardware:
        print(
            f"PASS rank={rank} kv_heads={kv_heads} depth={depth} "
            f"max_abs_output={errors[0]:.6g} max_abs_dq={errors[1]:.6g} "
            f"max_abs_dk={max(errors[2:2+depth]):.6g} max_abs_dv={max(errors[2+depth:]):.6g}",
            flush=True,
        )


def _worker(rank, init_method):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=45),
    )
    try:
        noncontiguous = dist.new_group([0, 2])
        singleton = dist.new_group([1])
        _compare(dist.group.WORLD, kv_heads=2, depth=3)
        if rank in (0, 2):
            _compare(noncontiguous, kv_heads=1, depth=1)
        else:
            _compare(singleton, kv_heads=4, depth=2)
    finally:
        dist.destroy_process_group()


@skipUnless(dist.is_gloo_available(), "requires Gloo")
class AscendRingTest(TestCase):
    def test_causal_ring_and_recurrent_diagonal_gradients(self):
        with TemporaryDirectory() as directory:
            mp.spawn(_worker, (Path(directory, "rdzv").as_uri(),), nprocs=3, join=True)


if __name__ == "__main__":
    # Direct hardware check: torchrun --standalone --nproc-per-node=2 this_file.py
    import torch_npu  # noqa: F401

    device = f"npu:{os.environ['LOCAL_RANK']}"
    torch.npu.set_device(device)
    dist.init_process_group("hccl")
    try:
        for kv_heads, depth in [(4, 1), (2, 3), (1, 2)]:
            _compare(dist.group.WORLD, kv_heads=kv_heads, depth=depth, device=device)
    finally:
        dist.destroy_process_group()
