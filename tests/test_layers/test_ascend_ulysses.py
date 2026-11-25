"""Exercise Ascend's sequence/head exchange and its adjoint with real collectives."""

from datetime import timedelta
from math import lcm
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from tests.test_layers.test_ascend_ring import fake_npu_kernels


def _check_ulysses(rank, world_size, store):
    from specforge.distributed import _native_usp_groups
    from specforge.modeling.draft.layers.attention.backends.ascend_usp import (
        ascend_usp_attention,
    )

    dist.init_process_group(
        "gloo",
        init_method=store,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.set_num_threads(1)
        # Consecutive pairs own independent samples; ring peers are strided
        # inside *one* sample when both parallel dimensions are active.
        draft_sp, _ = dist.new_subgroups_by_enumeration([[0, 1], [2, 3]])
        ulysses, ring = _native_usp_groups(2, 1, draft_sp)
        assert dist.get_process_group_ranks(ulysses) == [
            rank // 2 * 2,
            rank // 2 * 2 + 1,
        ]
        assert dist.get_process_group_ranks(ring) == [rank]
        u2, r2 = _native_usp_groups(2, 2, dist.group.WORLD)
        assert dist.get_process_group_ranks(u2) == dist.get_process_group_ranks(ulysses)
        assert dist.get_process_group_ranks(r2) == [rank % 2, rank % 2 + 2]

        # Vary both GQA ratio and whether the KV heads need replication.
        for q_heads, kv_heads, u_group, r_group in (
            (4, 1, ulysses, ring),
            (4, 2, ulysses, ring),
            (6, 3, ulysses, ring),
            (4, 4, u2, r2),
        ):
            size = dist.get_world_size(u_group) * dist.get_world_size(r_group)
            torch.manual_seed(100 + rank // size)
            tensors = [
                torch.randn(2, heads, 3 * size, 8, requires_grad=True)
                for heads in (q_heads, kv_heads, kv_heads)
            ]
            owned = slice(rank % size * 3, (rank % size + 1) * 3)
            local = [
                tensor[:, :, owned].detach().requires_grad_() for tensor in tensors
            ]
            q, k, v = tensors
            expected = F.scaled_dot_product_attention(
                q,
                k.repeat_interleave(q_heads // kv_heads, dim=1),
                v.repeat_interleave(q_heads // kv_heads, dim=1),
                is_causal=True,
            )
            weight = torch.randn_like(expected)
            with fake_npu_kernels():
                actual, cache_k, cache_v = ascend_usp_attention(
                    *local, (), (), ulysses_group=u_group, ring_group=r_group
                )
                torch.testing.assert_close(actual, expected[:, :, owned])
                assert (
                    cache_k.shape == cache_v.shape == (2, 6, lcm(2, kv_heads) // 2, 8)
                )

                # Each shard owns only its local loss. The inverse exchanges
                # must deliver every query's gradients to the original KV rank.
                actual.backward(weight[:, :, owned])
            expected.backward(weight)
            for shard, full in zip(local, tensors):
                torch.testing.assert_close(shard.grad, full.grad[:, :, owned])
    finally:
        dist.destroy_process_group()


@skipUnless(dist.is_gloo_available(), "requires Gloo")
class AscendUlyssesTest(TestCase):
    def test_ulysses_attention_and_gradients(self):
        with TemporaryDirectory() as directory:
            mp.spawn(
                _check_ulysses,
                args=(4, Path(directory, "rdzv").as_uri()),
                nprocs=4,
                join=True,
            )
