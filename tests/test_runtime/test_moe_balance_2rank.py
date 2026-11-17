# coding=utf-8
"""Two ranks pool their counts: the batch-wise load, the bias update and the metric agree."""

import unittest

import torch

CUDA = torch.cuda.is_available()
NGPU = torch.cuda.device_count() if CUDA else 0
WORLD = 2
E, H, T = 8, 16, 24


def _worker(rank, world_size):
    from tests.test_runtime import _fixtures as fx

    fx.init_rank_distributed(rank, world_size, port="29584")

    import torch.distributed as dist

    from specforge.modeling.draft.moe.balance import MoEBalancing, dispatch_balance
    from specforge.modeling.draft.moe.experts import RoutedExperts
    from specforge.modeling.draft.moe.layer import MoELayer, load_balance_loss
    from specforge.modeling.draft.moe.router import TopKRouter

    torch.manual_seed(0)  # the same weights on every rank
    gate = TopKRouter(
        E, H, 1, norm_topk_prob=False, initializer_range=0.5, expert_bias=True
    )
    experts = RoutedExperts(E, H, 8, initializer_range=0.5)
    layer = MoELayer(gate, experts, aux_loss_coef=0.01).cuda()  # batch-wise load
    torch.manual_seed(100 + rank)  # different tokens on each rank
    x = torch.randn(T, H, device="cuda")
    layer(x)  # the batch-wise loss pools both ranks' counts inside the forward
    got = layer.aux_loss
    pooled = layer.tokens_per_expert.clone()
    dist.all_reduce(pooled)
    pooled = pooled.float()
    scores, _, _ = gate(x)
    torch.testing.assert_close(got, load_balance_loss(scores, pooled))

    metrics = MoEBalancing(layer).on_step()  # one real all-reduce, then the bias moves
    delta = 0.001 * torch.sign(pooled.mean() - pooled)
    torch.testing.assert_close(gate.expert_bias, delta - delta.mean())
    assert torch.all(layer.tokens_per_expert == 0)
    torch.testing.assert_close(metrics["moe/step_balance"], dispatch_balance(pooled))
    gathered = [torch.empty_like(gate.expert_bias) for _ in range(world_size)]
    dist.all_gather(gathered, gate.expert_bias)
    for other in gathered:
        assert torch.equal(other, gate.expert_bias), "ranks disagree on the bias"
    dist.barrier()
    dist.destroy_process_group()


@unittest.skipUnless(CUDA and NGPU >= WORLD, f"requires >={WORLD} GPUs")
class TestMoEBalanceTwoRanks(unittest.TestCase):
    def test_pooled_counts_drive_the_load_the_bias_and_the_step_metric(self):
        import torch.multiprocessing as mp

        mp.spawn(_worker, args=(WORLD,), nprocs=WORLD, join=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
