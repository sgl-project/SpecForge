# coding=utf-8
"""Two ranks pool their counts: the bias, batch-wise load and step metric agree."""

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
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
    )

    from specforge.modeling.draft.moe.balance import (
        dispatch_balance,
        load_balance_loss,
        step_moe_balancing,
    )
    from specforge.modeling.draft.moe.families.qwen3_5_moe import build_moe

    torch.manual_seed(0)  # the same weights on every rank
    config = Qwen3_5MoeTextConfig(
        num_experts=E,
        num_experts_per_tok=1,
        hidden_size=H,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=0,
        norm_topk_prob=False,
        dflash_config={
            "moe_router_aux_loss_coef": 0.01,
            "moe_router_aux_loss_type": "batch_wise",
            "moe_router_selection_bias": True,
        },
    )
    layer = build_moe(config).cuda()
    gate = layer.gate
    torch.manual_seed(100 + rank)  # different tokens on each rank
    x = torch.randn(T, H, device="cuda")

    layer(x)  # the batch-wise loss pools both ranks' counts inside the forward
    got = gate.aux_loss
    pooled = gate.tokens_per_expert.clone()
    dist.all_reduce(pooled)
    scores, _, _ = gate.eval()(x)
    gate.train()
    torch.testing.assert_close(got, load_balance_loss(scores, pooled))

    metrics = step_moe_balancing(layer)  # one real all-reduce, then the bias moves
    delta = 0.001 * torch.sign(pooled.mean() - pooled)
    torch.testing.assert_close(gate.selection_bias, delta - delta.mean())
    assert torch.all(gate.tokens_per_expert == 0)
    torch.testing.assert_close(metrics["moe/step_balance"], dispatch_balance(pooled))
    gathered = [torch.empty_like(gate.selection_bias) for _ in range(world_size)]
    dist.all_gather(gathered, gate.selection_bias)
    for other in gathered:
        assert torch.equal(other, gate.selection_bias), "ranks disagree on the bias"
    dist.barrier()
    dist.destroy_process_group()


@unittest.skipUnless(CUDA and NGPU >= WORLD, f"requires >={WORLD} GPUs")
class TestMoEBalanceTwoRanks(unittest.TestCase):
    def test_pooled_counts_drive_the_bias_the_load_and_the_step_metric(self):
        import torch.multiprocessing as mp

        mp.spawn(_worker, args=(WORLD,), nprocs=WORLD, join=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
