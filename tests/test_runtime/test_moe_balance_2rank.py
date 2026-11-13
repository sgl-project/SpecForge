"""Two ranks pool their routed-token counts: one all-reduce moves the same bias on both."""

import unittest

import torch

CUDA = torch.cuda.is_available()
NGPU = torch.cuda.device_count() if CUDA else 0
WORLD = 2
RATE = 0.001


def _worker(rank, world_size):
    from tests.test_runtime import _fixtures as fx

    fx.init_rank_distributed(rank, world_size, port="29584")

    import torch.distributed as dist
    from torch import nn

    from specforge.algorithms.objectives import build_objective_model
    from specforge.algorithms.objectives.moe_balance import dispatch_balance
    from specforge.modeling import build_draft_model
    from specforge.modeling.draft.layers.mlp import MoEFFN
    from tests.test_algorithms.test_objectives import training_batch
    from tests.test_modeling.test_canonical_moe import moe_model_config

    config = moe_model_config()
    torch.manual_seed(0)  # the same weights on every rank
    model = build_draft_model(config).cuda()
    runtime = build_objective_model(
        model,
        options={"num_anchors": 2, "bias_update_rate": RATE},
        target_head=nn.Linear(32, config.vocab_size, bias=False).cuda(),
    )
    (layer,) = [m for m in model.modules() if isinstance(m, MoEFFN)]
    torch.manual_seed(100 + rank)  # different tokens on each rank
    batch = training_batch(config, device="cuda")
    runtime(batch, step=0, total_steps=2).loss.backward()
    (counts,) = runtime.balancer.counts
    gathered = [torch.empty_like(counts) for _ in range(world_size)]
    dist.all_gather(gathered, counts)
    assert not torch.equal(gathered[0], gathered[1]), "ranks routed alike"
    pooled = torch.stack(gathered).sum(0)

    metrics = runtime.balancer.step(process_group=dist.group.WORLD)  # one all-reduce
    load = pooled.float()
    delta = RATE * torch.sign(load.mean() - load)
    torch.testing.assert_close(layer.gate.expert_bias, delta - delta.mean())
    torch.testing.assert_close(metrics["moe/step_balance"], dispatch_balance(pooled))
    assert runtime.balancer.counts == [None]
    biases = [torch.empty_like(layer.gate.expert_bias) for _ in range(world_size)]
    dist.all_gather(biases, layer.gate.expert_bias)
    for other in biases:
        assert torch.equal(other, layer.gate.expert_bias), "ranks disagree on the bias"
    dist.barrier()
    dist.destroy_process_group()


@unittest.skipUnless(CUDA and NGPU >= WORLD, f"requires >={WORLD} GPUs")
class MoEBalanceTwoRankTest(unittest.TestCase):
    def test_pooled_counts_move_the_same_bias_on_every_rank(self):
        import torch.multiprocessing as mp

        mp.spawn(_worker, args=(WORLD,), nprocs=WORLD, join=True)


if __name__ == "__main__":
    unittest.main()
