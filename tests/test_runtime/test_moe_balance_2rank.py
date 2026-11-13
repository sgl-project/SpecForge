"""Two ranks pool their counts: one all-reduce moves the same bias; the aux loss and its router gradients match a reference."""

import unittest

import torch

CUDA = torch.cuda.is_available()
NGPU = torch.cuda.device_count() if CUDA else 0
WORLD = 2
RATE = 0.001
COEF = 0.5


def _worker(rank, world_size):
    from tests.test_runtime import _fixtures as fx

    fx.init_rank_distributed(rank, world_size, port="29584")

    import torch.distributed as dist
    import torch.nn.functional as F
    from torch import nn

    from specforge.algorithms.objectives import build_objective_model
    from specforge.algorithms.objectives.base import ObjectiveResult
    from specforge.algorithms.objectives.moe_balance import (
        AUX_LOSS_TYPES,
        dispatch_balance,
    )
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

    # The auxiliary loss: two micro-batches per rank, rank 1 keeping fewer blocks, both
    # modes; loss and router gradients against DeepSeek-V3's E / (K T) * sum_e count_e P_e.
    E, K = layer.config.num_experts, layer.config.num_experts_per_tok
    head = nn.Linear(32, config.vocab_size, bias=False).cuda()
    zero, one = torch.zeros((), device="cuda"), torch.ones((), device="cuda")
    for aux_loss_type in AUX_LOSS_TYPES:
        balanced = build_objective_model(
            model,
            options={
                "num_anchors": 2,
                "aux_loss_coef": COEF,
                "aux_loss_type": aux_loss_type,
            },
            target_head=head,
        )
        balancer, seen = balanced.balancer, []
        observe = balancer.observe

        def record(output, *, keep):  # the routing and keep mask each forward hands in
            seen.append((output.expert_routing[0], keep))
            return observe(output, keep=keep)

        balancer.observe = record
        step_counts = torch.zeros(E, device="cuda")
        step_kept = torch.zeros((), device="cuda")
        names = ("balancer", "reference")
        grads = {name: torch.zeros_like(layer.gate.weight) for name in names}
        kept_per_batch = []
        for micro in range(2):
            torch.manual_seed(1000 * micro + rank)
            batch = training_batch(config, device="cuda")
            if rank == 1:  # one valid anchor only: fewer kept blocks than rank 0
                batch["loss_mask"] = torch.tensor(
                    [[0, 0, 1, 1, 0, 0, 0, 0, 0]], device="cuda"
                )
            result = balanced(batch, step=0, total_steps=2)
            finished = balancer.finish(
                ObjectiveResult(zero, one, {}, result.balance),
                pooled=True,
                process_group=dist.group.WORLD,
            )
            routing, keep = seen[-1]
            kept = keep.sum().float()
            kept_per_batch.append(kept.clone())
            counts = torch.bincount(
                routing.expert_ids[keep].reshape(-1), minlength=E
            ).float()
            probs = F.normalize(routing.scores, p=1, dim=-1)[keep].mean(0)
            if aux_loss_type == "batch_wise":  # the step's load so far, over ranks
                dist.all_reduce(counts)
                dist.all_reduce(kept)
                step_counts, step_kept = step_counts + counts, step_kept + kept
                counts, kept = step_counts, step_kept
                assert torch.equal(
                    balancer.pooled[0].float(), counts
                ), "pooled load differs"
            reference = COEF * E / (K * kept) * (counts * probs).sum()
            torch.testing.assert_close(finished.loss, reference)
            (from_balancer,) = torch.autograd.grad(
                finished.loss, layer.gate.weight, retain_graph=True
            )
            (from_reference,) = torch.autograd.grad(reference, layer.gate.weight)
            torch.testing.assert_close(from_balancer, from_reference)
            grads["balancer"] += from_balancer
            grads["reference"] += from_reference
        kept_counts = [torch.empty_like(kept_per_batch[0]) for _ in range(world_size)]
        dist.all_gather(kept_counts, kept_per_batch[0])
        assert kept_counts[0] > kept_counts[1], "ranks kept the same number of blocks"
        for accumulated in grads.values():  # the data-parallel gradient reduction
            dist.all_reduce(accumulated)
        torch.testing.assert_close(grads["balancer"], grads["reference"])
        assert torch.any(grads["balancer"] != 0), "no router gradient from the aux loss"
        reduced = [torch.empty_like(grads["balancer"]) for _ in range(world_size)]
        dist.all_gather(reduced, grads["balancer"])
        assert all(torch.equal(other, grads["balancer"]) for other in reduced)
        assert set(balancer.step()) == {"moe/step_balance"} and balancer.pooled is None
    dist.barrier()
    dist.destroy_process_group()


@unittest.skipUnless(CUDA and NGPU >= WORLD, f"requires >={WORLD} GPUs")
class MoEBalanceTwoRankTest(unittest.TestCase):
    def test_two_ranks_pool_the_bias_update_and_the_aux_loss_gradients(self):
        import torch.multiprocessing as mp

        mp.spawn(_worker, args=(WORLD,), nprocs=WORLD, join=True)


if __name__ == "__main__":
    unittest.main()
