# coding=utf-8
"""An MoE DSpark draft through the canonical offline trainer runtime on two GPUs, FSDP and DDP."""

import math
import os
import tempfile
import unittest

import torch

from specforge.algorithms.builtin import builtin_algorithm_registry

CUDA = torch.cuda.is_available()
NGPU = torch.cuda.device_count() if CUDA else 0
WORLD = 2
ALGORITHM = builtin_algorithm_registry().resolve("dspark")
MOE_DRAFT = {
    "moe_preset": "qwen3_5_moe",
    "num_experts": 4,
    "num_experts_per_tok": 1,
    "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 0,
    "norm_topk_prob": False,
    "dflash_config": {
        "moe_router_aux_loss_coef": 0.01,
        "moe_router_aux_loss_type": "batch_wise",
        "moe_router_enable_expert_bias": True,
        "moe_router_bias_update_rate": 0.001,
    },
}


def _worker(rank, world_size, sharding):
    os.environ["FSDP_SHARDING"] = sharding  # the backend wraps NO_SHARD with DDP
    from tests.test_runtime import _fixtures as fx

    fx.init_rank_distributed(rank, world_size, port="29583")

    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy
    from torch.nn.parallel import DistributedDataParallel as DDP

    from specforge.launch import build_offline_runtime
    from specforge.modeling.draft.layers.ffn.moe import MoELayer
    from specforge.optimizer import BF16Optimizer

    hidden, sequence_length = 64, 12
    workdir = tempfile.mkdtemp(prefix=f"moe_offline_rank{rank}_")
    model, captured_width = fx.build_dspark(  # seeded: the same weights on every rank
        workdir,
        hidden=hidden,
        block_size=4,
        num_anchors=2,
        attention_backend="sdpa",
        draft_overrides=MOE_DRAFT,
    )
    feature_dir = fx.write_offline_files_dspark(
        os.path.join(workdir, "features"),
        n=8,  # four samples per rank: two optimizer steps of two micro-batches
        seq=sequence_length,
        captured_width=captured_width,
        target_hidden=hidden,
    )
    logged = []
    trainer = build_offline_runtime(
        algorithm=ALGORITHM,
        hidden_states_path=feature_dir,
        draft_model=model,
        target_head=None,
        optimizer_factory=lambda module: BF16Optimizer(
            module, lr=1e-3, max_grad_norm=0.5, warmup_ratio=0.0, total_steps=2
        ),
        run_id="moe-offline",
        output_dir=os.path.join(workdir, "out"),
        max_len=sequence_length,
        batch_size=1,
        accumulation_steps=2,
        num_epochs=1,
        max_steps=2,
        total_steps=2,
        logger=lambda metrics, _step: logged.append(metrics),
        log_interval=1,
    )

    layers = [m for m in model.draft_model.modules() if isinstance(m, MoELayer)]
    assert len(layers) == 1  # the preset replaced the dense MLP
    module = trainer.core.strategy.trainable_module()
    if sharding == "NO_SHARD":
        assert isinstance(module, DDP)
    else:
        assert isinstance(module, FSDP)
        # Two ranks shard for real; one rank silently falls back to NO_SHARD.
        assert module.sharding_strategy is ShardingStrategy.SHARD_GRAD_OP
    assert trainer.fit() == 2
    assert all(torch.isfinite(p).all() for p in module.parameters())
    # The registered hook ran: both metrics reached the log, the counts were reset.
    assert len(logged) == 2
    for name in ("moe/aux_loss", "moe/step_balance"):
        assert math.isfinite(logged[-1][name]), name
    assert torch.all(layers[0].tokens_per_expert == 0)
    # Every rank applied the same pooled bias update, under DDP without buffer sync.
    gate = layers[0].gate
    assert torch.isfinite(gate.expert_bias).all()
    gathered = [torch.empty_like(gate.expert_bias) for _ in range(world_size)]
    dist.all_gather(gathered, gate.expert_bias)
    assert all(torch.equal(other, gate.expert_bias) for other in gathered)
    dist.barrier()
    dist.destroy_process_group()


@unittest.skipUnless(CUDA and NGPU >= WORLD, f"requires >={WORLD} GPUs")
class TestMoEDraftLaunch(unittest.TestCase):
    def test_moe_dspark_trains_and_balances_from_precomputed_features(self):
        import torch.multiprocessing as mp

        for sharding in ("SHARD_GRAD_OP", "NO_SHARD"):  # FSDP, then DDP
            with self.subTest(sharding=sharding):
                mp.spawn(_worker, args=(WORLD, sharding), nprocs=WORLD, join=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
