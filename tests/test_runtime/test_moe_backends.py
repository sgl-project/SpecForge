# coding=utf-8
"""Gate: an MoE DFlash draft trains through both backends, FSDP (SHARD_GRAD_OP)
and DDP (NO_SHARD), on a 2-rank DP group with DISTINCT per-rank micro-batches.
Checks what the wrappers must not break: the router centering's running mean is
all-reduced (equal on both ranks, fp32 under FSDP's buffer casting), the aux loss
reaches the optimizer step, and the gathered state folds for export.
GPU-only (>=2 GPUs)."""

import json
import os
import tempfile
import unittest

import torch

CUDA = torch.cuda.is_available()
NGPU = torch.cuda.device_count() if CUDA else 0
WORLD = 2
HIDDEN = 64
MOE = {
    "moe_preset": "qwen3_5_moe",
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 48,
    "norm_topk_prob": False,
    "router_aux_loss_coef": 0.01,
    "dflash_config": {"mask_token_id": 0, "moe_router_centering": True},
}


def _draft_keys(state):
    """The draft's own key names, on CPU as a saved checkpoint would hand them over."""
    return {
        key.split("draft_model.", 1)[1]: value.detach().cpu()
        for key, value in state.items()
        if "draft_model." in key
    }


def _worker(rank, world_size, sharding, workdir, results_dir):
    os.environ["FSDP_SHARDING"] = sharding
    from tests.test_runtime import _fixtures as fx

    fx.init_rank_distributed(
        rank, world_size, port="29584" if sharding == "NO_SHARD" else "29585"
    )
    from specforge.modeling.draft.moe import MoELayer
    from specforge.modeling.draft.moe.util import fold_router_centering
    from specforge.optimizer import BF16Optimizer
    from specforge.runtime.contracts import TrainBatch
    from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
    from specforge.training.controller import TrainerCore
    from specforge.training.strategies.base import DFlashTrainStrategy

    # Same seed inside build_dflash on every rank: identical initial weights.
    model, width, _, _ = fx.build_dflash(
        os.path.join(workdir, f"rank{rank}"),
        hidden=HIDDEN,
        block_size=4,
        num_anchors=8,
        attention_backend="sdpa",
        draft_config_overrides=MOE,
    )
    vocab = int(model.draft_model.config.vocab_size)
    generator = torch.Generator().manual_seed(100 + rank)  # distinct data per rank
    seq = 64
    tensors = {
        "input_ids": torch.randint(0, vocab, (2, seq), generator=generator).cuda(),
        "loss_mask": (torch.rand(2, seq, generator=generator) > 0.2).long().cuda(),
        "hidden_states": torch.randn(2, seq, width, generator=generator)
        .to(torch.bfloat16)
        .cuda(),
    }
    batch = TrainBatch(
        sample_ids=[f"r{rank}a", f"r{rank}b"],
        strategy="dflash",
        tensors=tensors,
        metadata={},
    )

    def optimizer_factory(module):
        return BF16Optimizer(
            module, lr=1e-3, max_grad_norm=0.5, warmup_ratio=0.0, total_steps=4
        )

    backend = FSDPTrainingBackend(
        ParallelConfig.from_distributed(), optimizer_factory=optimizer_factory
    )
    backend.prepare_model(model, wrap=True, optimizer_target=model.draft_model)
    moe_layers = [m for m in backend.module.modules() if isinstance(m, MoELayer)]
    strategy = DFlashTrainStrategy(backend.module)
    core = TrainerCore(strategy, backend, accumulation_steps=2)
    reports = [core.train_step(batch) for _ in range(4)]
    boundary = [r for r in reports if r.optimizer_stepped]
    metrics = boundary[-1].metrics
    running_mean = moe_layers[0].centering.running_mean

    # state_dict() is collective; the gathered draft weights land on rank 0 only
    # (other ranks still see the replicated frozen target tables).
    draft_state = _draft_keys(backend.state_dict()["model"])
    fold = None
    if "layers.0.mlp.gate.weight" in draft_state:
        # Like export_to_hf: fold into a fresh, unwrapped draft built from the config.
        from specforge.modeling.draft.dflash import DFlashDraftModel

        export_model = DFlashDraftModel(model.draft_model.config)
        folded = fold_router_centering(export_model, draft_state)
        bias = folded["layers.0.mlp.gate.bias"]
        fold = {
            "has_centering_keys": any("centering" in key for key in folded),
            "bias_finite": bool(torch.isfinite(bias).all()),
            "bias_dtype": str(bias.dtype),
        }
    with open(os.path.join(results_dir, f"rank{rank}.json"), "w") as f:
        json.dump(
            {
                "wrapper": backend._wrapper_kind,
                "optimizer_steps": len(boundary),
                "loss": boundary[-1].loss,
                "grad_norm": boundary[-1].grad_norm,
                "aux_loss": metrics.get("moe/aux_loss"),
                "running_mean_dtype": str(running_mean.dtype),
                "running_mean": running_mean.float().cpu().tolist(),
                "fold": fold,
            },
            f,
        )


@unittest.skipUnless(CUDA and NGPU >= WORLD, f"requires >={WORLD} GPUs")
class TestMoEDraftBackends(unittest.TestCase):
    def _run(self, sharding):
        import torch.multiprocessing as mp

        workdir = tempfile.mkdtemp(prefix=f"moe_{sharding.lower()}_")
        results_dir = os.path.join(workdir, "results")
        os.makedirs(results_dir, exist_ok=True)
        mp.spawn(
            _worker,
            args=(WORLD, sharding, workdir, results_dir),
            nprocs=WORLD,
            join=True,
        )
        results = []
        for rank in range(WORLD):
            with open(os.path.join(results_dir, f"rank{rank}.json")) as f:
                results.append(json.load(f))
        return results

    def _check(self, results, wrapper):
        for rank, res in enumerate(results):
            self.assertEqual(res["wrapper"], wrapper, msg=f"rank {rank}")
            self.assertEqual(res["optimizer_steps"], 2, msg=f"rank {rank}")
            self.assertTrue(
                torch.isfinite(torch.tensor(res["loss"])), msg=f"rank {rank}"
            )
            self.assertTrue(
                torch.isfinite(torch.tensor(res["grad_norm"])), msg=f"rank {rank}"
            )
            self.assertIsNotNone(res["aux_loss"], msg=f"rank {rank}: aux loss missing")
            self.assertGreater(res["aux_loss"], 0.0, msg=f"rank {rank}")
            self.assertEqual(
                res["running_mean_dtype"], "torch.float32", msg=f"rank {rank}"
            )
        means = [torch.tensor(res["running_mean"]) for res in results]
        # Distinct batches per rank: only the all-reduce makes the means agree.
        self.assertGreater(
            float(means[0].abs().max()), 0.0, "running mean never updated"
        )
        self.assertLess(
            float((means[0] - means[1]).abs().max()),
            1e-6,
            "running mean diverged across ranks",
        )
        folds = [res["fold"] for res in results if res["fold"] is not None]
        self.assertGreaterEqual(len(folds), 1, "no rank saw the gathered full weights")
        for fold in folds:
            self.assertFalse(fold["has_centering_keys"])
            self.assertTrue(fold["bias_finite"])
            self.assertEqual(fold["bias_dtype"], "torch.float32")

    def test_fsdp_shard_grad_op(self):
        self._check(self._run("SHARD_GRAD_OP"), "fsdp")

    def test_ddp_no_shard(self):
        self._check(self._run("NO_SHARD"), "ddp")


if __name__ == "__main__":
    unittest.main(verbosity=2)
