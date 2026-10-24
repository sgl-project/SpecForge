# coding=utf-8
"""An MoE DSpark draft through the canonical offline trainer runtime on one GPU."""

import math
import os
import tempfile
import unittest

import torch

from specforge.algorithms.builtin import builtin_algorithm_registry

CUDA = torch.cuda.is_available()
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
        "moe_router_selection_bias": True,
        "moe_router_bias_update_rate": 0.001,
    },
}


@unittest.skipUnless(CUDA, "MoE offline launcher requires CUDA")
class TestMoEOfflineLaunch(unittest.TestCase):
    def test_moe_dspark_trains_and_balances_from_precomputed_features(self):
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29583")

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        from specforge.launch import build_offline_runtime
        from specforge.modeling.draft.moe import MoELayer
        from specforge.optimizer import BF16Optimizer

        hidden, sequence_length = 64, 12
        workdir = tempfile.mkdtemp(prefix="moe_offline_")
        model, captured_width = fx.build_dspark(
            workdir,
            hidden=hidden,
            block_size=4,
            num_anchors=2,
            attention_backend="sdpa",
            draft_overrides=MOE_DRAFT,
        )
        feature_dir = fx.write_offline_files_dspark(
            os.path.join(workdir, "features"),
            n=4,
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
            num_epochs=1,
            max_steps=2,
            total_steps=2,
            logger=lambda metrics, _step: logged.append(metrics),
            log_interval=1,
        )

        layers = [m for m in model.draft_model.modules() if isinstance(m, MoELayer)]
        self.assertEqual(len(layers), 1)  # the preset replaced the dense MLP
        module = trainer.core.strategy.trainable_module()
        self.assertIsInstance(module, FSDP)
        self.assertEqual(trainer.fit(), 2)
        self.assertTrue(all(torch.isfinite(p).all() for p in module.parameters()))
        # Both balancing metrics reached the step's log, and the step hook ran.
        self.assertEqual(len(logged), 2)
        for name in ("moe/aux_loss", "moe/step_balance"):
            self.assertTrue(math.isfinite(logged[-1][name]), name)
        gate = layers[0].gate
        self.assertTrue(torch.isfinite(gate.selection_bias).all())
        self.assertTrue(torch.all(gate.tokens_per_expert == 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
