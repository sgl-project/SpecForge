"""A MoE draft trains through the canonical trainer, checkpoints its bias, and resumes."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from specforge.application.composition import build_application_run
from specforge.config import Config
from specforge.config.draft.layer import LayerSpec
from specforge.config.draft.mlp import DenseFFNSpec
from specforge.loading import load_draft
from specforge.modeling.draft.layers.mlp import EXPERT_KERNELS, MoEFFN
from tests.test_algorithms.test_objectives import training_batch
from tests.test_config.draft_fixtures import draft_config
from tests.test_modeling.test_draft_moe_ffn import expert_spec, moe_spec, router_spec
from tests.test_runtime.test_canonical_training import run_config, target_directory


def moe_draft_config(intermediate_size=16):
    node = moe_spec(
        experts=expert_spec(intermediate_size),
        shared_expert=expert_spec(16),
        shared_expert_gate=True,
        router=router_spec(expert_bias=True),
    )
    return draft_config(
        mlp_configs={
            "dense": DenseFFNSpec(intermediate_size=64, hidden_act="silu"),
            "moe": node,
        },
        layer_configs={
            "full_dense": LayerSpec(attention="full", mlp="dense"),
            "full_moe": LayerSpec(attention="full", mlp="moe"),
        },
        layer_types=("full_dense", "full_moe"),
    )


def write_features(root, draft):
    features = root / "features"
    features.mkdir()
    for index in range(2):
        batch = training_batch(draft)
        torch.save(
            {
                key: batch[key][0].detach()
                for key in (
                    "input_ids",
                    "loss_mask",
                    "hidden_states",
                    "target_last_hidden_states",
                )
            },
            features / f"{index}.ckpt",
        )


def moe_run_config(target, draft, kernel, bias_update_rate=0.0, aux_loss_coef=0.0):
    return run_config(
        target,
        draft,
        model={
            "target_model_path": str(target),
            "torch_dtype": "float32",
            "moe_experts_kernel": kernel,
        },
        training={
            "attention_backend": "eager",
            "max_steps": 1,
            "total_steps": 2,
            "save_interval": 1,
            "log_interval": 1,
            "objective_options": {
                "num_anchors": 2,
                "bias_update_rate": bias_update_rate,
                "aux_loss_coef": aux_loss_coef,
                "aux_loss_type": "batch_wise",
            },
        },
    )


@unittest.skipUnless(torch.cuda.is_available(), "FSDP integration requires CUDA")
class CanonicalMoETrainingTest(unittest.TestCase):
    def setUp(self):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                "nccl",
                init_method="tcp://127.0.0.1:29571",
                rank=0,
                world_size=1,
            )
            self.addCleanup(torch.distributed.destroy_process_group)

    def _train_and_resume(self, root, kernel, bias_update_rate=0.0, aux_loss_coef=0.0):
        target = target_directory(root)
        draft = moe_draft_config()
        write_features(root, draft)
        cfg = moe_run_config(target, draft, kernel, bias_update_rate, aux_loss_coef)
        build_application_run(cfg).run()
        checkpoint = root / "output" / "specforge-run-step1"
        model = load_draft(checkpoint, target=target, dtype=torch.float32)
        (layer,) = [m for m in model.modules() if isinstance(m, MoEFFN)]
        bias = layer.gate.expert_bias
        self.assertEqual(bias.dtype, torch.float32)
        if bias_update_rate:
            # One optimizer step moved the bias by a zero-mean sign step; it is saved.
            self.assertAlmostEqual(bias.abs().max().item(), bias_update_rate, places=7)
            torch.testing.assert_close(bias.sum(), torch.zeros(()), atol=1e-6, rtol=0)
        else:
            self.assertEqual(torch.count_nonzero(bias).item(), 0)
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        raw = cfg.model_dump()
        raw.update(draft=None, resume=str(checkpoint))
        raw["training"]["max_steps"] = 2
        build_application_run(Config.model_validate(raw)).run()
        final = root / "output" / "specforge-run-step2"
        state = torch.load(
            final / "training_state.pt", map_location="cpu", weights_only=True
        )
        self.assertEqual(state["global_step"], 2)
        self.assertEqual(state["canonical_moe_experts_kernel"], kernel)
        trained = state["draft_state_dict"]
        for suffix in ("gate.weight", "experts.gate_up_proj", "experts.down_proj"):
            name = next(key for key in trained if key.endswith(suffix))
            self.assertFalse(torch.equal(before[name], trained[name]), name)
            self.assertTrue(torch.isfinite(trained[name]).all(), name)
        bias_key = next(key for key in trained if key.endswith("gate.expert_bias"))
        if bias_update_rate:
            # The resumed step moved the restored bias again.
            self.assertFalse(torch.equal(before[bias_key], trained[bias_key]))
            self.assertEqual(trained[bias_key].dtype, torch.float32)
        else:
            self.assertEqual(torch.count_nonzero(trained[bias_key]).item(), 0)
        # The kernel is part of the resume contract, like the Liger choice.
        other = next(name for name in EXPERT_KERNELS if name != kernel)
        raw["model"]["moe_experts_kernel"] = other
        raw["training"]["max_steps"] = 3
        with self.assertRaisesRegex(ValueError, "canonical_moe_experts_kernel"):
            build_application_run(Config.model_validate(raw)).run()

    def test_moe_draft_trains_saves_and_resumes_with_each_kernel(self):
        for kernel in sorted(EXPERT_KERNELS):
            with self.subTest(kernel=kernel), TemporaryDirectory() as temp:
                self._train_and_resume(Path(temp), kernel)

    def test_bias_update_rate_moves_the_bias_across_steps_and_resume(self):
        with TemporaryDirectory() as temp:
            self._train_and_resume(
                Path(temp), "grouped_mm", bias_update_rate=0.001, aux_loss_coef=0.01
            )

    def test_misaligned_widths_fail_at_startup_unless_the_loop_kernel_is_chosen(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            target = target_directory(root)
            draft = moe_draft_config(intermediate_size=18)
            write_features(root, draft)
            with self.assertRaisesRegex(
                ValueError, "cannot run a MoE layer .*intermediate_size 18"
            ):
                build_application_run(moe_run_config(target, draft, "grouped_mm")).run()
            with self.assertRaisesRegex(ValueError, "unknown moe_experts_kernel"):
                build_application_run(moe_run_config(target, draft, "triton")).run()
            build_application_run(moe_run_config(target, draft, "loop")).run()
            self.assertTrue((root / "output" / "specforge-run-step1").is_dir())


if __name__ == "__main__":
    unittest.main()
