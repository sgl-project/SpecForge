"""A MoE draft trains through the canonical trainer, checkpoints its bias, and resumes."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from specforge.application.composition import build_application_run
from specforge.config import Config
from specforge.config.draft.layer import LayerSpec
from specforge.config.draft.mlp import DenseFFNSpec, MoESharedExpertSpec
from specforge.loading import load_draft
from specforge.modeling.draft.layers.mlp import MoEFFN
from tests.test_algorithms.test_objectives import training_batch
from tests.test_config.draft_fixtures import draft_config
from tests.test_modeling.test_draft_moe_ffn import moe_spec, router_spec
from tests.test_runtime.test_canonical_training import run_config, target_directory


def moe_draft_config():
    node = moe_spec(
        intermediate_size=16,
        shared_expert=MoESharedExpertSpec(
            intermediate_size=16, hidden_act="silu", gated=True
        ),
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


@unittest.skipUnless(torch.cuda.is_available(), "FSDP integration requires CUDA")
class CanonicalMoETrainingTest(unittest.TestCase):
    def test_moe_draft_trains_saves_and_resumes(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(
                    "nccl",
                    init_method=f"file://{root / 'rendezvous'}",
                    rank=0,
                    world_size=1,
                )
                self.addCleanup(torch.distributed.destroy_process_group)
            target = target_directory(root)
            draft = moe_draft_config()
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
            cfg = run_config(
                target,
                draft,
                training={
                    "attention_backend": "eager",
                    "max_steps": 1,
                    "total_steps": 2,
                    "save_interval": 1,
                    "log_interval": 1,
                    "objective_options": {"num_anchors": 2},
                },
            )
            build_application_run(cfg).run()
            checkpoint = root / "output" / "specforge-run-step1"
            model = load_draft(checkpoint, target=target, dtype=torch.float32)
            (layer,) = [m for m in model.modules() if isinstance(m, MoEFFN)]
            self.assertEqual(layer.gate.expert_bias.dtype, torch.float32)
            # No balancing hook yet: the bias is saved, restored and still zero.
            self.assertEqual(torch.count_nonzero(layer.gate.expert_bias).item(), 0)
            before = {
                name: tensor.clone() for name, tensor in model.state_dict().items()
            }
            raw = cfg.model_dump()
            raw.update(draft=None, resume=str(checkpoint))
            raw["training"]["max_steps"] = 2
            build_application_run(Config.model_validate(raw)).run()
            final = root / "output" / "specforge-run-step2"
            state = torch.load(
                final / "training_state.pt", map_location="cpu", weights_only=True
            )
            self.assertEqual(state["global_step"], 2)
            trained = state["draft_state_dict"]
            for suffix in ("gate.weight", "experts.gate_up_proj", "experts.down_proj"):
                name = next(key for key in trained if key.endswith(suffix))
                self.assertFalse(torch.equal(before[name], trained[name]), name)
                self.assertTrue(torch.isfinite(trained[name]).all(), name)


if __name__ == "__main__":
    unittest.main()
