"""Exercise canonical loading, data ports, trainer checkpoints and replay."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import save_file
from transformers import Qwen3Config

from specforge.application.composition import build_application_run, resolve_run
from specforge.config import Config
from specforge.config.draft.codec import encode_config
from specforge.loading import load_draft
from specforge.runtime.contracts import TrainBatch
from specforge.training.assembly import build_model_bundle
from specforge.training.strategies.base import StepContext
from specforge.training.strategies.canonical import CanonicalTrainStrategy
from tests.test_algorithms.test_objectives import configurations, training_batch
from tests.test_config.draft_fixtures import draft_config
from tests.test_modeling.test_canonical_eagle3 import model_config as eagle_config
from tests.test_modeling.test_canonical_peagle import model_config as peagle_config


def target_directory(root):
    target = root / "target"
    Qwen3Config(
        hidden_size=32,
        vocab_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
    ).save_pretrained(target)
    save_file(
        {
            "model.embed_tokens.weight": torch.randn(64, 32),
            "lm_head.weight": torch.randn(64, 32),
        },
        target / "model.safetensors",
    )
    return target


def run_config(target, draft, **updates):
    raw = dict(
        model={"target_model_path": str(target), "torch_dtype": "float32"},
        data={
            "hidden_states_path": str(target.parent / "features"),
            "dataloader_num_workers": 0,
        },
        training={"attention_backend": "eager", "max_steps": 1},
        draft=encode_config(draft),
        output_dir=str(target.parent / "output"),
    )
    raw.update(updates)
    return Config.model_validate(raw)


class CanonicalTrainingTest(unittest.TestCase):
    def test_training_parses_with_target_before_building_the_model(self):
        from unittest.mock import patch

        from specforge.training.model_loading import training_target_config

        with TemporaryDirectory() as temp:
            target = target_directory(Path(temp))
            config = json.loads((target / "config.json").read_text())
            config["logits_mup_width_multiplier"] = 8
            (target / "config.json").write_text(json.dumps(config))
            cfg = run_config(target, draft_config())
            del cfg.draft["target_logit_transform"]
            resolved = resolve_run(cfg, target_config=training_target_config(cfg))
            self.assertEqual(
                resolved.config.draft["target_logit_transform"],
                {"type": "mup", "multiplier": 8},
            )
            bundle = build_model_bundle(resolved.config, algorithm=resolved.algorithm)
            self.assertEqual(bundle.draft_config.target_logit_transform.multiplier, 8)
            resolved.config.draft["spec_config"]["target_features"][
                "target_layer_ids"
            ] = [99]
            with patch("specforge.loading.initialize_draft") as initialize:
                with self.assertRaisesRegex(ValueError, "target_layer_ids"):
                    build_model_bundle(resolved.config, algorithm=resolved.algorithm)
                initialize.assert_not_called()

    def test_all_architectures_resolve_and_assemble(self):
        with TemporaryDirectory() as temp:
            target = target_directory(Path(temp))
            configs = configurations()[:-2] + [
                eagle_config(target_width=32),
                peagle_config(target_width=32),
            ]
            for draft in configs:
                with self.subTest(architecture=draft.architecture):
                    cfg = run_config(target, draft)
                    if draft.architecture == "PEagleDraftModel":
                        cfg.training.objective_options = {"mask_token_id": 0}
                    resolved = resolve_run(cfg)
                    bundle = build_model_bundle(
                        resolved.config, algorithm=resolved.algorithm
                    )
                    self.assertEqual(bundle.draft_config, draft)
                    self.assertIs(bundle.model.draft_model, bundle.draft_model)
                    layout = resolved.algorithm.providers.offline_for(
                        "text"
                    ).capture_layout
                    self.assertEqual(
                        layout is None, draft.architecture == "DMTPDraftModel"
                    )
                    self.assertEqual(
                        bundle.strategy_kwargs.resume_contract["canonical_objective"],
                        bundle.model.objective_name,
                    )

    @unittest.skipUnless(torch.cuda.is_available(), "FSDP integration requires CUDA")
    def test_existing_trainer_saves_loadable_model_and_resumes(self):
        for tied in (False, True):
            with self.subTest(tied=tied):
                self._train_and_resume(tied)

    def _train_and_resume(self, tied):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            owns_group = not torch.distributed.is_initialized()
            if owns_group:
                torch.distributed.init_process_group(
                    "nccl" if torch.cuda.is_available() else "gloo",
                    init_method=f"file://{root / 'rendezvous'}",
                    rank=0,
                    world_size=1,
                )
                self.addCleanup(torch.distributed.destroy_process_group)
            target = target_directory(root)
            draft = draft_config(
                tie_word_embeddings=tied,
                **(
                    {"target_bindings": (), "target_logit_transform": None}
                    if tied
                    else {}
                ),
            )
            features = root / "features"
            features.mkdir()
            for index in range(2):
                batch = training_batch(draft)
                torch.save(
                    {
                        key: batch[key][0].detach()
                        for key in ("input_ids", "loss_mask", "hidden_states")
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
            first = build_application_run(cfg)
            first.run()
            checkpoint = root / "output" / "specforge-run-step1"
            self.assertTrue((checkpoint / "config.json").exists())
            model = load_draft(checkpoint, target=target, dtype=torch.float32)
            before = {
                name: tensor.clone() for name, tensor in model.state_dict().items()
            }
            raw = cfg.model_dump()
            raw.update(draft=None, resume=str(checkpoint))
            raw["training"]["max_steps"] = 2
            second = build_application_run(Config.model_validate(raw))
            second.run()
            final = root / "output" / "specforge-run-step2"
            state = torch.load(
                final / "training_state.pt", map_location="cpu", weights_only=True
            )
            self.assertEqual(state["global_step"], 2)
            if tied:
                self.assertNotIn("lm_head.weight", state["draft_state_dict"])
                self.assertIs(model.embed_tokens.weight, model.lm_head.weight)
            self.assertTrue(
                any(
                    not torch.equal(before[name], tensor)
                    for name, tensor in state["draft_state_dict"].items()
                )
            )
            raw["draft"] = encode_config(
                draft_config(hidden_size=32, layer_types=("full_dense",))
            )
            with self.assertRaisesRegex(ValueError, "exactly match"):
                resolve_run(Config.model_validate(raw))

    def test_dspine_replay_uses_window_ce_and_restores_rng(self):
        from specforge.algorithms.objectives import build_objective_model
        from tests.test_modeling.test_canonical_dspine import initialized_model

        model = initialized_model()
        runtime = build_objective_model(
            model,
            options={"num_anchors": 2, "alignment_warmup_steps": 0},
            target_head=torch.nn.Linear(32, 64, bias=False),
        )
        strategy = CanonicalTrainStrategy(
            runtime,
            name="dspine",
            objective="dspine",
            required_features=runtime.registration.inputs,
        )
        batch = TrainBatch(["a"], "dspine", training_batch(model.config.draft))
        output = strategy.forward_loss(
            batch, StepContext(global_step=1, total_steps=10, accumulation_steps=2)
        )
        state = torch.get_rng_state().clone()
        replayed = output.replay_loss(output.ratio_metrics)
        torch.testing.assert_close(state, torch.get_rng_state())
        torch.testing.assert_close(output.loss, replayed.loss)
        replayed.loss.backward()
        self.assertTrue(
            all(p.grad is not None for p in model.parameters() if p.requires_grad)
        )

    def test_reject_mixed_sources_and_backend(self):
        draft = draft_config()
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            run_config(
                Path("target"), draft, initialization={"checkpoint": "a"}, resume="b"
            )
        cfg = run_config(
            Path("target"), draft, training={"attention_backend": "flex_attention"}
        )
        with self.assertRaisesRegex(ValueError, "attention_backend"):
            resolve_run(cfg)
