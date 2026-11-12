"""Regression gates for canonical entry, evaluation and checkpoint publication."""

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from specforge.application.composition import resolve_run
from specforge.config import Config, apply_overrides
from specforge.config.draft.codec import encode_config
from specforge.training.canonical import capture_layers
from specforge.training.checkpoint import CheckpointManager
from specforge.training.strategies.canonical import CanonicalTrainStrategy
from tests.test_config.test_draft_mtp_spec import mtp_config
from tests.test_runtime.test_canonical_training import (
    draft_config,
    run_config,
    target_directory,
)


class CanonicalReviewTest(unittest.TestCase):
    def test_ddp_train_save_resume(self):
        from tests.test_runtime.test_canonical_training import CanonicalTrainingTest

        case = CanonicalTrainingTest()
        try:
            with patch.dict(os.environ, {"FSDP_SHARDING": "NO_SHARD"}):
                case._train_and_resume(False)
        finally:
            case.doCleanups()

    def test_depth_metrics_and_runtime_eval_mode(self):
        from specforge.algorithms.objectives import objective_registrations
        from specforge.eval import Evaluator
        from tests.test_runtime.test_evaluator_aggregation import _batch

        runtime = torch.nn.Module()
        runtime.registration = objective_registrations()["eagle3"]
        runtime.draft_model = torch.nn.Linear(1, 1)
        ratios = {
            "position_1_accuracy": (torch.tensor(1.0), torch.tensor(2.0)),
            "position_0_accuracy": (torch.tensor(3.0), torch.tensor(4.0)),
        }
        result = SimpleNamespace(
            loss=torch.tensor(1.0),
            ratios=ratios,
            numerator=torch.tensor(1.0),
            denominator=torch.tensor(1.0),
        )
        strategy = CanonicalTrainStrategy(
            runtime, name="eagle3", objective="eagle3", required_features=()
        )
        strategy.trainable_module().eval()
        self.assertFalse(runtime.training)
        self.assertFalse(runtime.draft_model.training)
        with patch.object(runtime, "forward", return_value=result):
            metrics = Evaluator().run(strategy.forward_loss, [_batch()])
        self.assertAlmostEqual(metrics["eval/avg_acc"], 0.75)
        self.assertAlmostEqual(metrics["eval/simulated_acc_len"], 1.125)
        strategy.trainable_module().train()
        self.assertTrue(runtime.training)

    def test_config_is_published_before_complete_state(self):
        with TemporaryDirectory() as temp:
            manager = CheckpointManager(temp, "run")
            path = Path(manager.checkpoint_dir(1))
            original = manager._atomic_save

            def save(state, destination):
                self.assertEqual(
                    json.loads((path / "config.json").read_text()), {"a": 1}
                )
                return original(state, destination)

            with patch.object(manager, "_atomic_save", side_effect=save):
                manager.save({"canonical_draft_config": {"a": 1}}, 1)
            with patch.object(
                manager, "_atomic_json", side_effect=OSError("interrupted")
            ):
                with self.assertRaises(Exception):
                    manager.save({"canonical_draft_config": {"a": 2}}, 2)
            self.assertFalse(
                (Path(manager.checkpoint_dir(2)) / "training_state.pt").exists()
            )

    def test_nested_target_and_final_hidden_capture(self):
        from transformers import Qwen3_5Config, Qwen3_5TextConfig

        from specforge.training.assembly import build_model_bundle

        with TemporaryDirectory() as temp:
            target = target_directory(Path(temp))
            config = Qwen3_5Config(
                text_config=Qwen3_5TextConfig(
                    hidden_size=32,
                    vocab_size=64,
                    num_hidden_layers=8,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    head_dim=8,
                    intermediate_size=64,
                    tie_word_embeddings=False,
                ).to_dict()
            )
            config.save_pretrained(target)
            cfg = run_config(
                target,
                draft_config(),
                training={"attention_backend": "eager", "loss": {"kind": "kl"}},
            )
            resolved = resolve_run(cfg)
            bundle = build_model_bundle(resolved.config, algorithm=resolved.algorithm)
            self.assertIs(bundle.target_head, bundle.draft_model.lm_head)
            self.assertEqual(
                capture_layers(cfg, mtp_config(), config.text_config.to_dict()), [0]
            )

    def test_unused_settings_fail_and_resume_override_is_independent(self):
        with TemporaryDirectory() as temp:
            target = target_directory(Path(temp))
            cfg = run_config(target, draft_config())
            for name, value in (
                ("objective_chunk_blocks", 1),
                ("objective_checkpointing", False),
                ("fused_logits_loss", False),
                ("mtp_objective_chunk_size", 1),
            ):
                changed = cfg.model_copy(deep=True)
                setattr(changed.training, name, value)
                with self.assertRaisesRegex(ValueError, name):
                    resolve_run(changed)
            cfg.training.objective_chunk_blocks = 128
            with self.assertLogs(
                "specforge.training.canonical", level="WARNING"
            ) as logs:
                resolve_run(cfg)
            self.assertIn("do not enable", logs.output[0])
            for name in ("embedding_key", "lm_head_key"):
                owned = run_config(
                    target,
                    draft_config(target_bindings=(), target_logit_transform=None),
                )
                owned.model.load_target_embedding = False
                setattr(owned.model, name, "unused.weight")
                with self.assertRaisesRegex(ValueError, name):
                    resolve_run(owned)
            cfg.model.vocab_mapping_path = "unused.pt"
            with self.assertRaisesRegex(ValueError, "vocab_mapping_path"):
                resolve_run(cfg)
            raw = run_config(target, draft_config()).model_dump()
            raw.update(draft=None, resume="first")
            cfg = apply_overrides(Config.model_validate(raw), ["resume=second"])
            self.assertEqual(cfg.resume_path, "second")
            self.assertIsNone(cfg.training.resume_from)
            self.assertIsNone(
                Config.model_validate(cfg.model_dump() | {"resume": None}).resume_path
            )
            path = Path(temp) / "checkpoint"
            path.mkdir()
            (path / "config.json").write_text(json.dumps(encode_config(draft_config())))
            torch.save({}, path / "training_state.pt")
            raw.update(resume=None)
            raw["training"]["resume_from"] = str(path)
            self.assertEqual(
                resolve_run(Config.model_validate(raw)).config.draft,
                encode_config(draft_config()),
            )

    def test_legacy_dspine_requires_explicit_matching_training_settings(self):
        from specforge.algorithms.objectives import resolve_objective
        from specforge.training.canonical import _validate_imported_objective
        from tests.test_config.test_draft_legacy_dspine import dspine_payload

        with TemporaryDirectory() as temp:
            source = Path(temp)
            payload = dspine_payload()
            payload["dspine_config"]["ce_weight"] = 0.3
            (source / "config.json").write_text(json.dumps(payload))
            cfg = run_config(
                source, draft_config(), initialization={"checkpoint": temp}
            )
            with self.assertRaisesRegex(ValueError, "objective_options.ce_weight"):
                _validate_imported_objective(cfg, resolve_objective("DSpineDraftModel"))
            _validate_imported_objective(
                cfg, resolve_objective("DSpineDraftModel", options={"ce_weight": 0.3})
            )

    def test_scratch_embedding_key_survives_resume(self):
        """A scratch run that needed model.embedding_key must resume with it."""
        from safetensors.torch import save_file

        from specforge.application.composition import build_application_run
        from tests.test_algorithms.test_objectives import training_batch

        with TemporaryDirectory() as temp:
            root = Path(temp)
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(
                    "gloo",
                    init_method=f"file://{root / 'rendezvous'}",
                    rank=0,
                    world_size=1,
                )
                self.addCleanup(torch.distributed.destroy_process_group)
            target = target_directory(root)
            save_file(
                {
                    "transformer.wte.weight": torch.randn(64, 32),
                    "lm_head.weight": torch.randn(64, 32),
                },
                target / "model.safetensors",
            )
            draft = draft_config(target_bindings=(), target_logit_transform=None)
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
                model={
                    "target_model_path": str(target),
                    "torch_dtype": "float32",
                    "embedding_key": "transformer.wte.weight",
                },
                training={
                    "attention_backend": "eager",
                    "max_steps": 1,
                    "total_steps": 2,
                    "save_interval": 1,
                    "objective_options": {"num_anchors": 2},
                },
            )
            raw = cfg.model_dump()
            raw.update(draft=None, resume=str(root / "output" / "specforge-run-step1"))
            raw["training"]["max_steps"] = 2
            with patch.dict(os.environ, {"FSDP_SHARDING": "NO_SHARD"}):
                build_application_run(cfg).run()
                build_application_run(Config.model_validate(raw)).run()
            self.assertTrue((root / "output" / "specforge-run-step2").is_dir())
