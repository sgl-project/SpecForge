"""Regression gates for canonical entry, evaluation and checkpoint publication."""

import dataclasses
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
        runtime.statistics = None  # no per-forward statistics in this fake
        runtime.draft_model = torch.nn.Linear(1, 1)
        runtime.draft_model.config = SimpleNamespace(draft=None)
        ratios = {
            "acc_1": (torch.tensor(1.0), torch.tensor(2.0)),
            "acc_0": (torch.tensor(3.0), torch.tensor(4.0)),
        }
        result = SimpleNamespace(
            loss=torch.tensor(1.0),
            ratios=ratios,
            sums={},
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
                    json.loads((path / "config.json").read_text()),
                    {
                        "a": 1,
                        "specforge_artifact": {"format": "specforge", "version": 1},
                    },
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
                training={
                    "attention_backend": "eager",
                    "loss": {"terms": {"token": {"kind": "kl"}}},
                },
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
                changed = cfg.model_dump()
                changed["training"][name] = value
                with self.assertRaisesRegex(ValueError, name):
                    Config.model_validate(changed)
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
                Config.model_validate(
                    cfg.model_dump()
                    | {"resume": None, "draft": encode_config(draft_config())}
                ).resume_path
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

    def test_explicit_weighting_survives_the_second_resolution(self):
        """resolve_run writes the options back; build_bundle resolves them again."""
        from specforge.algorithms.objectives import resolve_objective
        from specforge.training.canonical import _objective_settings
        from tests.test_modeling.test_canonical_dspine import (
            model_config as dspine_config,
        )
        from tests.test_modeling.test_canonical_eagle3 import (
            model_config as eagle_config,
        )

        with TemporaryDirectory() as temp:
            target = target_directory(Path(temp))
            for config in (eagle_config(), dspine_config()):
                with self.subTest(architecture=config.architecture):
                    cfg = run_config(
                        target, config, training={"weighting": {"kind": "uniform"}}
                    )
                    run = resolve_run(cfg)
                    again = resolve_objective(
                        config.architecture, **_objective_settings(run.config)
                    )
                    self.assertEqual(again.weighting.kind, "uniform")

    def test_dspine_resolves_alignment_warmup_at_configuration_time(self):
        from dataclasses import asdict

        from specforge.algorithms.objectives import objective_record, resolve_objective
        from specforge.training.canonical import _objective_settings
        from tests.test_modeling.test_canonical_dspine import model_config

        with TemporaryDirectory() as temp:
            target = target_directory(Path(temp))
            for steps, expected in [(None, 0.5), (12, 1.0), (0, 1.0)]:
                with self.subTest(steps=steps):
                    cfg = run_config(
                        target,
                        model_config(),
                        training={
                            "warmup_ratio": 0.04,
                            "objective_options": {"alignment_warmup_steps": steps},
                        },
                    )
                    run = resolve_run(cfg)
                    objective = resolve_objective(
                        "DSpineDraftModel", **_objective_settings(run.config)
                    ).objective
                    self.assertEqual(objective.alignment_scale(12, 600), expected)
                    # Older checkpoints stored the inherited ratio in the strategy options.
                    old = asdict(objective)
                    del old["alignment_warmup_ratio"]
                    self.assertEqual(
                        objective_record(
                            "dspine", old, training={"warmup_ratio": 0.04}
                        ),
                        asdict(objective),
                    )

    def test_logging_options_are_not_part_of_the_resume_record(self):
        from specforge.algorithms.objectives import objective_record

        saved = {"num_anchors": 8, "loss_decay_gamma": None}
        for name, logged in (
            ("dflash", {"teacher_metrics": True, "metric_top_k": 4}),
            ("dflash2", {"teacher_metrics": True}),
        ):
            with self.subTest(objective=name):
                self.assertEqual(objective_record(name, {**saved, **logged}), saved)
        self.assertEqual(objective_record("dspark", saved), saved)

    def test_built_resume_contract_is_the_record_written_before_logging_options(self):
        from specforge.training.assembly import build_model_bundle

        with TemporaryDirectory() as temp:
            target = target_directory(Path(temp))
            for options in ({}, {"teacher_metrics": True, "metric_top_k": 4}):
                with self.subTest(options=options):
                    cfg = run_config(
                        target,
                        draft_config(),
                        training={
                            "attention_backend": "eager",
                            "objective_options": options,
                        },
                    )
                    resolved = resolve_run(cfg)
                    bundle = build_model_bundle(
                        resolved.config, algorithm=resolved.algorithm
                    )
                    self.assertEqual(
                        bundle.strategy_kwargs.resume_contract[
                            "canonical_objective_options"
                        ],
                        {"num_anchors": 512, "loss_decay_gamma": None},
                    )

    def test_resume_accepts_a_record_saved_with_logging_options(self):
        from tests.test_runtime import test_checkpoint_resume as resume_tests

        harness = resume_tests.TestTrainerResumeEntrypoint()
        if not torch.distributed.is_initialized():
            harness.setUpClass()
            self.addCleanup(harness.tearDownClass)
        trained = {"num_anchors": 8, "loss_decay_gamma": None}

        def contract(options):
            return {
                "canonical_objective": "dflash",
                "canonical_objective_options": options,
            }

        with TemporaryDirectory() as temp:
            root = Path(temp)
            features = resume_tests._write_feature_files(str(root / "features"), n=4)
            trainer, _, _ = harness._make_trainer(
                str(root / "run"),
                feat_dir=features,
                max_steps=1,
                checkpoint_extra=contract(
                    {**trained, "teacher_metrics": True, "metric_top_k": 4}
                ),
            )
            trainer.fit()
            source = str((root / "run" / "rz-latest").resolve())
            harness._make_trainer(
                str(root / "run"),
                feat_dir=features,
                max_steps=2,
                checkpoint_extra=contract(trained),
                resume_from=source,
            )
            # the options that fix the trajectory are still compared
            with self.assertRaisesRegex(ValueError, "canonical_objective_options"):
                harness._make_trainer(
                    str(root / "run"),
                    feat_dir=features,
                    max_steps=2,
                    checkpoint_extra=contract({**trained, "num_anchors": 16}),
                    resume_from=source,
                )

    def test_legacy_compact_teacher_is_not_silently_accepted(self):
        from tests.test_modeling.test_canonical_eagle3 import (
            model_config as eagle_config,
        )
        from tests.test_modeling.test_canonical_peagle import (
            model_config as peagle_config,
        )

        with TemporaryDirectory() as temp:
            target = target_directory(Path(temp))
            for config in (eagle_config(), peagle_config()):
                with self.subTest(architecture=config.architecture):
                    cfg = run_config(target, config)
                    cfg.training.batch_size = 1
                    raw = cfg.model_dump()
                    raw["training"]["compact_teacher"] = True
                    with self.assertRaisesRegex(ValueError, "compact_teacher"):
                        Config.model_validate(raw)

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

    def test_padded_target_vocabulary_sizes_the_teacher(self):
        """A target that pads its tables gets a teacher with the padded rows."""
        from safetensors.torch import save_file

        from specforge.training.assembly import build_model_bundle
        from tests.test_modeling.test_canonical_eagle3 import (
            model_config as eagle_config,
        )

        with TemporaryDirectory() as temp:
            target = target_directory(Path(temp))
            payload = json.loads((target / "config.json").read_text())
            payload.update(vocab_size=60, padded_vocab_size=64)
            (target / "config.json").write_text(json.dumps(payload))
            save_file(
                {
                    "model.embed_tokens.weight": torch.randn(64, 32),
                    "lm_head.weight": torch.randn(64, 32),
                },
                target / "model.safetensors",
            )
            resolved = resolve_run(
                run_config(
                    target,
                    eagle_config(target_width=32),
                    training={"attention_backend": "eager", "batch_size": 1},
                )
            )
            bundle = build_model_bundle(resolved.config, algorithm=resolved.algorithm)
            self.assertEqual(bundle.target_head.weight.shape[0], 64)
            self.assertEqual(bundle.target_vocab_size, 64)
            narrow = dataclasses.replace(eagle_config(target_width=32), vocab_size=60)
            resolved = resolve_run(
                run_config(
                    target,
                    narrow,
                    model={
                        "target_model_path": str(target),
                        "torch_dtype": "float32",
                        "load_target_embedding": False,
                    },
                    training={"attention_backend": "eager", "batch_size": 1},
                )
            )
            with self.assertRaisesRegex(ValueError, "target logit width 64"):
                build_model_bundle(resolved.config, algorithm=resolved.algorithm)

    def test_vocab_mapping_cannot_replace_a_copied_vocabulary(self):
        """weights_from copies lm_head rows trained under the source's mapping."""
        from specforge.loading import initialize_draft
        from specforge.training.assembly import build_model_bundle
        from tests.test_modeling.test_canonical_eagle3 import install_mapping
        from tests.test_modeling.test_canonical_eagle3 import (
            model_config as eagle_config,
        )

        with TemporaryDirectory() as temp:
            root = Path(temp)
            target = target_directory(root)
            draft = eagle_config(target_width=32)
            source = initialize_draft(draft, target=target)
            install_mapping(source)
            source.save_pretrained(root / "source")
            t2d = torch.zeros(draft.vocab_size, dtype=torch.bool)
            t2d[: draft.draft_vocab_size] = True
            torch.save(
                {
                    "t2d": t2d,
                    "d2t": torch.zeros(draft.draft_vocab_size, dtype=torch.long),
                },
                root / "mapping.pt",
            )
            resolved = resolve_run(
                run_config(
                    target,
                    draft,
                    model={
                        "target_model_path": str(target),
                        "torch_dtype": "float32",
                        "vocab_mapping_path": str(root / "mapping.pt"),
                    },
                    initialization={
                        "weights_from": {"checkpoint": str(root / "source")}
                    },
                )
            )
            with self.assertRaisesRegex(ValueError, "vocab_mapping_path disagrees"):
                build_model_bundle(resolved.config, algorithm=resolved.algorithm)

    def test_objective_frozen_head_that_the_draft_disables_needs_no_weights(self):
        """DSpark freezes its confidence head at alpha 0, which a draft may omit."""
        from specforge.training.assembly import build_model_bundle
        from tests.test_modeling.test_canonical_dspark import model_config as dspark

        with TemporaryDirectory() as temp:
            target = target_directory(Path(temp))
            for confidence in (False, True):
                cfg = run_config(
                    target,
                    dspark(confidence=confidence),
                    training={
                        "attention_backend": "eager",
                        "loss": {"terms": {"confidence": {"coefficient": 0.0}}},
                    },
                )
                resolved = resolve_run(cfg)
                with self.subTest(confidence=confidence):
                    if confidence:
                        # a present but untrained head still needs a source
                        with self.assertRaisesRegex(ValueError, "require loaded"):
                            build_model_bundle(
                                resolved.config, algorithm=resolved.algorithm
                            )
                        continue
                    bundle = build_model_bundle(
                        resolved.config, algorithm=resolved.algorithm
                    )
                    self.assertIsNone(bundle.draft_model.confidence_head)
                    self.assertEqual(bundle.model.frozen_parameters, ())

    def test_default_target_embedding_copy_follows_the_draft_dtype(self):
        """Like legacy load_embedding; explicit from_target copies stay strict."""
        from safetensors.torch import load_file, save_file

        from specforge.checkpoint.initialization import make_initialization_plan
        from specforge.checkpoint.target import TargetWeights
        from specforge.modeling import build_draft_model
        from specforge.training.assembly import build_model_bundle
        from tests.test_modeling.test_canonical_eagle3 import model_config as eagle

        with TemporaryDirectory() as temp:
            target = target_directory(Path(temp))
            path = target / "model.safetensors"
            tables = {key: value.half() for key, value in load_file(path).items()}
            save_file(tables, path)
            model = {"target_model_path": str(target), "torch_dtype": "bfloat16"}
            cfg = run_config(target, eagle(target_width=32), model=model)
            resolved = resolve_run(cfg)
            bundle = build_model_bundle(resolved.config, algorithm=resolved.algorithm)
            embedding = bundle.draft_model.embed_tokens.weight.detach().cpu()
            self.assertEqual(embedding.dtype, torch.bfloat16)
            expected = tables["model.embed_tokens.weight"].to(torch.bfloat16)
            self.assertTrue(torch.equal(embedding, expected))
            draft = build_draft_model(
                eagle(target_width=32), device="meta", dtype=torch.bfloat16
            )
            store = TargetWeights(target)
            with self.assertRaisesRegex(ValueError, "narrow"):
                make_initialization_plan(
                    draft,
                    target_store=store,
                    target_config=store.config,
                    from_target={"embed_tokens.weight": "model.embed_tokens.weight"},
                )

    def test_cli_overrides_do_not_warn_about_unset_legacy_options(self):
        """apply_overrides marks every field set; only YAML values may warn."""
        with TemporaryDirectory() as temp:
            target = target_directory(Path(temp))
            cfg = apply_overrides(
                run_config(target, draft_config()), ["training.max_steps=2"]
            )
            with self.assertNoLogs("specforge.training.canonical", level="WARNING"):
                resolve_run(cfg)

    def test_frozen_order_does_not_change_the_resume_contract(self):
        """The same frozen set in another order must still resume."""
        from specforge.training.assembly import build_model_bundle

        with TemporaryDirectory() as temp:
            target = target_directory(Path(temp))
            contracts = []
            for frozen in (["lm_head", "embed_tokens"], ["embed_tokens", "lm_head"]):
                training = {"attention_backend": "eager", "frozen": frozen}
                resolved = resolve_run(
                    run_config(target, draft_config(), training=training)
                )
                bundle = build_model_bundle(
                    resolved.config, algorithm=resolved.algorithm
                )
                contract = bundle.strategy_kwargs.resume_contract
                contracts.append(contract["canonical_frozen"])
            self.assertEqual(contracts[0], contracts[1])

    def test_explicit_copy_of_a_tied_alias_replaces_the_default_embedding_copy(self):
        """from_target on a tied draft's lm_head already sets embed_tokens."""
        from safetensors.torch import load_file

        from specforge.training.assembly import build_model_bundle

        with TemporaryDirectory() as temp:
            target = target_directory(Path(temp))
            draft = draft_config(
                tie_word_embeddings=True,
                target_bindings=(),
                target_logit_transform=None,
            )
            cfg = run_config(
                target,
                draft,
                initialization={"from_target": {"lm_head.weight": "lm_head.weight"}},
            )
            resolved = resolve_run(cfg)
            bundle = build_model_bundle(resolved.config, algorithm=resolved.algorithm)
            head = load_file(target / "model.safetensors")["lm_head.weight"]
            embedding = bundle.draft_model.embed_tokens.weight.detach().cpu()
            self.assertTrue(torch.equal(embedding, head))

    def test_offline_records_need_the_supervision_the_objective_trains_on(self):
        """Legacy rejects these by name; canonical failed generically or skipped."""
        with TemporaryDirectory() as temp:
            target = target_directory(Path(temp))
            resolved = resolve_run(run_config(target, draft_config()))
            provider = resolved.algorithm.providers.offline_for("text")
            normalize = provider.build_normalizer(16)
            record = {
                "input_ids": torch.arange(4),
                "loss_mask": torch.tensor([1, 0, 1, 0]),
                "hidden_states": torch.zeros(4, 32 * 5),
            }
            with self.assertRaisesRegex(ValueError, "loss-mask requirement"):
                normalize(record)
            record["loss_mask"] = torch.tensor([1, 1, 0, 0])
            self.assertEqual(normalize(record)["loss_mask"].shape, (1, 4))
