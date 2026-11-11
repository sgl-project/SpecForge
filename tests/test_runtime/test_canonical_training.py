"""Exercise canonical loading, data ports, trainer checkpoints and replay."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import load_file, save_file
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
        model={
            "target_model_path": str(target),
            "torch_dtype": "float32",
            "use_liger_kernel": False,
        },
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
    def test_existing_offline_fields_and_packed_lengths(self):
        from specforge.runtime.data_plane.feature_store import LocalFeatureStore
        from specforge.training.canonical_data import (
            build_reader,
            collate,
            normalize_sample,
        )

        keys = ("input_ids", "loss_mask", "hidden_states", "target_last_hidden_states")
        with TemporaryDirectory() as temp:
            torch.save(
                dict(
                    input_ids=torch.arange(7),
                    loss_mask=torch.ones(7),
                    aux_hidden_state=torch.randn(1, 7, 24),
                    hidden_state=torch.randn(1, 7, 8),
                    lengths=torch.tensor([3, 4]),
                ),
                Path(temp) / "sample.ckpt",
            )
            reader = build_reader(
                temp, strategy="eagle3", keys=keys, validate_files=True
            )
            store = LocalFeatureStore()
            raw, handle = store.get(reader.read()[0])
            sample = normalize_sample(raw, keys=keys, max_len=5)
            batch = collate([sample], keys=keys)
            self.assertEqual(batch["lengths"].tolist(), [3, 2])
            self.assertEqual(batch["hidden_states"].shape, (1, 5, 24))
            self.assertEqual(batch["target_last_hidden_states"].shape, (1, 5, 8))
            self.assertEqual(batch["loss_mask"][0, -1], 0)
            store.release(handle)

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

    @unittest.skipUnless(torch.cuda.is_available(), "FSDP integration requires CUDA")
    def test_mtp_native_copy_sdpa_chunking_and_resume(self):
        self._train_and_resume(False, mtp=True)

    def _train_and_resume(self, tied, mtp=False):
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
            if mtp:
                from specforge.checkpoint.formats.sglang.converters import (
                    serving_weights,
                )
                from specforge.modeling import build_draft_model
                from tests.test_checkpoint.test_native_presets import (
                    NATIVE_MAPPING,
                    native_configs,
                )

                draft = next(native_configs(small=True))
                original = build_draft_model(draft)
                native = {
                    "mtp." + k: v
                    for k, v in serving_weights(draft, original.state_dict()).items()
                }
                tables = load_file(target / "model.safetensors")
                save_file(tables | native, target / "model.safetensors")
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
                            "target_last_hidden_states",
                            *(("hidden_states",) if not mtp else ()),
                        )
                    },
                    features / f"{index}.ckpt",
                )
            cfg = run_config(
                target,
                draft,
                training={
                    "attention_backend": "sdpa" if mtp else "eager",
                    "max_steps": 1,
                    "total_steps": 2,
                    "save_interval": 1,
                    "log_interval": 1,
                    "objective_options": (
                        {"objective_chunk_size": 2} if mtp else {"num_anchors": 2}
                    ),
                },
            )
            if mtp:
                raw = cfg.model_dump()
                raw["initialization"] = {"from_target": NATIVE_MAPPING}
                cfg = Config.model_validate(raw)
            first = build_application_run(cfg)
            first.run()
            checkpoint = root / "output" / "specforge-run-step1"
            self.assertTrue((checkpoint / "config.json").exists())
            payload = json.loads((checkpoint / "config.json").read_text())
            self.assertEqual(
                payload["specforge_artifact"], {"format": "specforge", "version": 1}
            )
            self.assertFalse((checkpoint / "artifact.json").exists())
            model = load_draft(checkpoint, target=target, dtype=torch.float32)
            before = {
                name: tensor.clone() for name, tensor in model.state_dict().items()
            }
            raw = cfg.model_dump()
            raw.update(draft=None, initialization=None, resume=str(checkpoint))
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
        torch.testing.assert_close(
            output.ratio_metrics["dspine/backbone_ce"],
            replayed.ratio_metrics["dspine/backbone_ce"],
        )
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
        resolve_run(cfg)
        cfg = run_config(
            Path("target"),
            draft,
            training={"attention_backend": "usp", "sp_ulysses_size": 2},
        )
        with self.assertRaisesRegex(ValueError, "requires EAGLE3"):
            resolve_run(cfg)
