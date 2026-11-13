"""Exercise canonical loading, data ports, trainer checkpoints and replay."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

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
    def test_mtp_training_requires_explicit_initialization(self):
        from tests.test_modeling.test_mtp_sdpa_chunking import owned_mtp_config

        with TemporaryDirectory() as tmp:
            target = target_directory(Path(tmp))
            cfg = run_config(target, owned_mtp_config())
            resolved = resolve_run(cfg)  # Capture planning does not initialize weights.
            with self.assertRaisesRegex(ValueError, "initialization: {}"):
                build_model_bundle(resolved.config, algorithm=resolved.algorithm)
            raw = cfg.model_dump()
            raw["initialization"] = {}
            resolved = resolve_run(Config.model_validate(raw))
            bundle = build_model_bundle(resolved.config, algorithm=resolved.algorithm)
            self.assertIsNotNone(bundle.draft_model)

    def test_target_tables_are_reused_and_owned_embedding_is_initialized(self):
        from safetensors.torch import load_file

        from specforge.checkpoint.tensors import TensorStore

        with TemporaryDirectory() as temp:
            target = target_directory(Path(temp))
            original = TensorStore.read
            reads = []

            def read(store, key, region=None):
                reads.append(key)
                return original(store, key, region)

            resolved = resolve_run(
                run_config(
                    target,
                    draft_config(),
                    training={"attention_backend": "eager", "loss": {"kind": "kl"}},
                )
            )
            with patch.object(TensorStore, "read", read):
                bundle = build_model_bundle(
                    resolved.config, algorithm=resolved.algorithm
                )
            self.assertIs(bundle.target_head, bundle.draft_model.lm_head)
            self.assertEqual(reads.count("lm_head.weight"), 1)
            draft = peagle_config(target_width=32)
            resolved = resolve_run(run_config(target, draft))
            bundle = build_model_bundle(resolved.config, algorithm=resolved.algorithm)
            expected = load_file(target / "model.safetensors")[
                "model.embed_tokens.weight"
            ]
            torch.testing.assert_close(
                bundle.draft_model.embed_tokens.weight.cpu(), expected, rtol=0, atol=0
            )

    def test_target_aliases_survive_config_roundtrip(self):
        with TemporaryDirectory() as temp:
            target = target_directory(Path(temp))
            save_file(
                {"embed_tokens.weight": torch.randn(64, 32)},
                target / "model.safetensors",
            )
            payload = json.loads((target / "config.json").read_text())
            payload["tie_word_embeddings"] = True
            (target / "config.json").write_text(json.dumps(payload))
            cfg = Config.model_validate(run_config(target, draft_config()).model_dump())
            resolved = resolve_run(cfg)
            bundle = build_model_bundle(resolved.config, algorithm=resolved.algorithm)
            torch.testing.assert_close(
                bundle.draft_model.embed_tokens.weight,
                bundle.draft_model.lm_head.weight,
            )

    def test_legacy_import_settings_and_full_resume_boundary(self):
        import shutil

        from specforge.training.canonical import enabled

        fixture = Path(__file__).resolve().parents[1] / "fixtures/legacy_peagle"
        with TemporaryDirectory() as temp:
            target = target_directory(Path(temp))
            fixture = Path(shutil.copytree(fixture, Path(temp) / "legacy"))
            torch.save({}, fixture / "training_state.pt")
            raw = dict(
                model={"target_model_path": str(target), "mask_token_id": 63},
                initialization={"checkpoint": str(fixture)},
                data={"hidden_states_path": str(target.parent / "features")},
                training={"attention_backend": "eager"},
            )
            resolved = resolve_run(Config.model_validate(raw))
            self.assertEqual(resolved.config.draft["spec_config"]["mask_token_id"], 63)
            raw.pop("initialization")
            raw["resume"] = str(fixture)
            with self.assertRaisesRegex(ValueError, "legacy full resume"):
                resolve_run(Config.model_validate(raw))
            raw.pop("resume")
            raw["training"] = {
                "resume_from": str(fixture),
                "attention_backend": "eager",
            }
            raw["model"]["draft_model_config"] = str(fixture)
            self.assertFalse(enabled(Config.model_validate(raw)))

    @unittest.skipUnless(torch.cuda.is_available(), "Liger requires CUDA")
    def test_liger_training_kernels_preserve_forward_and_gradients(self):
        import copy

        from specforge.config.draft.mlp import DenseFFNSpec
        from specforge.config.draft.norm import RMSNormSpec
        from specforge.modeling.draft.layers.kernels import configure_kernels
        from specforge.modeling.draft.layers.mlp.dense import DenseFFN
        from specforge.modeling.draft.layers.norm.rms import RMSNorm

        for dtype in (torch.float32, torch.bfloat16):
            for offset in (0, 1):
                reference = torch.nn.Sequential(
                    RMSNorm(
                        RMSNormSpec(has_weight=True, eps=1e-6, weight_offset=offset),
                        hidden_size=32,
                    ),
                    DenseFFN(
                        DenseFFNSpec(hidden_act="silu", intermediate_size=64),
                        hidden_size=32,
                    ),
                ).to(device="cuda", dtype=dtype)
                fused = copy.deepcopy(reference)
                configure_kernels(fused, use_liger=True)
                x = torch.randn(
                    2, 8, 32, device="cuda", dtype=dtype, requires_grad=True
                )
                y = x.detach().clone().requires_grad_()
                expected, actual = reference(x), fused(y)
                expected.float().square().mean().backward()
                actual.float().square().mean().backward()
                tolerance = 1e-5 if dtype == torch.float32 else 1e-2
                torch.testing.assert_close(
                    actual, expected, rtol=tolerance, atol=tolerance
                )
                torch.testing.assert_close(
                    x.grad, y.grad, rtol=tolerance, atol=tolerance
                )
                for p, q in zip(reference.parameters(), fused.parameters()):
                    torch.testing.assert_close(
                        p.grad, q.grad, rtol=tolerance, atol=tolerance
                    )
                self.assertEqual(
                    reference.state_dict().keys(), fused.state_dict().keys()
                )

    def test_training_parses_with_target_before_building_the_model(self):
        from unittest.mock import patch

        from specforge.training.model_loading import training_target_config

        with TemporaryDirectory() as temp:
            target = target_directory(Path(temp))
            config = json.loads((target / "config.json").read_text())
            config["logits_mup_width_multiplier"] = 8
            (target / "config.json").write_text(json.dumps(config))
            cfg = run_config(
                target,
                draft_config(),
                training={"attention_backend": "eager", "loss": {"kind": "kl"}},
            )
            del cfg.draft["target_logit_transform"]
            resolved = resolve_run(cfg, target_config=training_target_config(cfg))
            self.assertEqual(
                resolved.config.draft["target_logit_transform"],
                {"type": "mup", "multiplier": 8},
            )
            bundle = build_model_bundle(resolved.config, algorithm=resolved.algorithm)
            self.assertEqual(bundle.draft_config.target_logit_transform.multiplier, 8)
            from safetensors.torch import load_file

            expected = load_file(target / "model.safetensors")["lm_head.weight"] / 8
            torch.testing.assert_close(
                bundle.target_head.weight.cpu(), expected, rtol=0, atol=0
            )
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
                    cfg = run_config(target, draft, initialization={})
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

    def test_loss_selection_drives_capture_head_and_resume_contract(self):
        from specforge.algorithms.objectives import resolve_objective

        with TemporaryDirectory() as temp:
            target = target_directory(Path(temp))
            for kind in ("ce", "kl", "lk_lambda"):
                with self.subTest(loss=kind):
                    loss = {"kind": kind}
                    if kind == "lk_lambda":
                        loss["options"] = {"target": "teacher", "kl_scale": 0.7}
                    cfg = run_config(
                        target,
                        draft_config(),
                        training={
                            "attention_backend": "eager",
                            "loss": loss,
                            "weighting": {"kind": "dpace", "alpha": 0.3},
                        },
                    )
                    resolved = resolve_run(cfg)
                    bundle = build_model_bundle(
                        resolved.config, algorithm=resolved.algorithm
                    )
                    needs_teacher = kind != "ce"
                    layout = resolved.algorithm.providers.offline_for(
                        "text"
                    ).capture_layout
                    self.assertEqual(
                        layout.last_hidden_feature is not None, needs_teacher
                    )
                    self.assertEqual(bundle.target_head is not None, needs_teacher)
                    self.assertEqual(
                        "target_last_hidden_states" in bundle.model.registration.inputs,
                        needs_teacher,
                    )
                    contract = bundle.strategy_kwargs.resume_contract
                    self.assertEqual(contract["canonical_loss"]["kind"], kind)
                    self.assertEqual(contract["canonical_weighting"]["alpha"], 0.3)
                    if bundle.target_head is not None:
                        self.assertFalse(
                            any(
                                p.requires_grad for p in bundle.target_head.parameters()
                            )
                        )
            # MTP consumes final states but CE needs no teacher projection; EAGLE
            # still needs a teacher for vocabulary mapping when selecting CE.
            self.assertFalse(resolve_objective("MTPDraftModel").requires_teacher)
            self.assertTrue(
                resolve_objective(
                    "Eagle3DraftModel", loss={"kind": "ce"}
                ).requires_teacher
            )
            for changes in (
                {"loss": {"kind": "unknown"}},
                {"weighting": {"kind": "dpace", "alpha": 2}},
            ):
                cfg = run_config(
                    target,
                    draft_config(),
                    training={"attention_backend": "eager", **changes},
                )
                with patch("specforge.loading.initialize_draft") as initialize:
                    with self.assertRaises(ValueError):
                        resolve_run(cfg)
                    initialize.assert_not_called()
            raw = run_config(target, draft_config()).model_dump()
            raw["draft"] = None
            raw["training"]["loss"] = {"kind": "kl"}
            with self.assertRaisesRegex(ValueError, "canonical"):
                Config.model_validate(raw)

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
                    "attention_backend": "sdpa" if mtp else "eager",
                    "max_steps": 1,
                    "total_steps": 2,
                    "save_interval": 1,
                    "log_interval": 1,
                    "objective_options": (
                        {"objective_chunk_size": 2} if mtp else {"num_anchors": 2}
                    ),
                    "loss": {"kind": "ce" if mtp else "kl"},
                    "weighting": (
                        {"kind": "uniform"} if mtp else {"kind": "dpace", "alpha": 0.3}
                    ),
                },
            )
            cfg.model.use_liger_kernel = tied
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
            self.assertEqual(state["canonical_loss"]["kind"], "ce" if mtp else "kl")
            if mtp:
                self.assertEqual(state["canonical_weighting"]["kind"], "uniform")
            else:
                self.assertEqual(state["canonical_weighting"]["alpha"], 0.3)
            if tied:
                self.assertNotIn("lm_head.weight", state["draft_state_dict"])
                self.assertIs(model.embed_tokens.weight, model.lm_head.weight)
            self.assertTrue(
                any(
                    not torch.equal(before[name], tensor)
                    for name, tensor in state["draft_state_dict"].items()
                )
            )
            changes = [("loss", {"kind": "tv"})]
            if not mtp:
                changes.append(("weighting", {"kind": "uniform"}))
            for key, changed in changes:
                with self.subTest(changed=key):
                    incompatible = Config.model_validate(raw)
                    setattr(incompatible.training, key, changed)
                    with self.assertRaisesRegex(ValueError, f"canonical_{key}"):
                        build_application_run(incompatible)
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
        for accumulation in (1, 2):
            with self.subTest(accumulation=accumulation):
                initial_rng = torch.get_rng_state().clone()
                ctx = StepContext(
                    global_step=1, total_steps=10, accumulation_steps=accumulation
                )
                with (
                    patch(
                        "specforge.training.strategies.canonical.dist.is_initialized",
                        return_value=True,
                    ),
                    patch(
                        "specforge.training.strategies.canonical.dist.all_reduce"
                    ) as reduce,
                ):
                    output = strategy.forward_loss(batch, ctx)
                    reduce.assert_not_called()
                    state = torch.get_rng_state().clone()
                    # A fractional denominator must remain exact.
                    window = {
                        "dspine/backbone_ce": (torch.tensor(0.1), torch.tensor(0.2))
                    }
                    replayed = output.replay_loss(window)
                    output.replay_loss(window)
                    reduce.assert_called_once()
                torch.testing.assert_close(state, torch.get_rng_state())
                with torch.random.fork_rng():
                    torch.set_rng_state(initial_rng)
                    expected = runtime(
                        batch.tensors,
                        step=1,
                        total_steps=10,
                        alignment_ce=torch.tensor(0.5),
                    )
                torch.testing.assert_close(replayed.loss, expected.loss)
                replayed.loss.backward()
                self.assertTrue(
                    all(
                        p.grad is not None
                        for p in model.parameters()
                        if p.requires_grad
                    )
                )
        strategy.trainable_module().eval()
        self.assertFalse(runtime.training)
        with patch("specforge.training.strategies.canonical.dist.all_reduce") as reduce:
            evaluated = strategy.forward_loss(batch, StepContext(global_step=1))
            self.assertIsNone(evaluated.replay_loss)
            reduce.assert_not_called()

    def test_eagle3_sdpa_is_selectable_and_matches_eager_loss(self):
        """EAGLE3 declares SDPA; switching backends must not change its loss."""
        from torch import nn

        from specforge.algorithms.objectives import build_objective_model
        from specforge.modeling import build_draft_model
        from tests.test_modeling.test_canonical_eagle3 import install_mapping

        draft = eagle_config(target_width=32)
        with TemporaryDirectory() as temp:
            target = target_directory(Path(temp))
            resolve_run(
                run_config(target, draft, training={"attention_backend": "sdpa"})
            )
        torch.manual_seed(0)
        eager = build_draft_model(draft, backend="eager")
        sdpa = build_draft_model(draft, backend="sdpa")
        sdpa.load_state_dict(eager.state_dict())
        head = nn.Linear(32, draft.vocab_size, bias=False)
        batch = training_batch(draft)
        losses = []
        for model in (eager, sdpa):
            install_mapping(model)
            runtime = build_objective_model(
                model, options={"ttt_length": 2}, target_head=head
            )
            losses.append(runtime(batch, step=1, total_steps=10).loss)
        torch.testing.assert_close(losses[1], losses[0])

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
