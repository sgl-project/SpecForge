"""CPU contract tests for LongSpark training."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.algorithms.common.providers import ServerCaptureLayout
from specforge.algorithms.longspark import FEATURES
from specforge.algorithms.longspark.model import (
    FrozenFinalNorm,
    LongSparkTrainingModel,
    LongSparkTrainStrategy,
    objective,
)
from specforge.algorithms.longspark.providers import (
    build_training_model,
    capture_layers,
    collate,
    resume_contract,
)
from specforge.algorithms.longspark.sampling import has_trainable_block, sample_anchors
from specforge.modeling.draft.longspark import LongSparkConfig, LongSparkDraftModel
from specforge.runtime.contracts import TrainBatch
from specforge.training.strategies.base import StepContext


def tiny_model():
    config = LongSparkConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=48,
        target_layer_ids=[0, 2],
        mask_token_id=47,
        markov_rank=4,
        gradient_checkpointing=False,
    )
    return LongSparkTrainingModel(
        LongSparkDraftModel(config),
        nn.Embedding(48, 16),
        nn.Linear(16, 48, bias=False),
        FrozenFinalNorm(torch.ones(16), 1e-6),
        num_anchors=3,
        ce_weight=0.1,
        l1_weight=0.9,
        gamma=4.0,
    )


def features(length):
    return {
        "input_ids": torch.randint(0, 47, (1, length)),
        "loss_mask": torch.ones(1, length),
        "hidden_states": torch.randn(1, length, 32),
        "target_last_hidden_states": torch.randn(1, length, 16),
        "target_keys": torch.randn(1, length, 2, 2, 4),
        "target_values": torch.randn(1, length, 2, 2, 4),
    }


class TestLongSpark(unittest.TestCase):
    """Use TestCase so upstream's unittest discovery actually runs this gate."""

    def test_registration_and_capture_contract(self):
        registration = builtin_algorithm_registry().resolve("longspark")
        self.assertEqual(
            registration.spec.feature_contract("streaming", "text").required_tensors,
            set(FEATURES),
        )
        stream = registration.providers.server_streaming_for("text")
        self.assertFalse(registration.providers.offline)
        with self.assertRaisesRegex(KeyError, "offline"):
            registration.spec.feature_contract("offline", "text")
        self.assertEqual(stream.capture_method, "longspark")
        self.assertEqual(
            stream.layout.extra_artifacts,
            (("target_keys", "target_keys"), ("target_values", "target_values")),
        )
        target = SimpleNamespace(
            model_type="qwen3",
            hidden_size=4096,
            num_hidden_layers=36,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            vocab_size=151936,
        )
        self.assertEqual(
            capture_layers(None, LongSparkConfig(), target), [0, 9, 18, 26, 35]
        )
        target.hidden_size = 5120
        with self.assertRaisesRegex(ValueError, "hidden_size"):
            capture_layers(None, LongSparkConfig(), target)
        target.hidden_size = 4096
        target.num_hidden_layers = 28
        with self.assertRaisesRegex(ValueError, "target_layer_ids"):
            capture_layers(None, LongSparkConfig(), target)

    def test_extra_capture_artifacts_cannot_replace_builtin_artifacts(self):
        with self.assertRaises(ValueError):
            ServerCaptureLayout(
                aux_feature="hidden_states",
                last_hidden_feature=None,
                passthrough=(),
                extra_artifacts=(("aux", "keys"),),
            )

    def test_token_major_collation_and_missing_features(self):
        a, b = features(12), features(16)
        result = collate([a, b])
        self.assertEqual(result["target_keys"].shape, (2, 16, 2, 2, 4))
        self.assertEqual(result["loss_mask"][0, 12:].sum(), 0)
        del a["target_keys"]
        with self.assertRaisesRegex(KeyError, "target_keys"):
            collate([a])

    def test_anchor_offsets_and_reproducible_random_draws(self):
        ids = torch.arange(30)[None, :]
        mask = torch.zeros_like(ids)
        mask[:, 8:21] = 1
        a, b = [
            sample_anchors(
                ids,
                mask,
                block_size=7,
                num_anchors=4,
                generator=torch.Generator().manual_seed(9),
            )
            for _ in range(2)
        ]
        self.assertTrue(torch.equal(a.positions, b.positions))
        self.assertEqual(a.positions.unique().numel(), 4)
        self.assertTrue(torch.all(a.positions >= 7) and torch.all(a.positions <= 13))
        self.assertTrue(torch.equal(a.current_tokens, a.positions))
        self.assertTrue(
            torch.equal(a.labels, a.positions[:, None] + torch.arange(1, 8)[None, :])
        )
        self.assertTrue(has_trainable_block(mask, 7))
        self.assertFalse(has_trainable_block(torch.tensor([1] * 8), 7))

    def test_objective_uses_full_vocab_l1_and_hard_regen_labels(self):
        draft = torch.randn(2, 7, 13, requires_grad=True)
        teacher = torch.randn_like(draft)
        labels = torch.randint(0, 13, (2, 7))
        numerator, denominator, _ = objective(
            draft, teacher, labels, ce_weight=0.1, l1_weight=0.9, gamma=4.0
        )
        ce = torch.nn.functional.cross_entropy(
            draft.flatten(0, 1), labels.flatten(), reduction="none"
        ).reshape(2, 7)
        l1 = (draft.softmax(-1) - teacher.softmax(-1)).abs().sum(-1)
        weights = torch.exp(-torch.arange(7) / 4)[None, :].expand(2, -1)
        torch.testing.assert_close(numerator, ((0.1 * ce + 0.9 * l1) * weights).sum())
        torch.testing.assert_close(denominator, weights.sum())
        (numerator / denominator).backward()
        self.assertTrue(torch.isfinite(draft.grad).all())
        # Without loss_decay_gamma, positions are weighted uniformly.
        numerator, denominator, _ = objective(
            draft, teacher, labels, ce_weight=0.1, l1_weight=0.9, gamma=None
        )
        torch.testing.assert_close(numerator, (0.1 * ce + 0.9 * l1).sum())
        torch.testing.assert_close(denominator, torch.tensor(14.0))

    def test_equal_scalar_losses_do_not_hide_missing_position_decay(self):
        # Identical per-position examples have identical scalar means under
        # uniform and decayed weights. Their gradients must still differ.
        torch.manual_seed(17)
        draft = torch.randn(2, 1, 13).repeat(1, 7, 1).requires_grad_()
        teacher = torch.randn(2, 1, 13).repeat(1, 7, 1)
        labels = torch.randint(0, 13, (2, 1)).repeat(1, 7)
        for gamma in (2.0, 4.0):
            with self.subTest(gamma=gamma):
                numerator, denominator, _ = objective(
                    draft, teacher, labels, ce_weight=0.1, l1_weight=0.9, gamma=gamma
                )
                grad = torch.autograd.grad(numerator / denominator, draft)[0]
                relative = grad.abs().sum(dim=(0, 2))
                relative = relative / relative[0]
                expected = torch.exp(-torch.arange(7).float() / gamma)
                torch.testing.assert_close(relative, expected)
                self.assertFalse(torch.allclose(relative, torch.ones(7)))

    def test_yaml_gamma_reaches_effective_training_objective(self):
        from specforge.config import load_config

        root = Path(__file__).resolve().parents[2]
        recipes = (
            "examples/configs/online/disaggregated/external/qwen3-8b-longspark-disaggregated.yaml",
        )
        for recipe in recipes:
            for gamma in (2.0, 4.0):
                with self.subTest(recipe=recipe, gamma=gamma):
                    cfg = load_config(
                        str(root / recipe),
                        overrides=[
                            f"training.loss_decay_gamma={gamma}",
                            "model.torch_dtype=float32",
                        ],
                    )
                    self.assertEqual(cfg.deployment.trainer.nnodes, 1)
                    self.assertEqual(cfg.deployment.trainer.nproc_per_node, 8)
                    self.assertEqual(cfg.training.accumulation_steps, 64)
                    self.assertEqual(
                        cfg.deployment.trainer.nproc_per_node
                        * cfg.training.batch_size
                        * cfg.training.accumulation_steps,
                        512,
                    )
                    tiny = tiny_model()
                    target = SimpleNamespace(
                        embed_tokens=tiny.embed_tokens, lm_head=tiny.lm_head
                    )
                    # Mock large checkpoint I/O, not configuration wiring or loss.
                    with (
                        mock.patch(
                            "specforge.algorithms.longspark.providers.capture_layers",
                            return_value=[0, 9, 18, 26, 35],
                        ),
                        mock.patch(
                            "specforge.utils.get_local_device",
                            return_value=torch.device("cpu"),
                        ),
                        mock.patch(
                            "specforge.modeling.target.target_utils.TargetEmbeddingsAndHead.from_pretrained",
                            return_value=target,
                        ),
                        mock.patch(
                            "specforge.modeling.target.checkpoint.load_checkpoint_tensors",
                            return_value={"model.norm.weight": torch.ones(16)},
                        ),
                    ):
                        model = build_training_model(
                            cfg,
                            tiny.draft_model,
                            LongSparkConfig(),
                            SimpleNamespace(rms_norm_eps=1e-6),
                            None,
                        ).model
                    self.assertEqual(model.gamma, gamma)
                    self.assertEqual(model.num_anchors, cfg.training.num_anchors)
                    self.assertEqual(
                        (model.ce_weight, model.l1_weight),
                        (
                            cfg.training.dspark_ce_loss_alpha,
                            cfg.training.dspark_l1_loss_alpha,
                        ),
                    )
                    with mock.patch(
                        "specforge.algorithms.longspark.model.objective",
                        wraps=objective,
                    ) as loss:
                        numerator, denominator, _ = model(
                            **features(24),
                            anchor_generator=torch.Generator().manual_seed(9),
                        )
                    self.assertEqual(loss.call_args.kwargs["gamma"], gamma)
                    anchor_count = loss.call_args.args[2].shape[0]
                    self.assertEqual(anchor_count, 16)
                    expected = (
                        anchor_count * torch.exp(-torch.arange(7).float() / gamma).sum()
                    )
                    torch.testing.assert_close(denominator, expected)
                    (numerator / denominator).backward()
                    self.assertTrue(
                        torch.isfinite(
                            model.draft_model.markov_head.markov_w1.weight.grad
                        ).all()
                    )

    def test_tiny_training_backward_freezes_target_and_exports_only_draft(self):
        torch.manual_seed(123)
        model = tiny_model()
        strategy = LongSparkTrainStrategy(model)
        result = strategy.forward_loss(
            TrainBatch(["test"], "longspark", features(24)),
            StepContext(epoch=0, epoch_microstep=3),
        )
        self.assertTrue(torch.isfinite(result.loss))
        self.assertIsNone(result.loss_terms)
        result.loss.backward()
        for name, value in model.named_parameters():
            if name.startswith("draft_model."):
                self.assertTrue(value.requires_grad)
                self.assertIsNotNone(value.grad, name)
                self.assertTrue(torch.isfinite(value.grad).all(), name)
            else:
                self.assertFalse(value.requires_grad)
                self.assertIsNone(value.grad, name)
        self.assertEqual(
            set(strategy.checkpoint_state_filter(model.state_dict())),
            set(model.draft_model.state_dict()),
        )

    def test_draft_weights_materialize_and_export_without_checkpoint_files(self):
        from safetensors.torch import load, save

        from specforge.export.checkpoint_io import materialize_draft
        from specforge.export.to_hf import export_to_hf

        model = tiny_model()
        model.draft_model.config.architectures = ["LongSparkDraftModel"]
        strategy = LongSparkTrainStrategy(model)
        weights = strategy.checkpoint_state_filter(model.state_dict())
        # Real serialization in memory; smoke/unit gates create no model files.
        state = {"strategy": "longspark", "draft_state_dict": load(save(weights))}
        with tempfile.TemporaryDirectory(prefix="longspark-config-") as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(model.draft_model.config.to_dict()))
            restored = materialize_draft(state, str(config_path))
            self.assertEqual(set(restored.state_dict()), set(weights))
            self.assertEqual(
                restored.config.qwen_gradient_checkpointing,
                model.draft_model.config.qwen_gradient_checkpointing,
            )
            for key, value in weights.items():
                torch.testing.assert_close(
                    restored.state_dict()[key], value, rtol=0, atol=0
                )
            with (
                mock.patch(
                    "specforge.export.to_hf.resolve_training_state", return_value=state
                ),
                mock.patch(
                    "specforge.export.to_hf.apply_legacy_rope_scaling",
                    return_value=False,
                ),
                mock.patch.object(
                    LongSparkDraftModel, "save_pretrained", autospec=True
                ) as write,
            ):
                export_to_hf(
                    "memory://training-state", str(config_path), "memory://export"
                )
            self.assertEqual(set(write.call_args.kwargs["state_dict"]), set(weights))
            bad = dict(
                state, draft_state_dict={**weights, "not_a_draft.weight": torch.ones(1)}
            )
            with self.assertRaisesRegex(ValueError, "does not have"):
                materialize_draft(bad, str(config_path))
            missing = dict(weights)
            missing.pop(next(iter(missing)))
            with self.assertRaisesRegex(ValueError, "missing non-embedding"):
                materialize_draft(
                    dict(state, draft_state_dict=missing), str(config_path)
                )
            self.assertEqual(
                [p.name for p in Path(directory).iterdir()], ["config.json"]
            )

    def test_reference_checkpoint_names_load_into_the_draft(self):
        draft = tiny_model().draft_model
        legacy = {
            "global_memory.queries": (
                "global16_raw256_reference.global_query_generator.learned_query"
            ),
            "state_encoder.": "draft_state_encoder.",
            "block_input.": "parallel_shifted_input.",
            "layers.": "parallel_qwen_layers.",
            "markov_head.": "parallel_markov_head.",
        }
        state = {}
        for key, value in draft.state_dict().items():
            for current, old in legacy.items():
                if key.startswith(current):
                    key = old + key[len(current) :]
                    break
            state[key] = value.clone() + 1
        restored = tiny_model().draft_model
        restored.load_state_dict(state, strict=True)
        for key, value in draft.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value + 1)

    def test_resume_contract_records_the_actual_loss_and_anchor_recipe(self):
        model = tiny_model()
        cfg = SimpleNamespace()
        contract = resume_contract(cfg, model.draft_model, model)
        self.assertEqual(contract["longspark_loss_decay_gamma"], 4.0)
        self.assertEqual(contract["longspark_num_anchors"], 3)
        self.assertEqual(contract["longspark_block_size"], 7)
        self.assertEqual(contract["longspark_ce_loss_alpha"], 0.1)
        self.assertEqual(contract["longspark_l1_loss_alpha"], 0.9)
        model.gamma = 2.0
        self.assertNotEqual(contract, resume_contract(cfg, model.draft_model, model))

    def test_evaluation_does_not_add_a_per_batch_collective(self):
        model = tiny_model().eval()
        strategy = LongSparkTrainStrategy(model)
        with (
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_rank", return_value=0),
            mock.patch("torch.distributed.all_reduce") as reduce,
        ):
            result = strategy.forward_loss(
                TrainBatch(["eval"], "longspark", features(16)), StepContext()
            )
        reduce.assert_not_called()
        self.assertTrue(torch.isfinite(result.loss))
