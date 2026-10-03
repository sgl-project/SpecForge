import json
import unittest
from unittest import mock

import torch
from torch import nn
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import OnlineDominoModel
from specforge.algorithms.domino.providers import build_draft
from specforge.config import Config
from specforge.modeling.draft.dflash_kernels import load_liger_dflash_kernels
from specforge.modeling.draft.domino import DominoDraftModel


def draft_config():
    config = Qwen3Config(
        architectures=["DominoDraftModel"],
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=256,
        num_target_layers=4,
        block_size=16,
        bos_token_id=1,
        eos_token_id=2,
        layer_types=["full_attention"],
        dflash_config={
            "target_layer_ids": [0, 1],
            "projector_type": "domino",
            "mask_token_id": 255,
            "emb_dim": 16,
            "gru_hidden_dim": 32,
            "shift_label": True,
            "pure_draft_prefix_len": 1,
        },
    )
    config._attn_implementation = "sdpa"
    return config


def model(cache=False, liger=False, chunk=4, dtype=torch.bfloat16, device="cuda"):
    torch.manual_seed(33)
    config = draft_config()
    draft = DominoDraftModel(
        config,
        dflash_kernels=load_liger_dflash_kernels() if liger else None,
    )
    head = nn.Linear(64, 256, bias=False).requires_grad_(False)
    embedding = nn.Embedding(256, 64).requires_grad_(False)
    return OnlineDominoModel(
        draft,
        head,
        embedding,
        255,
        block_size=16,
        attention_backend="sdpa",
        num_anchors=12,
        loss_decay_gamma=7,
        objective_chunk_blocks=chunk,
        shift_label=True,
        cache_projection=cache,
    ).to(device=device, dtype=dtype)


def inputs(device):
    torch.manual_seed(11)
    tokens = torch.randint(3, 250, (2, 64), device=device)
    hidden = torch.randn(2, 64, 128, device=device, dtype=torch.bfloat16)
    mask = torch.ones(2, 64, device=device)
    mask[:, :8] = 0
    return tokens, hidden, mask


class DominoConfigurationTest(unittest.TestCase):
    def test_liger_flag_reaches_domino_modules(self):
        payload = {
            "model": {"target_model_path": "unused", "use_liger_kernel": True},
            "data": {"hidden_states_path": "unused"},
            "training": {"strategy": "domino", "total_steps": 10},
        }
        config = Config.model_validate(payload)
        with mock.patch(
            "specforge.algorithms.model_providers._device",
            return_value=torch.device("cpu"),
        ):
            draft = build_draft(config, draft_config())
        self.assertIn("liger_kernel", type(draft.norm).__module__)
        self.assertIn("liger_kernel", type(draft.layers[0].mlp).__module__)
        config.model.use_liger_kernel = False
        with mock.patch(
            "specforge.algorithms.model_providers._device",
            return_value=torch.device("cpu"),
        ):
            baseline = build_draft(config, draft_config())
        self.assertIn("transformers", type(baseline.norm).__module__)
        self.assertEqual(set(draft.state_dict()), set(baseline.state_dict()))

    def test_projection_flag_rejects_other_algorithms(self):
        for strategy in ("dflash", "dspark", "eagle3"):
            with self.subTest(strategy=strategy), self.assertRaisesRegex(
                ValueError, "requires training.strategy=domino"
            ):
                Config.model_validate(
                    {
                        "model": {"target_model_path": "unused"},
                        "data": {"hidden_states_path": "unused"},
                        "training": {
                            "strategy": strategy, "domino_cache_projection": True
                        },
                    }
                )

    def test_dispatch_policy_is_domino_only_and_checks_watermark(self):
        payload = {
            "model": {"target_model_path": "unused", "target_backend": "sglang"},
            "data": {"train_data_path": "unused"},
            "training": {
                "strategy": "dflash", "batch_size": 2, "accumulation_steps": 2
            },
            "deployment": {
                "mode": "disaggregated",
                "trainer": {
                    "nnodes": 4, "nproc_per_node": 8, "master_addr": "localhost"
                },
                "disaggregated": {
                    "backend": "mooncake",
                    "control_dir": "unused",
                    "consumer_state_dir": "/tmp/consumer-state",
                    "server_urls": ["http://localhost:30000"],
                },
            },
            "runtime": {
                "consumer_dispatch": "domino_balanced",
                "in_flight_high_watermark": 255,
            },
        }
        with self.assertRaisesRegex(ValueError, "requires training.strategy=domino"):
            Config.model_validate(payload)
        payload["training"]["strategy"] = "domino"
        with self.assertRaisesRegex(ValueError, "at least two global batches"):
            Config.model_validate(payload)
        payload["runtime"]["in_flight_high_watermark"] = 256
        self.assertEqual(
            Config.model_validate(payload).runtime.consumer_dispatch, "domino_balanced"
        )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class DominoOptimizationParityTest(unittest.TestCase):
    def compare(self, baseline, optimized, *, exact):
        optimized.load_state_dict(baseline.state_dict(), strict=True)
        tensors = inputs("cuda")
        outputs = []
        for candidate in (baseline, optimized):
            torch.manual_seed(71)
            loss, accuracy, metrics = candidate(
                *tensors,
                lambda_base=0.37,
                collect_detailed_metrics=True,
            )
            loss.backward()
            outputs.append((loss.detach(), accuracy.detach(), metrics))
        tolerance = dict(rtol=0, atol=0) if exact else dict(rtol=0.02, atol=0.02)
        torch.testing.assert_close(outputs[0][0], outputs[1][0], **tolerance)
        for name in ("base_loss", "final_loss"):
            torch.testing.assert_close(
                outputs[0][2][name], outputs[1][2][name], **tolerance
            )
        gradients = []
        for candidate in (baseline, optimized):
            gradients.append(
                {
                    name: parameter.grad
                    for name, parameter in candidate.named_parameters()
                    if parameter.requires_grad
                }
            )
        self.assertEqual(gradients[0].keys(), gradients[1].keys())
        for name, expected in gradients[0].items():
            self.assertIsNotNone(expected, name)
            actual = gradients[1][name]
            self.assertTrue(torch.isfinite(actual).all(), name)
            torch.testing.assert_close(actual, expected, **tolerance, msg=name)
        if not exact:
            expected = torch.cat(
                [value.float().flatten() for value in gradients[0].values()]
            )
            actual = torch.cat(
                [value.float().flatten() for value in gradients[1].values()]
            )
            relative_error = float((actual - expected).norm() / expected.norm())
            cosine = float(
                torch.nn.functional.cosine_similarity(actual, expected, dim=0)
            )
            self.assertLess(relative_error, 0.02)
            self.assertGreater(cosine, 0.999)
            print(json.dumps({
                "domino_liger_gradient_relative_l2": relative_error,
                "domino_liger_gradient_cosine": cosine,
                "loss_absolute_difference": float(
                    (outputs[0][0] - outputs[1][0]).abs()
                ),
            }))

    def test_cached_projection_matches_loss_and_all_gradients(self):
        self.compare(model(), model(cache=True), exact=True)

    def test_cached_projection_with_liger_matches(self):
        self.compare(model(liger=True), model(cache=True, liger=True), exact=True)

    def test_liger_preserves_loss_and_gradients(self):
        self.compare(model(), model(liger=True), exact=False)

    def test_no_checkpoint_chunk_zero_remains_compatible(self):
        self.compare(model(chunk=0), model(cache=True, chunk=0), exact=True)

    def test_trainable_target_head_falls_back(self):
        baseline, optimized = model(), model(cache=True)
        baseline.lm_head.requires_grad_(True)
        optimized.lm_head.requires_grad_(True)
        self.compare(baseline, optimized, exact=True)


if __name__ == "__main__":
    unittest.main()
