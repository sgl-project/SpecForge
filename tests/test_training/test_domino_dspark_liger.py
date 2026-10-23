import unittest
from unittest import mock

import torch
from torch import nn
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import (
    OnlineDominoModel,
    OnlineDSparkModel,
)
from specforge.algorithms.domino import providers as domino
from specforge.algorithms.dspark import providers as dspark
from specforge.config import Config
from specforge.modeling.draft.dflash_kernels import load_liger_dflash_kernels
from specforge.modeling.draft.domino import DominoDraftModel
from specforge.modeling.draft.dspark import DSparkDraftModel


def draft_config(strategy):
    config = Qwen3Config(
        architectures=[
            "DominoDraftModel" if strategy == "domino" else "DSparkDraftModel"
        ],
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
            "projector_type": strategy,
            "markov_rank": 8,
            "enable_confidence_head": False,
            "mask_token_id": 255,
            "emb_dim": 16,
            "gru_hidden_dim": 32,
            "shift_label": True,
            "pure_draft_prefix_len": 1,
        },
    )
    config._attn_implementation = "sdpa"
    return config


def model(strategy, *, liger=False):
    torch.manual_seed(33)
    draft_cls, online_cls = (
        (DominoDraftModel, OnlineDominoModel)
        if strategy == "domino"
        else (DSparkDraftModel, OnlineDSparkModel)
    )
    draft = draft_cls(
        draft_config(strategy),
        dflash_kernels=load_liger_dflash_kernels() if liger else None,
    )
    return online_cls(
        draft,
        nn.Linear(64, 256, bias=False).requires_grad_(False),
        nn.Embedding(256, 64).requires_grad_(False),
        255,
        block_size=16,
        attention_backend="sdpa",
        num_anchors=12,
        loss_decay_gamma=7,
        objective_chunk_blocks=4,
        **({"shift_label": True} if strategy == "domino" else {}),
    ).to(device="cuda", dtype=torch.bfloat16)


def inputs(device):
    torch.manual_seed(11)
    tokens = torch.randint(3, 250, (2, 64), device=device)
    hidden = torch.randn(2, 64, 128, device=device, dtype=torch.bfloat16)
    mask = torch.ones(2, 64, device=device)
    mask[:, :8] = 0
    return tokens, hidden, mask


class LigerConfigurationTest(unittest.TestCase):
    def test_explicit_opt_in_and_native_defaults(self):
        for strategy, provider in (("domino", domino), ("dspark", dspark)):
            state_keys = None
            for requested in (True, False, None):
                config = Config.model_validate(
                    {
                        "model": {
                            "target_model_path": "unused",
                            "use_liger_kernel": requested,
                        },
                        "data": {"hidden_states_path": "unused"},
                        "training": {"strategy": strategy, "total_steps": 10},
                    }
                )
                with (
                    self.subTest(strategy=strategy, requested=requested),
                    mock.patch(
                        "specforge.algorithms.model_providers._device",
                        return_value=torch.device("cpu"),
                    ),
                    mock.patch(
                        "specforge.modeling.draft.dflash_kernels.current_liger_platform",
                        return_value="cuda",
                    ),
                ):
                    draft = provider.build_draft(config, draft_config(strategy))
                    package = "liger_kernel" if requested else "transformers"
                    self.assertIn(package, type(draft.norm).__module__)
                    self.assertIn(package, type(draft.layers[0].mlp).__module__)
                    if state_keys is None:
                        state_keys = set(draft.state_dict())
                    self.assertEqual(state_keys, set(draft.state_dict()))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class LigerParityTest(unittest.TestCase):
    def compare(self, baseline, optimized):
        optimized.load_state_dict(baseline.state_dict(), strict=True)
        tensors = inputs("cuda")
        target = torch.randn(2, 64, 64, device="cuda", dtype=torch.bfloat16)
        outputs = []
        for candidate in (baseline, optimized):
            torch.manual_seed(71)
            options = (
                {"target_last_hidden_states": target}
                if isinstance(candidate, OnlineDSparkModel)
                else {"lambda_base": 0.37}
            )
            loss, accuracy, metrics = candidate(*tensors, **options)
            loss.backward()
            outputs.append((loss.detach(), accuracy.detach(), metrics))
        tolerance = dict(rtol=0.02, atol=0.02)
        torch.testing.assert_close(outputs[0][0], outputs[1][0], **tolerance)
        torch.testing.assert_close(outputs[0][1], outputs[1][1], **tolerance)
        torch.testing.assert_close(outputs[0][2], outputs[1][2], **tolerance)
        self.assertEqual(baseline.state_dict().keys(), optimized.state_dict().keys())
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
        expected = torch.cat(
            [value.float().flatten() for value in gradients[0].values()]
        )
        actual = torch.cat([value.float().flatten() for value in gradients[1].values()])
        self.assertLess(float((actual - expected).norm() / expected.norm()), 0.02)
        self.assertGreater(
            float(torch.nn.functional.cosine_similarity(actual, expected, dim=0)), 0.999
        )

    def test_liger_preserves_loss_and_gradients(self):
        for strategy in ("domino", "dspark"):
            with self.subTest(strategy=strategy):
                self.compare(model(strategy), model(strategy, liger=True))


if __name__ == "__main__":
    unittest.main()
