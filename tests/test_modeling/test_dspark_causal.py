import unittest
from itertools import product
from unittest.mock import patch

import torch
from torch import nn
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
from specforge.algorithms.dspark.providers import resume_contract
from specforge.modeling.draft.dspark import DSparkDraftModel


def build_model(causal=True, backend="sdpa", device="cpu"):
    config = Qwen3Config(
        architectures=["DSparkDraftModel"],
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        num_target_layers=4,
        head_dim=32,
        vocab_size=64,
        block_size=4,
        layer_types=["full_attention"] * 2,
        is_causal=causal,
        dflash_config={
            "projector_type": "dspark",
            "target_layer_ids": [0, 2],
            "mask_token_id": 63,
            "markov_head_type": "vanilla",
            "markov_rank": 8,
            "enable_confidence_head": False,
            "confidence_head_alpha": 0.0,
        },
        attn_implementation=backend,
    )
    draft = DSparkDraftModel(config)
    head = nn.Linear(128, 64, bias=False).requires_grad_(False)
    embedding = nn.Embedding(64, 128).requires_grad_(False)
    return OnlineDSparkModel(
        draft,
        head,
        embedding,
        mask_token_id=63,
        block_size=4,
        attention_backend=backend,
        num_anchors=2,
        dspark_confidence_head_alpha=0.0,
        objective_chunk_blocks=1,
    ).to(device)


class DSparkCausalTest(unittest.TestCase):

    def test_training_mask_and_future_isolation(self):
        for backend, causal in product(["sdpa", "flex_attention"], [False, True]):
            with self.subTest(backend=backend, causal=causal):
                if backend == "flex_attention" and (not torch.cuda.is_available()):
                    self.skipTest("Flex Attention requires CUDA")
                device = "cuda" if backend == "flex_attention" else "cpu"
                torch.manual_seed(42)
                model = build_model(causal, backend, device)
                tokens = torch.randint(0, 62, (1, 12), device=device)
                hidden = torch.randn(1, 12, 256, device=device)
                anchors = torch.tensor([[2, 7]], device=device)
                keep = torch.ones_like(anchors, dtype=torch.bool)
                noise = torch.randn(1, 8, 128, device=device)
                changed = noise.clone()
                changed[:, 2:] += 10 * torch.randn_like(changed[:, 2:])
                outputs = []
                with patch.object(
                    model, "_sample_anchor_positions", return_value=(anchors, keep)
                ):
                    for embeddings in (noise, changed):
                        with patch.object(
                            model, "_create_noise_embed", return_value=embeddings
                        ):
                            with torch.no_grad():
                                outputs.append(
                                    model._forward_draft_blocks(
                                        tokens, hidden, torch.ones_like(tokens)
                                    )[2]
                                )
                if causal:
                    torch.testing.assert_close(
                        outputs[0][:, :2], outputs[1][:, :2], atol=1e-06, rtol=1e-06
                    )
                else:
                    self.assertFalse(
                        torch.allclose(outputs[0][:, :2], outputs[1][:, :2])
                    )
                self.assertFalse(torch.allclose(outputs[0][:, 2:], outputs[1][:, 2:]))

    def test_no_confidence_parameters_and_markov_backward(self):
        model = build_model()
        self.assertIs(model.draft_model.confidence_head, None)
        self.assertFalse(
            any(("confidence" in name for (name, _) in model.named_parameters()))
        )
        tokens = torch.randint(0, 62, (2, 12))
        (loss, accuracy, metrics) = model(
            tokens,
            torch.randn(2, 12, 256),
            torch.ones_like(tokens),
            target_last_hidden_states=torch.randn(2, 12, 128),
        )
        self.assertTrue(torch.isfinite(loss) and torch.isfinite(accuracy))
        self.assertEqual(metrics["ratio_metrics"]["confidence_loss"][0].item(), 0)
        loss.backward()
        for name, parameter in model.draft_model.named_parameters():
            self.assertIsNot(parameter.grad, None, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(
            model.draft_model.markov_head.markov_w2.weight.grad.abs().sum(), 0
        )
        contract = resume_contract(None, model.draft_model, model)
        self.assertIs(contract["dspark_causal_block"], True)
        self.assertIs(model.draft_model.config.to_dict()["is_causal"], True)


if __name__ == "__main__":
    unittest.main()
