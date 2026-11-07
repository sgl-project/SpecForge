"""DSpark's confidence head on the pre-norm readout (DeepSeek-V4) and in fp32."""

import unittest
from unittest import mock

import torch
from torch import nn
from torch.testing import assert_close
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.dspark import AcceptRatePredictor, DSparkDraftModel


def _config(confidence_head_input=None):
    dflash_config = {
        "attention_mode": "gqa",
        "target_layer_ids": [1],
        "projector_type": "dspark",
        "markov_rank": 4,
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "mask_token_id": 31,
    }
    if confidence_head_input is not None:
        dflash_config["confidence_head_input"] = confidence_head_input
    config = Qwen3Config(
        architectures=["DSparkDraftModel"],
        block_size=3,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        num_target_layers=4,
        head_dim=4,
        max_position_embeddings=64,
        vocab_size=32,
        layer_types=["full_attention"],
        attention_dropout=0.0,
        dflash_config=dflash_config,
    )
    config._attn_implementation = "sdpa"
    return config


class ConfidenceInputTest(unittest.TestCase):
    def test_forward_returns_the_pre_norm_readout_on_request(self):
        model = DFlashDraftModel(_config())
        inputs = dict(
            position_ids=torch.arange(8)[None],
            noise_embedding=torch.randn(1, 3, 16),
            target_hidden=torch.randn(1, 5, 16),
        )
        post = model(**inputs)
        normed, pre = model(**inputs, return_pre_norm_hidden=True)
        assert_close(normed, post, atol=0, rtol=0)
        assert_close(model.norm(pre), post, atol=0, rtol=0)

    def test_pre_norm_input_reaches_the_confidence_head_chunk_by_chunk(self):
        for confidence_head_input in ("pre_norm", None):
            with self.subTest(confidence_head_input=confidence_head_input):
                draft = DSparkDraftModel(_config(confidence_head_input))
                self.assertEqual(
                    draft.confidence_head_input, confidence_head_input or "post_norm"
                )
                seen = []
                original = draft.predict_confidence

                def recording(hidden_states, **kwargs):
                    seen.append((hidden_states, kwargs.get("pre_norm_hidden_states")))
                    return original(hidden_states, **kwargs)

                model = OnlineDSparkModel(
                    draft,
                    nn.Linear(16, 32, bias=False).requires_grad_(False),
                    nn.Embedding(32, 16).requires_grad_(False),
                    mask_token_id=31,
                    block_size=3,
                    attention_backend="sdpa",
                    num_anchors=4,
                    dspark_confidence_head_alpha=1.0,
                    objective_chunk_blocks=2,
                )
                with mock.patch.object(
                    draft, "predict_confidence", side_effect=recording
                ):
                    loss, _, _ = model(
                        input_ids=torch.randint(0, 30, (2, 12)),
                        hidden_states=torch.randn(2, 12, 16),
                        loss_mask=torch.ones(2, 12),
                        target_last_hidden_states=torch.randn(2, 12, 16),
                    )
                loss.backward()
                self.assertTrue(seen)
                for post, pre in seen:
                    if confidence_head_input == "pre_norm":
                        # The chunk of the readout whose norm is the chunk the logits use.
                        assert_close(draft.norm(pre), post)
                    else:
                        self.assertIsNone(pre)
                self.assertIsNotNone(draft.confidence_head.proj.weight.grad)

    def test_pre_norm_mode_requires_the_readout_and_rejects_other_values(self):
        draft = DSparkDraftModel(_config("pre_norm"))
        with self.assertRaisesRegex(ValueError, "pre_norm"):
            draft.predict_confidence(
                torch.randn(1, 3, 16),
                prev_token_ids=torch.zeros(1, 3, dtype=torch.long),
            )
        with self.assertRaisesRegex(ValueError, "confidence_head_input"):
            DSparkDraftModel(_config("sideways"))

    def test_confidence_projection_computes_in_fp32(self):
        head = AcceptRatePredictor(8, bias=False).bfloat16()
        features = (torch.randn(2, 8) * 300).bfloat16()  # an unnormalized readout
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = head(features)
        self.assertEqual(output.dtype, torch.float32)
        expected = (features.float() @ head.proj.weight.float().T).squeeze(-1)
        assert_close(output, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
