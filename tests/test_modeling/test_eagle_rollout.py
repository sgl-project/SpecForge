"""Cached recurrent execution matches the general model forward and gradients."""

import copy
import unittest
from dataclasses import replace

import torch

from specforge.modeling import build_draft_model
from specforge.modeling.draft.models.eagle_rollout import FullEagleRollout
from tests.test_modeling.test_canonical_eagle3 import model_config


class EagleRolloutTest(unittest.TestCase):
    def test_cache_matches_full_forward_with_checkpoint_recomputation(self):
        self.check_backends(
            torch.float32,
            (
                [None, "eager", "sdpa", "flex_attention"]
                if torch.cuda.is_available()
                else [None, "eager", "sdpa"]
            ),
            3e-5,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA FlashAttention")
    def test_flash_cached_rollout_matches_sdpa(self):
        self.check_backends(torch.bfloat16, ["sdpa", "fa"], 0.025)

    def check_backends(self, dtype, backends, tolerance):
        for norm_output in (False, True):
            with self.subTest(norm_output=norm_output):
                self._check_backends(dtype, backends, tolerance, norm_output)

    def _check_backends(self, dtype, backends, tolerance, norm_output):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config = model_config(norm_output=norm_output)
        config = replace(
            config,
            attention_configs={
                name: replace(node, head_dim=16)
                for name, node in config.attention_configs.items()
            },
        )
        original = build_draft_model(config, device=device, dtype=dtype)
        ids = torch.arange(8, device=device)[None]
        features = torch.randn(
            1,
            8,
            original.config.draft.spec_config.target_features.input_width,
            device=device,
            dtype=dtype,
        )
        padding = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]], device=device)
        results = []
        for backend in backends:
            model = copy.deepcopy(original)
            model.gradient_checkpointing_enable()
            rollout = (
                FullEagleRollout(model)
                if backend is None
                else model.new_rollout(backend)
            )
            hidden = model.fusion(features)
            outputs = []
            for depth in range(3):
                hidden = rollout.step(ids + depth, hidden, padding, ids)
                outputs.append(hidden[:, :4])
            output = torch.stack(outputs)
            output.square().mean().backward()
            results.append(
                (
                    output,
                    {
                        name: p.grad
                        for name, p in model.named_parameters()
                        if p.grad is not None
                    },
                )
            )
        reference, gradients = results[0]
        for output, grads in results[1:]:
            torch.testing.assert_close(
                output, reference, atol=tolerance, rtol=tolerance
            )
            self.assertEqual(grads.keys(), gradients.keys())
            for name in grads:
                torch.testing.assert_close(
                    grads[name], gradients[name], atol=tolerance, rtol=tolerance
                )


if __name__ == "__main__":
    unittest.main()
