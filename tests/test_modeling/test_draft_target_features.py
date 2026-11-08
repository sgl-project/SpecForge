"""Target fusion preserves the existing projection-then-norm computation."""

import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from specforge.modeling.draft.models.target_features import TargetFeatureFusion
from tests.test_config.draft_fixtures import draft_config
from tests.test_modeling.test_draft_pretrained import DFlashDraftModel, model_config


class TargetFeatureTest(unittest.TestCase):
    def _parity(self, device, dtype, autocast=False):
        config = draft_config()
        node = config.spec_config.target_features
        actual = TargetFeatureFusion(
            node,
            target_hidden_size=24,
            hidden_size=32,
            norm=config.norm_config,
            device=device,
            dtype=dtype,
        )
        reference = nn.Sequential(
            nn.Linear(48, 32, bias=False), Qwen3RMSNorm(32, eps=1e-6)
        ).to(device=device, dtype=dtype)
        reference[0].load_state_dict(actual.fc.state_dict())
        reference[1].load_state_dict(actual.hidden_norm.state_dict())
        features = torch.randn(2, 5, 48, device=device, dtype=dtype, requires_grad=True)
        other = features.detach().clone().requires_grad_()
        with (
            torch.autocast(device, dtype=torch.bfloat16) if autocast else nullcontext()
        ):
            output, expected = actual(features), reference(other)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        upstream = torch.randn_like(output)
        output.backward(upstream)
        expected.backward(upstream)
        torch.testing.assert_close(features.grad, other.grad, rtol=0, atol=0)
        for parameter, ref in zip(actual.parameters(), reference.parameters()):
            self.assertIsNotNone(parameter.grad)
            torch.testing.assert_close(parameter.grad, ref.grad, rtol=0, atol=0)
        self.assertIs(actual.config, node)
        self.assertEqual(
            actual(torch.empty(2, 0, 48, device=device, dtype=dtype)).shape, (2, 0, 32)
        )

    def test_cpu_projection_and_gradient_parity(self):
        for dtype in (torch.float32, torch.bfloat16):
            self._parity("cpu", dtype)
        self._parity("cpu", torch.float32, autocast=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_projection_and_gradient_parity(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            self._parity("cuda", dtype)
        self._parity("cuda", torch.float32, autocast=True)

    def test_meta_materialization_and_generic_initialization(self):
        config = draft_config()
        module = TargetFeatureFusion(
            config.spec_config.target_features,
            target_hidden_size=32,
            hidden_size=32,
            norm=config.norm_config,
            device="meta",
            dtype=torch.bfloat16,
        )
        self.assertTrue(all(p.is_meta for p in module.parameters()))
        module.to_empty(device="cpu")
        module.apply(
            DFlashDraftModel(model_config(initializer_range=0))._initialize_weights
        )
        torch.testing.assert_close(module.fc.weight, torch.zeros_like(module.fc.weight))
        torch.testing.assert_close(
            module.hidden_norm.weight, torch.ones_like(module.hidden_norm.weight)
        )
        self.assertEqual(module.fc.weight.dtype, torch.bfloat16)

    def test_bad_dimensions_fail_before_projection(self):
        config = draft_config()
        args = dict(hidden_size=32, norm=config.norm_config)
        with patch("torch.nn.Linear") as allocate:
            with self.assertRaises(ValueError):
                TargetFeatureFusion(
                    config.spec_config.target_features, target_hidden_size=0, **args
                )
            allocate.assert_not_called()
        module = TargetFeatureFusion(
            config.spec_config.target_features, target_hidden_size=24, **args
        )
        with patch.object(module.fc, "forward") as project:
            for shape in ((2, 48), (2, 5, 64)):
                with self.assertRaises(ValueError):
                    module(torch.empty(shape))
            project.assert_not_called()


if __name__ == "__main__":
    unittest.main()
