"""Runtime kernels preserve composed component weights, outputs and gradients."""

import copy
import os
import unittest
from unittest.mock import patch

import torch

from specforge.config.draft.conv import GroupedCausalConvSpec
from specforge.modeling import build_draft_model
from specforge.modeling.draft.layers.conv import GroupedCausalConv
from specforge.training.kernels import configure_liger
from tests.test_algorithms.test_objective_optimizations import evaluate
from tests.test_algorithms.test_objectives import training_batch
from tests.test_config.draft_fixtures import draft_config


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class ComponentKernelTest(unittest.TestCase):
    def test_liger_preserves_loss_gradients_and_state_keys(self):
        config = draft_config(target_bindings=(), target_logit_transform=None)
        eager = build_draft_model(config, device="cuda")
        fused = copy.deepcopy(eager)
        configure_liger(fused, True, default_enabled=False)
        self.assertEqual(eager.state_dict().keys(), fused.state_dict().keys())
        batch = training_batch(config, "cuda")
        options = dict(num_anchors=2, teacher_metrics=False)
        left, right = (
            evaluate(model, batch, None, options) for model in (eager, fused)
        )
        torch.testing.assert_close(left[0], right[0])
        for name in left[1]:
            torch.testing.assert_close(
                left[1][name], right[1][name], atol=3e-5, rtol=3e-5
            )

    def test_grouped_convolution_fused_forward_and_backward(self):
        module = GroupedCausalConv(
            GroupedCausalConvSpec(kernel_size=3, group_size=4),
            hidden_size=32,
            block_size=4,
            device="cuda",
        )
        with torch.no_grad():
            module.kernel_projection.weight.normal_(std=0.02)
        x = torch.randn(2, 12, 32, device="cuda", requires_grad=True)
        outputs, gradients = [], []
        for enabled in ("0", "1"):
            with patch.dict(os.environ, SPECFORGE_DFLASH2_FUSED_CONV=enabled):
                prepared, coefficients = module.prepare(x)
                output = module.finish(prepared, coefficients)
                outputs.append(output)
                gradients.append(
                    torch.autograd.grad(
                        output.square().mean(), (x, *module.parameters())
                    )
                )
        torch.testing.assert_close(*outputs, atol=2e-5, rtol=2e-5)
        for left, right in zip(*gradients):
            torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
