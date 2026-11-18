"""SDPA never receives fully masked rows, including padded MTP queries."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from specforge.algorithms.objectives import build_objective_model
from specforge.modeling import build_draft_model
from tests.test_algorithms.test_objectives import training_batch
from tests.test_modeling.test_mtp_sdpa_chunking import owned_mtp_config


class SDPAEmptyRowsTest(unittest.TestCase):
    def test_padding_is_safe_in_forward_and_backward(self):
        config = owned_mtp_config()
        batch = training_batch(config)
        batch["attention_mask"][:, -3:] = 0
        reference = build_draft_model(config)
        model = build_draft_model(config, backend="sdpa")
        model.load_state_dict(reference.state_dict())
        original = F.scaled_dot_product_attention

        def checked(*args, **kwargs):
            self.assertTrue(kwargs["attn_mask"].any(-1).all())
            return original(*args, **kwargs)

        expected = build_objective_model(reference)(batch, step=0, total_steps=None)
        with patch.object(
            F, "scaled_dot_product_attention", side_effect=checked
        ) as sdpa:
            actual = build_objective_model(model)(batch, step=0, total_steps=None)
        self.assertTrue(sdpa.called)
        torch.testing.assert_close(actual.loss, expected.loss)
        actual.loss.backward()
        expected.loss.backward()
        for actual_param, expected_param in zip(
            model.parameters(), reference.parameters()
        ):
            torch.testing.assert_close(
                actual_param.grad, expected_param.grad, atol=2e-6, rtol=3e-4
            )
