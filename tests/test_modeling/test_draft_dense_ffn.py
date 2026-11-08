"""Dense FFN retains projection names, computation order, and gradient updates."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from specforge.config.draft.mlp import DenseFFNSpec
from specforge.modeling.draft.layers.mlp import DenseFFN
from specforge.modeling.draft.layers.mlp.dense import SUPPORTED_ACTIVATIONS
from specforge.modeling.draft.llama3_eagle import LlamaMLP
from tests.test_config.test_draft_legacy_dflash import CONFIGS
from tests.test_modeling.test_draft_pretrained import DFlashDraftModel, model_config


class DenseFFNTest(unittest.TestCase):
    def _assert_parity(self, device, dtypes):
        for activation in sorted(SUPPORTED_ACTIVATIONS):
            for dtype in dtypes:
                for reference_type in (Qwen3MLP, LlamaMLP):
                    with self.subTest(
                        activation=activation,
                        dtype=dtype,
                        reference=reference_type.__name__,
                    ):
                        node = DenseFFNSpec(intermediate_size=24, hidden_act=activation)
                        before = hash(node)
                        expected = reference_type(
                            SimpleNamespace(
                                hidden_size=16,
                                intermediate_size=node.intermediate_size,
                                hidden_act=activation,
                                pretraining_tp=1,
                            )
                        ).to(device=device, dtype=dtype)
                        actual = DenseFFN(
                            node, hidden_size=16, device=device, dtype=dtype
                        )
                        actual.load_state_dict(expected.state_dict(), strict=True)
                        self.assertEqual(
                            set(actual.state_dict()),
                            {"gate_proj.weight", "up_proj.weight", "down_proj.weight"},
                        )
                        left = torch.randn(
                            2, 3, 16, device=device, dtype=dtype
                        ).requires_grad_()
                        right = left.detach().clone().requires_grad_()
                        expected_output, actual_output = expected(left), actual(right)
                        torch.testing.assert_close(
                            actual_output, expected_output, rtol=0, atol=0
                        )
                        upstream = torch.randn_like(expected_output)
                        expected_output.backward(upstream)
                        actual_output.backward(upstream)
                        torch.testing.assert_close(
                            right.grad, left.grad, rtol=0, atol=0
                        )
                        for x, y in zip(actual.parameters(), expected.parameters()):
                            torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
                        for module in (actual, expected):
                            torch.optim.SGD(module.parameters(), lr=0.01).step()
                        torch.testing.assert_close(
                            tuple(actual.parameters()),
                            tuple(expected.parameters()),
                            rtol=0,
                            atol=0,
                        )
                        self.assertIs(actual.config, node)
                        self.assertEqual(hash(node), before)

    def test_cpu_outputs_gradients_and_updates_match(self):
        self._assert_parity("cpu", (torch.float32, torch.bfloat16))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_outputs_gradients_and_updates_match(self):
        self._assert_parity("cuda", (torch.float32, torch.float16, torch.bfloat16))

    def test_all_bundled_activation_choices_are_supported(self):
        for path in CONFIGS.glob("*.json"):
            payload = json.loads(path.read_text())
            with self.subTest(path=path.name):
                self.assertIn(payload["hidden_act"], SUPPORTED_ACTIVATIONS)

    def test_generic_base_initializes_child_projections_from_golden_config(self):
        owner = DFlashDraftModel(model_config(initializer_range=0))
        node = DenseFFNSpec(intermediate_size=24, hidden_act="silu")
        with torch.device("meta"):
            module = DenseFFN(node, hidden_size=16)
        module.to_empty(device="cpu")
        module.apply(owner._initialize_weights)
        for parameter in module.parameters():
            self.assertEqual(torch.count_nonzero(parameter).item(), 0)

    def test_unsupported_settings_fail_before_projections_are_allocated(self):
        node = DenseFFNSpec(intermediate_size=24, hidden_act="prelu")
        with patch("specforge.modeling.draft.layers.mlp.dense.nn.Linear") as allocate:
            with self.assertRaisesRegex(ValueError, "hidden_act"):
                DenseFFN(node, hidden_size=16)
            for width in (0, True, 1.5):
                with self.assertRaises(ValueError):
                    DenseFFN(
                        DenseFFNSpec(intermediate_size=24, hidden_act="silu"),
                        hidden_size=width,
                    )
            with self.assertRaises(TypeError):
                DenseFFN({}, hidden_size=16)
            allocate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
