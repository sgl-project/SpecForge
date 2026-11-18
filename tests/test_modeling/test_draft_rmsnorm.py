"""Node-driven RMSNorm preserves both existing parameterizations exactly."""

import unittest
from unittest.mock import patch

import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from specforge.config.draft.norm import RMSNormSpec
from specforge.modeling.draft.layers.norm import RMSNorm
from tests.test_modeling.mtp_reference import OffsetRMSNorm


class RMSNormTest(unittest.TestCase):
    def _assert_parity(self, device, dtypes):
        for reference_type, offset in (
            (Qwen3RMSNorm, 0),
            (LlamaRMSNorm, 0),
            (OffsetRMSNorm, 1),
        ):
            for input_dtype in dtypes:
                for weight_dtype in (torch.float32, input_dtype):
                    with self.subTest(
                        reference=reference_type.__name__,
                        input=input_dtype,
                        weight=weight_dtype,
                    ):
                        node = RMSNormSpec(
                            has_weight=True, eps=1e-5, weight_offset=offset
                        )
                        before = hash(node)
                        actual = RMSNorm(
                            node, hidden_size=16, device=device, dtype=weight_dtype
                        )
                        expected = reference_type(16, eps=node.eps).to(
                            device=device, dtype=weight_dtype
                        )
                        with torch.no_grad():
                            expected.weight.normal_()
                        actual.load_state_dict(expected.state_dict(), strict=True)
                        left = torch.randn(
                            2, 3, 16, device=device, dtype=input_dtype
                        ).requires_grad_()
                        right = left.detach().clone().requires_grad_()
                        expected_output = expected(left)
                        actual_output = actual(right)
                        torch.testing.assert_close(
                            actual_output, expected_output, rtol=0, atol=0
                        )
                        upstream = torch.randn_like(expected_output)
                        expected_output.backward(upstream)
                        actual_output.backward(upstream)
                        torch.testing.assert_close(
                            right.grad, left.grad, rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            actual.weight.grad, expected.weight.grad, rtol=0, atol=0
                        )
                        for module in (actual, expected):
                            torch.optim.SGD(module.parameters(), lr=0.01).step()
                        torch.testing.assert_close(
                            actual.weight, expected.weight, rtol=0, atol=0
                        )
                        self.assertIs(actual.config, node)
                        self.assertEqual(hash(node), before)

    def test_cpu_outputs_gradients_and_updates_match(self):
        self._assert_parity("cpu", (torch.float32, torch.bfloat16))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_outputs_gradients_and_updates_match(self):
        self._assert_parity("cuda", (torch.float32, torch.float16, torch.bfloat16))

    def test_initialization_and_meta_materialization_use_the_node(self):
        for offset in (0, 1):
            node = RMSNormSpec(has_weight=True, eps=1e-6, weight_offset=offset)
            module = RMSNorm(node, hidden_size=8)
            torch.testing.assert_close(module.weight, torch.full((8,), 1.0 - offset))
            with torch.device("meta"):
                module = RMSNorm(node, hidden_size=8)
            module.to_empty(device="cpu")
            module.reset_draft_parameters()
            torch.testing.assert_close(module.weight, torch.full((8,), 1.0 - offset))
            with torch.no_grad():
                module.weight.fill_(7)
            module.weight._is_hf_initialized = True
            module.reset_draft_parameters()
            torch.testing.assert_close(module.weight, torch.full((8,), 7.0))

    def test_invalid_inputs_fail_before_parameter_allocation(self):
        node = RMSNormSpec(has_weight=True, eps=1e-6, weight_offset=0)
        with patch("specforge.modeling.draft.layers.norm.rms.torch.empty") as allocate:
            for width in (0, -1, True, 1.5):
                with self.assertRaises(ValueError):
                    RMSNorm(node, hidden_size=width)
            with self.assertRaises(TypeError):
                RMSNorm({"eps": 1e-6}, hidden_size=8)
            allocate.assert_not_called()

    def test_liger_skips_weightless_norms(self):
        """Liger RMSNorm needs a weight; a weightless norm keeps the PyTorch path."""
        from specforge.modeling.draft.layers.kernels import configure_kernels

        weightless = RMSNorm(
            RMSNormSpec(eps=1e-6, weight_offset=0, has_weight=False), hidden_size=8
        )
        model = torch.nn.Sequential(weightless)
        kernels = (object(), object())
        with patch(
            "specforge.modeling.draft.layers.kernels.load_liger_kernels",
            return_value=kernels,
        ):
            configure_kernels(model, use_liger=True)
        self.assertIsNone(weightless.liger_norm)


if __name__ == "__main__":
    unittest.main()
