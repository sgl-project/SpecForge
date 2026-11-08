"""Block-local convolution preserves eager DFlash2 numerics and HF loading."""

import copy
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from torch.utils.checkpoint import checkpoint

from specforge.config.draft.conv import GroupedCausalConvSpec
from specforge.modeling.draft.dflash2 import DFlashGroupedConv
from specforge.modeling.draft.layers import GroupedCausalConv, build_component
from specforge.modeling.draft.pretrained import DraftPreTrainedModel
from tests.test_modeling.test_draft_pretrained import model_config


def make_conv(**kwargs):
    return build_component(
        "conv",
        GroupedCausalConvSpec(kernel_size=3, group_size=4),
        hidden_size=16,
        block_size=4,
        **kwargs,
    )


def run_pair(module, hidden):
    prepared, coefficients = module.prepare(hidden)
    return module.finish(torch.tanh(prepared), coefficients)


class DFlashDraftModel(DraftPreTrainedModel):
    """Test-only model for generic HF initialization and weight restoration."""

    def __init__(self, config):
        super().__init__(config)
        self.conv = make_conv()
        self.post_init()


class GroupedCausalConvTest(unittest.TestCase):
    def _parity(self, device, dtypes, autocast=False):
        for dtype in dtypes:
            for group, taps in ((1, 1), (4, 3), (16, 4)):
                with self.subTest(device=device, dtype=dtype, group=group, taps=taps):
                    node = GroupedCausalConvSpec(kernel_size=taps, group_size=group)
                    actual = GroupedCausalConv(
                        node, hidden_size=16, block_size=4, device=device, dtype=dtype
                    )
                    expected = DFlashGroupedConv(16, 4, taps, group).to(
                        device=device, dtype=dtype
                    )
                    with torch.no_grad():
                        for parameter in expected.parameters():
                            parameter.normal_(std=0.1)
                    actual.load_state_dict(expected.state_dict(), strict=True)
                    # Non-contiguous input, multiple batches and blocks.
                    left = torch.randn(2, 16, 8, device=device, dtype=dtype)
                    left = left.transpose(1, 2).requires_grad_()
                    right = left.detach().clone().requires_grad_()
                    with patch.object(
                        expected, "_fused_convolution", return_value=None
                    ):
                        with torch.autocast(
                            device, dtype=torch.bfloat16, enabled=autocast
                        ):
                            reference = run_pair(expected, left)
                            output = run_pair(actual, right)
                    torch.testing.assert_close(output, reference, rtol=0, atol=0)
                    upstream = torch.randn_like(output)
                    reference.backward(upstream)
                    output.backward(upstream)
                    torch.testing.assert_close(right.grad, left.grad, rtol=0, atol=0)
                    for name, parameter in actual.named_parameters():
                        self.assertIsNotNone(parameter.grad)
                        torch.testing.assert_close(
                            parameter.grad,
                            dict(expected.named_parameters())[name].grad,
                            rtol=0,
                            atol=0,
                        )
                    self.assertIs(actual.config, node)

    def test_cpu_output_and_gradients_match_eager_reference(self):
        self._parity("cpu", (torch.float32, torch.bfloat16))
        self._parity("cpu", (torch.float32,), autocast=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_output_and_gradients_match_eager_reference(self):
        self._parity("cuda", (torch.float32, torch.float16, torch.bfloat16))
        self._parity("cuda", (torch.float32,), autocast=True)

    def test_causality_and_block_isolation_include_dynamic_coefficients(self):
        module = make_conv()
        with torch.no_grad():
            module.base_kernel.fill_(0.5)
            module.kernel_projection.weight.fill_(0.01)
        hidden = torch.randn(1, 8, 16, requires_grad=True)
        output = run_pair(module, hidden)
        changed = hidden.detach().clone()
        changed[:, 3:] += 100
        torch.testing.assert_close(run_pair(module, changed)[:, :3], output[:, :3])
        changed = hidden.detach().clone()
        changed[:, :4] += 100
        torch.testing.assert_close(run_pair(module, changed)[:, 4:], output[:, 4:])
        output[:, 5].sum().backward()
        self.assertEqual(torch.count_nonzero(hidden.grad[:, :4]).item(), 0)
        self.assertEqual(torch.count_nonzero(hidden.grad[:, 6:]).item(), 0)
        self.assertGreater(torch.count_nonzero(hidden.grad[:, 4:6]).item(), 0)

    def test_interleaved_forwards_and_checkpointing_keep_coefficients_local(self):
        module = make_conv()
        with torch.no_grad():
            module.kernel_projection.weight.normal_(std=0.1)
        checked = copy.deepcopy(module)
        left = torch.randn(2, 8, 16, requires_grad=True)
        right = left.detach().clone().requires_grad_()
        prepared, coefficients = module.prepare(left)
        module.prepare(torch.randn_like(left))
        expected = module.finish(torch.tanh(prepared), coefficients)
        actual = checkpoint(run_pair, checked, right, use_reentrant=False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected.square().sum().backward()
        actual.square().sum().backward()
        for a, b in zip((right, *checked.parameters()), (left, *module.parameters())):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_identity_initialization_meta_and_partial_restore(self):
        module = make_conv(device="meta")
        module.to_empty(device="cpu")
        module.reset_draft_parameters()
        hidden = torch.randn(2, 8, 16)
        prepared, coefficients = module.prepare(hidden)
        torch.testing.assert_close(prepared, hidden, rtol=0, atol=0)
        torch.testing.assert_close(
            module.finish(hidden, coefficients), hidden, rtol=0, atol=0
        )
        for loaded_name in ("base_kernel", "kernel_projection.weight"):
            module = make_conv()
            with torch.no_grad():
                for name, parameter in module.named_parameters():
                    parameter.fill_(7)
                    parameter._is_hf_initialized = name == loaded_name
            module.reset_draft_parameters()
            fresh = make_conv()
            for name, parameter in module.named_parameters():
                expected = (
                    torch.full_like(parameter, 7)
                    if name == loaded_name
                    else fresh.state_dict()[name]
                )
                torch.testing.assert_close(parameter, expected, rtol=0, atol=0)

    def test_hf_initialization_and_roundtrip_preserve_weights(self):
        model = DFlashDraftModel(model_config())
        hidden = torch.randn(2, 8, 16)
        torch.testing.assert_close(
            run_pair(model.conv, hidden), torch.tanh(hidden), rtol=0, atol=0
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_(std=0.1)
        expected = run_pair(model.conv, hidden)
        model.post_init()
        torch.testing.assert_close(
            run_pair(model.conv, hidden), expected, rtol=0, atol=0
        )
        with TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = DFlashDraftModel.from_pretrained(directory)
        torch.testing.assert_close(
            restored.state_dict(), model.state_dict(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            run_pair(restored.conv, hidden), expected, rtol=0, atol=0
        )

    def test_invalid_dimensions_fail_before_allocation(self):
        node = GroupedCausalConvSpec(kernel_size=3, group_size=4)
        with patch(
            "specforge.modeling.draft.layers.conv.grouped_causal.torch.empty"
        ) as allocate:
            for width, block in (
                (0, 4),
                (True, 4),
                (15, 4),
                (16, 0),
                (16, 2),
                (16, 4.0),
            ):
                with self.assertRaises(ValueError):
                    GroupedCausalConv(node, hidden_size=width, block_size=block)
            with self.assertRaises(TypeError):
                GroupedCausalConv({}, hidden_size=16, block_size=4)
            allocate.assert_not_called()
        module = make_conv()
        for shape in ((1, 7, 16), (1, 8, 8), (8, 16)):
            with self.assertRaises(ValueError):
                module.prepare(torch.randn(shape))
        with self.assertRaisesRegex(ValueError, "coefficients"):
            module.finish(torch.randn(1, 8, 16), torch.randn(1, 4, 3, 4))


if __name__ == "__main__":
    unittest.main()
