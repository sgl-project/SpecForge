import builtins
import contextlib
import copy
import os
import unittest
from unittest import mock

import torch

from specforge.config.draft.conv import GroupedCausalConvSpec
from specforge.modeling.draft.layers.conv import GroupedCausalConv

FUSED_CONV_ENV = "SPECFORGE_DFLASH2_FUSED_CONV"

_CUDA = torch.cuda.is_available()


def _eager_env():
    return mock.patch.dict(os.environ, {FUSED_CONV_ENV: "0"})


@contextlib.contextmanager
def _count_fused_calls():
    """Count calls that reach the fused kernels, so parity is not vacuous."""

    from specforge.modeling.draft.layers.conv import triton

    convolve = triton.dflash2_grouped_conv_fused
    calls = []

    def counted(*args):
        calls.append(args[0].shape)
        return convolve(*args)

    with mock.patch.object(triton, "dflash2_grouped_conv_fused", side_effect=counted):
        yield calls


def _random_conv(hidden_size, block_size, taps, group_size, *, device, dtype, seed):
    generator = torch.Generator().manual_seed(seed)
    conv = GroupedCausalConv(
        GroupedCausalConvSpec(kernel_size=taps, group_size=group_size),
        hidden_size=hidden_size,
        block_size=block_size,
    )
    with torch.no_grad():
        conv.base_kernel.copy_(torch.randn(conv.base_kernel.shape, generator=generator))
        conv.kernel_projection.weight.copy_(
            torch.randn(conv.kernel_projection.weight.shape, generator=generator)
            * hidden_size**-0.5
        )
    return conv.to(device=device, dtype=dtype)


def _prepare_finish_step(conv, inputs, mixer, grad_seed):
    """Run prepare -> mixer -> finish and return outputs plus every gradient."""

    inputs = inputs.detach().clone().requires_grad_(True)
    mixer = mixer.detach().clone().requires_grad_(True)
    conv.zero_grad(set_to_none=True)
    prepared, output_kernel = conv.prepare(inputs)
    finished = conv.finish(prepared * mixer, output_kernel)
    torch.autograd.backward((prepared, finished), grad_seed)
    return {
        "prepared": prepared.detach(),
        "finished": finished.detach(),
        "grad_inputs": inputs.grad,
        "grad_mixer": mixer.grad,
        "grad_base_kernel": conv.base_kernel.grad,
        "grad_kernel_projection": conv.kernel_projection.weight.grad,
    }


def _convolve_step(conv, inputs, delta, side, grad_output, *, delta_view=None):
    """Run one side of the convolution with an explicit dynamic kernel.

    ``delta_view`` maps the leaf ``delta`` to the tensor the convolution sees,
    so a caller can pass a strided slice like ``prepare`` does.
    """

    inputs = inputs.detach().clone().requires_grad_(True)
    delta = delta.detach().clone().requires_grad_(True)
    conv.zero_grad(set_to_none=True)
    kernel = delta if delta_view is None else delta_view(delta)
    output = conv._convolve(inputs, kernel, side=side)
    output.backward(grad_output)
    return {
        "output": output.detach(),
        "grad_inputs": inputs.grad,
        "grad_delta": delta.grad,
        "grad_base_kernel": conv.base_kernel.grad,
    }


def _errors(actual, expected):
    """Return max-abs, max-abs relative to the largest value, and RMS errors."""

    diff = (actual.double() - expected.double()).abs()
    scale = expected.double().abs().max().clamp_min(1e-12)
    return (
        diff.max().item(),
        (diff.max() / scale).item(),
        diff.square().mean().sqrt().item(),
    )


class DFlash2FusedConvDispatchTest(unittest.TestCase):
    def test_cpu_uses_eager_without_loading_triton(self):
        conv = _random_conv(8, 4, 2, 4, device="cpu", dtype=torch.float32, seed=0)
        with mock.patch("builtins.__import__", wraps=builtins.__import__) as imported:
            hidden, coefficients = conv.prepare(torch.randn(2, 8, 8))
            conv.finish(hidden, coefficients)
        self.assertFalse(
            any("triton" in call.args[0] for call in imported.call_args_list)
        )

    @unittest.skipUnless(_CUDA, "requires CUDA")
    def test_missing_triton_falls_back_to_eager(self):
        real_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name == "triton":
                raise ModuleNotFoundError(name="triton")
            return real_import(name, *args, **kwargs)

        conv = _random_conv(8, 4, 2, 4, device="cuda", dtype=torch.float32, seed=0)
        hidden = torch.randn(1, 8, 8, device="cuda")
        with _eager_env():
            expected, _ = conv.prepare(hidden)
        with mock.patch("builtins.__import__", guarded):
            actual, _ = conv.prepare(hidden)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@unittest.skipUnless(_CUDA, "fused DFlash2 convolution requires CUDA")
class DFlash2FusedConvParityTest(unittest.TestCase):
    def assert_step_close(self, actual, expected, *, tolerance):
        """Compare each tensor relative to its largest reference magnitude."""

        self.assertEqual(actual.keys(), expected.keys())
        for name in expected:
            with self.subTest(tensor=name):
                self.assertIsNotNone(actual[name])
                self.assertEqual(actual[name].dtype, expected[name].dtype)
                self.assertEqual(actual[name].shape, expected[name].shape)
                _, relative, _ = _errors(actual[name], expected[name])
                self.assertLessEqual(relative, tolerance)

    def test_fp32_prepare_finish_matches_eager(self):
        # (batch, num_blocks, block_size, taps, group_size, hidden_size)
        cases = [
            (1, 3, 4, 2, 4, 16),
            (2, 5, 8, 2, 16, 320),
            (2, 4, 8, 3, 16, 48),
            (1, 2, 16, 4, 64, 256),
            (2, 3, 8, 1, 1, 24),
            (2, 3, 5, 5, 8, 40),
            (1, 2, 8, 2, 512, 1024),
            (2, 64, 8, 2, 16, 5120),
        ]
        for index, case in enumerate(cases):
            batch, num_blocks, block_size, taps, group_size, hidden_size = case
            with self.subTest(case=case):
                conv = _random_conv(
                    hidden_size,
                    block_size,
                    taps,
                    group_size,
                    device="cuda",
                    dtype=torch.float32,
                    seed=index,
                )
                shape = (batch, num_blocks * block_size, hidden_size)
                inputs = torch.randn(shape, device="cuda")
                mixer = torch.randn(shape, device="cuda")
                grad_seed = (
                    torch.randn(shape, device="cuda"),
                    torch.randn(shape, device="cuda"),
                )

                with _count_fused_calls() as calls:
                    fused = _prepare_finish_step(conv, inputs, mixer, grad_seed)
                with _eager_env():
                    eager = _prepare_finish_step(conv, inputs, mixer, grad_seed)

                self.assertEqual(len(calls), 2)
                self.assert_step_close(fused, eager, tolerance=1e-5)

    def test_convolve_gradients_match_eager_for_both_sides(self):
        conv = _random_conv(64, 8, 3, 16, device="cuda", dtype=torch.float32, seed=7)
        inputs = torch.randn(2, 24, 64, device="cuda")
        # A strided view, like the slices ``prepare`` takes of its projection.
        delta = torch.randn(2, 24, 2, 3, 4, device="cuda")
        grad_output = torch.randn(2, 24, 64, device="cuda")

        for side in (0, 1):
            with self.subTest(side=side):

                def delta_view(leaf):
                    return leaf[:, :, side]

                fused = _convolve_step(
                    conv, inputs, delta, side, grad_output, delta_view=delta_view
                )
                with _eager_env():
                    eager = _convolve_step(
                        conv, inputs, delta, side, grad_output, delta_view=delta_view
                    )
                self.assert_step_close(fused, eager, tolerance=1e-5)
                self.assertEqual(conv.base_kernel.grad[1 - side].abs().sum(), 0)
                self.assertEqual(fused["grad_delta"][:, :, 1 - side].abs().sum(), 0)

    def test_inference_mode_matches_eager(self):
        # ``spec_generate`` runs the draft one block at a time under
        # inference mode.
        conv = _random_conv(64, 8, 2, 16, device="cuda", dtype=torch.float32, seed=5)
        inputs = torch.randn(3, 8, 64, device="cuda")

        def prepare_finish():
            with torch.inference_mode():
                prepared, output_kernel = conv.prepare(inputs)
                return prepared, conv.finish(prepared, output_kernel)

        with _count_fused_calls() as calls:
            fused = prepare_finish()
        with _eager_env():
            eager = prepare_finish()

        self.assertEqual(len(calls), 2)
        for actual, expected in zip(fused, eager):
            _, relative, _ = _errors(actual, expected)
            self.assertLessEqual(relative, 1e-5)

    def test_double_backward_raises_instead_of_dropping_terms(self):
        conv = _random_conv(32, 4, 2, 8, device="cuda", dtype=torch.float32, seed=2)
        inputs = torch.randn(1, 8, 32, device="cuda", requires_grad=True)
        delta = torch.randn(1, 8, 2, 4, device="cuda", requires_grad=True)

        with _count_fused_calls() as calls:
            output = conv._convolve(inputs, delta, side=0)
        self.assertEqual(len(calls), 1)
        (grad_inputs,) = torch.autograd.grad(
            output.square().sum(), inputs, create_graph=True
        )
        with self.assertRaisesRegex(RuntimeError, "once_differentiable"):
            (grad_inputs.sum() + output.sum()).backward()

    def test_fused_entry_rejects_mismatched_shapes(self):
        from specforge.modeling.draft.layers.conv.triton import (
            dflash2_grouped_conv_fused,
        )

        inputs = torch.randn(1, 8, 32, device="cuda")
        delta = torch.randn(1, 8, 2, 4, device="cuda")
        base = torch.randn(2, 32, device="cuda")
        with self.assertRaisesRegex(ValueError, "base must have shape"):
            dflash2_grouped_conv_fused(inputs, delta, base[:, :16], 4, 8)
        with self.assertRaisesRegex(ValueError, "multiples of block_size"):
            dflash2_grouped_conv_fused(inputs, delta, base, 3, 8)
        with self.assertRaises(RuntimeError):
            dflash2_grouped_conv_fused(inputs, delta[:, :, :1], base, 4, 8)

    def test_first_tap_positions_have_zero_shifted_gradients(self):
        conv = _random_conv(32, 4, 3, 8, device="cuda", dtype=torch.float32, seed=3)
        inputs = torch.randn(1, 12, 32, device="cuda")
        delta = torch.randn(1, 12, 3, 4, device="cuda")

        fused = _convolve_step(conv, inputs, delta, 0, torch.ones_like(inputs))

        grad_delta = fused["grad_delta"].view(1, 3, 4, 3, 4)
        for tap in range(1, 3):
            self.assertEqual(grad_delta[:, :, :tap, tap].abs().sum(), 0)
            self.assertGreater(grad_delta[:, :, tap:, tap].abs().sum(), 0)

    @unittest.skipUnless(
        _CUDA and torch.cuda.is_bf16_supported(), "BF16 parity requires BF16 CUDA"
    )
    def test_bf16_is_at_least_as_close_to_fp32_as_eager(self):
        # (batch, num_blocks, block_size, taps, group_size, hidden_size)
        cases = [
            (1, 16, 8, 2, 16, 512),
            (2, 64, 8, 2, 16, 5120),
            (2, 8, 16, 3, 32, 256),
        ]
        for index, case in enumerate(cases):
            batch, num_blocks, block_size, taps, group_size, hidden_size = case
            with self.subTest(case=case):
                conv = _random_conv(
                    hidden_size,
                    block_size,
                    taps,
                    group_size,
                    device="cuda",
                    dtype=torch.bfloat16,
                    seed=index,
                )
                # Evaluate the same BF16 parameters exactly in FP64.
                reference_conv = copy.deepcopy(conv).double()
                shape = (batch, num_blocks * block_size, hidden_size)
                inputs = torch.randn(shape, device="cuda").bfloat16()
                delta_shape = (batch, shape[1], taps, hidden_size // group_size)
                delta = torch.randn(delta_shape, device="cuda").bfloat16()
                grad_output = torch.randn(shape, device="cuda").bfloat16()

                with _count_fused_calls() as calls:
                    fused = _convolve_step(conv, inputs, delta, 0, grad_output)
                self.assertEqual(len(calls), 1)
                with _eager_env():
                    eager = _convolve_step(conv, inputs, delta, 0, grad_output)
                    reference = _convolve_step(
                        reference_conv,
                        inputs.double(),
                        delta.double(),
                        0,
                        grad_output.double(),
                    )

                for name in reference:
                    with self.subTest(tensor=name):
                        self.assertEqual(fused[name].dtype, torch.bfloat16)
                        _, _, fused_rms = _errors(fused[name], reference[name])
                        _, _, eager_rms = _errors(eager[name], reference[name])
                        self.assertLessEqual(fused_rms, eager_rms)
                        # FP32 accumulation plus one BF16 rounding per output.
                        torch.testing.assert_close(
                            fused[name].double(),
                            reference[name],
                            atol=1e-5 * reference[name].abs().max().item(),
                            rtol=2**-8,
                        )


@unittest.skipUnless(_CUDA, "requires CUDA")
class DFlash2FusedConvModelParityTest(unittest.TestCase):
    def test_composed_model_matches_eager(self):
        from specforge.modeling import build_draft_model
        from tests.test_modeling.test_canonical_dflash2 import (
            dflash2_config,
            model_inputs,
        )

        model = build_draft_model(dflash2_config(), device="cuda")
        with torch.no_grad():
            for layer in model.backbone.layers:
                for conv in (layer.attention_conv, layer.mlp_conv):
                    conv.kernel_projection.weight.normal_(std=0.03)
        args, _ = model_inputs(model.config.draft, "cuda")
        outputs, gradients = [], []
        for enabled in ("0", "1"):
            model.zero_grad(set_to_none=True)
            with mock.patch.dict(os.environ, {FUSED_CONV_ENV: enabled}):
                output = model(*args).logits
                output.square().mean().backward()
            outputs.append(output.detach())
            gradients.append(
                {
                    n: p.grad.clone()
                    for n, p in model.named_parameters()
                    if p.grad is not None
                }
            )
        torch.testing.assert_close(*outputs, atol=2e-5, rtol=2e-5)
        self.assertEqual(gradients[0].keys(), gradients[1].keys())
        for name in gradients[0]:
            torch.testing.assert_close(
                gradients[0][name], gradients[1][name], atol=3e-5, rtol=3e-5
            )
