import builtins
import contextlib
import copy
import os
import sys
import unittest
from unittest import mock

import torch
from transformers import Qwen3Config

from specforge.modeling.draft import dflash2
from specforge.modeling.draft.dflash2 import (
    FUSED_CONV_ENV,
    DFlash2DraftModel,
    DFlashGroupedConv,
)

_CUDA = torch.cuda.is_available()


def _eager_env():
    return mock.patch.dict(os.environ, {FUSED_CONV_ENV: "0"})


@contextlib.contextmanager
def _count_fused_calls():
    """Count calls that reach the fused kernels, so parity is not vacuous."""

    convolve, supports_group_size = dflash2._load_fused_grouped_conv()
    calls = []

    def counted(*args):
        calls.append(args[0].shape)
        return convolve(*args)

    with mock.patch.object(
        dflash2,
        "_load_fused_grouped_conv",
        return_value=(counted, supports_group_size),
    ):
        yield calls


def _random_conv(hidden_size, block_size, taps, group_size, *, device, dtype, seed):
    generator = torch.Generator().manual_seed(seed)
    conv = DFlashGroupedConv(hidden_size, block_size, taps, group_size)
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
        inputs = torch.randn(2, 8, 8)

        with mock.patch.object(dflash2, "_load_fused_grouped_conv") as load:
            prepared, output_kernel = conv.prepare(inputs)
            conv.finish(prepared, output_kernel)

        load.assert_not_called()

    def test_missing_triton_disables_fused_path(self):
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "triton" or name.startswith("triton."):
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return real_import(name, *args, **kwargs)

        dflash2._load_fused_grouped_conv.cache_clear()
        try:
            with (
                mock.patch.dict(sys.modules),
                mock.patch("builtins.__import__", guarded_import),
            ):
                sys.modules.pop("specforge.modeling.draft.dflash2_conv_triton", None)
                self.assertIsNone(dflash2._load_fused_grouped_conv())
        finally:
            dflash2._load_fused_grouped_conv.cache_clear()

    @unittest.skipUnless(_CUDA, "fused DFlash2 convolution requires CUDA")
    def test_selects_fused_path_only_for_supported_inputs(self):
        def selected(group_size, activation_dtype, parameter_dtype, env="1"):
            conv = _random_conv(
                48, 4, 2, group_size, device="cuda", dtype=parameter_dtype, seed=0
            )
            inputs = torch.randn(1, 8, 48, device="cuda", dtype=activation_dtype)
            delta = torch.randn(
                1, 8, 2, 48 // group_size, device="cuda", dtype=activation_dtype
            )
            with mock.patch.dict(os.environ, {FUSED_CONV_ENV: env}):
                return conv._fused_convolution(inputs, delta, conv.base_kernel[0])

        self.assertIsNotNone(selected(16, torch.bfloat16, torch.bfloat16))
        self.assertIsNotNone(selected(16, torch.float32, torch.float32))
        self.assertIsNone(selected(16, torch.bfloat16, torch.bfloat16, env="0"))
        # Non-power-of-two groups do not tile.
        self.assertIsNone(selected(3, torch.float32, torch.float32))
        # Mixed dtypes promote in the eager path, so they stay there.
        self.assertIsNone(selected(16, torch.float32, torch.bfloat16))


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


@unittest.skipUnless(_CUDA, "fused DFlash2 convolution requires CUDA")
class DFlash2FusedConvModelParityTest(unittest.TestCase):
    def _config(self):
        config = Qwen3Config(
            architectures=["DFlash2DraftModel"],
            hidden_size=128,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
            num_target_layers=4,
            head_dim=32,
            max_position_embeddings=256,
            vocab_size=64,
            layer_types=["full_attention", "full_attention"],
            dflash_config={
                "block_size": 8,
                "conv_group_size": 16,
                "conv_kernel_size": 2,
                "mask_token_id": 63,
                "selector_rank": 8,
                "selector_top_k": 4,
                "target_layer_ids": [1],
            },
        )
        config._attn_implementation = "eager"
        return config

    def _model(self, dtype):
        torch.manual_seed(0)
        model = DFlash2DraftModel(self._config())
        with torch.no_grad():
            for layer in model.layers:
                for conv in (layer.attention_conv, layer.mlp_conv):
                    conv.base_kernel.normal_()
                    conv.kernel_projection.weight.normal_(std=0.05)
        return model.to(device="cuda", dtype=dtype)

    def _step(self, model, inputs, *, layer_only):
        model.zero_grad(set_to_none=True)
        noise, target_hidden, position_ids, attention_mask, grad_output = inputs
        noise = noise.detach().clone().requires_grad_(True)
        if layer_only:
            output = model.layers[0](
                target_hidden=target_hidden,
                hidden_states=noise,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=model.rotary_emb(noise, position_ids),
            )
        else:
            output = model(
                position_ids=position_ids,
                attention_mask=attention_mask,
                noise_embedding=noise,
                target_hidden=target_hidden,
            )
        output.backward(grad_output)
        grads = {
            name: parameter.grad
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        return output.detach(), noise.grad, grads

    def _inputs(self, model, batch_size, dtype, *, layer_only):
        config = model.config
        context_length = 13
        draft_length = 3 * model.block_size
        # The decoder layer consumes the already projected target context.
        context_width = config.hidden_size * (
            1 if layer_only else len(model.target_layer_ids)
        )
        noise = torch.randn(batch_size, draft_length, config.hidden_size)
        context = torch.randn(batch_size, context_length, context_width)
        position_ids = torch.arange(context_length + draft_length).expand(
            batch_size, -1
        )
        attention_mask = (
            torch.rand(batch_size, 1, draft_length, context_length + draft_length) < 0.7
        )
        attention_mask[..., context_length:] = True
        grad_output = torch.randn(batch_size, draft_length, config.hidden_size)
        return (
            noise.to(device="cuda", dtype=dtype),
            context.to(device="cuda", dtype=dtype),
            position_ids.cuda(),
            attention_mask.cuda(),
            grad_output.to(device="cuda", dtype=dtype),
        )

    def test_decoder_layer_and_model_match_eager(self):
        for dtype, atol, rtol in (
            (torch.float32, 1e-4, 1e-4),
            (torch.bfloat16, 5e-2, 5e-2),
        ):
            for batch_size in (1, 2):
                for layer_only in (True, False):
                    with self.subTest(
                        dtype=dtype, batch_size=batch_size, layer_only=layer_only
                    ):
                        model = self._model(dtype)
                        torch.manual_seed(batch_size)
                        inputs = self._inputs(
                            model, batch_size, dtype, layer_only=layer_only
                        )

                        with _count_fused_calls() as calls:
                            fused = self._step(model, inputs, layer_only=layer_only)
                        with _eager_env():
                            eager = self._step(model, inputs, layer_only=layer_only)

                        num_layers = 1 if layer_only else len(model.layers)
                        self.assertEqual(len(calls), 4 * num_layers)

                        torch.testing.assert_close(
                            fused[0], eager[0], atol=atol, rtol=rtol
                        )
                        torch.testing.assert_close(
                            fused[1], eager[1], atol=atol, rtol=rtol
                        )
                        self.assertEqual(fused[2].keys(), eager[2].keys())
                        self.assertTrue(
                            any("attention_conv.base_kernel" in n for n in fused[2])
                        )
                        for name in eager[2]:
                            with self.subTest(parameter=name):
                                _assert_relative_close(
                                    self, fused[2][name], eager[2][name], rtol
                                )


def _assert_relative_close(test, actual, expected, rtol):
    """Compare gradients relative to their largest magnitude."""

    _, relative, _ = _errors(actual, expected)
    test.assertLessEqual(relative, rtol)


if __name__ == "__main__":
    unittest.main()
