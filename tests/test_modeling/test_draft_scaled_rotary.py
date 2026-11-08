"""Static scaling rules match HF and preserve EAGLE's Llama3 step transition."""

import unittest
from dataclasses import replace

import torch
from transformers import LlamaConfig
from transformers.modeling_rope_utils import (
    _compute_linear_scaling_rope_parameters,
    _compute_llama3_parameters,
)

from specforge.config.draft.codec.settings import encode_rope
from specforge.config.draft.rope import (
    DefaultRoPESpec,
    LinearRoPESpec,
    Llama3RoPESpec,
    NoRoPESpec,
)
from specforge.modeling.draft.layers.rope import (
    DefaultRotaryEmbedding,
    LinearRotaryEmbedding,
    Llama3RotaryEmbedding,
    build_rotary,
)
from specforge.modeling.draft.llama3_eagle import (
    LlamaLinearScalingRotaryEmbedding,
    LlamaRotaryEmbedding,
)


def scaled_nodes():
    return (
        LinearRoPESpec(rope_theta=10000.0, partial_rotary_factor=1.0, factor=3.0),
        Llama3RoPESpec(
            rope_theta=500000.0,
            partial_rotary_factor=1.0,
            factor=8.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_max_position_embeddings=8192,
        ),
    )


class ScaledRotaryTest(unittest.TestCase):
    def _assert_hf_parity(self, device, dtypes):
        positions = torch.tensor([[0, 37, 8192, 65535], [9, 3, 11, 17]], device=device)
        for original, reference in zip(
            scaled_nodes(),
            (
                _compute_linear_scaling_rope_parameters,
                _compute_llama3_parameters,
            ),
        ):
            for partial in (1.0, 0.5):
                node = replace(original, partial_rotary_factor=partial)
                module = build_rotary(node, head_dim=128)
                config = LlamaConfig(
                    head_dim=128,
                    max_position_embeddings=131072,
                    rope_parameters=encode_rope(node),
                )
                frequencies, scale = reference(config, device=torch.device(device))
                angles = (
                    frequencies[None, :, None].expand(2, -1, 1)
                    @ positions[:, None, :].float()
                ).transpose(1, 2)
                angles = torch.cat((angles, angles), dim=-1)
                expected = (angles.cos() * scale, angles.sin() * scale)
                for dtype in dtypes:
                    with self.subTest(node=node, device=device, dtype=dtype):
                        module.to(device=device, dtype=dtype)
                        hidden = torch.empty(2, 4, 128, device=device, dtype=dtype)
                        with torch.autocast(device_type=device, dtype=torch.bfloat16):
                            actual = module(hidden, positions)
                        for left, right in zip(actual, expected):
                            torch.testing.assert_close(
                                left, right.to(dtype), rtol=0, atol=0
                            )
                # A later short request must not inherit any long-request state.
                short = module(hidden, positions[:, :1])
                for left, right in zip(short, actual):
                    torch.testing.assert_close(left, right[:, :1], rtol=0, atol=0)
                self.assertIs(module.config, node)
                self.assertEqual(module.state_dict(), {})

    def test_cpu_hf_frequency_and_position_parity(self):
        self._assert_hf_parity("cpu", (torch.float32, torch.bfloat16))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_hf_frequency_and_position_parity(self):
        self._assert_hf_parity("cuda", (torch.float32, torch.float16, torch.bfloat16))

    def test_llama3_eagle_smooth_step_and_unscaled_frequencies_match(self):
        original = scaled_nodes()[1]
        for node in (
            original,
            replace(original, high_freq_factor=1.0),
            replace(original, factor=1.0),
        ):
            module = build_rotary(node, head_dim=128)
            reference = LlamaRotaryEmbedding(
                dim=128,
                max_position_embeddings=64,
                base=node.rope_theta,
                scaling_factor=node.factor,
                low_freq_factor=node.low_freq_factor,
                high_freq_factor=node.high_freq_factor,
                orig_max_position=node.original_max_position_embeddings,
            )
            positions = torch.tensor([[0, 7, 31, 64]])
            hidden = torch.empty(1, 4, 128)
            actual = module(hidden, positions)
            expected = (reference.cos_cached, reference.sin_cached)
            for left, right in zip(actual, expected):
                torch.testing.assert_close(left, right[0, 0][positions], rtol=0, atol=0)

    def test_linear_eagle_position_scaling_roundoff_is_bounded(self):
        # HF divides frequencies; legacy EAGLE divides positions. A non-power-of-
        # two factor has FP32 roundoff, so this reference check is not bitwise.
        for factor in (3.0, 4.0):
            node = replace(scaled_nodes()[0], factor=factor)
            module = build_rotary(node, head_dim=128)
            reference = LlamaLinearScalingRotaryEmbedding(
                dim=128,
                max_position_embeddings=64,
                base=node.rope_theta,
                scaling_factor=factor,
            )
            positions = torch.arange(65).unsqueeze(0)
            actual = module(torch.empty(1, 65, 128), positions)
            for left, right in zip(
                actual, (reference.cos_cached, reference.sin_cached)
            ):
                torch.testing.assert_close(
                    left,
                    right[0, 0][positions],
                    rtol=0,
                    atol=0 if factor == 4.0 else 1e-5,
                )

    def test_builders_hold_nodes_and_disabled_rope_has_no_module(self):
        nodes = (
            DefaultRoPESpec(rope_theta=10000.0, partial_rotary_factor=1.0),
            *scaled_nodes(),
        )
        classes = (DefaultRotaryEmbedding, LinearRotaryEmbedding, Llama3RotaryEmbedding)
        for node, cls in zip(nodes, classes):
            with torch.device("meta"):
                module = build_rotary(node, head_dim=128)
            self.assertIs(type(module), cls)
            self.assertIs(module.config, node)
            module.to_empty(device="cpu")
            result = module(torch.empty(1, 2, 128), torch.tensor([[0, 9999]]))
            self.assertTrue(all(torch.isfinite(value).all() for value in result))
            with self.assertRaises(TypeError):
                cls(NoRoPESpec(), head_dim=128)
        self.assertIsNone(build_rotary(NoRoPESpec(), head_dim=128))
        with self.assertRaises(ValueError):
            build_rotary(NoRoPESpec(), head_dim=0)

    def test_unknown_nodes_cannot_fall_back_to_default(self):
        for node in ({"rope_type": "default"}, object()):
            with self.assertRaisesRegex(ValueError, "unsupported rotary node"):
                build_rotary(node, head_dim=128)


if __name__ == "__main__":
    unittest.main()
