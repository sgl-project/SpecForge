"""Dynamic NTK obeys the resolved length rule without retaining request history."""

import unittest

import torch
from transformers import LlamaConfig
from transformers.modeling_rope_utils import _compute_dynamic_ntk_parameters

from specforge.config.draft.codec.settings import encode_rope
from specforge.config.draft.rope import DefaultRoPESpec, DynamicNTKRoPESpec, NoRoPESpec
from specforge.modeling.draft.layers.rope import DynamicNTKRotaryEmbedding, build_rotary
from specforge.modeling.draft.llama3_eagle import LlamaDynamicNTKScalingRotaryEmbedding


def dynamic_node(partial=1.0):
    return DynamicNTKRoPESpec(
        rope_theta=10000.0,
        partial_rotary_factor=partial,
        factor=2.0,
        original_max_position_embeddings=64,
    )


class DynamicRotaryTest(unittest.TestCase):
    def _assert_hf_parity(self, device, dtypes):
        for partial in (1.0, 0.5):
            node = dynamic_node(partial)
            module = build_rotary(node, head_dim=128)
            settings = encode_rope(node)
            # HF places the original dynamic threshold on the outer config.
            config = LlamaConfig(
                head_dim=128,
                max_position_embeddings=settings.pop(
                    "original_max_position_embeddings"
                ),
                rope_parameters=settings,
            )
            # Cross the exact threshold, grow, then shrink both above and below it.
            for length in (63, 64, 65, 8192, 128, 32, 64):
                positions = torch.tensor(
                    [[0, 7, length - 1], [3, 1, 11]], device=device
                )
                frequencies, scale = _compute_dynamic_ntk_parameters(
                    config,
                    device=torch.device(device),
                    seq_len=length,
                )
                angles = (
                    frequencies[None, :, None].expand(2, -1, 1)
                    @ positions[:, None, :].float()
                ).transpose(1, 2)
                angles = torch.cat((angles, angles), dim=-1)
                expected = (angles.cos() * scale, angles.sin() * scale)
                for dtype in dtypes:
                    with self.subTest(
                        partial=partial, length=length, dtype=dtype, device=device
                    ):
                        module.to(device=device, dtype=dtype)
                        hidden = torch.empty(2, 3, 128, device=device, dtype=dtype)
                        with torch.autocast(device_type=device, dtype=torch.bfloat16):
                            actual = module(hidden, positions)
                        for left, right in zip(actual, expected):
                            torch.testing.assert_close(
                                left, right.to(dtype), rtol=0, atol=0
                            )
            self.assertIs(module.config, node)

    def test_cpu_matches_hf_frequency_rule_for_explicit_lengths(self):
        self._assert_hf_parity("cpu", (torch.float32, torch.bfloat16))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_matches_hf_frequency_rule_for_explicit_lengths(self):
        self._assert_hf_parity("cuda", (torch.float32, torch.float16, torch.bfloat16))

    def test_long_sequence_matches_eagle_cache_rebuilt_for_that_length(self):
        node = dynamic_node()
        reference = LlamaDynamicNTKScalingRotaryEmbedding(
            dim=128,
            max_position_embeddings=64,
            base=node.rope_theta,
            scaling_factor=node.factor,
        )
        # Legacy EAGLE prebuilds max_positions + 20 at initialization. Compare
        # its explicit-length frequency rule, not that history-dependent cache.
        for length in (128, 256):
            reference._set_cos_sin_cache(length, torch.device("cpu"), torch.float32)
            positions = torch.tensor([[0, 9, length - 1]])
            actual = build_rotary(node, head_dim=128)(torch.empty(1, 3, 128), positions)
            for left, right in zip(
                actual, (reference.cos_cached, reference.sin_cached)
            ):
                torch.testing.assert_close(left, right[0, 0][positions], rtol=0, atol=0)

    def test_length_comes_from_positions_and_short_calls_restore_default(self):
        node = dynamic_node()
        module = build_rotary(node, head_dim=128)
        hidden = torch.empty(1, 3, 128)
        positions = torch.tensor([[0, 9, 63]])
        expected = build_rotary(
            DefaultRoPESpec(
                rope_theta=node.rope_theta,
                partial_rotary_factor=node.partial_rotary_factor,
            ),
            head_dim=128,
        )(hidden, positions)
        module(hidden, torch.tensor([[10, 99, 65535]]))
        torch.testing.assert_close(module(hidden, positions), expected, rtol=0, atol=0)
        # Padded batch layout changes neither the maximum nor the selected phases.
        full = module(hidden, torch.arange(128).unsqueeze(0))
        selected = module(hidden, torch.tensor([[0, 17, 127]]))
        for left, right in zip(selected, full):
            torch.testing.assert_close(left, right[:, [0, 17, 127]], rtol=0, atol=0)

    def test_meta_cast_empty_positions_and_invalid_inputs(self):
        node = dynamic_node(0.5)
        with torch.device("meta"):
            module = build_rotary(node, head_dim=128)
        module.to_empty(device="cpu").bfloat16()
        hidden = torch.empty(1, 3, 128, dtype=torch.bfloat16)
        positions = torch.tensor([[0, 128, 65535]])
        expected = build_rotary(node, head_dim=128)(hidden.float(), positions)
        for left, right in zip(module(hidden, positions), expected):
            torch.testing.assert_close(left, right.bfloat16(), rtol=0, atol=0)
        for value in module(hidden, torch.empty(1, 0, dtype=torch.long)):
            self.assertEqual(value.shape, (1, 0, 64))
        self.assertEqual(module.state_dict(), {})
        with self.assertRaises(TypeError):
            DynamicNTKRotaryEmbedding(NoRoPESpec(), head_dim=128)
        with self.assertRaises(ValueError):
            build_rotary(node, head_dim=4)
        with self.assertRaisesRegex(ValueError, "position_ids"):
            module(hidden, torch.arange(3))


if __name__ == "__main__":
    unittest.main()
