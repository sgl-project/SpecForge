"""YaRN uses resolved frequency/amplitude settings without applying MLA scaling."""

import unittest
from dataclasses import replace

import torch
from transformers import Qwen3Config
from transformers.modeling_rope_utils import _compute_yarn_parameters

from specforge.config.draft.codec.components import encode_component
from specforge.config.draft.codec.legacy.eagle_rope import resolve_eagle_rope
from specforge.config.draft.rope import NoRoPESpec, YaRNSpec
from specforge.modeling.draft.layers.rope import (
    YaRNRotaryEmbedding,
    apply_rotary,
    build_rotary,
)
from tests.test_config.test_draft_eagle_rope_rules import yarn_settings
from tests.test_modeling.mtp_reference import apply_rotary_pos_emb
from tests.test_modeling.rotary_reference import LlamaYarnRotaryEmbedding


def yarn_node(**overrides):
    return YaRNSpec(
        **(
            dict(
                interleaved=False,
                rope_theta=10000.0,
                partial_rotary_factor=1.0,
                factor=16.0,
                original_max_position_embeddings=4096,
                attention_factor=1.37,
                beta_fast=32.0,
                beta_slow=1.0,
                truncate=True,
                mscale_all_dim=0.707,
            )
            | overrides
        )
    )


class YaRNRotaryTest(unittest.TestCase):
    def _assert_hf_parity(self, device, dtypes):
        positions = torch.tensor([[0, 37, 4096, 65535], [7, 1, 11, 29]], device=device)
        for node in (
            yarn_node(),
            yarn_node(partial_rotary_factor=0.5),
            yarn_node(truncate=False),
            yarn_node(factor=1.0),
            yarn_node(beta_fast=8.0, beta_slow=8.0, truncate=False),
            yarn_node(original_max_position_embeddings=1),
        ):
            config = Qwen3Config(
                head_dim=128,
                max_position_embeddings=int(
                    node.factor * node.original_max_position_embeddings
                ),
                rope_parameters=encode_component("rope", node),
            )
            inv_freq, scale = _compute_yarn_parameters(
                config, device=torch.device(device)
            )
            angles = (
                inv_freq[None, :, None].expand(2, -1, 1) @ positions[:, None, :].float()
            ).transpose(1, 2)
            angles = torch.cat((angles, angles), dim=-1)
            expected_fp32 = (angles.cos() * scale, angles.sin() * scale)
            module = build_rotary(node, head_dim=128)
            for dtype in dtypes:
                with self.subTest(node=node, device=device, dtype=dtype):
                    module.to(device=device, dtype=dtype)
                    hidden = torch.empty(2, 4, 128, device=device, dtype=dtype)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        actual = module(hidden, positions)
                    expected = expected_fp32
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    query = torch.randn(
                        2, 2, 4, 128, device=device, dtype=dtype, requires_grad=True
                    )
                    query_copy = query.detach().clone().requires_grad_()
                    output = apply_rotary(query, actual)
                    reference, _ = apply_rotary_pos_emb(
                        query_copy.float(), query_copy.float(), *expected
                    )
                    reference = reference.to(dtype)
                    torch.testing.assert_close(output, reference, rtol=0, atol=0)
                    upstream = torch.randn_like(output)
                    output.backward(upstream)
                    reference.backward(upstream)
                    torch.testing.assert_close(
                        query.grad, query_copy.grad, rtol=0, atol=0
                    )
            self.assertIs(module.config, node)

    def test_cpu_reference_outputs_and_gradients(self):
        self._assert_hf_parity("cpu", (torch.float32, torch.bfloat16))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_reference_outputs_and_gradients(self):
        self._assert_hf_parity("cuda", (torch.float32, torch.float16, torch.bfloat16))

    def test_eagle_codec_amplitude_ratio_matches_existing_cache(self):
        for settings in (
            yarn_settings(),
            {
                k: v
                for k, v in yarn_settings().items()
                if k != "original_max_position_embeddings"
            },
            yarn_settings(mscale=0, mscale_all_dim=0),
            yarn_settings(mscale=1, mscale_all_dim=0),
        ):
            node = resolve_eagle_rope(
                {
                    "rope_scaling": settings,
                    "rope_theta": 10000,
                    "max_position_embeddings": 64,
                }
            )
            reference = LlamaYarnRotaryEmbedding(
                dim=128,
                max_position_embeddings=64,
                base=node.rope_theta,
                scaling_factor=settings["factor"],
                original_max_position_embeddings=settings.get(
                    "original_max_position_embeddings", 64
                ),
                beta_fast=settings["beta_fast"],
                beta_slow=settings["beta_slow"],
                mscale=settings["mscale"],
                mscale_all_dim=settings["mscale_all_dim"],
            )
            positions = torch.tensor([[0, 9, 31, 64]])
            actual = build_rotary(node, head_dim=128)(torch.empty(1, 4, 128), positions)
            for left, right in zip(
                actual, (reference.cos_cached, reference.sin_cached)
            ):
                torch.testing.assert_close(left, right[0, 0][positions], rtol=0, atol=0)

    def test_mla_full_score_scale_is_not_applied_to_rotary_prefix(self):
        node = yarn_node()
        other = replace(node, mscale_all_dim=0)
        self.assertGreater(node.mla_qk_scale_multiplier, other.mla_qk_scale_multiplier)
        hidden, positions = torch.empty(1, 3, 128), torch.tensor([[0, 4096, 9999]])
        left = build_rotary(node, head_dim=128)(hidden, positions)
        right = build_rotary(other, head_dim=128)(hidden, positions)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        torch.testing.assert_close(
            left[0][:, 0], torch.full((1, 128), node.attention_factor)
        )

    def test_meta_loading_keeps_all_rotary_state_derived(self):
        node = yarn_node(partial_rotary_factor=0.5, truncate=False)
        with torch.device("meta"):
            module = build_rotary(node, head_dim=128)
        module.to_empty(device="cpu").bfloat16()
        hidden = torch.empty(1, 3, 128, dtype=torch.bfloat16)
        positions = torch.tensor([[0, 8192, 65535]])
        expected = build_rotary(node, head_dim=128)(hidden.float(), positions)
        for left, right in zip(module(hidden, positions), expected):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertEqual(module.state_dict(), {})
        self.assertEqual(list(module.parameters()), [])
        with self.assertRaises(TypeError):
            YaRNRotaryEmbedding(NoRoPESpec(), head_dim=128)
        with self.assertRaises(ValueError):
            build_rotary(node, head_dim=6)


if __name__ == "__main__":
    unittest.main()
