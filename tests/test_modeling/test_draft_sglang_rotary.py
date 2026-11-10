"""Serving-reference checks for the canonical RoPE rules and precision policy."""

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from specforge.config.draft.rope import (
    DefaultRoPESpec,
    DynamicNTKRoPESpec,
    LinearRoPESpec,
)
from specforge.modeling.draft.layers.rope import apply_rotary, build_rotary
from tests.test_modeling.test_draft_yarn_rotary import yarn_node


@unittest.skipUnless(
    torch.cuda.is_available() and importlib.util.find_spec("sglang"),
    "SGLang CUDA runtime is required for serving-reference tests",
)
class SGLangRotaryTest(unittest.TestCase):
    def setUp(self):
        # Only provide the standalone reference's runtime policy; run its real math.
        self.enterContext(
            patch(
                "sglang.srt.layers.rotary_embedding.base.get_exec",
                return_value=SimpleNamespace(
                    deterministic=SimpleNamespace(rl_on_policy_target=None)
                ),
            )
        )

    def test_tables_and_rotation_match_serving_reference(self):
        from sglang.srt.layers.rotary_embedding.base import (
            LinearScalingRotaryEmbedding,
            RotaryEmbedding,
        )
        from sglang.srt.layers.rotary_embedding.rope_variant import (
            DynamicNTKScalingRotaryEmbedding,
        )
        from sglang.srt.layers.rotary_embedding.utils import (
            apply_rotary_pos_emb_native_eager,
        )

        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            for partial in (1.0, 0.5):
                common = dict(
                    interleaved=False,
                    rope_theta=10000000.0,
                    partial_rotary_factor=partial,
                )
                for node, reference_type, extra in (
                    (DefaultRoPESpec(**common), RotaryEmbedding, {}),
                    (
                        LinearRoPESpec(**common, factor=2.5),
                        LinearScalingRotaryEmbedding,
                        {"scaling_factors": 2.5},
                    ),
                    (
                        DynamicNTKRoPESpec(
                            **common, factor=2.5, original_max_position_embeddings=32768
                        ),
                        DynamicNTKScalingRotaryEmbedding,
                        {"scaling_factor": 2.5},
                    ),
                ):
                    with self.subTest(device=device, node=node):
                        with torch.device(device):
                            reference = reference_type(
                                head_size=128,
                                rotary_dim=node.rotary_dim(128),
                                max_position_embeddings=65536 if not extra else 32768,
                                base=node.rope_theta,
                                is_neox_style=True,
                                dtype=torch.float32,
                                **extra,
                            )
                        positions = torch.tensor([[0, 32768, 65535]], device=device)
                        for dtype in (torch.float32, torch.bfloat16):
                            hidden = torch.empty(1, 3, 128, device=device, dtype=dtype)
                            tables = build_rotary(node, head_dim=128)(hidden, positions)
                            cos, sin = reference.cos_sin_cache[positions].chunk(
                                2, dim=-1
                            )
                            expected = (
                                torch.cat((cos, cos), -1),
                                torch.cat((sin, sin), -1),
                            )
                            torch.testing.assert_close(tables, expected, rtol=0, atol=0)
                            q = torch.randn(
                                1,
                                2,
                                3,
                                128,
                                device=device,
                                dtype=dtype,
                                requires_grad=True,
                            )
                            other = q.detach().clone().requires_grad_()
                            actual = apply_rotary(q, tables)
                            dim = tables[0].shape[-1]
                            rotated, _ = apply_rotary_pos_emb_native_eager(
                                other[..., :dim], other[..., :dim], *expected
                            )
                            output = torch.cat((rotated, other[..., dim:]), -1)
                            torch.testing.assert_close(actual, output, rtol=0, atol=0)
                            torch.testing.assert_close(
                                torch.autograd.grad(actual.sum(), q)[0],
                                torch.autograd.grad(output.sum(), other)[0],
                                rtol=0,
                                atol=0,
                            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA kernel reference is required")
    def test_cuda_kernel_pairing_and_partial_rotation(self):
        from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding

        for interleaved in (False, True):
            node = DefaultRoPESpec(
                interleaved=interleaved, rope_theta=10000.0, partial_rotary_factor=0.5
            )
            with torch.device("cuda"):
                reference = RotaryEmbedding(
                    128, 64, 65536, 10000.0, not interleaved, torch.bfloat16
                )
            positions = torch.tensor([[32768, 65535]], device="cuda")
            q = torch.randn(1, 2, 2, 128, device="cuda", dtype=torch.bfloat16)
            tables = build_rotary(node, head_dim=128)(q, positions)
            actual = apply_rotary(q, tables, interleaved=interleaved)
            packed = q.transpose(1, 2).reshape(2, -1).clone()
            expected, _ = reference.forward_cuda(
                positions.flatten(), packed, packed.clone()
            )
            expected = expected.view(1, 2, 2, 128).transpose(1, 2)
            torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.016)


class RotaryPrecisionTest(unittest.TestCase):
    def test_legacy_bf16_frequency_cast_is_an_intentional_difference(self):
        from transformers import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

        node = DefaultRoPESpec(
            interleaved=False, rope_theta=1e7, partial_rotary_factor=1.0
        )
        positions = torch.tensor([[32768, 65535]])
        hidden = torch.empty(1, 2, 128, dtype=torch.bfloat16)
        reference = Qwen3RotaryEmbedding(
            Qwen3Config(
                head_dim=128,
                rope_parameters={"rope_type": "default", "rope_theta": 1e7},
            )
        ).bfloat16()
        old = reference(hidden, positions)
        actual = build_rotary(node, head_dim=128)(hidden, positions)
        self.assertTrue(
            any(not torch.equal(a.bfloat16(), b) for a, b in zip(actual, old))
        )

    @unittest.skipUnless(
        torch.cuda.is_available(), "CUDA compiled legacy reference is required"
    )
    def test_legacy_rotation_rounding_is_an_explicit_difference(self):
        from transformers.models.qwen3.modeling_qwen3 import (
            apply_rotary_pos_emb as dflash_rotate,
        )

        from specforge.modeling.draft.llama3_eagle import (
            apply_rotary_pos_emb as eagle_rotate,
        )

        node = DefaultRoPESpec(
            interleaved=False, rope_theta=1e7, partial_rotary_factor=1.0
        )
        positions = torch.tensor([[32768, 65535]], device="cuda")
        hidden = torch.empty(1, 2, 128, device="cuda", dtype=torch.bfloat16)
        tables = build_rotary(node, head_dim=128)(hidden, positions)
        torch.manual_seed(42)
        q = torch.randn(1, 4, 2, 128, device="cuda", dtype=torch.bfloat16)
        rounded = tuple(t.bfloat16() for t in tables)
        eager, _ = dflash_rotate(q, q, *rounded)
        # Legacy EAGLE indexes a full cache, whereas DFlash takes selected tables.
        all_tables = build_rotary(node, head_dim=128)(
            hidden, torch.arange(65536, device="cuda")[None]
        )
        compiled, _ = eagle_rotate(
            q, q, *(t[None].bfloat16() for t in all_tables), positions
        )
        actual = apply_rotary(q, tables)
        self.assertFalse(torch.equal(actual, eager))
        self.assertFalse(torch.equal(actual, compiled))
        # Pin the one-final-cast choice, independently of either legacy path.
        first, second = q.float().chunk(2, dim=-1)
        expected = (
            q.float() * tables[0][:, None]
            + torch.cat((-second, first), -1) * tables[1][:, None]
        ).bfloat16()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_yarn_frequencies_are_cached_and_invalidated_on_cast(self):
        module = build_rotary(yarn_node(), head_dim=128)
        positions, hidden = torch.tensor([[0, 32768]]), torch.empty(1, 2, 128)
        with patch.object(module, "_frequencies", wraps=module._frequencies) as compute:
            first = module(hidden, positions)
            module(hidden, positions + 1)
            self.assertEqual(compute.call_count, 1)
            module.bfloat16()
            torch.testing.assert_close(
                module(hidden.bfloat16(), positions), first, rtol=0, atol=0
            )
            self.assertEqual(compute.call_count, 2)
        self.assertEqual(module.state_dict(), {})
