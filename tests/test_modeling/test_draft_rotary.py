"""Rotary preparation and application preserve full/partial reference behavior."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from specforge.config.draft.residual import StandardResidualSpec
from specforge.config.draft.rope import DefaultRoPESpec
from specforge.modeling.draft.backbone import DraftBackbone, LayerInputs
from specforge.modeling.draft.layers.residual import StandardResidual
from specforge.modeling.draft.layers.rope import DefaultRotaryEmbedding, apply_rotary
from tests.test_modeling.mtp_reference import (
    PartialRotaryEmbedding,
    apply_rotary_pos_emb,
)
from tests.test_modeling.rotary_reference import LlamaRotaryEmbedding, apply_mla_rope


class RotaryLayer(nn.Module):
    def forward(self, hidden, inputs):
        return apply_rotary(hidden.unsqueeze(1), inputs.position_embeddings).squeeze(1)


class RotaryTest(unittest.TestCase):
    def test_mla_rotation_uses_fp32_with_partial_tail_and_gradients(self):
        cases = [("cpu", torch.float32), ("cpu", torch.bfloat16)]
        if torch.cuda.is_available():
            cases.extend(
                ("cuda", dtype)
                for dtype in (torch.float32, torch.float16, torch.bfloat16)
            )
        for device, dtype in cases:
            for width in (4, 8):
                for interleaved in (False, True):
                    hidden = torch.randn(
                        2, 3, 5, 8, device=device, dtype=dtype, requires_grad=True
                    )
                    other = hidden.detach().clone().requires_grad_()
                    phase = torch.randn(
                        2, 5, width // 2, device=device, dtype=torch.float32
                    ).repeat(1, 1, 2)
                    cos, sin = phase.cos(), phase.sin()
                    expected = torch.cat(
                        (
                            apply_mla_rope(
                                hidden[..., :width].float(),
                                cos,
                                sin,
                                interleaved=interleaved,
                            ).to(dtype),
                            hidden[..., width:],
                        ),
                        dim=-1,
                    )
                    actual = apply_rotary(other, (cos, sin), interleaved=interleaved)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    upstream = torch.randn_like(actual)
                    actual.backward(upstream)
                    expected.backward(upstream)
                    torch.testing.assert_close(other.grad, hidden.grad, rtol=0, atol=0)

    def _assert_parity(self, device, dtypes):
        positions = torch.tensor([[0, 7, 19, 64], [31, 9, 2, 53]], device=device)
        for partial in (1.0, 0.5):
            node = DefaultRoPESpec(
                interleaved=False, rope_theta=10000.0, partial_rotary_factor=partial
            )
            module = DefaultRotaryEmbedding(node, head_dim=16)
            reference = PartialRotaryEmbedding(
                SimpleNamespace(
                    rope_theta=node.rope_theta, partial_rotary_factor=partial
                ),
                head_dim=16,
            ).to(device)
            hf_reference = Qwen3RotaryEmbedding(
                Qwen3Config(
                    head_dim=16,
                    rope_parameters={
                        "rope_type": "default",
                        "rope_theta": node.rope_theta,
                        "partial_rotary_factor": partial,
                    },
                ),
            ).to(device)
            eagle_reference = LlamaRotaryEmbedding(
                dim=node.rotary_dim(16),
                max_position_embeddings=64,
                base=node.rope_theta,
                device=device,
            )
            for dtype in dtypes:
                with self.subTest(device=device, partial=partial, dtype=dtype):
                    hidden = torch.randn(2, 4, 16, device=device, dtype=dtype)
                    actual = module(hidden, positions)
                    references = [
                        reference(hidden.float(), positions),
                        tuple(
                            cache[0, 0][positions]
                            for cache in (
                                eagle_reference.cos_cached,
                                eagle_reference.sin_cached,
                            )
                        ),
                    ]
                    # Qwen3's default rotary implementation always rotates the full
                    # head; partial rotation is covered by the MTP reference above.
                    if partial == 1.0:
                        references.append(hf_reference(hidden.float(), positions))
                    for expected in references:
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    q = torch.randn(
                        2, 4, 4, 16, device=device, dtype=dtype, requires_grad=True
                    )
                    k = torch.randn(
                        2, 2, 4, 16, device=device, dtype=dtype, requires_grad=True
                    )
                    q_copy = q.detach().clone().requires_grad_()
                    k_copy = k.detach().clone().requires_grad_()
                    expected = tuple(
                        t.to(dtype)
                        for t in apply_rotary_pos_emb(q.float(), k.float(), *actual)
                    )
                    outputs = (
                        apply_rotary(q_copy, actual),
                        apply_rotary(k_copy, actual),
                    )
                    for left, right in zip(outputs, expected):
                        torch.testing.assert_close(left, right, rtol=0, atol=0)
                        upstream = torch.randn_like(left)
                        left.backward(upstream)
                        right.backward(upstream)
                    for left, right in ((q_copy, q), (k_copy, k)):
                        torch.testing.assert_close(
                            left.grad, right.grad, rtol=0, atol=0
                        )
            self.assertIs(module.config, node)

    def test_cpu_reference_outputs_and_gradients(self):
        self._assert_parity("cpu", (torch.float32, torch.bfloat16))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_reference_outputs_and_gradients(self):
        self._assert_parity("cuda", (torch.float32, torch.float16, torch.bfloat16))

    def test_precision_survives_module_cast_meta_and_autocast(self):
        node = DefaultRoPESpec(
            interleaved=False, rope_theta=10000000.0, partial_rotary_factor=0.5
        )
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            with torch.device("meta"):
                module = DefaultRotaryEmbedding(node, head_dim=128)
            module.to_empty(device=device).bfloat16()
            hidden = torch.empty(2, 4, 128, device=device, dtype=torch.bfloat16)
            positions = torch.tensor([[0, 32768, 65535, 131071]], device=device)
            expected = DefaultRotaryEmbedding(node, head_dim=128)(
                hidden.float(), positions
            )
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                actual = module(hidden, positions)
            for left, right in zip(actual, expected):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            self.assertEqual(module.state_dict(), {})
            self.assertEqual(list(module.parameters()), [])

    def test_backbone_prepares_per_layer_positions_and_preserves_overrides(self):
        node = DefaultRoPESpec(
            interleaved=False, rope_theta=10000.0, partial_rotary_factor=0.5
        )
        rotary = DefaultRotaryEmbedding(node, head_dim=8)
        backbone = DraftBackbone(
            [RotaryLayer(), RotaryLayer()],
            final_norm=nn.Identity(),
            rotary=[rotary, rotary],
            residual=StandardResidual(StandardResidualSpec()),
        )
        hidden = torch.randn(2, 3, 8, requires_grad=True)
        positions = torch.arange(3).unsqueeze(0)
        supplied = rotary(hidden, positions + 10)
        inputs = [
            LayerInputs(position_ids=positions),
            LayerInputs(position_embeddings=supplied),
        ]
        expected = apply_rotary(hidden.unsqueeze(1), rotary(hidden, positions))
        expected = apply_rotary(expected, supplied).squeeze(1)
        actual = backbone(hidden, inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        grad = torch.autograd.grad(actual.sum(), hidden, retain_graph=True)[0]
        ref_grad = torch.autograd.grad(expected.sum(), hidden)[0]
        torch.testing.assert_close(grad, ref_grad, rtol=0, atol=0)
        self.assertIsNone(inputs[0].position_embeddings)
        self.assertIs(inputs[1].position_embeddings, supplied)

    def test_backbone_reuses_equivalent_rotary_only_within_forward(self):
        node = DefaultRoPESpec(
            interleaved=False, rope_theta=10000.0, partial_rotary_factor=0.5
        )
        rotary = [DefaultRotaryEmbedding(node, head_dim=8) for _ in range(2)]
        backbone = DraftBackbone(
            [RotaryLayer(), RotaryLayer()],
            final_norm=nn.Identity(),
            rotary=rotary,
            residual=StandardResidual(StandardResidualSpec()),
        )
        hidden, positions = torch.randn(1, 3, 8), torch.arange(3)[None]
        inputs = [LayerInputs(position_ids=positions)] * 2
        with (
            patch.object(rotary[0], "forward", wraps=rotary[0].forward) as first,
            patch.object(rotary[1], "forward", wraps=rotary[1].forward) as second,
        ):
            state = backbone.begin(hidden, inputs)
            self.assertIs(
                state.layer_inputs[0].position_embeddings,
                state.layer_inputs[1].position_embeddings,
            )
            self.assertEqual((first.call_count, second.call_count), (1, 0))
            backbone.begin(hidden, inputs)
            self.assertEqual(first.call_count, 2)
            backbone.begin(hidden, [inputs[0], LayerInputs(position_ids=positions + 1)])
            self.assertEqual(second.call_count, 1)

    def test_backbone_keeps_tables_of_custom_rotary_subclasses_per_module(self):
        """A subclass may hold state beyond its node, so equal nodes must not share."""

        class OffsetRotary(DefaultRotaryEmbedding):
            def __init__(self, config, *, head_dim, offset):
                super().__init__(config, head_dim=head_dim)
                self.offset = offset

            def _positions(self, position_ids, device):
                return super()._positions(position_ids, device) + self.offset

        node = DefaultRoPESpec(
            interleaved=False, rope_theta=10000.0, partial_rotary_factor=1.0
        )
        rotary = [OffsetRotary(node, head_dim=8, offset=offset) for offset in (0, 5)]
        backbone = DraftBackbone(
            [RotaryLayer(), RotaryLayer()],
            final_norm=nn.Identity(),
            rotary=rotary,
            residual=StandardResidual(StandardResidualSpec()),
        )
        hidden, positions = torch.randn(1, 3, 8), torch.arange(3)[None]
        state = backbone.begin(hidden, [LayerInputs(position_ids=positions)] * 2)
        for inputs, module in zip(state.layer_inputs, rotary):
            torch.testing.assert_close(
                inputs.position_embeddings, module(hidden, positions), rtol=0, atol=0
            )

    def test_backbone_memo_compiles_with_fresh_positions(self):
        from torch._dynamo.testing import CompileCounter

        node = DefaultRoPESpec(
            interleaved=False, rope_theta=10000.0, partial_rotary_factor=1.0
        )
        backbone = DraftBackbone(
            [RotaryLayer(), RotaryLayer()],
            final_norm=nn.Identity(),
            rotary=[DefaultRotaryEmbedding(node, head_dim=8) for _ in range(2)],
            residual=StandardResidual(StandardResidualSpec()),
        )

        def forward(hidden, positions):
            return backbone(hidden, [LayerInputs(position_ids=positions)] * 2)

        torch._dynamo.reset()
        counter = CompileCounter()
        compiled = torch.compile(forward, backend=counter, fullgraph=True)
        for i in range(6):
            hidden, positions = torch.randn(1, 3, 8), (torch.arange(3) + i)[None]
            torch.testing.assert_close(
                compiled(hidden, positions), forward(hidden, positions)
            )
        self.assertLessEqual(
            counter.frame_count, 2
        )  # Cold and populated frequency cache.

    def test_invalid_dimensions_and_layouts_fail(self):
        node = DefaultRoPESpec(
            interleaved=False, rope_theta=10000.0, partial_rotary_factor=0.5
        )
        for head_dim in (0, True, 3, 6):
            with self.assertRaises(ValueError):
                DefaultRotaryEmbedding(node, head_dim=head_dim)
        with self.assertRaises(TypeError):
            DefaultRotaryEmbedding({}, head_dim=8)
        with self.assertRaises(ValueError):
            DefaultRotaryEmbedding(node, head_dim=8)(
                torch.zeros(1, 3, 8), torch.arange(3)
            )
        hidden = torch.zeros(1, 2, 3, 8)
        for batch, length in ((1, 1), (2, 3)):
            with self.assertRaisesRegex(ValueError, "batch.*sequence"):
                apply_rotary(
                    hidden,
                    (torch.ones(batch, length, 8), torch.zeros(batch, length, 8)),
                )
        for width in (0, 3, 10):
            with self.assertRaises(ValueError):
                apply_rotary(
                    hidden, (torch.ones(1, 3, width), torch.zeros(1, 3, width))
                )


if __name__ == "__main__":
    unittest.main()
