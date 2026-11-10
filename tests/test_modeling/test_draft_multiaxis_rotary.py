"""Multi-axis section selection is internal to rotary preparation."""

import unittest
from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import nn

from specforge.config.draft.residual import StandardResidualSpec
from specforge.config.draft.rope import DefaultRoPESpec, MultiAxisRoPESpec, NoRoPESpec
from specforge.modeling.draft.backbone import DraftBackbone, LayerInputs
from specforge.modeling.draft.layers.residual import StandardResidual
from specforge.modeling.draft.layers.rope import (
    MultiAxisRotaryEmbedding,
    apply_rotary,
    build_rotary,
)
from specforge.modeling.draft.llama3_eagle import (
    LlamaMutiRotaryEmbedding,
    apply_multimodal_rotary_pos_emb,
)
from tests.test_modeling.test_draft_rotary import RotaryLayer


def multiaxis_node(partial=1.0, sections=(16, 24, 24)):
    return MultiAxisRoPESpec(
        interleaved=False,
        mrope_interleaved=False,
        rope_theta=10000.0,
        partial_rotary_factor=partial,
        mrope_section=sections,
    )


class MultiAxisRotaryTest(unittest.TestCase):
    def _assert_parity(self, device, dtypes):
        positions = torch.arange(24, device=device).reshape(3, 2, 4)
        positions[2, 1, 3] = 65535
        for node in (
            multiaxis_node(),
            multiaxis_node(sections=(0, 32, 32)),
            multiaxis_node(sections=(64, 0, 0)),
            multiaxis_node(partial=0.5, sections=(8, 12, 12)),
        ):
            dim = node.rotary_dim(128)
            module = build_rotary(node, head_dim=128)
            reference = LlamaMutiRotaryEmbedding(
                dim=dim,
                max_position_embeddings=64,
                base=node.rope_theta,
                device=device,
            )
            for dtype in dtypes:
                with self.subTest(node=node, device=device, dtype=dtype):
                    module.to(device=device, dtype=dtype)
                    hidden = torch.empty(2, 4, 128, device=device, dtype=dtype)
                    expected = reference(hidden.float(), positions)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        actual = module(hidden, positions)
                    for left, right in zip(actual, expected):
                        selected = torch.cat(
                            [
                                section[index % 3]
                                for index, section in enumerate(
                                    right.split(node.mrope_section * 2, dim=-1)
                                )
                            ],
                            dim=-1,
                        )
                        torch.testing.assert_close(left, selected, rtol=0, atol=0)
                    q = torch.randn(
                        2, 4, 4, 128, device=device, dtype=dtype, requires_grad=True
                    )
                    k = torch.randn(
                        2, 2, 4, 128, device=device, dtype=dtype, requires_grad=True
                    )
                    q_copy = q.detach().clone().requires_grad_()
                    k_copy = k.detach().clone().requires_grad_()
                    ref_q, ref_k = apply_multimodal_rotary_pos_emb(
                        q_copy[..., :dim].float(),
                        k_copy[..., :dim].float(),
                        *expected,
                        list(node.mrope_section),
                    )
                    reference_outputs = (
                        torch.cat((ref_q, q_copy[..., dim:]), dim=-1).to(dtype),
                        torch.cat((ref_k, k_copy[..., dim:]), dim=-1).to(dtype),
                    )
                    outputs = (apply_rotary(q, actual), apply_rotary(k, actual))
                    for left, right in zip(outputs, reference_outputs):
                        torch.testing.assert_close(left, right, rtol=0, atol=0)
                        upstream = torch.randn_like(left)
                        left.backward(upstream)
                        right.backward(upstream)
                    for left, right in ((q, q_copy), (k, k_copy)):
                        torch.testing.assert_close(
                            left.grad, right.grad, rtol=0, atol=0
                        )
            self.assertIs(module.config, node)

    def test_cpu_eagle_outputs_and_gradients(self):
        self._assert_parity("cpu", (torch.float32, torch.bfloat16))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_eagle_outputs_and_gradients(self):
        self._assert_parity("cuda", (torch.float32, torch.float16, torch.bfloat16))

    def test_interleaved_axes_match_transformers_and_text_positions(self):
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5TextRotaryEmbedding,
        )

        node = replace(multiaxis_node(0.25, (11, 11, 10)), mrope_interleaved=True)
        module = build_rotary(node, head_dim=256)
        positions = torch.arange(9).reshape(3, 1, 3)
        hidden = torch.empty(1, 3, 256)
        frequencies = module.frequencies(hidden.device)
        phases = positions.float().unsqueeze(-1) * frequencies
        expected = Qwen3_5TextRotaryEmbedding.recomposition_frequencies(
            SimpleNamespace(mrope_section=node.mrope_section), phases
        )
        torch.testing.assert_close(
            module(hidden, positions), (expected.cos(), expected.sin()), rtol=0, atol=0
        )
        default = build_rotary(
            DefaultRoPESpec(
                interleaved=False,
                rope_theta=node.rope_theta,
                partial_rotary_factor=0.25,
            ),
            head_dim=256,
        )
        torch.testing.assert_close(
            module(hidden, positions[0]), default(hidden, positions[0]), rtol=0, atol=0
        )

    def test_meta_forward(self):
        for node in (
            multiaxis_node(),
            replace(multiaxis_node(), mrope_interleaved=True),
            DefaultRoPESpec(
                interleaved=False, rope_theta=10000.0, partial_rotary_factor=1.0
            ),
        ):
            module = build_rotary(node, head_dim=128)
            positions = torch.empty(2, 4, dtype=torch.long, device="meta")
            for table in module(torch.empty(2, 4, 128, device="meta"), positions):
                self.assertTrue(table.is_meta)
                self.assertEqual(table.shape, (2, 4, 128))
                self.assertEqual(table.dtype, torch.float32)

    def test_text_positions_reduce_to_default_full_and_partial_rotary(self):
        positions = torch.tensor([[0, 17, 65535], [3, 4, 7]])
        for node in (multiaxis_node(), multiaxis_node(0.5, (8, 12, 12))):
            module = build_rotary(node, head_dim=128)
            default = build_rotary(
                DefaultRoPESpec(
                    interleaved=False,
                    rope_theta=node.rope_theta,
                    partial_rotary_factor=node.partial_rotary_factor,
                ),
                head_dim=128,
            )
            hidden = torch.empty(2, 3, 128)
            expected = default(hidden, positions)
            for actual in (
                module(hidden, positions),
                module(hidden, positions.unsqueeze(0).expand(3, -1, -1)),
            ):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_mixed_backbone_layers_need_no_axis_specific_attention_interface(self):
        node = multiaxis_node(0.5, (1, 0, 1))
        default = DefaultRoPESpec(
            interleaved=False, rope_theta=10000.0, partial_rotary_factor=1.0
        )
        rotary = [build_rotary(node, head_dim=8), build_rotary(default, head_dim=8)]
        backbone = DraftBackbone(
            [RotaryLayer(), RotaryLayer()],
            final_norm=nn.Identity(),
            rotary=rotary,
            residual=StandardResidual(StandardResidualSpec()),
        )
        hidden = torch.randn(2, 3, 8, requires_grad=True)
        positions = [torch.arange(18).reshape(3, 2, 3), torch.arange(6).reshape(2, 3)]
        inputs = [LayerInputs(position_ids=entry) for entry in positions]
        expected = hidden.unsqueeze(1)
        for module, entry in zip(rotary, positions):
            expected = apply_rotary(expected, module(hidden, entry))
        actual = backbone(hidden, inputs)
        torch.testing.assert_close(actual, expected.squeeze(1), rtol=0, atol=0)
        state = backbone.begin(hidden, inputs)
        for _ in rotary:
            state = backbone.run_next_layer(state)
        torch.testing.assert_close(backbone.finalize(state), actual, rtol=0, atol=0)
        expected_grad = torch.autograd.grad(expected.sum(), hidden, retain_graph=True)[
            0
        ]
        actual_grad = torch.autograd.grad(actual.sum(), hidden)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
        self.assertTrue(all(entry.position_embeddings is None for entry in inputs))

    def test_meta_loading_and_dtype_cast_do_not_round_frequencies(self):
        node = multiaxis_node()
        with torch.device("meta"):
            module = build_rotary(node, head_dim=128)
        module.to_empty(device="cpu").bfloat16()
        positions = torch.tensor([[[0, 65535]], [[9, 10000]], [[31, 65536]]])
        hidden = torch.empty(1, 2, 128, dtype=torch.bfloat16)
        expected = build_rotary(node, head_dim=128)(hidden.float(), positions)
        for left, right in zip(module(hidden, positions), expected):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertEqual(module.state_dict(), {})

    def test_invalid_sections_and_position_layouts_fail(self):
        with self.assertRaises(TypeError):
            MultiAxisRotaryEmbedding(NoRoPESpec(), head_dim=128)
        with self.assertRaises(ValueError):
            build_rotary(multiaxis_node(), head_dim=64)
        module = build_rotary(multiaxis_node(), head_dim=128)
        for positions in (
            torch.zeros(4),
            torch.zeros(2, 1, 4),
            torch.zeros(3, 1, 1, 4),
        ):
            with self.assertRaisesRegex(ValueError, "multi-axis position_ids"):
                module(torch.empty(1, 4, 128), positions)


if __name__ == "__main__":
    unittest.main()
