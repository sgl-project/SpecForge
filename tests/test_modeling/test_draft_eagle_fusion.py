"""Model-owned EAGLE fusion matches the legacy first-layer wiring."""

import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import LlamaConfig

from specforge.config.draft.mlp import DenseFFNSpec
from specforge.config.draft.norm import RMSNormSpec
from specforge.modeling.draft.layers import build_component
from specforge.modeling.draft.layers.attention.backends.eager import eager_attention
from specforge.modeling.draft.layers.inputs import LayerInputs
from specforge.modeling.draft.models.eagle_fusion import (
    AuxiliaryFeatureProjection,
    EagleDecoderStage,
)
from specforge.modeling.draft.peagle import PEagleFirstLayer
from tests.test_config.test_draft_eagle3_spec import eagle_config
from tests.test_modeling.test_draft_gqa import build, layout_for, make_node
from tests.test_modeling.test_draft_mla import prepare


def eager_reference(*, query, key, value, block_mask, enable_gqa, **kwargs):
    # Swap only the legacy flex kernel for the canonical eager kernel: BLAS may round
    # a stride-0 GQA repeat differently from repeat_interleave. This oracle tests
    # fusion and residual wiring; attention components have separate oracles.
    return eager_attention(
        query,
        key,
        value,
        attention_mask=block_mask,
        scale=query.shape[-1] ** -0.5,
        dropout_p=0.0,
        training=False,
    )


def stage(*, norm_before_residual=False, device="cpu", dtype=torch.float32):
    return EagleDecoderStage(
        build(
            make_node(qk_norm=False, attention_bias=False, attention_dropout=0),
            input_size=32,
            device=device,
            dtype=dtype,
        ),
        build_component(
            "mlp",
            DenseFFNSpec(intermediate_size=32, hidden_act="silu"),
            hidden_size=16,
            device=device,
            dtype=dtype,
        ),
        hidden_size=16,
        norm=RMSNormSpec(has_weight=True, eps=1e-6, weight_offset=0),
        norm_before_residual=norm_before_residual,
        device=device,
        dtype=dtype,
    )


class EagleFusionTest(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()
        self.enterContext(patch("torch._dynamo.config.disable", True))

    def test_auxiliary_projection_normalizes_chunks_before_projection(self):
        config = eagle_config()
        for normalized in (False, True):
            with self.subTest(normalized=normalized):
                actual = AuxiliaryFeatureProjection(
                    config.spec_config.target_features,
                    hidden_size=16,
                    normalize_features=normalized,
                    norm=config.norm_config,
                )
                # Call the unchanged legacy projection on a minimal owner so the
                # test is independent of legacy model construction and caches.
                from specforge.modeling.draft.llama3_eagle import (
                    LlamaForCausalLMEagle3,
                    LlamaRMSNorm,
                )

                reference = nn.Module()
                reference.target_hidden_size = 32
                reference.fc = nn.Linear(96, 16, bias=False)
                reference.fc_norm = (
                    nn.ModuleList(
                        [LlamaRMSNorm(32, eps=config.norm_config.eps) for _ in range(3)]
                    )
                    if normalized
                    else None
                )
                reference.load_state_dict(actual.state_dict(), strict=True)
                features = torch.randn(2, 4, 96, requires_grad=True)
                other = features.detach().clone().requires_grad_()
                output = actual(features)
                expected = LlamaForCausalLMEagle3.project_hidden_states(
                    reference, other
                )
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                grad = torch.randn_like(output)
                output.backward(grad)
                expected.backward(grad)
                torch.testing.assert_close(features.grad, other.grad, rtol=0, atol=0)
                for name, parameter in actual.named_parameters():
                    torch.testing.assert_close(
                        parameter.grad,
                        dict(reference.named_parameters())[name].grad,
                        rtol=0,
                        atol=0,
                    )
                self.assertIs(actual.config, config.spec_config.target_features)

    def _stage_parity(self, device, dtype):
        torch.manual_seed(0)
        # Canonical RoPE rotates in FP32 before casting; legacy rotates in dtype.
        # Low-precision rotation rounds differently; FP32 eager math is exact.
        tolerance = (
            2e-2 if dtype == torch.bfloat16 else 1e-2 if dtype == torch.float16 else 0
        )
        for normalized_residual in (False, True):
            with self.subTest(
                device=device, dtype=dtype, normalized_residual=normalized_residual
            ):
                actual = stage(
                    norm_before_residual=normalized_residual, device=device, dtype=dtype
                )
                config = LlamaConfig(
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    rms_norm_eps=1e-6,
                    hidden_act="silu",
                )
                reference = PEagleFirstLayer(
                    config, norm_before_residual=normalized_residual
                ).to(device=device, dtype=dtype)
                reference.load_state_dict(actual.state_dict(), strict=True)
                hidden = torch.randn(
                    2, 4, 32, device=device, dtype=dtype, requires_grad=True
                )
                other = hidden.detach().clone().requires_grad_()
                layout = layout_for(hidden)
                causal = (
                    layout.key_positions[:, None, None, :]
                    <= layout.query_positions[:, None, :, None]
                )
                angles = torch.randn(2, 4, 4, device=device, dtype=dtype).repeat(
                    1, 1, 2
                )
                embeddings = (angles.cos(), angles.sin())
                output = actual(
                    hidden,
                    LayerInputs(
                        attention=prepare(actual.self_attn, layout, hidden.shape[0]),
                        position_embeddings=tuple(
                            table.float() for table in embeddings
                        ),
                    ),
                )
                with patch(
                    "specforge.modeling.draft.peagle.compile_friendly_flex_attention",
                    eager_reference,
                ):
                    expected = reference(other, causal, embeddings)
                torch.testing.assert_close(
                    output, expected, rtol=tolerance, atol=tolerance
                )
                grad = torch.randn_like(output)
                output.backward(grad)
                expected.backward(grad)
                torch.testing.assert_close(
                    hidden.grad, other.grad, rtol=tolerance, atol=tolerance
                )
                for name, parameter in actual.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    torch.testing.assert_close(
                        parameter.grad,
                        dict(reference.named_parameters())[name].grad,
                        rtol=tolerance,
                        atol=tolerance,
                    )

    def test_cpu_first_stage_forward_and_backward(self):
        for dtype in (torch.float32, torch.bfloat16):
            self._stage_parity("cpu", dtype)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_first_stage_forward_and_backward(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            self._stage_parity("cuda", dtype)

    def test_invalid_inputs_fail_before_attention(self):
        actual = stage()
        hidden = torch.randn(2, 4, 32)
        layout = layout_for(hidden)
        prepared = prepare(actual.self_attn, layout, 2)
        with patch.object(actual.self_attn, "forward") as attention:
            for value, inputs, message in (
                (
                    hidden[..., :16],
                    LayerInputs(attention=prepared),
                    r"2 \* hidden_size",
                ),
                (hidden, LayerInputs(), "requires prepared attention"),
                (
                    hidden,
                    LayerInputs(attention=prepared, target_context=hidden),
                    "not a target KV context",
                ),
            ):
                with self.assertRaisesRegex(ValueError, message):
                    actual(value, inputs)
            attention.assert_not_called()
        config = eagle_config()
        projection = AuxiliaryFeatureProjection(
            config.spec_config.target_features,
            hidden_size=16,
            normalize_features=True,
            norm=config.norm_config,
        )
        with self.assertRaises(ValueError):
            projection(torch.empty(2, 4, 32))

    def test_meta_device_reaches_all_components(self):
        model = stage(device="meta", dtype=torch.bfloat16)
        self.assertTrue(
            all(p.is_meta and p.dtype == torch.bfloat16 for p in model.parameters())
        )


if __name__ == "__main__":
    unittest.main()
