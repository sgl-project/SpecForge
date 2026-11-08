"""Composed decoder layers match full reference layers and backbone execution."""

import copy
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch
from torch import nn
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5DecoderLayer,
)

from specforge.config.draft.conv import GroupedCausalConvSpec
from specforge.config.draft.mlp import DenseFFNSpec
from specforge.config.draft.norm import RMSNormSpec
from specforge.config.draft.residual import StandardResidualSpec
from specforge.config.draft.rope import DefaultRoPESpec, NoRoPESpec
from specforge.modeling.draft.backbone import DraftBackbone
from specforge.modeling.draft.dflash import Qwen3DFlashDecoderLayer
from specforge.modeling.draft.dflash2 import DFlashGroupedConv, Qwen3DFlash2DecoderLayer
from specforge.modeling.draft.dflash_kernels import DEFAULT_DFLASH_KERNELS
from specforge.modeling.draft.layers import DecoderLayer, build_component
from specforge.modeling.draft.layers.attention.masks import dense_attention_mask
from specforge.modeling.draft.layers.inputs import LayerInputs
from specforge.modeling.draft.layers.residual import StandardResidual
from specforge.modeling.draft.layers.rope import build_rotary
from tests.test_modeling.test_draft_gqa import build, layout_for, make_node


def make_layer(
    node,
    *,
    rotary_dim=8,
    offset=0,
    device="cpu",
    dtype=torch.float32,
    convolution=False,
):
    conv_node = GroupedCausalConvSpec(kernel_size=3, group_size=4)
    convolutions = (
        tuple(
            build_component(
                "conv",
                conv_node,
                hidden_size=16,
                block_size=4,
                device=device,
                dtype=dtype,
            )
            for _ in range(2)
        )
        if convolution
        else None
    )
    return DecoderLayer(
        build(node, rotary_dim=rotary_dim, offset=offset, device=device, dtype=dtype),
        build_component(
            "mlp",
            DenseFFNSpec(intermediate_size=32, hidden_act="silu"),
            hidden_size=16,
            device=device,
            dtype=dtype,
        ),
        hidden_size=16,
        norm=RMSNormSpec(eps=1e-6, weight_offset=offset),
        sublayer_conv=convolutions,
        device=device,
        dtype=dtype,
    )


class DecoderTest(unittest.TestCase):
    def _assert_reference(self, device, dtypes, autocast=False):
        for kind in ("qwen3", "gated", "context", "convolution"):
            gated = kind == "gated"
            node = make_node(output_gate=gated)
            rotary_dim = 4 if gated else 8
            for dtype in dtypes:
                with self.subTest(
                    kind=kind, device=device, dtype=dtype, autocast=autocast
                ):
                    args = dict(
                        hidden_size=16,
                        intermediate_size=32,
                        hidden_act="silu",
                        num_attention_heads=2,
                        num_key_value_heads=1,
                        head_dim=8,
                        num_hidden_layers=1,
                        layer_types=["full_attention"],
                        attention_bias=True,
                        attention_dropout=0.1,
                        rms_norm_eps=1e-6,
                    )
                    config = (
                        Qwen3_5Config(text_config=args).text_config
                        if gated
                        else Qwen3Config(**args)
                    )
                    config._attn_implementation = "eager"
                    if kind == "convolution":
                        reference = Qwen3DFlash2DecoderLayer(
                            config,
                            0,
                            DEFAULT_DFLASH_KERNELS,
                            attention_conv=DFlashGroupedConv(16, 4, 3, 4),
                            mlp_conv=DFlashGroupedConv(16, 4, 3, 4),
                        )
                        with torch.no_grad():
                            for conv in (reference.attention_conv, reference.mlp_conv):
                                for parameter in conv.parameters():
                                    parameter.normal_(std=0.1)
                    elif kind == "context":
                        reference = Qwen3DFlashDecoderLayer(
                            config, 0, DEFAULT_DFLASH_KERNELS
                        )
                    else:
                        reference = (
                            Qwen3_5DecoderLayer if gated else Qwen3DecoderLayer
                        )(config, 0)
                    reference.to(device=device, dtype=dtype)
                    actual = make_layer(
                        node,
                        rotary_dim=rotary_dim,
                        offset=int(gated),
                        device=device,
                        dtype=dtype,
                        convolution=kind == "convolution",
                    )
                    actual.load_state_dict(reference.state_dict(), strict=True)
                    hidden = torch.randn(
                        2,
                        8 if kind == "convolution" else 3,
                        16,
                        device=device,
                        dtype=dtype,
                        requires_grad=True,
                    )
                    other = hidden.detach().clone().requires_grad_()
                    context = None
                    if kind in ("context", "convolution"):
                        # Feature fusion may already have produced BF16 context
                        # while the decoder's residual/norm stream remains FP32.
                        context_dtype = torch.bfloat16 if autocast else dtype
                        context = torch.randn(
                            2,
                            2,
                            16,
                            device=device,
                            dtype=context_dtype,
                            requires_grad=True,
                        )
                    other_context = (
                        None
                        if context is None
                        else context.detach().clone().requires_grad_()
                    )
                    layout = layout_for(hidden, 0 if context is None else 2)
                    rope = build_rotary(
                        DefaultRoPESpec(
                            rope_theta=10000.0, partial_rotary_factor=rotary_dim / 8
                        ),
                        head_dim=8,
                    )
                    embeddings = rope(hidden, layout.key_positions)
                    entry = LayerInputs(
                        attention_layout=layout,
                        target_context=context,
                        position_embeddings=embeddings,
                    )
                    mask = dense_attention_mask(
                        layout, is_causal=True, sliding_window=None
                    )
                    bias = torch.zeros_like(mask, dtype=dtype).masked_fill(
                        ~mask, -torch.inf
                    )
                    torch.manual_seed(37)
                    with (
                        torch.autocast(device_type=device, dtype=torch.bfloat16)
                        if autocast
                        else nullcontext()
                    ):
                        output = actual(hidden, entry)
                    torch.manual_seed(37)
                    with (
                        patch.dict("os.environ", {"SPECFORGE_DFLASH2_FUSED_CONV": "0"}),
                        (
                            torch.autocast(device_type=device, dtype=torch.bfloat16)
                            if autocast
                            else nullcontext()
                        ),
                    ):
                        if context is None:
                            expected = reference(
                                other,
                                position_embeddings=embeddings,
                                attention_mask=bias,
                            )
                        else:
                            expected = reference(
                                hidden_states=other,
                                target_hidden=other_context,
                                position_embeddings=embeddings,
                                attention_mask=bias,
                            )
                    torch.testing.assert_close(output, expected, rtol=0, atol=0)
                    upstream = torch.randn_like(output)
                    output.backward(upstream)
                    expected.backward(upstream)
                    torch.testing.assert_close(hidden.grad, other.grad, rtol=0, atol=0)
                    if context is not None:
                        torch.testing.assert_close(
                            context.grad, other_context.grad, rtol=0, atol=0
                        )
                    for name, parameter in actual.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        torch.testing.assert_close(
                            parameter.grad,
                            dict(reference.named_parameters())[name].grad,
                            rtol=0,
                            atol=0,
                        )

    def test_cpu_whole_layer_outputs_and_gradients(self):
        self._assert_reference("cpu", (torch.float32, torch.bfloat16))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_whole_layer_outputs_and_gradients(self):
        self._assert_reference("cuda", (torch.float32, torch.float16, torch.bfloat16))

    def test_autocast_with_fp32_residuals_and_bf16_context(self):
        self._assert_reference("cpu", (torch.float32,), autocast=True)
        if torch.cuda.is_available():
            self._assert_reference("cuda", (torch.float32,), autocast=True)

    def test_backbone_sequential_manual_and_checkpoint_paths_are_identical(self):
        cases = [("cpu", torch.float32)]
        if torch.cuda.is_available():
            cases.append(("cuda", torch.bfloat16))
        for device, dtype, convolution in [
            (device, dtype, conv) for device, dtype in cases for conv in (False, True)
        ]:
            layers = [
                make_layer(
                    make_node(sliding_window=2),
                    device=device,
                    dtype=dtype,
                    convolution=convolution,
                ),
                make_layer(
                    make_node(rope_parameters=NoRoPESpec(), is_causal=False),
                    rotary_dim=0,
                    device=device,
                    dtype=dtype,
                    convolution=convolution,
                ),
            ]
            base = DraftBackbone(
                layers,
                final_norm=build_component(
                    "norm",
                    RMSNormSpec(eps=1e-6, weight_offset=0),
                    hidden_size=16,
                    device=device,
                    dtype=dtype,
                ),
                rotary=[
                    build_rotary(
                        DefaultRoPESpec(rope_theta=10000.0, partial_rotary_factor=1.0),
                        head_dim=8,
                    ),
                    None,
                ],
                residual=StandardResidual(StandardResidualSpec()),
            )
            if convolution:
                # Nonidentity kernels exercise both sides during recomputation.
                with torch.no_grad():
                    for layer in layers:
                        for conv in (layer.attention_conv, layer.mlp_conv):
                            conv.kernel_projection.weight.normal_(std=0.1)
            hidden_base = torch.randn(
                1, 8 if convolution else 3, 16, device=device, dtype=dtype
            )
            context_base = torch.randn(1, 2, 16, device=device, dtype=dtype)
            expected = None
            for manual, checkpointed in (
                (False, False),
                (True, False),
                (False, True),
                (True, True),
            ):
                model = copy.deepcopy(base)
                model.gradient_checkpointing = checkpointed
                hidden = hidden_base.clone().requires_grad_()
                context = context_base.clone().requires_grad_()
                layout = layout_for(hidden, 2)
                entries = [
                    LayerInputs(
                        attention_layout=layout,
                        target_context=context,
                        position_ids=layout.key_positions,
                    )
                    for _ in layers
                ]
                torch.manual_seed(43)
                if manual:
                    state = model.begin(hidden, entries)
                    for _ in layers:
                        state = model.run_next_layer(state)
                    output = model.finalize(state)
                else:
                    output = model(hidden, entries)
                output.square().sum().backward()
                result = [
                    output.detach(),
                    hidden.grad,
                    context.grad,
                    *(parameter.grad for parameter in model.parameters()),
                ]
                self.assertTrue(all(value is not None for value in result))
                if expected is None:
                    expected = result
                else:
                    torch.testing.assert_close(result, expected, rtol=0, atol=0)
                self.assertTrue(
                    all(entry.position_embeddings is None for entry in entries)
                )

    def test_invalid_layout_streams_and_convolution_contract_fail(self):
        module = make_layer(make_node())
        hidden = torch.randn(1, 3, 16)
        layout = layout_for(hidden)
        with patch.object(module.self_attn, "forward") as attention:
            for entry in (LayerInputs(), LayerInputs(attention_mask=layout.visibility)):
                with self.assertRaisesRegex(ValueError, "attention_layout"):
                    module(hidden, entry)
            with self.assertRaisesRegex(ValueError, "single"):
                module(hidden.unsqueeze(0), LayerInputs(attention_layout=layout))
            attention.assert_not_called()
        with self.assertRaisesRegex(TypeError, "module pair"):
            DecoderLayer(
                nn.Identity(),
                nn.Identity(),
                hidden_size=16,
                norm=RMSNormSpec(eps=1e-6, weight_offset=0),
                sublayer_conv=nn.Identity(),
            )
        with patch.object(module.self_attn, "forward", return_value=torch.zeros(16)):
            with self.assertRaisesRegex(ValueError, "residual shape"):
                module(hidden, LayerInputs(attention_layout=layout))

    def test_convolutions_are_independent_and_identity_matches_plain_layer(self):
        node = make_node(rope_parameters=NoRoPESpec())
        wrapped = make_layer(node, rotary_dim=0, convolution=True)
        plain = make_layer(node, rotary_dim=0)
        plain.load_state_dict(
            {
                name: value
                for name, value in wrapped.state_dict().items()
                if not name.startswith(("attention_conv.", "mlp_conv."))
            },
            strict=True,
        )
        self.assertIs(wrapped.attention_conv.config, wrapped.mlp_conv.config)
        for a, b in zip(
            wrapped.attention_conv.parameters(), wrapped.mlp_conv.parameters()
        ):
            self.assertNotEqual(a.data_ptr(), b.data_ptr())
        hidden = torch.randn(1, 8, 16, requires_grad=True)
        other = hidden.detach().clone().requires_grad_()
        entry = LayerInputs(attention_layout=layout_for(hidden))
        torch.manual_seed(11)
        output = wrapped(hidden, entry)
        torch.manual_seed(11)
        expected = plain(other, entry)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        output.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(hidden.grad, other.grad, rtol=0, atol=0)
        for name, parameter in plain.named_parameters():
            torch.testing.assert_close(
                dict(wrapped.named_parameters())[name].grad,
                parameter.grad,
                rtol=0,
                atol=0,
            )
        for conv in (wrapped.attention_conv, wrapped.mlp_conv):
            for parameter in conv.parameters():
                self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0)
        for invalid, error in (
            ((wrapped.attention_conv,), TypeError),
            ((nn.Identity(), nn.Identity()), TypeError),
            ((wrapped.attention_conv, wrapped.attention_conv), ValueError),
        ):
            with self.assertRaises(error):
                DecoderLayer(
                    nn.Identity(),
                    nn.Identity(),
                    hidden_size=16,
                    norm=RMSNormSpec(eps=1e-6, weight_offset=0),
                    sublayer_conv=invalid,
                )

    def test_context_dtype_validation_only_allows_autocast_supported_pairs(self):
        module = make_layer(make_node(rope_parameters=NoRoPESpec()), rotary_dim=0)
        hidden = torch.randn(1, 3, 16)
        layout = layout_for(hidden, 2)
        with self.assertRaisesRegex(ValueError, "autocast"):
            module(
                hidden,
                LayerInputs(
                    attention_layout=layout,
                    target_context=torch.zeros(1, 2, 16, dtype=torch.bfloat16),
                ),
            )
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            for dtype in (torch.long, torch.float64):
                with self.assertRaises(ValueError):
                    module(
                        hidden,
                        LayerInputs(
                            attention_layout=layout,
                            target_context=torch.zeros(1, 2, 16, dtype=dtype),
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
