"""Configuration assembly preserves layer identity and mixed-stack execution."""

import copy
import unittest
from dataclasses import dataclass, replace
from typing import ClassVar
from unittest.mock import patch

import torch

from specforge.config.draft.architecture import DFlash2Spec
from specforge.config.draft.conv import ConvSpec, GroupedCausalConvSpec
from specforge.config.draft.layer import LayerSpec
from specforge.config.draft.registry import _SCHEMAS, register_component_schema
from specforge.config.draft.rope import DefaultRoPESpec, NoRoPESpec
from specforge.modeling.draft.layers import GQAAttention, MLAAttention
from specforge.modeling.draft.layers.inputs import LayerInputs
from specforge.modeling.draft.layers.registry import (
    _BUILDERS,
    register_component_builder,
)
from specforge.modeling.draft.models.assembly import build_decoder_backbone
from specforge.modeling.draft.models.eagle_backbone import build_eagle_backbone
from tests.test_config.draft_fixtures import draft_config
from tests.test_config.test_draft_eagle3_spec import eagle_config
from tests.test_config.test_draft_mtp_spec import mtp_config
from tests.test_config.test_draft_peagle_spec import peagle_config
from tests.test_modeling.test_draft_gqa import layout_for
from tests.test_modeling.test_draft_mla import mla_node


def mixed_config(*, convolution=False, latent_rope=None):
    base = draft_config()
    conv = "grouped" if convolution else None
    return replace(
        base,
        architecture="DFlash2DraftModel" if convolution else base.architecture,
        attention_configs={
            "full": base.attention_configs["full"],
            "latent": mla_node(
                rope_parameters=NoRoPESpec() if latent_rope is None else latent_rope,
                attention_dropout=0,
            ),
        },
        rope_parameters=DefaultRoPESpec(
            interleaved=False, rope_theta=10000, partial_rotary_factor=0.5
        ),
        conv_configs=(
            {"grouped": GroupedCausalConvSpec(kernel_size=3, group_size=4)}
            if convolution
            else {}
        ),
        layer_configs={
            "plain": LayerSpec(attention="full", mlp="dense", sublayer_conv=conv),
            "latent": LayerSpec(attention="latent", mlp="dense", sublayer_conv=conv),
        },
        layer_types=("latent", "plain", "plain"),
        spec_config=(
            DFlash2Spec(
                block=base.spec_config.block,
                target_features=base.spec_config.target_features,
                selector_rank=4,
                selector_top_k=4,
                output_multiplier=1,
                final_logit_softcapping=None,
            )
            if convolution
            else base.spec_config
        ),
    )


class DecoderAssemblyTest(unittest.TestCase):
    def test_named_layers_reuse_nodes_but_not_parameters(self):
        config = mixed_config(convolution=True)
        config_hash = hash(config)
        model = build_decoder_backbone(config, backend="eager")
        self.assertIsInstance(model.layers[0].self_attn, MLAAttention)
        self.assertIsInstance(model.layers[1].self_attn, GQAAttention)
        self.assertIsNone(model.rotary[0])
        self.assertEqual(model.layers[0].self_attn.rotary_dim, 0)
        self.assertEqual(model.rotary[1].rotary_dim, 4)
        self.assertIs(model.rotary[1].config, config.rope_parameters)
        for layer, name in zip(model.layers, config.layer_types):
            definition = config.layer_configs[name]
            self.assertIs(
                layer.self_attn.config, config.attention_configs[definition.attention]
            )
            self.assertIs(layer.mlp.config, config.mlp_configs[definition.mlp])
            self.assertIs(layer.input_layernorm.config, config.norm_config)
            for conv in (layer.attention_conv, layer.mlp_conv):
                self.assertIs(
                    conv.config, config.conv_configs[definition.sublayer_conv]
                )
                self.assertEqual(
                    conv.block_size, config.spec_config.protocol.num_model_slots
                )
        # No accidental sharing across repeated layer references, sublayer norms,
        # or the two convolution instances selected by a single LayerSpec field.
        parameters = list(model.named_parameters(remove_duplicate=False))
        self.assertEqual(len(parameters), len({id(p) for _, p in parameters}))
        self.assertEqual(len(parameters), len({p.data_ptr() for _, p in parameters}))
        self.assertIs(model.final_norm.config, config.norm_config)
        self.assertIs(model.residual.config, config.residual_config)
        self.assertEqual(hash(config), config_hash)

    def test_explicit_meta_device_and_dtype(self):
        model = build_decoder_backbone(
            mixed_config(convolution=True),
            backend="eager",
            device="meta",
            dtype=torch.bfloat16,
        )
        for parameter in model.parameters():
            self.assertEqual(parameter.device.type, "meta")
            self.assertEqual(parameter.dtype, torch.bfloat16)

    def _assert_checkpoint_parity(self, device):
        for convolution, latent_rope in (
            (
                False,
                DefaultRoPESpec(
                    interleaved=False, rope_theta=50000, partial_rotary_factor=1
                ),
            ),
            (True, None),
        ):
            with self.subTest(device=device, convolution=convolution):
                config = mixed_config(convolution=convolution, latent_rope=latent_rope)
                model = build_decoder_backbone(config, backend="eager", device=device)
                checkpointed = copy.deepcopy(model)
                checkpointed.gradient_checkpointing = True
                hidden = torch.randn(2, 8, 32, device=device, requires_grad=True)
                context = torch.randn(2, 3, 32, device=device, requires_grad=True)
                other = hidden.detach().clone().requires_grad_()
                other_context = context.detach().clone().requires_grad_()
                layout = layout_for(hidden, context_length=3)

                def inputs(target):
                    return [
                        LayerInputs(
                            attention_layout=layout,
                            position_ids=layout.key_positions,
                            target_context=target,
                        )
                        for _ in config.layer_types
                    ]

                entries = inputs(context)
                output = model(hidden, entries)
                expected = checkpointed(other, inputs(other_context))
                self.assertEqual(output.shape, hidden.shape)
                self.assertTrue(torch.isfinite(output).all())
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                # RoPE preparation must not mutate caller-owned LayerInputs.
                self.assertTrue(
                    all(entry.position_embeddings is None for entry in entries)
                )
                upstream = torch.randn_like(output)
                output.backward(upstream)
                expected.backward(upstream)
                torch.testing.assert_close(hidden.grad, other.grad, rtol=0, atol=0)
                torch.testing.assert_close(
                    context.grad, other_context.grad, rtol=0, atol=0
                )
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    torch.testing.assert_close(
                        parameter.grad,
                        dict(checkpointed.named_parameters())[name].grad,
                        rtol=0,
                        atol=0,
                    )

    def test_cpu_mixed_stack_outputs_and_checkpoint_gradients(self):
        self._assert_checkpoint_parity("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_mixed_stack_outputs_and_checkpoint_gradients(self):
        self._assert_checkpoint_parity("cuda")

    def test_missing_builders_fail_before_allocating_weights(self):
        config = mixed_config(convolution=True)
        for key in (
            ("attention", "mla"),
            ("mlp", "dense"),
            ("conv", "grouped_causal"),
            ("norm", "rms"),
            ("residual", "standard"),
        ):
            with (
                self.subTest(key=key),
                patch.dict(_BUILDERS),
                patch("torch.nn.Linear") as allocate,
            ):
                _BUILDERS.pop(key)
                with self.assertRaisesRegex(ValueError, "no modeling builder"):
                    build_decoder_backbone(config, backend="eager")
                allocate.assert_not_called()

    def test_protocol_less_conv_is_explicit_and_eagle_rejects_it(self):
        @dataclass(frozen=True, kw_only=True)
        class SlotlessConvSpec(ConvSpec):
            type: ClassVar[str] = "test_slotless"

        class SlotlessConv(torch.nn.Module):
            def __init__(self, config, *, block_size, **kwargs):
                super().__init__()
                self.config = config
                if block_size is not None:
                    raise ValueError("expected no block protocol")

            def prepare(self, hidden):
                return hidden, None

            def finish(self, hidden, coefficients):
                return hidden

        with patch.dict(_SCHEMAS), patch.dict(_BUILDERS):
            register_component_schema("conv", SlotlessConvSpec)
            register_component_builder("conv", SlotlessConvSpec, SlotlessConv)
            for base in (mtp_config(), eagle_config(), peagle_config()):
                config = replace(
                    base,
                    conv_configs={"slotless": SlotlessConvSpec()},
                    layer_configs={
                        name: replace(layer, sublayer_conv="slotless")
                        for name, layer in base.layer_configs.items()
                    },
                )
                if config.architecture == "MTPDraftModel":
                    model = build_decoder_backbone(config, backend="eager")
                    self.assertIsInstance(model.layers[0].attention_conv, SlotlessConv)
                else:
                    with patch("torch.nn.Linear") as allocate:
                        with self.assertRaisesRegex(
                            ValueError, "does not support sublayer_conv"
                        ):
                            build_eagle_backbone(config, backend="eager")
                        allocate.assert_not_called()

    def test_nonstandard_fusion_and_unsupported_backend_fail_early(self):
        with patch("torch.nn.Linear") as allocate:
            for config in (eagle_config(), peagle_config()):
                with self.assertRaisesRegex(ValueError, "input fusion"):
                    build_decoder_backbone(config, backend="eager")
            with self.assertRaisesRegex(ValueError, "eager"):
                build_decoder_backbone(draft_config(), backend="flex")
            with self.assertRaisesRegex(TypeError, "DraftConfig"):
                build_decoder_backbone({}, backend="eager")
            allocate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
