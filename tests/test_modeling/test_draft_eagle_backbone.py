"""EAGLE assembly preserves legacy eager wiring and shared checkpoint execution."""

import copy
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
from torch import nn
from transformers import LlamaConfig

from specforge.config.draft.layer import LayerSpec
from specforge.config.draft.rope import NoRoPESpec
from specforge.modeling.draft.layers import DecoderLayer, MLAAttention
from specforge.modeling.draft.layers.inputs import LayerInputs
from specforge.modeling.draft.layers.registry import _BUILDERS
from specforge.modeling.draft.llama3_eagle import LlamaRMSNorm
from specforge.modeling.draft.models.eagle_backbone import build_eagle_backbone
from specforge.modeling.draft.models.eagle_fusion import (
    AuxiliaryFeatureProjection,
    EagleDecoderStage,
)
from specforge.modeling.draft.peagle import PEagleFirstLayer, PEagleStandardLayer
from tests.test_config.draft_fixtures import draft_config
from tests.test_config.test_draft_eagle3_spec import eagle_config
from tests.test_config.test_draft_peagle_spec import peagle_config
from tests.test_modeling.test_draft_eagle_fusion import eager_reference
from tests.test_modeling.test_draft_gqa import layout_for
from tests.test_modeling.test_draft_mla import mla_node


def config_for(kind, option):
    base = (
        peagle_config(layer_types=("full_dense",) * 3)
        if kind == "peagle"
        else eagle_config()
    )
    return replace(
        base,
        attention_configs={
            "full": replace(base.attention_configs["full"], qk_norm=False)
        },
        spec_config=replace(
            base.spec_config,
            **(
                {"norm_before_residual": option}
                if kind == "peagle"
                else {"norm_output": option}
            ),
        ),
    )


class EagleBackboneTest(unittest.TestCase):
    def _parity(self, device):
        for kind in ("eagle3", "peagle"):
            for option in (False, True):
                with self.subTest(device=device, kind=kind, option=option):
                    config = config_for(kind, option)
                    actual = build_eagle_backbone(
                        config, backend="eager", device=device
                    )
                    reference_config = LlamaConfig(
                        hidden_size=32,
                        intermediate_size=64,
                        num_attention_heads=4,
                        num_key_value_heads=2,
                        head_dim=8,
                        rms_norm_eps=1e-6,
                        hidden_act="silu",
                        pretraining_tp=1,
                    )
                    references = nn.ModuleList(
                        [
                            PEagleFirstLayer(
                                reference_config,
                                norm_before_residual=(
                                    option if kind == "peagle" else False
                                ),
                            ),
                            *[
                                PEagleStandardLayer(reference_config)
                                for _ in config.layer_types[1:]
                            ],
                        ]
                    ).to(device)
                    references.load_state_dict(actual.layers.state_dict(), strict=True)
                    final_norm = (
                        LlamaRMSNorm(32).to(device)
                        if kind == "peagle" or option
                        else nn.Identity()
                    )
                    final_norm.load_state_dict(
                        actual.final_norm.state_dict(), strict=True
                    )
                    hidden = torch.randn(2, 4, 64, device=device, requires_grad=True)
                    other = hidden.detach().clone().requires_grad_()
                    layout = layout_for(hidden)
                    inputs = [
                        LayerInputs(
                            attention_layout=layout, position_ids=layout.key_positions
                        )
                        for _ in config.layer_types
                    ]
                    embeddings = actual.rotary[0](hidden, layout.key_positions)
                    causal = (
                        layout.key_positions[:, None, None, :]
                        <= layout.query_positions[:, None, :, None]
                    )
                    output = actual(hidden, inputs)
                    expected = other
                    with patch(
                        "specforge.modeling.draft.peagle.compile_friendly_flex_attention",
                        eager_reference,
                    ):
                        for layer in references:
                            expected = layer(expected, causal, embeddings)
                    expected = final_norm(expected)
                    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
                    upstream = torch.randn_like(output)
                    output.backward(upstream)
                    expected.backward(upstream)
                    torch.testing.assert_close(
                        hidden.grad, other.grad, rtol=1e-5, atol=1e-5
                    )
                    for name, parameter in actual.layers.named_parameters():
                        torch.testing.assert_close(
                            parameter.grad,
                            dict(references.named_parameters())[name].grad,
                            rtol=1e-5,
                            atol=1e-5,
                        )
                    for name, parameter in actual.final_norm.named_parameters():
                        torch.testing.assert_close(
                            parameter.grad,
                            dict(final_norm.named_parameters())[name].grad,
                            rtol=1e-5,
                            atol=1e-5,
                        )

    def test_cpu_legacy_eager_wiring(self):
        self._parity("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_legacy_eager_wiring(self):
        self._parity("cuda")

    def test_projection_and_mixed_layers_checkpoint_together(self):
        base = config_for("peagle", True)
        config = replace(
            base,
            attention_configs={
                **base.attention_configs,
                "latent": mla_node(rope_parameters=NoRoPESpec(), attention_dropout=0),
            },
            layer_configs={
                **base.layer_configs,
                "latent": LayerSpec(attention="latent", mlp="dense"),
            },
            layer_types=("full_dense", "latent", "full_dense"),
        )
        model = build_eagle_backbone(config, backend="eager")
        self.assertIsInstance(model.layers[0], EagleDecoderStage)
        self.assertIsInstance(model.layers[1], DecoderLayer)
        self.assertIsInstance(model.layers[1].self_attn, MLAAttention)
        self.assertIsNone(model.rotary[1])
        self.assertEqual(model.layers[0].self_attn.q_proj.in_features, 64)
        self.assertEqual(model.layers[2].self_attn.q_proj.in_features, 32)
        projection = AuxiliaryFeatureProjection(
            config.spec_config.target_features,
            hidden_size=32,
            normalize_features=False,
            norm=config.norm_config,
        )
        checkpointed, other_projection = copy.deepcopy(model), copy.deepcopy(projection)
        checkpointed.gradient_checkpointing = True
        features = torch.randn(2, 4, 96, requires_grad=True)
        embeds = torch.randn(2, 4, 32, requires_grad=True)
        other_features, other_embeds = (
            features.detach().clone().requires_grad_(),
            embeds.detach().clone().requires_grad_(),
        )
        layout = layout_for(embeds)
        inputs = [
            LayerInputs(attention_layout=layout, position_ids=layout.key_positions)
            for _ in config.layer_types
        ]
        output = model(torch.cat((embeds, projection(features)), -1), inputs)
        expected = checkpointed(
            torch.cat((other_embeds, other_projection(other_features)), -1), inputs
        )
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        upstream = torch.randn_like(output)
        output.backward(upstream)
        expected.backward(upstream)
        for value, other in ((features, other_features), (embeds, other_embeds)):
            torch.testing.assert_close(value.grad, other.grad, rtol=0, atol=0)
        for module, other in ((model, checkpointed), (projection, other_projection)):
            for name, parameter in module.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                torch.testing.assert_close(
                    parameter.grad,
                    dict(other.named_parameters())[name].grad,
                    rtol=0,
                    atol=0,
                )

    def test_repeated_definitions_have_independent_weights_and_meta_support(self):
        config = config_for("peagle", False)
        model = build_eagle_backbone(
            config, backend="eager", device="meta", dtype=torch.bfloat16
        )
        parameters = list(model.named_parameters(remove_duplicate=False))
        self.assertEqual(len(parameters), len({id(p) for _, p in parameters}))
        self.assertTrue(
            all(p.is_meta and p.dtype == torch.bfloat16 for _, p in parameters)
        )

    def test_invalid_assembly_fails_before_parameter_allocation(self):
        with patch("torch.nn.Linear") as allocate:
            with self.assertRaisesRegex(ValueError, "Eagle3Spec or PEagleSpec"):
                build_eagle_backbone(draft_config(), backend="eager")
            with self.assertRaisesRegex(ValueError, "eager"):
                build_eagle_backbone(eagle_config(), backend="flex")
            with patch.dict(_BUILDERS):
                _BUILDERS.pop(("norm", "rms"))
                with self.assertRaisesRegex(ValueError, "no modeling builder"):
                    build_eagle_backbone(eagle_config(), backend="eager")
            allocate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
