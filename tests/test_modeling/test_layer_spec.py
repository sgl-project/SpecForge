"""Component contracts, per-layer execution and legacy export boundaries."""

import copy
import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import torch

from specforge.config.draft import AutoDraftConfig, GQASpec, register_component
from specforge.config.draft.contract import draft_config_contract
from specforge.modeling.auto import AutoDraftModel
from tests.test_modeling.test_all_draft_configs import architecture_payloads


class LayerSpecTest(unittest.TestCase):
    def payload(self, architecture="DFlashDraftModel", **updates):
        return {**architecture_payloads()[architecture], **updates}

    def run_model(self, model):
        noise = torch.randn(1, 4, model.config.hidden_size, requires_grad=True)
        target = torch.randn(
            1,
            3,
            model.config.hidden_size
            * len(model.config.dflash_config["target_layer_ids"]),
            requires_grad=True,
        )
        output = model(
            position_ids=torch.arange(7)[None],
            noise_embedding=noise,
            target_hidden=target,
        )
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(noise.grad).all())
        self.assertTrue(torch.isfinite(target.grad).all())
        return output

    def test_legacy_and_explicit_layers_match_outputs_and_gradients(self):
        legacy = AutoDraftConfig.from_inline(self.payload())
        canonical = AutoDraftConfig.from_inline(legacy.to_dict())
        first, second = AutoDraftModel.from_config(legacy), AutoDraftModel.from_config(
            canonical
        )
        second.load_state_dict(first.state_dict())
        torch.manual_seed(12)
        a = self.run_model(first)
        torch.manual_seed(12)
        b = self.run_model(second)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        for left, right in zip(first.parameters(), second.parameters()):
            if left.grad is not None:
                torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)

    def test_mixed_projection_widths_windows_and_ffns_reach_forward(self):
        config = AutoDraftConfig.from_inline(
            self.payload(
                num_hidden_layers=2,
                layers=[
                    {
                        "attention": {
                            "type": "gqa",
                            "head_dim": 8,
                            "sliding_window": 2,
                            "is_causal": False,
                        },
                        "ffn": {"intermediate_size": 24},
                    },
                    {
                        "attention": {
                            "type": "mla",
                            "num_attention_heads": 2,
                            "kv_lora_rank": 8,
                            "qk_nope_head_dim": 4,
                            "qk_rope_head_dim": 4,
                            "v_head_dim": 6,
                            "is_causal": True,
                        },
                        "ffn": {"intermediate_size": 40},
                    },
                ],
            )
        )
        model = AutoDraftModel.from_config(config)
        self.assertEqual(model.layers[0].self_attn.q_proj.out_features, 32)
        self.assertEqual(model.layers[0].mlp.gate_proj.out_features, 24)
        self.assertEqual(model.layers[1].mlp.gate_proj.out_features, 40)
        self.assertEqual(model.layers[0].self_attn.sliding_window, 2)
        self.assertTrue(model.layers[1].self_attn.is_causal)
        self.run_model(model)
        self.assertEqual(len(draft_config_contract(config)["layers"]), 2)
        with self.assertRaisesRegex(ValueError, "legacy serving"):
            config.to_hf_config()

    def test_convolution_is_per_layer_and_algorithm_options_do_not_own_it(self):
        payload = self.payload(
            "DFlash2DraftModel",
            num_hidden_layers=2,
            layers=[
                {"conv": {"kernel_size": 2, "group_size": 4}},
                {"conv": {"kernel_size": 3, "group_size": 8}},
            ],
        )
        config = AutoDraftConfig.from_inline(payload)
        self.assertNotIn("conv_kernel_size", config.dflash_config)
        self.assertEqual(config.layers[1].conv.kernel_size, 3)
        self.run_model(AutoDraftModel.from_config(config))

    def test_new_component_needs_no_algorithm_allowlist(self):
        register_component(
            "attention",
            "test_gqa",
            GQASpec,
            builders={
                "context_then_draft": "specforge.modeling.draft.models.layers.attention:build_gqa",
            },
        )
        config = AutoDraftConfig.from_inline(
            self.payload(layers={"attention": {"type": "test_gqa"}})
        )
        self.run_model(AutoDraftModel.from_config(config))
        with self.assertRaisesRegex(ValueError, "eagle_cache"):
            AutoDraftConfig.from_inline(
                self.payload(
                    "LlamaForCausalLMEagle3", layers={"attention": {"type": "test_gqa"}}
                )
            )

    def test_contracts_and_typos_fail_before_allocation(self):
        cases = [
            (self.payload(layers={"attention": {"type": "dsa"}}), "Unknown"),
            (self.payload(layers={"ffn": {"routing": "typo"}}), "Unknown"),
            (self.payload(layers={"attention": {"is_causal": "true"}}), "boolean"),
            (
                self.payload(
                    "DSpineDraftModel", layers={"attention": {"is_causal": False}}
                ),
                "causal",
            ),
            (
                self.payload(
                    "LlamaForCausalLMEagle3",
                    layers={
                        "attention": {
                            "type": "mla",
                            "kv_lora_rank": 4,
                            "qk_nope_head_dim": 4,
                            "qk_rope_head_dim": 4,
                            "v_head_dim": 4,
                        }
                    },
                ),
                "eagle_cache",
            ),
        ]
        with patch("torch.nn.Linear", side_effect=AssertionError("allocated")):
            for data, error in cases:
                with (
                    self.subTest(error=error),
                    self.assertRaisesRegex(ValueError, error),
                ):
                    AutoDraftConfig.from_inline(data)

    def test_mixed_full_and_swa_is_legacy_representable(self):
        config = AutoDraftConfig.from_inline(
            self.payload(
                num_hidden_layers=2,
                layers=[
                    {"attention": {"is_causal": False}},
                    {"attention": {"sliding_window": 3, "is_causal": True}},
                ],
            )
        )
        adapter = config.to_hf_config()
        self.assertEqual(adapter.layer_types, ["full_attention", "sliding_attention"])
        self.assertIsNone(adapter.is_causal)
        self.assertEqual(AutoDraftConfig.resolve(adapter).layers, config.layers)

    def test_per_layer_windows_are_not_silently_exported(self):
        config = AutoDraftConfig.from_inline(
            self.payload(
                num_hidden_layers=2,
                layers=[
                    {"attention": {"sliding_window": 2}},
                    {"attention": {"sliding_window": 4}},
                ],
            )
        )
        self.run_model(AutoDraftModel.from_config(config))
        with self.assertRaisesRegex(ValueError, "sliding windows"):
            config.to_hf_config()

    def test_runtime_layers_are_immutable_and_share_common_settings(self):
        payload = self.payload(
            num_hidden_layers=2, layers={"attention": {"is_causal": True}}
        )
        config = AutoDraftConfig.from_inline(payload)
        payload["layers"]["attention"]["is_causal"] = False
        self.assertIsInstance(config.layers, tuple)
        with self.assertRaises(FrozenInstanceError):
            config.layers[0].attention.is_causal = False
        model = AutoDraftModel.from_config(config)
        for layer in model.layers:
            self.assertIs(layer.self_attn.config.model, config)
        self.assertEqual(copy.deepcopy(config), config)

    def test_uniform_layers_reuse_rotary_computation(self):
        model = AutoDraftModel.from_config(
            AutoDraftConfig.from_inline(
                self.payload(
                    num_hidden_layers=2, layers={"attention": {"is_causal": False}}
                )
            )
        )
        with (
            patch.object(
                model.rotary_emb, "forward", wraps=model.rotary_emb.forward
            ) as first,
            patch.object(
                model.extra_rotary_emb[0],
                "forward",
                wraps=model.extra_rotary_emb[0].forward,
            ) as second,
        ):
            self.run_model(model)
        self.assertEqual(first.call_count, 1)
        self.assertEqual(second.call_count, 0)

    def test_native_mtp_checks_and_uses_effective_layer_dimensions(self):
        from specforge.config.draft.mtp import Qwen3_5MTPConfig

        payload = self.payload(
            "Qwen3_5MTPDraftModel", layers={"attention": {"head_dim": 8}}
        )
        config = AutoDraftConfig.from_inline(payload)
        model = AutoDraftModel.from_config(config)
        self.assertEqual(model.mtp.layers[0].self_attn.head_dim, 8)
        self.assertEqual(model.mtp.rotary_emb.inv_freq.numel(), 2)
        native = AutoDraftConfig.from_inline(
            self.payload("Qwen3_5MTPDraftModel")
        ).to_hf_config()
        with self.assertRaisesRegex(ValueError, "head_dim"):
            Qwen3_5MTPConfig.validate_native_config(config, native)
