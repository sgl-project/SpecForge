"""Runtime settings are fixed; HF adapters cannot change the training model."""

import copy
import json
import pickle
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import mock

import torch
from transformers import Qwen3Config

from specforge.config.draft import AutoDraftConfig, DraftModelConfig
from specforge.config.draft.contract import draft_config_contract
from specforge.config.draft.eagle3 import EaglePretrainedConfig
from specforge.config.draft.registry import DRAFT_CONFIG_REGISTRY
from specforge.modeling.auto import AutoDraftModel
from tests.test_modeling.test_all_draft_configs import architecture_payloads


class ImmutableDraftConfigTest(unittest.TestCase):
    def test_inline_parsing_does_not_use_hf_constructors(self):
        with (
            mock.patch.object(
                Qwen3Config, "from_dict", side_effect=AssertionError("HF parse")
            ),
            mock.patch.object(
                EaglePretrainedConfig,
                "from_dict",
                side_effect=AssertionError("HF parse"),
            ),
        ):
            for name, payload in architecture_payloads().items():
                with self.subTest(architecture=name):
                    payload["draft_vocab_size"] = None
                    config = AutoDraftConfig.from_inline(payload)
                    self.assertIsInstance(config, DraftModelConfig)
                    self.assertEqual(config.draft_vocab_size, config.vocab_size)
                    self.assertFalse(config.tie_word_embeddings)

    def test_inline_defaults_preserve_existing_checkpoint_contracts(self):
        for name, payload in architecture_payloads().items():
            with self.subTest(architecture=name):
                inline = AutoDraftConfig.from_inline(payload)
                legacy = AutoDraftConfig.from_dict(payload)
                self.assertEqual(
                    draft_config_contract(inline), draft_config_contract(legacy)
                )

    def test_repository_configs_keep_their_existing_checkpoint_contracts(self):
        for path in (Path(__file__).resolve().parents[2] / "configs").glob("*.json"):
            payload = json.loads(path.read_text())
            if payload.get("architectures", [None])[0] not in DRAFT_CONFIG_REGISTRY:
                continue
            with self.subTest(config=path.name):
                self.assertEqual(
                    draft_config_contract(AutoDraftConfig.from_inline(payload)),
                    draft_config_contract(AutoDraftConfig.from_dict(payload)),
                )

    def test_inline_eagle_defaults_run_a_forward(self):
        payload = architecture_payloads()["LlamaForCausalLMEagle3"]
        model = AutoDraftModel.from_config(AutoDraftConfig.from_inline(payload)).eval()
        with torch.no_grad():
            result = model(
                inputs_embeds=torch.randn(1, 3, model.config.hidden_size),
                hidden_states=torch.randn(1, 3, model.config.target_hidden_size * 3),
            )
        self.assertEqual(result.shape, (1, 3, model.config.hidden_size))
        self.assertTrue(torch.isfinite(result).all())

    def test_nested_settings_are_immutable_and_do_not_alias_input(self):
        payload = architecture_payloads()["DFlashDraftModel"]
        payload["extension"] = {"layers": [{"kind": "gqa"}]}
        config = AutoDraftConfig.from_inline(payload)
        original = config.to_dict()
        payload["extension"]["layers"][0]["kind"] = "mla"
        with self.assertRaises(FrozenInstanceError):
            config.block_size = 99
        with self.assertRaises(TypeError):
            config.dflash_config["mask_token_id"] = 99
        with self.assertRaises(TypeError):
            config.extension["layers"][0]["kind"] = "mla"
        with self.assertRaises(TypeError):
            config.layer_types[0] = "sliding_attention"
        plain = config.to_dict()
        plain["extension"]["layers"].append({"kind": "mla"})
        self.assertEqual(config.to_dict(), original)

    def test_hf_adapters_are_independent(self):
        config = AutoDraftConfig.from_inline(
            architecture_payloads()["DFlashDraftModel"]
        )
        original = config.to_dict()
        adapter = config.to_hf_config()
        adapter.block_size = 99
        adapter.dflash_config["target_layer_ids"].append(999)
        self.assertEqual(config.to_dict(), original)
        self.assertEqual(config.to_hf_config().block_size, config.block_size)

    def test_models_and_layers_share_the_resolved_object(self):
        for name, payload in architecture_payloads().items():
            with self.subTest(architecture=name):
                config = AutoDraftConfig.from_inline(payload)
                model = AutoDraftModel.from_config(config)
                self.assertIs(model.config, config)
                self.assertNotIn("draft_options", vars(model))
                for module in model.modules():
                    child_config = getattr(module, "config", None)
                    if isinstance(child_config, DraftModelConfig):
                        self.assertIs(child_config, config)
                with self.assertRaisesRegex(AttributeError, "fixed"):
                    model.config = config.to_hf_config()
                cloned = copy.deepcopy(model)
                self.assertIs(cloned.config, config)
                self.assertIsNot(next(cloned.parameters()), next(model.parameters()))

    def test_config_and_nested_values_can_be_pickled(self):
        config = AutoDraftConfig.from_inline(
            architecture_payloads()["DSpineDraftModel"]
        )
        restored = pickle.loads(pickle.dumps(config))
        self.assertEqual(restored.to_dict(), config.to_dict())
        self.assertEqual(
            dict(pickle.loads(pickle.dumps(config.dspine_config))),
            dict(config.dspine_config),
        )
        with self.assertRaises(TypeError):
            restored.dspine_config["top_k"] = 99

    def test_inline_models_export_and_reload_with_the_same_weights_and_contract(self):
        for name, payload in architecture_payloads().items():
            with (
                self.subTest(architecture=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                config = AutoDraftConfig.from_inline(payload)
                model = AutoDraftModel.from_config(config)
                original = config.to_dict()
                model.save_pretrained(directory)
                restored = AutoDraftModel.from_pretrained(directory, config=config)
                self.assertEqual(config.to_dict(), original)
                self.assertEqual(
                    draft_config_contract(restored.config),
                    draft_config_contract(config),
                )
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(restored.state_dict()[key], value)

    def test_invalid_inline_settings_fail_before_model_allocation(self):
        payload = architecture_payloads()["DFlashDraftModel"]
        with mock.patch("torch.nn.Linear", side_effect=AssertionError("allocated")):
            for invalid in (0, -1, True, 2.5):
                with self.subTest(draft_vocab_size=invalid):
                    with self.assertRaisesRegex(ValueError, "draft_vocab_size"):
                        AutoDraftModel.from_config(
                            AutoDraftConfig.from_inline(
                                {**payload, "draft_vocab_size": invalid}
                            )
                        )


if __name__ == "__main__":
    unittest.main()
