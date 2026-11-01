"""Draft construction uses user settings; checkpoints only provide weights."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml
from transformers import PretrainedConfig

from specforge.config.draft import AutoDraftConfig
from specforge.config.draft.serialization.checkpoint import read_checkpoint_contract
from specforge.config.draft.serialization.contract import draft_config_contract
from specforge.config.draft.serialization.format import encode_format
from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.draft.registry import DRAFT_REGISTRY


class DraftConfigTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "run.yaml"
        self.payload = {
            "architectures": ["LlamaForCausalLMEagle3"],
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "eagle_config": {"eagle_aux_hidden_state_layer_ids": [1, 2, 3]},
        }
        self.write_recipe(self.payload)

    def write_recipe(self, payload):
        self.path.write_text(yaml.safe_dump({"model": {"draft_model_config": payload}}))

    def test_yaml_is_the_only_file_configuration_entry_point(self):
        config = AutoDraftConfig.from_yaml(self.path)
        self.assertEqual(
            config.to_dict(), AutoDraftConfig.from_inline(self.payload).to_dict()
        )
        self.assertFalse(hasattr(AutoDraftConfig, "from_file"))
        self.assertFalse(hasattr(AutoDraftConfig, "from_pretrained"))
        for value in (
            self.payload,
            {"model": {}},
            {"model": {"draft_model_config": "config.json"}},
        ):
            self.path.write_text(yaml.safe_dump(value))
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "model.draft_model_config"),
            ):
                AutoDraftConfig.from_yaml(self.path)

    def test_null_vocab_defaults_without_changing_user_input(self):
        payload = {**self.payload, "draft_vocab_size": None}
        self.write_recipe(payload)
        config = AutoDraftConfig.from_yaml(self.path)
        self.assertEqual(config.draft_vocab_size, payload["vocab_size"])
        self.assertIsNone(payload["draft_vocab_size"])

    def test_invalid_scalar_and_architecture_fail_before_allocation(self):
        for value in (False, 0, -1, 1.5, "128"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "draft_vocab_size"),
            ):
                AutoDraftConfig.from_inline({**self.payload, "draft_vocab_size": value})
        for value in (None, [], "LlamaForCausalLMEagle3", [1], ["a", "b"]):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "exactly one"),
            ):
                AutoDraftConfig.from_inline({**self.payload, "architectures": value})

    def test_all_pretrained_entry_points_require_resolved_user_config(self):
        model_cls = DRAFT_REGISTRY["LlamaForCausalLMEagle3"]
        with mock.patch.object(PretrainedConfig, "get_config_dict") as hub:
            for loader in (AutoDraftModel.from_pretrained, model_cls.from_pretrained):
                for source in ("org/draft", self.directory.name):
                    for config in (
                        None,
                        "config.json",
                        self.payload,
                        PretrainedConfig(),
                    ):
                        with self.subTest(loader=loader, source=source, config=config):
                            with self.assertRaisesRegex(ValueError, "user's YAML"):
                                loader(source, config=config)
            hub.assert_not_called()

    def test_source_options_and_explicit_config_reach_weight_loader(self):
        config = AutoDraftConfig.from_yaml(self.path)
        model_cls = DRAFT_REGISTRY["LlamaForCausalLMEagle3"]
        with mock.patch.object(model_cls, "from_pretrained") as load:
            AutoDraftModel.from_pretrained(
                "org/draft",
                config=config,
                revision="branch",
                subfolder="draft",
                local_files_only=True,
            )
        self.assertIs(load.call_args.kwargs["config"], config)
        self.assertEqual(load.call_args.kwargs["revision"], "branch")
        self.assertEqual(load.call_args.kwargs["subfolder"], "draft")

    def test_canonical_hub_metadata_is_comparison_only_and_never_mutated(self):
        config = AutoDraftConfig.from_yaml(self.path)
        payload = encode_format(config)
        original = copy.deepcopy(payload)
        model_cls = DRAFT_REGISTRY["LlamaForCausalLMEagle3"]
        with (
            mock.patch.object(
                PretrainedConfig, "get_config_dict", return_value=(payload, {})
            ) as hub,
            mock.patch(
                "transformers.modeling_utils.PreTrainedModel.from_pretrained"
            ) as weights,
        ):
            model_cls.from_pretrained(
                "org/draft",
                config=config,
                revision="branch",
                subfolder="draft",
                local_files_only=True,
            )
        self.assertEqual(payload, original)
        self.assertFalse(hasattr(config, "_name_or_path"))
        self.assertEqual(hub.call_args.kwargs["revision"], "branch")
        self.assertEqual(hub.call_args.kwargs["subfolder"], "draft")
        self.assertEqual(
            weights.call_args.kwargs["config"].layers, config.to_dict()["layers"]
        )

    def test_checkpoint_mismatch_fails_before_model_or_weights_are_loaded(self):
        config = AutoDraftConfig.from_yaml(self.path)
        payload = encode_format(config)
        payload["layers"][0]["ffn"]["intermediate_size"] += 8
        with (
            mock.patch.object(
                PretrainedConfig, "get_config_dict", return_value=(payload, {})
            ),
            mock.patch(
                "transformers.modeling_utils.PreTrainedModel.from_pretrained"
            ) as weights,
        ):
            with self.assertRaisesRegex(ValueError, "intermediate_size"):
                AutoDraftModel.from_pretrained("org/draft", config=config)
        weights.assert_not_called()
        self.assertEqual(config.layers[0].ffn.intermediate_size, 64)

    def test_metadata_reader_supports_legacy_local_config_without_constructing_model(
        self,
    ):
        path = Path(self.directory.name) / "config.json"
        path.write_text(json.dumps(self.payload))
        expected = draft_config_contract(AutoDraftConfig.from_dict(self.payload))
        for source in (path, path.parent, path.as_uri()):
            with self.subTest(source=source):
                self.assertEqual(read_checkpoint_contract(source), expected)

    def test_weights_roundtrip_uses_yaml_config_for_each_family(self):
        for architecture in ("LlamaForCausalLMEagle3", "DFlashDraftModel"):
            with self.subTest(architecture=architecture):
                self.write_recipe(
                    {
                        **self.payload,
                        "architectures": [architecture],
                        "num_target_layers": 4,
                        "block_size": 4,
                    }
                )
                config = AutoDraftConfig.from_yaml(self.path)
                before = config.to_dict()
                model = AutoDraftModel.from_config(config)
                folder = Path(self.directory.name) / architecture
                model.save_pretrained(folder)
                restored = AutoDraftModel.from_pretrained(
                    folder, config=config, local_files_only=True
                )
                self.assertEqual(config.to_dict(), before)
                torch.testing.assert_close(
                    restored.state_dict(), model.state_dict(), rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
