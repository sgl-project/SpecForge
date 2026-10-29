"""All draft config sources share normalization and registry dispatch."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from transformers import LlamaConfig, PretrainedConfig, Qwen3Config

from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
from specforge.modeling.draft.config import AutoDraftConfig
from specforge.modeling.draft.registry import DRAFT_REGISTRY
from specforge.training.model_loading import load_draft_config_source


class DraftConfigTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "config.json"
        self.payload = {
            "architectures": ["LlamaForCausalLMEagle3"],
            "model_type": "llama",
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "eagle_config": {"eagle_aux_hidden_state_layer_ids": [1, 2, 3]},
        }

    def write(self, payload):
        self.path.write_text(json.dumps(payload), encoding="utf-8")

    def test_sources_and_legacy_entry_points_are_equivalent(self):
        self.write(self.payload)
        expected = AutoDraftConfig.from_dict(self.payload).to_dict()
        for source in (self.path, self.path.parent, self.path.as_uri()):
            with self.subTest(source=source):
                self.assertEqual(
                    AutoDraftConfig.from_pretrained(source).to_dict(), expected
                )
                self.assertEqual(load_draft_config_source(source).to_dict(), expected)
        self.assertEqual(AutoDraftModelConfig.from_file(self.path).to_dict(), expected)
        self.assertFalse(expected["tie_word_embeddings"])
        self.assertEqual(expected["draft_vocab_size"], 128)

    def test_null_vocab_rejected_by_all_entry_points(self):
        payload = {**self.payload, "draft_vocab_size": None}
        self.write(payload)
        loaders = (
            lambda: AutoDraftConfig.from_dict(payload),
            lambda: AutoDraftModelConfig.from_file(self.path),
            lambda: load_draft_config_source(self.path.parent),
            lambda: AutoDraftModel.from_pretrained(self.path.parent),
        )
        for loader in loaders:
            with self.subTest(loader=loader):
                with self.assertRaisesRegex(
                    ValueError, "draft_vocab_size cannot be null"
                ):
                    loader()

    def test_normalization_is_independent_and_preserves_custom_fields(self):
        payload = {**self.payload, "tie_word_embeddings": True, "use_cache": False}
        before = copy.deepcopy(payload)
        config = AutoDraftConfig.from_dict(payload)
        self.assertFalse(config.tie_word_embeddings)
        self.assertFalse(config.use_cache)
        self.assertEqual(config.eagle_config, payload["eagle_config"])
        config.eagle_config["eagle_aux_hidden_state_layer_ids"].append(4)
        self.assertEqual(payload, before)

    def test_missing_vocab_uses_config_class_default(self):
        payload = {"architectures": ["LlamaForCausalLMEagle3"]}
        config = AutoDraftConfig.from_dict(payload)
        self.assertEqual(config.draft_vocab_size, LlamaConfig().vocab_size)

    def test_invalid_vocab_and_architectures_fail_before_construction(self):
        for value in (False, 0, -1, 1.5, "128"):
            with (
                self.subTest(vocab=value),
                self.assertRaisesRegex(ValueError, "positive integer"),
            ):
                AutoDraftConfig.from_dict({**self.payload, "draft_vocab_size": value})
        for value in (None, [], "LlamaForCausalLMEagle3", [1], [[]], ["a", "b"]):
            with (
                self.subTest(architectures=value),
                self.assertRaisesRegex(ValueError, "exactly one"),
            ):
                AutoDraftConfig.from_dict({**self.payload, "architectures": value})
        with self.assertRaisesRegex(ValueError, "available"):
            AutoDraftConfig.from_dict({**self.payload, "architectures": ["unknown"]})

    def test_invalid_local_sources_do_not_trigger_hub_requests(self):
        with mock.patch.object(PretrainedConfig, "get_config_dict") as hub:
            for source in (self.path, self.path.parent, self.path.as_uri()):
                with self.subTest(source=source), self.assertRaises(FileNotFoundError):
                    AutoDraftConfig.from_pretrained(source)
            self.path.write_text("{invalid", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid draft config JSON"):
                AutoDraftConfig.from_pretrained(self.path)
            self.write([])
            with self.assertRaisesRegex(ValueError, "JSON object"):
                AutoDraftConfig.from_pretrained(self.path)
            hub.assert_not_called()

    def test_hub_uses_registered_config_without_remote_python(self):
        payload = {
            **self.payload,
            "architectures": ["DFlashDraftModel"],
            "model_type": "unregistered_remote_model",
            "auto_map": {"AutoConfig": "remote.CustomConfig"},
            "_commit_hash": "test-commit",
        }
        with mock.patch.object(
            PretrainedConfig, "get_config_dict", return_value=(payload, {})
        ) as hub:
            config = AutoDraftConfig.from_pretrained(
                "org/draft",
                revision="test-revision",
                local_files_only=True,
                token="test-token",
                subfolder="draft",
                trust_remote_code=True,
            )
        self.assertIsInstance(config, Qwen3Config)
        self.assertEqual(config._name_or_path, "org/draft")
        self.assertEqual(config._commit_hash, "test-commit")
        self.assertEqual(hub.call_args.kwargs["revision"], "test-revision")
        self.assertEqual(hub.call_args.kwargs["subfolder"], "draft")
        self.assertTrue(hub.call_args.kwargs["local_files_only"])

    def test_pretrained_model_passes_source_options_and_normalized_config(self):
        model_cls = DRAFT_REGISTRY["LlamaForCausalLMEagle3"]
        with (
            mock.patch.object(
                PretrainedConfig, "get_config_dict", return_value=(self.payload, {})
            ) as hub,
            mock.patch.object(
                model_cls, "from_pretrained", return_value=object()
            ) as load,
        ):
            result = AutoDraftModel.from_pretrained(
                "org/draft",
                revision="branch",
                subfolder="draft",
                local_files_only=True,
            )
        self.assertIs(result, load.return_value)
        self.assertEqual(hub.call_args.kwargs["revision"], "branch")
        self.assertEqual(hub.call_args.kwargs["subfolder"], "draft")
        config = load.call_args.kwargs["config"]
        self.assertFalse(config.tie_word_embeddings)
        self.assertEqual(config.draft_vocab_size, 128)

    def test_explicit_config_is_preserved_for_pretrained_model(self):
        config = AutoDraftConfig.from_dict(self.payload)
        model_cls = DRAFT_REGISTRY["LlamaForCausalLMEagle3"]
        with (
            mock.patch.object(AutoDraftConfig, "from_pretrained") as parse,
            mock.patch.object(model_cls, "from_pretrained") as load,
        ):
            AutoDraftModel.from_pretrained("org/draft", config=config)
        parse.assert_not_called()
        self.assertIs(load.call_args.kwargs["config"], config)

    def test_save_reload_and_local_subfolder(self):
        config = AutoDraftConfig.from_dict(self.payload)
        nested = self.path.parent / "draft"
        config.save_pretrained(nested)
        restored = AutoDraftConfig.from_pretrained(self.path.parent, subfolder="draft")
        self.assertEqual(restored.to_dict(), config.to_dict())

    def test_pretrained_roundtrip_preserves_weights_for_both_config_families(self):
        for architecture in ("LlamaForCausalLMEagle3", "DFlashDraftModel"):
            with self.subTest(architecture=architecture):
                payload = {
                    **self.payload,
                    "architectures": [architecture],
                    "model_type": (
                        "llama" if architecture.startswith("Llama") else "qwen3"
                    ),
                    "num_target_layers": 4,
                    "block_size": 4,
                    "head_dim": 8,
                }
                model = AutoDraftModel.from_config(AutoDraftConfig.from_dict(payload))
                folder = self.path.parent / architecture
                model.save_pretrained(folder)
                restored = AutoDraftModel.from_pretrained(folder, local_files_only=True)
                self.assertIs(type(restored.config), type(model.config))
                self.assertEqual(restored.config.draft_vocab_size, 128)
                self.assertEqual(
                    restored.state_dict().keys(), model.state_dict().keys()
                )
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(
                        restored.state_dict()[key], value, rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
