"""HF metadata round trips without copying or overriding draft structure."""

import copy
import json
import unittest
from tempfile import TemporaryDirectory

import torch
from transformers import PreTrainedConfig

from specforge.config.draft.codec import encode_config
from specforge.modeling.draft.hf_config import SpecForgeDraftConfig
from tests.test_config.draft_fixtures import draft_config


class DraftHFConfigTest(unittest.TestCase):
    def test_one_structural_source_and_read_only_hf_views(self):
        draft = draft_config()
        config = SpecForgeDraftConfig(draft)
        self.assertIsInstance(config, PreTrainedConfig)
        self.assertIs(config.draft, draft)
        for name in (
            "hidden_size",
            "head_dim",
            "layer_types",
            "spec_config",
            "initializer_range",
        ):
            self.assertFalse(hasattr(config, name))
            with self.assertRaises(AttributeError):
                setattr(config, name, 1)
        for name in ("draft", "_draft"):
            with self.assertRaises(AttributeError):
                setattr(config, name, draft_config(hidden_size=64))
        config.architectures.append("OtherModel")
        self.assertEqual(config.architectures, [draft.architecture])
        config.architectures = [draft.architecture]
        config.tie_word_embeddings = draft.tie_word_embeddings
        for name, value in (
            ("architectures", ["OtherModel"]),
            ("tie_word_embeddings", True),
        ):
            with self.assertRaises(ValueError):
                setattr(config, name, value)
        self.assertEqual(hash(config.draft), hash(draft))

    def test_full_canonical_serialization_keeps_export_metadata(self):
        draft = draft_config()
        metadata = {
            "auto_map": {"AutoModel": "model.DFlashDraftModel"},
            "dtype": "bfloat16",
        }
        config = SpecForgeDraftConfig(draft, **metadata)
        metadata["auto_map"].clear()
        saved = json.loads(config.to_json_string())
        for name, value in encode_config(draft).items():
            self.assertEqual(saved[name], value)
        self.assertEqual(saved["auto_map"], {"AutoModel": "model.DFlashDraftModel"})
        self.assertNotIn("_draft", saved)
        restored = SpecForgeDraftConfig.from_dict(saved)
        self.assertEqual(restored.draft, draft)
        self.assertEqual(restored.dtype, torch.bfloat16)
        self.assertEqual(restored.to_dict(), config.to_dict())
        self.assertEqual(copy.deepcopy(config).draft, draft)

    def test_runtime_overrides_and_unused_model_arguments(self):
        payload = encode_config(draft_config())
        before = copy.deepcopy(payload)
        config, unused = SpecForgeDraftConfig.from_dict(
            payload,
            return_unused_kwargs=True,
            attn_implementation="eager",
            dtype="float32",
            device_map="cpu",
            _from_auto=True,
        )
        self.assertEqual(config._attn_implementation, "eager")
        self.assertEqual(config.dtype, torch.float32)
        self.assertEqual(unused, {"device_map": "cpu"})
        self.assertEqual(payload, before)
        for name in (
            "hidden_size",
            "head_dim",
            "layer_types",
            "architectures",
            "tie_word_embeddings",
        ):
            with self.assertRaisesRegex(ValueError, "override draft structure"):
                SpecForgeDraftConfig.from_dict(payload, **{name: 1})

    def test_hf_save_and_reload_keep_structure_and_metadata(self):
        config = SpecForgeDraftConfig(
            draft_config(),
            output_attentions=True,
            auto_map={"AutoModel": "model.DFlashDraftModel"},
        )
        with TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            restored = SpecForgeDraftConfig.from_pretrained(directory)
        self.assertEqual(restored.draft, config.draft)
        self.assertEqual(restored.auto_map, config.auto_map)
        self.assertTrue(restored.output_attentions)
        with self.assertRaises(AttributeError):
            restored.update({"hidden_size": 1})

    def test_missing_unknown_and_misspelled_structure_still_fails(self):
        payload = encode_config(draft_config())
        for name in ("hidden_size", "norm_config", "spec_config"):
            incomplete = dict(payload)
            incomplete.pop(name)
            with self.assertRaises(ValueError):
                SpecForgeDraftConfig.from_dict(incomplete)
        with self.assertRaisesRegex(ValueError, "hidden_szie"):
            SpecForgeDraftConfig.from_dict(payload | {"hidden_szie": 32})
        with self.assertRaisesRegex(ValueError, "metadata"):
            SpecForgeDraftConfig(draft_config(), hidden_size=32)
        with self.assertRaises(TypeError):
            SpecForgeDraftConfig()


if __name__ == "__main__":
    unittest.main()
