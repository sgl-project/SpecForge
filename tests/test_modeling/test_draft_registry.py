# coding=utf-8
"""E gate: draft-architecture registry drives the auto loaders.

Adding a draft architecture = a class + @register_draft; both auto loaders
(config-from-file and model-from-config) resolve it from the registry with the
legacy hardcoded mappings kept only as fallback.
"""

import json
import os
import tempfile
import unittest

from transformers import LlamaConfig, Qwen3Config

from specforge.modeling.auto import AutoDraftModelConfig, AutoEagle3DraftModel
from specforge.modeling.draft import (
    DRAFT_REGISTRY,
    DFlashDraftModel,
    LlamaForCausalLMEagle3,
    PEagleDraftModel,
    available_drafts,
    register_draft,
    resolve_draft,
)

TINY_EAGLE3 = {
    "architectures": ["LlamaForCausalLMEagle3"],
    "model_type": "llama",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 1,
    "max_position_embeddings": 512,
    "vocab_size": 256,
    "draft_vocab_size": 64,
    "tie_word_embeddings": False,
}

TINY_DFLASH = {
    "architectures": ["DFlashDraftModel"],
    "model_type": "qwen3",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 1,
    "head_dim": 16,
    "max_position_embeddings": 512,
    "vocab_size": 256,
}

TINY_PEAGLE = {
    **TINY_EAGLE3,
    "architectures": ["PEagleDraftModel"],
}


def _write(cfg: dict) -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(cfg, f)
    return path


class DraftRegistryTest(unittest.TestCase):
    def test_builtin_architectures_registered(self):
        self.assertIn("LlamaForCausalLMEagle3", available_drafts())
        self.assertIn("DFlashDraftModel", available_drafts())
        self.assertIn("PEagleDraftModel", available_drafts())
        self.assertIs(resolve_draft("LlamaForCausalLMEagle3"), LlamaForCausalLMEagle3)
        self.assertIs(resolve_draft("DFlashDraftModel"), DFlashDraftModel)
        self.assertIs(resolve_draft("PEagleDraftModel"), PEagleDraftModel)

    def test_unknown_architecture_raises_with_available_list(self):
        with self.assertRaises(KeyError) as ctx:
            resolve_draft("NoSuchDraft")
        self.assertIn("LlamaForCausalLMEagle3", str(ctx.exception))

    def test_register_is_idempotent_but_rejects_conflicts(self):
        cls = resolve_draft("LlamaForCausalLMEagle3")
        self.assertIs(register_draft(cls), cls)  # same class again: fine

        class Impostor:
            config_class = LlamaConfig

        with self.assertRaises(ValueError):
            register_draft(Impostor, name="LlamaForCausalLMEagle3")

    def test_register_requires_config_class_and_honors_name(self):
        class NoConfig:
            pass

        with self.assertRaises(TypeError):
            register_draft(NoConfig)

        @register_draft(name="tiny_fake_draft")
        class FakeDraft:
            config_class = LlamaConfig

        self.addCleanup(DRAFT_REGISTRY.pop, "tiny_fake_draft", None)
        self.assertIs(resolve_draft("tiny_fake_draft"), FakeDraft)


class AutoLoaderRegistryTest(unittest.TestCase):
    def test_from_file_resolves_eagle3_via_registry(self):
        path = _write(TINY_EAGLE3)
        self.addCleanup(os.unlink, path)
        config = AutoDraftModelConfig.from_file(path)
        self.assertIsInstance(config, LlamaConfig)
        self.assertEqual(config.draft_vocab_size, 64)

    def test_from_file_resolves_dflash_via_registry(self):
        # Not in the legacy _config_mapping: only the registry makes this load.
        path = _write(TINY_DFLASH)
        self.addCleanup(os.unlink, path)
        config = AutoDraftModelConfig.from_file(path)
        self.assertIsInstance(config, Qwen3Config)

    def test_from_file_unknown_architecture_raises(self):
        path = _write({**TINY_EAGLE3, "architectures": ["NoSuchDraft"]})
        self.addCleanup(os.unlink, path)
        with self.assertRaises(ValueError) as ctx:
            AutoDraftModelConfig.from_file(path)
        self.assertIn("available", str(ctx.exception))

    def test_from_config_resolves_class_via_architectures(self):
        path = _write(TINY_EAGLE3)
        self.addCleanup(os.unlink, path)
        config = AutoDraftModelConfig.from_file(path)
        model = AutoEagle3DraftModel.from_config(config)
        self.assertIsInstance(model, LlamaForCausalLMEagle3)

    def test_from_config_resolves_peagle_via_registry(self):
        path = _write(TINY_PEAGLE)
        self.addCleanup(os.unlink, path)
        config = AutoDraftModelConfig.from_file(path)
        model = AutoEagle3DraftModel.from_config(config)
        self.assertIsInstance(model, PEagleDraftModel)

    def test_peagle_direct_save_reload(self):
        path = _write(TINY_PEAGLE)
        self.addCleanup(os.unlink, path)
        config = AutoDraftModelConfig.from_file(path)
        model = PEagleDraftModel(config)
        with tempfile.TemporaryDirectory() as output_dir:
            model.save_pretrained(output_dir)
            reloaded = PEagleDraftModel.from_pretrained(output_dir)
        self.assertIsInstance(reloaded, PEagleDraftModel)

if __name__ == "__main__":
    unittest.main(verbosity=2)
