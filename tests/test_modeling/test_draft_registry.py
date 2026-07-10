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
    DominoDraftModel,
    DSparkDraftModel,
    LlamaForCausalLMEagle3,
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
    "block_size": 4,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 1,
    "num_target_layers": 4,
    "head_dim": 16,
    "max_position_embeddings": 512,
    "vocab_size": 256,
}

TINY_DSPARK = {
    **TINY_DFLASH,
    "architectures": ["DSparkDraftModel"],
    "layer_types": ["full_attention"],
    "dflash_config": {
        "projector_type": "dspark",
        "markov_rank": 8,
        "markov_head_type": "vanilla",
        "confidence_head_alpha": 1.0,
        "confidence_head_with_markov": True,
    },
}

TINY_LEGACY_DSPARK = {
    **TINY_DSPARK,
    "architectures": ["DFlashDraftModel"],
}

TINY_DOMINO = {
    **TINY_DFLASH,
    "architectures": ["DominoDraftModel"],
    "layer_types": ["full_attention"],
    "dflash_config": {
        "projector_type": "domino",
        "emb_dim": 16,
        "gru_hidden_dim": 16,
        "pure_draft_prefix_len": 0,
        "shift_label": False,
    },
}

TINY_LEGACY_DOMINO = {
    **TINY_DOMINO,
    "architectures": ["DFlashDraftModel"],
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
        self.assertIn("DominoDraftModel", available_drafts())
        self.assertIn("DSparkDraftModel", available_drafts())
        self.assertIs(resolve_draft("LlamaForCausalLMEagle3"), LlamaForCausalLMEagle3)
        self.assertIs(resolve_draft("DFlashDraftModel"), DFlashDraftModel)
        self.assertIs(resolve_draft("DominoDraftModel"), DominoDraftModel)
        self.assertIs(resolve_draft("DSparkDraftModel"), DSparkDraftModel)

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

    def test_from_config_builds_dspark_as_explicit_draft(self):
        path = _write(TINY_DSPARK)
        self.addCleanup(os.unlink, path)
        config = AutoDraftModelConfig.from_file(path)
        model = AutoEagle3DraftModel.from_config(config)
        self.assertIsInstance(model, DSparkDraftModel)
        self.assertIsInstance(model, DFlashDraftModel)
        self.assertEqual(model.projector_type, "dspark")
        self.assertIsNotNone(model.markov_head)
        self.assertIsNotNone(model.confidence_head)

    def test_from_config_maps_legacy_dspark_projector_to_explicit_draft(self):
        path = _write(TINY_LEGACY_DSPARK)
        self.addCleanup(os.unlink, path)
        config = AutoDraftModelConfig.from_file(path)
        model = AutoEagle3DraftModel.from_config(config)
        self.assertIsInstance(model, DSparkDraftModel)
        self.assertIsInstance(model, DFlashDraftModel)
        self.assertEqual(model.projector_type, "dspark")
        self.assertIsNotNone(model.markov_head)

    def test_from_config_maps_legacy_domino_projector_to_explicit_draft(self):
        path = _write(TINY_LEGACY_DOMINO)
        self.addCleanup(os.unlink, path)
        config = AutoDraftModelConfig.from_file(path)
        model = AutoEagle3DraftModel.from_config(config)
        self.assertIsInstance(model, DominoDraftModel)
        self.assertIsInstance(model, DFlashDraftModel)
        self.assertEqual(model.projector_type, "domino")
        self.assertIsNotNone(model.prefix_gru)

    def test_from_config_builds_domino_as_explicit_draft(self):
        path = _write(TINY_DOMINO)
        self.addCleanup(os.unlink, path)
        config = AutoDraftModelConfig.from_file(path)
        model = AutoEagle3DraftModel.from_config(config)
        self.assertIsInstance(model, DominoDraftModel)
        self.assertIsInstance(model, DFlashDraftModel)
        self.assertEqual(model.projector_type, "domino")
        self.assertIsNotNone(model.prefix_gru)
        self.assertIsNotNone(model.embed_proj)

    def test_domino_uses_direct_state_dict_keys(self):
        path = _write(TINY_DOMINO)
        self.addCleanup(os.unlink, path)
        config = AutoDraftModelConfig.from_file(path)
        model = AutoEagle3DraftModel.from_config(config)
        keys = set(model.state_dict())
        self.assertTrue(any(key.startswith("prefix_gru.") for key in keys))
        self.assertTrue(any(key.startswith("embed_proj.") for key in keys))
        self.assertFalse(any(key.startswith("logit_head.") for key in keys))

    def test_dspark_uses_direct_state_dict_keys(self):
        path = _write(TINY_DSPARK)
        self.addCleanup(os.unlink, path)
        config = AutoDraftModelConfig.from_file(path)
        model = AutoEagle3DraftModel.from_config(config)
        keys = set(model.state_dict())
        self.assertTrue(any(key.startswith("markov_head.") for key in keys))
        self.assertTrue(any(key.startswith("confidence_head.") for key in keys))
        self.assertFalse(any(key.startswith("logit_head.") for key in keys))

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
