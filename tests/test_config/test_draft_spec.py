"""Draft settings resolve without constructing a model or loading weights."""

import pickle
import unittest
from dataclasses import FrozenInstanceError

from specforge.config.draft import AutoDraftConfig


class DraftSpecTest(unittest.TestCase):
    def setUp(self):
        self.data = {
            "architectures": ["LlamaForCausalLMEagle3"],
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
        }

    def test_defaults_freezing_and_adapter_isolation(self):
        self.data.update(draft_vocab_size=None, extension={"indices": [1, 2]})
        config = AutoDraftConfig.from_inline(self.data)
        self.assertEqual(config.draft_vocab_size, 64)
        self.data["extension"]["indices"].append(3)
        self.assertEqual(config.extension["indices"], (1, 2))
        with self.assertRaises(FrozenInstanceError):
            config.hidden_size = 32
        with self.assertRaises(TypeError):
            config.extension["indices"] = ()
        adapter = config.to_hf_config()
        adapter.hidden_size = 32
        self.assertEqual(config.hidden_size, 16)
        self.assertEqual(pickle.loads(pickle.dumps(config)).to_dict(), config.to_dict())

    def test_invalid_dimensions_and_scalars(self):
        for field, value in (
            ("draft_vocab_size", False),
            ("head_dim", 3),
            ("num_hidden_layers", 2),
            ("fc_norm", "false"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                AutoDraftConfig.from_inline({**self.data, field: value})
