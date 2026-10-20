# coding=utf-8
"""build_ffn keeps a draft layer's dense MLP unless the config names a moe_preset."""

import unittest

from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from specforge.modeling.draft import DFlashDraftModel
from specforge.modeling.draft.moe import build_ffn

TINY_DFLASH = {
    "architectures": ["DFlashDraftModel"],
    "model_type": "qwen3",
    "block_size": 4,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "num_target_layers": 4,
    "head_dim": 16,
    "max_position_embeddings": 512,
    "vocab_size": 256,
}


class BuildFfnTest(unittest.TestCase):
    def test_without_moe_preset_returns_the_dense_mlp(self):
        config = Qwen3Config(**TINY_DFLASH)
        built = []

        def dense(dense_config):
            built.append(dense_config)
            return Qwen3MLP(dense_config)

        self.assertIsInstance(build_ffn(config, dense=dense), Qwen3MLP)
        self.assertEqual(built, [config])

    def test_moe_fields_without_a_preset_stay_dense(self):
        # configs/qwen-8b-moe.json carries Qwen3-MoE fields but no moe_preset.
        config = Qwen3Config(**TINY_DFLASH, num_experts=8, num_experts_per_tok=1)
        self.assertIsInstance(build_ffn(config, dense=Qwen3MLP), Qwen3MLP)

    def test_moe_preset_is_not_implemented_yet(self):
        config = Qwen3Config(**TINY_DFLASH, moe_preset="qwen3_moe")
        with self.assertRaisesRegex(NotImplementedError, "qwen3_moe"):
            build_ffn(config, dense=Qwen3MLP)

    def test_dflash_layers_build_their_mlp_through_build_ffn(self):
        model = DFlashDraftModel(Qwen3Config(**TINY_DFLASH))
        self.assertTrue(all(isinstance(layer.mlp, Qwen3MLP) for layer in model.layers))


if __name__ == "__main__":
    unittest.main(verbosity=2)
