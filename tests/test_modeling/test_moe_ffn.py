# coding=utf-8
"""build_ffn: the dense MLP by default, the moe_preset family's MoELayer otherwise."""

import tempfile
import unittest

import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from specforge.modeling.draft import DFlashDraftModel
from specforge.modeling.draft.moe import MoELayer, build_ffn

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
TINY_MOE = {
    "moe_preset": "qwen3_5_moe",
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 48,
    "norm_topk_prob": False,
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
        config = Qwen3Config(**TINY_DFLASH, num_experts=8, num_experts_per_tok=1)
        self.assertIsInstance(build_ffn(config, dense=Qwen3MLP), Qwen3MLP)

    def test_moe_preset_builds_the_family_moe_layer(self):
        config = Qwen3Config(**TINY_DFLASH, **TINY_MOE)
        self.assertIsInstance(build_ffn(config, dense=Qwen3MLP), MoELayer)

    def test_unknown_moe_preset_raises(self):
        config = Qwen3Config(**TINY_DFLASH, moe_preset="qwen3_moe")
        with self.assertRaisesRegex(ValueError, "qwen3_moe"):
            build_ffn(config, dense=Qwen3MLP)

    def test_dflash_layers_build_their_mlp_through_build_ffn(self):
        dense = DFlashDraftModel(Qwen3Config(**TINY_DFLASH))
        self.assertTrue(all(isinstance(layer.mlp, Qwen3MLP) for layer in dense.layers))
        moe = DFlashDraftModel(Qwen3Config(**TINY_DFLASH, **TINY_MOE))
        self.assertTrue(all(isinstance(layer.mlp, MoELayer) for layer in moe.layers))

    def test_from_pretrained_inits_missing_moe_weights_and_keeps_loaded_ones(self):
        model = DFlashDraftModel(Qwen3Config(**TINY_DFLASH, **TINY_MOE))
        state = model.state_dict()
        missing = {"layers.0.mlp.experts.gate_up_proj", "layers.1.mlp.gate.weight"}
        kept = "layers.0.mlp.experts.down_proj"
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(
                tmp, state_dict={k: v for k, v in state.items() if k not in missing}
            )
            loaded = DFlashDraftModel.from_pretrained(tmp).state_dict()
        for key in missing:
            self.assertTrue(torch.isfinite(loaded[key]).all(), key)
            self.assertTrue(0 < loaded[key].std() < 0.1, key)
        torch.testing.assert_close(loaded[kept], state[kept])


if __name__ == "__main__":
    unittest.main(verbosity=2)
