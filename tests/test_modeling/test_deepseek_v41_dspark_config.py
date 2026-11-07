"""configs/deepseek-v4.1-flash-moe-dspark.json: DeepSeek-V4.1's own DSpark draft."""

import json
import unittest
from pathlib import Path

import torch
from torch import nn
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.modeling.draft.layers.attention.deepseek_v41 import DeepseekV41Attention
from specforge.modeling.draft.layers.ffn.moe import MoELayer

CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "deepseek-v4.1-flash-moe-dspark.json"
)
# DeepSeek-V4.1-Flash: hidden, q rank, head dim, o groups, o rank, the draft's routed
# experts, expert width, vocabulary, Markov rank and hyper-connection streams.
H, R, D, G, O, E, I, V, M, HC = 5120, 1280, 512, 8, 1024, 128, 2304, 129280, 256, 4
HEADS, LAYERS, TARGETS = 64, 3, 3


def _config(**overrides):
    payload = json.loads(CONFIG.read_text())
    payload["dflash_config"].update(overrides.pop("dflash_config", {}))
    payload.update(overrides)
    config = Qwen3Config(**payload)
    config._attn_implementation = "eager"
    return config


def _expected_shapes():
    """The official mtp.* tensors under SpecForge names (official names in comments)."""
    shapes = {
        "fc.weight": (H, TARGETS * H),  # main_proj
        "hidden_norm.weight": (H,),  # main_norm
        "norm.weight": (H,),
        "markov_head.markov_w1.weight": (V, M),  # markov_head.embed
        "markov_head.markov_w2.weight": (V, M),  # markov_head.head
        "confidence_head.proj.weight": (1, H + M),
    }
    for layer in range(LAYERS):
        prefix = f"layers.{layer}."
        shapes.update(
            {
                prefix + "self_attn.q_a_proj.weight": (R, H),  # wq_a
                prefix + "self_attn.q_a_norm.weight": (R,),  # q_norm
                prefix + "self_attn.q_b_proj.weight": (HEADS * D, R),  # wq_b
                prefix + "self_attn.kv_proj.weight": (D, H),  # wkv
                prefix + "self_attn.kv_norm.weight": (D,),
                prefix + "self_attn.o_a_proj.weight": (G * O, HEADS * D // G),  # wo_a
                prefix + "self_attn.o_b_proj.weight": (H, G * O),  # wo_b
                prefix + "self_attn.sinks": (HEADS,),  # attn_sink
                prefix + "mlp.gate.weight": (E, H),
                prefix + "mlp.gate.expert_bias": (E,),  # gate.bias (text tokens)
                prefix + "mlp.experts.gate_up_proj": (E, 2 * I, H),  # experts.*.w1, w3
                prefix + "mlp.experts.down_proj": (E, H, I),  # experts.*.w2
                prefix + "mlp.shared_expert.gate_proj.weight": (I, H),
                prefix + "mlp.shared_expert.up_proj.weight": (I, H),
                prefix + "mlp.shared_expert.down_proj.weight": (H, I),
                prefix + "input_layernorm.weight": (H,),  # attn_norm
                prefix + "post_attention_layernorm.weight": (H,),  # ffn_norm
            }
        )
        for residual in ("attention_residual", "ffn_residual"):  # hc_attn_*, hc_ffn_*
            shapes.update(
                {
                    f"{prefix}{residual}.mix_weight": ((2 + HC) * HC, HC * H),
                    f"{prefix}{residual}.mix_scale": (3,),
                    f"{prefix}{residual}.mix_bias": ((2 + HC) * HC,),
                }
            )
    return shapes


class DeepseekV41DSparkConfigTest(unittest.TestCase):
    def test_full_size_state_dict_matches_the_official_draft(self):
        with torch.device("meta"):
            model = DSparkDraftModel(_config())
        self.assertEqual(model.attention_mode, "deepseek_v41")
        self.assertEqual(model.context_window, 128)
        self.assertEqual(model.block_size, 5)
        self.assertEqual(model.target_layer_ids, [37, 38, 39])
        self.assertEqual(model.residual_options.kind, "mhc")
        self.assertIsInstance(model.layers[0].self_attn, DeepseekV41Attention)
        self.assertIsInstance(model.layers[0].mlp, MoELayer)
        self.assertIsNone(model.confidence_head.proj.bias)
        shapes = {name: tuple(t.shape) for name, t in model.state_dict().items()}
        self.assertEqual(shapes, _expected_shapes())
        # Every element of the official mtp.* tensors except the three image-token
        # routing biases (gate.bias_vl), which a text draft does not carry: three
        # stages of attention, MoE, hyper-connections and norms, main_proj and
        # main_norm on the first stage, the single-pass mHC head (no hc_head
        # parameters) on the last.
        attention = H * R + R + HEADS * D * R + D * H + D + HEADS * D * O + H * G * O
        moe = E * 3 * I * H + 3 * I * H + E * H + E
        hc = 2 * ((2 + HC) * HC * HC * H + (2 + HC) * HC + 3)
        stage = attention + HEADS + moe + hc + 2 * H
        head_stage = H + 2 * V * M + (H + M)
        expected = LAYERS * stage + (TARGETS * H * H + H) + head_stage
        self.assertEqual(sum(t.numel() for t in model.state_dict().values()), expected)

    def test_small_scale_dspark_objective_trains_every_parameter(self):
        config = _config(
            hidden_size=32,
            num_attention_heads=4,
            head_dim=16,
            qk_rope_head_dim=8,
            q_lora_rank=16,
            o_groups=2,
            o_lora_rank=8,
            n_routed_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=16,
            intermediate_size=16,
            vocab_size=64,
            num_target_layers=4,
            max_position_embeddings=64,
            dflash_config={
                "target_layer_ids": [1],
                "markov_rank": 4,
                "mask_token_id": 63,
                "context_window": 3,
            },
        )
        draft = DSparkDraftModel(config)
        model = OnlineDSparkModel(
            draft,
            nn.Linear(32, 64, bias=False).requires_grad_(False),
            nn.Embedding(64, 32).requires_grad_(False),
            mask_token_id=63,
            block_size=5,
            attention_backend="eager",
            num_anchors=2,
            dspark_confidence_head_alpha=1.0,
            objective_chunk_blocks=1,
        )
        loss, _, _ = model(
            input_ids=torch.randint(0, 60, (2, 16)),
            hidden_states=torch.randn(2, 16, 32),
            loss_mask=torch.ones(2, 16),
            target_last_hidden_states=torch.randn(2, 16, 32),
        )
        loss.backward()
        for name, parameter in draft.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
