"""CPU tests for config-gated EAGLE-4 draft modules."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

import torch
import torch._dynamo
from transformers import LlamaConfig

try:
    from specforge.algorithms.eagle3.model import OnlineEagle3Model
except ImportError:
    OnlineEagle3Model = None

from specforge.modeling._mask_utils import _make_causal_mask
from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.draft.eagle4 import LlamaForCausalLMEagle4, candidate_logits
from specforge.modeling.draft.eagle4_moe import Eagle4MoE
from specforge.modeling.draft.eagle4_structure import (
    EAGLE4_ARCH,
    draft_sliding_window,
    draft_uses_rope,
    parse_efficient_lm_head_config,
)
from specforge.modeling.draft.flex_attention import generate_eagle3_mask
from specforge.modeling.draft.llama3_eagle import (
    LlamaDecoderLayer,
    LlamaForCausalLMEagle3,
    NoRotaryEmbedding,
)

torch._dynamo.config.disable = True


def _config(**overrides) -> LlamaConfig:
    config = LlamaConfig(
        vocab_size=overrides.get("vocab_size", 16),
        hidden_size=overrides.get("hidden_size", 8),
        intermediate_size=overrides.get("intermediate_size", 16),
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        pad_token_id=0,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        initializer_range=0.02,
    )
    config.draft_vocab_size = overrides.get("draft_vocab_size", config.vocab_size)
    config.architectures = [EAGLE4_ARCH]
    for key, value in overrides.items():
        setattr(config, key, value)
    if "architectures" not in overrides:
        config.architectures = [EAGLE4_ARCH]
    return config


def _inputs(batch=2, seq=4, hidden=8):
    generator = torch.Generator().manual_seed(0)
    hidden_states = torch.randn(batch, seq, hidden * 3, generator=generator)
    inputs_embeds = torch.randn(batch, seq, hidden, generator=generator)
    return hidden_states, inputs_embeds


class Eagle4StructureTest(unittest.TestCase):
    def test_sliding_window_requires_eagle4_architecture(self):
        other = LlamaConfig()
        other.sliding_window = 8
        other.architectures = ["LlamaForCausalLMEagle3"]
        self.assertIsNone(draft_sliding_window(other))
        enabled = _config(sliding_window=8)
        self.assertEqual(8, draft_sliding_window(enabled))
        self.assertIsNone(draft_sliding_window(_config()))

    def test_no_rope_flags(self):
        self.assertTrue(draft_uses_rope(_config()))
        self.assertFalse(draft_uses_rope(_config(use_rope=False)))
        self.assertFalse(draft_uses_rope(_config(position_embedding_type="none")))
        self.assertFalse(draft_uses_rope(_config(partial_rotary_factor=0)))

    def test_efficient_head_config_validation(self):
        self.assertIsNone(parse_efficient_lm_head_config(_config()))
        parsed = parse_efficient_lm_head_config(
            _config(
                efficient_lm_head_config={
                    "type": "low_rank",
                    "rank": 4,
                    "num_candidates": 3,
                }
            )
        )
        self.assertEqual(
            {"type": "low_rank", "rank": 4, "num_candidates": 3},
            parsed,
        )
        with self.assertRaisesRegex(ValueError, "low_rank"):
            parse_efficient_lm_head_config(
                _config(efficient_lm_head_config={"type": "full", "rank": 4})
            )


class Eagle4FusionTest(unittest.TestCase):
    def test_eagle3_qkv_width_stays_2h(self):
        config = LlamaConfig(
            vocab_size=16,
            draft_vocab_size=8,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=16,
            pad_token_id=0,
        )
        model = LlamaForCausalLMEagle3(config, attention_backend="sdpa")
        self.assertEqual(16, model.midlayer.self_attn.q_proj.in_features)

    def test_attn_norm_switches_qkv_width_and_eh_proj(self):
        off = LlamaForCausalLMEagle4(_config(), attention_backend="sdpa")
        self.assertEqual(16, off.midlayer.self_attn.q_proj.in_features)
        self.assertIsNone(off.midlayer.eh_proj)

        on = LlamaForCausalLMEagle4(_config(attn_norm=True), attention_backend="sdpa")
        self.assertEqual(8, on.midlayer.self_attn.q_proj.in_features)
        self.assertEqual(16, on.midlayer.eh_proj.in_features)
        self.assertEqual(8, on.midlayer.eh_proj.out_features)
        hidden, embeds = _inputs()
        out = on(hidden_states=hidden, inputs_embeds=embeds, ttt_length=1)
        self.assertEqual((2, 4, 8), tuple(out.shape))

    def test_fc_norm_and_norm_output_flags(self):
        model = LlamaForCausalLMEagle4(
            _config(fc_norm=True, norm_output=True),
            attention_backend="sdpa",
        )
        self.assertEqual(3, len(model.fc_norm))
        raw = torch.randn(1, 3, 8)
        torch.testing.assert_close(model.next_depth_hidden(raw), model.norm(raw))
        logits = model.compute_logits(raw)
        torch.testing.assert_close(logits, model.lm_head(model.norm(raw)))

    def test_no_rope_is_identity_rotation(self):
        model = LlamaForCausalLMEagle4(_config(use_rope=False), attention_backend="sdpa")
        rotary = model.midlayer.self_attn.rotary_emb
        self.assertIsInstance(rotary, NoRotaryEmbedding)
        query = torch.randn(1, 2, 4, 4)
        cos, sin = rotary(query, seq_len=4)
        self.assertTrue(torch.allclose(cos, torch.ones_like(cos)))
        self.assertTrue(torch.allclose(sin, torch.zeros_like(sin)))
        hidden, embeds = _inputs()
        out = model(hidden_states=hidden, inputs_embeds=embeds)
        self.assertTrue(torch.isfinite(out).all())

    def test_sliding_window_mask_hides_keys_behind_the_window(self):
        mask = _make_causal_mask(
            (1, 6),
            torch.float32,
            torch.device("cpu"),
            window=3,
        )
        self.assertEqual(0.0, mask[0, 0, 5, 4].item())
        self.assertEqual(0.0, mask[0, 0, 5, 5].item())
        self.assertLess(mask[0, 0, 5, 1].item(), 0.0)
        self.assertLess(mask[0, 0, 5, 2].item(), 0.0)

        model = LlamaForCausalLMEagle4(
            _config(sliding_window=3), attention_backend="sdpa"
        )
        hidden = torch.ones(1, 6, 8)
        attn = model.prepare_decoder_attention_mask(
            torch.ones(1, 6), hidden, 1, 6, 0
        )
        self.assertLess(attn[0, 0, 5, 1].item(), 0.0)
        self.assertEqual(0.0, attn[0, 0, 5, 4].item())

    def test_flex_mask_windows_context_and_keeps_suffix_diagonal(self):
        mask_mod = generate_eagle3_mask(
            seq_lengths=torch.tensor([8]),
            Q_LEN=8,
            KV_LEN=16,
            lck=1,
            sliding_window=3,
        )
        b = torch.tensor(0)
        h = torch.tensor(0)
        self.assertFalse(bool(mask_mod(b, h, torch.tensor(5), torch.tensor(1))))
        self.assertTrue(bool(mask_mod(b, h, torch.tensor(5), torch.tensor(4))))
        self.assertTrue(bool(mask_mod(b, h, torch.tensor(5), torch.tensor(13))))

    def test_fa_and_usp_reject_a_sliding_window(self):
        config = _config(sliding_window=4)
        for backend in ("fa", "usp"):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, "flex_attention"):
                    LlamaDecoderLayer(config, attention_backend=backend)


class Eagle4MoETest(unittest.TestCase):
    def _moe_config(self, **overrides):
        return _config(
            enable_moe=True,
            n_routed_experts=2,
            num_experts_per_tok=1,
            moe_intermediate_size=8,
            enable_post_mlp_norm=True,
            **overrides,
        )

    def test_moe_replaces_dense_mlp_and_post_mlp_norm_is_next_depth(self):
        model = LlamaForCausalLMEagle4(self._moe_config(), attention_backend="sdpa")
        self.assertIsInstance(model.midlayer.mlp, Eagle4MoE)
        self.assertIsNotNone(model.midlayer.post_mlp_layernorm)
        raw = torch.randn(2, 3, 8)
        torch.testing.assert_close(
            model.next_depth_hidden(raw),
            model.midlayer.post_mlp_layernorm(raw),
        )
        hidden, embeds = _inputs()
        out = model(hidden_states=hidden, inputs_embeds=embeds)
        self.assertEqual((2, 4, 8), tuple(out.shape))
        self.assertTrue(torch.isfinite(out).all())

    def test_router_bias_updates_after_a_training_step(self):
        model = LlamaForCausalLMEagle4(self._moe_config(), attention_backend="sdpa")
        model.train()
        hidden, embeds = _inputs()
        model(hidden_states=hidden, inputs_embeds=embeds)
        router = model.midlayer.mlp.gate
        self.assertGreater(int(router.load.sum().item()), 0)
        router.load.copy_(torch.tensor([8, 0], device=router.load.device))
        before = router.e_score_correction_bias.clone()
        imbalance = model.update_moe_router_bias()
        self.assertGreater(imbalance, 1.0)
        self.assertFalse(torch.equal(before, router.e_score_correction_bias))
        self.assertEqual(0, int(router.load.sum().item()))

    def test_enable_moe_requires_experts(self):
        with self.assertRaisesRegex(ValueError, "n_routed_experts"):
            LlamaForCausalLMEagle4(_config(enable_moe=True), attention_backend="sdpa")


class Eagle4CandidateHeadTest(unittest.TestCase):
    def _head_config(self, **overrides):
        return _config(
            efficient_lm_head_config={
                "type": "low_rank",
                "rank": 4,
                "num_candidates": 3,
            },
            **overrides,
        )

    def test_pruned_vocab_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "full-vocab"):
            LlamaForCausalLMEagle4(
                self._head_config(draft_vocab_size=8, vocab_size=16),
                attention_backend="sdpa",
            )

    def test_candidate_shapes_and_exact_rescore(self):
        model = LlamaForCausalLMEagle4(self._head_config(), attention_backend="sdpa")
        self.assertEqual((4, 8), tuple(model.efficient_lm_head_down_proj.weight.shape))
        self.assertEqual((16, 4), tuple(model.efficient_lm_head.weight.shape))
        self.assertEqual((16, 8), tuple(model.lm_head.weight.shape))
        self.assertFalse(hasattr(model, "t2d"))
        self.assertFalse(hasattr(model, "d2t"))

        hidden = torch.randn(2, 3, 8)
        logits = candidate_logits(
            hidden,
            model.efficient_lm_head_down_proj.weight,
            model.efficient_lm_head.weight,
            model.lm_head.weight,
            3,
        )
        self.assertEqual((2, 3, 16), tuple(logits.shape))
        approx = torch.nn.functional.linear(
            torch.nn.functional.linear(
                hidden, model.efficient_lm_head_down_proj.weight
            ),
            model.efficient_lm_head.weight,
        )
        indices = approx.topk(3, dim=-1).indices
        dense = torch.nn.functional.linear(hidden, model.lm_head.weight)
        torch.testing.assert_close(logits.gather(-1, indices), dense.gather(-1, indices))
        fill = torch.finfo(logits.dtype).min
        blocked = torch.ones_like(logits, dtype=torch.bool).scatter_(-1, indices, False)
        self.assertTrue(torch.all(logits[blocked] == fill))
        self.assertEqual(3, int((~blocked)[0, 0].sum().item()))

    def test_compute_logits_matches_dense_on_chosen_rows(self):
        model = LlamaForCausalLMEagle4(self._head_config(), attention_backend="sdpa")
        raw = torch.randn(1, 2, 8)
        logits = model.compute_logits(raw)
        normed = model.norm(raw)
        dense = model.lm_head(normed)
        approx = torch.nn.functional.linear(
            torch.nn.functional.linear(
                normed, model.efficient_lm_head_down_proj.weight
            ),
            model.efficient_lm_head.weight,
        )
        indices = approx.topk(3, dim=-1).indices
        torch.testing.assert_close(logits.gather(-1, indices), dense.gather(-1, indices))
        fill = torch.finfo(logits.dtype).min
        blocked = torch.ones_like(logits, dtype=torch.bool).scatter_(-1, indices, False)
        self.assertTrue(torch.all(logits[blocked] == fill))

    def test_factors_receive_gradients(self):
        model = LlamaForCausalLMEagle4(self._head_config(), attention_backend="sdpa")
        raw = torch.randn(2, 3, 8, requires_grad=True)
        logits = model.compute_logits(raw)
        logits[torch.isfinite(logits)].sum().backward()
        self.assertIsNotNone(model.efficient_lm_head.weight.grad)
        self.assertIsNotNone(model.efficient_lm_head_down_proj.weight.grad)

    def test_svd_init_and_export_roundtrip(self):
        model = LlamaForCausalLMEagle4(self._head_config(), attention_backend="sdpa")
        model.init_efficient_head_from_dense()
        self.assertEqual((16, 4), tuple(model.efficient_lm_head.weight.shape))
        state = model.state_dict()
        self.assertIn("lm_head.weight", state)
        self.assertIn("efficient_lm_head.weight", state)
        self.assertIn("efficient_lm_head_down_proj.weight", state)
        self.assertNotIn("t2d", state)
        self.assertNotIn("d2t", state)

        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            with open(os.path.join(directory, "config.json"), encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual([EAGLE4_ARCH], payload["architectures"])
            self.assertEqual("low_rank", payload["efficient_lm_head_config"]["type"])
            self.assertEqual(4, payload["efficient_lm_head_config"]["rank"])
            reloaded = AutoDraftModel.from_pretrained(
                directory, local_files_only=True
            )
            self.assertIsInstance(reloaded, LlamaForCausalLMEagle4)
            torch.testing.assert_close(
                reloaded.efficient_lm_head.weight,
                model.efficient_lm_head.weight,
            )
            torch.testing.assert_close(reloaded.lm_head.weight, model.lm_head.weight)

    def test_candidate_recall_is_one_when_target_is_the_argmax(self):
        model = LlamaForCausalLMEagle4(self._head_config(), attention_backend="sdpa")
        with torch.no_grad():
            model.lm_head.weight.zero_()
            model.lm_head.weight[3].fill_(1.0)
            model.efficient_lm_head_down_proj.weight.copy_(
                torch.eye(4, 8)[:4]
            )
            model.efficient_lm_head.weight.zero_()
            model.efficient_lm_head.weight[3, 0] = 10.0
        hidden = torch.zeros(1, 2, 8)
        hidden[..., 0] = 1.0
        target = torch.full((1, 2), 3, dtype=torch.long)
        recall = model.candidate_recall(hidden, target)
        self.assertGreater(float(recall), 0.99)


@unittest.skipUnless(
    OnlineEagle3Model is not None, "OnlineEagle3Model imports require triton"
)
class Eagle4TTTHookTest(unittest.TestCase):
    def test_ttt_uses_next_depth_hidden_and_identity_vocab_without_t2d(self):
        model = LlamaForCausalLMEagle4(
            _config(
                efficient_lm_head_config={
                    "type": "low_rank",
                    "rank": 4,
                    "num_candidates": 4,
                },
                norm_output=True,
            ),
            attention_backend="sdpa",
        ).eval()
        self.assertFalse(hasattr(model, "t2d"))
        wrapper = OnlineEagle3Model(model, length=1, attention_backend="sdpa")
        batch, seq, hidden, vocab = 1, 4, 8, 16
        with mock.patch.object(
            model, "next_depth_hidden", wraps=model.next_depth_hidden
        ) as next_depth:
            plosses, *_ = wrapper(
                input_ids=torch.ones(batch, seq, dtype=torch.long),
                attention_mask=torch.ones(batch, seq),
                target=torch.randn(batch, seq, vocab),
                loss_mask=torch.ones(batch, seq, 1),
                hidden_states=torch.randn(batch, seq, hidden * 3),
            )
        self.assertTrue(next_depth.called)
        self.assertEqual(1, len(plosses))
        self.assertTrue(torch.isfinite(plosses[0]))
        self.assertIsNotNone(wrapper.last_candidate_recall)


if __name__ == "__main__":
    unittest.main()
