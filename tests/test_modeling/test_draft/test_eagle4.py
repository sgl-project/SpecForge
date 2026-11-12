"""CPU tests for config-gated EAGLE-4 draft modules."""

from __future__ import annotations

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
from specforge.modeling.draft.eagle4 import LlamaForCausalLMEagle4
from specforge.modeling.draft.eagle4_moe import Eagle4MoE, relu2, resolve_act_fn
from specforge.modeling.draft.eagle4_structure import (
    EAGLE4_ARCH,
    draft_sliding_window,
    draft_uses_rope,
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

    def test_attention_residual_preserves_unnormalized_fusion(self):
        for attn_norm in (False, True):
            with self.subTest(attn_norm=attn_norm):
                torch.manual_seed(42)
                layer = LlamaForCausalLMEagle4(
                    _config(attn_norm=attn_norm), attention_backend="sdpa"
                ).midlayer
                hidden = torch.randn(1, 3, 8, requires_grad=True)
                embeds = torch.randn(1, 3, 8, requires_grad=True)
                joined = torch.cat(
                    (layer.input_layernorm(embeds), layer.hidden_norm(hidden)),
                    dim=-1,
                )
                residual = layer.eh_proj(joined) if attn_norm else hidden
                attention_input = (
                    layer.pre_attention_layernorm(residual) if attn_norm else joined
                )
                if attn_norm:
                    self.assertFalse(torch.allclose(residual, attention_input))
                # Zero attention and MLP isolate the skip connection, while
                # recording the input verifies normalization still feeds QKV.
                with (
                    mock.patch.object(
                        layer.self_attn,
                        "forward",
                        return_value=torch.zeros_like(hidden),
                    ) as attention,
                    mock.patch.object(
                        layer.mlp, "forward", side_effect=torch.zeros_like
                    ),
                ):
                    output = layer(input_emb=embeds, hidden_states=hidden)
                torch.testing.assert_close(output, residual)
                torch.testing.assert_close(
                    attention.call_args.kwargs["hidden_states"], attention_input
                )
                actual_grad = torch.autograd.grad(
                    output.sum(), hidden, retain_graph=True
                )[0]
                expected_grad = torch.autograd.grad(residual.sum(), hidden)[0]
                torch.testing.assert_close(actual_grad, expected_grad)

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
        model = LlamaForCausalLMEagle4(
            _config(use_rope=False), attention_backend="sdpa"
        )
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

        cached = _make_causal_mask(
            (1, 1),
            torch.float32,
            torch.device("cpu"),
            past_key_values_length=5,
            window=3,
        )
        torch.testing.assert_close(
            cached[0, 0, 0] == 0,
            torch.tensor([False, False, False, True, True, True]),
        )

        model = LlamaForCausalLMEagle4(
            _config(sliding_window=3), attention_backend="sdpa"
        )
        hidden = torch.ones(1, 6, 8)
        attn = model.prepare_decoder_attention_mask(torch.ones(1, 6), hidden, 1, 6, 0)
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

    def test_relu2_resolves_when_act2fn_omits_it(self):
        missing = mock.MagicMock()
        missing.__getitem__.side_effect = KeyError("relu2")
        with mock.patch("specforge.modeling.draft.eagle4_moe.ACT2FN", missing):
            self.assertIs(resolve_act_fn("relu2"), relu2)
        x = torch.tensor([-1.0, 2.0])
        torch.testing.assert_close(relu2(x), torch.tensor([0.0, 4.0]))

    def test_latent_relu2_moe_constructs_and_runs(self):
        model = LlamaForCausalLMEagle4(
            self._moe_config(
                moe_architecture="nemotron_latent_relu2",
                hidden_act="relu2",
                moe_latent_size=4,
                moe_shared_expert_intermediate_size=8,
                n_shared_experts=1,
            ),
            attention_backend="sdpa",
        )
        mlp = model.midlayer.mlp
        self.assertTrue(mlp.latent)
        self.assertIsNotNone(mlp.shared_experts)
        hidden, embeds = _inputs()
        out = model(hidden_states=hidden, inputs_embeds=embeds)
        self.assertEqual((2, 4, 8), tuple(out.shape))
        self.assertTrue(torch.isfinite(out).all())


class Eagle4DenseHeadTest(unittest.TestCase):
    def test_efficient_head_config_is_export_metadata_only(self):
        model = LlamaForCausalLMEagle4(
            _config(
                efficient_lm_head_config={
                    "type": "low_rank",
                    "rank": 4,
                    "num_candidates": 3,
                }
            ),
            attention_backend="sdpa",
        )
        self.assertFalse(any("efficient_lm_head" in k for k in model.state_dict()))
        hidden = torch.randn(1, 3, 8)
        logits = model.compute_logits(hidden)
        torch.testing.assert_close(logits, model.lm_head(model.norm(hidden)))
        logits.sum().backward()
        self.assertIsNotNone(model.lm_head.weight.grad)
        with self.assertRaisesRegex(ValueError, "draft_vocab_size"):
            LlamaForCausalLMEagle4(_config(draft_vocab_size=8))

    def test_dense_export_roundtrip_preserves_identity_mapping(self):
        model = LlamaForCausalLMEagle4(_config(), attention_backend="sdpa")
        hidden = torch.randn(1, 3, 8)
        torch.testing.assert_close(
            model.compute_logits(hidden), model.lm_head(model.norm(hidden))
        )
        self.assertTrue(bool(model.t2d.all()))
        self.assertTrue(bool((model.d2t == 0).all()))
        self.assertFalse(any("efficient_lm_head" in k for k in model.state_dict()))
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            reloaded = AutoDraftModel.from_pretrained(directory, local_files_only=True)
            torch.testing.assert_close(
                model.compute_logits(hidden), reloaded.compute_logits(hidden)
            )

    def test_forward_returns_raw_hidden_for_single_logit_normalization(self):
        model = LlamaForCausalLMEagle4(
            _config(norm_output=True), attention_backend="sdpa"
        )
        hidden, embeds = _inputs()
        raw = torch.randn(2, 4, 8)
        with mock.patch.object(model, "backbone", return_value=raw):
            output = model(hidden_states=hidden, inputs_embeds=embeds)
        torch.testing.assert_close(output, raw)
        torch.testing.assert_close(
            model.compute_logits(output), model.lm_head(model.norm(raw))
        )
        torch.testing.assert_close(model.next_depth_hidden(output), model.norm(raw))


@unittest.skipUnless(
    OnlineEagle3Model is not None, "OnlineEagle3Model imports require triton"
)
class Eagle4TTTHookTest(unittest.TestCase):
    def test_dense_ttt_uses_next_depth_hidden_and_backpropagates(self):
        model = LlamaForCausalLMEagle4(
            _config(attn_norm=True, norm_output=True), attention_backend="sdpa"
        )
        wrapper = OnlineEagle3Model(model, length=2, attention_backend="sdpa")

        def reference_loss(logits, target_p, mask, target_on_draft=None):
            # Exercise the TTT plumbing on CPU; the fused kernel has separate tests.
            loss = -(mask * target_p * logits.float().log_softmax(-1)).sum(-1).mean()
            if target_on_draft is None:
                return loss
            rows = torch.minimum(logits.float().softmax(-1), target_on_draft).sum(-1)
            return loss, rows

        with (
            mock.patch.object(
                model, "next_depth_hidden", wraps=model.next_depth_hidden
            ) as next_depth,
            mock.patch(
                "specforge.algorithms.eagle3.model.LogSoftmaxLoss.apply",
                side_effect=reference_loss,
            ),
        ):
            plosses, *_ = wrapper(
                input_ids=torch.ones(1, 4, dtype=torch.long),
                attention_mask=torch.ones(1, 4),
                target=torch.randn(1, 4, 16),
                loss_mask=torch.ones(1, 4, 1),
                hidden_states=torch.randn(1, 4, 24),
            )
            loss = sum(plosses)
            loss.backward()
        self.assertEqual(next_depth.call_count, 2)
        self.assertEqual(len(plosses), 2)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.lm_head.weight.grad).all())
        self.assertTrue(torch.isfinite(model.midlayer.eh_proj.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
