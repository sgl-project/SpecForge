# coding=utf-8
"""Tests for DSpark (DFlash backbone + Markov / confidence heads + L1 distillation).

Ported/adapted from TorchSpec PR #129 ``tests/test_dspark.py`` to SpecForge's
DFlash structure. Pins the DSpark wiring so future refactors can't silently break
the objective:

1. DSparkConfig / DSparkDraftModel: head construction, base relationship.
2. forward returns the 6-tuple with detached per-component losses.
3. Loss-wiring invariants:
   - internal identity: combined loss == ce_a*ce + l1_a*l1 + cf_a*conf (world 1)
   - all-masked batch -> loss 0
   - gradients reach markov + confidence + backbone; target embedding stays frozen
   - next-token convention: every within-block slot is supervised (B predictions)
4. Markov / confidence head unit math.
5. CE-only path runs without target last_hidden_states.

Runs on CPU with ``attention_backend="sdpa"`` (flex_attention needs CUDA). The
real DSpark modules are loaded directly via importlib + stub parent packages so
the heavy ``specforge`` package ``__init__`` (sglang/TF target stack) is not
imported. Run with ``USE_TF=0`` on macOS to avoid the torch/TF OpenMP clash.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]


def _stub_pkg(name: str, path: Path) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]
    sys.modules[name] = mod
    return mod


def _load(modname: str, relpath: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(modname, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


# Build stub parent packages, then load the real leaf modules in dependency order.
_stub_pkg("specforge", REPO / "specforge")
_stub_pkg("specforge.core", REPO / "specforge" / "core")
_stub_pkg("specforge.modeling", REPO / "specforge" / "modeling")
_stub_pkg("specforge.modeling.draft", REPO / "specforge" / "modeling" / "draft")

_load("specforge.modeling.draft.dflash", "specforge/modeling/draft/dflash.py")
_dspark_draft = _load(
    "specforge.modeling.draft.dspark", "specforge/modeling/draft/dspark.py"
)
_load("specforge.core.dflash", "specforge/core/dflash.py")
_core_dspark = _load("specforge.core.dspark", "specforge/core/dspark.py")

AcceptRatePredictor = _dspark_draft.AcceptRatePredictor
DSparkConfig = _dspark_draft.DSparkConfig
DSparkDraftModel = _dspark_draft.DSparkDraftModel
VanillaMarkov = _dspark_draft.VanillaMarkov
OnlineDSparkModel = _core_dspark.OnlineDSparkModel

import torch.nn as nn  # noqa: E402

CE_A, L1_A, CF_A = 0.1, 0.9, 1.0


def _make_dspark_config(
    H=64,
    V=128,
    num_target_layers=2,
    num_hidden_layers=1,
    markov_rank=16,
    enable_confidence_head=True,
    confidence_head_with_markov=True,
):
    return DSparkConfig(
        hidden_size=H,
        intermediate_size=256,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=V,
        rms_norm_eps=1e-6,
        max_position_embeddings=512,
        rope_theta=10000.0,
        sliding_window=None,
        layer_types=["full_attention"] * num_hidden_layers,
        attn_implementation="sdpa",
        # DFlash-carried fields
        block_size=4,
        num_target_layers=num_target_layers,
        dflash_config={"mask_token_id": V - 1},
        # DSpark fields
        markov_rank=markov_rank,
        markov_head_type="vanilla",
        enable_confidence_head=enable_confidence_head,
        confidence_head_with_markov=confidence_head_with_markov,
    )


def _make_dspark_model(block_size=4, num_anchors=6, H=64, V=128, **cfg_kw):
    config = _make_dspark_config(H=H, V=V, **cfg_kw)
    config.block_size = block_size
    config._attn_implementation = "sdpa"
    draft = DSparkDraftModel(config).to(dtype=torch.float32)
    draft.mask_token_id = V - 1

    target_embed = nn.Embedding(V, H)
    target_lm_head = nn.Linear(H, V, bias=False)
    target_embed.requires_grad_(False)
    target_lm_head.requires_grad_(False)

    return OnlineDSparkModel(
        draft_model=draft,
        target_lm_head=target_lm_head,
        target_embed_tokens=target_embed,
        mask_token_id=V - 1,
        block_size=block_size,
        attention_backend="sdpa",
        num_anchors=num_anchors,
        loss_decay_gamma=4.0,
        ce_loss_alpha=CE_A,
        l1_loss_alpha=L1_A,
        confidence_head_alpha=CF_A,
    ).to(dtype=torch.float32)


def _batch(B=2, S=24, H=64, V=128, all_masked=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, V, (B, S), generator=g)
    # SpecForge fused context feature: one captured layer -> width H.
    hidden_states = torch.randn(B, S, H, generator=g)
    loss_mask = torch.zeros(B, S) if all_masked else torch.ones(B, S)
    if not all_masked:
        loss_mask[:, :2] = 0  # prompt
    last_hidden_states = torch.randn(B, S, H, generator=g)
    return dict(
        input_ids=input_ids,
        hidden_states=hidden_states,
        loss_mask=loss_mask,
        last_hidden_states=last_hidden_states,
    )


class TestDSparkConfig(unittest.TestCase):
    def test_subclasses_qwen3_and_attrs(self):
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

        cfg = _make_dspark_config(markov_rank=32)
        self.assertIsInstance(cfg, Qwen3Config)
        self.assertEqual(cfg.model_type, "dspark")
        self.assertEqual(cfg.markov_rank, 32)
        self.assertTrue(cfg.enable_confidence_head)

    def test_draft_model_heads(self):
        cfg = _make_dspark_config(H=64, markov_rank=16)
        m = DSparkDraftModel(cfg)
        self.assertIsInstance(m.markov_head, VanillaMarkov)
        self.assertIsInstance(m.confidence_head, AcceptRatePredictor)
        # confidence input = hidden + markov_rank when fused
        self.assertEqual(m.confidence_head.proj.in_features, 64 + 16)

    def test_no_heads(self):
        cfg = _make_dspark_config(
            markov_rank=0,
            enable_confidence_head=False,
            confidence_head_with_markov=False,
        )
        m = DSparkDraftModel(cfg)
        self.assertIsNone(m.markov_head)
        self.assertIsNone(m.confidence_head)


class TestDSparkForward(unittest.TestCase):
    def test_returns_six_tuple_with_detached_components(self):
        m = _make_dspark_model()
        out = m(**_batch())
        self.assertEqual(len(out), 6)
        loss, acc, lpp, app, cpp, comps = out
        self.assertEqual(set(comps), {"ce_loss", "l1_loss", "confidence_loss"})
        for v in comps.values():
            self.assertTrue(torch.isfinite(v).all())
            self.assertFalse(v.requires_grad)  # detached for logging
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(lpp.shape[0], m.block_size)

    def test_internal_loss_identity(self):
        # At world_size==1 the combined loss must equal the alpha-weighted sum of
        # the logged components (same denominator) — so the components are a
        # faithful decomposition of what's actually optimized.
        m = _make_dspark_model()
        loss, _, _, _, _, comps = m(**_batch(seed=1))
        recomputed = (
            CE_A * comps["ce_loss"]
            + L1_A * comps["l1_loss"]
            + CF_A * comps["confidence_loss"]
        )
        self.assertTrue(
            torch.allclose(loss, recomputed, atol=1e-4),
            f"{loss.item()} vs {recomputed.item()}",
        )

    def test_all_masked_raises_guard(self):
        # SpecForge's anchor sampler refuses an all-masked sample (the dataloader
        # filters these out upstream via min_loss_tokens = 2*block_size). This
        # differs from TorchSpec, where an all-masked batch yields loss 0. The
        # per-label masking ("masked tokens contribute zero") is still exercised
        # by the prompt mask (loss_mask[:, :2]=0) in the other forward tests.
        m = _make_dspark_model()
        with self.assertRaises(ValueError):
            m(**_batch(all_masked=True))

    def test_next_token_convention_all_slots_supervised(self):
        # Every within-block slot predicts a real token (B predictions), unlike
        # DFlash where slot 0 is the masked anchor. With a long fully supervised
        # sequence, every position should accumulate supervised tokens.
        m = _make_dspark_model(block_size=4, num_anchors=8)
        b = _batch(B=2, S=40)
        b["loss_mask"] = torch.ones(2, 40)
        _, _, _, _, count_per_position, _ = m(**b)
        self.assertEqual(count_per_position.shape[0], 4)
        self.assertTrue(
            (count_per_position > 0).all(),
            f"some slot unsupervised: {count_per_position.tolist()}",
        )

    def test_grad_flow_and_frozen_embedding(self):
        m = _make_dspark_model()
        loss, *_ = m(**_batch(seed=2))
        loss.backward()
        draft = m.draft_model
        self.assertIsNotNone(draft.markov_head.markov_w2.weight.grad)
        self.assertGreater(draft.markov_head.markov_w2.weight.grad.abs().sum().item(), 0)
        self.assertIsNotNone(draft.confidence_head.proj.weight.grad)
        self.assertGreater(
            draft.confidence_head.proj.weight.grad.abs().sum().item(), 0
        )
        # backbone context projection (SpecForge's `fc`) gets gradient
        self.assertIsNotNone(draft.fc.weight.grad)
        self.assertGreater(draft.fc.weight.grad.abs().sum().item(), 0)
        # target embedding is frozen (lives on the wrapper, not the draft)
        self.assertIsNone(m.embed_tokens.weight.grad)

    def test_ce_only_without_target(self):
        # ce-only (l1=0, no confidence) must run without last_hidden_states.
        m = _make_dspark_model(
            markov_rank=16,
            enable_confidence_head=False,
            confidence_head_with_markov=False,
        )
        m.l1_loss_alpha = 0.0
        m.ce_loss_alpha = 1.0
        m.confidence_head_alpha = 0.0
        b = _batch()
        b["last_hidden_states"] = None
        loss, *_ = m(**b)
        self.assertTrue(torch.isfinite(loss))


class TestHeadMath(unittest.TestCase):
    def test_vanilla_markov_is_bigram_bias(self):
        torch.manual_seed(0)
        mk = VanillaMarkov(vocab_size=50, markov_rank=8)
        base = torch.randn(2, 3, 4, 50)
        prev = torch.randint(0, 50, (2, 3, 4))
        out = mk.apply_block_logits(base, token_ids=prev)
        expected = base + mk.markov_w2(mk.markov_w1(prev))
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_confidence_head_is_linear(self):
        torch.manual_seed(0)
        head = AcceptRatePredictor(20)
        feats = torch.randn(2, 3, 4, 20)
        out = head(feats)
        expected = head.proj(feats).squeeze(-1)
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))
        self.assertEqual(out.shape, (2, 3, 4))


if __name__ == "__main__":
    unittest.main()
