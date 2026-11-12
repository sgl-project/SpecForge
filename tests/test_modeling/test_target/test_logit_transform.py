"""SpecForge's target logits reproduce what SGLang serves, per architecture.

Each case restates the served formula from SGLang's model file. The teacher
(shared head plus target-only transform, and ``TargetHead``) must equal it, and
the shared head alone must equal what SGLang's draft paths project through.
"""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.core.compact_teacher import compute_target_from_hidden
from specforge.modeling.target.logit_transform import (
    IDENTITY,
    TargetLogitTransform,
    resolve_target_logit_transform,
)
from specforge.modeling.target.target_head import TargetHead

MUSE_GLIMMER_SCALE = 0.19611613513818404
MUSE_SPARK_SCALE = 0.17677669529663687


def _cap(logits, cap):
    return torch.tanh(logits / cap) * cap


def _config(architecture, nested=False, **fields):
    if nested:
        return SimpleNamespace(
            architectures=[architecture], text_config=SimpleNamespace(**fields)
        )
    return SimpleNamespace(architectures=[architecture], **fields)


def _plain(h, w):
    return h @ w.T


# name -> (config, served(h, W), draft(h, W)); draft is SGLang's projection of
# draft hidden through the borrowed target head.
CASES = {
    "inkling": (
        _config(
            "InklingForConditionalGeneration",
            nested=True,
            logits_mup_width_multiplier=24.0,
        ),
        lambda h, w: (h / 24.0) @ w.T,
        lambda h, w: (h / 24.0) @ w.T,
    ),
    "minicpm": (
        _config("MiniCPMForCausalLM", hidden_size=8, dim_model_base=2),
        lambda h, w: (h / (8 / 2)) @ w.T,
        _plain,
    ),
    "falcon_h1": (
        _config("FalconH1ForCausalLM", lm_head_multiplier=0.5),
        lambda h, w: 0.5 * (h @ w.T),
        _plain,
    ),
    "teleflm": (
        _config(
            "TeleFLMForCausalLM", use_mup=True, output_mult=2.0, mup_scale_factor=8.0
        ),
        lambda h, w: (2.0 / 8.0) * (h @ w.T),
        _plain,
    ),
    "cohere": (
        _config("CohereForCausalLM", logit_scale=0.0625),
        lambda h, w: 0.0625 * (h @ w.T),
        _plain,
    ),
    "cohere2_moe_default": (
        _config("Cohere2MoeForCausalLM"),
        lambda h, w: 0.0625 * (h @ w.T),
        _plain,
    ),
    "granite": (
        _config("GraniteMoeHybridForCausalLM", logits_scaling=8.0),
        lambda h, w: (1 / 8.0) * (h @ w.T),
        _plain,
    ),
    "kimi_k3": (
        _config("KimiK3ForConditionalGeneration", nested=True, logit_scale=0.5),
        lambda h, w: 0.5 * (h @ w.T),
        _plain,
    ),
    "muse_glimmer_flat_defaults": (
        _config("MuseGlimmerForCausalLM"),
        lambda h, w: _cap(MUSE_GLIMMER_SCALE * (h @ w.T), 20.0),
        _plain,
    ),
    "muse_glimmer_nested": (
        _config(
            "MuseGlimmerForConditionalGeneration",
            nested=True,
            output_multiplier=0.25,
            final_logit_softcapping=10.0,
        ),
        lambda h, w: _cap(0.25 * (h @ w.T), 10.0),
        _plain,
    ),
    "muse_spark_defaults": (
        _config("MuseSparkForConditionalGeneration", nested=True),
        lambda h, w: _cap(MUSE_SPARK_SCALE * (h @ w.T), 20.0),
        _plain,
    ),
    "gemma2": (
        _config("Gemma2ForCausalLM", final_logit_softcapping=30.0),
        lambda h, w: _cap(h @ w.T, 30.0),
        _plain,
    ),
    "beam_default": (
        _config("BeamForCausalLM"),
        lambda h, w: _cap(h @ w.T, 30.0),
        _plain,
    ),
    "grok": (
        _config(
            "Grok1ForCausalLM",
            output_multiplier_scale=0.5,
            final_logit_softcapping=30.0,
        ),
        lambda h, w: _cap(h @ (w * 0.5).T, 30.0),
        lambda h, w: h @ (w * 0.5).T,
    ),
    "baichuan2": (
        _config("BaichuanForCausalLM", vocab_size=125696),
        lambda h, w: h @ F.normalize(w, dim=1).T,
        lambda h, w: h @ F.normalize(w, dim=1).T,
    ),
    "bailing_norm_head": (
        _config("BailingMoeV2ForCausalLM", norm_head=True, tie_word_embeddings=False),
        lambda h, w: h @ F.normalize(w, dim=0, eps=1e-7).T,
        lambda h, w: h @ F.normalize(w, dim=0, eps=1e-7).T,
    ),
    "plain": (_config("Qwen3ForCausalLM"), _plain, _plain),
}


def _inputs():
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(2, 3, 8, generator=generator)
    weight = torch.randn(16, 8, generator=generator)
    return hidden, weight


def _target_head(weight, transform):
    head = TargetHead(SimpleNamespace(hidden_size=8, vocab_size=16))
    with torch.no_grad():
        head.fc.weight.copy_(weight)
    head.apply_logit_transform(transform)
    return head


class TargetLogitTransformTest(unittest.TestCase):
    def test_teacher_logits_match_served_logits(self):
        hidden, weight = _inputs()
        for name, (config, served, _) in CASES.items():
            with self.subTest(name):
                transform = resolve_target_logit_transform(config)
                shared = hidden @ transform.fold_into_head(weight).T
                torch.testing.assert_close(
                    transform.apply_teacher(shared), served(hidden, weight)
                )

    def test_shared_head_matches_sglang_draft_projection(self):
        hidden, weight = _inputs()
        for name, (config, _, draft) in CASES.items():
            with self.subTest(name):
                transform = resolve_target_logit_transform(config)
                torch.testing.assert_close(
                    hidden @ transform.fold_into_head(weight).T, draft(hidden, weight)
                )

    def test_target_head_and_compact_teacher_return_served_logits(self):
        hidden, weight = _inputs()
        t2d = torch.zeros(16, dtype=torch.bool)
        t2d[[0, 3, 7, 12]] = True
        loss_mask = torch.ones(2, 3, 1, dtype=torch.long)
        for name, (config, served, _) in CASES.items():
            with self.subTest(name):
                head = _target_head(weight, resolve_target_logit_transform(config))
                expected = served(hidden, weight)
                torch.testing.assert_close(head(hidden), expected)

                target_p, on_draft, token_ids, _ = compute_target_from_hidden(
                    hidden,
                    head.fc.weight,
                    t2d,
                    loss_mask,
                    chunk_size=5,
                    softcap=head.logit_softcap,
                )
                draft_logits = expected[..., t2d]
                torch.testing.assert_close(
                    target_p, torch.softmax(draft_logits, dim=-1)
                )
                torch.testing.assert_close(
                    on_draft,
                    torch.exp(
                        draft_logits - torch.logsumexp(expected, -1, keepdim=True)
                    ),
                )
                self.assertTrue(torch.equal(token_ids, expected.argmax(dim=-1)))

    def test_dflash_teacher_logits_apply_the_target_only_transform(self):
        hidden, weight = _inputs()
        config, served, _ = CASES["muse_spark_defaults"]
        model = SimpleNamespace(
            target_logit_transform=resolve_target_logit_transform(config),
            _project_logits=lambda h: h @ weight.T,
        )
        torch.testing.assert_close(
            OnlineDFlashModel._teacher_logits(model, hidden), served(hidden, weight)
        )

    def test_mup_fold_keeps_the_bf16_division(self):
        # The fold replaced an in-place ``W.div_(m)`` on the bf16 head.
        weight = torch.randn(64, 32).to(torch.bfloat16)
        config = CASES["inkling"][0]
        folded = resolve_target_logit_transform(config).fold_into_head(weight)
        self.assertEqual(folded.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(folded, weight.clone().div_(24.0)))

    def test_mup_divisor_does_not_depend_on_the_architecture(self):
        # SGLang's DSpark and DFlash workers read the field from any target.
        config = SimpleNamespace(
            text_config=SimpleNamespace(logits_mup_width_multiplier=4.0)
        )
        self.assertEqual(resolve_target_logit_transform(config).head_divisor, 4.0)

    def test_identity_paths_return_their_inputs(self):
        hidden, weight = _inputs()
        self.assertIs(IDENTITY.apply_teacher(hidden), hidden)
        self.assertIs(IDENTITY.fold_into_head(weight), weight)
        transform = resolve_target_logit_transform(CASES["plain"][0])
        self.assertTrue(transform.head_is_identity and transform.teacher_is_identity)

    def test_teacher_keeps_the_head_dtype(self):
        # bf16 teachers stay bf16: a scaled target costs no extra vocab memory.
        hidden, weight = (tensor.to(torch.bfloat16) for tensor in _inputs())
        config, served, _ = CASES["muse_spark_defaults"]
        transform = resolve_target_logit_transform(config)
        teacher = transform.apply_teacher(hidden @ weight.T)
        head = _target_head(weight, transform).to(torch.bfloat16)
        for actual in (teacher, head(hidden)):
            self.assertEqual(actual.dtype, torch.bfloat16)
            torch.testing.assert_close(
                actual.float(),
                served(hidden.float(), weight.float()),
                rtol=2e-2,
                atol=2e-2,
            )

    def test_non_positive_softcap_is_disabled(self):
        for value in (0, -1.0, None):
            with self.subTest(value=value):
                config = _config("Gemma2ForCausalLM", final_logit_softcapping=value)
                self.assertIsNone(resolve_target_logit_transform(config).softcap)

    def test_architecture_gated_fields(self):
        cases = (
            _config("BaichuanForCausalLM", vocab_size=64000),
            _config(
                "BailingMoeV2ForCausalLM", norm_head=True, tie_word_embeddings=True
            ),
            _config("TeleFLMForCausalLM", use_mup=False, output_mult=2.0),
            _config("GraniteForCausalLM"),
        )
        for config in cases:
            with self.subTest(config.architectures[0]):
                transform = resolve_target_logit_transform(config)
                self.assertTrue(transform.head_is_identity)
                self.assertTrue(transform.teacher_is_identity)

    def test_unknown_architecture_ignores_and_warns_on_specific_fields(self):
        config = _config("Qwen3ForCausalLM", logit_scale=0.5)
        with self.assertLogs(
            "specforge.modeling.target.logit_transform", level="WARNING"
        ) as logs:
            transform = resolve_target_logit_transform(config)
        self.assertIs(transform.teacher_is_identity, True)
        self.assertIn("logit_scale=0.5", logs.output[0])

    def test_invalid_transforms_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "output_multiplier_scale"):
            resolve_target_logit_transform(_config("Grok1ForCausalLM"))
        for fields in (
            {"head_divisor": 0.0},
            {"logit_scale": float("nan")},
            {"softcap": -1.0},
        ):
            with self.subTest(fields=fields):
                with self.assertRaises(ValueError):
                    TargetLogitTransform(**fields)


if __name__ == "__main__":
    unittest.main()
