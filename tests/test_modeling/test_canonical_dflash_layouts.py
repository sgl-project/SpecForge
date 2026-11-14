"""Canonical DFlash layouts match explicit causal/window visibility per layer."""

import copy
import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from specforge.config.draft.codec.legacy import decode_legacy_config
from specforge.modeling import build_draft_model
from specforge.modeling.draft.layers.attention import gqa
from specforge.modeling.draft.layers.attention.masks import AttentionLayout
from specforge.modeling.draft.layers.inputs import LayerInputs

CONFIGS = Path(__file__).resolve().parents[2] / "tests/fixtures/legacy_drafts"
HIDDEN, VOCAB, MASK, WINDOW, CONTEXT = 32, 64, 60, 6, 48


def small_payload(name, **overrides):
    payload = json.loads((CONFIGS / name).read_text())
    payload.update(
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=VOCAB,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
    )
    if payload.get("use_sliding_window"):
        # A window much shorter than the context, so anchors straddle it.
        payload["sliding_window"] = WINDOW
    method = dict(payload["dflash_config"], mask_token_id=MASK)
    if "conv_group_size" in method:
        method.update(conv_group_size=4, selector_rank=4, selector_top_k=3)
    payload["dflash_config"] = method
    return payload | overrides


def mla(payload):
    payload = copy.deepcopy(payload)
    payload["dflash_config"]["attention_mode"] = "mla"
    payload.pop("head_dim")
    payload.update(
        q_lora_rank=None,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        rope_interleave=True,
    )
    return payload


def training_layout(anchors, keep, block):
    """`prepare_blocks` visibility: context before the anchor, own block, kept blocks."""
    batch, count = anchors.shape
    queries = (anchors[..., None] + torch.arange(block)).flatten(1)
    keys = torch.cat((torch.arange(CONTEXT).expand(batch, -1), queries), dim=1)
    kv = torch.arange(CONTEXT + count * block).view(1, 1, 1, -1)
    q = torch.arange(count * block).view(1, 1, -1, 1)
    anchor = anchors.view(batch, 1, count, 1).repeat_interleave(block, 2)
    visible = ((kv < CONTEXT) & (kv < anchor)) | (
        (kv >= CONTEXT) & ((kv - CONTEXT) // block == q // block)
    )
    visible &= keep.view(batch, 1, count, 1).repeat_interleave(block, 2)
    return (
        AttentionLayout(
            query_positions=queries, key_positions=keys, visibility=visible
        ),
        keys,
    )


class CanonicalDFlashLayoutTest(unittest.TestCase):
    # Compare attention layout, not fused versus unfused convolution rounding.
    @patch.dict("os.environ", {"SPECFORGE_DFLASH2_FUSED_CONV": "0"})
    def _parity(self, payload, dtype=torch.float32):
        layer_count = payload.get("num_target_layers") or 64
        target = {
            "hidden_size": HIDDEN,
            "num_hidden_layers": layer_count,
            "vocab_size": VOCAB,
        }
        config = decode_legacy_config(payload, target_config=target)
        block = config.spec_config.block.block_size
        torch.manual_seed(0)
        model = build_draft_model(config, dtype=dtype)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_(std=0.1)
        embedding, head = (torch.randn(VOCAB, HIDDEN, dtype=dtype) for _ in range(2))
        model.embed_tokens.load_target_weight(embedding)
        model.lm_head.load_target_weight(head)
        reference = copy.deepcopy(model)
        anchors = torch.tensor([[1, 9, 20, 30], [0, 15, 25, 33]])
        keep = torch.tensor([[True] * 4, [True, False, True, True]])
        layout, keys = training_layout(anchors, keep, block)
        ids = torch.full((2, 4 * block), MASK)
        ids[:, ::block] = torch.randint(0, MASK, (2, 4))
        features = torch.randn(
            2,
            CONTEXT,
            len(config.spec_config.target_features.target_layer_ids) * HIDDEN,
            dtype=dtype,
        )
        entries = [
            LayerInputs(attention_layout=layout, position_ids=keys)
            for _ in config.layer_types
        ]
        output = model(ids, features, entries)
        explicit = []
        q = layout.query_positions[:, None, :, None]
        k = keys[:, None, None, :]
        for layer in reference.backbone.layers:
            attention = layer.self_attn
            spec = attention.config
            visible = layout.visibility.clone()
            if spec.is_causal:
                visible &= k <= q
            if spec.sliding_window is not None:
                visible &= (k - q).abs() < spec.sliding_window
            # All visibility rules are now expressed in the independent mask.
            attention.config = replace(spec, is_causal=False, sliding_window=None)
            explicit.append(
                LayerInputs(
                    attention_layout=replace(layout, visibility=visible),
                    position_ids=keys,
                )
            )
        expected = reference(ids, features, explicit)
        torch.testing.assert_close(
            output.hidden_states, expected.hidden_states, rtol=0, atol=0
        )
        torch.testing.assert_close(output.logits, expected.logits, rtol=0, atol=0)

    def test_bundled_recipes_match_explicit_visibility_under_training_layouts(self):
        for name in (
            "qwen3.6-27b-dflash.json",  # sliding causal + full bidirectional
            "qwen3.6-27b-dflash2.json",  # bidirectional sliding window
            "qwen3.8-27b-dflash2.json",
            "qwen3-8b-dflash.json",  # full bidirectional
        ):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(recipe=name, dtype=dtype):
                    self._parity(small_payload(name), dtype=dtype)

    def test_mla_attention_matches_explicit_visibility_under_training_layouts(self):
        payload = mla(small_payload("qwen3.8-27b-dflash2.json"))
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self._parity(copy.deepcopy(payload), dtype=dtype)

    def _mha_bfloat16_matches_explicit_visibility(self):
        for name in ("qwen3-8b-dflash.json", "qwen3.8-27b-dflash2.json"):
            with self.subTest(recipe=name):
                payload = small_payload(name, num_key_value_heads=4)
                payload["dflash_config"]["attention_mode"] = "mha"
                self._parity(payload, dtype=torch.bfloat16)

    def test_mha_rotary_preserves_fp32_layout(self):
        key = torch.randn(2, 16, 4, 8, dtype=torch.bfloat16).transpose(1, 2)
        cos, sin = torch.randn(2, 16, 8), torch.randn(2, 16, 8)
        first, second = key.float().chunk(2, dim=-1)
        expected = (
            key.float() * cos[:, None] + torch.cat((-second, first), -1) * sin[:, None]
        )
        expected = expected.to(key.dtype)
        actual = gqa.apply_rotary(key, (cos, sin))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.stride(), expected.stride())

    def test_cpu_mha_bfloat16_matches_explicit_visibility(self):
        self._mha_bfloat16_matches_explicit_visibility()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_mha_bfloat16_matches_explicit_visibility(self):
        with torch.device("cuda"):
            self._mha_bfloat16_matches_explicit_visibility()

    def test_explicit_causality_matches_explicit_visibility(self):
        for causal in (True, False):
            with self.subTest(is_causal=causal):
                self._parity(small_payload("qwen3.6-27b-dflash.json", is_causal=causal))


if __name__ == "__main__":
    unittest.main()
