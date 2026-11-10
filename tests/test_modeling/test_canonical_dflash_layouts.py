"""Canonical DFlash/DFlash2 match legacy under multi-block training layouts."""

import copy
import json
import unittest
from pathlib import Path

import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from specforge.config.draft.codec.legacy import decode_legacy_config
from specforge.modeling import build_draft_model
from specforge.modeling.draft.dflash import DFlashDraftModel as LegacyDFlash
from specforge.modeling.draft.dflash import _rope_config
from specforge.modeling.draft.dflash2 import DFlash2DraftModel as LegacyDFlash2
from specforge.modeling.draft.dflash_mask import create_dflash_sdpa_mask
from specforge.modeling.draft.layers.attention.masks import AttentionLayout
from specforge.modeling.draft.layers.inputs import LayerInputs
from tests.test_modeling.test_canonical_dflash import legacy_name

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
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
    def _parity(self, payload):
        layer_count = payload.get("num_target_layers") or 64
        target = {
            "hidden_size": HIDDEN,
            "num_hidden_layers": layer_count,
            "vocab_size": VOCAB,
        }
        config = decode_legacy_config(payload, target_config=target)
        is_dflash2 = config.architecture == "DFlash2DraftModel"
        block = config.spec_config.block.block_size
        torch.manual_seed(0)
        model = build_draft_model(config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_(std=0.1)
        embedding, head = torch.randn(VOCAB, HIDDEN), torch.randn(VOCAB, HIDDEN)
        model.embed_tokens.load_target_weight(embedding)
        model.lm_head.load_target_weight(head)
        legacy_config = Qwen3Config.from_dict(payload)
        legacy_config._attn_implementation = "eager"
        legacy = (LegacyDFlash2 if is_dflash2 else LegacyDFlash)(legacy_config)
        # Same FP32 frequency rule as canonical RoPE.
        legacy.rotary_emb = Qwen3RotaryEmbedding(
            _rope_config(legacy_config, legacy.attention_mode)
        )
        legacy.load_state_dict(
            {legacy_name(n) or n: v for n, v in model.state_dict().items()},
            strict=True,
        )
        anchors = torch.tensor([[1, 9, 20, 30], [0, 15, 25, 33]])
        keep = torch.tensor([[True] * 4, [True, False, True, True]])
        layout, keys = training_layout(anchors, keep, block)
        ids = torch.full((2, 4 * block), MASK)
        ids[:, ::block] = torch.randint(0, MASK, (2, 4))
        features = torch.randn(
            2,
            CONTEXT,
            len(config.spec_config.target_features.target_layer_ids) * HIDDEN,
        )
        entries = [
            LayerInputs(attention_layout=layout, position_ids=keys)
            for _ in config.layer_types
        ]
        output = model(ids, features, entries)
        args = dict(
            anchor_positions=anchors,
            block_keep_mask=keep,
            S=CONTEXT,
            block_size=block,
            device="cpu",
            is_causal=getattr(legacy_config, "is_causal", None),
        )
        masks = {}
        if "full_attention" in legacy.layer_types:
            masks["full_attention"] = create_dflash_sdpa_mask(**args)
        if legacy.sliding_window is not None:
            masks["sliding_attention"] = create_dflash_sdpa_mask(
                **args, sliding_window=legacy.sliding_window
            )
        hidden = legacy(
            keys,
            attention_mask=masks,
            noise_embedding=torch.nn.functional.embedding(ids, embedding),
            target_hidden=features,
        )
        logits = hidden @ head.T
        if is_dflash2:
            logits = legacy.transform_unary_logits(logits)
        torch.testing.assert_close(output.hidden_states, hidden, rtol=0, atol=0)
        torch.testing.assert_close(output.logits, logits, rtol=0, atol=0)

    def test_bundled_recipes_match_legacy_under_training_layouts(self):
        for name in (
            "qwen3.6-27b-dflash.json",  # sliding causal + full bidirectional
            "qwen3.6-27b-dflash2.json",  # bidirectional sliding window
            "qwen3.8-27b-dflash2.json",
            "qwen3-8b-dflash.json",  # full bidirectional
        ):
            with self.subTest(recipe=name):
                self._parity(small_payload(name))

    def test_mla_attention_matches_legacy_under_training_layouts(self):
        self._parity(mla(small_payload("qwen3.8-27b-dflash2.json")))

    def test_explicit_causality_matches_legacy(self):
        for causal in (True, False):
            with self.subTest(is_causal=causal):
                self._parity(small_payload("qwen3.6-27b-dflash.json", is_causal=causal))


if __name__ == "__main__":
    unittest.main()
