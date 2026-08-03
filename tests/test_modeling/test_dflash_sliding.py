import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.modeling.draft.dflash import (
    DFlashDraftModel,
    resolve_dflash_attention_layout,
)


def _draft_config(layer_types, sliding_window=None):
    config = Qwen3Config(
        architectures=["DFlashDraftModel"],
        block_size=2,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=len(layer_types),
        num_target_layers=6,
        head_dim=4,
        max_position_embeddings=64,
        vocab_size=32,
        layer_types=list(layer_types),
        sliding_window=sliding_window,
        use_sliding_window=sliding_window is not None,
    )
    config._attn_implementation = "sdpa"
    return config


class _CaptureLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_mask = None

    def forward(self, *, hidden_states, attention_mask, **_):
        self.attention_mask = attention_mask
        return hidden_states


class _RotaryStub(nn.Module):
    def forward(self, *_):
        return (torch.empty(0), torch.empty(0))


def _capture_model(layer_types, sliding_window=None):
    model = DFlashDraftModel(_draft_config(layer_types, sliding_window))
    capture_layers = [_CaptureLayer() for _ in layer_types]
    model.layers = nn.ModuleList(capture_layers)
    model.fc = nn.Identity()
    model.hidden_norm = nn.Identity()
    model.norm = nn.Identity()
    model.rotary_emb = _RotaryStub()
    return model, capture_layers


def _forward(model, attention_mask):
    noise_embedding = torch.randn(1, 2, model.config.hidden_size)
    target_hidden = torch.randn(1, 4, model.config.hidden_size)
    position_ids = torch.arange(6).unsqueeze(0)
    return model(
        position_ids=position_ids,
        noise_embedding=noise_embedding,
        target_hidden=target_hidden,
        attention_mask=attention_mask,
    )


class TestDFlashSlidingDispatch(unittest.TestCase):
    def test_full_only_model_keeps_single_mask_compatibility(self):
        model, layers = _capture_model(["full_attention", "full_attention"])
        full_mask = torch.tensor([1])

        _forward(model, full_mask)

        self.assertIs(layers[0].attention_mask, full_mask)
        self.assertIs(layers[1].attention_mask, full_mask)

    def test_online_wrapper_builds_both_masks_for_hybrid_model(self):
        model, layers = _capture_model(
            ["sliding_attention", "full_attention"],
            sliding_window=4,
        )
        wrapper = OnlineDFlashModel(
            draft_model=model,
            target_lm_head=nn.Identity(),
            target_embed_tokens=nn.Embedding(32, model.config.hidden_size),
            mask_token_id=31,
            block_size=2,
            attention_backend="sdpa",
            num_anchors=1,
        )
        anchors = torch.tensor([[2]])
        keep = torch.tensor([[True]])
        noise_embedding = torch.randn(1, 2, model.config.hidden_size)
        full_mask = torch.tensor([1])
        sliding_mask = torch.tensor([2])

        with (
            mock.patch.object(
                wrapper,
                "_sample_anchor_positions",
                return_value=(anchors, keep),
            ),
            mock.patch.object(
                wrapper,
                "_create_noise_embed",
                return_value=noise_embedding,
            ),
            mock.patch(
                "specforge.algorithms.common.dflash_family_model."
                "create_dflash_sdpa_mask",
                side_effect=(full_mask, sliding_mask),
            ) as create_mask,
        ):
            wrapper._forward_draft_blocks(
                input_ids=torch.ones(1, 4, dtype=torch.long),
                hidden_states=torch.randn(1, 4, model.config.hidden_size),
                loss_mask=torch.ones(1, 4),
            )

        self.assertEqual(create_mask.call_count, 2)
        self.assertIs(layers[0].attention_mask, sliding_mask)
        self.assertIs(layers[1].attention_mask, full_mask)


class TestDFlashSlidingConfig(unittest.TestCase):
    def test_checked_in_qwen36_config_preserves_hybrid_layout(self):
        config_path = (
            Path(__file__).resolve().parents[2] / "configs" / "qwen3.6-27b-dflash.json"
        )
        config = Qwen3Config.from_json_file(str(config_path))

        layer_types, sliding_window = resolve_dflash_attention_layout(config)

        self.assertEqual(
            list(layer_types),
            [
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
        )
        self.assertEqual(sliding_window, 2048)

    def test_configures_attention_modules_from_layer_types(self):
        model = DFlashDraftModel(
            _draft_config(
                ["sliding_attention", "full_attention", "sliding_attention"],
                sliding_window=7,
            )
        )

        self.assertEqual(
            list(model.layer_types),
            ["sliding_attention", "full_attention", "sliding_attention"],
        )
        self.assertEqual(model.sliding_window, 7)
        self.assertEqual(model.layers[0].self_attn.sliding_window, 7)
        self.assertIsNone(model.layers[1].self_attn.sliding_window)
        self.assertEqual(model.layers[2].self_attn.sliding_window, 7)

    def test_rejects_invalid_attention_layouts(self):
        cases = (
            (["full_attention"], None),
            (["full_attention", "unknown"], None),
            (["sliding_attention", "full_attention"], None),
            (["sliding_attention", "full_attention"], 0),
            (["sliding_attention", "full_attention"], -1),
        )
        for layer_types, sliding_window in cases:
            with self.subTest(
                layer_types=layer_types,
                sliding_window=sliding_window,
            ):
                config = _draft_config(["full_attention", "full_attention"])
                config.layer_types = layer_types
                config.sliding_window = sliding_window
                with self.assertRaises(ValueError):
                    resolve_dflash_attention_layout(config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
