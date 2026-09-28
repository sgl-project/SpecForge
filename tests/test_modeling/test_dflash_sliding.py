import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn
from transformers import DynamicCache, Qwen3Config

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.modeling.draft.dflash import (
    DFlashDraftModel,
    resolve_dflash_attention_layout,
)
from specforge.modeling.draft.dflash_mask import build_block_mask, build_dense_mask


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
        self.kernel_options = None

    def forward(self, *, hidden_states, attention_mask, kernel_options=None, **_):
        self.attention_mask = attention_mask
        self.kernel_options = kernel_options
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


def _forward(model, attention_mask, **kwargs):
    noise_embedding = torch.randn(1, 2, model.config.hidden_size)
    target_hidden = torch.randn(1, 4, model.config.hidden_size)
    position_ids = torch.arange(6).unsqueeze(0)
    return model(
        position_ids=position_ids,
        noise_embedding=noise_embedding,
        target_hidden=target_hidden,
        attention_mask=attention_mask,
        **kwargs,
    )


def _reference_generation_mask(context_length, query_length, causal, window):
    return torch.tensor(
        [
            [
                (not causal or key <= context_length + query)
                and (window is None or abs(key - context_length - query) < window)
                for key in range(context_length + query_length)
            ]
            for query in range(query_length)
        ]
    )[None, None]


class TestDFlashGenerationMasks(unittest.TestCase):
    @torch.no_grad()
    def test_forward_fallback_matches_explicit_mask(self):
        for implementation in ("eager", "sdpa"):
            for window in (None, 1, 3):
                for causal, override in (
                    (None, None),
                    (False, None),
                    (True, None),
                    (False, True),
                    (True, False),
                ):
                    for cached_length, query_length in ((0, 4), (3, 4), (3, 1)):
                        with self.subTest(
                            backend=implementation,
                            window=window,
                            causal=causal,
                            override=override,
                            cache=cached_length,
                            query=query_length,
                        ):
                            config = _draft_config(
                                [
                                    (
                                        "full_attention"
                                        if window is None
                                        else "sliding_attention"
                                    )
                                ],
                                window,
                            )
                            config._attn_implementation = implementation
                            if causal is not None:
                                config.is_causal = causal
                            model = DFlashDraftModel(config).eval()
                            self.assertEqual(
                                model.layers[0].self_attn.is_causal,
                                window is not None if causal is None else causal,
                            )
                            resolved = causal if override is None else override
                            if resolved is None:
                                resolved = window is not None
                            inputs = dict(
                                noise_embedding=torch.randn(2, query_length, 8),
                                target_hidden=torch.randn(2, 2, model.fc.in_features),
                                position_ids=torch.arange(
                                    cached_length,
                                    cached_length + 2 + query_length,
                                )[None],
                            )
                            if override is not None:
                                inputs["is_causal"] = override
                            actual_cache, expected_cache = (
                                DynamicCache(),
                                DynamicCache(),
                            )
                            if cached_length:
                                k = torch.randn(2, 1, cached_length, 4)
                                v = torch.randn_like(k)
                                actual_cache.update(k, v, 0)
                                expected_cache.update(k.clone(), v.clone(), 0)
                            actual = model(**inputs, past_key_values=actual_cache)
                            mask = _reference_generation_mask(
                                cached_length + 2,
                                query_length,
                                resolved,
                                window,
                            )
                            expected = model(
                                **inputs,
                                past_key_values=expected_cache,
                                attention_mask=mask,
                            )
                            torch.testing.assert_close(
                                actual, expected, rtol=1e-5, atol=1e-6
                            )

    def test_default_masks_are_shared_by_layer_type(self):
        for implementation, builder in (
            ("sdpa", build_dense_mask),
            ("flex_attention", build_block_mask),
        ):
            model, layers = _capture_model(
                ["full_attention", "sliding_attention"] * 2,
                sliding_window=3,
            )
            model.config._attn_implementation = implementation
            model.config.is_causal = True
            name = (
                "build_block_mask"
                if implementation == "flex_attention"
                else "build_dense_mask"
            )
            with mock.patch(
                "specforge.modeling.draft.dflash." + name,
                wraps=builder,
            ) as create_mask:
                _forward(model, None)
            self.assertEqual(create_mask.call_count, 2)
            self.assertIs(layers[0].attention_mask, layers[2].attention_mask)
            self.assertIs(layers[1].attention_mask, layers[3].attention_mask)
            for layer, window in ((layers[0], None), (layers[1], 3)):
                expected = _reference_generation_mask(4, 2, True, window)
                if implementation == "flex_attention":
                    actual = layer.attention_mask.mask_mod(
                        0,
                        0,
                        torch.arange(2)[:, None],
                        torch.arange(6)[None],
                    )
                    torch.testing.assert_close(actual, expected[0, 0])
                else:
                    torch.testing.assert_close(layer.attention_mask, expected)

    def test_no_fallback_for_supplied_masks_or_flash_attention(self):
        model, layers = _capture_model(
            ["full_attention", "sliding_attention"],
            sliding_window=3,
        )
        model.config.is_causal = True
        for supplied in (
            torch.ones(1, 1, 2, 6, dtype=torch.bool),
            {"full_attention": object(), "sliding_attention": object()},
        ):
            with mock.patch(
                "specforge.modeling.draft.dflash.build_dense_mask",
            ) as create_mask:
                _forward(model, supplied)
            create_mask.assert_not_called()
            for layer_type, layer in zip(model.layer_types, layers):
                expected = (
                    supplied[layer_type] if isinstance(supplied, dict) else supplied
                )
                self.assertIs(layer.attention_mask, expected)
        model.config._attn_implementation = "flash_attention_2"
        with mock.patch(
            "specforge.modeling.draft.dflash.build_dense_mask",
        ) as create_mask:
            _forward(model, None)
        create_mask.assert_not_called()
        self.assertTrue(all(layer.attention_mask is None for layer in layers))

    def test_bidirectional_full_attention_remains_unmasked(self):
        model, layers = _capture_model(["full_attention"])
        for causal in (None, False):
            model.config.is_causal = causal
            with mock.patch(
                "specforge.modeling.draft.dflash.build_dense_mask",
            ) as create_mask:
                _forward(model, None)
            create_mask.assert_not_called()
            self.assertIsNone(layers[0].attention_mask)


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
                "specforge.algorithms.common.dflash_family_model." "build_dense_mask",
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
        self.assertTrue(all(layer.kernel_options is None for layer in layers))

    def test_online_wrapper_forces_standard_triton_flex_backend(self):
        model, layers = _capture_model(["full_attention"])
        wrapper = OnlineDFlashModel(
            draft_model=model,
            target_lm_head=nn.Identity(),
            target_embed_tokens=nn.Embedding(32, model.config.hidden_size),
            mask_token_id=31,
            block_size=2,
            attention_backend="flex_attention",
            num_anchors=1,
        )
        anchors = torch.tensor([[2]])
        keep = torch.tensor([[True]])

        with (
            mock.patch.object(
                wrapper,
                "_sample_anchor_positions",
                return_value=(anchors, keep),
            ),
            mock.patch.object(
                wrapper,
                "_create_noise_embed",
                return_value=torch.randn(1, 2, model.config.hidden_size),
            ),
            mock.patch(
                "specforge.algorithms.common.dflash_family_model." "build_block_mask",
                return_value=torch.tensor([1]),
            ),
        ):
            wrapper._forward_draft_blocks(
                input_ids=torch.ones(1, 4, dtype=torch.long),
                hidden_states=torch.randn(1, 4, model.config.hidden_size),
                loss_mask=torch.ones(1, 4),
            )

        expected_kernel_options = (
            {"BACKEND": "TRITON"}
            if torch.__version__ >= "2.11"
            else {"FORCE_USE_FLEX_ATTENTION": True}
        )
        self.assertEqual(layers[0].kernel_options, expected_kernel_options)


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
