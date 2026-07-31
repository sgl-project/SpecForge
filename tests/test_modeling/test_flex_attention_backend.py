import os
import unittest
from unittest import mock

import torch
from torch.nn.attention.flex_attention import flex_attention
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import (
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
)
from specforge.modeling.draft.dflash import DFlashDraftModel, Qwen3DFlashAttention
from specforge.modeling.draft.dflash_kernels import DEFAULT_DFLASH_KERNELS
from specforge.modeling.draft.flex_attention_backend import flex_attention_backend


class FlexAttentionBackendTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "FlexAttention requires CUDA")
    def test_sliding_block_mask_matches_sdpa(self):
        torch.manual_seed(0)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        context_len, draft_block_size = 16, 4
        anchors = torch.tensor([[8, 12]], device=device)
        keep_blocks = torch.ones(1, 2, dtype=torch.bool, device=device)
        query_len = anchors.shape[1] * draft_block_size
        kv_len = context_len + query_len
        query = torch.randn(1, 2, query_len, 64, device=device, dtype=dtype)
        key = torch.randn(1, 2, kv_len, 64, device=device, dtype=dtype)
        value = torch.randn(1, 2, kv_len, 64, device=device, dtype=dtype)

        block_mask = create_dflash_block_mask(
            anchor_positions=anchors,
            block_keep_mask=keep_blocks,
            S=context_len,
            block_size=draft_block_size,
            device=device,
            sliding_window=8,
        )
        dense_mask = create_dflash_sdpa_mask(
            anchor_positions=anchors,
            block_keep_mask=keep_blocks,
            S=context_len,
            block_size=draft_block_size,
            device=device,
            sliding_window=8,
        )
        compiled_attention = torch.compile(
            lambda q, k, v, mask: flex_attention(
                q,
                k,
                v,
                block_mask=mask,
            ),
            fullgraph=True,
        )

        flex_output = compiled_attention(query, key, value, block_mask)
        sdpa_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=dense_mask,
        )
        torch.testing.assert_close(
            flex_output,
            sdpa_output,
            atol=3e-3,
            rtol=2e-2,
        )

    def test_mixed_layer_types_select_their_own_masks(self):
        config = Qwen3Config(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_hidden_layers=2,
            num_target_layers=4,
            head_dim=8,
            layer_types=["sliding_attention", "full_attention"],
            use_sliding_window=True,
            sliding_window=8,
            attention_dropout=0.0,
            block_size=2,
            dflash_config={"target_layer_ids": [1, 2]},
        )
        config._attn_implementation = "eager"
        model = DFlashDraftModel(config)
        sliding_mask = torch.ones(1, 1, 2, 5, dtype=torch.bool)
        full_mask = torch.ones(1, 1, 2, 5, dtype=torch.bool)

        for layer in model.layers:
            layer.self_attn.forward = mock.Mock(
                side_effect=lambda hidden_states, **kwargs: (hidden_states, None)
            )

        model(
            position_ids=torch.arange(5).unsqueeze(0),
            noise_embedding=torch.randn(1, 2, config.hidden_size),
            target_hidden=torch.randn(1, 3, 2 * config.hidden_size),
            attention_mask={
                "sliding_attention": sliding_mask,
                "full_attention": full_mask,
            },
        )

        self.assertIs(
            model.layers[0].self_attn.forward.call_args.kwargs["attention_mask"],
            sliding_mask,
        )
        self.assertIs(
            model.layers[1].self_attn.forward.call_args.kwargs["attention_mask"],
            full_mask,
        )

    def test_eager_converts_boolean_mask_to_additive_mask(self):
        config = Qwen3Config(
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=1,
            num_key_value_heads=1,
            num_hidden_layers=1,
            head_dim=8,
            layer_types=["sliding_attention"],
            sliding_window=8,
            attention_dropout=0.0,
        )
        config._attn_implementation = "eager"
        attention = Qwen3DFlashAttention(
            config,
            layer_idx=0,
            kernels=DEFAULT_DFLASH_KERNELS,
        )
        boolean_mask = torch.tensor([[[[True, False]]]])
        cos = torch.ones(1, 2, config.head_dim)
        sin = torch.zeros_like(cos)

        with mock.patch(
            "specforge.modeling.draft.dflash.eager_attention_forward",
            return_value=(torch.zeros(1, 1, 1, config.head_dim), None),
        ) as eager:
            attention(
                hidden_states=torch.randn(1, 1, config.hidden_size),
                target_hidden=torch.randn(1, 1, config.hidden_size),
                position_embeddings=(cos, sin),
                attention_mask=boolean_mask,
            )

        additive_mask = eager.call_args.args[4]
        self.assertEqual(additive_mask[0, 0, 0, 0].item(), 0.0)
        self.assertEqual(
            additive_mask[0, 0, 0, 1].item(),
            torch.finfo(additive_mask.dtype).min,
        )

    # This correctness regression test can be deleted when we require
    # torch>=2.13; it tests the Torch 2.11 Inductor monkeypatch for CuteDSL
    # operations in patch_inductor_cutedsl_lowerings().
    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10,
        "FLASH FlexAttention correctness requires a Blackwell CUDA device",
    )
    def test_flash_matches_triton_forward_and_backward(self):
        torch.manual_seed(0)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        batch_size, num_query_heads, num_key_value_heads = 1, 3, 1
        context_len, head_dim = 256, 64
        num_blocks, draft_block_size = 4, 64
        query_len = num_blocks * draft_block_size
        kv_len = context_len + query_len
        anchors = torch.tensor([[64, 128, 192, 224]], device=device)
        keep_blocks = torch.ones(
            (batch_size, num_blocks), dtype=torch.bool, device=device
        )

        inputs = (
            torch.randn(
                batch_size,
                num_query_heads,
                query_len,
                head_dim,
                device=device,
                dtype=dtype,
            ),
            torch.randn(
                batch_size,
                num_key_value_heads,
                kv_len,
                head_dim,
                device=device,
                dtype=dtype,
            ),
            torch.randn(
                batch_size,
                num_key_value_heads,
                kv_len,
                head_dim,
                device=device,
                dtype=dtype,
            ),
        )

        def run_backend(backend, flex_block_size=None):
            block_mask = create_dflash_block_mask(
                anchor_positions=anchors,
                block_keep_mask=keep_blocks,
                S=context_len,
                block_size=draft_block_size,
                device=device,
                flex_block_size=flex_block_size,
                sliding_window=128,
            )

            compiled_attention = torch.compile(
                lambda query, key, value, mask: flex_attention(
                    query,
                    key,
                    value,
                    block_mask=mask,
                    enable_gqa=True,
                    kernel_options={"BACKEND": backend},
                ),
                fullgraph=True,
            )

            query, key, value = [
                tensor.detach().clone().requires_grad_(True) for tensor in inputs
            ]
            output = compiled_attention(query, key, value, block_mask)
            output.float().square().mean().backward()
            torch.cuda.synchronize()
            return output.detach(), tuple(
                tensor.grad.detach() for tensor in (query, key, value)
            )

        triton_output, triton_grads = run_backend("TRITON")
        with mock.patch.dict(os.environ, {"SPECFORGE_FLEX_ATTENTION_BACKEND": "FLASH"}):
            self.assertEqual(flex_attention_backend(), "FLASH")
            flash_output, flash_grads = run_backend("FLASH", (256, 128))

        self.assertTrue(torch.isfinite(flash_output).all())
        torch.testing.assert_close(flash_output, triton_output, atol=3e-3, rtol=2e-2)
        for flash_grad, triton_grad in zip(flash_grads, triton_grads):
            self.assertTrue(torch.isfinite(flash_grad).all())
            torch.testing.assert_close(flash_grad, triton_grad, atol=5e-6, rtol=2e-2)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10,
        "FLASH FlexAttention correctness requires a Blackwell CUDA device",
    )
    def test_dflash_flash_attention_forward_backward_smoke(self):
        config = Qwen3Config(
            hidden_size=256,
            intermediate_size=512,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=1,
            head_dim=64,
            layer_types=["full_attention"],
            attention_dropout=0.0,
        )
        config._attn_implementation = "flex_attention"
        attention = Qwen3DFlashAttention(
            config,
            layer_idx=0,
            kernels=DEFAULT_DFLASH_KERNELS,
        ).to(device="cuda", dtype=torch.bfloat16)
        hidden_states = torch.randn(
            1,
            256,
            config.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        target_hidden = torch.randn(
            1,
            256,
            config.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        block_mask = create_dflash_block_mask(
            anchor_positions=torch.tensor([[64, 128, 192, 224]], device="cuda"),
            block_keep_mask=torch.ones(1, 4, dtype=torch.bool, device="cuda"),
            S=256,
            block_size=64,
            device=torch.device("cuda"),
            flex_block_size=(256, 128),
        )
        cos = torch.ones(1, 512, config.head_dim, device="cuda", dtype=torch.bfloat16)
        sin = torch.zeros_like(cos)

        with mock.patch.dict(os.environ, {"SPECFORGE_FLEX_ATTENTION_BACKEND": "FLASH"}):
            output, weights = attention(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                position_embeddings=(cos, sin),
                attention_mask=block_mask,
            )
            output.float().square().mean().backward()
            torch.cuda.synchronize()

        self.assertIsNone(weights)
        self.assertIsNotNone(hidden_states.grad)
        self.assertIsNotNone(target_hidden.grad)
        self.assertTrue(torch.isfinite(hidden_states.grad).all())
        self.assertTrue(torch.isfinite(target_hidden.grad).all())


if __name__ == "__main__":
    unittest.main()
