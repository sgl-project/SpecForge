import os
import unittest
from unittest import mock

import torch
from torch.nn.attention.flex_attention import flex_attention
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import create_dflash_block_mask
from specforge.modeling.draft.dflash import Qwen3DFlashAttention
from specforge.modeling.draft.dflash_kernels import DEFAULT_DFLASH_KERNELS
from specforge.modeling.draft.flex_attention_backend import flex_attention_backend


class FlexAttentionBackendTest(unittest.TestCase):
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
