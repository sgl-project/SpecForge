"""MTP consumes the Transformers [batch, query, head, dim] output contract."""

import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config

from specforge.modeling.draft.mtp.qwen3_5 import Qwen3MTPAttention


class MTPAttentionLayoutTest(unittest.TestCase):
    def test_backend_output_preserves_query_head_order_and_gradients(self):
        for backend in ("eager", "sdpa"):
            for gated in (False, True):
                with self.subTest(backend=backend, gated=gated):
                    config = Qwen3Config(
                        hidden_size=8,
                        num_attention_heads=2,
                        num_key_value_heads=1,
                        head_dim=4,
                        attn_output_gate=gated,
                    )
                    config._attn_implementation = backend
                    attention = Qwen3MTPAttention(config, layer_idx=0)
                    with torch.no_grad():
                        attention.o_proj.weight.copy_(torch.eye(8))
                    hidden = torch.randn(2, 3, 8)
                    # Q != H exposes shape confusion; values also distinguish
                    # every query/head pair even when Q happens to equal H.
                    returned = torch.arange(48.0).reshape(2, 3, 2, 4).requires_grad_()
                    expected = returned.detach().clone().requires_grad_()
                    reference = expected.reshape(2, 3, 8)
                    if gated:
                        gate = (
                            attention.q_proj(hidden)
                            .view(2, 3, 2, 8)
                            .chunk(2, dim=-1)[1]
                            .reshape(2, 3, 8)
                        )
                        reference = reference * gate.detach().sigmoid()
                    embeddings = (torch.ones(2, 3, 4), torch.zeros(2, 3, 4))

                    def run_backend(*args, **kwargs):
                        return returned, None

                    with (
                        patch(
                            "specforge.modeling.draft.mtp.qwen3_5.eager_attention_forward",
                            run_backend,
                        ),
                        patch(
                            "specforge.modeling.draft.mtp.qwen3_5.ALL_ATTENTION_FUNCTIONS",
                            {"sdpa": run_backend},
                        ),
                    ):
                        output, _ = attention(hidden, position_embeddings=embeddings)
                    torch.testing.assert_close(output, reference, rtol=0, atol=0)
                    upstream = torch.randn_like(output)
                    output.backward(upstream)
                    reference.backward(upstream)
                    torch.testing.assert_close(
                        returned.grad, expected.grad, rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
