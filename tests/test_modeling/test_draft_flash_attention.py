"""FlashAttention padding, position contracts, and real CUDA kernel parity."""

import unittest
from dataclasses import replace

import torch

from specforge.modeling.draft.layers.attention.backends import (
    AttentionPreparationScope,
    PaddingMask,
    resolve_attention_backend,
)
from specforge.modeling.draft.layers.attention.masks import AttentionLayout


def make_layout(device):
    qvalid = torch.tensor(
        [[False, True, True, True], [True, True, False, False]], device=device
    )
    kvalid = torch.tensor(
        [
            [False, True, True, True, True, True],
            [True, True, True, False, False, False],
        ],
        device=device,
    )
    return AttentionLayout(
        query_positions=torch.tensor([[0, 12, 13, 14], [21, 22, 0, 0]], device=device),
        key_positions=torch.tensor(
            [[0, 10, 11, 12, 13, 14], [20, 21, 22, 0, 0, 0]], device=device
        ),
        visibility=PaddingMask(qvalid, kvalid),
        active_queries=qvalid,
    )


class FlashAttentionTest(unittest.TestCase):
    def test_preparation_and_rejected_layouts(self):
        backend = resolve_attention_backend("fa")
        layout = make_layout("cpu")
        with AttentionPreparationScope() as scope:

            def prepare(value):
                return scope.prepare(
                    backend,
                    value,
                    batch_size=2,
                    num_heads=4,
                    is_causal=True,
                    sliding_window=3,
                )

            prepared = prepare(layout)
            self.assertIs(prepared, prepare(layout))
            self.assertEqual(prepared.mask.cu_query.tolist(), [0, 3, 5])
            self.assertEqual(prepared.mask.cu_key.tolist(), [0, 5, 8])
            with self.assertRaisesRegex(ValueError, "aligned"):
                prepare(replace(layout, query_positions=layout.query_positions + 1))
            with self.assertRaisesRegex(ValueError, "PaddingMask"):
                prepare(replace(layout, visibility=lambda b, h, q, k: q >= k))
            positions = layout.query_positions.clone()
            positions[0, 2] += 1
            with self.assertRaisesRegex(ValueError, "consecutive"):
                prepare(replace(layout, query_positions=positions))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_output_lse_and_gradients(self):
        # Import failures on a CUDA test runner must not silently skip the backend.
        import flash_attn

        self.assertTrue(callable(flash_attn.flash_attn_varlen_func))
        torch.manual_seed(3)
        layout = make_layout("cuda")
        reversed_layout = replace(
            layout,
            query_positions=layout.key_positions,
            key_positions=layout.query_positions,
            visibility=PaddingMask(
                layout.visibility.key_valid, layout.visibility.query_valid
            ),
            active_queries=layout.visibility.key_valid,
        )
        for layout in (layout, reversed_layout):
            self.check_parity(layout)

    def check_parity(self, layout):
        for causal, window in ((False, None), (True, None), (False, 2), (True, 2)):
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(causal=causal, window=window, dtype=dtype):
                    values = [
                        torch.randn(2, h, n, 32, device="cuda", dtype=dtype)
                        for h, n in (
                            (4, layout.query_positions.shape[1]),
                            (2, layout.key_positions.shape[1]),
                            (2, layout.key_positions.shape[1]),
                        )
                    ]
                    results = []
                    for backend in ("eager", "fa"):
                        q, k, v = [t.clone().requires_grad_() for t in values]
                        with AttentionPreparationScope() as scope:
                            prepared = scope.prepare(
                                resolve_attention_backend(backend),
                                layout,
                                batch_size=2,
                                num_heads=4,
                                is_causal=causal,
                                sliding_window=window,
                            )
                        result = prepared.attend(
                            q,
                            k,
                            v,
                            scale=32**-0.5,
                            dropout_p=0,
                            training=True,
                            return_lse=True,
                        )
                        if backend == "fa":
                            evaluation = prepared.attend(
                                q, k, v, scale=32**-0.5, dropout_p=0.5, training=False
                            )
                            self.assertIsNone(evaluation.lse)
                            torch.testing.assert_close(evaluation.output, result.output)
                        output = result.mask_output(
                            result.output.transpose(1, 2).flatten(2) + 1
                        )
                        output.float().square().mean().backward()
                        self.assertEqual(
                            torch.count_nonzero(output[~layout.active_queries]).item(),
                            0,
                        )
                        results.append((output, result.lse, q.grad, k.grad, v.grad))
                    for actual, expected in zip(results[1], results[0]):
                        torch.testing.assert_close(
                            actual,
                            expected,
                            atol=0.04 if dtype == torch.bfloat16 else 0.005,
                            rtol=0.03,
                        )
