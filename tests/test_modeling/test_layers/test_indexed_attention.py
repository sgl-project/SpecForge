import unittest

import torch
from torch.testing import assert_close

from specforge.modeling.draft.dflash_kernels import DEFAULT_DFLASH_KERNELS
from specforge.modeling.draft.layers.config import AttentionConfig
from specforge.modeling.draft.layers.factory import build_attention
from tests.test_modeling.test_layers.test_composition import config, context


def sparse_config(kind, backend="reference", mode="joint"):
    cfg = config("mla" if kind == "dsa" else "gqa")
    cfg.dflash_config["attention_mode"] = kind
    cfg.sparse_attention = dict(backend=backend, indexer_mode=mode, query_chunk_size=2)
    cfg.index_n_heads = 2
    cfg.index_head_dim = 8
    cfg.index_topk = 3
    cfg.indexer_n_heads = 2
    cfg.indexer_kv_heads = 1
    cfg.indexer_head_dim = 8
    cfg.indexer_budget = 4
    cfg.indexer_compress_ratio = 2
    return cfg


class IndexedAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)

    def module(self, kind, backend="reference", mode="joint"):
        return build_attention(
            sparse_config(kind, backend, mode),
            0,
            DEFAULT_DFLASH_KERNELS,
            AttentionConfig(kind, is_causal=True),
        )

    def test_indexer_scores_match_independent_formulas(self):
        from torch.nn import functional as F

        from specforge.modeling.draft.layers.attention.dsa import DSAIndexer
        from specforge.modeling.draft.layers.attention.qsa import QSAIndexer

        x = torch.randn(1, 6, 16)
        qr = torch.randn(1, 6, 8)
        angles = torch.randn(1, 6, 2).repeat(1, 1, 2)
        cos, sin = angles.cos(), angles.sin()

        def rope(t, c, s):
            rotated = torch.cat((-t[..., 2:4], t[..., :2]), -1)
            return torch.cat((t[..., :4] * c + rotated * s, t[..., 4:]), -1)

        cfg = sparse_config("dsa")
        dsa = DSAIndexer(cfg)
        actual = dsa(x, qr, x, (cos, sin))
        q = F.linear(qr, dsa.wq_b.weight).reshape(1, 6, 2, 8)
        k = F.layer_norm(
            F.linear(x, dsa.wk.weight),
            (8,),
            dsa.k_norm.weight,
            dsa.k_norm.bias,
            cfg.rms_norm_eps,
        )
        q = rope(q, cos[:, :, None], sin[:, :, None])
        k = rope(k, cos, sin)
        weights = F.linear(x, dsa.weights_proj.weight) / 4
        expected = (
            torch.einsum("bqhd,bkd->bqhk", q, k).relu() * weights[..., None]
        ).sum(2)
        assert_close(actual, expected)

        qsa = QSAIndexer(sparse_config("qsa"))

        def norm(t, module):
            return (
                t
                * torch.rsqrt(t.square().mean(-1, keepdim=True) + module.eps)
                * (1 + module.weight)
            )

        projected = F.linear(x, qsa.index_qk_proj.weight)
        q = norm(projected[..., :16].reshape(1, 6, 2, 8), qsa.q_layernorm)
        k = norm(projected[..., 16:].reshape(1, 3, 2, 8).mean(2), qsa.k_layernorm)
        q = rope(q, cos[:, :, None], sin[:, :, None])
        k = rope(k, cos[:, ::2], sin[:, ::2])
        expected = torch.einsum("bqhd,bkd->bqhk", q, k).relu().sum(2) / 8**0.5
        assert_close(qsa(x, x, (cos, sin)), expected)

    def test_gathered_matches_reference_forward_and_all_gradients(self):
        for kind in ("qsa", "dsa"):
            with self.subTest(kind=kind):
                ref = self.module(kind)
                gathered = self.module(kind, "gathered")
                gathered.load_state_dict(ref.state_dict())
                hidden = torch.randn(1, 7, 16)
                grads = []
                outputs = []
                for module in (ref, gathered):
                    x = hidden.clone().requires_grad_(True)
                    out = module(x, context(x), output_auxiliary_losses=True)
                    objective = (
                        out.hidden_states.square().mean()
                        + out.auxiliary_losses[0].value
                    )
                    objective.backward()
                    outputs.append(out.hidden_states)
                    grads.append([x.grad, *[p.grad for p in module.parameters()]])
                assert_close(*outputs, atol=1e-6, rtol=1e-5)
                for a, b in zip(*grads):
                    self.assertIsNotNone(a)
                    assert_close(a, b, atol=2e-6, rtol=2e-5)

    def test_auxiliary_loss_only_trains_indexer(self):
        for kind in ("dsa", "qsa"):
            module = self.module(kind)
            x = torch.randn(2, 9, 16, requires_grad=True)
            out = module(x, context(x), output_auxiliary_losses=True)
            out.auxiliary_losses[0].value.backward()
            self.assertIsNone(x.grad)
            total = 0
            for name, p in module.named_parameters():
                if name.startswith("indexer."):
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
                    total += p.grad.abs().sum().item()
                else:
                    self.assertIsNone(p.grad, name)
            self.assertGreater(total, 0)

    def test_full_budget_matches_dense_attention(self):
        for kind, dense_kind in (("dsa", "mla"), ("qsa", "gqa")):
            cfg = sparse_config(kind, mode="frozen")
            cfg.index_topk = 100
            cfg.indexer_budget = 100
            sparse = build_attention(
                cfg, 0, DEFAULT_DFLASH_KERNELS, AttentionConfig(kind, is_causal=True)
            )
            dense = build_attention(
                cfg,
                0,
                DEFAULT_DFLASH_KERNELS,
                AttentionConfig(dense_kind, is_causal=True),
            )
            dense.load_state_dict(
                {
                    k: v
                    for k, v in sparse.state_dict().items()
                    if not k.startswith("indexer.")
                }
            )
            x = torch.randn(2, 7, 16)
            assert_close(
                sparse(x, context(x)).hidden_states,
                dense(x, context(x)).hidden_states,
                atol=1e-6,
                rtol=1e-5,
            )

    def test_causal_selection_cannot_leak_future_hidden_states(self):
        for kind in ("dsa", "qsa"):
            module = self.module(kind, mode="frozen").eval()
            x = torch.randn(1, 9, 16)
            changed = x.clone()
            changed[:, 5:] += 100
            first = module(x, context(x)).hidden_states
            second = module(changed, context(changed)).hidden_states
            assert_close(first[:, :5], second[:, :5], atol=0, rtol=0)

    def test_masks_empty_rows_and_required_training_output(self):
        from dataclasses import replace

        for kind in ("qsa", "dsa"):
            module = self.module(kind)
            x = torch.randn(1, 5, 16)
            with self.assertRaisesRegex(ValueError, "output_auxiliary_losses"):
                module(x, context(x))
            ctx = replace(
                context(x), attention_mask=torch.zeros(1, 1, 5, 5, dtype=torch.bool)
            )
            out = module(x, ctx, output_auxiliary_losses=True)
            assert_close(out.hidden_states, torch.zeros_like(x))
            assert_close(out.auxiliary_losses[0].value, torch.tensor(0.0))
            out.auxiliary_losses[0].value.backward()
            self.assertTrue(
                all(
                    p.grad is None or torch.isfinite(p.grad).all()
                    for p in module.parameters()
                )
            )
            with self.assertRaisesRegex(NotImplementedError, "cache"):
                module(x, ctx, use_cache=True)

    def test_short_qsa_prefix_with_no_complete_blocks(self):
        module = self.module("qsa")
        x = torch.randn(1, 1, 16, requires_grad=True)
        out = module(x, context(x), output_auxiliary_losses=True)
        self.assertTrue(torch.isfinite(out.hidden_states).all())
        self.assertEqual(out.auxiliary_losses[0].value.item(), 0)
        (out.hidden_states.sum() + out.auxiliary_losses[0].value).backward()

    def test_blockmask_matches_tensor_visibility(self):
        from dataclasses import replace

        from torch.nn.attention.flex_attention import create_block_mask

        for kind in ("dsa", "qsa"):
            module = self.module(kind, "gathered", "frozen").eval()
            hidden = torch.randn(1, 7, 16)
            blockmask = create_block_mask(
                lambda b, h, q, k: (q >= k) & (q != 3),
                B=1,
                H=None,
                Q_LEN=7,
                KV_LEN=7,
                device="cpu",
            )
            positions = torch.arange(7)
            mask = (positions[:, None] >= positions[None, :]) & (
                positions[:, None] != 3
            )
            first = module(hidden, replace(context(hidden), attention_mask=blockmask))
            second = module(
                hidden, replace(context(hidden), attention_mask=mask[None, None])
            )
            assert_close(first.hidden_states, second.hidden_states, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
