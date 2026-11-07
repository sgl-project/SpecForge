"""Attention sinks on the eager and flex dense backends."""

import unittest

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from torch.testing import assert_close
from transformers import Qwen3Config

from specforge.modeling.draft.dflash_kernels import DEFAULT_DFLASH_KERNELS
from specforge.modeling.draft.layers.attention.base import AttentionContext
from specforge.modeling.draft.layers.attention.gqa import GQAAttention
from specforge.modeling.draft.layers.config import AttentionConfig

B, S, H, KV, D = 2, 5, 4, 2, 16  # flex needs head_dim >= 16 on CUDA


def _config(implementation):
    cfg = Qwen3Config(
        hidden_size=H * D,
        num_attention_heads=H,
        num_key_value_heads=KV,
        head_dim=D,
        num_hidden_layers=1,
        layer_types=["full_attention"],
        attention_dropout=0.0,
    )
    cfg._attn_implementation = implementation
    return cfg


class SinkAttention(GQAAttention):
    """GQA plus a per-head sink logit under HF's name ``sinks``."""

    def _init_projections(self, config, kernels):
        super()._init_projections(config, kernels)
        self.sinks = nn.Parameter(torch.linspace(-1.0, 1.0, H))


def _module(cls, implementation):
    torch.manual_seed(0)
    return cls(_config(implementation), 0, DEFAULT_DFLASH_KERNELS, AttentionConfig())


def _identity_rope(x):
    return x.new_ones(1, S, D), x.new_zeros(1, S, D)


def _allowed():
    """A random boolean mask with one query that sees no key at all."""
    torch.manual_seed(1)
    allowed = torch.rand(B, 1, S, S) < 0.6
    allowed[:, :, 0, 0] = True
    allowed[1, :, 3, :] = False
    return allowed


def _mask_for(implementation, allowed):
    if implementation != "flex_attention":
        return allowed

    def mask_mod(b, h, q_idx, kv_idx):
        return allowed[b, 0, q_idx, kv_idx]

    return create_block_mask(mask_mod, B, None, S, S, device=allowed.device)


def _reference(module, x, allowed):
    """softmax over [keys, sink] with the sink column dropped, then o_proj."""
    q, k, v = module._compute_qkv(x, x[:, :0], _identity_rope(x))
    k = k.repeat_interleave(H // KV, dim=1)
    v = v.repeat_interleave(H // KV, dim=1)
    scores = (q @ k.transpose(-1, -2)) * module.scaling
    scores = scores.masked_fill(~allowed, float("-inf"))
    sink = module.sinks.view(1, -1, 1, 1).expand(B, H, S, 1)
    probs = torch.cat([scores, sink], -1).softmax(-1)[..., :-1]
    return module.o_proj((probs @ v).transpose(1, 2).reshape(B, S, -1))


def _run(module, x, mask):
    output = module(x, AttentionContext(_identity_rope(x), attention_mask=mask))
    return output.hidden_states


def _grads(module, loss):
    module.zero_grad()
    loss.backward()
    return {n: p.grad.clone() for n, p in module.named_parameters()}


class AttentionSinkTest(unittest.TestCase):
    def _check_against_reference(self, implementation, device):
        module = _module(SinkAttention, implementation).to(device)
        x = torch.randn(B, S, H * D, device=device)
        allowed = _allowed().to(device)
        mask = _mask_for(implementation, allowed)
        output = _run(module, x, mask)
        assert_close(output, _reference(module, x, allowed), rtol=1e-4, atol=1e-5)
        # The query without keys gives zeros, not the sink's share of nothing.
        self.assertTrue(torch.all(output[1, 3] == 0))
        expected = _grads(module, _reference(module, x, allowed).square().sum())
        grads = _grads(module, _run(module, x, mask).square().sum())
        self.assertFalse(torch.all(grads["sinks"] == 0))
        for name, grad in expected.items():
            assert_close(grads[name], grad, rtol=1e-4, atol=1e-5, msg=name)

    def test_eager_matches_the_sink_reference(self):
        self._check_against_reference("eager", "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_flex_matches_the_sink_reference(self):
        self._check_against_reference("flex_attention", "cuda")

    def _check_sinks_are_learned(self, implementation, device):
        """Only the sinks train, against outputs made with hidden target sinks."""
        module = _module(SinkAttention, implementation).to(device)
        x = torch.randn(B, S, H * D, device=device)
        mask = _mask_for(implementation, _allowed().to(device))
        target_sinks = torch.linspace(1.5, -1.5, H, device=device)
        with torch.no_grad():
            module.sinks.copy_(target_sinks)
            target = _run(module, x, mask)
            module.sinks.zero_()
        for name, parameter in module.named_parameters():
            parameter.requires_grad_(name == "sinks")
        optimizer = torch.optim.Adam([module.sinks], lr=0.1)
        losses = []
        for _ in range(300):
            loss = nn.functional.mse_loss(_run(module, x, mask), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0] * 1e-2)
        assert_close(module.sinks, target_sinks, rtol=0, atol=0.05)

    def test_the_sinks_are_learned_under_eager(self):
        self._check_sinks_are_learned("eager", "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_the_sinks_are_learned_under_flex(self):
        self._check_sinks_are_learned("flex_attention", "cuda")

    def test_cpu_flex_refuses_sinks(self):
        # Inductor has no CPU logsumexp output for flex; the backend says so.
        module = _module(SinkAttention, "flex_attention")
        with self.assertRaisesRegex(ValueError, "CUDA"):
            _run(
                module,
                torch.randn(B, S, H * D),
                _mask_for("flex_attention", _allowed()),
            )

    def test_sdpa_and_flash_refuse_sinks(self):
        module = _module(SinkAttention, "sdpa")
        with self.assertRaisesRegex(ValueError, "sink"):
            _run(module, torch.randn(B, S, H * D), _allowed())

    def test_without_a_sink_the_sdpa_path_is_unchanged(self):
        allowed = _allowed()
        plain = _module(GQAAttention, "sdpa")
        eager = _module(GQAAttention, "eager")
        eager.load_state_dict(plain.state_dict())
        x = torch.randn(B, S, H * D)
        assert_close(
            _run(plain, x, allowed), _run(eager, x, allowed), rtol=1e-5, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
