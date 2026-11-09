"""The DeepSeek-V4 attention kind against a port of DeepSeek's reference math."""

import unittest

import torch
from torch.nn.attention.flex_attention import create_block_mask
from torch.testing import assert_close
from transformers import Qwen3Config

from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.dflash_kernels import DEFAULT_DFLASH_KERNELS
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.modeling.draft.layers.attention.base import AttentionContext
from specforge.modeling.draft.layers.attention.deepseek_v4 import DeepseekV4Attention
from specforge.modeling.draft.layers.config import AttentionConfig
from specforge.modeling.draft.layers.factory import build_attention

B, CTX, S, H, D, RD = 2, 3, 4, 4, 16, 8  # flex needs head_dim >= 16 on CUDA
HIDDEN, Q_RANK, GROUPS, O_RANK = 32, 16, 2, 8
V4_FLASH = dict(
    hidden_size=4096,
    num_attention_heads=64,
    head_dim=512,
    qk_rope_head_dim=64,
    q_lora_rank=1024,
    o_groups=8,
    o_lora_rank=1024,
)


def _config(implementation="eager", **overrides):
    fields = dict(
        hidden_size=HIDDEN,
        num_attention_heads=H,
        num_key_value_heads=1,
        head_dim=D,
        qk_rope_head_dim=RD,
        q_lora_rank=Q_RANK,
        o_groups=GROUPS,
        o_lora_rank=O_RANK,
        num_hidden_layers=1,
        layer_types=["full_attention"],
        attention_dropout=0.0,
        rope_theta=10000.0,
    )
    fields.update(overrides)
    cfg = Qwen3Config(**fields)
    cfg._attn_implementation = implementation
    return cfg


def _module(implementation, **overrides):
    torch.manual_seed(0)
    module = DeepseekV4Attention(
        _config(implementation, **overrides),
        0,
        DEFAULT_DFLASH_KERNELS,
        AttentionConfig(kind="deepseek_v4"),
    )
    with torch.no_grad():
        module.sinks.copy_(torch.linspace(-1.0, 1.0, H))
    return module


def _position_embeddings(key_len, device):
    """Model-level cos/sin over the rope slice, as DFlash's rotary emits them."""
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, RD, 2, device=device) / RD))
    freqs = torch.outer(torch.arange(key_len, device=device).float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos()[None], emb.sin()[None]


def _freqs_cis(positions):
    """DeepSeek's precompute_freqs_cis without YaRN: the draft's rope type."""
    freqs = 1.0 / (10000.0 ** (torch.arange(0, RD, 2, device=positions.device) / RD))
    angles = torch.outer(positions.float(), freqs)
    return torch.polar(torch.ones_like(angles), angles)


def _rotate(x, freqs_cis, inverse=False):
    """DeepSeek's apply_rotary_emb: complex pairs, the conjugate undoes them."""
    pairs = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    shape = [1, x.shape[1]] + [1] * (x.ndim - 3) + [pairs.shape[-1]]
    freqs_cis = freqs_cis.conj() if inverse else freqs_cis
    return torch.view_as_real(pairs * freqs_cis.view(shape)).flatten(-2).to(x.dtype)


def _reference(module, x, prefix, allowed):
    """DeepSeek's inference/model.py Attention written out in plain PyTorch."""
    eps = module.config.rms_norm_eps
    positions = torch.arange(CTX + S, device=x.device)
    q = module.q_b_proj(module.q_a_norm(module.q_a_proj(x))).view(B, S, H, D)
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)
    q_rope = _rotate(q[..., -RD:], _freqs_cis(positions[-S:]))
    q = torch.cat([q[..., :-RD], q_rope], -1)
    kv = module.kv_norm(module.kv_proj(torch.cat([prefix, x], dim=1)))
    kv = torch.cat([kv[..., :-RD], _rotate(kv[..., -RD:], _freqs_cis(positions))], -1)
    scores = torch.einsum("bshd,btd->bhst", q, kv) * module.scaling
    scores = scores.masked_fill(~allowed, float("-inf"))
    sink = module.sinks.view(1, H, 1, 1).expand(B, H, S, 1)
    probs = torch.cat([scores, sink], -1).softmax(-1)[..., :-1]
    o = torch.einsum("bhst,btd->bshd", probs, kv)
    o_rope = _rotate(o[..., -RD:], _freqs_cis(positions[-S:]), inverse=True)
    o = torch.cat([o[..., :-RD], o_rope], -1).reshape(B, S, GROUPS, -1)
    wo_a = module.o_a_proj.weight.view(GROUPS, O_RANK, -1)
    return module.o_b_proj(torch.einsum("bsgd,grd->bsgr", o, wo_a).flatten(2))


def _allowed(device):
    """A random mask over [context, block] keys with one query that sees nothing."""
    torch.manual_seed(1)
    allowed = torch.rand(B, 1, S, CTX + S, device=device) < 0.6
    allowed[:, :, :, 0] = True
    allowed[1, :, 2, :] = False
    return allowed


def _mask_for(implementation, allowed):
    if implementation != "flex_attention":
        return allowed

    def mask_mod(b, h, q_idx, kv_idx):
        return allowed[b, 0, q_idx, kv_idx]

    return create_block_mask(mask_mod, B, None, S, CTX + S, device=allowed.device)


def _run(module, x, prefix, mask):
    cos, sin = _position_embeddings(CTX + S, x.device)
    context = AttentionContext((cos, sin), attention_mask=mask, key_value_prefix=prefix)
    return module(x, context).hidden_states


def _grads(module, loss):
    module.zero_grad()
    loss.backward()
    return {n: p.grad.clone() for n, p in module.named_parameters()}


def _draft_config(architecture, implementation="eager"):
    dflash_config = {"attention_mode": "deepseek_v4", "target_layer_ids": [1]}
    if architecture == "DSparkDraftModel":
        dflash_config.update(
            {
                "projector_type": "dspark",
                "markov_rank": 4,
                "enable_confidence_head": True,
            }
        )
    return _config(
        implementation,
        architectures=[architecture],
        block_size=3,
        intermediate_size=48,
        num_target_layers=4,
        max_position_embeddings=64,
        vocab_size=64,
        attention_bias=False,
        dflash_config=dflash_config,
    )


class DeepseekV4AttentionTest(unittest.TestCase):
    def _check_against_reference(self, implementation, device):
        module = _module(implementation).to(device)
        x = torch.randn(B, S, HIDDEN, device=device)
        prefix = torch.randn(B, CTX, HIDDEN, device=device)
        allowed = _allowed(device)
        mask = _mask_for(implementation, allowed)
        output = _run(module, x, prefix, mask)
        expected = _reference(module, x, prefix, allowed)
        assert_close(output, expected, rtol=1e-4, atol=1e-5)
        self.assertTrue(torch.all(output[1, 2] == 0))
        grads = _grads(module, _run(module, x, prefix, mask).square().sum())
        expected_grads = _grads(
            module, _reference(module, x, prefix, allowed).square().sum()
        )
        self.assertFalse(torch.all(grads["sinks"] == 0))
        for name, grad in expected_grads.items():
            assert_close(grads[name], grad, rtol=1e-4, atol=1e-5, msg=name)

    def test_eager_matches_the_reference(self):
        self._check_against_reference("eager", "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_flex_matches_the_reference(self):
        self._check_against_reference("flex_attention", "cuda")

    def test_parameter_names_follow_transformers(self):
        names = {name for name, _ in _module("eager").named_parameters()}
        self.assertEqual(
            names,
            {
                "q_a_proj.weight",
                "q_a_norm.weight",
                "q_b_proj.weight",
                "kv_proj.weight",
                "kv_norm.weight",
                "o_a_proj.weight",
                "o_b_proj.weight",
                "sinks",
            },
        )

    def test_v4_flash_shapes_match_the_checkpoint(self):
        with torch.device("meta"):
            module = DeepseekV4Attention(
                _config(**V4_FLASH),
                0,
                DEFAULT_DFLASH_KERNELS,
                AttentionConfig(kind="deepseek_v4"),
            )
        shapes = {name: tuple(p.shape) for name, p in module.named_parameters()}
        self.assertEqual(
            shapes,
            {
                "q_a_proj.weight": (1024, 4096),
                "q_a_norm.weight": (1024,),
                "q_b_proj.weight": (32768, 1024),
                "kv_proj.weight": (512, 4096),
                "kv_norm.weight": (512,),
                "o_a_proj.weight": (8192, 4096),
                "o_b_proj.weight": (4096, 8192),
                "sinks": (64,),
            },
        )

    def test_factory_builds_the_kind(self):
        module = build_attention(
            _config(), 0, DEFAULT_DFLASH_KERNELS, AttentionConfig(kind="deepseek_v4")
        )
        self.assertIsInstance(module, DeepseekV4Attention)

    def test_config_validation(self):
        for overrides, message in (
            ({"num_key_value_heads": 2}, "one KV head"),
            ({"o_groups": 3}, "o_groups"),
            ({"qk_rope_head_dim": D}, "qk_rope_head_dim"),
            ({"o_lora_rank": None}, "missing"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                _module("eager", **overrides)

    def test_dflash_and_dspark_select_the_kind(self):
        for cls in (DFlashDraftModel, DSparkDraftModel):
            model = cls(_draft_config(cls.__name__))
            self.assertEqual(model.attention_mode, "deepseek_v4")
            self.assertIsInstance(model.layers[0].self_attn, DeepseekV4Attention)
            cos, _ = model.rotary_emb(torch.zeros(1, 5, 1), torch.arange(5)[None])
            self.assertEqual(cos.shape[-1], RD)

    def test_dflash_forward_runs_one_block(self):
        model = DFlashDraftModel(_draft_config("DFlashDraftModel"))
        context_length, block_size = 6, model.block_size
        output = model(
            position_ids=torch.arange(context_length + block_size)[None].expand(B, -1),
            noise_embedding=torch.randn(B, block_size, HIDDEN),
            target_hidden=torch.randn(B, context_length, HIDDEN),
        )
        self.assertEqual(tuple(output.shape), (B, block_size, HIDDEN))


if __name__ == "__main__":
    unittest.main(verbosity=2)
