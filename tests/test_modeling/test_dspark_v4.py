# coding=utf-8
"""Tests for the DeepSeek-V4 official-architecture DSpark draft model."""

import pytest
import torch

from specforge.modeling.draft.dspark_v4 import (
    DSparkV4DraftConfig,
    DSparkV4DraftModel,
    _fake_quant_fp8_ue8m0,
    _hc_split_sinkhorn,
)


def tiny_config(**overrides):
    dflash_config = {
        "target_layer_ids": [1, 2, 3],
        "mask_token_id": 7,
        "markov_rank": 8,
        "markov_head_type": "vanilla",
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "kv_fake_quant": True,
        "moe_bias_update_rate": 0.001,
    }
    defaults = dict(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=3,
        num_target_layers=6,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        qk_rope_head_dim=8,
        q_lora_rank=16,
        o_lora_rank=8,
        o_groups=2,
        sliding_window=8,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        block_size=5,
        architectures=["DSparkV4DraftModel"],
        dflash_config=dflash_config,
    )
    defaults.update(overrides)
    return DSparkV4DraftConfig(**defaults)


def build_model(**overrides):
    torch.manual_seed(0)
    model = DSparkV4DraftModel(tiny_config(**overrides))
    # Break the deliberate zero-init symmetry so every path is exercised.
    state = {
        k: torch.randn_like(v) * 0.05 if v.is_floating_point() else v
        for k, v in model.state_dict().items()
    }
    for key in state:
        if "hc_" in key and key.endswith("scale"):
            state[key] = torch.ones_like(state[key])
    model.load_state_dict(state)
    return model


def run_forward(model, batch=2, seq=24, n_blocks=3, keep=None):
    cfg = model.config
    torch.manual_seed(1)
    features = torch.randn(batch, seq, 3 * cfg.hidden_size)
    noise = torch.randn(batch, n_blocks * cfg.block_size, cfg.hidden_size)
    anchors = torch.randint(0, seq - cfg.block_size, (batch, n_blocks)).sort(-1).values
    if keep is None:
        keep = torch.ones(batch, n_blocks, dtype=torch.bool)
    out = model(
        noise_embedding=noise,
        target_hidden=features,
        anchor_positions=anchors,
        block_keep_mask=keep,
    )
    return out, anchors


class TestDSparkV4Model:
    def test_state_dict_naming(self):
        model = build_model()
        keys = set(model.state_dict())
        assert "mtp.0.main_proj.weight" in keys
        assert "mtp.0.main_norm.weight" in keys
        assert "mtp.2.norm.weight" in keys
        assert "mtp.2.hc_head_fn" in keys
        assert "mtp.1.ffn.gate.bias" in keys
        assert "mtp.0.attn.attn_sink" in keys
        # The heads live at the module root (FSDP: used outside stage
        # forwards); the bundler maps them back to the official
        # mtp.<last>.* names at export time.
        assert "markov_head.markov_w1.weight" in keys
        assert "markov_head.markov_w2.weight" in keys
        assert "confidence_head.proj.weight" in keys
        # main_proj/main_norm only on stage 0; hc_head only on the last.
        assert "mtp.1.main_proj.weight" not in keys
        assert "mtp.0.hc_head_fn" not in keys

    def test_load_accepts_official_head_naming(self):
        model = build_model()
        state = dict(model.state_dict())
        for head in ("markov_head.", "confidence_head."):
            for key in [k for k in state if k.startswith(head)]:
                state["mtp.2." + key] = state.pop(key)
        fresh = DSparkV4DraftModel(tiny_config())
        missing, unexpected = fresh.load_state_dict(state, strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        assert torch.equal(
            fresh.confidence_head.proj.weight, model.confidence_head.proj.weight
        )

    def test_state_dict_round_trip(self):
        model = build_model()
        state = model.state_dict()
        fresh = DSparkV4DraftModel(tiny_config())
        missing, unexpected = fresh.load_state_dict(state, strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        out_a, _ = run_forward(model.eval())
        out_b, _ = run_forward(fresh.eval())
        assert torch.equal(out_a, out_b)

    def test_from_pretrained_round_trip(self):
        """Warm start loads via HF from_pretrained; transformers 5 runs
        _init_weights AFTER loading and only per-param loaded flags prevent
        re-initialization — raw .data writes would silently clobber every
        loaded tensor."""
        import os
        import tempfile

        from safetensors.torch import save_file

        model = build_model()
        state = model.state_dict()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny_config()
            cfg.architectures = ["DSparkV4DraftModel"]
            cfg.save_pretrained(tmp)
            save_file(
                {k: v.contiguous() for k, v in state.items()},
                os.path.join(tmp, "model.safetensors"),
            )
            from specforge.modeling.auto import AutoDraftModel

            loaded, info = AutoDraftModel.from_pretrained(
                tmp, config=cfg, output_loading_info=True
            )
        assert not info["missing_keys"] and not info["unexpected_keys"]
        loaded_state = dict(loaded.state_dict())
        for key, value in state.items():
            assert torch.equal(loaded_state[key], value), key

    def test_forward_and_backward(self):
        model = build_model()
        out, _ = run_forward(model)
        assert out.shape == (2, 15, 32)
        assert torch.isfinite(out).all()
        prenorm = model.pop_confidence_hidden()
        assert prenorm.shape == out.shape
        assert model.pop_confidence_hidden() is None  # popped exactly once
        prev_ids = torch.randint(0, 128, (2, 3, 5))
        conf = model.predict_confidence(
            prenorm.reshape(2, 3, 5, -1), prev_token_ids=prev_ids
        )
        assert conf.shape == (2, 3, 5)
        logits = model.apply_logits_head(
            torch.randn(2, 3, 5, 128), prev_token_ids=prev_ids, hidden_states=None
        )
        loss = out.square().mean() + conf.square().mean() + logits.square().mean()
        loss.backward()
        grads = {
            n for n, p in model.named_parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        }
        # Not every tiny-MoE expert is routed to, but every structural path
        # must carry gradient.
        for required in (
            "mtp.0.main_proj.weight",
            "mtp.0.attn.wq_a.weight",
            "mtp.0.attn.wkv.weight",
            "mtp.0.attn.attn_sink",
            "mtp.1.ffn.gate.weight",
            "mtp.1.ffn.shared_experts.w1.weight",
            "mtp.2.hc_head_fn",
            "mtp.2.norm.weight",
            # heads live on the root module (FSDP); state_dict hooks rename
            # them to the official mtp.<last>.* keys
            "markov_head.markov_w2.weight",
            "confidence_head.proj.weight",
        ):
            assert required in grads, f"no gradient reached {required}"

    def test_context_isolation(self):
        """A block's output only depends on context strictly before its anchor
        and within the window."""
        model = build_model().eval()
        cfg = model.config
        seq, block = 24, cfg.block_size
        torch.manual_seed(2)
        features = torch.randn(1, seq, 3 * cfg.hidden_size)
        noise = torch.randn(1, block, cfg.hidden_size)
        anchors = torch.tensor([[10]])
        keep = torch.ones(1, 1, dtype=torch.bool)
        with torch.no_grad():
            base = model(
                noise_embedding=noise, target_hidden=features,
                anchor_positions=anchors, block_keep_mask=keep,
            )
            # Perturbing features at/after the anchor must not change output.
            after = features.clone()
            after[:, 10:] += 100.0
            out_after = model(
                noise_embedding=noise, target_hidden=after,
                anchor_positions=anchors, block_keep_mask=keep,
            )
            # Perturbing features before the window start must not either.
            before = features.clone()
            before[:, : 10 - cfg.sliding_window] += 100.0
            out_before = model(
                noise_embedding=noise, target_hidden=before,
                anchor_positions=anchors, block_keep_mask=keep,
            )
            # Perturbing inside the window MUST change the output.
            inside = features.clone()
            inside[:, 10 - 2] += 100.0
            out_inside = model(
                noise_embedding=noise, target_hidden=inside,
                anchor_positions=anchors, block_keep_mask=keep,
            )
        assert torch.equal(base, out_after)
        assert torch.equal(base, out_before)
        assert not torch.equal(base, out_inside)

    def test_blocks_are_independent(self):
        """Blocks must not attend to each other's tokens."""
        model = build_model().eval()
        cfg = model.config
        block = cfg.block_size
        torch.manual_seed(3)
        features = torch.randn(1, 24, 3 * cfg.hidden_size)
        noise = torch.randn(1, 2 * block, cfg.hidden_size)
        anchors = torch.tensor([[5, 12]])
        keep = torch.ones(1, 2, dtype=torch.bool)
        with torch.no_grad():
            base = model(
                noise_embedding=noise, target_hidden=features,
                anchor_positions=anchors, block_keep_mask=keep,
            )
            other = noise.clone()
            other[:, block:] += 100.0  # perturb block 1 tokens only
            out = model(
                noise_embedding=other, target_hidden=features,
                anchor_positions=anchors, block_keep_mask=keep,
            )
        assert torch.equal(base[:, :block], out[:, :block])
        assert not torch.equal(base[:, block:], out[:, block:])

    def test_stage_gradient_checkpointing_matches(self):
        model = build_model()
        model.train()
        for stage in model.mtp:
            stage.ffn.bias_update_rate = 0.0  # keep routing state fixed
        out, anchors = run_forward(model)
        model.pop_confidence_hidden()
        out.square().sum().backward()
        grads = {
            n: p.grad.clone() for n, p in model.named_parameters()
            if p.grad is not None
        }
        model.zero_grad(set_to_none=True)
        model.stage_gradient_checkpointing = True
        out2, _ = run_forward(model)
        model.pop_confidence_hidden()
        assert torch.equal(out, out2)
        out2.square().sum().backward()
        for name, expected in grads.items():
            got = dict(model.named_parameters())[name].grad
            assert got is not None, name
            assert torch.allclose(got, expected, rtol=1e-5, atol=1e-7), name

    def test_gate_bias_stays_fp32_and_updates(self):
        model = build_model().to(torch.bfloat16)
        for stage in model.mtp:
            assert stage.ffn.gate.bias.dtype == torch.float32
        model.train()
        before = [stage.ffn.gate.bias.clone() for stage in model.mtp]
        run_forward(model.float())
        model = model  # bias updated in-place during forward (training mode)
        changed = any(
            not torch.equal(before[i], model.mtp[i].ffn.gate.bias)
            for i in range(len(model.mtp))
        )
        assert changed

    def test_gate_bias_frozen_in_eval(self):
        model = build_model().eval()
        before = [stage.ffn.gate.bias.clone() for stage in model.mtp]
        run_forward(model)
        for i, stage in enumerate(model.mtp):
            assert torch.equal(before[i], stage.ffn.gate.bias)

    def test_fake_quant_is_idempotent_and_bounded(self):
        torch.manual_seed(0)
        x = torch.randn(4, 128) * 3
        q1 = _fake_quant_fp8_ue8m0(x.clone())
        q2 = _fake_quant_fp8_ue8m0(q1.clone())
        assert torch.allclose(q1, q2)
        assert (q1 - x).abs().max() < 0.5

    def test_sinkhorn_is_doubly_stochastic(self):
        torch.manual_seed(0)
        mixes = torch.randn(2, 3, 24)
        pre, post, comb = _hc_split_sinkhorn(
            mixes, torch.ones(3), torch.zeros(24), 4, 20, 1e-6
        )
        assert pre.shape == (2, 3, 4) and post.shape == (2, 3, 4)
        assert comb.shape == (2, 3, 4, 4)
        assert torch.allclose(comb.sum(-1), torch.ones(2, 3, 4), atol=1e-3)
        assert torch.allclose(comb.sum(-2), torch.ones(2, 3, 4), atol=1e-3)


class TestOnlineDSparkV4Integration:
    def test_native_backend_end_to_end(self):
        from specforge.algorithms.common.dflash_family_model import (
            OnlineDSparkModel,
        )

        model = build_model()
        cfg = model.config
        vocab, hidden = cfg.vocab_size, cfg.hidden_size
        lm_head = torch.nn.Linear(hidden, vocab, bias=False)
        embed = torch.nn.Embedding(vocab, hidden)
        online = OnlineDSparkModel(
            draft_model=model,
            target_lm_head=lm_head,
            target_embed_tokens=embed,
            mask_token_id=7,
            block_size=cfg.block_size,
            attention_backend="native",
            num_anchors=4,
            loss_decay_gamma=4.0,
            objective_chunk_blocks=2,
        )
        torch.manual_seed(0)
        batch, seq = 2, 32
        input_ids = torch.randint(0, vocab, (batch, seq))
        hidden_states = torch.randn(batch, seq, 3 * hidden)
        last_hidden = torch.randn(batch, seq, hidden)
        loss_mask = torch.ones(batch, seq)
        loss, accuracy, metrics = online(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
            target_last_hidden_states=last_hidden,
        )
        assert torch.isfinite(loss)
        loss.backward()
        assert 0.0 <= accuracy.item() <= 1.0
        ratio = metrics["ratio_metrics"]
        assert "confidence_loss" in ratio and "accuracy_position" in ratio
        # confidence gradient must flow into the confidence head
        conf_grad = model.confidence_head.proj.weight.grad
        assert conf_grad is not None and torch.isfinite(conf_grad).all()

    def test_native_backend_requires_capable_model(self):
        from specforge.algorithms.common.dflash_family_model import (
            OnlineDSparkModel,
        )
        from specforge.modeling.draft.dspark import DSparkDraftModel  # noqa: F401

        model = build_model()
        model.native_block_attention = False
        online = OnlineDSparkModel(
            draft_model=model,
            target_lm_head=torch.nn.Linear(32, 128, bias=False),
            target_embed_tokens=torch.nn.Embedding(128, 32),
            mask_token_id=7,
            block_size=5,
            attention_backend="native",
            num_anchors=4,
        )
        with pytest.raises(ValueError, match="native"):
            online(
                input_ids=torch.randint(0, 128, (1, 16)),
                hidden_states=torch.randn(1, 16, 96),
                loss_mask=torch.ones(1, 16),
                target_last_hidden_states=torch.randn(1, 16, 32),
            )
