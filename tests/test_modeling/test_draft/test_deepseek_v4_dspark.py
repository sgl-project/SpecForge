"""Arithmetic, information-flow, checkpoint and real-objective V4 tests."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
from specforge.modeling.draft.deepseek_v4_dspark import (
    DeepseekV4BlockLayout,
    DeepseekV4DSparkConfig,
    DeepseekV4DSparkDraftModel,
    apply_v4_rope,
    hc_post,
    hc_split_sinkhorn,
)

ROOT = Path(__file__).resolve().parents[3]


def tiny_config(**kwargs):
    dc = dict(target_layer_ids=[4, 5, 6], mask_token_id=127, markov_rank=8)
    dc.update(kwargs.pop("dflash_config", {}))
    values = dict(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=3,
        num_target_layers=7,
        num_attention_heads=4,
        head_dim=8,
        q_lora_rank=8,
        qk_rope_head_dim=4,
        o_groups=2,
        o_lora_rank=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=12,
        hc_mult=2,
        sliding_window=4,
        attention_chunk_blocks=1,
        block_size=10,
        dflash_config=dc,
    )
    values.update(kwargs)
    c = DeepseekV4DSparkConfig(**values)
    c._attn_implementation = "sdpa"
    return c


def inputs(c, anchors=None, s=20):
    anchors = torch.tensor([[3, 12]]) if anchors is None else anchors
    b, n = anchors.shape
    pos = torch.cat(
        (
            torch.arange(s).expand(b, -1),
            (anchors[:, :, None] + torch.arange(c.block_size)).flatten(1),
        ),
        1,
    )
    return dict(
        position_ids=pos,
        attention_mask=DeepseekV4BlockLayout(
            anchors, torch.ones_like(anchors, dtype=torch.bool), s, c.block_size
        ),
        noise_embedding=torch.randn(b, n * c.block_size, c.hidden_size),
        target_hidden=torch.randn(
            b, s, c.hidden_size * len(c.dflash_config["target_layer_ids"])
        ),
    )


def online_model(draft):
    c = draft.config
    head, embed = nn.Linear(c.hidden_size, c.vocab_size, bias=False), nn.Embedding(
        c.vocab_size, c.hidden_size
    )
    head.requires_grad_(False)
    embed.requires_grad_(False)
    return OnlineDSparkModel(
        draft_model=draft,
        target_lm_head=head,
        target_embed_tokens=embed,
        mask_token_id=draft.mask_token_id,
        block_size=draft.block_size,
        attention_backend="sdpa",
        num_anchors=3,
        objective_chunk_blocks=2,
        dspark_ce_loss_alpha=0.1,
        dspark_l1_loss_alpha=0.9,
        dspark_confidence_head_alpha=1,
    )


class V4ArithmeticTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(2)

    def test_rope_interleaving_inverse_and_large_positions(self):
        x = torch.randn(2, 3, 4, 8)
        pos = torch.tensor([[0, 7, 999999], [9, 11, 1048575]])
        rotated = apply_v4_rope(x, pos, 4, 10000)
        angle = pos[..., None].float() * torch.tensor([1.0, 0.01])
        z = torch.view_as_complex(x[..., -4:].reshape(2, 3, 4, 2, 2))
        expected = torch.view_as_real(
            z * torch.polar(torch.ones_like(angle), angle).unsqueeze(2)
        ).flatten(-2)
        torch.testing.assert_close(rotated[..., -4:], expected)
        torch.testing.assert_close(
            apply_v4_rope(rotated, pos, 4, 10000, inverse=True), x
        )

    def test_hc_post_source_destination_orientation(self):
        x, residual = torch.randn(2, 3, 8), torch.randn(2, 3, 2, 8)
        post, comb = torch.rand(2, 3, 2), torch.rand(2, 3, 2, 2)
        reference = torch.stack(
            [
                post[..., j, None] * x
                + sum(comb[..., i, j, None] * residual[..., i, :] for i in range(2))
                for j in range(2)
            ],
            -2,
        )
        torch.testing.assert_close(hc_post(x, residual, post, comb), reference)

    def test_sinkhorn_normalization_and_gradients(self):
        mixes = torch.randn(2, 3, 24, requires_grad=True)
        pre, post, comb = hc_split_sinkhorn(
            mixes, torch.ones(3), torch.zeros(24), 4, 20, 1e-6
        )
        torch.testing.assert_close(
            comb.sum(-1), torch.ones(2, 3, 4), atol=5e-6, rtol=5e-6
        )
        torch.testing.assert_close(
            comb.sum(-2), torch.ones(2, 3, 4), atol=5e-6, rtol=5e-6
        )
        (pre.sum() + post.sum() + (comb * torch.rand_like(comb)).sum()).backward()
        self.assertTrue(torch.isfinite(mixes.grad).all())

    def test_attention_matches_dense_reference_and_sink_gradient(self):
        c = tiny_config()
        model = DeepseekV4DSparkDraftModel(c)
        attn = model.mtp[0].attn
        data = inputs(c)
        x, main, pos, layout = (
            data["noise_embedding"],
            data["target_hidden"][..., :32],
            data["position_ids"],
            data["attention_mask"],
        )
        actual = attn(x, main, pos, layout)
        q = attn.wq_b(attn.q_norm(attn.wq_a(x))).view(1, 20, 4, 8)
        q = F.rms_norm(q, (8,), eps=c.rms_norm_eps)
        q = apply_v4_rope(q, pos[:, 20:], 4, c.rope_theta).transpose(1, 2)
        kv = attn.kv_norm(attn.wkv(torch.cat((main, x), 1))).unsqueeze(2)
        kv = apply_v4_rope(kv, pos, 4, c.rope_theta).transpose(1, 2)
        allow = torch.zeros(20, 40, dtype=torch.bool)
        for block, anchor in enumerate(layout.anchors[0].tolist()):
            allow[block * 10 : (block + 1) * 10, max(0, anchor - 4) : anchor] = True
            allow[
                block * 10 : (block + 1) * 10, 20 + block * 10 : 20 + (block + 1) * 10
            ] = True
        logits = (q @ kv.transpose(-2, -1)) / 8**0.5
        logits = logits.masked_fill(~allow, -torch.inf)
        sink = attn.attn_sink[None, :, None, None].expand(1, 4, 20, 1)
        out = torch.cat((logits, sink), -1).softmax(-1)[..., :-1] @ kv
        out = apply_v4_rope(
            out.transpose(1, 2), pos[:, 20:], 4, c.rope_theta, inverse=True
        )
        out = torch.einsum(
            "bsgd,grd->bsgr", out.reshape(1, 20, 2, -1), attn.wo_a.weight.view(2, 8, -1)
        )
        expected = attn.wo_b(out.flatten(2))
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=2e-5)
        a = torch.autograd.grad(
            actual.square().sum(), tuple(attn.parameters()), retain_graph=True
        )
        e = torch.autograd.grad(expected.square().sum(), tuple(attn.parameters()))
        for ag, eg in zip(a, e):
            torch.testing.assert_close(ag, eg, atol=1e-6, rtol=3e-4)
        self.assertGreater(a[0].abs().sum().item(), 0)

    def test_moe_matches_per_token_reference(self):
        c = tiny_config()
        moe = DeepseekV4DSparkDraftModel(c).mtp[0].ffn
        x = torch.randn(2, 4, 32, requires_grad=True)
        actual = moe(x)
        flat = x.reshape(-1, 32)
        scores = F.softplus(F.linear(flat, moe.gate.weight)).sqrt()
        ids = (scores + moe.gate.bias).topk(2, -1).indices
        weights = scores.gather(-1, ids)
        weights = weights / weights.sum(-1, keepdim=True) * c.routed_scaling_factor
        expected = torch.cat(
            [
                moe.shared_experts(flat[i : i + 1])
                + sum(
                    moe.experts[ids[i, j]](flat[i : i + 1], weights[i, j])
                    for j in range(2)
                )
                for i in range(len(flat))
            ]
        ).reshape_as(x)
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        self.assertGreater(moe.gate.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is not None for p in moe.parameters()))


class V4TrainingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(2)

    def test_training_uses_one_compact_sliding_layout(self):
        draft = DeepseekV4DSparkDraftModel(tiny_config())
        online = online_model(draft)
        with (
            patch(
                "specforge.algorithms.common.dflash_family_model.create_dflash_sdpa_mask",
                side_effect=AssertionError("native V4 must not allocate a dense mask"),
            ),
            patch.object(
                draft,
                "build_training_attention_mask",
                wraps=draft.build_training_attention_mask,
            ) as build_mask,
        ):
            online._forward_draft_blocks(
                torch.randint(128, (2, 40)),
                torch.randn(2, 40, 96),
                torch.ones(2, 40),
            )
        build_mask.assert_called_once()
        self.assertEqual(build_mask.call_args.kwargs["sliding_window"], 4)

    def test_no_future_or_cross_block_teacher_leakage(self):
        model = DeepseekV4DSparkDraftModel(tiny_config()).eval()
        data = inputs(model.config)
        with torch.no_grad():
            baseline = model(**data)
            altered = {
                **data,
                "target_hidden": data["target_hidden"].clone(),
                "noise_embedding": data["noise_embedding"].clone(),
            }
            altered["target_hidden"][:, 3:] += 15
            altered["noise_embedding"][:, 10:] += 10
            torch.testing.assert_close(model(**altered)[:, :10], baseline[:, :10])
            # Same-block future noise IS visible to the first query.
            altered = {**data, "noise_embedding": data["noise_embedding"].clone()}
            altered["noise_embedding"][:, 9] += torch.randn(32) * 10
            self.assertFalse(
                torch.allclose(model(**altered)[:, 0], baseline[:, 0], atol=1e-6)
            )

    def test_block10_objective_backward_checkpointing_and_optimizer(self):
        draft = DeepseekV4DSparkDraftModel(tiny_config()).train()
        draft.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        online = online_model(draft)
        optimizer = torch.optim.AdamW(draft.parameters(), lr=1e-3)
        before = draft.mtp[0].main_proj.weight.detach().clone()
        loss, _, extra = online(
            input_ids=torch.randint(0, 128, (2, 32)),
            hidden_states=torch.randn(2, 32, 96),
            loss_mask=torch.ones(2, 32),
            target_last_hidden_states=torch.randn(2, 32, 32),
        )
        loss = loss[0] if isinstance(loss, (list, tuple)) else loss
        self.assertTrue(torch.isfinite(loss).all())
        loss.backward()
        for name, param in draft.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
        for param in (
            draft.mtp[0].main_proj.weight,
            draft.mtp[0].attn.attn_sink,
            draft.mtp[-1].hc_head_fn,
            draft.mtp[-1].norm.weight,
            draft.markov_head.markov_w2.weight,
            draft.confidence_head.proj.weight,
        ):
            self.assertGreater(param.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, draft.mtp[0].main_proj.weight))

    def test_objective_chunks_preserve_masked_loss_and_all_head_gradients(self):
        draft = DeepseekV4DSparkDraftModel(tiny_config()).train()
        online = online_model(draft)
        hidden = torch.randn(2, 170, 32, requires_grad=True)
        target = torch.randn_like(hidden)
        ids = torch.randint(128, (2, 17, 10))
        mask = torch.ones_like(ids, dtype=torch.bool)
        mask[:, ::3, 6:] = False
        mask[:, ::7] = False
        positions = torch.arange(170).reshape(1, 17, 10).expand(2, -1, -1)
        reference = None
        for chunk in (0, 1, 4, 8, 16):
            online.zero_grad(set_to_none=True)
            hidden.grad = None
            online.objective_chunk_blocks = chunk
            loss, metrics = online._compute_dspark_loss(
                output_hidden=hidden,
                target_ids=ids,
                eval_mask=mask,
                prev_token_ids=ids.roll(1, -1),
                safe_label_indices=positions,
                target_last_hidden_states=target,
                anchor_positions=positions[..., 0],
                collect_detailed_metrics=True,
            )
            loss.backward()
            grads = {
                name: p.grad.clone()
                for name, p in draft.named_parameters()
                if p.grad is not None
            }
            grads["input"] = hidden.grad.clone()
            if reference is None:
                reference = (loss.detach(), metrics, grads)
                continue
            torch.testing.assert_close(loss, reference[0], rtol=2e-6, atol=2e-6)
            self.assertEqual(grads.keys(), reference[2].keys())
            for name, grad in grads.items():
                torch.testing.assert_close(
                    grad,
                    reference[2][name],
                    rtol=3e-5,
                    atol=3e-7,
                    msg=lambda text: f"chunk={chunk}, {name}: {text}",
                )
            for name, pair in metrics["ratio_metrics"].items():
                for value, ref in zip(pair, reference[1]["ratio_metrics"][name]):
                    torch.testing.assert_close(value, ref, rtol=2e-6, atol=2e-6)

    def test_checkpointing_preserves_full_objective_and_all_draft_gradients(self):
        draft = DeepseekV4DSparkDraftModel(tiny_config()).train()
        online = online_model(draft)
        data = dict(
            input_ids=torch.randint(128, (2, 40)),
            hidden_states=torch.randn(2, 40, 96),
            loss_mask=torch.ones(2, 40),
            target_last_hidden_states=torch.randn(2, 40, 32),
        )
        reference = None
        for checkpointing in (False, True):
            online.zero_grad(set_to_none=True)
            if checkpointing:
                draft.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            torch.manual_seed(813)
            loss, _, _ = online(**data)
            loss.backward()
            grads = {name: p.grad.clone() for name, p in draft.named_parameters()}
            if reference is None:
                reference = loss.detach(), grads
                continue
            torch.testing.assert_close(loss, reference[0], rtol=0, atol=0)
            for name, grad in grads.items():
                torch.testing.assert_close(
                    grad,
                    reference[1][name],
                    rtol=1e-5,
                    atol=1e-7,
                    msg=lambda text: f"{name}: {text}",
                )

    def test_confidence_prenorm_logits_postnorm_and_no_bias(self):
        draft = DeepseekV4DSparkDraftModel(tiny_config())
        hidden = torch.randn(2, 3, 10, 32)
        prev = torch.randint(0, 128, (2, 3, 10))
        predicted = draft.predict_confidence(hidden, prev_token_ids=prev)
        reference = F.linear(
            torch.cat((hidden, draft.markov_head.get_prev_embeddings(prev)), -1),
            draft.confidence_head.proj.weight,
        ).squeeze(-1)
        torch.testing.assert_close(predicted, reference)
        torch.testing.assert_close(
            draft.prepare_lm_head_hidden(hidden), draft.mtp[-1].norm(hidden)
        )
        self.assertIsNone(draft.confidence_head.proj.bias)

    def test_save_reload_has_no_missing_or_reinitialized_weights(self):
        model = DeepseekV4DSparkDraftModel(tiny_config())
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            loaded, info = DeepseekV4DSparkDraftModel.from_pretrained(
                tmp, output_loading_info=True
            )
            self.assertFalse(info["missing_keys"])
            self.assertFalse(info["unexpected_keys"])
            for key, value in model.state_dict().items():
                torch.testing.assert_close(
                    value, loaded.state_dict()[key], rtol=0, atol=0
                )

    def test_architecture_registration(self):
        from specforge.algorithms.dspark.providers import COMPATIBLE_DRAFT_ARCHITECTURES
        from specforge.modeling.draft import resolve_draft

        self.assertIn("DeepseekV4DSparkDraftModel", COMPATIBLE_DRAFT_ARCHITECTURES)
        self.assertIs(
            resolve_draft("DeepseekV4DSparkDraftModel"), DeepseekV4DSparkDraftModel
        )

    def test_sglang_export_preserves_native_keys_and_block10(self):
        from safetensors import safe_open

        from specforge.export.to_sglang import export_to_sglang

        model = DeepseekV4DSparkDraftModel(tiny_config())
        model.mtp[0].ffn.gate.bias.copy_(
            torch.tensor([0.1234567, -0.8765432, 0.0000123, 1.00321])
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model.config.architectures = ["DeepseekV4DSparkDraftModel"]
            model.config.save_pretrained(root)
            torch.save(
                {"strategy": "dspark", "draft_state_dict": model.state_dict()},
                root / "state.pt",
            )
            output = root / "serving"
            export_to_sglang(
                str(root / "state.pt"), str(root / "config.json"), str(output)
            )
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["architectures"], ["DeepseekV4ForCausalLMDSpark"])
            self.assertEqual(config["dspark_block_size"], 10)
            self.assertEqual(config["num_nextn_predict_layers"], 3)
            self.assertEqual(config["num_hidden_layers"], 7)
            self.assertIsNone(config["quantization_config"])
            self.assertTrue(config["sample_from_anchor"])
            with safe_open(str(output / "model.safetensors"), framework="pt") as f:
                self.assertEqual(set(f.keys()), set(model.state_dict()))
                for k, v in model.state_dict().items():
                    expected = v if k in dict(model.named_buffers()) else v.bfloat16()
                    torch.testing.assert_close(f.get_tensor(k), expected)

    def test_bf16_placement_keeps_exact_router_bias(self):
        draft = DeepseekV4DSparkDraftModel(tiny_config())
        bias = torch.tensor([0.1234567, -0.8765432, 0.0000123, 1.00321])
        draft.mtp[0].ffn.gate.bias.copy_(bias)
        draft.to(dtype=torch.bfloat16)
        online_model(draft).to(dtype=torch.bfloat16)
        torch.testing.assert_close(draft.mtp[0].ffn.gate.bias, bias, rtol=0, atol=0)

    def test_native_architecture_and_checkpointing_through_training_assembly(self):
        from specforge.algorithms.builtin import builtin_algorithm_registry
        from specforge.config import Config
        from specforge.training.assembly import _load_draft

        with tempfile.TemporaryDirectory() as tmp:
            config = tiny_config(use_gradient_checkpointing=True)
            config.architectures = ["DeepseekV4DSparkDraftModel"]
            config.save_pretrained(tmp)
            cfg = Config(
                model={
                    "target_model_path": "unused",
                    "draft_model_config": str(Path(tmp) / "config.json"),
                    "torch_dtype": "float32",
                },
                data={"hidden_states_path": "unused"},
                training={
                    "strategy": "dspark",
                    "attention_backend": "sdpa",
                },
            )
            _, draft = _load_draft(cfg, builtin_algorithm_registry().resolve("dspark"))
            self.assertIsInstance(draft, DeepseekV4DSparkDraftModel)
            self.assertTrue(draft.is_gradient_checkpointing)
            for stage in draft.mtp:
                self.assertFalse(
                    stage._gradient_checkpointing_func.keywords["use_reentrant"]
                )

        self.assertFalse(
            DeepseekV4DSparkDraftModel(tiny_config()).is_gradient_checkpointing
        )
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            tiny_config(use_gradient_checkpointing="false")

    def test_full_official_checkpoint_key_and_shape_contract(self):
        if not os.environ.get("DEEPSEEK_V4_PATH"):
            self.skipTest("set DEEPSEEK_V4_PATH to test an official checkpoint")
        target = Path(os.environ["DEEPSEEK_V4_PATH"])
        if not (target / "config.json").exists():
            self.skipTest("official checkpoint not available")
        from scripts.warmstart_deepseek_v4_dspark import import_checkpoint

        report = import_checkpoint(
            target,
            ROOT / "configs/deepseek-v4-flash-0731-dspark.json",
            "unused",
            dry_run=True,
        )
        self.assertEqual(report["block_size"], 10)
        self.assertEqual(report["tensor_count"], 2376)
        self.assertEqual(report["randomly_initialized"], [])


if __name__ == "__main__":
    unittest.main()
