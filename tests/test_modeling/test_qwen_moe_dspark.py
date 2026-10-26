"""Qwen MoE DSpark drafter: routing, balance loss, gated attention and partial RoPE."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from torch.testing import assert_close

from specforge.modeling.draft.dflash import _rope_config, apply_rotary_pos_emb
from specforge.modeling.draft.qwen_moe import QwenSparseMoE

try:
    from transformers.models.qwen4_exp import modeling_qwen4_exp
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
except ImportError:  # transformers < 5.16
    modeling_qwen4_exp = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
FLASH_NEXT_CONFIGS = [
    os.path.join(REPO_ROOT, "configs", f"qwen3.8-flash-next-dspark-moe{n}.json")
    for n in (512, 64)
]


def moe_config(**overrides):
    values = dict(
        hidden_size=32,
        num_experts=8,
        num_experts_per_tok=3,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        norm_topk_prob=True,
        router_aux_loss_coef=0.001,
        hidden_act="silu",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def tiny_draft_config(**overrides):
    from specforge.modeling.auto import AutoDraftModelConfig

    with open(FLASH_NEXT_CONFIGS[1], encoding="utf-8") as stream:
        payload = json.load(stream)
    payload.update(
        hidden_size=64,
        head_dim=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        layer_types=["full_attention"] * 2,
        num_experts=8,
        num_experts_per_tok=3,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        intermediate_size=16,
        vocab_size=300,
    )
    payload["dflash_config"] = dict(
        payload["dflash_config"], target_layer_ids=[0, 1], mask_token_id=299
    )
    payload.update(overrides)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as stream:
        json.dump(payload, stream)
    try:
        config = AutoDraftModelConfig.from_file(stream.name)
    finally:
        os.unlink(stream.name)
    config._attn_implementation = "sdpa"
    return config


class TestShippedConfigs(unittest.TestCase):
    def test_both_arms_differ_only_in_expert_count(self):
        payloads = []
        for path in FLASH_NEXT_CONFIGS:
            with open(path, encoding="utf-8") as stream:
                payloads.append(json.load(stream))
        self.assertEqual(payloads[0]["num_experts"], 512)
        self.assertEqual(payloads[1]["num_experts"], 64)
        self.assertEqual(
            {k: v for k, v in payloads[0].items() if k != "num_experts"},
            {k: v for k, v in payloads[1].items() if k != "num_experts"},
        )

    def test_configs_mirror_the_target_where_serving_needs_them(self):
        with open(FLASH_NEXT_CONFIGS[0], encoding="utf-8") as stream:
            payload = json.load(stream)
        # Full-attention taps only; each tap's next layer exists and has no PLE.
        self.assertEqual(
            payload["dflash_config"]["target_layer_ids"], [3, 15, 23, 35, 43]
        )
        self.assertEqual(payload["num_target_layers"], 48)
        self.assertEqual(payload["block_size"], 7)
        self.assertEqual(payload["dflash_config"]["mask_token_id"], 248070)
        # The target MTP layer's shapes, so it can warm-start the drafter.
        self.assertEqual(
            (
                payload["hidden_size"],
                payload["num_attention_heads"],
                payload["num_key_value_heads"],
                payload["head_dim"],
            ),
            (2560, 24, 2, 256),
        )
        self.assertEqual(payload["moe_intermediate_size"], 640)
        self.assertEqual(payload["num_experts_per_tok"], 10)
        self.assertTrue(payload["attn_output_gate"])
        self.assertEqual(payload["partial_rotary_factor"], 0.25)


@unittest.skipIf(modeling_qwen4_exp is None, "transformers has no qwen4_exp")
class TestMatchesTheTarget(unittest.TestCase):
    """The drafter's FFN and RoPE against Qwen3.8-Flash-Next's own modules."""

    def test_moe_block_matches_the_target_moe_block(self):
        torch.manual_seed(0)
        target_config = Qwen4ExpTextConfig(
            hidden_size=32,
            num_experts=8,
            num_experts_per_tok=3,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
            num_hidden_layers=1,
            layer_types=["linear_attention"],
        )
        target = modeling_qwen4_exp.Qwen4ExpTextSparseMoeBlock(target_config).eval()
        with torch.no_grad():
            for parameter in target.parameters():
                parameter.normal_(0.0, 0.3)
        draft = QwenSparseMoE(moe_config()).eval()
        with torch.no_grad():
            draft.gate.weight.copy_(target.gate.weight)
            for expert in range(8):
                gate_up = target.experts.gate_up_proj[expert]
                draft.experts[expert].gate_proj.weight.copy_(gate_up[:16])
                draft.experts[expert].up_proj.weight.copy_(gate_up[16:])
                draft.experts[expert].down_proj.weight.copy_(
                    target.experts.down_proj[expert]
                )
            draft.shared_expert.load_state_dict(target.shared_expert.state_dict())
            draft.shared_expert_gate.load_state_dict(
                target.shared_expert_gate.state_dict()
            )
        x = torch.randn(2, 9, 32)
        assert_close(draft(x), target(x), rtol=1e-5, atol=1e-6)

    def test_partial_rope_matches_the_target_rope(self):
        draft = tiny_draft_config(head_dim=256, num_attention_heads=2)
        target_config = Qwen4ExpTextConfig(
            hidden_size=64,
            head_dim=256,
            num_attention_heads=2,
            num_hidden_layers=1,
            layer_types=["linear_attention"],
            rope_parameters={
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default",
            },
        )
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

        positions = torch.arange(11)[None]
        draft_rope = Qwen3RotaryEmbedding(_rope_config(draft, "gqa"))
        target_rope = modeling_qwen4_exp.Qwen4ExpTextRotaryEmbedding(target_config)
        cos, sin = draft_rope(torch.zeros(1, 11, 64), positions)
        # Text-only: the three M-RoPE sections share one position.
        target_cos, target_sin = target_rope(
            torch.zeros(1, 11, 64), positions[None].expand(3, -1, -1)
        )
        assert_close(cos, target_cos, rtol=1e-6, atol=1e-6)
        assert_close(sin, target_sin, rtol=1e-6, atol=1e-6)

        q = torch.randn(1, 2, 11, 256)
        k = torch.randn(1, 2, 11, 256)
        ours = apply_rotary_pos_emb(q, k, cos, sin)
        theirs = modeling_qwen4_exp.apply_rotary_pos_emb(q, k, cos, sin)
        assert_close(ours[0], theirs[0])
        assert_close(ours[1], theirs[1])


class TestRoutingAndBalance(unittest.TestCase):
    def _balance(self, moe, x, mask=None):
        moe.zero_grad()
        moe(x, balance_mask=mask)
        (loss,) = moe.pop_aux_losses()
        loss.backward()
        return loss.detach(), moe.gate.weight.grad.clone(), moe.load_balance()

    def test_padding_tokens_change_neither_the_loss_nor_the_router_gradient(self):
        torch.manual_seed(5)
        moe = QwenSparseMoE(moe_config()).train()
        valid = torch.randn(6, 32)
        padded = torch.cat([valid, 10.0 * torch.randn(4, 32)])
        mask = torch.tensor([True] * 6 + [False] * 4)

        expected = self._balance(moe, valid)
        actual = self._balance(moe, padded, mask)
        unmasked = self._balance(moe, padded)

        assert_close(actual[0], expected[0])
        assert_close(actual[1], expected[1])
        self.assertEqual(actual[2], expected[2])
        # Without the mask the padding is counted; this is what was fixed.
        self.assertFalse(torch.allclose(unmasked[0], expected[0]))

    def test_a_mask_of_the_wrong_length_is_refused(self):
        moe = QwenSparseMoE(moe_config()).train()
        with self.assertRaisesRegex(ValueError, "balance mask"):
            moe(torch.randn(4, 32), balance_mask=torch.ones(3, dtype=torch.bool))

    def test_balance_loss_is_the_switch_term(self):
        moe = QwenSparseMoE(moe_config()).train()
        x = torch.randn(11, 32)
        moe(x)
        (loss,) = moe.pop_aux_losses()
        _, indices, probabilities = moe.route(x)
        fraction = torch.bincount(indices.reshape(-1), minlength=8).float() / 11
        assert_close(loss, 8 * (fraction * probabilities.mean(0)).sum())
        self.assertEqual(moe.pop_aux_losses(), [])

    def test_balance_loss_is_top_k_when_routing_is_uniform(self):
        moe = QwenSparseMoE(moe_config(num_experts=4, num_experts_per_tok=1)).train()
        with torch.no_grad():
            # Token i is routed to expert i, with probability spread evenly
            # enough that every expert's mean probability is 1/4.
            moe.gate.weight.copy_(torch.eye(4, 32) * 5.0)
        x = torch.eye(4, 32)
        moe(x)
        (loss,) = moe.pop_aux_losses()
        assert_close(loss, torch.tensor(1.0))

    def test_eval_and_zero_coefficient_record_no_loss(self):
        moe = QwenSparseMoE(moe_config()).eval()
        moe(torch.randn(5, 32))
        self.assertEqual(moe.pop_aux_losses(), [])
        moe = QwenSparseMoE(moe_config(router_aux_loss_coef=0.0)).train()
        moe(torch.randn(5, 32))
        self.assertEqual(moe.pop_aux_losses(), [])
        self.assertIsNotNone(moe.load_balance())

    def test_unrouted_experts_still_receive_a_gradient(self):
        moe = QwenSparseMoE(moe_config(num_experts=8, num_experts_per_tok=1)).train()
        moe(torch.randn(1, 32)).sum().backward()
        for expert in moe.experts:
            for parameter in expert.parameters():
                self.assertIsNotNone(parameter.grad)


class TestDraftModel(unittest.TestCase):
    def _construction_dtypes(self, config):
        """Parameter dtypes as each MoE block finishes construction."""

        from unittest import mock

        from specforge.modeling.auto import AutoDraftModel

        seen = []
        original = QwenSparseMoE.__init__

        def recording_init(module, *args, **kwargs):
            original(module, *args, **kwargs)
            seen.append(module.gate.weight.dtype)

        with mock.patch.object(QwenSparseMoE, "__init__", recording_init):
            model = AutoDraftModel.from_config(config, torch_dtype=torch.bfloat16)
        return seen, model

    def test_moe_drafters_are_built_in_the_requested_dtype(self):
        # Built in fp32 and cast afterwards, the 512-expert drafter would peak
        # at twice its bf16 size before the cast.
        seen, model = self._construction_dtypes(tiny_draft_config())
        self.assertEqual(seen, [torch.bfloat16] * 2)
        self.assertEqual({p.dtype for p in model.parameters()}, {torch.bfloat16})
        self.assertEqual(torch.get_default_dtype(), torch.float32)

    def test_dense_drafters_keep_their_fp32_construction(self):
        from unittest import mock

        from specforge.modeling.auto import AutoDraftModel
        from specforge.modeling.draft.dflash import DFlashDraftModel

        seen = []
        original = DFlashDraftModel.__init__

        def recording_init(module, *args, **kwargs):
            original(module, *args, **kwargs)
            seen.append(module.fc.weight.dtype)

        with mock.patch.object(DFlashDraftModel, "__init__", recording_init):
            model = AutoDraftModel.from_config(
                tiny_draft_config(num_experts=0), torch_dtype=torch.bfloat16
            )
        self.assertEqual(seen, [torch.float32])
        self.assertEqual(model.fc.weight.dtype, torch.bfloat16)

    def test_moe_layers_gated_attention_and_partial_rope_are_built(self):
        from specforge.modeling.auto import AutoDraftModel

        model = AutoDraftModel.from_config(tiny_draft_config())
        layer = model.layers[0]
        self.assertIsInstance(layer.mlp, QwenSparseMoE)
        self.assertTrue(layer.self_attn.output_gate)
        self.assertEqual(layer.self_attn.q_proj.out_features, 4 * 32 * 2)
        self.assertEqual(model.rotary_emb.inv_freq.numel(), 32 // 4 // 2)

    def test_forward_records_one_balance_term_per_layer(self):
        from specforge.modeling.auto import AutoDraftModel

        model = AutoDraftModel.from_config(tiny_draft_config()).train()
        noise = torch.randn(1, 7, 64)
        target_hidden = torch.randn(1, 5, 2 * 64)
        position_ids = torch.arange(12)[None]
        out = model(
            position_ids=position_ids,
            noise_embedding=noise,
            target_hidden=target_hidden,
        )
        self.assertEqual(out.shape, (1, 7, 64))
        aux = model.pop_moe_aux_loss()
        self.assertEqual(aux.dim(), 0)
        self.assertIsNone(model.pop_moe_aux_loss())
        self.assertEqual(
            set(model.moe_load_metrics()),
            {"moe_max_load_ratio", "moe_unused_expert_frac"},
        )

    def test_output_gate_scales_attention_by_its_sigmoid(self):
        from specforge.modeling.auto import AutoDraftModel

        gated = AutoDraftModel.from_config(tiny_draft_config()).eval()
        plain = AutoDraftModel.from_config(
            tiny_draft_config(attn_output_gate=False)
        ).eval()
        attention, reference = gated.layers[0].self_attn, plain.layers[0].self_attn
        with torch.no_grad():
            # Interleave per head as [query | gate]; a large gate bias -> sigmoid 1.
            weight = attention.q_proj.weight.view(4, 2, 32, 64)
            weight[:, 0] = reference.q_proj.weight.view(4, 32, 64)
            weight[:, 1] = 0.0
            for name in ("k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
                getattr(attention, name).load_state_dict(
                    getattr(reference, name).state_dict()
                )
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

        hidden = torch.randn(1, 3, 64)
        context = torch.randn(1, 4, 64)
        rope = Qwen3RotaryEmbedding(_rope_config(tiny_draft_config(), "gqa"))
        embeddings = rope(hidden, torch.arange(7)[None])
        args = dict(
            hidden_states=hidden,
            target_hidden=context,
            position_embeddings=embeddings,
            attention_mask=None,
        )
        # Zero gate logits halve the output: sigmoid(0) = 0.5.
        assert_close(attention(**args)[0], 0.5 * reference(**args)[0])


class TestObjective(unittest.TestCase):
    def _forward(self, model, coefficient):
        for block in model.draft_model.moe_blocks():
            block.aux_loss_coef = coefficient
        torch.manual_seed(7)
        generator = torch.Generator().manual_seed(8)
        return model(
            input_ids=torch.randint(0, 298, (1, 40), generator=generator),
            hidden_states=torch.randn(1, 40, 2 * 64, generator=generator),
            loss_mask=torch.ones(1, 40, dtype=torch.long),
            target_last_hidden_states=torch.randn(1, 40, 64, generator=generator),
        )

    def test_dspark_loss_carries_the_router_balance_term(self):
        from torch import nn

        from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
        from specforge.modeling.auto import AutoDraftModel

        torch.manual_seed(0)
        model = OnlineDSparkModel(
            draft_model=AutoDraftModel.from_config(tiny_draft_config()),
            target_lm_head=nn.Linear(64, 300, bias=False),
            target_embed_tokens=nn.Embedding(300, 64),
            mask_token_id=299,
            block_size=7,
            attention_backend="sdpa",
            num_anchors=3,
        ).train()

        loss, _, metrics = self._forward(model, 0.25)
        aux_numerator, aux_denominator = metrics["ratio_metrics"]["moe_aux_loss"]
        self.assertIn("moe_max_load_ratio", metrics["ratio_metrics"])
        baseline, _, baseline_metrics = self._forward(model, 0.0)
        self.assertNotIn("moe_aux_loss", baseline_metrics["ratio_metrics"])
        assert_close(loss - baseline, 0.25 * aux_numerator / aux_denominator)
        loss.backward()
        self.assertIsNotNone(model.draft_model.layers[0].mlp.gate.weight.grad)

    def test_padded_blocks_stay_out_of_the_balance_term(self):
        """Two rows with 3 and 1 valid blocks: the reported balance term must be
        the Switch formula over the 4 valid blocks' 28 tokens, not all 42."""

        from torch import nn

        from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
        from specforge.modeling.auto import AutoDraftModel

        torch.manual_seed(0)
        model = OnlineDSparkModel(
            draft_model=AutoDraftModel.from_config(tiny_draft_config()),
            target_lm_head=nn.Linear(64, 300, bias=False),
            target_embed_tokens=nn.Embedding(300, 64),
            mask_token_id=299,
            block_size=7,
            attention_backend="sdpa",
            num_anchors=3,
        ).train()
        keep = torch.tensor([[True, True, True], [True, False, False]])
        anchors = torch.tensor([[2, 10, 20], [5, 12, 30]])
        model._sample_anchor_positions = lambda *_, **__: (anchors, keep)
        blocks = model.draft_model.moe_blocks()
        logits = []
        handles = [
            block.gate.register_forward_hook(
                lambda _m, _i, out: logits.append(out.detach())
            )
            for block in blocks
        ]
        seen = []
        draft_forward = model.draft_model.forward

        def record(*args, **kwargs):
            seen.append(kwargs.get("moe_balance_mask"))
            return draft_forward(*args, **kwargs)

        model.draft_model.forward = record
        generator = torch.Generator().manual_seed(8)
        _, _, metrics = model(
            input_ids=torch.randint(0, 298, (2, 40), generator=generator),
            hidden_states=torch.randn(2, 40, 2 * 64, generator=generator),
            loss_mask=torch.ones(2, 40, dtype=torch.long),
            target_last_hidden_states=torch.randn(2, 40, 64, generator=generator),
        )
        for handle in handles:
            handle.remove()

        token_mask = keep.repeat_interleave(7, dim=1)

        def switch(layer_logits, counted):
            probabilities = layer_logits.float().softmax(-1)[counted]
            indices = probabilities.topk(3, dim=-1).indices
            fraction = torch.bincount(indices.reshape(-1), minlength=8).float()
            fraction = fraction / counted.sum()
            return 8 * (fraction * probabilities.mean(0)).sum()

        valid = token_mask.reshape(-1)
        expected = torch.stack([switch(l, valid) for l in logits]).mean()
        everything = torch.stack(
            [switch(l, torch.ones_like(valid)) for l in logits]
        ).mean()
        reported = metrics["ratio_metrics"]["moe_aux_loss"][0]
        assert_close(reported, expected)
        # The padded tokens do route differently, so counting them would show.
        self.assertFalse(torch.allclose(everything, expected))
        # The wrapper hands the drafter the mask as a forward argument.
        self.assertEqual(len(seen), 1)
        self.assertTrue(torch.equal(seen[0], token_mask))

    def test_checkpointed_recompute_keeps_padding_out_of_the_router_gradient(self):
        """Recomputing the layers in backward must see the same balance mask.

        With the mask held as module state and cleared after the forward, the
        recompute counted every token: the loss matched but the router
        gradient did not.
        """

        from torch import nn

        from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
        from specforge.modeling.auto import AutoDraftModel

        keep = torch.tensor([[True, True, True], [True, False, False]])
        anchors = torch.tensor([[2, 10, 20], [5, 12, 30]])

        def router_gradient(checkpointed):
            torch.manual_seed(0)
            model = OnlineDSparkModel(
                draft_model=AutoDraftModel.from_config(tiny_draft_config()),
                target_lm_head=nn.Linear(64, 300, bias=False),
                target_embed_tokens=nn.Embedding(300, 64),
                mask_token_id=299,
                block_size=7,
                attention_backend="sdpa",
                num_anchors=3,
            ).train()
            for block in model.draft_model.moe_blocks():
                block.aux_loss_coef = 0.25
            if checkpointed:
                model.draft_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            model._sample_anchor_positions = lambda *_, **__: (anchors, keep)
            generator = torch.Generator().manual_seed(8)
            loss, _, _ = model(
                input_ids=torch.randint(0, 298, (2, 40), generator=generator),
                hidden_states=torch.randn(2, 40, 2 * 64, generator=generator),
                loss_mask=torch.ones(2, 40, dtype=torch.long),
                target_last_hidden_states=torch.randn(2, 40, 64, generator=generator),
            )
            loss.backward()
            return loss.detach(), model.draft_model.layers[0].mlp.gate.weight.grad

        plain_loss, plain_gradient = router_gradient(checkpointed=False)
        recomputed_loss, recomputed_gradient = router_gradient(checkpointed=True)
        assert_close(recomputed_loss, plain_loss)
        assert_close(recomputed_gradient, plain_gradient)
