"""Full DSpark composition preserves head math and recurrent block boundaries."""

import copy
import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from specforge.config.draft.architecture.block import DraftOnlyBlockSpec
from specforge.config.draft.architecture.dspark import (
    ConfidenceHeadSpec,
    MarkovHeadSpec,
)
from specforge.config.draft.layer import LayerSpec
from specforge.config.draft.mlp import DenseFFNSpec
from specforge.modeling import build_draft_model
from specforge.modeling.draft.dspark import DSparkDraftModel as LegacyDSparkDraftModel
from specforge.modeling.draft.layers.inputs import LayerInputs
from specforge.modeling.draft.layers.mlp import MoEFFN
from specforge.modeling.draft.models import DSparkDraftModel
from tests.test_config.draft_fixtures import draft_config
from tests.test_config.test_draft_dspark_spec import dspark_spec
from tests.test_modeling.legacy_fixtures import legacy_dflash_config
from tests.test_modeling.legacy_rope_reference import legacy_fp32_rope
from tests.test_modeling.test_canonical_dflash import legacy_name
from tests.test_modeling.test_draft_gqa import layout_for
from tests.test_modeling.test_draft_moe_ffn import expert_spec, moe_spec, router_spec


def model_config(kind="rnn", *, confidence=True, with_markov=True, bound=False):
    return draft_config(
        architecture="DSparkDraftModel",
        spec_config=dspark_spec(
            block=DraftOnlyBlockSpec(block_size=4),
            markov=None if kind is None else MarkovHeadSpec(type=kind, rank=4),
            confidence=(
                ConfidenceHeadSpec(with_markov=with_markov) if confidence else None
            ),
        ),
        **({} if bound else {"target_bindings": (), "target_logit_transform": None}),
    )


def model_inputs(config, device="cpu", dtype=torch.float32):
    ids = torch.arange(16, device=device).reshape(2, 8)
    features = torch.randn(2, 3, 64, device=device, dtype=dtype, requires_grad=True)
    layout = layout_for(torch.empty(2, 8, 32, device=device), context_length=3)
    entries = [
        LayerInputs(attention_layout=layout, position_ids=layout.key_positions)
        for _ in config.layer_types
    ]
    return (ids, features, entries), torch.randint(64, (2, 8), device=device)


def moe_model_config(expert_bias=True):
    """A dense layer followed by a MoE layer, as drafts of DeepSeek-style targets."""
    node = moe_spec(
        experts=expert_spec(16),
        shared_expert=expert_spec(16),
        shared_expert_gate=True,
        router=router_spec(expert_bias=expert_bias),
    )
    return replace(
        model_config(),
        mlp_configs={
            "dense": DenseFFNSpec(intermediate_size=64, hidden_act="silu"),
            "moe": node,
        },
        layer_configs={
            "full_dense": LayerSpec(attention="full", mlp="dense"),
            "full_moe": LayerSpec(attention="full", mlp="moe"),
        },
        layer_types=("full_dense", "full_moe"),
    )


def moe_layers(model):
    return [module for module in model.modules() if isinstance(module, MoEFFN)]


class CanonicalDSparkTest(unittest.TestCase):
    @legacy_fp32_rope()
    def _parity(self, device, dtype):
        # Canonical RoPE computes in FP32 before casting back to the model dtype.
        tolerance = (
            2e-2
            if dtype == torch.bfloat16
            else 1e-2 if dtype == torch.float16 else 1e-5
        )
        torch.manual_seed(0)
        for kind, with_markov in (
            (None, False),
            ("vanilla", True),
            ("gated", False),
            ("rnn", True),
        ):
            with self.subTest(device=device, dtype=dtype, kind=kind):
                config = model_config(kind, with_markov=with_markov)
                actual = build_draft_model(config, device=device, dtype=dtype)
                legacy = legacy_dflash_config(
                    block_size=4,
                    dflash_config={
                        "target_layer_ids": [1, 3],
                        "mask_token_id": 63,
                        "markov_rank": 4 if kind else 0,
                        "markov_head_type": kind,
                        "enable_confidence_head": True,
                        "confidence_head_with_markov": with_markov,
                    },
                )
                reference = LegacyDSparkDraftModel(legacy).to(
                    device=device, dtype=dtype
                )
                reference.rotary_emb = Qwen3RotaryEmbedding(legacy).to(device)
                reference.load_state_dict(
                    {
                        legacy_name(n) or n: v
                        for n, v in actual.state_dict().items()
                        if n not in ("embed_tokens.weight", "lm_head.weight")
                    },
                    strict=True,
                )
                embedding, head = copy.deepcopy(actual.embed_tokens), copy.deepcopy(
                    actual.lm_head
                )
                (ids, features, entries), previous = model_inputs(config, device, dtype)
                other = features.detach().clone().requires_grad_()
                output = actual(ids, features, entries, predecessor_ids=previous)
                hidden = reference(
                    entries[0].position_ids,
                    noise_embedding=embedding(ids),
                    target_hidden=other,
                )
                base = head(hidden)
                logits = reference.apply_logits_head(
                    base.reshape(2, 2, 4, 64),
                    prev_token_ids=previous.reshape(2, 2, 4),
                    hidden_states=hidden.reshape(2, 2, 4, 32),
                ).reshape(2, 8, 64)
                confidence = reference.predict_confidence(
                    hidden, prev_token_ids=previous
                )
                torch.testing.assert_close(
                    (output.hidden_states, output.logits, output.confidence_scores),
                    (hidden, logits, confidence),
                    rtol=tolerance,
                    atol=tolerance,
                )
                grad, confidence_grad = torch.randn_like(logits), torch.randn_like(
                    confidence
                )
                (
                    output.logits.mul(grad).sum()
                    + output.confidence_scores.mul(confidence_grad).sum()
                ).backward()
                (
                    logits.mul(grad).sum() + confidence.mul(confidence_grad).sum()
                ).backward()
                torch.testing.assert_close(
                    features.grad, other.grad, rtol=tolerance, atol=tolerance
                )
                for name, parameter in actual.named_parameters():
                    ref = (
                        embedding.weight
                        if name == "embed_tokens.weight"
                        else (
                            head.weight
                            if name == "lm_head.weight"
                            else dict(reference.named_parameters())[
                                legacy_name(name) or name
                            ]
                        )
                    )
                    self.assertIsNotNone(parameter.grad, name)
                    torch.testing.assert_close(
                        parameter.grad, ref.grad, rtol=tolerance, atol=tolerance
                    )

    def test_cpu_legacy_composition(self):
        for dtype in (torch.float32, torch.bfloat16):
            self._parity("cpu", dtype)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_legacy_composition(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            self._parity("cuda", dtype)

    def test_recurrence_resets_at_block_boundaries(self):
        actual = build_draft_model(model_config())
        args, previous = model_inputs(actual.config.draft)
        output = actual(*args, predecessor_ids=previous)
        changed = previous.clone()
        changed[:, :4] = (changed[:, :4] + 1) % 64
        other = actual(*args, predecessor_ids=changed)
        torch.testing.assert_close(
            other.logits[:, 4:], output.logits[:, 4:], rtol=0, atol=0
        )
        torch.testing.assert_close(
            other.confidence_scores[:, 4:],
            output.confidence_scores[:, 4:],
            rtol=0,
            atol=0,
        )
        self.assertFalse(torch.equal(other.logits[:, :4], output.logits[:, :4]))

    def test_binding_checkpoint_and_hf_restore(self):
        config = model_config(bound=True)
        actual = build_draft_model(config)
        weights = {name: torch.randn(64, 32) for name in config.target_bindings}
        for name, value in weights.items():
            getattr(actual, name).load_target_weight(value)
        checkpointed = copy.deepcopy(actual)
        checkpointed.gradient_checkpointing_enable()
        args, previous = model_inputs(config)
        output = actual(*args, predecessor_ids=previous)
        expected = checkpointed(*args, predecessor_ids=previous)
        torch.testing.assert_close(
            (output.logits, output.confidence_scores),
            (expected.logits, expected.confidence_scores),
            rtol=0,
            atol=0,
        )
        (output.logits.sum() + output.confidence_scores.sum()).backward()
        (expected.logits.sum() + expected.confidence_scores.sum()).backward()
        for name, parameter in actual.named_parameters():
            self.assertEqual(parameter.grad is not None, parameter.requires_grad, name)
            if parameter.requires_grad:
                torch.testing.assert_close(
                    parameter.grad,
                    dict(checkpointed.named_parameters())[name].grad,
                    rtol=0,
                    atol=0,
                )
        with TemporaryDirectory() as path:
            actual.save_pretrained(path)
            restored = DSparkDraftModel.from_pretrained(path)
        for name, value in weights.items():
            module = getattr(restored, name)
            if module.weight.is_meta:
                module.to_empty(device="cpu")
            module.load_target_weight(value)
        result = restored(*args, predecessor_ids=previous)
        torch.testing.assert_close(result.logits, output.logits, rtol=0, atol=0)
        torch.testing.assert_close(
            result.confidence_scores, output.confidence_scores, rtol=0, atol=0
        )
        self.assertNotIn("lm_head.weight", actual.state_dict())

    def test_optional_heads_hidden_only_meta_and_input_contract(self):
        config = model_config(None, confidence=False)
        actual = build_draft_model(config)
        args, _ = model_inputs(config)
        output = actual(*args)
        self.assertIsNone(actual.markov_head)
        self.assertIsNone(actual.confidence_head)
        self.assertIsNone(output.confidence_scores)
        torch.testing.assert_close(
            output.logits, actual.lm_head(output.hidden_states), rtol=0, atol=0
        )
        headed = build_draft_model(model_config())
        hidden = headed(*args, output_logits=False)
        self.assertIsNone(hidden.logits)
        self.assertIsNone(hidden.confidence_scores)
        with self.assertRaisesRegex(ValueError, "requires predecessor_ids"):
            headed(*args)
        with self.assertRaisesRegex(ValueError, "whole draft blocks"):
            headed(args[0][:, :-1], args[1], args[2])
        with self.assertRaisesRegex(ValueError, "integer IDs"):
            headed(*args, predecessor_ids=args[0].float())
        meta = build_draft_model(model_config(), device="meta", dtype=torch.bfloat16)
        self.assertTrue(
            all(p.is_meta and p.dtype == torch.bfloat16 for p in meta.parameters())
        )
        tied = build_draft_model(replace(config, tie_word_embeddings=True))
        self.assertIs(tied.embed_tokens.weight, tied.lm_head.weight)

    def test_moe_layer_build_forward_backward_and_activation_checkpointing(self):
        config = moe_model_config()
        model = build_draft_model(config)
        (layer,) = moe_layers(model)
        self.assertIs(layer.config, config.mlp_configs["moe"])
        checkpointed = copy.deepcopy(model)
        checkpointed.gradient_checkpointing_enable()
        args, previous = model_inputs(config)
        output = model(*args, predecessor_ids=previous)
        expected = checkpointed(*args, predecessor_ids=previous)
        torch.testing.assert_close(
            (expected.logits, expected.confidence_scores),
            (output.logits, output.confidence_scores),
            rtol=0,
            atol=0,
        )
        (output.logits.sum() + output.confidence_scores.sum()).backward()
        (expected.logits.sum() + expected.confidence_scores.sum()).backward()
        # The MoE layer's routing travels out as a typed output; dense layers add none.
        (routing,) = output.expert_routing
        self.assertEqual(tuple(routing.expert_ids.shape), (16, 2))
        self.assertTrue(
            torch.equal(routing.expert_ids, expected.expert_routing[0].expert_ids)
        )
        dense = build_draft_model(model_config())
        self.assertIsNone(dense(*args, predecessor_ids=previous).expert_routing)
        others = dict(checkpointed.named_parameters())
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            self.assertIsNotNone(parameter.grad, name)
            torch.testing.assert_close(
                parameter.grad, others[name].grad, rtol=0, atol=0
            )

    def test_moe_restore_keeps_the_fp32_expert_bias_and_checks_the_node(self):
        config = moe_model_config()
        model = build_draft_model(config)
        (layer,) = moe_layers(model)
        with torch.no_grad():
            layer.gate.expert_bias.copy_(torch.tensor([0.5, -0.25, 0.0, 0.125]))
        args, previous = model_inputs(config)
        output = model(*args, predecessor_ids=previous).logits
        with TemporaryDirectory() as path:
            model.save_pretrained(path)
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(dtype=dtype):
                    restored = DSparkDraftModel.from_pretrained(path, dtype=dtype)
                    (restored_layer,) = moe_layers(restored)
                    self.assertEqual(restored_layer.gate.weight.dtype, dtype)
                    self.assertEqual(
                        restored_layer.gate.expert_bias.dtype, torch.float32
                    )
                    torch.testing.assert_close(
                        restored_layer.gate.expert_bias,
                        layer.gate.expert_bias,
                        rtol=0,
                        atol=0,
                    )
                    if dtype == torch.float32:
                        result = restored(*args, predecessor_ids=previous).logits
                        torch.testing.assert_close(result, output, rtol=0, atol=0)
            # A node without the bias rejects a checkpoint that carries one.
            config_path = Path(path, "config.json")
            payload = json.loads(config_path.read_text())
            payload["mlp_configs"]["moe"]["router"]["expert_bias"] = False
            config_path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "unexpected_keys"):
                DSparkDraftModel.from_pretrained(path)


if __name__ == "__main__":
    unittest.main()
