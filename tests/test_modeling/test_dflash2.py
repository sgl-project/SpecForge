import unittest
from types import SimpleNamespace

import torch
from torch import nn
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.algorithms.dflash.providers import resume_contract
from specforge.modeling.draft.dflash2 import (
    CandidateSelector,
    DFlash2DraftModel,
    DFlashGroupedConv,
    Qwen3DFlash2DecoderLayer,
)
from specforge.training.strategies.base import DFlashTrainStrategy, StepContext


def _tiny_config(**dflash_overrides):
    method_config = {
        "block_size": 4,
        "conv_group_size": 4,
        "conv_kernel_size": 2,
        "mask_token_id": 31,
        "selector_rank": 4,
        "selector_top_k": 3,
        "target_layer_ids": [1],
        **dflash_overrides,
    }
    return Qwen3Config(
        architectures=["DFlash2DraftModel"],
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        num_target_layers=4,
        head_dim=4,
        max_position_embeddings=64,
        vocab_size=32,
        layer_types=["full_attention"],
        dflash_config=method_config,
    )


class DFlash2ArchitectureTest(unittest.TestCase):
    def test_builds_sglang_compatible_modules_and_keys(self):
        model = DFlash2DraftModel(_tiny_config())

        self.assertEqual(model.block_size, 4)
        self.assertIsInstance(model.layers[0], Qwen3DFlash2DecoderLayer)
        keys = set(model.state_dict())
        self.assertIn("layers.0.attention_conv.base_kernel", keys)
        self.assertIn("layers.0.attention_conv.kernel_projection.weight", keys)
        self.assertIn("layers.0.mlp_conv.base_kernel", keys)
        self.assertIn("candidate_selector.predecessor_codebook", keys)
        self.assertIn("candidate_selector.successor_codebook", keys)
        self.assertIn("candidate_selector.hidden_projection.weight", keys)

    def test_rejects_incomplete_checkpoint_switches(self):
        with self.assertRaisesRegex(ValueError, "conv_kernel_size"):
            DFlash2DraftModel(_tiny_config(conv_kernel_size=None))
        with self.assertRaisesRegex(ValueError, "selector_rank"):
            DFlash2DraftModel(_tiny_config(selector_rank=None))
        with self.assertRaisesRegex(ValueError, "must not exceed block_size"):
            DFlash2DraftModel(_tiny_config(conv_kernel_size=5))

    def test_applies_public_unary_logit_transform(self):
        model = DFlash2DraftModel(
            _tiny_config(output_multiplier=0.2, final_logit_softcapping=2.0)
        )
        logits = torch.tensor([[-100.0, 0.0, 100.0]], dtype=torch.bfloat16)

        actual = model.transform_unary_logits(logits)

        expected = logits.float() * 0.2
        expected = torch.tanh(expected / 2.0) * 2.0
        torch.testing.assert_close(actual, expected)

    def test_resume_contract_tracks_dflash2_specific_semantics(self):
        model = DFlash2DraftModel(_tiny_config())
        training_model = SimpleNamespace(
            attention_backend="eager",
            block_size=4,
            dpace_alpha=0.1,
            loss_decay_gamma=0.9,
            loss_type="dflash",
            lk_loss_type="lambda",
            kl_scale=0.9,
            kl_decay=0.8,
            mask_token_id=31,
            num_anchors=8,
            selector_loss_alpha=0.75,
            selector_ramp_ratio=0.2,
            selector_warmup_ratio=0.1,
        )

        contract = resume_contract(None, model, training_model)

        self.assertEqual(contract["dflash2_conv_kernel_size"], 2)
        self.assertEqual(contract["dflash2_conv_group_size"], 4)
        self.assertEqual(contract["dflash2_selector_rank"], 4)
        self.assertEqual(contract["dflash2_selector_top_k"], 3)
        self.assertEqual(contract["dflash2_selector_loss_alpha"], 0.75)
        self.assertEqual(contract["dflash2_selector_warmup_ratio"], 0.1)
        self.assertEqual(contract["dflash2_selector_ramp_ratio"], 0.2)
        self.assertEqual(contract["dflash_lk_loss_type"], "lambda")

    def test_backward_reaches_convolution_parameters(self):
        config = _tiny_config()
        config._attn_implementation = "eager"
        model = DFlash2DraftModel(config)
        context_length = 3
        noise = torch.randn(1, model.block_size, config.hidden_size)
        target_hidden = torch.randn(1, context_length, config.hidden_size)
        position_ids = torch.arange(context_length + model.block_size).unsqueeze(0)
        attention_mask = torch.ones(
            1,
            1,
            model.block_size,
            context_length + model.block_size,
            dtype=torch.bool,
        )

        output = model(
            position_ids=position_ids,
            attention_mask=attention_mask,
            noise_embedding=noise,
            target_hidden=target_hidden,
        )
        output.square().mean().backward()

        grad = model.layers[0].attention_conv.kernel_projection.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(grad.abs().sum().item(), 0.0)


class DFlash2GroupedConvTest(unittest.TestCase):
    def test_identity_initialization_preserves_inputs(self):
        conv = DFlashGroupedConv(4, block_size=3, taps=2, group_size=2)
        nn.init.zeros_(conv.kernel_projection.weight)
        inputs = torch.randn(2, 6, 4)

        prepared, output_kernel = conv.prepare(inputs)
        finished = conv.finish(inputs, output_kernel)

        torch.testing.assert_close(prepared, inputs)
        torch.testing.assert_close(finished, inputs)

    def test_shifted_tap_does_not_cross_block_boundaries(self):
        conv = DFlashGroupedConv(2, block_size=3, taps=2, group_size=1)
        nn.init.zeros_(conv.kernel_projection.weight)
        with torch.no_grad():
            conv.base_kernel.zero_()
            conv.base_kernel[:, 1].fill_(1.0)
        inputs = torch.tensor(
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                    [7.0, 8.0],
                    [9.0, 10.0],
                    [11.0, 12.0],
                ]
            ]
        )
        expected = torch.tensor(
            [[[0.0, 0.0], [1.0, 2.0], [3.0, 4.0], [0.0, 0.0], [7.0, 8.0], [9.0, 10.0]]]
        )

        actual, _ = conv.prepare(inputs)

        torch.testing.assert_close(actual, expected)


class CandidateSelectorTest(unittest.TestCase):
    def test_fresh_selector_is_a_unary_noop(self):
        selector = CandidateSelector(
            hidden_size=4,
            vocab_size=8,
            state_rank=3,
            top_k=2,
            initializer_range=0.2,
        )
        unary_logits = torch.randn(2, 3, 2)

        scores = selector.score_candidates(
            candidate_ids=torch.randint(0, 8, (2, 3, 2)),
            unary_logits=unary_logits,
            hidden_states=torch.randn(2, 3, 4),
            predecessor_ids=torch.randint(0, 8, (2, 3)),
        )

        torch.testing.assert_close(scores, unary_logits)
        torch.testing.assert_close(
            selector.successor_codebook,
            torch.zeros_like(selector.successor_codebook),
        )

    def test_training_objective_uses_serving_unary_transform(self):
        class Draft(nn.Module):
            def __init__(self):
                super().__init__()
                self.candidate_selector = object()

            @staticmethod
            def transform_unary_logits(logits):
                return logits.float() * 0.5

        model = OnlineDFlashModel(
            draft_model=Draft(),
            target_lm_head=nn.Identity(),
            target_embed_tokens=nn.Embedding(3, 3),
            mask_token_id=2,
            block_size=2,
            attention_backend="eager",
            selector_loss_alpha=0.0,
        )
        hidden = torch.tensor([[[[0.0, 0.0, 0.0], [0.0, 2.0, 1.0]]]])
        target_ids = torch.tensor([[[0, 0]]])
        weights = torch.tensor([[[0.0, 1.0]]])

        terms = model._dflash_objective_chunk_terms(
            hidden,
            target_ids,
            weights,
            target_ids,
        )

        expected = torch.nn.functional.cross_entropy(
            hidden[0, 0, 1].unsqueeze(0) * 0.5,
            torch.tensor([0]),
        )
        torch.testing.assert_close(terms[0], expected)

    def test_scores_unary_plus_predecessor_transition(self):
        selector = CandidateSelector(
            hidden_size=2,
            vocab_size=5,
            state_rank=2,
            top_k=2,
            initializer_range=0.02,
        )
        with torch.no_grad():
            selector.hidden_projection.weight.copy_(torch.eye(2))
            selector.predecessor_codebook.zero_()
            selector.successor_codebook.zero_()
            selector.predecessor_codebook[1] = torch.tensor([2.0, 3.0])
            selector.successor_codebook[2] = torch.tensor([5.0, 7.0])
            selector.successor_codebook[4] = torch.tensor([11.0, 13.0])

        scores = selector.score_candidates(
            candidate_ids=torch.tensor([[2, 4]]),
            unary_logits=torch.tensor([[0.5, 1.5]]),
            hidden_states=torch.tensor([[17.0, 19.0]]),
            predecessor_ids=torch.tensor([1]),
        )
        expected = torch.tensor(
            [[0.5 + 2 * 17 * 5 + 3 * 19 * 7, 1.5 + 2 * 17 * 11 + 3 * 19 * 13]]
        )
        torch.testing.assert_close(scores, expected)

    def test_lattice_rows_match_realized_predecessor_scores(self):
        torch.manual_seed(1)
        selector = CandidateSelector(
            hidden_size=4,
            vocab_size=16,
            state_rank=3,
            top_k=4,
            initializer_range=0.2,
        )
        candidate_ids = torch.randint(0, 16, (2, 3, 4))
        unary_logits = torch.randn(2, 3, 4)
        hidden_states = torch.randn(2, 3, 4)
        anchor_ids = torch.tensor([2, 7])

        lattice = selector.build_lattice(
            candidate_ids=candidate_ids,
            unary_logits=unary_logits,
            hidden_states=hidden_states,
            anchor_token_ids=anchor_ids,
        )

        predecessor_ids = anchor_ids
        for position in range(candidate_ids.shape[1]):
            realized = selector.score_candidates(
                candidate_ids=candidate_ids[:, position],
                unary_logits=unary_logits[:, position],
                hidden_states=hidden_states[:, position],
                predecessor_ids=predecessor_ids,
            )
            if position == 0:
                expected = lattice[:, position, 0]
            else:
                previous_candidates = candidate_ids[:, position - 1]
                previous_index = (
                    previous_candidates.eq(predecessor_ids.unsqueeze(-1))
                    .long()
                    .argmax(dim=-1)
                )
                expected = lattice[:, position].gather(
                    1,
                    previous_index[:, None, None].expand(-1, 1, selector.top_k),
                )[:, 0]
            torch.testing.assert_close(realized, expected)
            selected = realized.argmax(dim=-1, keepdim=True)
            predecessor_ids = candidate_ids[:, position].gather(1, selected)[:, 0]

    def test_selector_loss_masks_targets_outside_strict_topk(self):
        class Draft(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))
                self.candidate_selector = CandidateSelector(
                    hidden_size=4,
                    vocab_size=4,
                    state_rank=2,
                    top_k=2,
                    initializer_range=0.02,
                )

            @staticmethod
            def transform_unary_logits(logits):
                return logits.float()

        draft = Draft()
        with torch.no_grad():
            draft.candidate_selector.predecessor_codebook.zero_()
            draft.candidate_selector.successor_codebook.zero_()
            draft.candidate_selector.hidden_projection.weight.zero_()
        model = OnlineDFlashModel(
            draft_model=draft,
            target_lm_head=nn.Identity(),
            target_embed_tokens=nn.Embedding(4, 4),
            mask_token_id=3,
            block_size=2,
            attention_backend="eager",
            selector_loss_alpha=1.0,
        )
        hidden = torch.tensor([[[[0.0, 0.0, 0.0, 0.0], [0.0, 3.0, 2.0, 1.0]]]])
        targets = torch.tensor([[[0, 0]]])
        weights = torch.tensor([[[0.0, 1.0]]])
        predecessors = torch.tensor([[[0, 1]]])

        terms = model._dflash_objective_chunk_terms(
            hidden,
            targets,
            weights,
            predecessors,
        )

        base_num = terms[0]
        selector_num = terms[6]
        selector_den = terms[10]
        covered = terms[11]
        base_ce = torch.nn.functional.cross_entropy(
            hidden[0, 0, 1].unsqueeze(0),
            torch.tensor([0]),
        )
        torch.testing.assert_close(base_num, base_ce)
        self.assertEqual(selector_num.item(), 0.0)
        self.assertEqual(selector_den.item(), 0.0)
        self.assertEqual(covered.item(), 0.0)

        # When the target is covered, train the selector over exactly the same
        # strict unary top-k candidate set used by serving.
        covered_targets = torch.tensor([[[0, 2]]])
        covered_terms = model._dflash_objective_chunk_terms(
            hidden,
            covered_targets,
            weights,
            predecessors,
        )
        covered_base_ce = torch.nn.functional.cross_entropy(
            hidden[0, 0, 1].unsqueeze(0),
            torch.tensor([2]),
        )
        selector_ce = torch.nn.functional.cross_entropy(
            torch.tensor([[3.0, 2.0]]),
            torch.tensor([1]),
        )
        torch.testing.assert_close(covered_terms[0], covered_base_ce)
        torch.testing.assert_close(covered_terms[6], selector_ce)
        self.assertEqual(covered_terms[10].item(), 1.0)
        self.assertEqual(covered_terms[11].item(), 1.0)

        # D-PACE uses the unary target probability to derive one detached
        # position weight, and that same weight must scale both objectives.
        model.loss_type = "dpace"
        model.dpace_alpha = 0.5
        dpace_terms = model._dflash_objective_chunk_terms(
            hidden,
            covered_targets,
            weights,
            predecessors,
        )
        unary_probability = torch.exp(-covered_base_ce)
        dpace_weight = 0.5 * unary_probability + 0.5
        torch.testing.assert_close(dpace_terms[0], covered_base_ce * dpace_weight)
        torch.testing.assert_close(dpace_terms[6], selector_ce * dpace_weight)
        torch.testing.assert_close(dpace_terms[10], dpace_weight)

    def test_selector_keeps_ce_when_base_uses_tv(self):
        class Draft(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))
                self.candidate_selector = CandidateSelector(
                    hidden_size=4,
                    vocab_size=4,
                    state_rank=2,
                    top_k=2,
                    initializer_range=0.02,
                )

            @staticmethod
            def transform_unary_logits(logits):
                return logits.float()

        draft = Draft()
        with torch.no_grad():
            draft.candidate_selector.predecessor_codebook.zero_()
            draft.candidate_selector.successor_codebook.zero_()
            draft.candidate_selector.hidden_projection.weight.zero_()
        model = OnlineDFlashModel(
            draft_model=draft,
            target_lm_head=nn.Identity(),
            target_embed_tokens=nn.Embedding(4, 4),
            mask_token_id=3,
            block_size=2,
            attention_backend="eager",
            selector_loss_alpha=1.0,
            lk_loss_type="tv",
        )
        output_hidden = torch.tensor(
            [[[0.0, 0.0, 0.0, 0.0], [0.0, 3.0, 2.0, 1.0]]]
        )
        model._forward_draft_blocks = lambda **_kwargs: (
            torch.tensor([[0]]),
            torch.tensor([[True]]),
            output_hidden,
        )

        loss, _accuracy, metrics = model(
            input_ids=torch.tensor([[0, 2]]),
            hidden_states=torch.zeros(1, 2, 4),
            loss_mask=torch.ones(1, 2),
        )

        target_probability = output_hidden[0, 1].softmax(dim=-1)[2]
        selector_ce = torch.nn.functional.cross_entropy(
            torch.tensor([[3.0, 2.0]]),
            torch.tensor([1]),
        )
        torch.testing.assert_close(loss, (1.0 - target_probability) + selector_ce)
        selector_num, selector_den = metrics["ratio_metrics"]["selector_loss"]
        torch.testing.assert_close(selector_num / selector_den, selector_ce)

    def test_selector_objective_backpropagates_to_all_selector_factors(self):
        class Draft(nn.Module):
            def __init__(self):
                super().__init__()
                self.candidate_selector = CandidateSelector(
                    hidden_size=4,
                    vocab_size=4,
                    state_rank=2,
                    top_k=2,
                    initializer_range=0.2,
                )

            @staticmethod
            def transform_unary_logits(logits):
                return logits.float()

        draft = Draft()
        with torch.no_grad():
            draft.candidate_selector.successor_codebook.normal_(std=0.2)
        model = OnlineDFlashModel(
            draft_model=draft,
            target_lm_head=nn.Identity(),
            target_embed_tokens=nn.Embedding(4, 4),
            mask_token_id=3,
            block_size=2,
            attention_backend="eager",
        )
        hidden = torch.tensor(
            [[[[0.0, 0.0, 0.0, 0.0], [0.0, 3.0, 2.0, 1.0]]]],
            requires_grad=True,
        )
        terms = model._dflash_objective_chunk_terms(
            hidden,
            torch.tensor([[[0, 2]]]),
            torch.tensor([[[0.0, 1.0]]]),
            torch.tensor([[[0, 1]]]),
        )
        (terms[0] + terms[6]).backward()

        selector = draft.candidate_selector
        for parameter in (
            selector.predecessor_codebook,
            selector.successor_codebook,
            selector.hidden_projection.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0)


class DFlashSelectorScheduleTest(unittest.TestCase):
    def test_warmup_then_linear_ramp_reaches_configured_alpha(self):
        model = SimpleNamespace(
            selector_loss_alpha=0.9,
            selector_warmup_ratio=0.2,
            selector_ramp_ratio=0.3,
        )
        strategy = DFlashTrainStrategy(model)

        self.assertEqual(
            strategy._selector_loss_alpha(StepContext(global_step=0, total_steps=10)),
            0.0,
        )
        self.assertEqual(
            strategy._selector_loss_alpha(StepContext(global_step=1, total_steps=10)),
            0.0,
        )
        self.assertAlmostEqual(
            strategy._selector_loss_alpha(StepContext(global_step=2, total_steps=10)),
            0.3,
        )
        self.assertAlmostEqual(
            strategy._selector_loss_alpha(StepContext(global_step=3, total_steps=10)),
            0.6,
        )
        self.assertAlmostEqual(
            strategy._selector_loss_alpha(StepContext(global_step=4, total_steps=10)),
            0.9,
        )

    def test_missing_schedule_context_preserves_configured_alpha(self):
        model = SimpleNamespace(
            selector_loss_alpha=0.75,
            selector_warmup_ratio=0.2,
            selector_ramp_ratio=0.3,
        )

        self.assertEqual(DFlashTrainStrategy(model)._selector_loss_alpha(None), 0.75)


if __name__ == "__main__":
    unittest.main(verbosity=2)
