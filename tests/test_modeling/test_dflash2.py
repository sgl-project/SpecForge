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
            mask_token_id=31,
            num_anchors=8,
            selector_loss_alpha=0.75,
        )

        contract = resume_contract(None, model, training_model)

        self.assertEqual(contract["dflash2_conv_kernel_size"], 2)
        self.assertEqual(contract["dflash2_conv_group_size"], 4)
        self.assertEqual(contract["dflash2_selector_rank"], 4)
        self.assertEqual(contract["dflash2_selector_top_k"], 3)
        self.assertEqual(contract["dflash2_selector_loss_alpha"], 0.75)

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

    def test_selector_loss_uses_gold_replacement_when_target_is_not_topk(self):
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

        combined_num, _, _, _, selector_num, selector_den, _, covered = terms
        base_ce = torch.nn.functional.cross_entropy(
            hidden[0, 0, 1].unsqueeze(0),
            torch.tensor([0]),
        )
        selector_ce = torch.nn.functional.cross_entropy(
            torch.tensor([[3.0, 0.0]]),
            torch.tensor([1]),
        )
        torch.testing.assert_close(selector_num, selector_ce)
        torch.testing.assert_close(combined_num, base_ce + selector_ce)
        self.assertEqual(selector_den.item(), 1.0)
        self.assertEqual(covered.item(), 0.0)

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
        terms[0].backward()

        selector = draft.candidate_selector
        for parameter in (
            selector.predecessor_codebook,
            selector.successor_codebook,
            selector.hidden_projection.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
