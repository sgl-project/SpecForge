import copy
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn.functional as functional

from specforge.algorithms.dspark.providers import resume_contract
from specforge.algorithms.model_providers import build_dspark_model
from specforge.config import Config
from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.draft.dspark import DSparkDraftModel
from tests.test_modeling.test_dspark_causal import build_model
from tests.test_modeling.test_dspark_target_kv import config, wrapper


class DSparkConvDPaceTest(unittest.TestCase):
    def test_kv_conv_dpace_and_dpard_full_backward(self):
        for objective in ("dpace", "dpard"):
            with self.subTest(objective=objective):
                torch.manual_seed(42)
                draft_config = config()
                draft_config.is_causal = True
                draft_config.dflash_config.update(
                    conv_kernel_size=2,
                    conv_group_size=4,
                    enable_confidence_head=False,
                )
                model = wrapper(DSparkDraftModel(draft_config))
                model.dspark_confidence_head_alpha = 0.0
                if objective == "dpard":
                    model.dspark_objective = "dpard"
                else:
                    model.loss_type = "dpace"
                tokens = torch.randint(0, 30, (2, 16))
                loss, _accuracy, _metrics = model(
                    input_ids=tokens,
                    hidden_states=None,
                    loss_mask=torch.ones_like(tokens),
                    target_kv=torch.randn(2, 16, 2, 2, 2, 4),
                    target_last_hidden_states=torch.randn(2, 16, 16),
                )
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                convolution_parameters = [
                    parameter
                    for name, parameter in model.draft_model.named_parameters()
                    if "_conv." in name
                ]
                self.assertTrue(convolution_parameters)
                for parameter in convolution_parameters:
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_identity_conv_preserves_hidden_and_kv_models(self):
        for conditioning in ("hidden", "kv"):
            with self.subTest(conditioning=conditioning):
                torch.manual_seed(14)
                kv_config = config()
                kv_config.is_causal = True
                baseline = (
                    build_model()
                    if conditioning == "hidden"
                    else wrapper(DSparkDraftModel(kv_config))
                )
                conv_config = copy.deepcopy(baseline.draft_model.config)
                conv_config.dflash_config.update(conv_kernel_size=2, conv_group_size=4)
                changed = copy.deepcopy(baseline)
                changed.draft_model = DSparkDraftModel(conv_config)
                missing, unexpected = changed.draft_model.load_state_dict(
                    baseline.draft_model.state_dict(), strict=False
                )
                self.assertFalse(unexpected)
                self.assertTrue(missing)
                self.assertTrue(all("_conv." in name for name in missing))
                width = baseline.embed_tokens.embedding_dim
                vocabulary = baseline.embed_tokens.num_embeddings
                tokens = torch.randint(0, vocabulary - 1, (2, 16))
                kwargs = dict(
                    input_ids=tokens,
                    loss_mask=torch.ones_like(tokens),
                    target_last_hidden_states=torch.randn(2, 16, width),
                )
                if conditioning == "hidden":
                    kwargs["hidden_states"] = torch.randn(2, 16, 256)
                else:
                    kwargs.update(
                        hidden_states=None, target_kv=torch.randn(2, 16, 2, 2, 2, 4)
                    )
                outputs = []
                for model in (baseline, changed):
                    torch.manual_seed(83)
                    outputs.append(model(**kwargs))
                torch.testing.assert_close(outputs[0][0], outputs[1][0])
                torch.testing.assert_close(outputs[0][1], outputs[1][1])
                outputs[1][0].backward()
                for name, parameter in changed.draft_model.named_parameters():
                    if "_conv." in name:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                        self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
                with tempfile.TemporaryDirectory() as directory:
                    changed.draft_model.save_pretrained(directory)
                    restored = AutoDraftModel.from_pretrained(
                        directory, attn_implementation="sdpa"
                    )
                    for name, value in changed.draft_model.state_dict().items():
                        torch.testing.assert_close(
                            restored.state_dict()[name], value, rtol=0, atol=0
                        )

    def test_dpace_matches_independent_markov_ce_l1_reference(self):
        torch.manual_seed(7)
        model = build_model().double()
        model.loss_type = "dpace"
        model.dpace_alpha = 0.5
        hidden = torch.randn(2, 3, 4, 128, dtype=torch.float64, requires_grad=True)
        target_hidden = torch.randn_like(hidden)
        target_ids = torch.randint(0, 64, (2, 3, 4))
        previous = torch.randint(0, 64, (2, 3, 4))
        valid = torch.tensor(
            [
                [[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]],
                [[1, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
            dtype=torch.bool,
        )
        scale = model._dspark_loss_weight_mask(valid)
        actual = model._dspark_objective_chunk_terms(
            hidden, previous, target_ids, scale, valid, target_hidden
        )
        logits = model.draft_model.apply_logits_head(
            model.lm_head(hidden), prev_token_ids=previous, hidden_states=hidden
        )
        ce = functional.cross_entropy(
            logits.flatten(0, 2), target_ids.flatten(), reduction="none"
        ).reshape_as(target_ids)
        l1 = (
            (
                logits.float().softmax(-1)
                - model.lm_head(target_hidden).float().softmax(-1)
            )
            .abs()
            .sum(-1)
        )
        weights = torch.zeros_like(ce)
        with torch.no_grad():
            probability = torch.exp(-ce.float())
            for sequence in range(2):
                anchors = valid[sequence].any(-1).sum()
                for block in range(3):
                    survival = 1.0
                    prefixes = []
                    for position in range(4):
                        if valid[sequence, block, position]:
                            survival *= (
                                0.5
                                + 0.5 * probability[sequence, block, position].item()
                            )
                            prefixes.append(survival)
                        else:
                            prefixes.append(0.0)
                    for position in range(4):
                        if valid[sequence, block, position]:
                            weights[sequence, block, position] = (
                                sum(prefixes[position:]) / anchors
                            )
        expected_ce, expected_l1 = (ce * weights).sum(), (l1 * weights).sum()
        torch.testing.assert_close(actual[0], expected_ce, rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(
            actual[1].double(), expected_l1, rtol=2e-6, atol=2e-6
        )
        actual_gradient = torch.autograd.grad(
            0.1 * actual[0] + 0.9 * actual[1], hidden, retain_graph=True
        )[0]
        expected_gradient = torch.autograd.grad(
            0.1 * expected_ce + 0.9 * expected_l1, hidden
        )[0]
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=2e-5, atol=1e-7
        )

    def test_dpace_chunking_and_accumulation_equivalence(self):
        torch.manual_seed(3)
        baseline = build_model().double()
        baseline.loss_type = "dpace"
        baseline.num_anchors = 3
        baseline.objective_chunk_blocks = 0
        target_ids = torch.randint(0, 64, (2, 3, 4))
        previous = torch.randint(0, 64, (2, 3, 4))
        hidden = torch.randn(2, 12, 128, dtype=torch.float64)
        target = torch.randn(2, 16, 128, dtype=torch.float64)
        indices = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]] * 2)
        valid = torch.ones_like(target_ids, dtype=torch.bool)
        valid[0, 2] = False
        valid[1, 1:, 2:] = False

        def evaluate(model, selection):
            return model._compute_dspark_loss(
                output_hidden=hidden[selection],
                target_ids=target_ids[selection],
                eval_mask=valid[selection],
                prev_token_ids=previous[selection],
                safe_label_indices=indices[selection],
                target_last_hidden_states=target[selection],
                anchor_positions=indices[selection, :, 0] - 1,
                collect_detailed_metrics=False,
            )[0]

        reference = evaluate(baseline, slice(None))
        reference.backward()
        for chunk in (1, 2):
            for microbatch in (False, True):
                model = copy.deepcopy(baseline)
                model.zero_grad(set_to_none=True)
                model.objective_chunk_blocks = chunk
                losses = (
                    [evaluate(model, slice(index, index + 1)) / 2 for index in range(2)]
                    if microbatch
                    else [evaluate(model, slice(None))]
                )
                for loss in losses:
                    loss.backward()
                torch.testing.assert_close(sum(losses), reference, rtol=1e-6, atol=1e-7)
                for (name, expected), (_, actual) in zip(
                    baseline.named_parameters(), model.named_parameters()
                ):
                    if expected.grad is not None:
                        torch.testing.assert_close(
                            actual.grad, expected.grad, rtol=2e-5, atol=1e-7, msg=name
                        )

    def test_configuration_and_resume_contract(self):
        reference = build_model()
        options = dict(
            draft_model=reference.draft_model,
            target_lm_head=reference.lm_head,
            target_embed_tokens=reference.embed_tokens,
            mask_token_id=63,
            block_size=4,
        )
        cfg = Config.model_validate(
            dict(
                model=dict(target_model_path="unused"),
                data=dict(hidden_states_path="unused"),
                training=dict(strategy="dspark", loss_type="dpace", dpace_alpha=0.3),
            )
        )
        with mock.patch(
            "specforge.algorithms.model_providers._build_dflash_family_model",
            side_effect=lambda config, draft, tokenizer, factory: factory(options),
        ):
            model = build_dspark_model(cfg, reference.draft_model, None, None, None)
        self.assertEqual(model.loss_type, "dpace")
        self.assertEqual(model.dpace_alpha, 0.3)
        contract = resume_contract(None, model.draft_model, model)
        self.assertEqual(contract["dspark_loss_type"], "dpace")
        self.assertEqual(contract["dspark_dpace_alpha"], 0.3)
        for field in ("conv_kernel_size", "conv_group_size"):
            cfg = copy.deepcopy(reference.draft_model.config)
            cfg.dflash_config[field] = 2
            with self.assertRaisesRegex(ValueError, "convolution requires integer"):
                DSparkDraftModel(cfg)


if __name__ == "__main__":
    unittest.main()
