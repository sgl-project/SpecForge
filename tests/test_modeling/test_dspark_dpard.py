import copy
import unittest
from unittest import mock

import torch

from specforge.algorithms.dspark.providers import resume_contract
from specforge.algorithms.dspark_kv.providers import resume_contract as kv_contract
from specforge.algorithms.model_providers import build_dspark_model
from specforge.config import Config
from specforge.core.dpard import dpard_token_terms
from specforge.modeling.draft.dspark import DSparkDraftModel
from tests.test_modeling.test_dspark_target_kv import config, wrapper


def build_model(conditioning="kv", objective="dpard"):
    draft_config = config()
    draft_config.dflash_config["enable_confidence_head"] = False
    if conditioning == "hidden":
        draft_config.dflash_config = {
            name: value
            for name, value in draft_config.dflash_config.items()
            if not name.startswith("target_kv_") and name != "conditioning_source"
        }
    model = wrapper(DSparkDraftModel(draft_config))
    model.dspark_objective = objective
    model.dspark_confidence_head_alpha = 0.0
    return model


class DPardTest(unittest.TestCase):
    def test_formula_mask_detachment_and_analytic_gradient(self):
        torch.manual_seed(12)
        draft = torch.randn(2, 4, 7, requires_grad=True)
        target = torch.randn_like(draft, requires_grad=True)
        valid = torch.tensor([[True, True, True, False], [True, False, True, False]])
        for rho in (0.0, 0.5, 1.0):
            with self.subTest(rho=rho):
                renyi, credit = dpard_token_terms(draft, target, valid, rho)
                target_prob = target.detach().softmax(-1)
                draft_prob = draft.detach().softmax(-1)
                affinity = (target_prob * draft_prob).sqrt().sum(-1)
                torch.testing.assert_close(
                    renyi, -2 * affinity.log(), atol=1e-6, rtol=1e-5
                )
                acceptance = torch.minimum(target_prob, draft_prob).sum(-1)
                expected_credit = torch.zeros_like(credit)
                for sequence in range(2):
                    reach = 1.0
                    prefixes = []
                    for position in range(4):
                        if valid[sequence, position]:
                            reach *= (
                                rho + (1 - rho) * acceptance[sequence, position].item()
                            )
                        prefixes.append(reach if valid[sequence, position] else 0.0)
                    for position in range(4):
                        if valid[sequence, position]:
                            expected_credit[sequence, position] = sum(
                                prefixes[position:]
                            )
                torch.testing.assert_close(credit, expected_credit)
                self.assertFalse(credit.requires_grad)
                loss = (renyi * credit).sum() / valid.sum()
                actual, teacher_gradient = torch.autograd.grad(
                    loss, (draft, target), allow_unused=True
                )
                normalized_affinity = (
                    target_prob * draft_prob
                ).sqrt() / affinity.unsqueeze(-1)
                expected = (
                    (draft_prob - normalized_affinity)
                    * credit.unsqueeze(-1)
                    / valid.sum()
                )
                torch.testing.assert_close(actual, expected, atol=1e-7, rtol=2e-5)
                self.assertIsNone(teacher_gradient)

    def test_equal_distributions_extremes_and_empty_mask(self):
        for dtype in (torch.float32, torch.bfloat16):
            logits = torch.tensor(
                [[[1000.0, -1000.0], [-1000.0, 1000.0]]],
                dtype=dtype,
                requires_grad=True,
            )
            valid = torch.ones(1, 2, dtype=torch.bool)
            renyi, credit = dpard_token_terms(logits, logits.detach(), valid)
            torch.testing.assert_close(renyi, torch.zeros_like(renyi))
            torch.testing.assert_close(credit, torch.tensor([[2.0, 1.0]]))
            opposite, _ = dpard_token_terms(logits, -logits.detach(), valid)
            self.assertTrue(torch.isfinite(opposite).all())
            empty_renyi, empty_credit = dpard_token_terms(
                logits, -logits.detach(), ~valid
            )
            (empty_renyi * empty_credit).sum().backward()
            torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_chunked_loss_matches_reference_and_global_token_normalization(self):
        torch.manual_seed(19)
        model = build_model()
        model.loss_decay_gamma = 7.0
        hidden = torch.randn(2, 12, 16, requires_grad=True)
        target = torch.randn(2, 16, 16, requires_grad=True)
        target_ids = torch.randint(0, 31, (2, 3, 4))
        previous = torch.randint(0, 31, (2, 3, 4))
        indices = torch.arange(1, 13).reshape(1, 3, 4).expand(2, -1, -1)
        valid = torch.ones_like(target_ids, dtype=torch.bool)
        valid[0, 2] = False
        valid[1, 1:, 2:] = False
        arguments = dict(
            output_hidden=hidden,
            target_ids=target_ids,
            eval_mask=valid,
            prev_token_ids=previous,
            safe_label_indices=indices,
            target_last_hidden_states=target,
            anchor_positions=indices[:, :, 0] - 1,
            collect_detailed_metrics=True,
        )
        logits = model.draft_model.apply_logits_head(
            model.lm_head(hidden).reshape(2, 3, 4, 32),
            prev_token_ids=previous,
            hidden_states=hidden.reshape(2, 3, 4, 16),
        )
        teacher_logits = model.lm_head(target.detach()[:, :12]).reshape_as(logits)
        renyi, credit = dpard_token_terms(logits, teacher_logits, valid)
        expected = (renyi * credit).sum() / valid.sum()
        expected_gradient = torch.autograd.grad(expected, hidden, retain_graph=True)[0]
        for chunk in (0, 1, 2):
            model.objective_chunk_blocks = chunk
            loss, metrics = model._compute_dspark_loss(**arguments)
            torch.testing.assert_close(loss, expected)
            actual_gradient = torch.autograd.grad(loss, hidden)[0]
            torch.testing.assert_close(
                actual_gradient, expected_gradient, atol=1e-7, rtol=2e-5
            )
            numerator, denominator = metrics["ratio_metrics"]["dpard_loss"]
            torch.testing.assert_close(numerator / denominator, expected)
        model.dspark_ce_loss_alpha = 42.0
        model.dspark_l1_loss_alpha = 0.0
        model.loss_decay_gamma = 0.3
        loss, _ = model._compute_dspark_loss(**arguments)
        torch.testing.assert_close(loss, expected)
        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=2),
            mock.patch(
                "torch.distributed.all_reduce",
                side_effect=lambda value, **kwargs: value.add_(7.0),
            ),
        ):
            loss, _ = model._compute_dspark_loss(**arguments)
        torch.testing.assert_close(loss, expected * valid.sum() * 2 / (valid.sum() + 7))
        arguments["target_last_hidden_states"] = None
        with self.assertRaisesRegex(ValueError, "target_last_hidden_states"):
            model._compute_dspark_loss(**arguments)

    def test_hidden_and_kv_full_forward_backward(self):
        for conditioning in ("hidden", "kv"):
            with self.subTest(conditioning=conditioning):
                torch.manual_seed(6)
                model = build_model(conditioning)
                tokens = torch.randint(0, 31, (2, 9))
                target = torch.randn(2, 9, 16, requires_grad=True)
                kwargs = dict(
                    input_ids=tokens,
                    loss_mask=torch.ones_like(tokens),
                    hidden_states=(
                        torch.randn(2, 9, 32) if conditioning == "hidden" else None
                    ),
                    target_last_hidden_states=target,
                )
                if conditioning == "kv":
                    kwargs["target_kv"] = torch.randn(2, 9, 2, 2, 2, 4)
                loss, _, metrics = model(**kwargs)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertIsNone(target.grad)
                self.assertIn("dpard_loss", metrics["ratio_metrics"])
                gradients = [
                    parameter.grad
                    for parameter in model.draft_model.parameters()
                    if parameter.grad is not None
                ]
                self.assertTrue(gradients)
                self.assertTrue(
                    all(torch.isfinite(gradient).all() for gradient in gradients)
                )
                self.assertGreater(
                    sum(gradient.abs().sum().item() for gradient in gradients), 0
                )

    def test_configuration_provider_and_resume_contract(self):
        for strategy in ("dspark", "dspark_kv"):
            raw = dict(
                model=dict(target_model_path="unused"),
                data=dict(hidden_states_path="unused"),
                training=dict(
                    strategy=strategy, dspark_objective="dpard", dspark_dpard_rho=0.3
                ),
            )
            cfg = Config.model_validate(raw)
            reference = build_model()
            options = dict(
                draft_model=reference.draft_model,
                target_lm_head=reference.lm_head,
                target_embed_tokens=reference.embed_tokens,
                mask_token_id=31,
                block_size=4,
            )
            with mock.patch(
                "specforge.algorithms.model_providers._build_dflash_family_model",
                side_effect=lambda config, draft, tokenizer, factory: factory(options),
            ):
                model = build_dspark_model(cfg, reference.draft_model, None, None, None)
            contract = (kv_contract if strategy == "dspark_kv" else resume_contract)(
                None, model.draft_model, model
            )
            self.assertEqual(contract[f"{strategy}_objective"], "dpard")
            self.assertEqual(contract[f"{strategy}_dpard_rho"], 0.3)
            for change in (
                dict(strategy="dflash"),
                dict(loss_type="dpace"),
                dict(lk_loss_type="lambda"),
                dict(dspark_dpard_rho=1.1),
            ):
                invalid = copy.deepcopy(raw)
                invalid["training"].update(change)
                with self.assertRaises(ValueError):
                    Config.model_validate(invalid)
        baseline = build_model(objective="ce_l1")
        contract = resume_contract(None, baseline.draft_model, baseline)
        self.assertNotIn("dspark_objective", contract)

    def test_confidence_auxiliary_is_unchanged(self):
        torch.manual_seed(71)
        baseline = wrapper(DSparkDraftModel(config()), chunk=0)
        baseline.loss_decay_gamma = 2.0
        changed = copy.deepcopy(baseline)
        changed.dspark_objective = "dpard"
        tokens = torch.randint(0, 31, (2, 9))
        kwargs = dict(
            input_ids=tokens,
            hidden_states=None,
            loss_mask=torch.ones_like(tokens),
            target_last_hidden_states=torch.randn(2, 9, 16),
            target_kv=torch.randn(2, 9, 2, 2, 2, 4),
        )
        outputs = []
        for model in (baseline, changed):
            torch.manual_seed(38)
            outputs.append(model(**kwargs))
        for name in (
            "confidence_loss",
            "confidence_abs_error",
            "ce_loss",
            "l1_loss",
            "acc",
        ):
            for actual, expected in zip(
                outputs[1][2]["ratio_metrics"][name],
                outputs[0][2]["ratio_metrics"][name],
            ):
                torch.testing.assert_close(actual, expected)
        metrics = outputs[1][2]["ratio_metrics"]
        expected = metrics["dpard_loss"][0] / metrics["dpard_loss"][1]
        expected += (
            changed.dspark_confidence_head_alpha
            * metrics["confidence_loss"][0]
            / metrics["confidence_loss"][1]
        )
        torch.testing.assert_close(outputs[1][0], expected)
        outputs[1][0].backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in changed.draft_model.confidence_head.parameters()
            )
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_bf16_gpu_chunking_and_optimized_path(self):
        for conditioning in ("hidden", "kv"):
            with self.subTest(conditioning=conditioning):
                torch.manual_seed(42)
                baseline = build_model(conditioning).cuda().bfloat16()
                baseline.objective_chunk_blocks = 0
                optimized = copy.deepcopy(baseline)
                optimized.objective_chunk_blocks = 1
                if hasattr(optimized, "compile_objective"):
                    optimized.flatten_projection = True
                    optimized.cache_projection = True
                    optimized.compile_objective = True
                tokens = torch.randint(0, 31, (2, 9), device="cuda")
                kwargs = dict(
                    input_ids=tokens,
                    loss_mask=torch.ones_like(tokens),
                    target_last_hidden_states=torch.randn(
                        2, 9, 16, device="cuda"
                    ).bfloat16(),
                    hidden_states=(
                        torch.randn(2, 9, 32, device="cuda").bfloat16()
                        if conditioning == "hidden"
                        else None
                    ),
                )
                if conditioning == "kv":
                    kwargs["target_kv"] = torch.randn(
                        2, 9, 2, 2, 2, 4, device="cuda"
                    ).bfloat16()
                outputs = []
                for model in (baseline, optimized):
                    torch.manual_seed(8)
                    output = model(**kwargs)
                    output[0].backward()
                    outputs.append(output)
                torch.testing.assert_close(
                    outputs[0][0], outputs[1][0], rtol=0.01, atol=0.002
                )
                self.assertTrue(torch.isfinite(outputs[1][0]))
                for (name, original), (_, actual) in zip(
                    baseline.named_parameters(), optimized.named_parameters()
                ):
                    if original.grad is not None:
                        self.assertIsNotNone(actual.grad, name)
                        torch.testing.assert_close(
                            actual.grad, original.grad, rtol=0.05, atol=0.002, msg=name
                        )


if __name__ == "__main__":
    unittest.main()
