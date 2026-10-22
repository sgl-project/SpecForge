"""DPARD reference tests adapted from SpecForge PR #973 (25899d3)."""
import copy
import sys
import unittest
from unittest import mock
import torch
import test_dflash_losses as fixtures
from specforge.core.dpard import dpard_token_terms


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


    def test_dspark_objective_chunking_and_gradient(self):
        logits, ids, mask, hidden, anchors, keep = fixtures._sample_tensors()
        torch.manual_seed(19)
        head = torch.nn.Linear(4, logits.shape[-1], bias=False).double()
        target_hidden = torch.randn_like(hidden, requires_grad=True)
        targets, valid = fixtures._dspark_targets_and_mask(ids, mask, anchors, keep, logits.shape[2])
        def build(chunk, **kwargs):
            return fixtures._make_dspark_model(
                logits, anchors, keep, lm_head=copy.deepcopy(head),
                draft_model=fixtures._LearnableDSparkDraft(4).double(),
                dspark_objective='dpard', dspark_confidence_head_alpha=0.0,
                objective_chunk_blocks=chunk, **kwargs,
            )
        baseline = build(0)
        signal = baseline.draft_model.signal
        draft_logits = baseline.lm_head(signal).expand(*targets.shape, -1)
        indices = (anchors[..., None] + torch.arange(logits.shape[2])).clamp(max=ids.shape[1]-2)
        teacher = torch.gather(target_hidden.detach(), 1, indices.flatten(1)[..., None].expand(-1, -1, 4))
        teacher_logits = baseline.lm_head(teacher).reshape_as(draft_logits)
        renyi, credit = dpard_token_terms(draft_logits, teacher_logits, valid)
        expected = (renyi * credit).sum() / valid.sum()
        expected_gradient = torch.autograd.grad(expected, signal)[0]
        for chunk in (0, 1, 2):
            model = build(chunk, dspark_ce_loss_alpha=42.0, dspark_l1_loss_alpha=0.0, loss_decay_gamma=0.3)
            loss, _, metrics = model(ids, hidden, mask, target_last_hidden_states=target_hidden)
            torch.testing.assert_close(loss, expected, check_dtype=False)
            loss.backward()
            torch.testing.assert_close(model.draft_model.signal.grad, expected_gradient, atol=1e-7, rtol=2e-5)
            self.assertIsNone(target_hidden.grad)
            num, den = metrics['ratio_metrics']['dpard_loss']
            torch.testing.assert_close(num/den, expected)
        with mock.patch('torch.distributed.is_initialized', return_value=True), \
             mock.patch('torch.distributed.get_world_size', return_value=2), \
             mock.patch('torch.distributed.all_reduce', side_effect=lambda value, **kwargs: value.add_(7.0)):
            loss, _, _ = baseline(ids, hidden, mask, target_last_hidden_states=target_hidden)
        torch.testing.assert_close(loss, expected * valid.sum() * 2 / (valid.sum() + 7), check_dtype=False)
        with self.assertRaisesRegex(ValueError, 'target_last_hidden_states'):
            build(0, dspark_l1_loss_alpha=0.0)(ids, hidden, mask)

    def test_confidence_and_diagnostic_metrics_unchanged(self):
        logits, ids, mask, hidden, anchors, keep = fixtures._sample_tensors()
        head = torch.nn.Linear(4, logits.shape[-1], bias=False).double()
        outputs = []
        for objective in ('ce_l1', 'dpard'):
            model = fixtures._make_dspark_model(
                logits, anchors, keep, lm_head=copy.deepcopy(head),
                draft_model=fixtures._LearnableDSparkDraft(4).double(),
                dspark_objective=objective, loss_decay_gamma=4.0,
            )
            outputs.append(model(ids, hidden, mask, target_last_hidden_states=hidden))
        old, new = [result[2]['ratio_metrics'] for result in outputs]
        for name, pair in old.items():
            for before, after in zip(pair, new[name]):
                torch.testing.assert_close(before, after)
        expected = new['dpard_loss'][0]/new['dpard_loss'][1]
        expected += new['confidence_loss'][0]/new['confidence_loss'][1]
        torch.testing.assert_close(outputs[1][0], expected)
        outputs[1][0].backward()
        self.assertTrue(torch.isfinite(model.draft_model.signal.grad).all())

    def test_configuration_wiring_and_resume_contract(self):
        from specforge.config import Config
        from specforge.algorithms import model_providers
        from specforge.algorithms.dspark.providers import resume_contract
        from types import SimpleNamespace
        raw = dict(model=dict(target_model_path='unused', draft_model_config='unused'),
                   data=dict(hidden_states_path='unused'),
                   training=dict(strategy='dspark', dspark_objective='dpard', dspark_dpard_rho=0.3))
        cfg = Config.model_validate(raw)
        common = dict(draft_model=fixtures._FixedDraft(4), target_lm_head=torch.nn.Linear(4, 13),
                      target_embed_tokens=torch.nn.Embedding(13, 4), mask_token_id=0,
                      block_size=5, attention_backend='sdpa')
        with mock.patch.object(model_providers, '_build_dflash_family_model',
                side_effect=lambda cfg, draft, tokenizer, factory: factory(common)), \
             mock.patch.dict(sys.modules, {'specforge.algorithms.common.dflash_family_model': fixtures._dflash_module}):
            model = model_providers.build_dspark_model(cfg, None, None, None, None)
        self.assertEqual(model.dspark_objective, 'dpard')
        self.assertEqual(model.dspark_dpard_rho, 0.3)
        draft = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=2),
                                target_layer_ids=[1, 3])
        # The local version also records attention layout; use its default helper.
        with mock.patch('specforge.modeling.draft.dflash.dflash_attention_resume_contract',
                        return_value={}, create=True):
            contract = resume_contract(cfg, draft, model)
            self.assertEqual(contract['dspark_objective'], 'dpard')
            self.assertEqual(contract['dspark_dpard_rho'], 0.3)
            self.assertEqual(contract['dspark_dpard_normalization'], 'valid_token_mean')
            model.dspark_objective = 'ce_l1'
            self.assertEqual(resume_contract(cfg, draft, model)['dspark_objective'], 'ce_l1')
        for change in (dict(strategy='dflash'), dict(loss_type='dpace'),
                       dict(lk_loss_type='lambda'), dict(dspark_dpard_rho=1.1)):
            invalid = copy.deepcopy(raw)
            invalid['training'].update(change)
            with self.assertRaises(ValueError):
                Config.model_validate(invalid)
        for change in (dict(loss_type='dpace'), dict(dspark_dpard_rho=-0.1)):
            with self.assertRaises(ValueError):
                fixtures.OnlineDSparkModel(**common, dspark_objective='dpard', **change)

    def test_real_dspark_forward_backward_and_bf16_chunking(self):
        from transformers import Qwen3Config
        from specforge.modeling.draft.dspark import DSparkDraftModel
        from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
        config = Qwen3Config(
            architectures=['DSparkDraftModel'], hidden_size=128,
            intermediate_size=256, num_attention_heads=4, num_key_value_heads=2,
            num_hidden_layers=2, num_target_layers=4, head_dim=32, vocab_size=64,
            block_size=4, layer_types=['full_attention']*2,
            dflash_config=dict(projector_type='dspark', target_layer_ids=[0, 2],
                mask_token_id=63, markov_head_type='vanilla', markov_rank=8,
                enable_confidence_head=False, confidence_head_alpha=0.0),
            attn_implementation='sdpa',
        )
        cases = [('cpu', torch.float32)]
        if torch.cuda.is_available():
            cases.append(('cuda', torch.bfloat16))
        for device, dtype in cases:
            with self.subTest(device=device, dtype=dtype):
                torch.manual_seed(42)
                baseline = OnlineDSparkModel(
                    DSparkDraftModel(config), torch.nn.Linear(128, 64, bias=False),
                    torch.nn.Embedding(64, 128), mask_token_id=63, block_size=4,
                    attention_backend='sdpa', num_anchors=2, dspark_objective='dpard',
                    dspark_confidence_head_alpha=0.0, objective_chunk_blocks=0,
                ).to(device=device, dtype=dtype)
                chunked = copy.deepcopy(baseline)
                chunked.objective_chunk_blocks = 1
                tokens = torch.randint(0, 62, (2, 12), device=device)
                hidden = torch.randn(2, 12, 256, device=device, dtype=dtype)
                target = torch.randn(2, 12, 128, device=device, dtype=dtype, requires_grad=True)
                outputs = []
                for model in (baseline, chunked):
                    torch.manual_seed(71)
                    loss, _, metrics = model(tokens, hidden, torch.ones_like(tokens),
                                              target_last_hidden_states=target)
                    loss.backward()
                    outputs.append(loss)
                    self.assertTrue(torch.isfinite(loss))
                    self.assertIn('dpard_loss', metrics['ratio_metrics'])
                self.assertIsNone(target.grad)
                tolerance = dict(atol=0.002, rtol=0.05) if dtype == torch.bfloat16 else dict(atol=1e-6, rtol=1e-4)
                torch.testing.assert_close(outputs[0], outputs[1], **tolerance)
                gradients = 0
                for (name, original), (_, actual) in zip(baseline.named_parameters(), chunked.named_parameters()):
                    if original.grad is not None:
                        gradients += 1
                        self.assertTrue(torch.isfinite(actual.grad).all(), name)
                        torch.testing.assert_close(original.grad, actual.grad, **tolerance, msg=name)
                self.assertGreater(gradients, 0)


if __name__ == '__main__':
    unittest.main()
