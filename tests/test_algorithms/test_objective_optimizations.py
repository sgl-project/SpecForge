"""Optimized paths preserve loss and trainable gradients on composed models."""

import copy
import unittest

import torch
from torch import nn

from specforge.algorithms.objectives import build_objective_model
from specforge.modeling import build_draft_model
from tests.test_algorithms.test_objectives import configurations, training_batch
from tests.test_modeling.test_canonical_eagle3 import install_mapping, model_config


def evaluate(model, batch, teacher, options):
    model = copy.deepcopy(model)
    runtime = build_objective_model(
        model, target_head=copy.deepcopy(teacher), options=options
    )
    torch.manual_seed(31)
    result = runtime(batch, step=1, total_steps=10, collect_detailed_metrics=False)
    result.loss.backward()
    gradients = {
        name: p.grad.clone() for name, p in model.named_parameters() if p.requires_grad
    }
    return result.loss.detach(), gradients


class ObjectiveOptimizationTest(unittest.TestCase):
    def assert_equivalent(self, left, right, *, tolerance=3e-5):
        torch.testing.assert_close(left[0], right[0], atol=tolerance, rtol=tolerance)
        self.assertEqual(left[1].keys(), right[1].keys())
        for name in left[1]:
            with self.subTest(parameter=name):
                torch.testing.assert_close(
                    left[1][name], right[1][name], atol=tolerance, rtol=tolerance
                )

    def test_disabling_selector_freezes_it_but_warmup_keeps_its_graph(self):
        config = configurations()[1]
        for alpha in (0.0, 1.0):
            model = build_draft_model(config)
            runtime = build_objective_model(
                model,
                options=dict(
                    num_anchors=2,
                    selector_loss_alpha=alpha,
                    selector_warmup_steps=5,
                    fused_logits_loss=False,
                    teacher_metrics=False,
                ),
            )
            runtime(
                training_batch(config),
                step=0,
                total_steps=10,
                collect_detailed_metrics=False,
            ).loss.backward()
            for parameter in model.candidate_selector.parameters():
                self.assertEqual(parameter.requires_grad, alpha > 0)
                self.assertEqual(parameter.grad is not None, alpha > 0)

    def test_block_chunking_checkpointing_and_fused_losses(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for config in configurations()[:5]:
            model = build_draft_model(config, device=device)
            teacher = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False, device=device
            )
            if hasattr(model, "initialize_transfer_space"):
                model.embed_tokens.load_target_weight(torch.randn_like(teacher.weight))
                model.lm_head.load_target_weight(teacher.weight)
                model.initialize_transfer_space()
            batch = training_batch(config, device)
            variants = (
                [
                    {"loss_type": "dpace", "lk_loss_type": "lambda"},
                    {"lk_loss_type": "tv"},
                ]
                if "DFlash" in config.architecture
                else [{}]
            )
            for extra in variants:
                with self.subTest(architecture=config.architecture, options=extra):
                    options = dict(num_anchors=4, loss_decay_gamma=2.0, **extra)
                    reference = evaluate(
                        model,
                        batch,
                        teacher,
                        dict(
                            options,
                            objective_chunk_blocks=0,
                            objective_checkpointing=False,
                            fused_logits_loss=False,
                        ),
                    )
                    optimized = evaluate(
                        model,
                        batch,
                        teacher,
                        dict(
                            options,
                            objective_chunk_blocks=1,
                            objective_checkpointing=True,
                            fused_logits_loss=True,
                        ),
                    )
                    self.assert_equivalent(reference, optimized)

    def test_eagle_compact_teacher_and_trimmed_loss(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config = model_config()
        model = build_draft_model(config)
        install_mapping(model)
        model = model.to(device)
        teacher = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False, device=device
        )
        batch = training_batch(config, device)
        options = dict(ttt_length=3)
        reference = evaluate(model, batch, teacher, options)
        for extra in (
            {"trim_loss_positions": True},
            {"compact_teacher": True, "compact_teacher_chunk_size": 7},
        ):
            with self.subTest(options=extra):
                self.assert_equivalent(
                    reference, evaluate(model, batch, teacher, dict(options, **extra))
                )


if __name__ == "__main__":
    unittest.main()
