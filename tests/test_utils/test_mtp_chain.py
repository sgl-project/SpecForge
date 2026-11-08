"""MTP chain causality, gradient accumulation, and training metric regressions."""

import contextlib
import copy
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from specforge.core.mtp import OnlineMTPModel
from specforge.modeling.draft.mtp import Qwen3_5MTPDraftModel
from specforge.runtime.contracts import TrainBatch
from specforge.training.controller import TrainerCore
from specforge.training.strategies.base import MTPTrainStrategy


def model(steps=3, backend="eager"):
    torch.manual_seed(1701)
    config = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        max_position_embeddings=128,
        pad_token_id=0,
        rms_norm_eps=1e-6,
        attn_output_gate=True,
        partial_rotary_factor=0.5,
        rope_theta=10000000.0,
        mtp_config={"share_lm_head": True},
        tie_word_embeddings=False,
    )
    config._attn_implementation = backend
    draft = Qwen3_5MTPDraftModel(config)
    draft.embed_tokens.requires_grad_(False)
    draft.mtp.lm_head.requires_grad_(False)
    return OnlineMTPModel(draft, num_steps=steps, ploss_decay=0.8).eval()


def tensors():
    torch.manual_seed(43)
    ids, hidden, mask = (
        torch.randint(1, 64, (2, 8)),
        torch.randn(2, 8, 32),
        torch.ones(2, 8),
    )
    mask[:, :4] = 0
    mask[0, 6:] = 0
    ids[0, 6:] = 0
    hidden[0, 6:] = 0
    return ids, hidden, mask


def batch(ids, hidden, mask):
    return TrainBatch(
        sample_ids=[str(i) for i in range(len(ids))],
        strategy="mtp",
        tensors={
            "input_ids": ids,
            "target_last_hidden_states": hidden,
            "loss_mask": mask,
        },
    )


class CPUBackend:
    def __init__(self, current):
        self.model = current

    def accumulation_context(self, *, is_boundary):
        return contextlib.nullcontext()

    def backward(self, loss, *, is_boundary):
        loss.backward()

    def scale_gradients(self, factor):
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(factor)

    def step(self):
        return torch.tensor(0.0)


class MTPChainTest(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)

    def test_future_hidden_states_do_not_change_prefix(self):
        for backend in ("eager", "sdpa"):
            for steps in (1, 3):
                with self.subTest(backend=backend, steps=steps):
                    current = model(steps, backend)
                    ids, hidden, mask = tensors()
                    changed = hidden.clone()
                    changed[:, 1:] += 7 * torch.randn_like(changed[:, 1:])
                    outputs = []
                    handle = current.draft_model.mtp.register_forward_hook(
                        lambda _, args, result: outputs.append(result.detach().clone())
                    )
                    with torch.no_grad():
                        current(input_ids=ids, hidden_states=hidden, loss_mask=mask)
                        first = outputs[:]
                        outputs.clear()
                        current(input_ids=ids, hidden_states=changed, loss_mask=mask)
                    handle.remove()
                    self.assertEqual(len(first), steps)
                    self.assertEqual(len(outputs), steps)
                    for before, after in zip(first, outputs):
                        torch.testing.assert_close(
                            before[:, 0], after[:, 0], rtol=0, atol=0
                        )

    def test_accumulation_matches_full_batch_gradients_and_metrics(self):
        for backend in ("eager", "sdpa"):
            for steps in (1, 3):
                with self.subTest(backend=backend, steps=steps):
                    current = model(steps, backend)
                    split = copy.deepcopy(current)
                    ids, hidden, mask = tensors()
                    large = TrainerCore(
                        MTPTrainStrategy(current),
                        CPUBackend(current),
                        accumulation_steps=1,
                    )
                    micro = TrainerCore(
                        MTPTrainStrategy(split), CPUBackend(split), accumulation_steps=2
                    )
                    expected = large.train_step(batch(ids, hidden, mask))
                    micro.train_step(batch(ids[:1], hidden[:1], mask[:1]))
                    actual = micro.train_step(batch(ids[1:], hidden[1:], mask[1:]))
                    torch.testing.assert_close(
                        current.draft_model.mtp.fc.weight.grad,
                        split.draft_model.mtp.fc.weight.grad,
                        rtol=1e-5,
                        atol=1e-6,
                    )
                    keys = [
                        "loss",
                        "kl_loss",
                        "acc",
                        *[f"ploss_{i}" for i in range(steps)],
                        *[f"acc_{i}" for i in range(steps)],
                    ]
                    for key in keys:
                        self.assertAlmostEqual(
                            actual.metrics[key], expected.metrics[key], places=5
                        )
                    self.assertAlmostEqual(
                        actual.metrics["kl_loss"], actual.loss, places=5
                    )

    def test_unequal_supervision_is_rejected_before_forward(self):
        current = model()
        ids, hidden, mask = tensors()
        mask[0, 2] = 1
        with mock.patch.object(current, "forward", wraps=current.forward) as forward:
            with self.assertRaisesRegex(ValueError, "equal supervision counts"):
                MTPTrainStrategy(current).forward_loss(batch(ids, hidden, mask))
            forward.assert_not_called()

    def test_single_step_accepts_full_supervision(self):
        current = model(1)
        ids, hidden, mask = tensors()
        mask.fill_(1)
        result = MTPTrainStrategy(current).forward_loss(batch(ids, hidden, mask))
        result.loss.backward()
        self.assertTrue(torch.isfinite(current.draft_model.mtp.fc.weight.grad).all())

    def test_short_sequence_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "sequence length"):
            model()(
                input_ids=torch.ones(1, 3, dtype=torch.long),
                hidden_states=torch.zeros(1, 3, 32),
                loss_mask=torch.zeros(1, 3),
            )

    def test_chain_loss_matches_weighted_step_losses(self):
        ids, hidden, mask = tensors()
        loss, _, _, plosses, denoms = model()(
            input_ids=ids, hidden_states=hidden, loss_mask=mask
        )
        self.assertEqual(len(plosses), 3)
        self.assertEqual(len(denoms), 3)
        torch.testing.assert_close(
            loss.detach(), sum(0.8**i * value for i, value in enumerate(plosses))
        )
        self.assertTrue(all(not value.requires_grad for value in plosses))

    def test_cached_attention_matches_explicit_per_row_reference(self):
        attention = model().draft_model.mtp.layers[0].self_attn
        torch.manual_seed(44)
        query = torch.randn(2, 4, 5, 8)
        keys = [torch.randn_like(query) for _ in range(3)]
        values = [torch.randn_like(query) for _ in range(3)]
        mask = torch.zeros(1, 1, 5, 5).masked_fill(
            torch.ones(5, 5, dtype=torch.bool).triu(1), torch.finfo(query.dtype).min
        )
        actual = attention._chain_attention(query, [keys, values], mask)
        expected = []
        for row in range(5):
            row_keys = torch.cat(
                [
                    keys[0][:, :, : row + 1],
                    *[key[:, :, row : row + 1] for key in keys[1:]],
                ],
                dim=2,
            )
            row_values = torch.cat(
                [
                    values[0][:, :, : row + 1],
                    *[value[:, :, row : row + 1] for value in values[1:]],
                ],
                dim=2,
            )
            weights = torch.softmax(
                query[:, :, row : row + 1]
                @ row_keys.transpose(2, 3)
                * attention.scaling,
                dim=-1,
            )
            expected.append(weights @ row_values)
        expected = torch.cat(expected, dim=2).transpose(1, 2).contiguous()
        torch.testing.assert_close(actual, expected)

    def test_rope_reads_v5_parameters_and_legacy_fallback(self):
        from specforge.modeling.draft.mtp.qwen3_5 import PartialRotaryEmbedding

        for config in (
            SimpleNamespace(
                rope_parameters={"rope_theta": 10000000.0, "partial_rotary_factor": 0.5}
            ),
            SimpleNamespace(rope_theta=10000000.0, partial_rotary_factor=0.5),
        ):
            with self.subTest(config=config):
                actual = PartialRotaryEmbedding(config, 8).inv_freq
                expected = 1.0 / (10000000.0 ** (torch.arange(0, 4, 2).float() / 4))
                torch.testing.assert_close(actual, expected)

    def test_configuration_reaches_model_and_resume_contract(self):
        from pydantic import ValidationError

        from specforge.algorithms.mtp.providers import (
            build_training_model,
            resume_contract,
        )
        from specforge.config.schema import TrainingConfig

        for steps, decay in ((1, 0.8), (3, 0.6)):
            with self.subTest(steps=steps, decay=decay):
                config = SimpleNamespace(
                    training=TrainingConfig(mtp_num_steps=steps, mtp_ploss_decay=decay)
                )
                draft = model().draft_model
                parts = build_training_model(config, draft, None, None, None)
                self.assertEqual(parts.model.num_steps, steps)
                self.assertEqual(parts.model.ploss_decay, decay)
                contract = resume_contract(config, draft, parts.model)
                self.assertEqual(contract["mtp_num_steps"], steps)
                self.assertEqual(contract["mtp_ploss_decay"], decay)
        for values in (
            {"mtp_num_steps": 0},
            {"mtp_ploss_decay": 0},
            {"mtp_ploss_decay": 1.1},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValidationError):
                    TrainingConfig(**values)


if __name__ == "__main__":
    unittest.main()
