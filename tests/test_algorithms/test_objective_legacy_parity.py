"""Compare canonical training losses and gradients with the legacy wrappers."""

import copy
import unittest
from dataclasses import replace

import torch

from specforge.algorithms.objectives import build_objective_model
from specforge.modeling import build_draft_model
from tests.test_algorithms.test_objectives import training_batch


def assert_loss_and_gradients(new, old, new_loss, old_loss, names):
    torch.testing.assert_close(new_loss, old_loss, rtol=2e-5, atol=2e-6)
    new_loss.backward()
    old_loss.backward()
    parameters = dict(old.named_parameters())
    for name, parameter in new.draft_model.named_parameters():
        if name in names:
            torch.testing.assert_close(
                parameter.grad,
                parameters[names[name]].grad,
                rtol=2e-4,
                atol=2e-6,
                msg=lambda message: f"{name}: {message}",
            )


class LegacyObjectiveParityTest(unittest.TestCase):
    def test_mtp_matches_explicit_shifted_loss_with_native_layers(self):
        from specforge.config.draft.norm import RMSNormSpec
        from specforge.config.draft.rope import DefaultRoPESpec
        from tests.test_config.test_draft_mtp_spec import mtp_config
        from tests.test_modeling.test_canonical_mtp import (
            reference_forward,
            reference_model,
        )

        config = mtp_config(target_bindings=(), target_logit_transform=None)
        config = replace(
            config,
            norm_config=RMSNormSpec(has_weight=True, eps=1e-6, weight_offset=1),
            attention_configs={
                "full": replace(config.attention_configs["full"], output_gate=True)
            },
            rope_parameters=DefaultRoPESpec(
                interleaved=False, rope_theta=10000000, partial_rotary_factor=0.5
            ),
        )
        model = build_draft_model(config)
        reference = reference_model(config)
        reference.load_state_dict(model.state_dict())
        new = build_objective_model(model)
        batch = training_batch(config)
        batch["attention_mask"].fill_(1)
        actual = new(batch, step=0, total_steps=None).loss
        # Historical MTP alignment: h[t-1] and token[t] predict token[t+1].
        ids = batch["input_ids"]
        _, logits = reference_forward(
            reference,
            ids[:, 1:],
            batch["target_last_hidden_states"][:, :-1].detach(),
            torch.arange(1, ids.shape[1]).expand_as(ids[:, 1:]),
        )
        losses = torch.nn.functional.cross_entropy(
            logits[:, :-1].flatten(0, 1), ids[:, 2:].flatten(), reduction="none"
        )
        mask = batch["loss_mask"][:, 2:].flatten().float()
        expected = (losses * mask).sum() / mask.sum()
        assert_loss_and_gradients(
            new,
            reference,
            actual,
            expected,
            {n: n for n, _ in model.named_parameters()},
        )

    def test_dmtp_matches_explicit_suffix_loss(self):
        # DMTP has no legacy Online wrapper. Check its label alignment independently.
        from specforge.algorithms.objectives.inputs import gather
        from tests.test_algorithms.test_objectives import dmtp_config

        config = dmtp_config()
        model = build_draft_model(config)
        reference = copy.deepcopy(model)
        batch = training_batch(config)
        batch["loss_mask"][0, 4] = 0
        calls = []
        model.register_forward_pre_hook(
            lambda module, args, kwargs: calls.append((args, kwargs)), with_kwargs=True
        )
        runtime = build_objective_model(model, options={"num_anchors": 2})
        actual = runtime(batch, step=0, total_steps=None).loss
        args, kwargs = calls[0]
        logits = reference(*args, **{**kwargs, "output_logits": True}).logits.reshape(
            1, -1, 4, config.vocab_size
        )
        # Fixture tokens equal positions; the first input is the external candidate.
        first_positions = args[0].reshape(1, -1, 4)[..., 0]
        positions = first_positions[..., None] + torch.arange(4)
        labels = gather(batch["input_ids"], positions)
        mask = gather(batch["loss_mask"], positions).bool() & (positions < 9)
        mask[..., 0] = False
        ce = torch.nn.functional.cross_entropy(
            logits.flatten(0, 2), labels.flatten(), reduction="none"
        ).reshape_as(labels)
        expected = (ce * mask).sum() / mask.sum()
        assert_loss_and_gradients(
            runtime,
            reference,
            actual,
            expected,
            {n: n for n, _ in model.named_parameters()},
        )
