from dataclasses import replace
from unittest.mock import patch

import torch

from specforge.algorithms.objectives.execution_options import LossExecution
from specforge.algorithms.objectives.protocols.parallel_reconsideration import (
    halt_probabilities,
    reconsideration_token_loss,
    rollout_inputs,
    select_halt,
)
from specforge.algorithms.objectives.registry import build_objective_model
from specforge.modeling import build_draft_model
from specforge.modeling.draft.layers.attention.kv import AttentionKV
from specforge.modeling.draft.models.eagle3 import Eagle3Output
from tests.test_modeling.test_canonical_eagle3 import (
    install_mapping,
    model_config,
    model_inputs,
)


def test_halt_probabilities_and_detached_objective():
    halts = torch.tensor([0.25, 0.4])
    torch.testing.assert_close(
        halt_probabilities(torch.logit(halts)), torch.tensor([0.25, 0.30, 0.45])
    )
    ce = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    mass = torch.tensor([0.25, 0.30, 0.45], requires_grad=True)
    loss = reconsideration_token_loss(ce, mass)
    torch.testing.assert_close(loss, torch.tensor(5.15))
    d_ce, d_mass = torch.autograd.grad(loss, (ce, mass))
    torch.testing.assert_close(d_ce, torch.tensor([1.0, 0.30, 0.45]))
    torch.testing.assert_close(d_mass, ce.detach())


def test_halt_selects_the_same_depth_for_feature_and_cache():
    outputs = []
    for depth in range(3):
        hidden = torch.full((1, 3, 2), float(depth), requires_grad=True)
        key = torch.full((1, 2, 3, 4), float(10 + depth), requires_grad=True)
        value = torch.full((1, 2, 3, 4), float(20 + depth), requires_grad=True)
        outputs.append(Eagle3Output(hidden_states=hidden, kv=AttentionKV(key, value)))
    hidden, kv = select_halt(outputs, torch.tensor([[0, 2, 1]]))
    torch.testing.assert_close(hidden[0, :, 0], torch.tensor([0.0, 2.0, 1.0]))
    torch.testing.assert_close(kv.key[0, 0, :, 0], torch.tensor([10.0, 12.0, 11.0]))
    torch.testing.assert_close(kv.value[0, 0, :, 0], torch.tensor([20.0, 22.0, 21.0]))
    (hidden.sum() + kv.key.sum() + kv.value.sum()).backward()
    torch.testing.assert_close(
        outputs[2].kv.key.grad[0, 0, :, 0], torch.tensor([0.0, 1.0, 0.0])
    )


def test_reconsideration_reuses_position_and_excludes_the_old_slot():
    config = replace(
        model_config(normalize=True, norm_output=True),
        architecture="ParallelReconsiderationDraftModel",
    )
    torch.manual_seed(19)
    model = build_draft_model(config)
    install_mapping(model)
    ids, features, _ = model_inputs(config)
    mask = torch.ones_like(ids, dtype=torch.bool)
    positions = torch.arange(ids.shape[1]).expand_as(ids)
    ordinary = rollout_inputs(config, mask, positions, 0, mixed=False)
    revised = rollout_inputs(config, mask, positions, 0, mixed=True)
    torch.testing.assert_close(ordinary[0].position_ids, revised[0].position_ids)
    origin = model(ids, features, ordinary)
    expected = model.reconsider(origin.hidden_states, revised, kv_prefix=origin.kv)

    key, value = origin.kv.key.clone(), origin.kv.value.clone()
    key[:, :, 0] += 100
    value[:, :, 0] += 100
    changed = model.reconsider(
        origin.hidden_states, revised, kv_prefix=AttentionKV(key, value)
    )
    torch.testing.assert_close(changed.logits[:, 0], expected.logits[:, 0])
    assert not torch.allclose(changed.logits[:, 1], expected.logits[:, 1])
    assert not torch.allclose(expected.logits[:, 0], origin.logits[:, 0])


def test_native_objective_updates_and_round_trips_model(tmp_path):
    torch.manual_seed(43)
    config = replace(
        model_config(normalize=True, norm_output=True),
        architecture="ParallelReconsiderationDraftModel",
    )
    model = build_draft_model(config, backend="sdpa")
    install_mapping(model)
    ids, features, _ = model_inputs(config)
    batch = {
        "input_ids": ids,
        "hidden_states": features,
        "loss_mask": torch.ones_like(ids),
        "target_logits": torch.nn.functional.one_hot(
            torch.tensor([[0, 6, 10, 14], [2, 8, 12, 16]]), config.vocab_size
        ).float(),
    }
    objective = build_objective_model(model, options={"ttt_length": 2})
    torch.manual_seed(11)
    full = objective(batch, step=0, total_steps=2)
    objective.execution = LossExecution(chunk_size=2)
    torch.manual_seed(11)
    chunked = objective(batch, step=0, total_steps=2)
    torch.testing.assert_close(full.loss, chunked.loss)
    assert full.denominator.item() == 14
    torch.testing.assert_close(
        full.ratios["halt_entropy"], chunked.ratios["halt_entropy"]
    )
    assert torch.isfinite(chunked.loss)
    chunked.loss.backward()
    parameters = (model.self_query.weight, model.halt_gate.weight)
    previous = [p.detach().clone() for p in parameters]
    for p in parameters:
        assert torch.isfinite(p.grad).all() and p.grad.norm() > 0
    torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    assert all(not torch.equal(p, old) for p, old in zip(parameters, previous))

    model.save_pretrained(tmp_path)
    loaded = type(model).from_pretrained(tmp_path)
    assert loaded.config.draft == model.config.draft
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], parameter)


def test_two_rollouts_sample_separately_and_commit_one_halt_slot():
    config = replace(
        model_config(normalize=True, norm_output=True),
        architecture="ParallelReconsiderationDraftModel",
    )
    model = build_draft_model(config)
    install_mapping(model)
    ids, features, _ = model_inputs(config)
    batch = {
        "input_ids": ids,
        "hidden_states": features,
        "loss_mask": torch.ones_like(ids),
        "target_logits": torch.nn.functional.one_hot(
            ids * 2, config.vocab_size
        ).float(),
    }
    runtime = build_objective_model(model, options={"ttt_length": 2})
    calls = []

    def record_forward(module, args, kwargs):
        calls.append((args[0], kwargs.get("kv_prefix")))

    handle = model.register_forward_pre_hook(record_forward, with_kwargs=True)
    choices = iter((2, 1, 7))
    try:
        with patch.object(
            torch,
            "multinomial",
            side_effect=lambda probabilities, count: probabilities.new_full(
                (probabilities.shape[0], count), next(choices), dtype=torch.long
            ),
        ):
            assert torch.isfinite(runtime(batch, step=0, total_steps=1).loss)
    finally:
        handle.remove()

    seed, one, mixed = calls
    assert seed[1] is None
    torch.testing.assert_close(one[0], torch.full_like(ids, 2))
    torch.testing.assert_close(mixed[0], torch.full_like(ids, 14))
    length = ids.shape[1]
    assert one[1].key.shape[2] == length
    assert mixed[1].key.shape[2] == 2 * length
    torch.testing.assert_close(mixed[1].key[:, :, :length], one[1].key)
    assert not torch.allclose(mixed[1].key[:, :, length:], one[1].key)


def test_empty_eligible_minibatch_keeps_a_valid_optimizer_denominator():
    config = replace(
        model_config(normalize=True, norm_output=True),
        architecture="ParallelReconsiderationDraftModel",
    )
    model = build_draft_model(config)
    install_mapping(model)
    ids, features, _ = model_inputs(config)
    runtime = build_objective_model(model, options={"ttt_length": 2})
    result = runtime(
        {
            "input_ids": ids,
            "hidden_states": features,
            "loss_mask": torch.ones_like(ids),
            "target_logits": torch.nn.functional.one_hot(
                torch.ones_like(ids), config.vocab_size
            ).float(),
        },
        step=0,
        total_steps=1,
    )
    # All teacher IDs are odd whereas the draft contains the even IDs.
    assert result.denominator.item() > 0
    assert result.loss.item() == 0
    result.loss.backward()
    assert model.self_query.weight.grad.norm().item() == 0
