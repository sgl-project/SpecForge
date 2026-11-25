"""Check MMFlash in the canonical objective and offline feature runtime."""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from specforge.algorithms.objectives import build_objective_model, objective_record
from specforge.algorithms.objectives.inputs import gather, prepare_blocks
from specforge.application import resolve_run
from specforge.config import load_config
from specforge.modeling.draft.models import build_draft_model
from specforge.training.canonical_data import normalize_sample
from tests.test_config.draft_fixtures import draft_config


def test_sampled_loss_and_gradients_match_reference_across_chunks():
    torch.manual_seed(3)
    config = draft_config(target_bindings=(), target_logit_transform=None)
    model = build_draft_model(config)
    runtime = build_objective_model(
        model,
        name="mmflash",
        options={"num_anchors": 3},
        loss={"execution": {"chunk_size": 2}},
    )
    reference = build_draft_model(config)
    reference.load_state_dict(model.state_dict())
    mask = torch.ones(2, 16)
    mask[:, :2] = 0
    mask[0, 6] = 0
    mask[1, 13:] = 0
    batch = {
        "input_ids": torch.arange(32).reshape(2, 16),
        "loss_mask": mask,
        "hidden_states": torch.randn(2, 16, 64),
        "visual_score": torch.stack((torch.arange(16) * 570, torch.full((16,), -1))),
    }

    torch.manual_seed(29)
    measured = runtime(batch, step=0, total_steps=1, collect_detailed_metrics=False)
    measured.loss.backward()
    torch.manual_seed(29)
    valid = mask[:, :9].bool()
    priority = torch.rand(valid.shape).masked_fill(~valid, 2.0)
    anchors = priority.argsort(dim=1)[:, :3].sort(dim=1).values

    torch.manual_seed(29)
    blocks = prepare_blocks(config, batch, num_anchors=3, full_block_anchors=True)
    torch.testing.assert_close(blocks.anchors, anchors)
    output = reference(blocks.input_ids, batch["hidden_states"], blocks.layer_inputs)
    logits = output.logits.reshape(2, 3, 8, -1)
    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), blocks.labels.flatten(), reduction="none"
    ).reshape_as(blocks.labels)
    with torch.no_grad():
        p = (-ce).exp()
        prefix = (0.5 + 0.5 * p).masked_fill(blocks.weights == 0, 1).cumprod(-1)
        visual = (
            gather(batch["visual_score"], blocks.label_positions).clamp_min(0) / 10_000
        )
        weights = blocks.weights * prefix * (1 + visual * (1 - p))
    expected = (ce * weights).sum() / 2
    expected.backward()

    torch.testing.assert_close(measured.denominator, torch.tensor(2.0))
    torch.testing.assert_close(measured.loss, expected)
    for actual, wanted in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, wanted.grad)


def test_recipe_resolves_an_offline_multimodal_feature_contract_for_both_arms():
    root = Path(__file__).resolve().parents[2]
    recipe = str(
        root / "examples/configs/offline/colocated/qwen3.5-4b-mmflash-a100.yaml"
    )
    for options, objective in [
        ([], "mmflash"),
        (
            [
                "training.objective=dflash",
                "training.objective_options.full_block_anchors=true",
            ],
            "dflash",
        ),
    ]:
        run = resolve_run(load_config(recipe, overrides=options))
        contract = run.algorithm.spec.feature_contract(
            "offline", run.config.model.input_modality
        )
        assert run.config.training.objective == objective
        assert contract.modality == "multimodal"
        assert ("visual_score" in contract.required_tensors) == (objective == "mmflash")
    cfg = load_config(
        recipe, overrides=["training.weighting={kind: exponential, gamma: 7}"]
    )
    with pytest.raises(ValueError, match="chain_positions"):
        resolve_run(cfg)


def test_scored_record_rejects_bad_channel_before_truncation():
    tensors = {
        "input_ids": torch.arange(4),
        "visual_score": torch.tensor([0, 12, 31, 20001]),
    }
    with pytest.raises(ValueError, match="visual_score"):
        normalize_sample(tensors, keys=tuple(tensors), max_len=3)


def test_paired_sampler_is_recorded_only_when_it_changes_dflash():
    old = objective_record("dflash", {"num_anchors": 2}, training={})
    new = {"num_anchors": 2, "full_block_anchors": False}
    assert objective_record("dflash", new) == old
    new["full_block_anchors"] = True
    assert objective_record("dflash", new) == new
