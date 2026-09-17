"""Frozen-cache replay must keep example identity and metric denominators."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from specforge.eval.checkpoint import (
    evaluate_cached_features,
    load_feature_index,
    run_checkpoint_evaluation,
)


class CountedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, hidden_states, loss_mask, **kwargs):
        assert not self.training and not torch.is_grad_enabled()
        numerator, denominator = input_ids[0, :2]
        ratio = numerator / denominator
        return (
            ratio,
            ratio,
            {
                "random_anchor": torch.rand(()),
                "ratio_metrics": {"ce_loss": (numerator, denominator)},
                "sum_metrics": {"eligible": denominator},
            },
        )


@pytest.fixture
def index(tmp_path):
    rows = []
    for number, pair in enumerate(((1, 2), (9, 10))):
        path = tmp_path / f"{number}.pt"
        torch.save(
            {
                "input_ids": torch.tensor([pair]),
                "loss_mask": torch.ones(1, 2),
                "hidden_states": torch.ones(1, 2, 3),
                "target_last_hidden_states": torch.ones(1, 2, 3),
            },
            path,
        )
        rows.append(
            {"id": str(number), "path": path.name, "group_hash": f"{number + 19:064x}"}
        )
    manifest = tmp_path / "index.json"
    manifest.write_text(json.dumps({"rows": rows}))
    return manifest


def test_pool_denominators_and_keep_per_example_evidence(index):
    report = evaluate_cached_features(CountedModel(), load_feature_index(index))
    assert report["mean"]["loss"] == pytest.approx(0.7)
    assert report["mean"]["ce_loss"] == pytest.approx(10 / 12)
    assert report["ratio_totals"]["ce_loss"] == [10, 12]
    assert report["sum_totals"]["eligible"] == 12
    assert report["rows"][1]["ratio_metrics"]["ce_loss"] == [9, 10]
    assert (
        report["rows"][0]["sha256"]
        == hashlib.sha256((index.parent / "0.pt").read_bytes()).hexdigest()
    )


def test_seed_is_per_example_and_replay_restores_rng_and_training_mode(index):
    rows = load_feature_index(index)
    rows[0]["seed"], rows[1]["seed"] = 123, 456
    model = CountedModel()
    state = torch.random.get_rng_state().clone()
    first = evaluate_cached_features(model, rows)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert model.training
    second = evaluate_cached_features(model, rows[::-1])
    assert {r["id"]: r["metrics"] for r in first["rows"]} == {
        r["id"]: r["metrics"] for r in second["rows"]
    }


@pytest.mark.parametrize("bad", ["nonfinite", "denominator", "schema"])
def test_reject_invalid_metric_populations(index, bad):
    class InvalidModel(CountedModel):
        def forward(self, input_ids, **kwargs):
            loss, accuracy, metrics = super().forward(input_ids=input_ids, **kwargs)
            if bad == "nonfinite":
                loss = float("nan")
            elif bad == "denominator":
                metrics["ratio_metrics"]["ce_loss"] = (1, 0)
            elif input_ids[0, 0] == 9:
                metrics["ratio_metrics"]["teacher_only"] = (1, 2)
            return loss, accuracy, metrics

    with pytest.raises(ValueError):
        evaluate_cached_features(InvalidModel(), load_feature_index(index))


@pytest.mark.parametrize("bad", ["empty", "duplicate", "missing", "seed", "limit"])
def test_reject_invalid_index(index, bad):
    manifest = json.loads(index.read_text())
    if bad == "empty":
        manifest["rows"] = []
    elif bad == "duplicate":
        manifest["rows"].append(manifest["rows"][0])
    elif bad == "missing":
        manifest["rows"][0]["path"] = "missing.pt"
    elif bad == "seed":
        manifest["rows"][0]["seed"] = -1
    index.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        load_feature_index(index, 0 if bad == "limit" else None)


@pytest.mark.parametrize("bad", ["digest", "shape", "nan", "mask"])
def test_reject_bad_cache_and_restore_model(index, bad):
    rows = load_feature_index(index)
    path = Path(rows[0]["path"])
    if bad == "digest":
        rows[0]["sha256"] = "0" * 64
    else:
        features = torch.load(path, weights_only=True)
        if bad == "shape":
            features["hidden_states"] = torch.ones(2, 2, 3)
        elif bad == "nan":
            features["hidden_states"][0, 0, 0] = float("nan")
        else:
            features["loss_mask"].zero_()
        torch.save(features, path)
    model = CountedModel()
    with pytest.raises(ValueError):
        evaluate_cached_features(model, rows)
    assert model.training


def test_cli_runs_weights_only_and_records_resolved_recipe(index, monkeypatch):
    from specforge.cli import main
    from specforge.config import Config

    config = Config.model_validate(
        {
            "model": {
                "target_model_path": "target",
                "draft_model_config": "draft.json",
            },
            "data": {"hidden_states_path": str(index.parent)},
            "training": {"strategy": "dflash", "resume_from": "old-training-state"},
        }
    )
    model = CountedModel()
    draft_config = SimpleNamespace(to_dict=lambda: {"block_size": 8})

    def build(cfg, **kwargs):
        assert cfg.training.resume_from is None
        assert cfg.model.draft_checkpoint_path is None
        return SimpleNamespace(
            model=model, draft_model=model, draft_config=draft_config
        )

    def load(draft, checkpoint, **kwargs):
        assert draft is model and checkpoint == "saved-checkpoint"
        assert kwargs["allow_missing_embedding"]
        return SimpleNamespace(checkpoint_format="specforge")

    monkeypatch.setattr("specforge.cli.load_config", lambda *args: config)
    monkeypatch.setattr("specforge.training.assembly.build_model_bundle", build)
    monkeypatch.setattr("specforge.training.model_loading.warm_start_draft_model", load)
    output = index.parent / "report.json"
    assert (
        main(
            [
                "evaluate-checkpoint",
                "--config",
                str(index),
                "--checkpoint",
                "saved-checkpoint",
                "--feature-index",
                str(index),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    assert report["examples"] == 2 and report["optimizer_updates"] == 0
    assert report["checkpoint"] == "saved-checkpoint"
    assert report["recipe"]["training"]["resume_from"] is None
    assert config.training.resume_from == "old-training-state"
    with pytest.raises(FileExistsError):
        run_checkpoint_evaluation(config, "saved-checkpoint", index, output)


def test_real_dflash2_checkpoint_replays_identical_metrics(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    from specforge.config import Config
    from specforge.modeling.draft.dflash2 import DFlash2DraftModel
    from tests.test_modeling.test_dflash2 import _tiny_config

    monkeypatch.setenv("SPECFORGE_DEVICE", "cpu")
    torch.manual_seed(9)
    draft_config = _tiny_config()
    draft_config._attn_implementation = "eager"
    draft = DFlash2DraftModel(draft_config)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    torch.save(
        {"strategy": "dflash", "draft_state_dict": draft.state_dict()},
        checkpoint / "training_state.pt",
    )
    draft_path = tmp_path / "draft.json"
    draft_config.to_json_file(draft_path)
    target = tmp_path / "target"
    target.mkdir()
    draft_config.save_pretrained(target)
    save_file(
        {
            "model.embed_tokens.weight": torch.randn(32, 16),
            "lm_head.weight": torch.randn(32, 16),
        },
        target / "model.safetensors",
    )
    feature_path = tmp_path / "features.pt"
    torch.save(
        {
            "input_ids": torch.randint(0, 30, (1, 12)),
            "loss_mask": torch.ones(1, 12),
            "hidden_states": torch.randn(1, 12, 16),
            "target_last_hidden_states": torch.randn(1, 12, 16),
        },
        feature_path,
    )
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps({"rows": [{"id": "tiny", "path": feature_path.name, "seed": 17}]})
    )
    config = Config.model_validate(
        {
            "model": {
                "target_model_path": str(target),
                "draft_model_config": str(draft_path),
                "mask_token_id": 31,
                "torch_dtype": "float32",
            },
            "data": {"hidden_states_path": str(tmp_path)},
            "training": {
                "strategy": "dflash",
                "attention_backend": "eager",
                "num_anchors": 3,
            },
        }
    )
    first = run_checkpoint_evaluation(
        config, str(checkpoint), index, tmp_path / "first.json"
    )
    second = run_checkpoint_evaluation(
        config, str(checkpoint), index, tmp_path / "second.json"
    )
    assert first["mean"] == second["mean"]
    assert first["ratio_totals"] == second["ratio_totals"]
    assert "dflash2/selector/greedy_accepted_length" in first["mean"]
    assert "position_1/selector/greedy_prefix_survival" in first["mean"]
