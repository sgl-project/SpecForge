"""Stage selection and checkpoint export compatibility across EAGLE4 phases."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from specforge.algorithms.model_providers import build_eagle4_draft
from specforge.export.to_hf import export_to_hf
from specforge.export.to_sglang import export_to_sglang
from specforge.modeling.draft.eagle4 import LlamaForCausalLMEagle4
from tests.test_modeling.test_draft.test_eagle4 import _config


@pytest.mark.parametrize("source_format", ["runtime", "hf"])
@pytest.mark.parametrize("exporter", [export_to_hf, export_to_sglang])
def test_stage1_checkpoint_to_stage2_training_and_export(
    tmp_path, monkeypatch, source_format, exporter
):
    """Exercise real warm-start loading, SVD initialization, freezing and export."""
    torch.manual_seed(7)
    head = {"type": "low_rank", "rank": 1, "num_candidates": 4}
    stage1_config = _config(
        eagle4_training_stage="stage1", efficient_lm_head_config=head
    )
    stage1 = LlamaForCausalLMEagle4(stage1_config, attention_backend="sdpa")
    token_ids = torch.randint(0, 16, (2, 16))
    target_features = torch.randn(2, 16, 24)
    optimizer = torch.optim.SGD(stage1.parameters(), lr=0.1)
    logits = stage1.compute_logits(
        stage1(
            hidden_states=target_features,
            inputs_embeds=stage1.embed_input_ids(token_ids),
        )
    )
    torch.nn.functional.cross_entropy(
        logits.reshape(-1, 16), token_ids.flatten()
    ).backward()
    optimizer.step()
    dense_weights = {key: value.clone() for key, value in stage1.state_dict().items()}
    assert not any("efficient_lm_head" in key for key in dense_weights)
    stage1_path = tmp_path / "stage1"
    stage1_path.mkdir()
    torch.save(
        {"strategy": "eagle4", "draft_state_dict": dense_weights},
        stage1_path / "training_state.pt",
    )
    config_path = tmp_path / "stage1.json"
    stage1_config.to_json_file(str(config_path))
    source = stage1_path
    if source_format == "hf":
        source = tmp_path / "stage1-export"
        export_to_hf(str(stage1_path), str(config_path), str(source))

    cfg = SimpleNamespace(
        model=SimpleNamespace(
            vocab_mapping_path="",
            draft_checkpoint_path=str(source),
            torch_dtype="float32",
            load_target_embedding=False,
            cache_dir=str(tmp_path),
            trust_remote_code=False,
        ),
        training=SimpleNamespace(attention_backend="sdpa", strategy="eagle4"),
    )
    monkeypatch.setattr(
        "specforge.algorithms.model_providers._device", lambda: torch.device("cpu")
    )
    stage2_config = _config(
        eagle4_training_stage="stage2", efficient_lm_head_config=head
    )
    stage2 = build_eagle4_draft(cfg, stage2_config)
    factor_keys = {"efficient_lm_head.weight", "efficient_lm_head_down_proj.weight"}
    assert {
        name for name, p in stage2.named_parameters() if p.requires_grad
    } == factor_keys
    for key, value in dense_weights.items():
        torch.testing.assert_close(stage2.state_dict()[key], value, rtol=0, atol=0)
    if source_format == "hf":
        exported = load_file(str(source / "model.safetensors"))
        for key in factor_keys:
            torch.testing.assert_close(
                stage2.state_dict()[key], exported[key], rtol=0, atol=0
            )
    u, singular, vh = torch.linalg.svd(
        dense_weights["lm_head.weight"], full_matrices=False
    )
    torch.testing.assert_close(
        stage2.efficient_lm_head.weight @ stage2.efficient_lm_head_down_proj.weight,
        (u[:, :1] * singular[:1]) @ vh[:1],
    )
    before = {key: value.clone() for key, value in stage2.state_dict().items()}
    optimizer = torch.optim.SGD(
        [p for p in stage2.parameters() if p.requires_grad], lr=0.1
    )
    hidden = stage2(
        hidden_states=target_features, inputs_embeds=stage2.embed_input_ids(token_ids)
    )
    loss, _ = stage2.retrieval_loss(hidden)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    for name, parameter in stage2.named_parameters():
        if name in factor_keys:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert torch.count_nonzero(parameter.grad) > 0
        else:
            assert parameter.grad is None
    optimizer.step()
    stage2.update_moe_router_bias()
    weights = stage2.state_dict()
    for key, value in before.items():
        if key in factor_keys:
            assert not torch.equal(weights[key], value)
        else:
            torch.testing.assert_close(weights[key], value, rtol=0, atol=0)
    stage2_path = tmp_path / "stage2"
    stage2_path.mkdir()
    torch.save(
        {"strategy": "eagle4", "draft_state_dict": weights},
        stage2_path / "training_state.pt",
    )
    stage2_config.to_json_file(str(tmp_path / "stage2.json"))
    destination = tmp_path / "stage2-export"
    exporter(str(stage2_path), str(tmp_path / "stage2.json"), str(destination))
    saved = load_file(str(destination / "model.safetensors"))
    for key, value in weights.items():
        if key in saved:
            torch.testing.assert_close(saved[key], value, rtol=0, atol=0)
    for key in factor_keys | {"lm_head.weight"}:
        torch.testing.assert_close(saved[key], weights[key], rtol=0, atol=0)


def test_legacy_head_metadata_keeps_dense_stage1_training():
    config = _config(
        efficient_lm_head_config={"type": "low_rank", "rank": 4, "num_candidates": 8}
    )
    delattr(config, "eagle4_training_stage")
    model = LlamaForCausalLMEagle4(config, attention_backend="sdpa")
    assert model.efficient_lm_head is None
    assert not any("efficient_lm_head" in key for key in model.state_dict())
    hidden = torch.randn(1, 3, 24)
    inputs = model.embed_input_ids(torch.tensor([[1, 2, 3]]))
    logits = model.compute_logits(model(hidden_states=hidden, inputs_embeds=inputs))
    loss = logits.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert model.lm_head.weight.grad is not None
    assert model.fc.weight.grad is not None


@pytest.mark.parametrize("exporter", [export_to_hf, export_to_sglang])
def test_stage2_export_preserves_learned_factors_and_dense_weights(exporter):
    config = _config(
        eagle4_training_stage="stage2",
        efficient_lm_head_config={"type": "low_rank", "rank": 4, "num_candidates": 8},
    )
    model = LlamaForCausalLMEagle4(config, attention_backend="sdpa")
    model.init_efficient_head_from_dense()
    model.freeze_non_candidate_parameters()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    hidden = torch.randn(1, 3, 8)
    loss, _ = model.retrieval_loss(hidden)
    loss.backward()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.add_(parameter.grad, alpha=-0.01)
    weights = model.state_dict()
    assert torch.equal(weights["lm_head.weight"], before["lm_head.weight"])
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config.to_json_file(str(root / "draft.json"))
        torch.save(
            {"strategy": "eagle4", "draft_state_dict": weights},
            root / "training_state.pt",
        )
        exporter(str(root), str(root / "draft.json"), str(root / "export"))
        saved = load_file(str(root / "export/model.safetensors"))
        for name in (
            "lm_head.weight",
            "efficient_lm_head.weight",
            "efficient_lm_head_down_proj.weight",
        ):
            torch.testing.assert_close(saved[name], weights[name], rtol=0, atol=0)


@pytest.mark.parametrize("stage", ["invalid", 2])
def test_invalid_training_stage_fails(stage):
    with pytest.raises(ValueError, match="eagle4_training_stage"):
        LlamaForCausalLMEagle4(_config(eagle4_training_stage=stage))


def test_candidate_stage_requires_head_configuration():
    with pytest.raises(ValueError, match="stage2 requires"):
        LlamaForCausalLMEagle4(_config(eagle4_training_stage="stage2"))
