"""Stage selection and checkpoint export compatibility across EAGLE4 phases."""

import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from specforge.export.to_hf import export_to_hf
from specforge.export.to_sglang import export_to_sglang
from specforge.modeling.draft.eagle4 import LlamaForCausalLMEagle4
from tests.test_modeling.test_draft.test_eagle4 import _config


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
