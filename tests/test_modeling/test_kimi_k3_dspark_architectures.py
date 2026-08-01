import json
from pathlib import Path

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from specforge.config import Config
from specforge.modeling.draft.kimi_k3_dspark import (
    KimiK3DraftKDAAttention,
    KimiK3DSpark4KDA1MLADraftModel,
    KimiK3DSpark5MLADraftModel,
)

ROOT = Path(__file__).resolve().parents[2]


def _tiny_config(architecture: str, *, layers: int = 5) -> Qwen3Config:
    config = Qwen3Config(
        architectures=[architecture],
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        mla_use_nope=True,
        mla_use_output_gate=True,
        rope_interleave=True,
        max_position_embeddings=128,
        vocab_size=64,
        block_size=3,
        num_target_layers=8,
        dflash_config={
            "projector_type": "dspark",
            "target_layer_ids": [0, 1, 2, 3, 4],
            "mask_token_id": 63,
            "markov_rank": 4,
            "enable_confidence_head": True,
            "confidence_head_with_markov": True,
            "confidence_head_alpha": 1.0,
        },
        draft_layer_types=["kda", "kda", "mla", "kda", "kda"],
        linear_attn_config={
            "backend": "reference",
            "gate_lower_bound": -5.0,
            "head_dim": 4,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
        },
        layer_types=["full_attention"] * layers,
        attention_bias=False,
    )
    config._attn_implementation = "sdpa"
    return config


@pytest.mark.parametrize(
    ("filename", "architecture"),
    [
        ("kimi-k3-dspark-5mla.json", "KimiK3DSpark5MLADraftModel"),
        (
            "kimi-k3-dspark-4kda-1mla.json",
            "KimiK3DSpark4KDA1MLADraftModel",
        ),
    ],
)
def test_production_configs_match_k3_target(filename, architecture):
    config = json.loads((ROOT / "configs" / filename).read_text())
    assert config["architectures"] == [architecture]
    assert config["block_size"] == 7
    assert config["num_hidden_layers"] == 5
    assert config["dflash_config"]["target_layer_ids"] == [11, 23, 47, 71, 83]
    assert config["mla_use_output_gate"] is True
    assert {
        "num_attention_heads": config["num_attention_heads"],
        "q_lora_rank": config["q_lora_rank"],
        "kv_lora_rank": config["kv_lora_rank"],
        "qk_nope_head_dim": config["qk_nope_head_dim"],
        "qk_rope_head_dim": config["qk_rope_head_dim"],
        "v_head_dim": config["v_head_dim"],
    } == {
        "num_attention_heads": 96,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
    }


@pytest.mark.parametrize(
    "filename",
    [
        "kimi-k3-dspark-5mla-openperfectblend-disaggregated.yaml",
        "kimi-k3-dspark-4kda-1mla-openperfectblend-disaggregated.yaml",
    ],
)
def test_training_recipes_preserve_reference_run_contract(filename):
    config = Config.from_file(str(ROOT / "examples" / "configs" / filename))
    assert config.data.max_length == 4096
    assert config.training.batch_size == 8
    assert config.training.accumulation_steps == 32
    assert config.training.num_epochs == 10
    assert config.training.learning_rate == pytest.approx(6e-4)
    assert config.training.warmup_ratio == pytest.approx(0.04)
    assert config.training.num_anchors == 512
    assert config.training.save_interval == 250
    assert config.training.log_interval == 10
    assert config.tracking.report_to == "wandb"
    assert config.tracking.wandb_offline is False
    assert "kimi-k3-openperfectblend-regen-439c2fdc" in (config.data.train_data_path)


def _forward_backward(model):
    batch, context_len, query_len = 1, 5, 6
    config = model.config
    target_hidden = torch.randn(
        batch,
        context_len,
        len(config.dflash_config["target_layer_ids"]) * config.hidden_size,
    )
    noise_embedding = torch.randn(
        batch, query_len, config.hidden_size, requires_grad=True
    )
    position_ids = torch.arange(context_len + query_len).expand(batch, -1)
    output = model(
        position_ids=position_ids,
        target_hidden=target_hidden,
        noise_embedding=noise_embedding,
    )
    assert output.shape == (batch, query_len, config.hidden_size)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert noise_embedding.grad is not None
    assert torch.isfinite(noise_embedding.grad).all()


def test_tiny_5mla_forward_and_backward():
    model = KimiK3DSpark5MLADraftModel(_tiny_config("KimiK3DSpark5MLADraftModel"))
    _forward_backward(model)
    assert all(layer.self_attn.use_output_gate for layer in model.layers)


def test_tiny_4kda_1mla_forward_and_backward():
    model = KimiK3DSpark4KDA1MLADraftModel(
        _tiny_config("KimiK3DSpark4KDA1MLADraftModel")
    )
    _forward_backward(model)
    assert [type(layer.self_attn).__name__ for layer in model.layers] == [
        "KimiK3DraftKDAAttention",
        "KimiK3DraftKDAAttention",
        "KimiK3DraftMLAAttention",
        "KimiK3DraftKDAAttention",
        "KimiK3DraftKDAAttention",
    ]
    first_kda = model.layers[0].self_attn
    assert "q_conv1d.weight" in first_kda.state_dict()
    assert "q_conv1d.bias" not in first_kda.state_dict()


def test_kda_resets_state_between_proposal_blocks():
    config = _tiny_config("KimiK3DSpark4KDA1MLADraftModel")
    attention = KimiK3DraftKDAAttention(config, layer_idx=0)
    first = torch.randn(1, config.block_size, config.hidden_size)
    second = torch.randn_like(first)
    baseline = attention(torch.cat((first, second), dim=1))[0]
    changed = attention(torch.cat((first, second + 20.0), dim=1))[0]
    torch.testing.assert_close(
        baseline[:, : config.block_size],
        changed[:, : config.block_size],
    )


def test_hybrid_rejects_noncanonical_layer_order():
    config = _tiny_config("KimiK3DSpark4KDA1MLADraftModel")
    config.draft_layer_types = ["mla", "kda", "kda", "kda", "kda"]
    with pytest.raises(ValueError, match="layer pattern"):
        KimiK3DSpark4KDA1MLADraftModel(config)


def test_5mla_rejects_noncanonical_layer_count():
    config = _tiny_config("KimiK3DSpark5MLADraftModel", layers=4)
    with pytest.raises(ValueError, match="exactly 5 layers"):
        KimiK3DSpark5MLADraftModel(config)


@pytest.mark.parametrize("field", ["mla_use_nope", "mla_use_output_gate"])
def test_mla_requires_k3_attention_features(field):
    config = _tiny_config("KimiK3DSpark5MLADraftModel")
    setattr(config, field, False)
    with pytest.raises(ValueError, match=field):
        KimiK3DSpark5MLADraftModel(config)
