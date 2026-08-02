import json
from pathlib import Path

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from specforge.config import Config
from specforge.data.template import TEMPLATE_REGISTRY
from specforge.modeling.draft import kimi_k3_dspark
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


def test_reference_full_attention_config_is_exact_old_architecture():
    config = json.loads(
        (ROOT / "configs" / "kimi-k3-dspark-fullattn-gqa16.json").read_text()
    )
    assert config["architectures"] == ["DSparkDraftModel"]
    assert config["block_size"] == 7
    assert config["num_hidden_layers"] == 5
    assert config["layer_types"] == ["full_attention"] * 5
    assert config["num_attention_heads"] == 64
    assert config["num_key_value_heads"] == 16
    assert config["dflash_config"]["target_layer_ids"] == [7, 23, 51, 67, 83]
    assert config["rope_scaling"] is None


def test_reference_full_attention_recipe_preserves_exact_old_contract():
    config = Config.from_file(
        str(
            ROOT
            / "examples"
            / "configs"
            / "kimi-k3-dspark-fullattn-openperfectblend-disaggregated.yaml"
        )
    )
    assert config.model.target_model_path == "/workspace/models/Kimi-K3"
    assert config.data.max_length == 4096
    assert config.data.dataloader_num_workers == 4
    assert config.training.batch_size == 1
    assert config.training.accumulation_steps == 32
    assert config.deployment.trainer.nnodes == 2
    assert config.deployment.trainer.nproc_per_node == 8
    assert (
        config.deployment.trainer.nnodes
        * config.deployment.trainer.nproc_per_node
        * config.training.batch_size
        == 16
    )
    assert (
        config.deployment.trainer.nnodes
        * config.deployment.trainer.nproc_per_node
        * config.training.batch_size
        * config.training.accumulation_steps
        == 512
    )
    assert config.training.num_epochs == 10
    assert config.training.total_steps == 9173
    assert config.training.learning_rate == pytest.approx(6e-4)
    assert config.training.lr_scheduler == "cosine"
    assert config.training.warmup_ratio == pytest.approx(0.04)
    assert config.training.num_anchors == 512
    assert config.training.max_checkpoints == 3
    assert config.runtime.producer_lease == 16
    assert len(config.deployment.disaggregated.server_urls) == 2
    assert (
        config.deployment.disaggregated.inbox_server_url
        == "http://trainer-node-0:35900"
    )
    assert "openperfectblend-regen-9caaf705" in config.data.train_data_path
    assert config.tracking.report_to == "wandb"


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
    assert config.training.batch_size == 2
    assert config.training.accumulation_steps == 32
    assert (
        config.deployment.trainer.nnodes
        * config.deployment.trainer.nproc_per_node
        * config.training.batch_size
        == 16
    )
    assert (
        config.deployment.trainer.nnodes
        * config.deployment.trainer.nproc_per_node
        * config.training.batch_size
        * config.training.accumulation_steps
        == 512
    )
    assert config.training.num_epochs == 10
    assert config.training.learning_rate == pytest.approx(6e-4)
    assert config.training.warmup_ratio == pytest.approx(0.04)
    assert config.training.num_anchors == 512
    assert config.training.save_interval == 250
    assert config.training.log_interval == 10
    optimizer_quantum = (
        config.deployment.trainer.nnodes
        * config.deployment.trainer.nproc_per_node
        * config.training.batch_size
        * config.training.accumulation_steps
    )
    max_capture_overshoot = (
        len(config.deployment.disaggregated.server_urls)
        * config.runtime.producer_concurrency
        * config.runtime.producer_lease
    )
    # Two windows are required for capture and training to overlap.  Once one
    # window is acknowledged, hysteresis must resume capture even at the
    # maximum number of concurrently leased refs.
    assert config.runtime.in_flight_high_watermark >= 2 * optimizer_quantum
    assert (
        config.runtime.in_flight_low_watermark
        >= config.runtime.in_flight_high_watermark
        + max_capture_overshoot
        - optimizer_quantum
    )
    # Each capture HTTP request uses the source job's complete 16-sample
    # optimizer microbatch to amortize auxiliary-state aggregation.  This is a
    # producer request size, independent of the DP8 consumer's per-rank batch.
    assert config.runtime.producer_lease == 16
    assert config.runtime.producer_concurrency == 2
    assert config.model.sglang_enable_symm_mem is False
    assert config.model.sglang_max_running_requests == 16
    # K3 consumes five linear-attention cache entries per live request; 40
    # silently caps SGLang at eight even when max_running_requests is 16.
    assert config.model.sglang_max_mamba_cache_size == 80
    assert len(config.deployment.disaggregated.server_urls) >= 2
    # Two worst-case 4,096-token optimizer windows carry about 336 GiB of
    # captured features; byte throttling must not serialize the pipeline.
    assert config.runtime.resident_high_watermark_bytes >= 360777252864
    assert config.runtime.resident_low_watermark_bytes >= 180388626432
    assert (
        config.runtime.feature_store_max_resident_bytes
        >= config.runtime.resident_high_watermark_bytes
    )
    assert config.tracking.report_to == "wandb"
    assert config.tracking.wandb_offline is False
    assert config.data.chat_template in TEMPLATE_REGISTRY.get_all_template_names()
    assert "kimi-k3-openperfectblend-regen-439c2fdc" in (config.data.train_data_path)


def test_kimi_k3_template_matches_target_xtml_contract():
    template = TEMPLATE_REGISTRY.get("kimi-k3-thinking")
    assert template.assistant_header == (
        '<|open|>message role="assistant"<|sep|><|open|>think<|sep|>'
    )
    assert template.user_header == '<|open|>message role="user"<|sep|>'
    assert template.end_of_turn_token == "<|end_of_msg|>"
    assert template.parser_type == "thinking"
    assert template.enable_thinking is False
    assert template.ignore_token == ["<|end_of_msg|>"]


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


def test_mla_absorbed_and_expanded_attention_are_algebraically_equivalent():
    torch.manual_seed(7)
    batch, queries, keys, heads = 2, 3, 5, 4
    nope_dim, latent_dim, value_dim = 6, 8, 7
    q_nope = torch.randn(batch, queries, heads, nope_dim, dtype=torch.float64)
    kv_latent = torch.randn(batch, keys, latent_dim, dtype=torch.float64)
    w_kc = torch.randn(heads, nope_dim, latent_dim, dtype=torch.float64)
    w_vc = torch.randn(heads, value_dim, latent_dim, dtype=torch.float64)

    q_absorbed = torch.einsum("bqhd,hdk->bqhk", q_nope, w_kc)
    absorbed_scores = torch.einsum("bqhk,bsk->bhqs", q_absorbed, kv_latent)
    k_expanded = torch.einsum("bsk,hdk->bshd", kv_latent, w_kc)
    expanded_scores = torch.einsum("bqhd,bshd->bhqs", q_nope, k_expanded)
    torch.testing.assert_close(absorbed_scores, expanded_scores)

    probabilities = absorbed_scores.softmax(dim=-1)
    latent_output = torch.einsum("bhqs,bsk->bhqk", probabilities, kv_latent)
    absorbed_output = torch.einsum("bhqk,hvk->bqhv", latent_output, w_vc)
    v_expanded = torch.einsum("bsk,hvk->bshv", kv_latent, w_vc)
    expanded_output = torch.einsum("bhqs,bshv->bqhv", probabilities, v_expanded)
    torch.testing.assert_close(absorbed_output, expanded_output)


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


def test_fla_kda_splits_independent_blocks_below_cuda_grid_z_limit(monkeypatch):
    calls = []

    def fake_chunk_kda(**kwargs):
        q = kwargs["q"]
        calls.append(int(q.shape[0]))
        assert q.shape[0] * q.shape[2] <= 8
        assert kwargs["output_final_state"] is False
        assert kwargs["use_qk_l2norm_in_kernel"] is True
        assert kwargs["use_gate_in_kernel"] is True
        assert kwargs["use_beta_sigmoid_in_kernel"] is True
        output = (
            q
            + kwargs["k"]
            + kwargs["v"]
            + kwargs["g"]
            + kwargs["beta"].unsqueeze(-1)
            + kwargs["A_log"].view(1, 1, -1, 1)
            + kwargs["dt_bias"].view(1, 1, q.shape[2], q.shape[3])
        )
        return output, None

    monkeypatch.setattr(kimi_k3_dspark, "_CUDA_MAX_GRID_DIM_Z", 8)
    monkeypatch.setattr(kimi_k3_dspark, "_load_fla_chunk_kda", lambda: fake_chunk_kda)

    shape = (5, 2, 3, 4)
    q, k, v, gate = (torch.randn(shape, requires_grad=True) for _ in range(4))
    beta = torch.randn(shape[:-1], requires_grad=True)
    A_log = torch.randn(shape[2], requires_grad=True)
    dt_bias = torch.randn(shape[2] * shape[3], requires_grad=True)
    output = kimi_k3_dspark._fla_kda(
        q,
        k,
        v,
        gate,
        beta,
        A_log,
        dt_bias,
        lower_bound=-5.0,
    )

    assert calls == [2, 2, 1]
    assert output.shape == shape
    output.sum().backward()
    for tensor in (q, k, v, gate, beta, A_log, dt_bias):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


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
