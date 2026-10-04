"""DSpine causal isolation, refinement parity, and persistent transfer state."""

import copy
import json
import math

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn
from transformers import Qwen3Config

from scripts.convert_dspine_checkpoint import convert_checkpoint
from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.algorithms.common.dflash_family_model import (
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
)
from specforge.algorithms.dspine.model import OnlineDSpineModel
from specforge.config import Config
from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.dspine import DSpineConfig, DSpineDraftModel
from specforge.runtime.contracts import TrainBatch
from specforge.training.strategies.base import StepContext


def tiny_config(head_dim=4, **options):
    return Qwen3Config(
        architectures=["DSpineDraftModel"],
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=16 // head_dim,
        num_key_value_heads=min(2, 16 // head_dim),
        num_hidden_layers=3,
        num_target_layers=8,
        head_dim=head_dim,
        vocab_size=32,
        rms_norm_eps=1e-6,
        block_size=4,
        layer_types=["full_attention"] * 3,
        dflash_config={"mask_token_id": 31, "target_layer_ids": [1, 4, 6]},
        dspine_config={"transfer_rank": 8, "message_size": 4, "top_k": 3, **options},
        attn_implementation="sdpa",
    )


def initialized_model(**options):
    torch.manual_seed(17)
    model = DSpineDraftModel(tiny_config(**options))
    head = torch.randn(32, 16)
    model.initialize_transfer_space(head, chunk_size=7)
    return model, head


def forward_inputs(model):
    anchors = torch.tensor([[2, 5], [1, 4]])
    length = 7
    token_ids = torch.randint(0, 30, (2, 2, 4))
    positions = torch.cat(
        (
            torch.arange(length).expand(2, -1),
            (anchors[..., None] + torch.arange(4)).flatten(1),
        ),
        dim=1,
    )
    return dict(
        position_ids=positions,
        attention_mask=create_dflash_sdpa_mask(
            anchors,
            torch.ones_like(anchors, dtype=torch.bool),
            length,
            4,
            "cpu",
            causal_block=True,
        ),
        noise_embedding=torch.randn(2, 8, 16),
        target_hidden=torch.randn(2, length, 48),
        block_token_ids=token_ids,
    )


def test_zero_writes_are_identity_against_the_shared_causal_backbone():
    model, _ = initialized_model()
    inputs = forward_inputs(model)
    backbone = DFlashDraftModel(copy.deepcopy(model.config))
    backbone.load_state_dict(
        {
            key: value
            for key, value in model.state_dict().items()
            if key in backbone.state_dict()
        },
        strict=True,
    )
    expected = backbone(
        **{key: value for key, value in inputs.items() if key != "block_token_ids"}
    )
    actual = model(**inputs)
    torch.testing.assert_close(
        actual.hidden_states.flatten(1, 2), expected, atol=0, rtol=0
    )
    torch.testing.assert_close(
        model.injection.predecessor_gate.weight,
        torch.zeros_like(model.injection.predecessor_gate.weight),
    )


def test_future_slots_and_other_blocks_cannot_change_a_prefix():
    model, _ = initialized_model()
    with torch.no_grad():
        for write in model.injection.writes:
            write.weight.normal_(std=0.1)
    inputs = forward_inputs(model)
    expected = model(**inputs).hidden_states
    changed = {**inputs, "noise_embedding": inputs["noise_embedding"].clone()}
    changed["noise_embedding"][:, 2:] += 10 * torch.randn_like(
        changed["noise_embedding"][:, 2:]
    )
    actual = model(**changed).hidden_states
    torch.testing.assert_close(actual[:, 0, :2], expected[:, 0, :2], atol=0, rtol=0)
    changed = {**inputs, "block_token_ids": inputs["block_token_ids"].clone()}
    changed["block_token_ids"][..., 1:] = (changed["block_token_ids"][..., 1:] + 3) % 30
    torch.testing.assert_close(model(**changed).hidden_states, expected, atol=0, rtol=0)


def test_injection_preserves_anchors_and_only_replaces_shallow_predecessors():
    model, _ = initialized_model(replacement_layers=1)
    with torch.no_grad():
        for write in model.injection.writes:
            write.weight.normal_(std=0.1)
    hidden = torch.randn(2, 2, 4, 16)
    labels = torch.randint(0, 30, (2, 2, 4))
    replace = torch.tensor([[True, False], [False, True]])
    for layer in range(3):
        normal = model.injection(hidden, layer, labels, model.transfer_codes)
        replaced = model.injection(hidden, layer, labels, model.transfer_codes, replace)
        torch.testing.assert_close(
            replaced[..., 0, :], hidden[..., 0, :], atol=0, rtol=0
        )
        torch.testing.assert_close(replaced[~replace], normal[~replace], atol=0, rtol=0)
        if layer:
            torch.testing.assert_close(replaced, normal, atol=0, rtol=0)
        else:
            assert not torch.equal(replaced[replace, 2:], normal[replace, 2:])


def test_transition_cache_matches_sequential_last_write_selection():
    model, head = initialized_model()
    with torch.no_grad():
        model.injection.writes[-1].weight.normal_(std=0.1)
    hidden = torch.randn(2, 2, 3, 16)
    candidates = torch.randn(2, 2, 3, 32).topk(3).indices
    anchor = torch.randint(0, 32, (2, 2))
    scores = model.transition_scores(hidden, candidates, anchor, head)
    cached = model.select_cached(scores, candidates)
    predecessor = anchor
    selected = []
    for position in range(3):
        refined = model.refine(hidden[..., position, :], predecessor)
        direct = torch.einsum(
            "...h,...kh->...k", refined, F.embedding(candidates[..., position, :], head)
        )
        predecessor = (
            candidates[..., position, :]
            .gather(-1, direct.argmax(-1)[..., None])
            .squeeze(-1)
        )
        selected.append(predecessor)
    torch.testing.assert_close(cached, torch.stack(selected, dim=-1), atol=0, rtol=0)
    # Every cached row must match direct refinement, not only the selected path.
    for position in range(3):
        for index in range(3):
            predecessor = (
                anchor if position == 0 else candidates[..., position - 1, index]
            )
            refined = model.refine(hidden[..., position, :], predecessor)
            direct = torch.einsum(
                "...h,...kh->...k",
                refined,
                F.embedding(candidates[..., position, :], head),
            )
            torch.testing.assert_close(scores[..., position, index, :], direct)


def test_zero_state_refinement_has_finite_gradients():
    model, _ = initialized_model()
    hidden = torch.zeros(2, 16, requires_grad=True)
    model.refine(hidden, torch.tensor([1, 2])).sum().backward()
    assert torch.isfinite(hidden.grad).all()


@pytest.mark.parametrize("window", [None, 3])
def test_flex_and_dense_causal_masks_agree(window):
    anchors = torch.tensor([[2, 5], [1, 4]])
    keep = torch.tensor([[True, True], [True, False]])
    arguments = dict(
        anchor_positions=anchors,
        block_keep_mask=keep,
        S=7,
        block_size=4,
        device="cpu",
        sliding_window=window,
        causal_block=True,
    )
    dense = create_dflash_sdpa_mask(**arguments)
    block_mask = create_dflash_block_mask(**arguments)
    for batch in range(2):
        for query in range(8):
            keys = torch.arange(15)
            actual = block_mask.mask_mod(
                torch.tensor(batch), torch.tensor(0), torch.tensor(query), keys
            )
            torch.testing.assert_close(actual, dense[batch, 0, query])
    assert not dense[1, 0, 4:].any()
    assert not dense[0, 0, 0, 8:11].any()


def test_fixed_whitened_codes_and_learned_readouts_survive_hf_reload(tmp_path):
    model, _ = initialized_model()
    assert not model.transfer_codes.requires_grad
    torch.testing.assert_close(model.transfer_codes.norm(dim=-1), torch.ones(32))
    with torch.no_grad():
        model.injection.readouts[0].weight.add_(0.2)
    inputs = forward_inputs(model)
    expected = model(**inputs).hidden_states
    model.save_pretrained(tmp_path)
    reloaded = DSpineDraftModel.from_pretrained(tmp_path, attn_implementation="sdpa")
    assert reloaded.transfer_ready.item()
    torch.testing.assert_close(
        reloaded(**inputs).hidden_states, expected, atol=0, rtol=0
    )
    torch.testing.assert_close(
        reloaded.transfer_codes, model.transfer_codes, atol=0, rtol=0
    )


def test_converter_retains_reference_weights_and_codes_without_reinitializing(tmp_path):
    model, _ = initialized_model()
    state = {
        name: tensor
        for name, tensor in model.state_dict().items()
        if name not in {"transfer_codes", "transfer_ready"}
    }
    save_file(state, tmp_path / "model.safetensors")
    save_file({"codes": model.transfer_codes}, tmp_path / "transfer.safetensors")
    (tmp_path / "config.json").write_text(
        json.dumps({"config": model.config.to_dict()})
    )
    output = tmp_path / "export"
    convert_checkpoint(
        tmp_path / "model.safetensors",
        tmp_path / "config.json",
        tmp_path / "transfer.safetensors",
        output,
    )
    reloaded = DSpineDraftModel.from_pretrained(output)
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(
            reloaded.state_dict()[name], expected, atol=0, rtol=0
        )
    assert reloaded.block_size == 16
    with pytest.raises(FileExistsError):
        convert_checkpoint(
            tmp_path / "model.safetensors",
            tmp_path / "config.json",
            tmp_path / "transfer.safetensors",
            output,
        )


def test_registry_resolves_dspine_and_uninitialized_transfer_space_is_rejected():
    model = AutoDraftModel.from_config(tiny_config())
    assert isinstance(model, DSpineDraftModel)
    with pytest.raises(RuntimeError, match="transfer space"):
        model(**forward_inputs(model))


@pytest.mark.parametrize(
    "options",
    [
        {"transfer_rank": 17},
        {"top_k": 0},
        {"message_size": 0},
        {"alignment_weight": float("nan")},
        {"replacement_hold_ratio": 0.5},
        {"top_k": 2.5},
    ],
)
def test_invalid_options_fail_before_training(options):
    with pytest.raises(ValueError, match="DSpine"):
        DSpineDraftModel(tiny_config(**options))


def test_schedules_reach_the_paper_endpoints():
    options = DSpineConfig()
    assert options.schedule(0, 600) == (0, 0.5)
    assert options.schedule(100, 600)[1] == 0.5
    assert options.schedule(150, 600)[1] == pytest.approx(0.25)
    assert options.schedule(200, 600)[1] == 0
    assert options.schedule(146, 600)[0] == 0.5
    assert options.schedule(0, None)[1] == 0


def training_model(chunk_size=0, head_dim=4):
    draft, head_weight = initialized_model(head_dim=head_dim)
    head = nn.Linear(16, 32, bias=False)
    head.weight.data.copy_(head_weight)
    with torch.no_grad():
        for write in draft.injection.writes:
            write.weight.normal_(std=0.05)
    return OnlineDSpineModel(
        draft_model=draft,
        target_lm_head=head,
        target_embed_tokens=nn.Embedding(32, 16),
        mask_token_id=31,
        block_size=4,
        num_anchors=4,
        attention_backend="sdpa",
        objective_chunk_blocks=chunk_size,
    )


def training_inputs():
    return dict(
        input_ids=torch.randint(0, 30, (2, 12)),
        hidden_states=torch.randn(2, 12, 48),
        target_last_hidden_states=torch.randn(2, 12, 16),
        loss_mask=torch.tensor(
            [[0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1]]
        ),
    )


@pytest.mark.parametrize("chunk_size", [1, 3])
def test_chunked_objective_matches_full_loss_and_all_parameter_gradients(chunk_size):
    full = training_model()
    chunked = copy.deepcopy(full)
    chunked.objective_chunk_blocks = chunk_size
    inputs = training_inputs()
    torch.manual_seed(42)
    expected, _, _ = full(**inputs, global_step=100, total_steps=600)
    expected.backward()
    torch.manual_seed(42)
    actual, _, _ = chunked(**inputs, global_step=100, total_steps=600)
    actual.backward()
    torch.testing.assert_close(actual, expected)
    for name, parameter in full.draft_model.named_parameters():
        compared = dict(chunked.draft_model.named_parameters())[name]
        assert parameter.grad is not None and compared.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(compared.grad, parameter.grad, atol=2e-5, rtol=2e-4)
    assert full.lm_head.weight.grad is None
    assert full.embed_tokens.weight.grad is None
    assert full.draft_model.transfer_codes.grad is None


def test_tail_after_a_mask_gap_does_not_contribute_teacher_loss(monkeypatch):
    model = training_model()
    inputs = training_inputs()
    inputs = {name: value[:1] for name, value in inputs.items()}
    inputs["loss_mask"] = torch.tensor([[0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]])
    monkeypatch.setattr(
        model,
        "_sample_anchor_positions",
        lambda *args, **kwargs: (torch.tensor([[1]]), torch.tensor([[True]])),
    )
    expected, _, metrics = model(**inputs, global_step=300, total_steps=600)
    torch.testing.assert_close(
        metrics["loss_terms"][1], torch.tensor(1 + math.exp(-1 / 7))
    )
    inputs["target_last_hidden_states"][:, 3:] += 100
    actual, _, _ = model(**inputs, global_step=300, total_steps=600)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_refinement_ignores_missing_labels_and_negligible_teacher_mass():
    model = training_model()
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.weight[:, 0] = torch.arange(32)
    hidden = torch.zeros(1, 3, 16)
    hidden[..., 0] = 10
    teacher = -hidden
    terms = model._objective_terms(
        hidden,
        torch.randn_like(hidden),
        torch.randn(1, 3, 3, 8),
        torch.zeros(1, 3, dtype=torch.long),
        torch.zeros(1, 3, dtype=torch.long),
        teacher,
        torch.ones(1, 3),
        torch.ones(1, 3, dtype=torch.bool),
    )
    assert terms[2].item() == 0
    assert all(torch.isfinite(term) for term in terms)


def test_strategy_runs_an_optimizer_step_and_exports_the_fixed_codes():
    model = training_model(chunk_size=2)
    registration = builtin_algorithm_registry().resolve("dspine")
    strategy = registration.providers.step.build(model)
    inputs = training_inputs()
    batch = TrainBatch(["first", "second"], "dspine", inputs)
    before = model.draft_model.injection.writes[0].weight.detach().clone()
    output = strategy.forward_loss(batch, StepContext(global_step=100, total_steps=600))
    numerator, denominator = output.loss_terms
    torch.testing.assert_close(output.loss, numerator / denominator)
    output.loss.backward()
    optimizer = torch.optim.SGD(model.draft_model.parameters(), lr=0.01)
    optimizer.step()
    assert not torch.equal(before, model.draft_model.injection.writes[0].weight)
    exported = strategy.checkpoint_state_filter(model.state_dict())
    assert set(exported) == set(model.draft_model.state_dict())
    assert exported["transfer_ready"].item()
    torch.testing.assert_close(
        exported["transfer_codes"], model.draft_model.transfer_codes
    )
    with pytest.raises(ValueError, match="target_last_hidden_states"):
        strategy.forward_loss(
            TrainBatch(
                ["missing"],
                "dspine",
                {
                    key: value
                    for key, value in inputs.items()
                    if key != "target_last_hidden_states"
                },
            )
        )


def test_provider_loads_real_target_tensors_and_does_not_reset_a_resumed_transfer_space(
    tmp_path,
):
    target = tiny_config()
    target.save_pretrained(tmp_path)
    save_file(
        {
            "model.embed_tokens.weight": torch.randn(32, 16),
            "lm_head.weight": torch.randn(32, 16),
        },
        tmp_path / "model.safetensors",
    )
    config = Config.model_validate(
        dict(
            model=dict(
                target_model_path=str(tmp_path), torch_dtype="float32", mask_token_id=31
            ),
            data=dict(hidden_states_path=str(tmp_path / "features")),
            training=dict(strategy="dspine", attention_backend="sdpa", num_anchors=2),
        )
    )
    draft = DSpineDraftModel(tiny_config())
    provider = builtin_algorithm_registry().resolve("dspine").providers
    parts = provider.model.build_training_model(
        config, draft, draft.config, target, None
    )
    assert draft.transfer_ready.item()
    with torch.no_grad():
        draft.injection.readouts[0].weight.add_(1)
    learned = draft.injection.readouts[0].weight.detach().clone()
    codes = draft.transfer_codes.clone()
    again = provider.model.build_training_model(
        config, draft, draft.config, target, None
    )
    torch.testing.assert_close(
        draft.injection.readouts[0].weight, learned, atol=0, rtol=0
    )
    torch.testing.assert_close(draft.transfer_codes, codes, atol=0, rtol=0)
    assert parts.capture_layers == again.capture_layers == [1, 4, 6]
    contract = provider.step.resume_contract(config, draft, again.model)
    assert dict(contract["dspine_options"])["transfer_rank"] == 8


def test_offline_provider_retains_dspine_identity_and_teacher_features(tmp_path):
    inputs = {name: value[:1] for name, value in training_inputs().items()}
    torch.save(inputs, tmp_path / "sample.ckpt")
    registration = builtin_algorithm_registry().resolve("dspine")
    provider = registration.providers.offline_for("text")
    reader = provider.build_reader(
        str(tmp_path), run_id="dspine-test", ttt_length=4, max_len=12
    )
    records = reader.read()
    assert len(records) == 1
    assert records[0].strategy == "dspine"
    assert "target_last_hidden_states" in records[0].feature_keys
    assert (
        registration.providers.server_streaming_for("text").capture_method == "dspark"
    )


def test_bf16_training_keeps_frozen_state_and_gradients_finite():
    model = training_model(chunk_size=2).to(dtype=torch.bfloat16)
    inputs = training_inputs()
    for name in ("hidden_states", "target_last_hidden_states"):
        inputs[name] = inputs[name].to(torch.bfloat16)
    loss, _, _ = model(**inputs, global_step=150, total_steps=600)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.draft_model.parameters()
    )
    assert not model.draft_model.transfer_codes.requires_grad


@pytest.mark.parametrize(
    "step", [100, 300], ids=["teacher-replacement", "predicted-predecessors"]
)
def test_bf16_injection_and_refinement_accept_fp32_transfer_buffers(step):
    model = training_model(chunk_size=2).to(dtype=torch.bfloat16)
    model.draft_model.transfer_codes = model.draft_model.transfer_codes.float()
    inputs = training_inputs()
    for name in ("hidden_states", "target_last_hidden_states"):
        inputs[name] = inputs[name].to(torch.bfloat16)
    loss, _, _ = model(**inputs, global_step=step, total_steps=600)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.draft_model.transfer_codes.dtype == torch.float32
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.draft_model.parameters()
    )


def test_gradient_checkpointing_preserves_the_complete_training_objective():
    model = training_model(chunk_size=2)
    checkpointed = copy.deepcopy(model)
    checkpointed.draft_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    inputs = training_inputs()
    torch.manual_seed(100)
    expected, _, _ = model(**inputs, global_step=146, total_steps=600)
    expected.backward()
    torch.manual_seed(100)
    actual, _, _ = checkpointed(**inputs, global_step=146, total_steps=600)
    actual.backward()
    torch.testing.assert_close(actual, expected)
    for original, checked in zip(
        model.draft_model.parameters(), checkpointed.draft_model.parameters()
    ):
        torch.testing.assert_close(checked.grad, original.grad)
