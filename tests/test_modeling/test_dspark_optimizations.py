import copy
from unittest import mock

import pytest
import torch
import torch.nn.functional as functional

from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
from specforge.algorithms.dspark.providers import resume_contract
from specforge.algorithms.model_providers import build_dspark_model
from specforge.config import Config
from tests.test_modeling.test_dspark_causal import build_model


@pytest.mark.parametrize("flatten,cache", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("chunk_size", [0, 2])
@pytest.mark.parametrize(
    "dtype,device",
    [(torch.float64, "cpu"), (torch.float32, "cuda"), (torch.bfloat16, "cuda")],
)
def test_loss_metrics_and_all_gradients_match(
    flatten, cache, chunk_size, dtype, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(33)
    baseline = build_model().to(device=device, dtype=dtype)
    baseline.num_anchors = 5
    baseline.objective_chunk_blocks = chunk_size
    baseline.loss_decay_gamma = 7.0
    optimized = copy.deepcopy(baseline)
    optimized.flatten_projection = flatten
    optimized.cache_projection = cache
    tokens = torch.randint(0, 62, (2, 32), device=device)
    hidden = torch.randn(2, 32, 256, device=device, dtype=dtype)
    target = torch.randn(2, 64, 128, device=device, dtype=dtype)[:, ::2]
    mask = torch.ones_like(tokens)
    mask[:, :4] = 0
    mask[:, 22:25] = 0
    outputs = []
    gradients = []
    for model in (baseline, optimized):
        torch.manual_seed(71)
        output = model(tokens, hidden, mask, target_last_hidden_states=target)
        output[0].backward()
        outputs.append(output)
        gradients.append(
            {
                name: parameter.grad.detach().float().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
        )
    tolerance = (
        dict(rtol=0, atol=0)
        if not flatten
        else dict(
            rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
            atol=0.002 if dtype == torch.bfloat16 else 1e-6,
        )
    )
    torch.testing.assert_close(outputs[0][0], outputs[1][0], **tolerance)
    torch.testing.assert_close(outputs[0][1], outputs[1][1], **tolerance)
    for name, terms in outputs[0][2]["ratio_metrics"].items():
        for expected, actual in zip(terms, outputs[1][2]["ratio_metrics"][name]):
            torch.testing.assert_close(expected, actual, **tolerance, msg=name)
    assert gradients[0].keys() == gradients[1].keys()
    for name, expected in gradients[0].items():
        torch.testing.assert_close(expected, gradients[1][name], **tolerance, msg=name)
    expected = torch.cat([value.flatten() for value in gradients[0].values()])
    actual = torch.cat([value.flatten() for value in gradients[1].values()])
    relative_error = (actual - expected).norm() / expected.norm()
    cosine = functional.cosine_similarity(expected, actual, dim=0)
    assert relative_error < (0.01 if dtype == torch.bfloat16 else 1e-5)
    assert cosine > 0.9999
    assert baseline.state_dict().keys() == optimized.state_dict().keys()
    assert resume_contract(None, baseline.draft_model, baseline) == resume_contract(
        None, optimized.draft_model, optimized
    )


@pytest.mark.parametrize("trainable_head", [False, True])
def test_cache_reuses_only_frozen_projection_gemms(trainable_head):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA kernel events")
    model = build_model(device="cuda")
    model.flatten_projection = True
    model.cache_projection = True
    model.lm_head.requires_grad_(trainable_head)
    tokens = torch.randint(0, 62, (2, 12), device="cuda")
    hidden = torch.randn(2, 12, 256, device="cuda")
    target = torch.randn(2, 12, 128, device="cuda")
    counts = []
    for enabled in (False, True):
        model.cache_projection = enabled
        model.zero_grad(set_to_none=True)
        torch.manual_seed(71)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as profiler:
            loss, _, _ = model(
                tokens,
                hidden,
                torch.ones_like(tokens),
                target_last_hidden_states=target,
            )
            loss.backward()
            torch.cuda.synchronize()
        counts.append(
            sum(
                len(event.kernels)
                for event in profiler.events()
                if event.name == "aten::mm"
                and event.input_shapes == [[8, 128], [128, 64]]
            )
        )
    assert counts[0] > 0
    assert counts[1] == (counts[0] if trainable_head else counts[0] // 2)


@pytest.mark.parametrize(
    "option", ["dspark_flatten_projection", "dspark_cache_projection"]
)
def test_config_flags_are_dspark_only_and_reach_model(option):
    payload = {
        "model": {"target_model_path": "unused"},
        "data": {"hidden_states_path": "unused"},
        "training": {"strategy": "dspark", option: True},
    }
    config = Config.model_validate(payload)
    reference = build_model()
    common = {
        "draft_model": reference.draft_model,
        "target_lm_head": reference.lm_head,
        "target_embed_tokens": reference.embed_tokens,
        "mask_token_id": 63,
        "block_size": 4,
    }
    with mock.patch(
        "specforge.algorithms.model_providers._build_dflash_family_model",
        side_effect=lambda config, draft, tokenizer, factory: factory(common),
    ):
        model = build_dspark_model(config, reference.draft_model, None, None, None)
    assert isinstance(model, OnlineDSparkModel)
    assert getattr(model, option.removeprefix("dspark_")) is True
    for strategy in ("dflash", "domino", "dspine", "eagle3"):
        payload["training"]["strategy"] = strategy
        with pytest.raises(ValueError, match="require training.strategy=dspark"):
            Config.model_validate(payload)
