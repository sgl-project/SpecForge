"""Objective-window and resume contracts for the native TorchTitan runtime."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torchtitan")

from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.trainer import Trainer as TitanTrainer

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.training.torchtitan.data import FeatureDataLoader
from specforge.training.torchtitan.loss import SpecForgeObjectiveLoss
from specforge.training.torchtitan.runtime import (
    SpecForgeTitanTrainer,
    partition_anchor_blocks,
)


def preparation_model(loss_type="dflash", gamma=2.0):
    model = OnlineDFlashModel.__new__(OnlineDFlashModel)
    torch.nn.Module.__init__(model)
    model.block_size = 4
    model.num_anchors = 3
    model.loss_type = loss_type
    model.loss_decay_gamma = gamma
    model.selector_loss_alpha = 0.75
    model.selector_warmup_ratio = 0.0
    model.selector_ramp_ratio = 0.0
    return model


def test_pipeline_rejects_nonlinear_lk_lambda_before_titan_initialization(monkeypatch):
    def unexpected_initialization(*args, **kwargs):
        pytest.fail("An unsupported objective must fail before distributed setup")

    monkeypatch.setattr(TitanTrainer, "__init__", unexpected_initialization)
    config = SimpleNamespace(
        training=SimpleNamespace(disable_cuda_graphs=True),
        parallelism=SimpleNamespace(pipeline_parallel_degree=2),
        model_spec=SimpleNamespace(
            model=SimpleNamespace(objective={"lk_loss_type": "lambda"})
        ),
    )
    with pytest.raises(
        ValueError, match="Pipeline parallelism does not support LK-lambda"
    ):
        SpecForgeTitanTrainer(config)


@pytest.mark.parametrize("gamma", [None, 2.0])
def test_prepared_denominator_matches_manual_mask_and_decay(gamma):
    model = preparation_model(gamma=gamma)
    mask = torch.tensor([[0, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1]])
    anchors = torch.tensor([[1, 2, 0], [0, 2, 4]])
    keep = torch.tensor([[True, True, False], [True, True, True]])
    expected = 0.0
    for row in range(2):
        for column in range(3):
            for offset in range(1, 4):
                target = anchors[row, column].item() + offset
                if keep[row, column] and target < 6 and mask[row, target]:
                    expected += (
                        1
                        if gamma is None
                        else torch.exp(torch.tensor(-(offset - 1) / gamma)).item()
                    )
    actual = model.prepared_objective_denominator(mask, anchors, keep)
    torch.testing.assert_close(actual, torch.tensor(expected))


@pytest.mark.parametrize(
    "loss_type", ["dflash", "dpace", "dpace-continuation-value-only"]
)
def test_block_context_partition_preserves_objective_measure(loss_type):
    model = preparation_model(loss_type=loss_type)
    mask = torch.tensor([[1, 1, 1, 1, 0, 1, 1]])
    anchors = torch.tensor([[0, 1, 5]])
    keep = torch.ones_like(anchors, dtype=torch.bool)
    full_den = model.prepared_objective_denominator(mask, anchors, keep)
    _, full_weights = model._dflash_weight_mask(mask, anchors, keep)
    global_scale = model._sequence_anchor_scale(full_weights).squeeze(-1)
    local_dens = []
    for rank in range(4):
        local_anchors, local_keep = partition_anchor_blocks(
            anchors, keep, rank=rank, degree=4
        )
        assert local_anchors.shape == (1, 1)
        if loss_type == "dflash":
            local_dens.append(
                model.prepared_objective_denominator(mask, local_anchors, local_keep)
            )
        else:
            local_scale, _ = partition_anchor_blocks(
                global_scale, keep, rank=rank, degree=4
            )
            _, weights = model._dflash_weight_mask(mask, local_anchors, local_keep)
            local_dens.append(((weights > 0).any(-1).float() * local_scale).sum())
    torch.testing.assert_close(torch.stack(local_dens).sum(), full_den)


def test_window_loss_gradient_is_global_ratio_not_mean_of_local_means():
    criterion = SpecForgeObjectiveLoss.Config().build()
    weight = torch.tensor(2.0, requires_grad=True)
    denominator = torch.tensor(11.0)
    for x, local_den in [(3.0, 1.0), (5.0, 10.0)]:
        numerator = (weight - x).square()
        output = (
            numerator / local_den,
            torch.tensor(0),
            {
                "loss_terms": (numerator, torch.tensor(local_den)),
                "_torchtitan_normalizer": denominator,
            },
        )
        loss, _ = criterion(output, torch.tensor([1]), torch.tensor(999))
        loss.backward()
    torch.testing.assert_close(weight.grad, torch.tensor(-8.0 / 11.0))


def test_dspark_pipeline_chunks_share_logical_batch_denominator():
    criterion = SpecForgeObjectiveLoss.Config(algorithm="dspark").build()
    weight = torch.tensor(1.0, requires_grad=True)
    # Unequal valid counts in two physical PP microbatches must produce the
    # logical local-batch ratio, then average the two accumulation groups.
    for numerators, denominators in [((1, 20), (1, 10)), ((9, 24), (3, 6))]:
        normalizer = torch.tensor(float(sum(denominators) * 2))
        for numerator, denominator in zip(numerators, denominators):
            raw = weight * numerator
            output = (
                999 * raw,
                torch.tensor(0),
                {
                    "loss_terms": (raw, torch.tensor(float(denominator))),
                    "_torchtitan_normalizer": normalizer,
                },
            )
            loss, _ = criterion(output, None, None)
            packed_loss, _ = criterion(torch.stack([raw, normalizer]), None)
            torch.testing.assert_close(packed_loss, loss)
            loss.backward()
    torch.testing.assert_close(weight.grad, torch.tensor((21 / 11 + 33 / 9) / 2))


def test_native_trainer_keeps_training_and_backward_implementations():
    for name in (
        "train",
        "forward_backward_step",
        "_forward_backward_body",
        "pp_forward_backward_step",
    ):
        assert getattr(SpecForgeTitanTrainer, name) is getattr(TitanTrainer, name)


def test_objective_logger_omits_only_token_derived_maximum_and_forwards_close():
    from torchtitan.components.metrics import BaseLogger

    from specforge.training.torchtitan.metrics import ObjectiveMetricLogger

    class Recorder(BaseLogger):
        def __init__(self):
            self.calls = []
            self.closed = False

        def log(self, metrics, step):
            self.calls.append((metrics, step))

        def close(self):
            self.closed = True

    recorder = Recorder()
    logger = ObjectiveMetricLogger(recorder)
    native_metrics = {
        "loss_metrics/global_avg_loss": 1.25,
        "loss_metrics/global_max_loss": float("nan"),
        "grad_norm": 0.5,
        "throughput(tps)": 128.0,
        "memory/max_active(GiB)": 2.0,
        "optimizer/lr": 0.001,
    }
    logger.log(native_metrics, 7)
    expected = dict(native_metrics)
    del expected["loss_metrics/global_max_loss"]
    assert recorder.calls == [(expected, 7)]
    assert "loss_metrics/global_max_loss" in native_metrics
    validation = {"validation_metrics/loss": 1.0}
    logger.log(validation, 8)
    assert recorder.calls[-1] == (validation, 8)
    logger.close()
    assert recorder.closed


@pytest.mark.parametrize("tp_enabled", [False, True])
def test_training_step_delegates_to_titan_with_only_scoped_norm_compatibility(
    monkeypatch, tp_enabled
):
    import specforge.training.torchtitan.runtime as runtime

    events = []

    class NormScope:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, *args):
            events.append("exit")

    def native_step(self, iterator):
        events.append("native")
        return iterator

    monkeypatch.setattr(runtime, "HeterogeneousGradientNorms", NormScope)
    monkeypatch.setattr(TitanTrainer, "train_step", native_step)
    trainer = SpecForgeTitanTrainer.__new__(SpecForgeTitanTrainer)
    trainer.parallel_dims = SimpleNamespace(tp_enabled=tp_enabled)
    sentinel = object()
    assert trainer.train_step(sentinel) is sentinel
    assert events == (["enter", "native", "exit"] if tp_enabled else ["native"])


def test_mesh_optimizer_groups_preserve_checkpoint_parameter_names(monkeypatch):
    from torchtitan.components.optimizer.utils import get_flat_optim_state_dict

    import specforge.training.torchtitan.parallelize as parallelize

    class MeshParameter(torch.nn.Parameter):
        pass

    monkeypatch.setattr(parallelize, "DTensor", MeshParameter)
    parameters = [MeshParameter(torch.tensor(float(index))) for index in range(3)]
    for parameter, mesh in zip(parameters, ("dp", "dp_tp", "dp")):
        parameter.device_mesh = mesh
    names = ["fc.weight", "layers.0.q_proj.weight", "norm.weight"]
    optimizer = torch.optim.AdamW(
        [{"params": parameters, "param_names": names, "lr": 0.01, "weight_decay": 0.0}]
    )
    parallelize.group_optimizer_parameters_by_mesh(
        [optimizer], [], SimpleNamespace(tp_enabled=True)
    )
    assert [group["param_names"] for group in optimizer.param_groups] == [
        ["fc.weight", "norm.weight"],
        ["layers.0.q_proj.weight"],
    ]
    for index, parameter in enumerate(parameters):
        parameter.grad = torch.full_like(parameter, index + 1)
    optimizer.step()
    state = get_flat_optim_state_dict(optimizer)
    assert {
        name.removeprefix("state.").removesuffix(".exp_avg")
        for name in state
        if name.endswith(".exp_avg")
    } == set(names)
    for index, name in enumerate(names):
        torch.testing.assert_close(
            state[f"state.{name}.exp_avg"], torch.tensor((index + 1) * 0.1)
        )


def source_factory(**kwargs):
    for index in range(4):
        yield {
            "input_ids": torch.full(
                (1, 6), kwargs["epoch"] * 10 + index, dtype=torch.long
            ),
            "loss_mask": torch.ones(1, 6, dtype=torch.long),
            "hidden_states": torch.zeros(1, 6, 8),
        }


def test_dataloader_cursor_resumes_next_batch_and_epoch():
    config = FeatureDataLoader.Config(source_factory=source_factory, epochs=2)
    loader = config.build(dp_rank=0, dp_world_size=1, local_batch_size=1)
    iterator = iter(loader)
    next(iterator)
    next(iterator)
    resumed = config.build(dp_rank=0, dp_world_size=1, local_batch_size=1)
    resumed.load_state_dict(loader.state_dict())
    observed = [int(inputs["input"][0, 0]) for inputs, _ in resumed]
    assert observed == [2, 3, 10, 11, 12, 13]
    iterator.close()


@pytest.mark.parametrize("epoch,cursor", [(-1, 0), (0, -1), (2, 0), (1, 1)])
def test_dataloader_rejects_invalid_checkpoint_position(epoch, cursor):
    loader = FeatureDataLoader.Config(source_factory=source_factory).build(
        dp_rank=0, dp_world_size=1
    )
    state = {"dp_rank_0": {"epoch": epoch, "cursor": cursor, "dp_world_size": 1}}
    with pytest.raises(ValueError, match="Invalid saved feature"):
        loader.load_state_dict(state)


def test_dataloader_rejects_cursor_past_end_without_silent_replay():
    loader = FeatureDataLoader.Config(source_factory=source_factory).build(
        dp_rank=0, dp_world_size=1
    )
    loader.load_state_dict({"dp_rank_0": {"epoch": 0, "cursor": 5, "dp_world_size": 1}})
    with pytest.raises(ValueError, match="cursor exceeds"):
        next(iter(loader))


def test_pipeline_padding_keeps_feature_alignment_and_masks_padding():
    config = FeatureDataLoader.Config(
        source_factory=source_factory, pad_to_seq_len=True
    )
    loader = config.build(dp_rank=0, dp_world_size=1, local_batch_size=1, seq_len=8)
    inputs, labels = next(iter(loader))
    assert inputs["input"].shape == (1, 8)
    assert inputs["hidden_states"].shape == (1, 8, 8)
    assert inputs["loss_mask"].tolist() == [[1, 1, 1, 1, 1, 1, 0, 0]]
    assert labels[0, -2:].tolist() == [-100, -100]


def test_checkpoint_restores_next_anchor_sample_and_rejects_changed_contract(
    monkeypatch,
):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        torch.cuda, "get_rng_state", lambda device: torch.tensor([7], dtype=torch.uint8)
    )
    restored_cuda = []
    monkeypatch.setattr(
        torch.cuda, "set_rng_state", lambda state, device: restored_cuda.append(state)
    )
    trainer = SpecForgeTitanTrainer.__new__(SpecForgeTitanTrainer)
    trainer._anchor_generator = torch.Generator().manual_seed(45)
    trainer.config = SimpleNamespace(resume_contract={"teacher": "fingerprint"})
    trainer.step = 4
    trainer.ntokens_seen = 96
    trainer.device = "cuda"
    state = trainer.state_dict()
    expected = torch.rand(12, generator=trainer._anchor_generator)
    trainer.load_state_dict(state)
    torch.testing.assert_close(
        torch.rand(12, generator=trainer._anchor_generator), expected
    )
    assert len(restored_cuda) == 1
    trainer.config.resume_contract = {"teacher": "different"}
    with pytest.raises(ValueError, match="training contract changed"):
        trainer.load_state_dict(state)


def test_native_window_preparation_has_one_denominator_and_no_partial_update():
    trainer = SpecForgeTitanTrainer.__new__(SpecForgeTitanTrainer)
    model = preparation_model()
    trainer.model_parts = [SimpleNamespace(training_model=model)]
    trainer.config = SimpleNamespace(
        model_spec=SimpleNamespace(model=SimpleNamespace(algorithm="dflash")),
        schedule_total_steps=100,
        training=SimpleNamespace(steps=2),
    )
    mesh = SimpleNamespace(size=lambda: 1, get_local_rank=lambda: 0)
    trainer.parallel_dims = SimpleNamespace(
        get_optional_mesh=lambda *args, **kwargs: mesh, pp_enabled=False
    )
    trainer.metrics_processor = SimpleNamespace(
        should_log=lambda step: False,
        ntokens_since_last_log=0,
        data_loading_times=[],
    )
    trainer.gradient_accumulation_steps = 2
    trainer.num_pipeline_parallel_microbatches = 1
    trainer._anchor_generator = torch.Generator().manual_seed(1)
    trainer.device = torch.device("cpu")
    trainer.step = 1
    batches = []
    for lengths in (6, 4, 3):
        mask = torch.ones(1, lengths, dtype=torch.long)
        batches.append(({"input": mask, "loss_mask": mask}, mask))
    iterator = trainer.batch_generator(batches)
    first, _ = next(iterator)
    second, _ = next(iterator)
    expected = sum(
        model.prepared_objective_denominator(
            x["loss_mask"], x["anchor_positions"], x["block_keep_mask"]
        )
        for x in (first, second)
    )
    torch.testing.assert_close(first["objective_normalizer"], expected)
    assert first["objective_normalizer"] is second["objective_normalizer"]
    with pytest.raises(DataloaderExhaustedError):
        next(iterator)


@pytest.mark.parametrize("algorithm", ["dflash", "dflash2", "dspark"])
def test_pipeline_stages_preserve_objective_and_all_context_gradients(algorithm):
    from specforge.training.torchtitan.model import SpecForgeTitanModel
    from specforge.training.torchtitan.pipeline import SpecForgePipelineStage

    config = SpecForgeTitanModel.Config(
        algorithm=algorithm,
        draft_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_hidden_layers": 2,
            "num_target_layers": 4,
            "head_dim": 8,
            "vocab_size": 32,
            "max_position_embeddings": 32,
            "dflash_config": {
                "block_size": 4,
                "mask_token_id": 31,
                "target_layer_ids": [1, 2],
                "conv_group_size": 4,
                "conv_kernel_size": 2,
                "selector_rank": 4,
                "selector_top_k": 4,
                "markov_rank": 4,
                "enable_confidence_head": True,
            },
        },
        objective={
            "attention_backend": "eager",
            "num_anchors": 3,
            "objective_chunk_blocks": 0,
            "loss_decay_gamma": 2.0,
        },
    )
    torch.manual_seed(21)
    full = config.build().bfloat16()
    stages = [
        SpecForgePipelineStage(full, stage_index=i, num_stages=2) for i in range(2)
    ]
    input_ids = torch.randint(0, 31, (1, 8))
    mask = torch.tensor([[0, 1, 1, 1, 1, 0, 1, 1]])
    anchors = torch.tensor([[1, 2, 6]])
    keep = torch.ones_like(anchors, dtype=torch.bool)
    denominator = full.training_model.prepared_objective_denominator(
        mask, anchors, keep
    )
    inputs = {
        "hidden_states": torch.randn(1, 8, 32).bfloat16(),
        "target_last_hidden_states": torch.randn(1, 8, 16).bfloat16(),
        "loss_mask": mask,
        "anchor_positions": anchors,
        "block_keep_mask": keep,
        "objective_normalizer": denominator,
        "collect_detailed_metrics": False,
    }
    criterion = SpecForgeObjectiveLoss.Config(algorithm=algorithm).build()
    full_output = full(input_ids, **inputs)
    # Prepared CPU measure also matches the actual objective's denominator.
    torch.testing.assert_close(full_output[2]["loss_terms"][1], denominator)
    full_loss, _ = criterion(full_output, None)
    full_loss.backward()
    hidden, context = stages[0](input_ids, source_input_ids=input_ids, **inputs)
    staged_output = stages[1](hidden, context, source_input_ids=input_ids, **inputs)
    staged_loss, _ = criterion(staged_output, None)
    staged_loss.backward()
    torch.testing.assert_close(staged_loss, full_loss, rtol=0, atol=0)
    gradients = {}
    for stage in stages:
        for name, parameter in stage.named_parameters():
            if parameter.requires_grad:
                assert (
                    name not in gradients
                ), f"Trainable parameter duplicated over stages: {name}"
                gradients[name] = parameter.grad
    full_parameters = {
        name: value for name, value in full.named_parameters() if value.requires_grad
    }
    assert gradients.keys() == full_parameters.keys()
    for name, parameter in full_parameters.items():
        if parameter.grad is None:
            assert gradients[name] is None
        else:
            torch.testing.assert_close(
                gradients[name], parameter.grad, rtol=0.02, atol=0.001, msg=name
            )
