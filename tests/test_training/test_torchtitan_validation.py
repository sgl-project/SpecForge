"""Offline evaluation must preserve the objective and the training trajectory."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torchtitan")

from specforge.runtime.contracts import TrainBatch
from specforge.training.torchtitan.model import SpecForgeTitanModel
from specforge.training.torchtitan.validation import (
    OfflineEvalSourceFactory,
    SpecForgeValidator,
    _ObjectiveTotals,
)


class SingletonMesh:
    def size(self):
        return 1

    def get_local_rank(self):
        return 0

    def get_group(self):
        return None


class SingleRank:
    pp_enabled = False

    def get_optional_mesh(self, *args, **kwargs):
        return SingletonMesh()


class FreshSource:
    def __init__(self, batches):
        self.batches = batches
        self.closed = False
        self.iterator_closed = False

    def __iter__(self):
        try:
            for batch in self.batches:
                torch.rand(1)  # Loader side effects must not change training RNG.
                yield batch
        finally:
            self.iterator_closed = True

    def close(self):
        torch.rand(1)  # Cleanup is part of the independent evaluation RNG scope.
        self.closed = True


def make_validator(batches, algorithm="dflash", **overrides):
    sources, logs = [], []

    def factory(**kwargs):
        source = FreshSource(batches)
        sources.append(source)
        return source

    config = SpecForgeValidator.Config(
        enable=True,
        freq=2,
        num_batches=len(batches),
        source_factory=factory,
        algorithm=algorithm,
        seed=123,
        **overrides,
    )
    processor = SimpleNamespace(
        logger=SimpleNamespace(log=lambda metrics, step: logs.append((metrics, step))),
        time_last_log=0.0,
        ntokens_since_last_log=317,
        data_loading_times=[0.13, 0.17],
    )
    validator = config.build(
        parallel_dims=SingleRank(),
        dp_world_size=1,
        dp_rank=0,
        validation_context=nullcontext,
        metrics_processor=processor,
        seq_len=8,
        local_batch_size=1,
    )
    return validator, sources, logs


def tiny_model(algorithm):
    return (
        SpecForgeTitanModel.Config(
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
        .build()
        .bfloat16()
    )


def feature_batches():
    return [
        {
            "input_ids": torch.randint(0, 31, (1, 8)),
            "hidden_states": torch.randn(1, 8, 32).bfloat16(),
            "target_last_hidden_states": torch.randn(1, 8, 16).bfloat16(),
            "loss_mask": torch.tensor([mask]),
        }
        for mask in ([0, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1])
    ]


@pytest.mark.parametrize("algorithm", ["dflash", "dflash2", "dspark"])
def test_validation_matches_weighted_objective_and_preserves_training(algorithm):
    torch.manual_seed(19)
    model = tiny_model(algorithm)
    batches = feature_batches()
    validator, sources, logs = make_validator(batches, algorithm)
    model.train()
    model.training_model.lm_head.eval()
    modes = [module.training for module in model.modules()]
    parameter = next(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    parameter.grad = torch.ones_like(parameter)
    expected_grad = parameter.grad.clone()
    generator = torch.Generator().manual_seed(123)
    numerator, denominator = 0.0, 0.0
    with torch.no_grad():
        for batch in batches:
            inputs = validator._prepare(
                batch, model.training_model, generator, SingletonMesh(), 2
            )
            _, _, metrics = model(**inputs)
            numerator += metrics["loss_terms"][0].item()
            denominator += metrics["loss_terms"][1].item()
    rng = torch.random.get_rng_state().clone()
    validator.validate([model], 2)
    assert validator.last_metrics["eval/avg_loss"] == pytest.approx(
        numerator / denominator
    )
    assert [module.training for module in model.modules()] == modes
    torch.testing.assert_close(torch.random.get_rng_state(), rng)
    torch.testing.assert_close(parameter.grad, expected_grad)
    assert validator.metrics_processor.ntokens_since_last_log == 317
    assert validator.metrics_processor.data_loading_times == [0.13, 0.17]
    assert validator.metrics_processor.time_last_log > 0
    first = {
        key: value
        for key, value in validator.last_metrics.items()
        if key != "eval/time_s"
    }
    validator.validate([model], 2)
    assert {
        key: value
        for key, value in validator.last_metrics.items()
        if key != "eval/time_s"
    } == first
    assert len(sources) == 2 and all(
        source.closed and source.iterator_closed for source in sources
    )
    assert [step for _, step in logs] == [2, 2]
    assert not validator.should_validate(0)
    assert not validator.should_validate(1)
    assert validator.should_validate(2)


def test_metric_ratios_pool_their_denominators_and_omit_partitioned_walks():
    totals = _ObjectiveTotals()
    for numerator, denominator in ((2.0, 1.0), (90.0, 30.0)):
        totals.add(
            {
                "loss_terms": (torch.tensor(numerator), torch.tensor(denominator)),
                "ratio_metrics": {
                    "acc": (torch.tensor(denominator), torch.tensor(denominator)),
                    "dflash/hard_label/walk_accepted_length": (
                        torch.tensor(3.0),
                        torch.tensor(1.0),
                    ),
                },
                "sum_metrics": {"count": torch.tensor(1.0)},
            },
            context_degree=2,
        )
    metrics = totals.reduce(device="cpu", group=None, degree=1)
    assert metrics["eval/avg_loss"] == pytest.approx(92 / 31)
    assert metrics["eval/avg_loss"] != 2.5
    assert metrics["eval/avg_acc"] == 1.0
    assert metrics["eval/count"] == 2.0
    assert not any("walk_accepted_length" in key for key in metrics)


def test_early_eof_restores_modes_rng_and_closes_sources():
    model = tiny_model("dflash")
    validator, sources, _ = make_validator(feature_batches())
    validator.config.num_batches = 3
    model.eval()
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="ended before"):
        validator.validate([model], 2)
    assert not model.training
    torch.testing.assert_close(torch.random.get_rng_state(), rng)
    assert sources[0].closed and sources[0].iterator_closed


def test_pipeline_evaluation_rejected():
    with pytest.raises(ValueError, match="pipeline parallelism"):
        SpecForgeValidator.Config().build(
            parallel_dims=SimpleNamespace(pp_enabled=True),
            dp_world_size=1,
            dp_rank=0,
            validation_context=nullcontext,
            metrics_processor=None,
            seq_len=8,
            local_batch_size=1,
        )


def test_factory_visits_uneven_dataset_once_and_zero_weights_padding(monkeypatch):
    import specforge.runtime.data_plane as data_plane

    sources, stores = [], []

    class Store:
        def __init__(self, run_id):
            self.closed = False
            stores.append(self)

        def close(self):
            self.closed = True

    class Loader(FreshSource):
        def __init__(self, store, *, refs, batch_size, **kwargs):
            super().__init__(
                [
                    TrainBatch(
                        sample_ids=[
                            str(value) for value in refs[offset : offset + batch_size]
                        ],
                        strategy="dflash",
                        tensors={"loss_mask": torch.ones(batch_size, 8)},
                    )
                    for offset in range(0, len(refs), batch_size)
                ]
            )
            sources.append(self)

    monkeypatch.setattr(data_plane, "FeatureDataLoader", Loader)
    monkeypatch.setattr(data_plane, "LocalFeatureStore", Store)
    provider = SimpleNamespace(
        build_collator=lambda: None,
        build_normalizer=lambda *args, **kwargs: None,
    )
    algorithm = SimpleNamespace(
        name="dflash",
        providers=SimpleNamespace(offline_for=lambda _: provider),
    )
    cfg = SimpleNamespace(
        model=SimpleNamespace(input_modality="text"),
        data=SimpleNamespace(dataloader_num_workers=0),
        training=SimpleNamespace(ttt_length=1),
        run_id="test",
    )
    factory = OfflineEvalSourceFactory(cfg, algorithm, list(range(5)))
    seen, lengths = [], []
    for rank in range(2):
        batches = list(
            factory(dp_rank=rank, dp_world_size=2, local_batch_size=2, seq_len=8)
        )
        lengths.append(len(batches))
        for batch in batches:
            for sample_id, mask in zip(batch.sample_ids, batch.tensors["loss_mask"]):
                if mask.any():
                    seen.append(int(sample_id))
    assert lengths == [2, 2]
    assert sorted(seen) == list(range(5))
    assert all(source.closed and source.iterator_closed for source in sources)
    assert all(store.closed for store in stores)


def test_cleanup_attempts_all_sources_without_hiding_primary_failure():
    from specforge.training.torchtitan.validation import _close_all

    closed = []

    def failed_close():
        closed.append("first")
        raise RuntimeError("cleanup error")

    first = SimpleNamespace(close=failed_close)
    second = SimpleNamespace(close=lambda: closed.append("second"))
    with pytest.raises(ValueError, match="forward error"):
        try:
            raise ValueError("forward error")
        finally:
            _close_all(first, second)
    assert closed == ["first", "second"]
    with pytest.raises(RuntimeError, match="cleanup error"):
        _close_all(first, second)
