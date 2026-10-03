"""Online IDs become durable only after a completed native optimizer step."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torchtitan")

from torchtitan.components.dataloader import DataloaderExhaustedError

from specforge.runtime.contracts import TrainBatch
from specforge.training.torchtitan.online import (
    OnlineFeatureDataLoader,
    SpecForgeOnlineTrainer,
)
from specforge.training.torchtitan.runtime import SpecForgeTitanTrainer


def online_loader():
    events = []

    class Source:
        ack = False

        def __iter__(self):
            try:
                for index in range(4):
                    yield TrainBatch(
                        sample_ids=[f"sample-{index}"],
                        strategy="dflash",
                        tensors={
                            "input_ids": torch.ones(1, 8, dtype=torch.long),
                            "hidden_states": torch.ones(1, 8, 32),
                            "loss_mask": torch.ones(1, 8),
                        },
                    )
            finally:
                events.append("iterator-close")

        def close(self):
            events.append("source-close")

    def acknowledge(trainer_id, sample_ids, **kwargs):
        assert kwargs["optimizer_durable"]
        events.append(("durable", list(sample_ids), kwargs["global_step"]))

    resources = SimpleNamespace(
        controller=SimpleNamespace(ack_train_refs=acknowledge),
        queue=SimpleNamespace(ack_ids=lambda ids: events.append(("inbox", list(ids)))),
        start=lambda: events.append("start"),
        on_success=lambda step: events.append(("success", step)),
        on_failure=lambda exc: events.append(("failure", type(exc))),
        close=lambda: events.append("resources-close"),
    )

    def factory(**kwargs):
        events.append(("open", kwargs["resume_step"]))
        return Source(), resources

    loader = OnlineFeatureDataLoader.Config(source_factory=factory).build(
        dp_rank=0, dp_world_size=1, seq_len=8, local_batch_size=1
    )
    return loader, events


def test_restore_precedes_queue_open_and_ack_follows_complete_optimizer_window(
    monkeypatch,
):
    loader, events = online_loader()
    state = {"dp_rank_0": {"committed_step": 7, "dp_world_size": 1}}
    loader.load_state_dict(state)
    assert events == []
    trainer = SpecForgeOnlineTrainer.__new__(SpecForgeOnlineTrainer)
    trainer.dataloader = loader
    trainer.step = 8
    iterator = iter(loader)

    def native_step(self, data_iterator):
        next(data_iterator)
        next(data_iterator)
        assert not any(
            isinstance(event, tuple) and event[0] == "durable" for event in events
        )
        with pytest.raises(RuntimeError, match="before its optimizer ACK"):
            loader.state_dict()
        events.append("optimizer")
        return "native-result"

    monkeypatch.setattr(SpecForgeTitanTrainer, "train_step", native_step)
    assert trainer.train_step(iterator) == "native-result"
    assert events == [
        ("open", 7),
        "start",
        "optimizer",
        ("durable", ["sample-0", "sample-1"], 8),
        ("inbox", ["sample-0", "sample-1"]),
    ]
    assert loader.state_dict()["dp_rank_0"]["committed_step"] == 8
    loader.close()
    assert events[-2:] == ["iterator-close", "source-close"]


@pytest.mark.parametrize(
    "error", [RuntimeError("backward failed"), DataloaderExhaustedError()]
)
def test_failed_or_incomplete_window_never_acks(monkeypatch, error):
    loader, events = online_loader()
    trainer = SpecForgeOnlineTrainer.__new__(SpecForgeOnlineTrainer)
    trainer.dataloader, trainer.step = loader, 1

    def native_step(self, iterator):
        next(iterator)
        raise error

    monkeypatch.setattr(SpecForgeTitanTrainer, "train_step", native_step)
    with pytest.raises(type(error)):
        trainer.train_step(iter(loader))
    assert not any(
        isinstance(event, tuple) and event[0] == "durable" for event in events
    )
    assert loader.pending_ids == ["sample-0"]
    assert loader.committed_step == 0
    assert trainer.step == (0 if isinstance(error, DataloaderExhaustedError) else 1)
    loader.close()


def test_failed_durable_commit_does_not_advance_inbox_or_checkpoint():
    loader, events = online_loader()
    next(iter(loader))

    def failure(*args, **kwargs):
        raise RuntimeError("ledger unavailable")

    loader.resources.controller.ack_train_refs = failure
    with pytest.raises(RuntimeError, match="ledger unavailable"):
        loader.commit(1)
    assert loader.committed_step == 0
    assert loader.pending_ids == ["sample-0"]
    assert not any(isinstance(event, tuple) and event[0] == "inbox" for event in events)
    loader.close()


def test_native_train_lifecycle_marks_failure_before_final_cleanup(monkeypatch):
    loader, events = online_loader()
    trainer = SpecForgeOnlineTrainer.__new__(SpecForgeOnlineTrainer)
    trainer.dataloader = loader
    trainer.config = SimpleNamespace(training=SimpleNamespace(steps=2))
    trainer.step = 0

    def native_train(self):
        next(iter(loader))
        raise ValueError("failure")

    monkeypatch.setattr(SpecForgeTitanTrainer, "train", native_train)
    with pytest.raises(ValueError, match="failure"):
        trainer.train()
    assert events.index(("failure", ValueError)) < events.index("source-close")
    assert events[-1] == "resources-close"
    assert not any(
        isinstance(event, tuple) and event[0] == "success" for event in events
    )


def test_native_resume_step_validates_retained_ledger_without_legacy_checkpoint(
    tmp_path, monkeypatch
):
    from specforge.launch import build_online_consumer_resources
    from specforge.runtime.control_plane.controller import DataFlowController
    from specforge.runtime.control_plane.metadata_store import SQLiteMetadataStore
    from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel
    from tests.test_runtime.test_disagg_online_shared_plane import (
        _FakeDistributor,
        _ref,
        _RetainingFeatureStore,
    )

    database = str(tmp_path / "ledger.sqlite")
    ledger = SQLiteMetadataStore(database)
    controller = DataFlowController("run0", metadata_store=ledger)
    controller.commit_samples("producer", [_ref("s0"), _ref("s1")])
    ledger.record_train_ack(["s0"], global_step=7, optimizer_durable=True)
    ledger.close()
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)
    monkeypatch.setattr(
        "specforge.runtime.data_plane.ref_distributor.RefDistributor", _FakeDistributor
    )
    monkeypatch.setattr(
        "specforge.launch._checkpoint_global_step",
        lambda _: pytest.fail("Native resume must not parse a legacy checkpoint"),
    )
    channel = StreamingRefChannel(str(tmp_path / "refs.jsonl"))
    resources = build_online_consumer_resources(
        run_id="run0",
        feature_store=_RetainingFeatureStore(),
        channel=channel,
        metadata_db_path=database,
        resume_from="native-dcp/step-7",
        resume_step=7,
        async_ack=False,
    )
    assert resources.distributor.kwargs["skip_ids"] == {"s0"}
    assert resources.distributor.kwargs["requeued_ids"] == {"s1"}
    resources.close()
    with pytest.raises(RuntimeError, match="behind"):
        build_online_consumer_resources(
            run_id="run0",
            feature_store=_RetainingFeatureStore(),
            channel=channel,
            metadata_db_path=database,
            resume_from="native-dcp/step-8",
            resume_step=8,
            async_ack=False,
        )


@pytest.mark.parametrize(
    "max_steps,total_steps,expected",
    [
        (None, 100, (3, 100)),
        (1, 100, (1, 100)),
        (10, None, (3, 10)),
        (None, None, (3, 3)),
    ],
)
def test_online_stop_uses_finite_producer_length_independently_of_lr_horizon(
    max_steps, total_steps, expected
):
    from specforge.training.torchtitan.frontend import _online_training_horizon

    cfg = SimpleNamespace(
        training=SimpleNamespace(max_steps=max_steps, total_steps=total_steps)
    )
    assert _online_training_horizon(cfg, 3) == expected


def test_completed_checkpoint_resume_still_reconciles_ledger_and_signals_success(
    monkeypatch,
):
    loader, events = online_loader()
    trainer = SpecForgeOnlineTrainer.__new__(SpecForgeOnlineTrainer)
    trainer.dataloader = loader
    trainer.config = SimpleNamespace(training=SimpleNamespace(steps=7))

    def native_train(self):
        self.step = 7
        loader.load_state_dict({"dp_rank_0": {"committed_step": 7, "dp_world_size": 1}})
        # Native loop correctly executes no optimizer steps after loading step 7.

    monkeypatch.setattr(SpecForgeTitanTrainer, "train", native_train)
    trainer.train()
    assert events == [
        ("open", 7),
        "start",
        "source-close",
        ("success", 7),
        "resources-close",
    ]
    assert not loader.started
