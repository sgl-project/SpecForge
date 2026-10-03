"""Stateful feature batches for the native TorchTitan training loop."""

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.loss import IGNORE_INDEX


class FeatureDataLoader(BaseDataLoader):
    """Adapt a deterministic, re-iterable SpecForge feature source.

    ``source_factory`` receives the *data parallel* rank and degree, rather than
    WORLD, so TP peers see the same batches. Sources must be replayable; online
    queues with consumption/acknowledgement semantics need a separate adapter.
    Cursor state is captured at optimizer boundaries by Titan's DCP manager.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        source_factory: Callable[..., Iterable] | None = None
        epochs: int = 1
        pad_to_seq_len: bool = False

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer=None,
        seq_len: int = 0,
        local_batch_size: int = 1,
        snapshot_every_n_steps=None,
    ):
        del tokenizer, snapshot_every_n_steps
        if config.source_factory is None or config.epochs <= 0:
            raise ValueError(
                "FeatureDataLoader requires a source_factory and epochs > 0"
            )
        self.source_factory = config.source_factory
        self.epochs = config.epochs
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.local_batch_size = local_batch_size
        self.seq_len = seq_len
        self.pad_to_seq_len = config.pad_to_seq_len
        self.epoch = 0
        self.cursor = 0

    def __iter__(self) -> Iterator[tuple[dict[str, Any], torch.Tensor]]:
        while self.epoch < self.epochs:
            source = self.source_factory(
                dp_rank=self.dp_rank,
                dp_world_size=self.dp_world_size,
                local_batch_size=self.local_batch_size,
                seq_len=self.seq_len,
                epoch=self.epoch,
            )
            if getattr(source, "queue", None) is not None:
                raise ValueError(
                    "TorchTitan feature source must be replayable, not a queue"
                )
            set_epoch = getattr(source, "set_epoch", None)
            if set_epoch is not None:
                set_epoch(self.epoch)
            seek = getattr(source, "seek", None)
            if seek is not None:
                seek(self.cursor)
            source_iterator = iter(source)
            try:
                for _ in range(self.cursor if seek is None else 0):
                    try:
                        next(source_iterator)
                    except StopIteration as exc:
                        raise ValueError(
                            "Saved feature cursor exceeds the current source"
                        ) from exc
                for batch in source_iterator:
                    tensors = batch.tensors if hasattr(batch, "tensors") else batch
                    tensors = dict(tensors)
                    required = {"input_ids", "hidden_states", "loss_mask"}
                    if not required.issubset(tensors):
                        raise ValueError(
                            f"Feature batch missing {required - tensors.keys()}"
                        )
                    if self.pad_to_seq_len:
                        for name in (
                            "input_ids",
                            "loss_mask",
                            "hidden_states",
                            "target_last_hidden_states",
                        ):
                            if name not in tensors:
                                continue
                            value = tensors[name]
                            if value.shape[1] > self.seq_len:
                                raise ValueError(
                                    "Feature sequence exceeds the configured pipeline length"
                                )
                            shape = list(value.shape)
                            shape[1] = self.seq_len
                            padded = value.new_zeros(shape)
                            padded[:, : value.shape[1]] = value
                            tensors[name] = padded
                    input_ids = tensors.pop("input_ids")
                    if (
                        input_ids.device.type != "cpu"
                        or tensors["loss_mask"].device.type != "cpu"
                    ):
                        raise ValueError(
                            "Feature input_ids and loss_mask must remain on CPU"
                        )
                    if (
                        input_ids.ndim != 2
                        or input_ids.shape[0] != self.local_batch_size
                    ):
                        raise ValueError(
                            "Feature batches must match training.local_batch_size"
                        )
                    labels = input_ids.clone()
                    labels.masked_fill_(tensors["loss_mask"] <= 0, IGNORE_INDEX)
                    self.cursor += 1
                    yield {"input": input_ids, **tensors}, labels
            finally:
                close = getattr(source_iterator, "close", None)
                if close is not None:
                    close()
                close = getattr(source, "close", None)
                if close is not None and source is not source_iterator:
                    close()
            self.epoch += 1
            self.cursor = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            f"dp_rank_{self.dp_rank}": {
                "epoch": self.epoch,
                "cursor": self.cursor,
                "dp_world_size": self.dp_world_size,
            }
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        state = state_dict[f"dp_rank_{self.dp_rank}"]
        if state["dp_world_size"] != self.dp_world_size:
            raise ValueError(
                "Feature cursor resume requires unchanged data parallel degree"
            )
        epoch, cursor = int(state["epoch"]), int(state["cursor"])
        if (
            epoch < 0
            or epoch > self.epochs
            or cursor < 0
            or (epoch == self.epochs and cursor)
        ):
            raise ValueError("Invalid saved feature epoch or cursor")
        self.epoch, self.cursor = epoch, cursor
