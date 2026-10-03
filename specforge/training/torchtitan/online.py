"""DP-only online feature consumption at native optimizer boundaries."""

import logging
from dataclasses import dataclass
from typing import Any, Callable

from torchtitan.components.dataloader import BaseDataLoader, DataloaderExhaustedError
from torchtitan.components.loss import IGNORE_INDEX

from .runtime import SpecForgeTitanTrainer


class OnlineFeatureDataLoader(BaseDataLoader):
    """Resume the retained ledger, never seek or replay a live feature queue."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        # Called only after native DCP restore, once per consumer attempt.
        source_factory: Callable[..., Any] | None = None

    def __init__(
        self,
        config,
        *,
        dp_world_size,
        dp_rank,
        local_batch_size=1,
        seq_len=0,
        **kwargs,
    ):
        del kwargs
        if config.source_factory is None:
            raise ValueError("Online features require a source factory")
        self.config = config
        self.dp_world_size, self.dp_rank = dp_world_size, dp_rank
        self.local_batch_size, self.seq_len = local_batch_size, seq_len
        self.committed_step = 0
        self.pending_ids = []
        self.resources = self.source = self.iterator = None
        self.started = False

    def ensure_open(self):
        """Reconcile even when a restored checkpoint already meets the stop step."""
        if self.resources is not None:
            return
        self.source, self.resources = self.config.source_factory(
            dp_rank=self.dp_rank,
            dp_world_size=self.dp_world_size,
            local_batch_size=self.local_batch_size,
            seq_len=self.seq_len,
            resume_step=self.committed_step,
        )
        if getattr(self.source, "ack", None) is not False:
            raise ValueError(
                "Online feature loading must defer ACK until optimizer commit"
            )
        self.resources.start()

    def __iter__(self):
        if self.started:
            raise RuntimeError("An online feature stream can only be consumed once")
        self.started = True
        self.ensure_open()
        self.iterator = iter(self.source)
        for batch in self.iterator:
            inputs = dict(batch.tensors)
            required = {"input_ids", "hidden_states", "loss_mask"}
            if not required.issubset(inputs):
                raise ValueError(
                    f"Online feature batch missing {required - inputs.keys()}"
                )
            input_ids = inputs.pop("input_ids")
            if (
                input_ids.device.type != "cpu"
                or inputs["loss_mask"].device.type != "cpu"
            ):
                raise ValueError("Online input_ids and loss_mask must remain on CPU")
            if input_ids.ndim != 2 or input_ids.shape[0] != self.local_batch_size:
                raise ValueError("Online feature batch must match local_batch_size")
            if len(batch.sample_ids) != self.local_batch_size:
                raise ValueError("Online features require one sample ID per batch row")
            labels = input_ids.clone()
            labels.masked_fill_(inputs["loss_mask"] <= 0, IGNORE_INDEX)
            self.pending_ids.extend(batch.sample_ids)
            yield {"input": input_ids, **inputs}, labels

    def commit(self, step):
        if self.resources is None or not self.pending_ids:
            raise RuntimeError(
                "An online optimizer step did not consume any sample IDs"
            )
        if step != self.committed_step + 1:
            raise ValueError(
                "Online optimizer steps must follow the retained ACK boundary"
            )
        self.resources.controller.ack_train_refs(
            f"torchtitan-dp-{self.dp_rank}",
            self.pending_ids,
            global_step=step,
            optimizer_durable=True,
        )
        self.resources.queue.ack_ids(self.pending_ids)
        self.pending_ids = []
        self.committed_step = step

    def close(self):
        objects = (
            self.iterator,
            self.source if self.source is not self.iterator else None,
        )
        self.iterator = self.source = None
        failure = None
        for obj in objects:
            close = getattr(obj, "close", None)
            if close is not None:
                try:
                    close()
                except BaseException as exc:
                    if failure is None:
                        failure = exc
        if failure is not None:
            raise failure

    def state_dict(self):
        if self.pending_ids:
            raise RuntimeError("Cannot checkpoint online data before its optimizer ACK")
        return {
            f"dp_rank_{self.dp_rank}": {
                "committed_step": self.committed_step,
                "dp_world_size": self.dp_world_size,
            }
        }

    def load_state_dict(self, state_dict):
        if self.started:
            raise RuntimeError("Cannot restore an already-open online stream")
        state = state_dict[f"dp_rank_{self.dp_rank}"]
        step = state["committed_step"]
        if (
            state["dp_world_size"] != self.dp_world_size
            or not isinstance(step, int)
            or step < 0
        ):
            raise ValueError(
                "Online resume requires a valid step and unchanged DP degree"
            )
        self.committed_step = step


class SpecForgeOnlineTrainer(SpecForgeTitanTrainer):
    """Add synchronous durable ACKs while retaining the native training loop."""

    @dataclass(kw_only=True, slots=True)
    class Config(SpecForgeTitanTrainer.Config):
        pass

    def __init__(self, config):
        parallelism = config.parallelism
        if any(
            getattr(parallelism, name) != 1
            for name in (
                "tensor_parallel_degree",
                "context_parallel_degree",
                "pipeline_parallel_degree",
            )
        ):
            raise ValueError("TorchTitan online training currently supports DP only")
        if (
            not config.training.disable_cuda_graphs
            or getattr(config.compile, "mode", None) == "aot_fx_trace"
        ):
            raise ValueError(
                "TorchTitan online features do not yet support graph capture"
            )
        if config.checkpoint.async_mode != "disabled":
            raise ValueError(
                "Online ACK boundaries require synchronous native checkpoints"
            )
        super().__init__(config)
        if not isinstance(self.dataloader, OnlineFeatureDataLoader):
            raise TypeError("Online training requires OnlineFeatureDataLoader")

    def train_step(self, data_iterator):
        if self.dataloader.committed_step != self.step - 1:
            raise ValueError("Restored online data and model steps do not match")
        try:
            result = super().train_step(data_iterator)
        except DataloaderExhaustedError:
            self.step -= 1  # Native loop increments before fetching its window.
            raise
        self.dataloader.commit(self.step)
        return result

    def train(self):
        failure = None
        try:
            result = super().train()
            if self.step != self.config.training.steps:
                raise RuntimeError(
                    "Online source ended before the configured optimizer boundary"
                )
            if self.dataloader.committed_step != self.step:
                raise ValueError("Restored online data and model steps do not match")
            self.dataloader.ensure_open()
            self.dataloader.close()
            self.dataloader.resources.on_success(self.step)
            return result
        except BaseException as exc:
            failure = exc
            if self.dataloader.resources is not None:
                self.dataloader.resources.on_failure(exc)
            raise
        finally:
            try:
                try:
                    self.dataloader.close()
                finally:
                    if self.dataloader.resources is not None:
                        self.dataloader.resources.close()
            except BaseException as cleanup_error:
                if failure is None:
                    if self.dataloader.resources is not None:
                        self.dataloader.resources.on_failure(cleanup_error)
                    raise
                logging.getLogger(__name__).exception(
                    "Online cleanup failed while preserving the primary training failure"
                )
