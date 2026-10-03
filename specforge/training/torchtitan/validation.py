"""Periodic offline feature evaluation through TorchTitan's validator hook."""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.distributed as dist
from torchtitan.components.validate import BaseValidator
from torchtitan.tools.logging import logger

from .numerics import compiler_numerics
from .runtime import partition_anchor_blocks


def _close_all(*objects):
    """Attempt every cleanup without hiding an active evaluation failure."""
    primary = sys.exc_info()[1]
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
        if primary is None:
            raise failure
        logger.error("Evaluation cleanup also failed: %s", failure)


@dataclass
class OfflineEvalSourceFactory:
    """Visit every reference once, with zero-weight padding for equal DP work."""

    cfg: Any
    algorithm: Any
    refs: list

    def __call__(self, *, dp_rank, dp_world_size, local_batch_size, seq_len, epoch=0):
        del epoch
        from specforge.runtime.data_plane import FeatureDataLoader, LocalFeatureStore

        if not self.refs:
            raise ValueError("The offline evaluation dataset is empty")
        provider = self.algorithm.providers.offline_for(self.cfg.model.input_modality)
        count = math.ceil(len(self.refs) / (dp_world_size * local_batch_size))
        indices = [
            index * dp_world_size + dp_rank for index in range(count * local_batch_size)
        ]
        padded_refs = [
            self.refs[index] if index < len(self.refs) else self.refs[0]
            for index in indices
        ]
        workers = self.cfg.data.dataloader_num_workers
        if workers is None:
            workers = self.algorithm.providers.model.default_dataloader_num_workers
        store = LocalFeatureStore(f"{self.cfg.run_id}-eval")
        source = iterator = None
        try:
            source = FeatureDataLoader(
                store,
                refs=padded_refs,
                device="cpu",
                batch_size=local_batch_size,
                collate_fn=provider.build_collator(),
                per_sample_transform=provider.build_normalizer(
                    seq_len,
                    ttt_length=self.cfg.training.ttt_length,
                    use_usp_preprocess=False,
                ),
                drop_last=True,
                strategy=self.algorithm.name,
                num_workers=workers,
                pin_memory=workers > 0,
            )
            iterator = iter(source)
            for batch_index, batch in enumerate(iterator):
                valid = torch.tensor(
                    [
                        index < len(self.refs)
                        for index in indices[
                            batch_index
                            * local_batch_size : (batch_index + 1)
                            * local_batch_size
                        ]
                    ],
                    dtype=torch.bool,
                )
                tensors = dict(batch.tensors)
                tensors["loss_mask"] = tensors["loss_mask"].clone()
                tensors["loss_mask"][~valid] = 0
                yield replace(batch, tensors=tensors)
        finally:
            _close_all(iterator, source if source is not iterator else None, store)


class _ObjectiveTotals:
    def __init__(self):
        self.ratios = {}
        self.sums = {}

    def add(self, metrics, *, context_degree):
        ratios = {"avg_loss": metrics["loss_terms"], **metrics.get("ratio_metrics", {})}
        for name, (numerator, denominator) in ratios.items():
            # A walk crosses consecutive anchors within a sequence. Separate
            # CP walks cannot be combined by adding their local ratios.
            if context_degree > 1 and name.endswith("/walk_accepted_length"):
                continue
            pair = torch.stack(
                (numerator.detach().double(), denominator.detach().double())
            )
            self.ratios[name] = self.ratios.get(name, 0) + pair
        for name, value in metrics.get("sum_metrics", {}).items():
            self.sums[name] = self.sums.get(name, 0) + value.detach().double()

    def reduce(self, *, device, group, degree):
        values = {f"ratio:{name}": tensor for name, tensor in self.ratios.items()}
        values.update({f"sum:{name}": tensor for name, tensor in self.sums.items()})
        shapes = {name: tuple(value.shape) for name, value in values.items()}
        if degree > 1:
            gathered = [None] * degree
            dist.all_gather_object(gathered, shapes, group=group)
            for rank_shapes in gathered:
                for name, shape in rank_shapes.items():
                    if name in shapes and shapes[name] != shape:
                        raise ValueError(
                            f"Evaluation metric shape differs across ranks: {name}"
                        )
                    shapes[name] = shape
        names = sorted(shapes)
        packed = torch.cat(
            [
                values.get(
                    name, torch.zeros(shapes[name], device=device, dtype=torch.float64)
                )
                .flatten()
                .to(device)
                for name in names
            ]
        )
        if degree > 1:
            dist.all_reduce(packed, group=group)
        if "ratio:avg_loss" not in shapes:
            raise ValueError("Evaluation produced no objective terms")
        output, cursor = {}, 0
        for key in names:
            width = math.prod(shapes[key])
            value = packed[cursor : cursor + width].reshape(shapes[key])
            cursor += width
            kind, name = key.split(":", 1)
            if kind == "ratio":
                if name == "avg_loss" and not bool(value[1] > 0):
                    raise ValueError("Evaluation produced no supervised targets")
                value = value[0] / value[1].clamp_min(1e-30)
            flat = value.flatten().cpu().tolist()
            if len(flat) == 1:
                output[f"eval/{name}"] = flat[0]
            else:
                output.update(
                    {
                        f"eval/{name}_{index}": scalar
                        for index, scalar in enumerate(flat)
                    }
                )
        output["eval/avg_acc"] = output.get("eval/acc", 0.0)
        if not all(math.isfinite(value) for value in output.values()):
            raise FloatingPointError("Offline evaluation produced a non-finite metric")
        return output


class SpecForgeValidator(BaseValidator):
    """Evaluate weighted objectives without advancing the training data or RNG."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseValidator.Config):
        enable: bool = False
        source_factory: Callable[..., Any] | None = None
        num_batches: int = 0
        algorithm: str = "dflash"
        seed: int = 42
        schedule_total_steps: int = 1

    def __init__(
        self,
        config,
        *,
        parallel_dims,
        dp_world_size,
        dp_rank,
        validation_context,
        metrics_processor,
        seq_len,
        local_batch_size,
        **kwargs,
    ):
        del kwargs
        super().__init__(config)
        if parallel_dims.pp_enabled:
            raise ValueError(
                "TorchTitan offline evaluation does not yet support pipeline parallelism"
            )
        if config.source_factory is None or config.num_batches < 1 or config.freq < 1:
            raise ValueError(
                "Offline evaluation requires a source, positive batch count and interval"
            )
        if config.algorithm not in {"dflash", "dflash2", "dspark"}:
            raise ValueError(f"Unsupported evaluation algorithm: {config.algorithm}")
        self.parallel_dims = parallel_dims
        self.dp_world_size, self.dp_rank = dp_world_size, dp_rank
        self.validation_context = validation_context
        self.metrics_processor = metrics_processor
        self.seq_len, self.local_batch_size = seq_len, local_batch_size
        self.last_metrics = {}

    def should_validate(self, step: int) -> bool:
        return self.config.enable and step > 0 and step % self.config.freq == 0

    def _prepare(self, batch, objective, generator, context_mesh, step):
        inputs = dict(batch.tensors if hasattr(batch, "tensors") else batch)
        mask = inputs["loss_mask"]
        if mask.device.type != "cpu" or mask.ndim != 2:
            raise ValueError(
                "Evaluation loss masks must be two-dimensional CPU tensors"
            )
        if mask.shape[0] != self.local_batch_size:
            raise ValueError(
                "Evaluation batches must include zero-weight padding to local_batch_size"
            )
        if bool(((mask[:, :-1] > 0.5) & (mask[:, 1:] > 0.5)).any()):
            anchors, keep = objective._sample_anchor_positions(
                mask.shape[1], mask, torch.device("cpu"), generator=generator
            )
        else:
            # A padded DP rank must still join the same FSDP/TP/CP forwards.
            anchors = torch.zeros((mask.shape[0], 1), dtype=torch.long)
            keep = torch.zeros_like(anchors, dtype=torch.bool)
        if self.config.algorithm != "dspark":
            completed = step
            warmup = int(
                self.config.schedule_total_steps * objective.selector_warmup_ratio
            )
            ramp = int(self.config.schedule_total_steps * objective.selector_ramp_ratio)
            alpha = (
                0.0
                if completed < warmup
                else objective.selector_loss_alpha
                * (1.0 if ramp <= 0 else min((completed - warmup + 1) / ramp, 1.0))
            )
            inputs["selector_loss_alpha"] = alpha
            if objective.loss_type != "dflash":
                _, weights = objective._dflash_weight_mask(mask, anchors, keep)
                scale = objective._sequence_anchor_scale(weights).squeeze(-1)
                scale, _ = partition_anchor_blocks(
                    scale,
                    keep,
                    rank=context_mesh.get_local_rank(),
                    degree=context_mesh.size(),
                )
                inputs["prepared_sequence_anchor_scale"] = scale.unsqueeze(-1)
        anchors, keep = partition_anchor_blocks(
            anchors,
            keep,
            rank=context_mesh.get_local_rank(),
            degree=context_mesh.size(),
        )
        inputs.update(
            anchor_positions=anchors,
            block_keep_mask=keep,
            collect_detailed_metrics=True,
        )
        return inputs

    @torch.no_grad()
    def validate(self, model_parts, step):
        if len(model_parts) != 1:
            raise ValueError("Offline evaluation requires one non-pipeline model part")
        model = model_parts[0]
        device = next(model.parameters()).device
        context_mesh = self.parallel_dims.get_optional_mesh(
            "cp", include_singleton_axes=True
        )
        loss_mesh = self.parallel_dims.get_optional_mesh(
            "loss", include_singleton_axes=True
        )
        generator = torch.Generator(device="cpu").manual_seed(
            self.config.seed + self.dp_rank
        )
        modes = [(module, module.training) for module in model.modules()]
        source = iterator = None
        started = time.perf_counter()
        totals = _ObjectiveTotals()
        devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            try:
                model.eval()
                source = self.config.source_factory(
                    dp_rank=self.dp_rank,
                    dp_world_size=self.dp_world_size,
                    local_batch_size=self.local_batch_size,
                    seq_len=self.seq_len,
                    epoch=0,
                )
                iterator = iter(source)
                for _ in range(self.config.num_batches):
                    try:
                        batch = next(iterator)
                    except StopIteration as exc:
                        raise ValueError(
                            "Evaluation source ended before its declared batch count"
                        ) from exc
                    inputs = self._prepare(
                        batch, model.training_model, generator, context_mesh, step
                    )
                    inputs = {
                        name: (
                            value.to(device)
                            if isinstance(value, torch.Tensor)
                            else value
                        )
                        for name, value in inputs.items()
                    }
                    with self.validation_context(), compiler_numerics():
                        _loss, _accuracy, metrics = model(**inputs)
                    totals.add(metrics, context_degree=context_mesh.size())
                self.last_metrics = totals.reduce(
                    device=device, group=loss_mesh.get_group(), degree=loss_mesh.size()
                )
                self.last_metrics["eval/time_s"] = time.perf_counter() - started
                self.metrics_processor.logger.log(self.last_metrics, step)
                logger.info(
                    "validation step: %d loss: %.6f accuracy: %.6f",
                    step,
                    self.last_metrics["eval/avg_loss"],
                    self.last_metrics["eval/avg_acc"],
                )
            finally:
                for module, was_training in modes:
                    module.training = was_training
                _close_all(iterator, source if source is not iterator else None)
                # Validation must not reset or consume the native training metrics
                # window. Exclude its wall time from the next training throughput.
                if hasattr(self.metrics_processor, "time_last_log"):
                    self.metrics_processor.time_last_log += (
                        time.perf_counter() - started
                    )
