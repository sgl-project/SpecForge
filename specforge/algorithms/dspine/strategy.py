"""DSpine schedule and optimizer-window loss contract."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.distributed as dist

from specforge.runtime.contracts import TrainBatch
from specforge.training.strategies.base import (
    DFlashTrainStrategy,
    StepContext,
    StepOutput,
    _cpu_max_valid_anchors,
)


class DSpineTrainStrategy(DFlashTrainStrategy):
    name = "dspine"
    required_features = {
        "input_ids",
        "hidden_states",
        "loss_mask",
        "target_last_hidden_states",
    }

    def forward_loss(
        self, batch: TrainBatch, ctx: StepContext | None = None
    ) -> StepOutput:
        if ctx is None or ctx.accumulation_steps == 1 or not torch.is_grad_enabled():
            return self._forward_loss(batch, ctx)

        device = self._device()
        devices = [] if device.type == "cpu" else [device.index]
        device_module = torch.get_device_module(device.type)
        cpu_rng = torch.get_rng_state()
        device_rng = [device_module.get_rng_state(index) for index in devices]
        with torch.no_grad():
            output = self._forward_loss(batch, ctx)
        # Keep replay inputs on the host, not activations for the whole window.
        replay_batch = replace(
            batch,
            tensors={
                name: value.detach().cpu() for name, value in batch.tensors.items()
            },
        )

        def replay_loss(
            window_ratios: dict[str, tuple[torch.Tensor, torch.Tensor]],
        ) -> StepOutput:
            ce, denominator = window_ratios["dspine/backbone_ce"]
            totals = torch.stack((ce, denominator))
            if dist.is_initialized():
                dist.all_reduce(totals)
            torch._assert_async(
                totals[1] > 0, "DSpine has no supervised proposal tokens"
            )
            with torch.random.fork_rng(devices=devices, device_type=device.type):
                torch.set_rng_state(cpu_rng)
                for index, state in zip(devices, device_rng):
                    device_module.set_rng_state(state, index)
                return self._forward_loss(replay_batch, ctx, totals[0] / totals[1])

        return replace(output, replay_loss=replay_loss)

    def _forward_loss(
        self,
        batch: TrainBatch,
        ctx: StepContext | None,
        alignment_ce: torch.Tensor | None = None,
    ) -> StepOutput:
        self.validate_batch(batch)
        tensors = batch.tensors
        inputs = {
            name: tensors[name].to(self._device(), non_blocking=True)
            for name in self.required_features
        }
        loss, accuracy, metrics = self.dflash_model(
            **inputs,
            max_valid_anchors=_cpu_max_valid_anchors(tensors["loss_mask"]),
            global_step=ctx.global_step if ctx is not None else 0,
            total_steps=ctx.total_steps if ctx is not None else None,
            collect_detailed_metrics=(
                ctx.collect_detailed_metrics if ctx is not None else True
            ),
            alignment_ce=alignment_ce,
        )
        return StepOutput(
            loss=loss,
            metrics={
                "accuracy": accuracy.detach(),
                "accuracy_denom": metrics["accuracy_denom"],
            },
            ratio_metrics=metrics["ratio_metrics"],
            loss_terms=metrics["loss_terms"],
        )
