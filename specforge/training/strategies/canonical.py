"""Adapt additive objective results to the existing distributed trainer."""

from dataclasses import replace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .base import DraftTrainStrategy, StepContext, StepOutput


class CanonicalTrainStrategy(DraftTrainStrategy):
    def __init__(self, model, *, name, objective, required_features):
        self.model = model
        self.name = name
        self.objective = objective
        self.required_features = set(required_features)
        seen = set()
        self._duplicate_parameters = set()
        inner = model.module if isinstance(model, DistributedDataParallel) else model
        self.window_statistic = inner.registration.training.window_statistic
        self.host_inputs = inner.registration.training.host_inputs
        self.draft_config = inner.draft_model.config.draft
        self.statistics = (
            inner.statistics
        )  # stepped and pooled here, not in the objective
        for key, value in inner.draft_model.named_parameters(remove_duplicate=False):
            if id(value) in seen:
                self._duplicate_parameters.add(key)
            seen.add(id(value))
        self._statistic_window = None
        self._window_value = None

    def trainable_module(self):
        return self.model

    def after_step(self, *, process_group=None):
        if self.statistics is None:
            return {}
        return self.statistics.step(process_group=process_group)

    def checkpoint_state_filter(self, state_dict):
        prefix = "draft_model."
        return {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
            and key[len(prefix) :] not in self._duplicate_parameters
        }

    def _forward(self, batch, ctx, window_value=None):
        self.validate_batch(batch)
        device = next(self.model.parameters()).device
        inputs = batch.tensors
        if self.host_inputs is not None:
            prepared = self.host_inputs(inputs, self.draft_config)
            if device.type != "cpu":
                # a pageable source makes the non_blocking copy wait for the host
                prepared = {
                    key: value.pin_memory() if value.device.type == "cpu" else value
                    for key, value in prepared.items()
                }
            inputs = inputs | prepared
        tensors = {
            key: value.to(device, non_blocking=True) for key, value in inputs.items()
        }
        result = self.model(
            tensors,
            step=ctx.global_step,
            total_steps=ctx.total_steps,
            **(
                {self.window_statistic.argument_name: window_value}
                if self.window_statistic
                else {}
            ),
        )
        if result.deferred is not None:
            # The objective's term is unfinished: pooling over ranks is the strategy's, as above.
            result = result.deferred.finish(
                result,
                pooled=self.model.training and torch.is_grad_enabled(),
                process_group=ctx.process_group,
            )
        positions = sorted(
            int(name.split("_")[1])
            for name in result.ratios
            if name.startswith("position_") and name.endswith("_accuracy")
        )
        counts = [
            result.ratios[f"position_{position}_accuracy"] for position in positions
        ]
        metrics = (
            {
                "acc_corrects": [n for n, _ in counts],
                "acc_denoms": [d for _, d in counts],
            }
            if counts
            else {}
        )
        return StepOutput(
            loss=result.loss,
            metrics=metrics,
            ratio_metrics=result.ratios,
            loss_terms=(result.numerator, result.denominator),
        )

    def forward_loss(self, batch, ctx=None):
        ctx = ctx or StepContext()
        if (
            self.window_statistic is None
            or not self.trainable_module().training
            or not torch.is_grad_enabled()
        ):
            return self._forward(batch, ctx)
        device = next(self.model.parameters()).device
        devices = [] if device.type == "cpu" else [device.index]
        device_module = torch.get_device_module(device.type)
        cpu_rng = torch.get_rng_state()
        device_rng = [device_module.get_rng_state(index) for index in devices]
        with torch.no_grad():
            output = self._forward(
                batch, ctx, window_value=torch.zeros((), device=device)
            )
        replay_batch = replace(
            batch,
            tensors={key: value.detach().cpu() for key, value in batch.tensors.items()},
        )

        def replay(window_ratios):
            if self._statistic_window is not window_ratios:
                numerator, denominator = window_ratios[self.window_statistic.ratio_name]
                totals = torch.stack((numerator, denominator))
                if dist.is_initialized():
                    dist.all_reduce(totals)
                denominator = totals[1].where(totals[1] > 0, 1)
                self._window_value = (totals[0] / denominator).detach()
                self._statistic_window = window_ratios
            with torch.random.fork_rng(devices=devices, device_type=device.type):
                torch.set_rng_state(cpu_rng)
                for index, state in zip(devices, device_rng):
                    device_module.set_rng_state(state, index)
                return self._forward(replay_batch, ctx, self._window_value)

        return replace(output, replay_loss=replay)
