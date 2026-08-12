# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""In-process SGLang capture adapter for colocated online training."""

from __future__ import annotations

from typing import List

from specforge.algorithms.common.providers import ServerStreamingProvider
from specforge.inference.batch_partition import TargetBatchPartition
from specforge.inference.capture import CaptureConfig
from specforge.runtime.contracts import PromptTask


class LocalSGLangCaptureAdapter:
    """Map local SGLang hidden-state rows to an algorithm streaming schema."""

    def __init__(
        self,
        capture_model,
        *,
        provider: ServerStreamingProvider,
        synchronize_after_capture: bool = True,
        batch_partition: TargetBatchPartition | None = None,
    ) -> None:
        self.capture_model = capture_model
        self.provider = provider
        self.synchronize_after_capture = bool(synchronize_after_capture)
        self.batch_partition = batch_partition or TargetBatchPartition()
        self.returns_local_batch = True

    def set_batch_partition(self, partition: TargetBatchPartition) -> None:
        self.batch_partition = partition

    def _synchronize(self) -> None:
        if not self.synchronize_after_capture:
            return
        import torch

        device = next(
            self.capture_model._backend.model_runner.model.parameters()
        ).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "npu":
            torch.npu.synchronize(device)

    def generate_features(
        self,
        tasks: List[PromptTask],
        *,
        capture: CaptureConfig,
    ):
        del capture
        import torch

        input_rows = [list(task.payload["input_ids"]) for task in tasks]
        aux_rows, last_rows = self.capture_model.capture_rows(input_rows)
        local_slice = self.batch_partition.local_slice(len(tasks))
        local_tasks = tasks[local_slice]
        if self.batch_partition.size > 1:
            # torch.split returns views into the full TP-wide packed output.
            # Clone only this rank's rows before dropping those views; otherwise
            # a one-sample feature would pin the entire TP batch's backing
            # allocation throughout draft forward/backward (especially costly
            # for 64K K3 captures).
            local_aux_rows = tuple(row.clone() for row in aux_rows[local_slice])
            local_last_rows = tuple(row.clone() for row in last_rows[local_slice])
        else:
            local_aux_rows = aux_rows
            local_last_rows = last_rows
        del aux_rows, last_rows
        self._synchronize()
        layout = self.provider.layout
        features = []
        for task, aux, last in zip(local_tasks, local_aux_rows, local_last_rows):
            row = {}
            device = aux.device
            for name, payload_key, trailing_shape in layout.passthrough:
                values = task.payload[payload_key]
                tensor = torch.as_tensor(values, device=device, dtype=torch.long)
                row[name] = tensor.reshape(1, len(values), *trailing_shape)
            if layout.attention_mask_feature is not None:
                row[layout.attention_mask_feature] = torch.ones(
                    (1, len(task.payload["input_ids"])),
                    device=device,
                    dtype=torch.long,
                )
            if layout.aux_feature is not None:
                row[layout.aux_feature] = aux.unsqueeze(0)
            if layout.last_hidden_feature is not None:
                row[layout.last_hidden_feature] = last.unsqueeze(0)
            row["__aux_layer_ids__"] = tuple(self.capture_model.capture_layers or ())
            features.append(row)
        return features


__all__ = ["LocalSGLangCaptureAdapter"]
