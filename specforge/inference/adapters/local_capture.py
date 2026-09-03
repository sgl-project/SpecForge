# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""In-process SGLang capture adapter for colocated online training."""

from __future__ import annotations

from typing import Dict, List

from specforge.algorithms.common.providers import (
    ServerCaptureLayout,
    ServerStreamingProvider,
)
from specforge.inference.batch_partition import TargetBatchPartition
from specforge.inference.capture import CaptureConfig
from specforge.runtime.contracts import PromptTask


def _feature_row(layout: ServerCaptureLayout, task: PromptTask, *, aux, last) -> Dict:
    """Map one captured row onto the algorithm's streaming feature names.

    Emits the same record the patched SGLang server produces for the
    disaggregated transport: passthrough payloads as ``(1, L, *trailing)`` long
    tensors, an all-ones attention mask (prompts are unpadded), and ``(1, L, H)``
    auxiliary / final hidden states.
    """
    import torch

    device = aux.device
    length = len(task.payload["input_ids"])
    row: Dict = {}
    for name, payload_key, trailing_shape in layout.passthrough:
        values = task.payload[payload_key]
        tensor = torch.as_tensor(values, device=device, dtype=torch.long)
        row[name] = tensor.reshape(1, len(values), *trailing_shape)
    if layout.attention_mask_feature is not None:
        row[layout.attention_mask_feature] = torch.ones(
            (1, length), device=device, dtype=torch.long
        )
    if layout.aux_feature is not None:
        row[layout.aux_feature] = aux.unsqueeze(0)
    if layout.last_hidden_feature is not None:
        row[layout.last_hidden_feature] = last.unsqueeze(0)
    return row


class LocalSGLangCaptureAdapter:
    """Feed a local SGLang capture engine's rows to a ``RolloutWorker``.

    The worker leases one TP-wide task batch; every TP peer captures all of it
    (the SGLang forward is collective) and this adapter returns only the peer's
    own contiguous slice (``returns_local_batch``), detached from the packed
    capture output so the other slices' memory is released before training.
    """

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
        """Order SGLang's streams before the trainer's FSDP collectives.

        Capture runs on this rank's current accelerator, so synchronizing the
        current device is sufficient.
        """
        if not self.synchronize_after_capture:
            return
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif getattr(torch, "npu", None) is not None and torch.npu.is_available():
            torch.npu.synchronize()

    def generate_features(
        self,
        tasks: List[PromptTask],
        *,
        capture: CaptureConfig,
    ) -> List[Dict]:
        # The capture contract (required tensors, aux layer ids) is enforced by
        # RolloutWorker.verify_capture against the layer ids recorded below.
        del capture
        aux_rows, last_rows = self.capture_model.capture_rows(
            [list(task.payload["input_ids"]) for task in tasks]
        )
        local = self.batch_partition.local_slice(len(tasks))
        if self.batch_partition.size > 1:
            # torch.split returns views into the full TP-wide packed output.
            # Clone only this rank's rows before dropping those views; otherwise
            # one sample's features would pin the whole TP batch's allocation
            # through draft forward/backward.
            local_aux = tuple(row.clone() for row in aux_rows[local])
            local_last = tuple(row.clone() for row in last_rows[local])
        else:
            local_aux, local_last = aux_rows, last_rows
        del aux_rows, last_rows
        self._synchronize()

        layer_ids = tuple(self.capture_model.capture_layers or ())
        features = []
        for task, aux, last in zip(tasks[local], local_aux, local_last):
            row = _feature_row(self.provider.layout, task, aux=aux, last=last)
            row["__aux_layer_ids__"] = layer_ids
            features.append(row)
        return features


__all__ = ["LocalSGLangCaptureAdapter"]
