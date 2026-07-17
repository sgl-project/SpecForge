# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Materialize a local ``FeatureSource`` into order-aligned ``SampleRef``s."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from specforge.inference.capture import (
    CaptureConfig,
    CaptureMismatchError,
    verify_capture,
)
from specforge.runtime.contracts import PromptTask, SampleRef


@dataclass(frozen=True)
class MaterializationFailure:
    """One task failed validation or persistence without failing its batch."""

    task_id: str
    reason: str
    retryable: bool = True


class MaterializingRefSource:
    """Adapt ``generate_features`` plus a store to the ``RefSource`` protocol.

    Windowed capture consumes order-aligned refs because it owns scheduling and
    retry decisions. Local target engines instead expose ``generate_features``.
    This adapter is the transport-neutral boundary between those APIs: it
    verifies each capture before persistence and returns a typed per-task
    failure when one record can be retried or rejected independently.
    """

    def __init__(
        self,
        feature_source: Any,
        feature_store: Any,
        *,
        run_id: str,
        strategy: str = "eagle3",
        target_model_version: str = "unknown",
        tokenizer_version: str = "unknown",
        draft_weight_version: Optional[str] = None,
    ) -> None:
        if not callable(getattr(feature_source, "generate_features", None)):
            raise TypeError(
                "feature_source must expose generate_features(tasks, capture=...)"
            )
        if not callable(getattr(feature_store, "put", None)):
            raise TypeError("feature_store must expose put(tensors, ...)")
        if not callable(getattr(feature_store, "abort", None)):
            raise TypeError("feature_store must expose abort(sample_id, reason=...)")
        self.feature_source = feature_source
        self.feature_store = feature_store
        self.run_id = run_id
        self.strategy = strategy
        self.target_model_version = target_model_version
        self.tokenizer_version = tokenizer_version
        self.draft_weight_version = draft_weight_version

    def _sample_id(self, task: PromptTask) -> str:
        return f"{self.run_id}:{task.task_id}"

    def _put_metadata(
        self, task: PromptTask, capture: CaptureConfig
    ) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source_task_id": task.task_id,
            "strategy": self.strategy,
            "target_repr": capture.target_repr,
            "vocab_map_version": capture.vocab_map_version,
            "ttt_length": capture.extra.get("ttt_length"),
            "target_model_version": self.target_model_version,
            "tokenizer_version": self.tokenizer_version,
            "draft_weight_version": self.draft_weight_version,
            "num_tokens": int(task.metadata.get("num_tokens", 0)),
        }

    def produce_refs(
        self, tasks: List[PromptTask], *, capture: CaptureConfig
    ) -> List[Union[SampleRef, MaterializationFailure]]:
        """Generate, validate, and persist exactly one aligned result per task."""
        features = self.feature_source.generate_features(tasks, capture=capture)
        if len(features) != len(tasks):
            reason = (
                f"generate_features returned {len(features)} feature records "
                f"for {len(tasks)} tasks"
            )
            return [
                MaterializationFailure(
                    task_id=task.task_id,
                    reason=reason,
                    retryable=False,
                )
                for task in tasks
            ]

        results: List[Union[SampleRef, MaterializationFailure]] = []
        for task, generated in zip(tasks, features):
            sample_id = self._sample_id(task)
            try:
                tensors = dict(generated)
            except (TypeError, ValueError) as exc:
                results.append(
                    MaterializationFailure(
                        task_id=task.task_id,
                        reason=f"invalid feature record: {exc}",
                        retryable=False,
                    )
                )
                continue
            recorded = tensors.pop("__aux_layer_ids__", None)
            try:
                verify_capture(
                    tensors,
                    capture,
                    sample_id=sample_id,
                    recorded_aux_layer_ids=recorded,
                )
            except CaptureMismatchError as exc:
                results.append(
                    MaterializationFailure(
                        task_id=task.task_id,
                        reason=str(exc),
                        retryable=False,
                    )
                )
                continue

            try:
                ref = self.feature_store.put(
                    tensors,
                    sample_id=sample_id,
                    metadata=self._put_metadata(task, capture),
                )
            except Exception as exc:
                try:
                    self.feature_store.abort(sample_id, reason=f"put_failed:{exc}")
                except Exception as cleanup_error:
                    exc.add_note(
                        f"failed to abort partial materialization: {cleanup_error}"
                    )
                results.append(
                    MaterializationFailure(
                        task_id=task.task_id,
                        reason=f"put_failed:{exc}",
                        retryable=True,
                    )
                )
                continue
            results.append(ref)
        return results


__all__ = ["MaterializationFailure", "MaterializingRefSource"]
