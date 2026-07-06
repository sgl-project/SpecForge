# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Server-side spec-capture rollout source (zero-copy Mooncake transport).

The *server* transport (vs the in-process ``PolicyFeatureAdapter``): a live
SGLang server patched with ``patches/sglang/v0.5.14/spec-capture.patch`` runs
the prefill and writes captured features straight into Mooncake in
:class:`MooncakeFeatureStore`'s key layout. Tensors never pass through this
process — the ``/generate`` response's ``meta_info["spec_capture"]`` carries
only key/shape/dtype, from which :meth:`SGLangServerCaptureAdapter.produce_refs`
builds committed-ready ``SampleRef``s.

Strategy-agnostic: the server knows only generic artifacts (``aux`` = capture
layers concatenated, ``last_hidden`` = post-norm final hidden) plus passthrough
tensors; :class:`ServerCaptureSchema` maps them onto each strategy's feature
names/shapes (eagle3 / dflash / domino below; add more via
:func:`register_server_capture_schema`). eagle3 stores the target as the last
hidden state (``target_repr="hidden_state"``, offline convention — the trainer
re-runs the frozen ``TargetHead``), not logits (~50x the traffic).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from specforge.inference.capture import (
    FeatureContract,
    FeatureContractError,
    verify_feature_contract_specs,
)
from specforge.runtime.contracts import SCHEMA_VERSION, FeatureSpec, PromptTask

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServerCaptureSchema:
    """Maps a strategy's feature names onto the server's capture artifacts.

    ``aux_feature`` / ``last_hidden_feature`` name the features fed by the two
    engine artifacts (None = not wanted). ``passthrough`` is
    ``(feature_name, payload_key, trailing_shape)`` for client tensors stored
    verbatim (``trailing_shape`` is appended after ``(1, L)``).
    ``attention_mask_feature`` is synthesized all-ones (PromptTasks are unpadded).
    """

    strategy: str
    aux_feature: Optional[str]
    last_hidden_feature: Optional[str]
    passthrough: Tuple[Tuple[str, str, Tuple[int, ...]], ...]
    attention_mask_feature: Optional[str] = None


_SERVER_CAPTURE_SCHEMAS: Dict[str, ServerCaptureSchema] = {}


def register_server_capture_schema(schema: ServerCaptureSchema) -> None:
    existing = _SERVER_CAPTURE_SCHEMAS.get(schema.strategy)
    if existing is not None and existing != schema:
        raise ValueError(
            f"server-capture schema for {schema.strategy!r} already registered"
        )
    _SERVER_CAPTURE_SCHEMAS[schema.strategy] = schema


def resolve_server_capture_schema(strategy: str) -> ServerCaptureSchema:
    schema = _SERVER_CAPTURE_SCHEMAS.get(strategy)
    if schema is None:
        raise KeyError(
            f"no server-capture schema registered for strategy {strategy!r}; "
            f"registered: {sorted(_SERVER_CAPTURE_SCHEMAS)}"
        )
    return schema


register_server_capture_schema(
    ServerCaptureSchema(
        strategy="eagle3",
        aux_feature="hidden_state",
        last_hidden_feature="target",  # target_repr="hidden_state"
        passthrough=(
            ("input_ids", "input_ids", ()),
            # (1, L): the hidden_state train path mirrors the OFFLINE
            # convention — TargetHead.preprocess adds the trailing mask dim.
            ("loss_mask", "loss_mask", ()),
        ),
        attention_mask_feature="attention_mask",
    )
)
register_server_capture_schema(
    ServerCaptureSchema(
        strategy="dflash",
        aux_feature="hidden_states",
        last_hidden_feature=None,  # hard-label training: no target distribution
        passthrough=(
            ("input_ids", "input_ids", ()),
            ("loss_mask", "loss_mask", ()),
        ),
    )
)
register_server_capture_schema(
    ServerCaptureSchema(
        strategy="domino",
        aux_feature="hidden_states",
        last_hidden_feature=None,
        passthrough=(
            ("input_ids", "input_ids", ()),
            ("loss_mask", "loss_mask", ()),
        ),
    )
)


@dataclass(frozen=True)
class ServerCaptureFailure:
    """Per-task failure from the server transport (worker fails just this task)."""

    task_id: str
    reason: str
    retryable: bool = True


def _default_post(url: str, json_body: Dict[str, Any], timeout: float):
    import requests

    resp = requests.post(url, json=json_body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


class SGLangServerCaptureAdapter:
    """RefSource over a live spec-capture SGLang server.

    Implements ``produce_refs`` (not ``generate_features``): the returned
    ``SampleRef``s point at tensors the SERVER already wrote to the Mooncake
    store, so the RolloutWorker commits them without any local put. The refs
    are verified against the :class:`FeatureContract` from their FeatureSpecs
    alone — same loud extraction-boundary guarantee, no tensor fetch.

    ``store`` must be the run's :class:`MooncakeFeatureStore` (its ``store_id``
    namespaces the keys and ``adopt()`` registers each ref so a later
    ``abort()``/``gc()`` on the producer side can free server-written objects).
    ``post_fn`` is injectable for tests.
    """

    def __init__(
        self,
        base_url: str,
        store,
        *,
        run_id: str,
        strategy: str = "eagle3",
        schema: Optional[ServerCaptureSchema] = None,
        timeout_s: float = 300.0,
        post_fn: Optional[Callable[..., Any]] = None,
        target_model_version: str = "unknown",
    ) -> None:
        if not hasattr(store, "adopt") or not hasattr(store, "store_id"):
            raise TypeError(
                "SGLangServerCaptureAdapter needs a MooncakeFeatureStore-like "
                "store exposing .store_id and .adopt(ref)"
            )
        self.base_url = base_url.rstrip("/")
        self.store = store
        self.run_id = run_id
        self.schema = schema or resolve_server_capture_schema(strategy)
        self.strategy = self.schema.strategy
        self.timeout_s = timeout_s
        self.post_fn = post_fn or _default_post
        self.target_model_version = target_model_version
        self._healthy = True

    # -- request construction -------------------------------------------------
    def _sample_id(self, task: PromptTask) -> str:
        return f"{self.run_id}:{task.task_id}"

    def _spec_capture_payload(self, task: PromptTask) -> Dict[str, Any]:
        input_ids = list(task.payload["input_ids"])
        length = len(input_ids)
        features: Dict[str, str] = {}
        if self.schema.aux_feature is not None:
            features["aux"] = self.schema.aux_feature
        if self.schema.last_hidden_feature is not None:
            features["last_hidden"] = self.schema.last_hidden_feature
        passthrough: List[Dict[str, Any]] = []
        for feature_name, payload_key, trailing in self.schema.passthrough:
            if payload_key == "input_ids":
                data = input_ids
            else:
                data = list(task.payload.get(payload_key, [1] * length))
            if len(data) != length:
                raise ValueError(
                    f"task {task.task_id}: payload {payload_key!r} length "
                    f"{len(data)} != input_ids length {length}"
                )
            passthrough.append(
                {
                    "name": feature_name,
                    "data": data,
                    "shape": [1, length, *trailing],
                    "dtype": "int64",
                }
            )
        if self.schema.attention_mask_feature is not None:
            passthrough.append(
                {
                    "name": self.schema.attention_mask_feature,
                    "data": [1] * length,
                    "shape": [1, length],
                    "dtype": "int64",
                }
            )
        return {
            "store_id": self.store.store_id,
            "sample_id": self._sample_id(task),
            # retried tasks re-capture under a bumped generation so a stale
            # ref from the failed attempt can never resolve (B5)
            "gen": int(task.attempt) + 1,
            "features": features,
            "passthrough": passthrough,
        }

    # -- ref construction ------------------------------------------------------
    def _ref_from_result(
        self, task: PromptTask, result: Dict[str, Any], contract: FeatureContract
    ):
        from specforge.runtime.contracts import SampleRef

        sample_id = str(result["sample_id"])
        gen = int(result["gen"])
        feats: Dict[str, Dict[str, Any]] = result["features"]
        specs: Dict[str, FeatureSpec] = {}
        nbytes = 0
        for name, meta in feats.items():
            shape = tuple(int(d) for d in meta["shape"])
            dtype = str(meta["dtype"])
            extra: Dict[str, Any] = {}
            if name == self.schema.last_hidden_feature:
                extra["target_repr"] = contract.target_repr
                if contract.vocab_map_version:
                    extra["target_meta"] = {
                        "vocab_map_version": contract.vocab_map_version
                    }
            specs[name] = FeatureSpec(name=name, shape=shape, dtype=dtype, **extra)
            nbytes += _spec_nbytes(shape, dtype)
        num_tokens = int(task.metadata.get("num_tokens", 0)) or len(
            task.payload["input_ids"]
        )
        return SampleRef(
            sample_id=sample_id,
            run_id=self.run_id,
            source_task_id=task.task_id,
            feature_store_uri=f"mooncake://{result['store_id']}/{sample_id}",
            feature_keys={n: f"{sample_id}/{n}" for n in specs},
            feature_specs=specs,
            strategy=self.strategy,
            schema_version=SCHEMA_VERSION,
            target_model_version=self.target_model_version,
            tokenizer_version=str(task.metadata.get("tokenizer_version", "unknown")),
            num_tokens=num_tokens,
            estimated_bytes=nbytes,
            metadata={
                "run_id": self.run_id,
                "source_task_id": task.task_id,
                "strategy": self.strategy,
                "target_repr": contract.target_repr,
                "vocab_map_version": contract.vocab_map_version,
                "transport": "sglang_server_capture",
                "generation": gen,  # the zero-copy get() locator
            },
        )

    # -- the RefSource entry point ----------------------------------------------
    def produce_refs(
        self, tasks: List[PromptTask], *, capture: FeatureContract
    ) -> List[Union["SampleRef", ServerCaptureFailure]]:  # noqa: F821
        """One batched server call -> one committed-ready ref per task.

        Order-aligned with ``tasks``; a per-task server error becomes a
        :class:`ServerCaptureFailure` (the worker fails just that lease). Any
        transport-level error raises — the worker fails the whole lease batch
        retryable, mirroring ``generate_features`` semantics.
        """
        body = {
            "input_ids": [list(t.payload["input_ids"]) for t in tasks],
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
            "spec_capture": [self._spec_capture_payload(t) for t in tasks],
        }
        rows = self.post_fn(
            f"{self.base_url}/generate", json_body=body, timeout=self.timeout_s
        )
        if isinstance(rows, dict):
            rows = [rows]
        if len(rows) != len(tasks):
            raise RuntimeError(
                f"spec-capture server returned {len(rows)} rows for "
                f"{len(tasks)} tasks"
            )
        out: List[Union[Any, ServerCaptureFailure]] = []
        for task, row in zip(tasks, rows):
            meta = row.get("meta_info") or {}
            # meta_info["spec_capture"] is the per-request result dict (or an
            # {"error": ...} marker) from the server's dedicated output field.
            result = meta.get("spec_capture")
            if not result:
                out.append(
                    ServerCaptureFailure(
                        task_id=task.task_id,
                        reason=(
                            "server_capture: response carries no spec_capture "
                            "result — is the server patched and launched with "
                            "--enable-spec-capture?"
                        ),
                        retryable=False,
                    )
                )
                continue
            if result.get("error"):
                out.append(
                    ServerCaptureFailure(
                        task_id=task.task_id,
                        reason=f"server_capture:{result['error']}",
                        retryable=True,
                    )
                )
                continue
            ref = self._ref_from_result(task, result, capture)
            try:
                verify_feature_contract_specs(
                    ref.feature_specs,
                    capture,
                    sample_id=ref.sample_id,
                    recorded_aux_layer_ids=(
                        tuple(result["aux_layer_ids"])
                        if result.get("aux_layer_ids") is not None
                        else None
                    ),
                    aux_feature_name=self.schema.aux_feature or "hidden_state",
                    target_feature_name=self.schema.last_hidden_feature or "target",
                )
            except FeatureContractError as exc:
                # Loud boundary failure; free the server-written keys so a
                # mismatched sample is never consumable.
                self.store.adopt(ref)
                self.store.abort(ref.sample_id, reason=f"contract:{exc}")
                out.append(
                    ServerCaptureFailure(
                        task_id=task.task_id, reason=str(exc), retryable=False
                    )
                )
                continue
            self.store.adopt(ref)
            out.append(ref)
        return out

    def health(self) -> Dict[str, Any]:
        return {
            "healthy": self._healthy,
            "backend": "sglang_server_capture",
            "base_url": self.base_url,
            "strategy": self.strategy,
        }


_DTYPE_BYTES = {
    "float64": 8,
    "int64": 8,
    "float32": 4,
    "int32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
}


def _spec_nbytes(shape: Tuple[int, ...], dtype: str) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n * _DTYPE_BYTES.get(dtype, 2)


__all__ = [
    "ServerCaptureSchema",
    "ServerCaptureFailure",
    "SGLangServerCaptureAdapter",
    "register_server_capture_schema",
    "resolve_server_capture_schema",
]
