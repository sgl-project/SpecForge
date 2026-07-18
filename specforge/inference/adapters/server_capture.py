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

The server knows only generic artifacts (``aux`` = capture layers concatenated,
``last_hidden`` = post-norm final hidden) plus passthrough tensors. The
application composition root injects the algorithm-owned
:class:`ServerCaptureSchema`; this transport does not resolve algorithms.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from specforge.inference.capture import (
    CaptureConfig,
    CaptureMismatchError,
    verify_capture_specs,
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

    aux_feature: Optional[str]
    last_hidden_feature: Optional[str]
    passthrough: Tuple[Tuple[str, str, Tuple[int, ...]], ...]
    attention_mask_feature: Optional[str] = None


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


def _flatten_list_wrappers(value: Any) -> List[Any]:
    """Flatten list-only response wrappers while leaving row objects intact."""
    if not isinstance(value, list):
        return [value]
    flattened: List[Any] = []
    for item in value:
        flattened.extend(_flatten_list_wrappers(item))
    return flattened


def _capture_result_for_task(
    value: Any, *, task_id: str, expected_sample_id: str
) -> Optional[Dict[str, Any]]:
    """Select this task's capture from scalar or batch-wrapped results."""
    candidates = [item for item in _flatten_list_wrappers(value) if item is not None]
    if not candidates:
        return None
    if not all(isinstance(item, dict) for item in candidates):
        types = sorted({type(item).__name__ for item in candidates})
        raise RuntimeError(
            "spec-capture server returned non-object capture results for task "
            f"{task_id}: {types}"
        )
    if len(candidates) == 1:
        return candidates[0]
    matches = [
        item for item in candidates if str(item.get("sample_id")) == expected_sample_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "spec-capture server returned an ambiguous batch result for task "
            f"{task_id}: expected sample_id={expected_sample_id!r}, "
            f"matches={len(matches)}, candidates={len(candidates)}"
        )
    return matches[0]


class SGLangServerCaptureAdapter:
    """RefSource over a live spec-capture SGLang server.

    Implements ``produce_refs`` (not ``generate_features``): the returned
    ``SampleRef``s point at tensors the SERVER already wrote to the Mooncake
    store, so the RolloutWorker commits them without any local put. The refs
    are verified against the :class:`CaptureConfig` from their FeatureSpecs
    alone — same loud extraction-boundary guarantee, no tensor fetch.

    ``store`` must be the run's :class:`MooncakeFeatureStore` (its ``store_id``
    namespaces the keys and ``adopt()`` registers each ref so a later
    generation-aware ``reclaim()``/``gc()`` on the producer side can free
    server-written objects).
    ``post_fn`` is injectable for tests.
    """

    def __init__(
        self,
        base_url: str,
        store,
        *,
        run_id: str,
        algorithm: str,
        schema: ServerCaptureSchema,
        request_input_adapter=None,
        timeout_s: float = 300.0,
        post_fn: Optional[Callable[..., Any]] = None,
        target_model_version: str = "unknown",
        capture_token: Optional[str] = None,
    ) -> None:
        required_store_api = (
            "adopt",
            "discard_external_attempts",
            "reclaim",
            "store_id",
            "track_external_attempt",
        )
        missing_store_api = [
            name for name in required_store_api if not hasattr(store, name)
        ]
        if missing_store_api:
            raise TypeError(
                "SGLangServerCaptureAdapter needs a MooncakeFeatureStore-like "
                f"store; missing {missing_store_api}"
            )
        if run_id != store.store_id:
            raise ValueError(
                "server capture requires run_id == store.store_id; got "
                f"{run_id!r} != {store.store_id!r}"
            )
        self.base_url = base_url.rstrip("/")
        self.store = store
        self.run_id = run_id
        if not isinstance(schema, ServerCaptureSchema):
            raise TypeError("schema must be an injected ServerCaptureSchema")
        if not algorithm:
            raise ValueError("algorithm must be non-empty")
        self.schema = schema
        self.strategy = algorithm
        if request_input_adapter is not None and not callable(
            getattr(request_input_adapter, "build_request_inputs", None)
        ):
            raise TypeError(
                "request_input_adapter must expose build_request_inputs(tasks)"
            )
        self.request_input_adapter = request_input_adapter
        self.timeout_s = timeout_s
        self.post_fn = post_fn or _default_post
        self.target_model_version = target_model_version
        self.capture_token = capture_token or os.environ.get(
            "SGLANG_SPEC_CAPTURE_TOKEN"
        )
        if not self.capture_token:
            raise ValueError(
                "server capture requires capture_token or SGLANG_SPEC_CAPTURE_TOKEN"
            )
        self._healthy = True
        self._rpc_calls = 0
        self._rpc_tasks = 0
        self._rpc_time_s = 0.0
        self._rpc_failures = 0
        self._result_failures = 0

    # -- request construction -------------------------------------------------
    def _sample_id(self, task: PromptTask) -> str:
        return f"{self.run_id}:{task.task_id}"

    def _request_inputs(self, tasks: List[PromptTask]) -> Dict[str, Any]:
        """Build model inputs while keeping transport fields runtime-owned."""

        if self.request_input_adapter is None:
            return {"input_ids": [list(task.payload["input_ids"]) for task in tasks]}
        request_inputs = self.request_input_adapter.build_request_inputs(tasks)
        if not isinstance(request_inputs, Mapping):
            raise TypeError(
                "ServerInputAdapter.build_request_inputs must return a mapping"
            )
        conflicts = sorted({"sampling_params", "spec_capture"} & set(request_inputs))
        if conflicts:
            raise ValueError(
                "ServerInputAdapter cannot set runtime-owned request fields: "
                f"{conflicts}"
            )
        if not request_inputs:
            raise ValueError(
                "ServerInputAdapter.build_request_inputs returned no model inputs"
            )
        return dict(request_inputs)

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
                if payload_key not in task.payload:
                    raise ValueError(
                        f"task {task.task_id}: required capture payload "
                        f"{payload_key!r} is missing"
                    )
                data = list(task.payload[payload_key])
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
            "auth_token": self.capture_token,
            "store_id": self.store.store_id,
            "sample_id": self._sample_id(task),
            # A task id is unique within a run, and retries happen only before
            # its ref is committed.  Keep the generation stable so a response
            # lost after the server write cannot strand gN when the retry writes
            # gN+1.  The server patch replaces these deterministic keys on a
            # retry (``replace``), bounding the attempt to one namespace.
            "gen": 1,
            "replace": int(task.attempt) > 0,
            "features": features,
            "passthrough": passthrough,
        }

    # -- ref construction ------------------------------------------------------
    def _ref_from_result(
        self, task: PromptTask, result: Dict[str, Any], capture: CaptureConfig
    ):
        from specforge.runtime.contracts import SampleRef

        sample_id = str(result["sample_id"])
        gen = int(result["gen"])
        feats: Dict[str, Dict[str, Any]] = result["features"]
        specs: Dict[str, FeatureSpec] = {}
        nbytes = 0
        for name, meta in feats.items():
            if not isinstance(meta, dict):
                raise TypeError(f"feature {name!r} metadata must be an object")
            raw_shape = meta["shape"]
            if not isinstance(raw_shape, (list, tuple)):
                raise TypeError(f"feature {name!r} shape must be a list or tuple")
            if not raw_shape:
                raise ValueError(f"feature {name!r} shape must not be empty")
            if any(type(d) is not int or d <= 0 for d in raw_shape):
                raise ValueError(
                    f"feature {name!r} shape must contain positive integers"
                )
            shape = tuple(raw_shape)
            dtype = meta["dtype"]
            if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES:
                raise ValueError(f"feature {name!r} has unsupported dtype {dtype!r}")
            extra: Dict[str, Any] = {}
            if name == self.schema.last_hidden_feature:
                extra["target_repr"] = capture.target_repr
                if capture.vocab_map_version:
                    extra["target_meta"] = {
                        "vocab_map_version": capture.vocab_map_version
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
                "target_repr": capture.target_repr,
                "vocab_map_version": capture.vocab_map_version,
                "transport": "sglang_server_capture",
                "server": self.base_url,  # which server captured it (provenance)
                "generation": gen,  # the zero-copy get() locator
            },
        )

    def _cleanup_ref(
        self,
        task: PromptTask,
        *,
        sample_id: str,
        store_id: str,
        gen: int,
        feature_names: Iterable[str],
    ) -> "SampleRef":  # noqa: F821
        """Build an exact-generation ref without trusting response tensor metadata."""
        from specforge.runtime.contracts import SampleRef

        names = tuple(sorted(feature_names))
        return SampleRef(
            sample_id=sample_id,
            run_id=self.run_id,
            source_task_id=task.task_id,
            feature_store_uri=f"mooncake://{store_id}/{sample_id}",
            feature_keys={name: f"{sample_id}/{name}" for name in names},
            feature_specs={},
            strategy=self.strategy,
            schema_version=SCHEMA_VERSION,
            target_model_version=self.target_model_version,
            tokenizer_version=str(task.metadata.get("tokenizer_version", "unknown")),
            num_tokens=0,
            estimated_bytes=0,
            metadata={"generation": gen},
        )

    @staticmethod
    def _aux_layer_ids_from_result(
        result: Dict[str, Any],
    ) -> Optional[Tuple[int, ...]]:
        raw = result.get("aux_layer_ids")
        if raw is None:
            return None
        if not isinstance(raw, (list, tuple)) or any(type(v) is not int for v in raw):
            raise ValueError("aux_layer_ids must be a list of integers")
        return tuple(raw)

    # -- the RefSource entry point ----------------------------------------------
    def produce_refs(
        self, tasks: List[PromptTask], *, capture: CaptureConfig
    ) -> List[Union["SampleRef", ServerCaptureFailure]]:  # noqa: F821
        """One batched server call -> one committed-ready ref per task.

        Order-aligned with ``tasks``; a per-task server error becomes a
        :class:`ServerCaptureFailure` (the worker fails just that lease). Any
        transport-level error raises — the worker fails the whole lease batch
        retryable, mirroring ``generate_features`` semantics.
        """
        body = self._request_inputs(tasks)
        body["sampling_params"] = {"temperature": 0.0, "max_new_tokens": 1}
        capture_payloads = [self._spec_capture_payload(t) for t in tasks]
        body["spec_capture"] = capture_payloads
        for payload in capture_payloads:
            feature_names = list(payload["features"].values())
            feature_names.extend(
                item["name"] for item in payload.get("passthrough", ())
            )
            self.store.track_external_attempt(
                payload["sample_id"],
                generation=int(payload["gen"]),
                feature_names=feature_names,
            )
        self._rpc_calls += 1
        self._rpc_tasks += len(tasks)
        rpc_started = time.perf_counter()
        try:
            rows = self.post_fn(
                f"{self.base_url}/generate", json_body=body, timeout=self.timeout_s
            )
        except BaseException:
            self._rpc_failures += 1
            raise
        finally:
            self._rpc_time_s += time.perf_counter() - rpc_started
        rows = _flatten_list_wrappers(rows)
        if len(rows) != len(tasks):
            raise RuntimeError(
                f"spec-capture server returned {len(rows)} rows for "
                f"{len(tasks)} tasks"
            )
        out: List[Union[Any, ServerCaptureFailure]] = []
        successful_refs: List["SampleRef"] = []  # noqa: F821
        for task, request_spec, row in zip(tasks, capture_payloads, rows):
            if not isinstance(row, dict):
                raise RuntimeError(
                    "spec-capture server returned a non-object row for task "
                    f"{task.task_id}: {type(row).__name__}"
                )
            meta = row.get("meta_info") or {}
            # meta_info["spec_capture"] is the per-request result dict (or an
            # {"error": ...} marker) from the server's dedicated output field.
            result = _capture_result_for_task(
                meta.get("spec_capture"),
                task_id=task.task_id,
                expected_sample_id=self._sample_id(task),
            )
            expected_identity = {
                "sample_id": self._sample_id(task),
                "store_id": str(self.store.store_id),
                "gen": 1,
            }
            expected_features = set(request_spec["features"].values()) | {
                item["name"] for item in request_spec["passthrough"]
            }
            cleanup_ref = self._cleanup_ref(
                task,
                sample_id=expected_identity["sample_id"],
                store_id=expected_identity["store_id"],
                gen=expected_identity["gen"],
                feature_names=expected_features,
            )

            def reject(reason: str, *, retryable: bool) -> ServerCaptureFailure:
                self.store.adopt(cleanup_ref)
                self.store.reclaim(cleanup_ref, reason=reason)
                return ServerCaptureFailure(
                    task_id=task.task_id,
                    reason=f"server_capture:{reason}",
                    retryable=retryable,
                )

            if not result:
                out.append(
                    reject(
                        "response carries no spec_capture result; is the server "
                        "patched and launched with --enable-spec-capture?",
                        retryable=False,
                    )
                )
                continue
            if result.get("error"):
                out.append(reject(str(result["error"]), retryable=True))
                continue
            try:
                actual_identity = {
                    "sample_id": str(result.get("sample_id")),
                    "store_id": str(result.get("store_id")),
                    "gen": int(result.get("gen", -1)),
                }
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "spec-capture server returned malformed object identity for "
                    f"task {task.task_id}: {exc}"
                ) from exc
            if actual_identity != expected_identity:
                raise RuntimeError(
                    "spec-capture server returned the wrong object identity for "
                    f"task {task.task_id}: {actual_identity} != "
                    f"{expected_identity}"
                )
            features = result.get("features")
            actual_features = set(features) if isinstance(features, dict) else set()
            if actual_features != expected_features:
                out.append(
                    reject(
                        "response feature set mismatch "
                        f"expected={sorted(expected_features)}, "
                        f"actual={sorted(actual_features)}",
                        retryable=False,
                    )
                )
                continue
            try:
                ref = self._ref_from_result(task, result, capture)
                recorded_aux_layer_ids = self._aux_layer_ids_from_result(result)
            except (KeyError, TypeError, ValueError) as exc:
                out.append(
                    reject(f"malformed feature metadata: {exc}", retryable=False)
                )
                continue
            # A capture shorter than the prompt is corrupt (classic cause: a
            # radix-cache prefix hit skips prefilling — and capturing — the
            # cached tokens; the patched scheduler refuses that config).
            expected_len = len(task.payload["input_ids"])
            short = {
                name: spec.shape
                for name, spec in ref.feature_specs.items()
                if len(spec.shape) >= 2 and spec.shape[1] != expected_len
            }
            if short:
                out.append(
                    reject(
                        f"captured seq len != prompt len {expected_len} for "
                        f"{short}; was the server started without "
                        "--disable-radix-cache?",
                        retryable=False,
                    )
                )
                continue
            try:
                if (
                    capture.aux_hidden_state_layer_ids
                    and recorded_aux_layer_ids is None
                ):
                    raise CaptureMismatchError(
                        f"[{ref.sample_id}] capture omitted aux-layer ids; "
                        "cannot verify requested layers "
                        f"{capture.aux_hidden_state_layer_ids}"
                    )
                verify_capture_specs(
                    ref.feature_specs,
                    capture,
                    sample_id=ref.sample_id,
                    recorded_aux_layer_ids=recorded_aux_layer_ids,
                    aux_feature_name=self.schema.aux_feature or "hidden_state",
                    target_feature_name=self.schema.last_hidden_feature or "target",
                )
            except CaptureMismatchError as exc:
                # Loud boundary failure; free the server-written keys so a
                # mismatched sample is never consumable.
                out.append(reject(f"capture contract mismatch: {exc}", retryable=False))
                continue
            # Do not transfer a successful row out of provisional ownership
            # until every row in the response has passed structural and
            # identity validation.  A later malformed row raises for the whole
            # lease batch; adopting this prefix now would make it invisible to
            # terminal provisional cleanup even though no refs are returned.
            successful_refs.append(ref)
            out.append(ref)
        for ref in successful_refs:
            self.store.adopt(ref)
        self._result_failures += sum(
            isinstance(result, ServerCaptureFailure) for result in out
        )
        return out

    def health(self) -> Dict[str, Any]:
        mean_batch_size = self._rpc_tasks / self._rpc_calls if self._rpc_calls else 0.0
        return {
            "healthy": self._healthy,
            "backend": "sglang_server_capture",
            "base_url": self.base_url,
            "strategy": self.strategy,
            "rpc_calls": self._rpc_calls,
            "rpc_tasks": self._rpc_tasks,
            "rpc_time_s": self._rpc_time_s,
            "rpc_failures": self._rpc_failures,
            "result_failures": self._result_failures,
            "mean_batch_size": mean_batch_size,
            "tasks_per_rpc_second": (
                self._rpc_tasks / self._rpc_time_s if self._rpc_time_s else 0.0
            ),
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
]
