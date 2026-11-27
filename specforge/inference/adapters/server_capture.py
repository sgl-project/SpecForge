# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Server-side spec-capture rollout source (zero-copy Mooncake transport).

An external SGLang capture server runs the prefill and writes captured
features straight into Mooncake in :class:`MooncakeFeatureStore`'s key layout.
The server runs the ``specforge-sglang-capture`` plugin
(``server_capture="plugin"``: the spec rides ``sampling_params.custom_params``).
``server_capture="auto"`` reads the backend from the server's ``/server_info``
before the first capture request.
Tensors never pass through this process — the ``/generate`` response's
``meta_info["spec_capture"]`` carries only key/shape/dtype, from which
:meth:`SGLangServerCaptureAdapter.produce_refs` builds committed-ready
``SampleRef``s.

The server knows only generic artifacts (``aux`` = capture layers
concatenated, ``last_hidden`` = post-norm final hidden) plus passthrough
tensors.  The application composition root injects an algorithm-owned
:class:`ServerCaptureSchema`; this transport never resolves an algorithm name.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

from specforge.inference.capture import (
    CaptureConfig,
    CaptureMismatchError,
    verify_capture_specs,
)
from specforge.inference.capture_protocol import (
    CaptureCapacityDeclined,
    CaptureProtocol,
    attempt_keys,
    request_digest,
)
from specforge.runtime.contracts import (
    SCHEMA_VERSION,
    FeatureSpec,
    PromptTask,
    SampleRef,
)
from specforge.runtime.data_plane.feature_store import abort_feature_ref

logger = logging.getLogger(__name__)

#: The supported capture integration.
SERVER_CAPTURE_BACKENDS = ("plugin",)
_SERVER_INFO_TIMEOUT_S = 30.0
_CAPTURE_HINTS = {
    "plugin": "the specforge-sglang-capture plugin with SPECFORGE_SPEC_CAPTURE=1",
}


def detect_server_capture(server_info: Mapping[str, Any], *, server_url: str) -> str:
    """The capture backend a running server advertises in ``/server_info``.

    The enabled plugin marks each scheduler with ``spec_capture_backend``.
    """
    states = server_info.get("internal_states") or []
    if any(
        isinstance(state, Mapping) and state.get("spec_capture_backend") == "plugin"
        for state in states
    ):
        return "plugin"
    raise RuntimeError(
        f"{server_url} advertises no spec capture in /server_info: launch it with "
        "specforge-sglang-capture plugin and SPECFORGE_SPEC_CAPTURE=1; or set "
        "deployment.disaggregated.server_capture (or DISAGG_SERVER_CAPTURE) to "
        "plugin"
    )


def _get_server_info(url: str, timeout: float) -> Mapping[str, Any]:
    import requests

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


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


class _SessionPost:
    """Default ``post_fn``: one keep-alive ``requests.Session`` per thread.

    ``requests.post`` opens and closes a connection for every ``/generate``.
    A ``Session`` is not documented as thread-safe and the producer drives one
    adapter from ``producer_concurrency`` threads, so each thread keeps its
    own. Transport errors still raise (no automatic resend), leaving retries
    to the lease-level policy of :meth:`SGLangServerCaptureAdapter.produce_refs`.
    """

    def __init__(self) -> None:
        self._local = threading.local()

    def _session(self):
        session = getattr(self._local, "session", None)
        if session is None:
            import requests

            session = requests.Session()
            self._local.session = session
        return session

    def __call__(self, url: str, json_body: Dict[str, Any], timeout: float):
        resp = self._session().post(url, json=json_body, timeout=timeout)
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
    ``abort()``/``gc()`` on the producer side can free server-written objects).
    ``post_fn`` is injectable for tests; the default reuses one HTTP
    connection per calling thread. ``server_info_fn(url, timeout)`` fetches
    ``/server_info`` when ``server_capture="auto"``.
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
        server_capture: str = "plugin",
        server_info_fn: Optional[Callable[[str, float], Mapping[str, Any]]] = None,
        hs_capture: bool = False,
        hs_prefix_cache: bool = False,
        control_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        if server_capture not in ("auto", *SERVER_CAPTURE_BACKENDS):
            raise ValueError(
                "server_capture must be one of "
                f"{('auto', *SERVER_CAPTURE_BACKENDS)}, got {server_capture!r}"
            )
        required_store_api = (
            "adopt",
            "discard_external_attempts",
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
        self.post_fn = post_fn or _SessionPost()
        if hs_prefix_cache and not hs_capture:
            raise ValueError("hs_prefix_cache requires explicit hs_capture opt-in")
        self.hs_capture = hs_capture
        self.hs_prefix_cache = hs_prefix_cache

        self._capture_protocol = (
            CaptureProtocol(base_url, control_fn or self.post_fn, timeout_s)
            if hs_capture
            else None
        )
        self.target_model_version = target_model_version
        #: ``auto`` until the first request reads the server's backend.
        self.server_capture = server_capture
        self.server_info_fn = server_info_fn or _get_server_info
        self._server_capture_lock = threading.Lock()
        self._healthy = True

    def _resolve_server_capture(self) -> str:
        with self._server_capture_lock:
            if self.server_capture == "auto":
                try:
                    info = self.server_info_fn(
                        f"{self.base_url}/server_info", _SERVER_INFO_TIMEOUT_S
                    )
                except Exception as exc:
                    # Raised before any write: the lease is retried, as for a
                    # failed /generate. A later request detects again.
                    raise RuntimeError(
                        f"cannot read {self.base_url}/server_info to detect its "
                        f"spec-capture backend: {exc}"
                    ) from exc
                self.server_capture = detect_server_capture(
                    info, server_url=self.base_url
                )
                logger.info(
                    "%s runs spec capture via the %s",
                    self.base_url,
                    self.server_capture,
                )
            return self.server_capture

    # -- request construction -------------------------------------------------
    def _sample_id(self, task: PromptTask) -> str:
        return f"{self.run_id}:{task.task_id}"

    def _request_inputs(self, tasks: List[PromptTask]) -> Dict[str, Any]:
        """Build only model-input fields, keeping transport keys runtime-owned."""

        if self.request_input_adapter is None:
            return {"input_ids": [list(task.payload["input_ids"]) for task in tasks]}
        request_inputs = self.request_input_adapter.build_request_inputs(tasks)
        if not isinstance(request_inputs, Mapping):
            raise TypeError(
                "ServerInputAdapter.build_request_inputs must return a mapping"
            )
        reserved = {"extra_key", "sampling_params", "spec_capture"}
        conflicts = sorted(reserved & set(request_inputs))
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
            "store_id": self.store.store_id,
            "sample_id": self._sample_id(task),
            # A task id is unique within a run, and retries happen only before
            # its ref is committed.  Keep the generation stable so a response
            # lost after the server write cannot strand gN when the retry writes
            # gN+1.  The server plugin replaces these deterministic keys on a
            # retry (``replace``), bounding the attempt to one namespace.
            "gen": 1,
            "replace": int(task.attempt) > 0,
            "features": features,
            "passthrough": passthrough,
        }

    # -- ref construction ------------------------------------------------------
    def _ref_from_result(
        self, task: PromptTask, result: Dict[str, Any], capture: CaptureConfig
    ) -> SampleRef:

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
                extra["target_repr"] = capture.target_repr
                if capture.vocab_map_version:
                    extra["target_meta"] = {
                        "vocab_map_version": capture.vocab_map_version
                    }
            specs[name] = FeatureSpec(name=name, shape=shape, dtype=dtype, **extra)
            nbytes += _spec_nbytes(shape, dtype)
        if result.get("publication_ownership") == "attempt-v1":
            feature_keys = dict(result["physical_keys"])
        else:
            feature_keys = {name: f"{sample_id}/{name}" for name in specs}
        ref = SampleRef(
            sample_id=sample_id,
            run_id=self.run_id,
            source_task_id=task.task_id,
            feature_store_uri=f"mooncake://{result['store_id']}/{sample_id}",
            feature_keys=feature_keys,
            feature_specs=specs,
            strategy=self.strategy,
            schema_version=SCHEMA_VERSION,
            target_model_version=self.target_model_version,
            tokenizer_version=str(task.metadata.get("tokenizer_version", "unknown")),
            num_tokens=len(task.payload["input_ids"]),
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
                "valid_anchor_count": sum(
                    left > 0.5 and right > 0.5
                    for left, right in zip(
                        task.payload.get("loss_mask", []),
                        task.payload.get("loss_mask", [])[1:],
                    )
                ),
            },
        )
        if result.get("protocol_version") == 1:
            ref.metadata["capture_fence"] = {
                **{
                    name: result[name]
                    for name in (
                        "server_epoch",
                        "store_id",
                        "sample_id",
                        "gen",
                        "attempt_id",
                        "reservation_token",
                    )
                },
                "protocol_version": 1,
                "control_url": self.base_url + "/spec_capture/control",
                "timeout_s": self.timeout_s,
                **{
                    name: result[name]
                    for name in (
                        "object_key_layout",
                        "publication_ownership",
                        "physical_keys",
                        "writer_drained",
                        "state",
                    )
                    if name in result
                },
            }
            ref.metadata["capture_diagnostics"] = {
                name: result.get(name)
                for name in (
                    "H",
                    "L",
                    "N",
                    "span",
                    "execution_epoch",
                    "rank",
                    "reused_tokens",
                    "capture_namespace",
                    "capture_signature",
                )
            }
        if "arena" in result:
            # Arena objects stay on the capture server; currently only online
            # NVLink capture uses the arena.
            from specforge.runtime.data_plane.mooncake_arena import REF_METADATA_KEY

            ref.metadata[REF_METADATA_KEY] = {
                "session": str(result["arena"]["session"]),
                "control": str(result["arena"]["control"]),
                "addresses": {
                    name: int(meta["address"]) for name, meta in feats.items()
                },
            }
        return ref

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
        prepared = self.prepare_refs_batches([tasks], capture=capture)
        self.register_prepared_batches(prepared)
        return self.produce_prepared(prepared[0], capture=capture)

    def prepare_refs_batches(self, task_batches, *, capture: CaptureConfig):
        """Build bounded lease groups without registering a writer yet."""
        self._resolve_server_capture()
        capability = (
            self._capture_protocol.capability()
            if self._capture_protocol is not None
            else None
        )
        if self.hs_capture:
            if (
                capability is None
                or capability.get("published_ownership") != "attempt-v1"
            ):
                raise RuntimeError(
                    "HS capture requires native plugin attempt-v1 capability"
                )
            if capability.get("cache_enabled") is not self.hs_prefix_cache:
                raise RuntimeError("HS cache setting disagrees with server capability")
        if capability is not None:
            required = ("prepare_capture_retry", "track_capture_fence")
            if any(not callable(getattr(self.store, name, None)) for name in required):
                raise TypeError("negotiated capture requires attempt-fenced store")
            requested_layers = tuple(capture.aux_hidden_state_layer_ids or ())
            schema = {item["name"]: item for item in capability["schema"]}
            if (
                requested_layers
                and schema.get("aux", {}).get("width") != capture.expected_aux_width
            ):
                raise CaptureMismatchError("server capability aux width mismatch")
            if (
                self.schema.last_hidden_feature is not None
                and schema.get("last_hidden", {}).get("width")
                != capture.target_hidden_size
            ):
                raise CaptureMismatchError(
                    "server capability last hidden width mismatch"
                )
            if (
                requested_layers
                and tuple(schema.get("aux", {}).get("layer_ids", ()))
                != requested_layers
            ):
                raise CaptureMismatchError("server capability aux layer order mismatch")
        prepared = []
        for tasks in task_batches:
            body = self._request_inputs(tasks)
            body["extra_key"] = [uuid.uuid4().hex for _ in tasks]
            payloads = [self._spec_capture_payload(task) for task in tasks]
            if capability is not None:
                for payload in payloads:
                    self.store.prepare_capture_retry(payload["sample_id"])
                if capability["cache_enabled"]:
                    body["extra_key"] = [capability["capture_namespace"]] * len(tasks)
            prepared.append(
                dict(
                    tasks=tasks,
                    body=body,
                    payloads=payloads,
                    capability=capability,
                    estimated_bytes=self._output_bytes(tasks, capability),
                )
            )
        return prepared

    def _output_bytes(
        self, tasks: list[PromptTask], capability: dict | None
    ) -> int | None:
        # The negotiated schema fixes the complete engine tensor width/dtype.
        # Passthrough tensors are int64 and have client-declared trailing dims.
        if capability is None:
            return None
        schema = {item["name"]: item for item in capability["schema"]}
        per_token = 0
        for artifact, feature in (
            ("aux", self.schema.aux_feature),
            ("last_hidden", self.schema.last_hidden_feature),
        ):
            if feature is not None:
                item = schema[artifact]
                width = int(item["width"])
                if width < 1 or item["dtype"] not in _DTYPE_BYTES:
                    raise ValueError("invalid negotiated capture output schema")
                per_token += width * _DTYPE_BYTES[item["dtype"]]
        for _name, _key, trailing in self.schema.passthrough:
            per_token += _spec_nbytes((1, *trailing), "int64")
        if self.schema.attention_mask_feature is not None:
            per_token += 8
        return sum(len(task.payload["input_ids"]) * per_token for task in tasks)

    def registration_capacity(self) -> int | None:
        """Return the negotiated endpoint's atomic begin bound before leasing."""
        if self._capture_protocol is None:
            return None
        capability = self._capture_protocol.capability()
        capacity = self._registration_capacity(capability)
        if capability is not None and not capability.get("begin_batch"):
            return 1
        return capacity

    @staticmethod
    def _registration_capacity(capability: dict | None) -> int | None:
        if capability is None:
            return None
        capacity = capability.get("limits", {}).get("attempt_capacity")
        if capacity is not None and (
            isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1
        ):
            raise RuntimeError("invalid capture attempt capacity")
        return capacity

    def register_prepared_batches(self, prepared):
        """Coalesce all currently ready leases into one endpoint begin_batch."""
        if not prepared:
            return
        capability = prepared[0]["capability"]
        if any(item["capability"] != capability for item in prepared):
            raise RuntimeError("capture capability changed within registration batch")
        if capability is None:
            return
        payloads = [payload for item in prepared for payload in item["payloads"]]
        tasks = [task for item in prepared for task in item["tasks"]]
        capacity = self._registration_capacity(capability)
        if capacity is not None and len(tasks) > capacity:
            # No attempt identity or local begin intent exists yet. The producer
            # splits ready groups before this boundary; direct callers receive a
            # local error rather than an ambiguous partial remote reservation.
            raise ValueError("registration batch exceeds capture attempt capacity")
        requests = []
        for item in prepared:
            item["begin_control_url"] = self._capture_protocol.url
        retryable_payloads = {
            id(payload)
            for item in prepared
            if item.get("registration_refused")
            for payload in item["payloads"]
        }
        for task, payload in zip(tasks, payloads):
            names = [
                *payload["features"].values(),
                *(x["name"] for x in payload.get("passthrough", ())),
            ]
            # Freeze the begin identity before crossing HTTP. Even a lost reply
            # must not let a late cleanup retarget a peer's replacement writer.
            if "attempt_id" in payload and id(payload) not in retryable_payloads:
                raise RuntimeError("prepared capture registration cannot be replayed")
            payload.update(
                attempt_id=payload.get("attempt_id", uuid.uuid4().hex),
                server_epoch=capability["server_epoch"],
                replace=False,
            )
            payload["request_digest"] = request_digest(
                task.payload["input_ids"], payload
            )
            requests.append(
                (payload, task.payload["input_ids"], payload["attempt_id"], names)
            )
        for item in prepared:
            item["registration_refused"] = False
        try:
            fences = self.store.begin_capture_attempts(
                self._capture_protocol, capability, requests
            )
        except CaptureCapacityDeclined:
            for item in prepared:
                item["registration_refused"] = True
            raise
        for payload, fence in zip(payloads, fences):
            payload.update(
                {
                    name: capability[name]
                    for name in (
                        "server_epoch",
                        "capture_namespace",
                        "capture_signature",
                    )
                }
            )
            payload.update(
                {
                    name: fence[name]
                    for name in ("protocol_version", "attempt_id", "reservation_token")
                }
            )
            if fence.get("object_key_layout") == "attempt-v1":
                payload["object_key_layout"] = "attempt-v1"

    @staticmethod
    def _verify_hs_feature_metadata(
        result: dict, payload: dict, capability: dict
    ) -> None:
        """Check the complete negotiated schema, including passthrough dtype."""
        schema = {item["name"]: item for item in capability["schema"]}
        n = result.get("N")
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise RuntimeError("negotiated capture returned invalid token count")
        expected = {
            name: ((1, n, int(schema[artifact]["width"])), schema[artifact]["dtype"])
            for artifact, name in payload["features"].items()
        }
        expected.update(
            {
                item["name"]: (tuple(item["shape"]), item["dtype"])
                for item in payload.get("passthrough", ())
            }
        )
        actual = result.get("features", {})
        if set(actual) != set(expected):
            raise RuntimeError("negotiated capture feature names mismatch")
        for name, (shape, dtype) in expected.items():
            if (
                tuple(actual[name].get("shape", ())) != shape
                or actual[name].get("dtype") != dtype
            ):
                raise RuntimeError(
                    "negotiated capture feature shape/dtype mismatch: " + name
                )

    def unreturned_prepared(self, prepared, refs):
        """Retain only per-task failures for exact-attempt retirement."""
        returned = {ref.sample_id for ref in refs}
        pairs = [
            (task, payload)
            for task, payload in zip(prepared["tasks"], prepared["payloads"])
            if payload["sample_id"] not in returned
        ]
        if not pairs:
            return None
        tasks, payloads = map(list, zip(*pairs))
        return {
            **prepared,
            "tasks": tasks,
            "payloads": payloads,
            "estimated_bytes": self._output_bytes(tasks, prepared["capability"]),
        }

    def discard_prepared_batches(self, prepared):
        # Resolve uncertain begin replies using the store's immutable original
        # intent, then retire that exact writer before releasing byte credits.
        for item in prepared:
            for payload in item["payloads"]:
                if "attempt_id" not in payload:
                    continue  # This lease never registered a server writer.
                self.store.discard_capture_attempt(
                    payload["sample_id"],
                    server_epoch=payload["server_epoch"],
                    attempt_id=payload["attempt_id"],
                    control_url=item["begin_control_url"],
                    reservation_token=payload.get("reservation_token"),
                )

    def produce_prepared(self, prepared, *, capture: CaptureConfig):
        tasks = prepared["tasks"]
        body = prepared["body"]
        capture_payloads = prepared["payloads"]
        capability = prepared["capability"]
        server_capture = self._resolve_server_capture()
        sampling_params = {"temperature": 0.0, "max_new_tokens": 0}
        body["sampling_params"] = [
            {
                **sampling_params,
                "custom_params": {
                    "spec_capture": json.dumps(payload, separators=(",", ":"))
                },
            }
            for payload in capture_payloads
        ]
        for payload in capture_payloads:
            feature_names = list(payload["features"].values())
            feature_names.extend(
                item["name"] for item in payload.get("passthrough", ())
            )
            self.store.track_external_attempt(
                payload["sample_id"],
                generation=int(payload["gen"]),
                feature_names=feature_names,
                server_url=self.base_url,
            )
        try:
            rows = self.post_fn(
                f"{self.base_url}/generate", json_body=body, timeout=self.timeout_s
            )
        except Exception as exc:
            if (
                self._capture_protocol is not None
                and getattr(getattr(exc, "response", None), "status_code", None) == 409
            ):
                self._capture_protocol.capability(refresh=True)
            raise
        rows = _flatten_list_wrappers(rows)
        if len(rows) != len(tasks):
            raise RuntimeError(
                f"spec-capture server returned {len(rows)} rows for "
                f"{len(tasks)} tasks"
            )
        capture_hint = _CAPTURE_HINTS[server_capture]
        out: List[Union[Any, ServerCaptureFailure]] = []
        successful_refs = []
        for task, row, payload in zip(tasks, rows, capture_payloads):
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
            if not result:
                out.append(
                    ServerCaptureFailure(
                        task_id=task.task_id,
                        reason=(
                            "server_capture: response carries no spec_capture "
                            f"result — is the server running {capture_hint}?"
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
            expected_identity = {
                "sample_id": self._sample_id(task),
                "store_id": str(self.store.store_id),
                "gen": 1,
            }
            actual_identity = {
                "sample_id": str(result.get("sample_id")),
                "store_id": str(result.get("store_id")),
                "gen": int(result.get("gen", -1)),
            }
            if actual_identity != expected_identity:
                raise RuntimeError(
                    "spec-capture server returned the wrong object identity for "
                    f"task {task.task_id}: {actual_identity} != "
                    f"{expected_identity}"
                )
            if capability is not None:
                for name in (
                    "protocol_version",
                    "server_epoch",
                    "capture_namespace",
                    "capture_signature",
                    "attempt_id",
                    "reservation_token",
                ):
                    if result.get(name) != payload.get(name):
                        if name == "server_epoch":
                            self._capture_protocol.invalidate_epoch(result.get(name))
                        raise RuntimeError(
                            "negotiated capture response identity mismatch: " + name
                        )
                if self.hs_capture:
                    if result.get("request_digest") != payload.get("request_digest"):
                        raise RuntimeError(
                            "negotiated capture response request digest mismatch"
                        )
                    self._verify_hs_feature_metadata(result, payload, capability)
                if capability.get("published_ownership") == "attempt-v1":
                    if (
                        result.get("publication_ownership") != "attempt-v1"
                        or result.get("state") != "PUBLISHED"
                        or not result.get("writer_drained")
                    ):
                        raise RuntimeError(
                            "capture response omitted drained ownership transfer"
                        )
                    names = [
                        *payload["features"].values(),
                        *(x["name"] for x in payload.get("passthrough", ())),
                    ]
                    keys = attempt_keys(result, names)
                    if result.get("physical_keys") != keys or any(
                        result.get("features", {}).get(name, {}).get("key") != key
                        for name, key in keys.items()
                    ):
                        raise RuntimeError("capture response immutable keys mismatch")
                if self.hs_capture:
                    N, L = len(task.payload["input_ids"]), result.get("L")
                    if (
                        isinstance(L, bool)
                        or not isinstance(L, int)
                        or not 0 <= L < N
                        or result.get("N") != N
                        or result.get("span") != [L, N]
                    ):
                        raise RuntimeError(
                            "negotiated capture returned invalid absolute span"
                        )
            ref = self._ref_from_result(task, result, capture)
            if capability is not None and ref.estimated_bytes > self._output_bytes(
                [task], capability
            ):
                raise RuntimeError(
                    "capture output exceeded negotiated byte reservation"
                )
            # A capture shorter than the prompt is corrupt: it means cache
            # isolation or another full-prefill invariant failed despite the
            # request's fresh namespace.
            expected_len = len(task.payload["input_ids"])
            short = {
                name: spec.shape
                for name, spec in ref.feature_specs.items()
                if len(spec.shape) >= 2 and spec.shape[1] != expected_len
            }
            if short:
                self.store.adopt(ref)
                abort_feature_ref(self.store, ref, reason="seq-len-mismatch")
                out.append(
                    ServerCaptureFailure(
                        task_id=task.task_id,
                        reason=(
                            f"server_capture: captured seq len != prompt len "
                            f"{expected_len} for {short}; the capture request "
                            "did not execute a complete prefill"
                        ),
                        retryable=False,
                    )
                )
                continue
            try:
                recorded_aux_layer_ids = result.get("aux_layer_ids")
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
                    recorded_aux_layer_ids=(
                        tuple(recorded_aux_layer_ids)
                        if recorded_aux_layer_ids is not None
                        else None
                    ),
                    aux_feature_name=self.schema.aux_feature or "hidden_state",
                    target_feature_name=self.schema.last_hidden_feature or "target",
                )
            except CaptureMismatchError as exc:
                # Loud boundary failure; free the server-written keys so a
                # mismatched sample is never consumable.
                self.store.adopt(ref)
                abort_feature_ref(self.store, ref, reason=f"contract:{exc}")
                out.append(
                    ServerCaptureFailure(
                        task_id=task.task_id, reason=str(exc), retryable=False
                    )
                )
                continue
            # Do not transfer a successful row out of provisional ownership
            # until every row in the response has passed structural and
            # identity validation.  A later malformed row raises for the whole
            # lease batch; adopting this prefix now would make it invisible to
            # terminal provisional cleanup even though no refs are returned.
            successful_refs.append(ref)
            out.append(ref)
        for index, ref in enumerate(successful_refs):
            try:
                self.store.adopt(ref)
            except Exception as exc:
                # No ref has left this call. An adopt may fail after installing
                # ownership, so include the current row in the exact-ref rollback.
                cleanup_errors = []
                for unpublished in successful_refs[: index + 1]:
                    try:
                        abort_feature_ref(
                            self.store, unpublished, reason="capture-adoption-failed"
                        )
                    except Exception as cleanup_exc:
                        cleanup_errors.append(f"{unpublished.sample_id}: {cleanup_exc}")
                if cleanup_errors:
                    raise RuntimeError(
                        f"capture adoption rollback failed: {cleanup_errors}"
                    ) from exc
                raise
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
    "SERVER_CAPTURE_BACKENDS",
    "ServerCaptureSchema",
    "ServerCaptureFailure",
    "SGLangServerCaptureAdapter",
    "detect_server_capture",
]
