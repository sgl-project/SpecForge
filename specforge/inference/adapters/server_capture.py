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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from specforge.inference.capture import (
    CaptureConfig,
    CaptureMismatchError,
    verify_capture_specs,
)
from specforge.runtime.contracts import SCHEMA_VERSION, FeatureSpec, PromptTask

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


def check_capture_outputs(
    server_info: Mapping[str, Any], layers: Sequence[int], *, server_url: str
) -> None:
    """Require a server that can cut ``layers`` into several outputs a request.

    An older plugin ignores a spec's ``outputs`` and would write nothing.
    """
    states = [
        state
        for state in server_info.get("internal_states") or []
        if isinstance(state, Mapping) and state.get("spec_capture_backend") == "plugin"
    ]
    if not states or not all(state.get("spec_capture_outputs") for state in states):
        raise RuntimeError(
            f"{server_url} cannot write several capture outputs per request; "
            "update the specforge-sglang-capture plugin in its environment"
        )
    needed = sorted(set(layers))
    for state in states:
        captured = list(state.get("spec_capture_aux_layer_ids") or ())
        if captured != sorted(captured) or set(needed) - set(captured):
            raise RuntimeError(
                f"{server_url} captures layers {captured}, but the drafters need "
                f"{needed} in ascending order; launch it with "
                "--return-aux-hidden-state-layer-ids "
                + " ".join(str(layer) for layer in needed)
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
class CaptureOutput:
    """One drafter's sample of each captured prompt.

    Its refs carry ``run_id`` and ``algorithm``, so several drafters can share
    one request. Once ``wanted()`` is false, later requests leave the output
    out: its drafter has finished.
    """

    run_id: str
    algorithm: str
    schema: ServerCaptureSchema
    wanted: Callable[[], bool] = lambda: True

    def __post_init__(self) -> None:
        if not isinstance(self.schema, ServerCaptureSchema):
            raise TypeError("schema must be an injected ServerCaptureSchema")
        if not self.algorithm:
            raise ValueError("algorithm must be non-empty")


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

    ``algorithm`` and ``schema`` describe a single drafter. Several drafters
    pass ``outputs`` instead: one request then asks the server for one sample
    per drafter, and ``run_id`` only names the request.
    """

    def __init__(
        self,
        base_url: str,
        store,
        *,
        run_id: str,
        algorithm: Optional[str] = None,
        schema: Optional[ServerCaptureSchema] = None,
        outputs: Optional[Sequence[CaptureOutput]] = None,
        request_input_adapter=None,
        timeout_s: float = 300.0,
        post_fn: Optional[Callable[..., Any]] = None,
        target_model_version: str = "unknown",
        server_capture: str = "plugin",
        server_info_fn: Optional[Callable[[str, float], Mapping[str, Any]]] = None,
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
        if outputs is None:
            outputs = [CaptureOutput(run_id=run_id, algorithm=algorithm, schema=schema)]
        elif algorithm is not None or schema is not None:
            raise ValueError("pass either outputs or algorithm and schema")
        if not outputs:
            raise ValueError("outputs must not be empty")
        self.outputs = tuple(outputs)
        self.schema = self.outputs[0].schema
        self.strategy = "+".join(output.algorithm for output in self.outputs)
        if request_input_adapter is not None and not callable(
            getattr(request_input_adapter, "build_request_inputs", None)
        ):
            raise TypeError(
                "request_input_adapter must expose build_request_inputs(tasks)"
            )
        self.request_input_adapter = request_input_adapter
        self.timeout_s = timeout_s
        self.post_fn = post_fn or _SessionPost()
        self.target_model_version = target_model_version
        #: ``auto`` until the first request reads the server's backend.
        self.server_capture = server_capture
        self.server_info_fn = server_info_fn or _get_server_info
        self._server_capture_lock = threading.Lock()
        self._outputs_checked = len(self.outputs) == 1
        self._healthy = True

    def _resolve_server_capture(self, layers: Sequence[int]) -> str:
        """The server's capture backend, read from ``/server_info`` if needed.

        The first request for several outputs also checks there that the
        server can cut ``layers``, every drafter's capture layers, for them.
        """
        with self._server_capture_lock:
            if self.server_capture == "auto" or not self._outputs_checked:
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
                if self.server_capture == "auto":
                    self.server_capture = detect_server_capture(
                        info, server_url=self.base_url
                    )
                    logger.info(
                        "%s runs spec capture via the %s",
                        self.base_url,
                        self.server_capture,
                    )
                if not self._outputs_checked:
                    check_capture_outputs(info, layers, server_url=self.base_url)
                    self._outputs_checked = True
            return self.server_capture

    # -- request construction -------------------------------------------------
    def _sample_id(self, task: PromptTask, run_id: Optional[str] = None) -> str:
        return f"{run_id or self.run_id}:{task.task_id}"

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

    def _output_payload(
        self, task: PromptTask, schema: ServerCaptureSchema
    ) -> Dict[str, Any]:
        """The server artifacts and client tensors one sample of *task* holds."""
        input_ids = list(task.payload["input_ids"])
        length = len(input_ids)
        features: Dict[str, str] = {}
        if schema.aux_feature is not None:
            features["aux"] = schema.aux_feature
        if schema.last_hidden_feature is not None:
            features["last_hidden"] = schema.last_hidden_feature
        passthrough: List[Dict[str, Any]] = []
        for feature_name, payload_key, trailing in schema.passthrough:
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
        if schema.attention_mask_feature is not None:
            passthrough.append(
                {
                    "name": schema.attention_mask_feature,
                    "data": [1] * length,
                    "shape": [1, length],
                    "dtype": "int64",
                }
            )
        return {"features": features, "passthrough": passthrough}

    def _spec_capture_payload(
        self,
        task: PromptTask,
        wanted: Sequence[Tuple[CaptureOutput, CaptureConfig]],
    ) -> Dict[str, Any]:
        """The request spec for *task*: one sample per wanted output.

        A single-output adapter sends the flat spec. Several outputs go in
        ``outputs``, each with the layers the server cuts for it.
        """

        def spec(sample_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "store_id": self.store.store_id,
                "sample_id": sample_id,
                # A task id is unique within a run, and retries happen only
                # before its ref is committed.  Keep the generation stable so
                # a response lost after the server write cannot strand gN when
                # the retry writes gN+1.  The server plugin replaces these
                # deterministic keys on a retry (``replace``), bounding the
                # attempt to one namespace.
                "gen": 1,
                "replace": int(task.attempt) > 0,
                **body,
            }

        if len(self.outputs) == 1:
            ((output, _),) = wanted
            return spec(
                self._sample_id(task, output.run_id),
                self._output_payload(task, output.schema),
            )
        return spec(
            self._sample_id(task),
            {
                "outputs": [
                    {
                        "sample_id": self._sample_id(task, output.run_id),
                        **self._output_payload(task, output.schema),
                        "layer_ids": list(capture.aux_hidden_state_layer_ids),
                    }
                    for output, capture in wanted
                ]
            },
        )

    # -- ref construction ------------------------------------------------------
    def _ref_from_result(
        self,
        task: PromptTask,
        result: Dict[str, Any],
        capture: CaptureConfig,
        output: CaptureOutput,
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
            if name == output.schema.last_hidden_feature:
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
        ref = SampleRef(
            sample_id=sample_id,
            run_id=output.run_id,
            source_task_id=task.task_id,
            feature_store_uri=f"mooncake://{result['store_id']}/{sample_id}",
            feature_keys={n: f"{sample_id}/{n}" for n in specs},
            feature_specs=specs,
            strategy=output.algorithm,
            schema_version=SCHEMA_VERSION,
            target_model_version=self.target_model_version,
            tokenizer_version=str(task.metadata.get("tokenizer_version", "unknown")),
            num_tokens=num_tokens,
            estimated_bytes=nbytes,
            metadata={
                "run_id": output.run_id,
                "source_task_id": task.task_id,
                "strategy": output.algorithm,
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

    def _ref_fault(
        self,
        task: PromptTask,
        ref,
        result: Dict[str, Any],
        capture: CaptureConfig,
        schema: ServerCaptureSchema,
    ) -> Optional[Tuple[str, str]]:
        """Why *ref* must not be used: its abort and failure reasons, or None."""
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
            return "seq-len-mismatch", (
                f"server_capture: captured seq len != prompt len "
                f"{expected_len} for {short}; the capture request "
                "did not execute a complete prefill"
            )
        try:
            recorded_aux_layer_ids = result.get("aux_layer_ids")
            if capture.aux_hidden_state_layer_ids and recorded_aux_layer_ids is None:
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
                aux_feature_name=schema.aux_feature or "hidden_state",
                target_feature_name=schema.last_hidden_feature or "target",
            )
        except CaptureMismatchError as exc:
            return f"contract:{exc}", str(exc)
        return None

    # -- the RefSource entry point ----------------------------------------------
    def produce_refs(
        self,
        tasks: List[PromptTask],
        *,
        capture: Union[CaptureConfig, Sequence[CaptureConfig]],
    ) -> List[Union["SampleRef", tuple, ServerCaptureFailure]]:  # noqa: F821
        """One batched server call -> the committed-ready ref(s) of each task.

        Order-aligned with ``tasks``. A single-output adapter returns one ref
        per task. With several outputs, ``capture`` holds one config per
        output and a task yields a tuple with one ref per wanted output: all
        of them or, when any output is bad, none.
        A per-task server error becomes a :class:`ServerCaptureFailure` (the
        worker fails just that lease). Any transport-level error raises — the
        worker fails the whole lease batch retryable, mirroring
        ``generate_features`` semantics.
        """
        captures = (capture,) if isinstance(capture, CaptureConfig) else tuple(capture)
        if len(captures) != len(self.outputs):
            raise ValueError(
                f"got {len(captures)} capture configs for {len(self.outputs)} outputs"
            )
        server_capture = self._resolve_server_capture(
            [layer for item in captures for layer in item.aux_hidden_state_layer_ids]
        )
        wanted = [pair for pair in zip(self.outputs, captures) if pair[0].wanted()]
        if not wanted:
            raise RuntimeError("no drafter wants capture output any more")
        body = self._request_inputs(tasks)
        # A fresh cache namespace forces a full prefill on every attempt while
        # leaving SGLang free to use the radix data structure required by hybrid
        # targets. Retries need a new key because the prior attempt may have
        # populated its namespace before the response was lost.
        body["extra_key"] = [uuid.uuid4().hex for _ in tasks]
        sampling_params = {"temperature": 0.0, "max_new_tokens": 0}
        capture_payloads = [self._spec_capture_payload(t, wanted) for t in tasks]
        # custom_params values are JSON scalars: one encoded spec per request.
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
            for sample in payload.get("outputs", (payload,)):
                feature_names = list(sample["features"].values())
                feature_names.extend(
                    item["name"] for item in sample.get("passthrough", ())
                )
                self.store.track_external_attempt(
                    sample["sample_id"],
                    generation=int(payload["gen"]),
                    feature_names=feature_names,
                    server_url=self.base_url,
                )
        rows = self.post_fn(
            f"{self.base_url}/generate", json_body=body, timeout=self.timeout_s
        )
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
                expected_sample_id=payload["sample_id"],
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
            # The result mirrors the request: one sample per requested output.
            requested = payload.get("outputs", (payload,))
            written = result.get("outputs") if "outputs" in payload else (result,)
            if not isinstance(written, (list, tuple)) or len(written) != len(requested):
                raise RuntimeError(
                    "spec-capture server did not return the "
                    f"{len(requested)} requested outputs for task {task.task_id}"
                )
            refs = []
            for sample, sample_result, (output, output_capture) in zip(
                requested, written, wanted
            ):
                expected_identity = {
                    "sample_id": sample["sample_id"],
                    "store_id": str(self.store.store_id),
                    "gen": 1,
                }
                actual_identity = {
                    "sample_id": str(sample_result.get("sample_id")),
                    "store_id": str(sample_result.get("store_id")),
                    "gen": int(sample_result.get("gen", -1)),
                }
                if actual_identity != expected_identity:
                    raise RuntimeError(
                        "spec-capture server returned the wrong object identity "
                        f"for task {task.task_id}: {actual_identity} != "
                        f"{expected_identity}"
                    )
                refs.append(
                    self._ref_from_result(task, sample_result, output_capture, output)
                )
            fault = None
            for ref, sample_result, (output, output_capture) in zip(
                refs, written, wanted
            ):
                fault = self._ref_fault(
                    task, ref, sample_result, output_capture, output.schema
                )
                if fault is not None:
                    break
            if fault is not None:
                # Loud boundary failure; free the server-written keys of every
                # output so no drafter can consume part of a bad capture.
                abort_reason, reason = fault
                for ref in refs:
                    self.store.adopt(ref)
                    self.store.abort(ref.sample_id, reason=abort_reason)
                out.append(
                    ServerCaptureFailure(
                        task_id=task.task_id, reason=reason, retryable=False
                    )
                )
                continue
            # Do not transfer a successful row out of provisional ownership
            # until every row in the response has passed structural and
            # identity validation.  A later malformed row raises for the whole
            # lease batch; adopting this prefix now would make it invisible to
            # terminal provisional cleanup even though no refs are returned.
            successful_refs.extend(refs)
            out.append(refs[0] if len(self.outputs) == 1 else tuple(refs))
        for ref in successful_refs:
            self.store.adopt(ref)
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
    "CaptureOutput",
    "ServerCaptureSchema",
    "ServerCaptureFailure",
    "SGLangServerCaptureAdapter",
    "detect_server_capture",
]
