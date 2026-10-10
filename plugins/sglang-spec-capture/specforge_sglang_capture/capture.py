# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Server-side spec capture on SGLang's forward-observer extension points.

A capture request carries its spec as a JSON string in
``sampling_params.custom_params["spec_capture"]``::

    {"store_id", "sample_id", "gen", "replace",
     "features": {"aux": <name>, "last_hidden": <name>},
     "passthrough": [{"name", "data", "shape", "dtype"}]}

Per scheduler process, on the attention-TP writer rank:

1. ``CaptureObserver`` (an SGLang ``ForwardObserver``) picks the prefill rows of
   capture requests out of the forward's hidden states: the aux-layer
   concatenation in ``hidden_states`` and the post-norm ``last_hidden_states``.
2. ``CaptureDeviceOutput.copy_to_host`` copies them to pinned host memory with
   the generation result, or keeps device tensors for RDMA device publication.
3. ``CaptureHostOutput.consume`` hands each finished request's rows to the
   Mooncake sink and holds the request's response (``Req.defer_output``).
4. ``PendingCaptures`` (an SGLang ``DeferredOutputSource``) releases requests in
   order once their objects are written; the streamer subclass then returns
   the result in ``meta_info["spec_capture"]``.

The module imports SGLang lazily so its logic is testable without a server.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from collections import deque
from concurrent.futures import Future
from concurrent.futures import wait as wait_futures
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import torch
from specforge_sglang_capture.sink import (
    ARTIFACT_AUX,
    ARTIFACT_LAST_HIDDEN,
    Sample,
    SpecCaptureSink,
    gpu_put_enabled,
)

logger = logging.getLogger(__name__)

#: ``custom_params`` key of the request spec and ``meta_info`` key of the result.
SPEC_CAPTURE_KEY = "spec_capture"
#: Bound on scheduler batches whose rows the sink still holds.
MAX_PENDING_BATCHES_ENV = "SGLANG_SPEC_CAPTURE_MAX_PENDING_BATCHES"

_UNPARSED = object()


@dataclasses.dataclass(frozen=True)
class CaptureItem:
    """One capture request's rows in a forward's hidden states."""

    rid: str
    spec: Dict[str, Any]
    start: int
    length: int
    error: Optional[str] = None


def request_spec(req) -> Optional[Dict[str, Any]]:
    """Return the capture spec a request carries (parsed once), or ``None``."""
    spec = getattr(req, "_spec_capture_spec", _UNPARSED)
    if spec is not _UNPARSED:
        return spec
    params = getattr(req.sampling_params, "custom_params", None)
    raw = params.get(SPEC_CAPTURE_KEY) if isinstance(params, dict) else None
    if raw is None:
        spec = None
    else:
        try:
            spec = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(spec, dict):
                raise ValueError(f"expected a JSON object, got {type(spec).__name__}")
        except ValueError as exc:
            spec = {"_parse_error": f"invalid spec_capture: {exc}"}
    req._spec_capture_spec = spec
    return spec


def error_result(spec: Dict[str, Any], error: str) -> Dict[str, Any]:
    return {"sample_id": str(spec.get("sample_id")), "error": error}


def collect_capture_items(
    reqs: Sequence[Any],
    extend_lens: Sequence[int],
    prefix_lens: Sequence[int],
) -> Tuple[List[CaptureItem], int]:
    """Locate capture requests' rows; return the items and the total row count.

    A capture must cover the whole prompt in this forward: a cached prefix or a
    chunked prefill would leave rows out.
    """
    items: List[CaptureItem] = []
    offset = 0
    for req, length, prefix_len in zip(reqs, extend_lens, prefix_lens, strict=True):
        spec = request_spec(req)
        if spec is not None:
            error = spec.get("_parse_error")
            prompt_len = len(req.origin_input_ids)
            if error is None and prefix_len:
                error = (
                    f"{prefix_len} prompt tokens were served from the prefix "
                    "cache; spec capture needs a full prefill"
                )
            elif error is None and length != prompt_len:
                error = (
                    f"the prefill covered {length} of {prompt_len} prompt tokens; "
                    "launch the server with --chunked-prefill-size -1"
                )
            items.append(CaptureItem(req.rid, spec, offset, length, error))
        offset += length
    return items, offset


def _missing_artifacts(spec: Dict[str, Any], available: Dict[str, bool]) -> List[str]:
    features = spec.get("features") or {}
    return [artifact for artifact in features if not available.get(artifact, False)]


class CaptureObserver:
    """ForwardObserver selecting capture requests' rows of each prefill."""

    def __init__(self, runtime: CaptureRuntime) -> None:
        self.runtime = runtime

    def after_forward(self, batch, forward_batch, logits_output, *, can_run_graph):
        if not self.runtime.is_writer or not forward_batch.forward_mode.is_extend():
            return None
        items, num_rows = collect_capture_items(
            batch.reqs,
            forward_batch.extend_seq_lens_cpu,
            forward_batch.extend_prefix_lens_cpu,
        )
        if not items:
            return None

        aux = logits_output.hidden_states
        last_hidden = logits_output.last_hidden_states
        batch_error = None
        if logits_output.hidden_states_token_indices is not None:
            batch_error = "the forward kept a subset of hidden-state rows"
        else:
            for tensor in (aux, last_hidden):
                if tensor is not None and tensor.shape[0] != num_rows:
                    batch_error = (
                        f"the forward captured {tensor.shape[0]} hidden-state "
                        f"rows for {num_rows} tokens"
                    )
        available = {
            ARTIFACT_AUX: aux is not None,
            ARTIFACT_LAST_HIDDEN: last_hidden is not None,
        }
        checked = []
        for item in items:
            if item.error is None:
                if batch_error is not None:
                    item = dataclasses.replace(item, error=batch_error)
                elif missing := _missing_artifacts(item.spec, available):
                    item = dataclasses.replace(
                        item,
                        error=(
                            f"the server did not capture {missing}; launch with "
                            "--aux-hidden-state-capture and "
                            "--return-hidden-states-mode full"
                        ),
                    )
            checked.append(item)
        return CaptureDeviceOutput.from_rows(
            self.runtime, checked, aux, last_hidden, copy=can_run_graph
        )


def _select_rows(
    items: List[CaptureItem], tensors: Sequence[Optional[torch.Tensor]], *, copy: bool
) -> Tuple[List[CaptureItem], List[Optional[torch.Tensor]]]:
    """Keep only the rows of capturable items, re-based onto the kept rows.

    ``copy`` forces new storage, since graph outputs are overwritten by later
    replays; gathering already allocates it.
    """
    spans = [(item.start, item.length) for item in items if item.error is None]
    present = [t for t in tensors if t is not None]
    if not spans or not present:
        return items, [None for _ in tensors]
    if _covers_all_rows(spans, present[0].shape[0]):
        return items, [t.clone() if copy and t is not None else t for t in tensors]
    index = torch.cat(
        [torch.arange(start, start + n, device=present[0].device) for start, n in spans]
    )
    kept = [t.index_select(0, index) if t is not None else None for t in tensors]
    rebased, cursor = [], 0
    for item in items:
        if item.error is None:
            rebased.append(dataclasses.replace(item, start=cursor))
            cursor += item.length
        else:
            rebased.append(item)
    return rebased, kept


def _covers_all_rows(spans: List[Tuple[int, int]], num_rows: int) -> bool:
    expected = 0
    for start, n in spans:
        if start != expected:
            return False
        expected = start + n
    return expected == num_rows


class CaptureDeviceOutput:
    """DeviceAuxiliaryOutput holding a forward's capture rows on the device."""

    def __init__(self, runtime, items, aux, last_hidden):
        self.runtime = runtime
        self.items = items
        self.aux = aux
        self.last_hidden = last_hidden

    @classmethod
    def from_rows(cls, runtime, items, aux, last_hidden, *, copy):
        items, (aux, last_hidden) = _select_rows(items, (aux, last_hidden), copy=copy)
        if last_hidden is not None and runtime.last_hidden_scale:
            # muP targets feed the logits processor LM-head-scaled states;
            # SpecForge folds that multiplier into its frozen target head, so
            # restore the pre-head-scale representation the trainer expects.
            last_hidden = last_hidden * runtime.last_hidden_scale
        return cls(runtime, items, aux, last_hidden)

    def copy_to_host(self, copy_tensor):
        if self.runtime.gpu_put:
            # Device publication: the sink waits for everything enqueued so far.
            ready = torch.cuda.Event()
            ready.record()
            return CaptureHostOutput(
                self.runtime, self.items, self.aux, self.last_hidden, ready_event=ready
            )
        return CaptureHostOutput(
            self.runtime,
            self.items,
            copy_tensor(self.aux) if self.aux is not None else None,
            copy_tensor(self.last_hidden) if self.last_hidden is not None else None,
        )


class CaptureHostOutput:
    """HostAuxiliaryOutput: publish finished capture requests, then hold them."""

    def __init__(self, runtime, items, aux, last_hidden, *, ready_event=None):
        self.runtime = runtime
        self.items = items
        self.aux = aux
        self.last_hidden = last_hidden
        self.ready_event = ready_event

    def _rows(self, tensor, item):
        if tensor is None:
            return None
        return tensor[item.start : item.start + item.length]

    def consume(self, batch, commits) -> None:
        reqs = {req.rid: req for req in batch.reqs}
        samples: List[Sample] = []
        held = []
        for item in self.items:
            req = reqs.get(item.rid)
            if req is None:
                continue
            error = item.error
            if error is None and not req.finished():
                error = (
                    "spec capture requests must finish at prefill (max_new_tokens=0)"
                )
            if error is not None:
                # Not held: the error result streams with this batch.
                self.runtime.results[req.rid] = error_result(item.spec, error)
                continue
            samples.append(
                (
                    item.spec,
                    self._rows(self.aux, item),
                    self._rows(self.last_hidden, item),
                )
            )
            held.append(req)
        if not samples:
            return
        for req in held:
            req.defer_output = True
        self.runtime.pending.submit(held, samples, ready_event=self.ready_event)


class PendingCaptures:
    """DeferredOutputSource: release held requests once their batch is written."""

    def __init__(self, runtime: CaptureRuntime, max_pending_batches: int) -> None:
        if max_pending_batches < 1:
            raise ValueError(f"{MAX_PENDING_BATCHES_ENV} must be >= 1")
        self.runtime = runtime
        self.max_pending_batches = max_pending_batches
        self._batches: Deque[Tuple[List[Any], List[Dict[str, Any]], Future]] = deque()

    def submit(self, reqs, samples: List[Sample], *, ready_event=None) -> None:
        future = self.runtime.sink.submit_samples(samples, ready_event=ready_event)
        self._batches.append((reqs, [spec for spec, _, _ in samples], future))
        if len(self._batches) > self.max_pending_batches:
            # Bound the retained rows: the next poll releases the oldest batch.
            wait_futures([self._batches[0][2]])

    def poll(self) -> List[Any]:
        released = []
        while self._batches and self._batches[0][2].done():
            reqs, specs, future = self._batches.popleft()
            try:
                results = future.result()
                if len(results) != len(reqs):
                    raise RuntimeError(
                        f"spec-capture sink returned {len(results)} results for "
                        f"{len(reqs)} requests"
                    )
            except Exception as exc:
                logger.error(
                    "spec-capture batch failed for %d requests: %s", len(reqs), exc
                )
                results = [error_result(spec, str(exc)) for spec in specs]
            for req, result in zip(reqs, results):
                self.runtime.results[req.rid] = result
            released.extend(reqs)
        return released

    def has_pending(self) -> bool:
        return bool(self._batches)


def capture_streamer_class(base: type) -> type:
    """Subclass ``base`` (a SchedulerOutputStreamer) to return capture results."""
    base_builds = getattr(base, "has_additional_customized_info", False)

    class SpecCaptureOutputStreamer(base):
        has_additional_customized_info = True
        # The runtime's per-request results, set once the runtime is installed.
        spec_capture_results: Optional[Dict[str, Dict[str, Any]]] = None

        def should_build_additional_customized_info(self) -> bool:
            if self.spec_capture_results:
                return True
            return base_builds and super().should_build_additional_customized_info()

        def build_additional_customized_info(self, reqs) -> Dict[str, list]:
            info = super().build_additional_customized_info(reqs) if base_builds else {}
            results = self.spec_capture_results
            if results:
                info[SPEC_CAPTURE_KEY] = [
                    [results.pop(req.rid)] if req.rid in results else [] for req in reqs
                ]
            return info

    SpecCaptureOutputStreamer.__name__ = f"SpecCapture{base.__name__}"
    SpecCaptureOutputStreamer.__qualname__ = SpecCaptureOutputStreamer.__name__
    return SpecCaptureOutputStreamer


class CaptureRuntime:
    """Per-scheduler capture state shared by the observer, outputs and streamer."""

    def __init__(
        self,
        *,
        sink: SpecCaptureSink,
        is_writer: bool,
        gpu_put: bool,
        max_pending_batches: int,
        last_hidden_scale: Optional[float] = None,
    ) -> None:
        self.sink = sink
        self.is_writer = is_writer
        self.gpu_put = gpu_put
        # The target's logits_mup_width_multiplier, if it declares one.
        self.last_hidden_scale = last_hidden_scale
        # rid -> result dict, read by the streamer when the request streams.
        self.results: Dict[str, Dict[str, Any]] = {}
        self.pending = PendingCaptures(self, max_pending_batches)
        self.observer = CaptureObserver(self)


def logits_mup_width_multiplier(model_runner) -> Optional[float]:
    """The target's ``logits_mup_width_multiplier``, or ``None``."""
    model_config = getattr(model_runner, "model_config", None)
    for name in ("hf_text_config", "hf_config"):
        config = getattr(model_config, name, None)
        value = getattr(config, "logits_mup_width_multiplier", None)
        if value:
            return float(value)
    return None


def install(scheduler) -> CaptureRuntime:
    """Validate the server configuration and wire capture into ``scheduler``."""
    from sglang.srt.runtime_context import get_exec, get_parallel, get_schedule

    model_runner = scheduler.tp_worker.model_runner
    features = get_exec().features
    if not (
        hasattr(type(model_runner), "forward_observer")
        and hasattr(scheduler, "register_deferred_output_source")
        and hasattr(features, "aux_hidden_state_capture")
    ):
        raise RuntimeError(
            "SPECFORGE_SPEC_CAPTURE needs an SGLang build with forward observers, "
            "deferred outputs and --aux-hidden-state-capture"
        )
    if features.aux_hidden_state_capture is None:
        raise RuntimeError("SPECFORGE_SPEC_CAPTURE needs --aux-hidden-state-capture")
    if features.return_hidden_states_mode != "full":
        raise RuntimeError(
            "SPECFORGE_SPEC_CAPTURE needs --return-hidden-states-mode full so every "
            "prefill captures all hidden-state rows"
        )
    if get_schedule().chunked_prefill_size != -1:
        raise RuntimeError(
            "SPECFORGE_SPEC_CAPTURE needs --chunked-prefill-size -1 so each prompt "
            "is captured by one forward"
        )

    runtime = CaptureRuntime(
        sink=SpecCaptureSink(features.aux_hidden_state_layer_ids),
        is_writer=get_parallel().attn_tp_rank == 0,
        gpu_put=gpu_put_enabled(),
        max_pending_batches=int(os.environ.get(MAX_PENDING_BATCHES_ENV, "2")),
        last_hidden_scale=logits_mup_width_multiplier(model_runner),
    )
    model_runner.forward_observer = runtime.observer
    scheduler.register_deferred_output_source(runtime.pending)
    scheduler.output_streamer.spec_capture_results = runtime.results
    logger.info(
        "SpecForge spec capture enabled (%s publication, writer=%s)",
        "device" if runtime.gpu_put else "host",
        runtime.is_writer,
    )
    return runtime
