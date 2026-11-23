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
   concatenation in ``hidden_states`` and the unscaled post-norm
   ``last_hidden_states``.
2. ``CaptureDeviceOutput.copy_to_host`` copies them to pinned host memory with
   the generation result, or keeps device tensors for device publication (RDMA,
   or the NVLink arena under ``MOONCAKE_PROTOCOL=nvlink``).
3. ``CaptureHostOutput.consume`` publishes each finished request's rows and
   passes its completion to ``Scheduler.defer_response``.
4. SGLang holds the request until ``CaptureCompletion`` is ready, then reads
   its result on the scheduler thread before streaming. The streamer subclass
   returns the result in ``meta_info["spec_capture"]``.

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
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import torch
from specforge_sglang_capture.prefix_cache import HiddenPrefixCache
from specforge_sglang_capture.sink import (
    ARTIFACT_AUX,
    ARTIFACT_LAST_HIDDEN,
    Sample,
    SpecCaptureSink,
    arena_bytes,
    gpu_put_enabled,
    nvlink_enabled,
)

logger = logging.getLogger(__name__)

#: ``custom_params`` key of the request spec and ``meta_info`` key of the result.
SPEC_CAPTURE_KEY = "spec_capture"
#: Bound on scheduler batches whose rows the sink still holds.
MAX_PENDING_BATCHES_ENV = "SGLANG_SPEC_CAPTURE_MAX_PENDING_BATCHES"
PREFIX_CACHE_BYTES_ENV = "SGLANG_SPEC_CAPTURE_PREFIX_CACHE_BYTES"
PREFIX_CACHE_MISS = "capture prefix hidden states are unavailable"

_UNPARSED = object()


@dataclasses.dataclass(frozen=True)
class CaptureItem:
    """One capture request's rows in a forward's hidden states."""

    rid: str
    spec: Dict[str, Any]
    start: int
    length: int
    error: Optional[str] = None
    prefix_len: int = 0


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
    result = {"sample_id": str(spec.get("sample_id")), "error": error}
    if error == PREFIX_CACHE_MISS:
        result["error_code"] = "prefix_cache_miss"
    return result


def prefix_eligible(req, spec):
    return (
        spec.get("reuse_prefix") is True
        and req.extra_key
        and req.sampling_params.max_new_tokens == 0
        and req.input_embeds is None
        and req.multimodal_inputs is None
        and req.lora_id is None
        and getattr(req, "positional_embed_overrides", None) is None
        and getattr(req, "session", None) is None
    )


def collect_capture_items(
    reqs: Sequence[Any],
    extend_lens: Sequence[int],
    prefix_lens: Sequence[int],
) -> Tuple[List[CaptureItem], int]:
    """Locate capture requests' rows; return the items and the total row count.

    The prefill must cover the entire suffix. Only explicitly opted-in text
    requests can supply the missing prefix from the hidden-state cache.
    """
    items: List[CaptureItem] = []
    offset = 0
    for req, length, prefix_len in zip(reqs, extend_lens, prefix_lens, strict=True):
        spec = request_spec(req)
        if spec is not None:
            error = spec.get("_parse_error")
            prompt_len = len(req.origin_input_ids)
            if error is None and prefix_len and not prefix_eligible(req, spec):
                error = (
                    f"{prefix_len} prompt tokens were served from the prefix "
                    "cache; spec capture needs a full prefill"
                )
            elif error is None and length + prefix_len != prompt_len:
                error = (
                    f"the prefill covered {length} of {prompt_len} prompt tokens; "
                    "launch the server with --chunked-prefill-size -1"
                )
            items.append(CaptureItem(req.rid, spec, offset, length, error, prefix_len))
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
        cache = self.runtime.prefix_cache
        if not forward_batch.forward_mode.is_extend() or (
            not self.runtime.is_writer and cache is None
        ):
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
                            "--enable-return-aux-hidden-states and "
                            "--return-hidden-states-mode full"
                        ),
                    )
            checked.append(item)
        if cache is not None and aux is not None and last_hidden is not None:
            req_by_id = {req.rid: req for req in batch.reqs}
            for item in checked:
                req = req_by_id[item.rid]
                if item.error is None and prefix_eligible(req, item.spec):
                    span = slice(item.start, item.start + item.length)
                    cache.stage(
                        req,
                        item.prefix_len,
                        item.prefix_len + item.length,
                        (aux[span], last_hidden[span]),
                        copy=can_run_graph,
                    )
        if not self.runtime.is_writer:
            return None
        if any(item.prefix_len and item.error is None for item in checked):
            return self._restore_prefixes(
                batch.reqs, checked, aux, last_hidden, copy=can_run_graph
            )
        return CaptureDeviceOutput.from_rows(
            self.runtime, checked, aux, last_hidden, copy=can_run_graph
        )

    def _restore_prefixes(self, reqs, items, aux, last_hidden, *, copy):
        cache = self.runtime.prefix_cache
        req_by_id = {req.rid: req for req in reqs}
        can_restore = cache is not None and aux is not None and last_hidden is not None
        parts = []
        rebased = []
        hits = False
        cursor = 0
        for i, item in enumerate(items):
            if item.error is not None:
                rebased.append(item)
                continue
            if item.prefix_len:
                prefix = (
                    cache.read(req_by_id[item.rid], item.prefix_len)
                    if can_restore
                    else None
                )
                if prefix is None:
                    item = dataclasses.replace(item, error=PREFIX_CACHE_MISS)
                    items[i] = item
                    rebased.append(item)
                    continue
                parts.append(prefix)
                hits = True
            if can_restore:
                span = slice(item.start, item.start + item.length)
                parts.append((aux[span], last_hidden[span]))
            length = item.prefix_len + item.length
            rebased.append(dataclasses.replace(item, start=cursor, length=length))
            cursor += length
        if not hits:
            return CaptureDeviceOutput.from_rows(
                self.runtime, items, aux, last_hidden, copy=copy
            )

        # The concatenations allocate fresh storage, including for graph outputs.
        aux_parts, last_parts = zip(*parts, strict=True)
        return CaptureDeviceOutput(
            self.runtime, rebased, torch.cat(aux_parts), torch.cat(last_parts)
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
        future = self.runtime.pending.submit(samples, ready_event=self.ready_event)
        for index, (req, (spec, _, _)) in enumerate(zip(held, samples, strict=True)):
            self.runtime.defer_response(
                req,
                CaptureCompletion(
                    future,
                    index,
                    len(samples),
                    req.rid,
                    spec,
                    self.runtime.results,
                    self.runtime.sink.discard_sample,
                ),
            )


@dataclasses.dataclass
class CaptureCompletion:
    """One response's metadata, backed by a shared batch publication future.

    SGLang calls these methods on the scheduler thread. Cancelling one request
    leaves the shared write running for its siblings and suppresses its metadata.
    Cleanup is queued on the sink's writer, after publication, even if the
    client has already tried to remove keys that did not exist yet.
    """

    future: Future
    index: int
    batch_size: int
    rid: str
    spec: Dict[str, Any]
    results: Dict[str, Dict[str, Any]]
    discard_sample: Callable[[Dict[str, Any]], Future]
    _cancelled: bool = False
    _resolved: bool = False

    def done(self) -> bool:
        return self._cancelled or self.future.done()

    def result(self) -> None:
        if self._cancelled or self._resolved:
            return
        try:
            results = self.future.result()
            if len(results) != self.batch_size:
                raise RuntimeError(
                    f"spec-capture sink returned {len(results)} results for "
                    f"{self.batch_size} requests"
                )
            result = results[self.index]
        except Exception as exc:
            logger.error("spec-capture failed for %s: %s", self.rid, exc)
            result = error_result(self.spec, str(exc))
        self.results[self.rid] = result
        self._resolved = True

    def cancel(self) -> bool:
        if self._resolved:
            return False
        if self._cancelled:
            return True
        self.discard_sample(self.spec)
        self._cancelled = True
        return True


class PendingCaptures:
    """Bound in-flight publication batches without retaining SGLang requests."""

    def __init__(self, runtime: CaptureRuntime, max_pending_batches: int) -> None:
        if max_pending_batches < 1:
            raise ValueError(f"{MAX_PENDING_BATCHES_ENV} must be >= 1")
        self.runtime = runtime
        self.max_pending_batches = max_pending_batches
        self._futures: Deque[Future] = deque()

    def submit(self, samples: List[Sample], *, ready_event=None) -> Future:
        while self._futures and self._futures[0].done():
            self._futures.popleft()
        future = self.runtime.sink.submit_samples(samples, ready_event=ready_event)
        self._futures.append(future)
        if len(self._futures) > self.max_pending_batches:
            wait_futures([self._futures.popleft()])
        return future


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
        defer_response: Callable[[Any, CaptureCompletion], None],
        prefix_cache: HiddenPrefixCache | None = None,
        prefix_capacity_tokens: int = 0,
    ) -> None:
        self.sink = sink
        self.is_writer = is_writer
        self.gpu_put = gpu_put
        self.defer_response = defer_response
        self.prefix_cache = prefix_cache
        self.prefix_capacity_tokens = prefix_capacity_tokens
        # rid -> result dict, read by the streamer when the request streams.
        self.results: Dict[str, Dict[str, Any]] = {}
        self.pending = PendingCaptures(self, max_pending_batches)
        self.observer = CaptureObserver(self)


def _configure_prefix_cache(tree, model_runner, features, budget, is_writer):
    if not budget:
        return None, 0
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.runtime_context import get_parallel

    if (
        type(tree) is not UnifiedRadixCache
        or tree.disable
        or tree.cache_controller is not None
        or tree.linker is not None
        or tree.logical_page_size != tree.page_size
        or tree.supports_mamba()
        or tree.tree_core.is_eagle
        or tree.uocr_doc_reuse is not None
        or get_parallel().pp_size != 1
    ):
        logger.info("SpecForge prefix reuse is unavailable on this radix cache")
        return None, 0

    layers = features.return_aux_hidden_state_layer_ids
    if not layers:
        return None, 0
    bytes_per_token = (
        model_runner.model_config.hidden_size
        * (len(layers) + 1)
        * model_runner.dtype.itemsize
    )
    hidden = HiddenPrefixCache(
        budget,
        tree.page_size,
        is_writer=is_writer,
        max_tokens=model_runner.max_total_num_tokens,
    )
    pages = hidden.capacity(tree.page_size * bytes_per_token)
    if not pages:
        return None, 0

    tree._spec_capture_hidden = hidden
    return hidden, pages * tree.page_size


def install(scheduler) -> CaptureRuntime:
    """Validate the server configuration and wire capture into ``scheduler``."""
    from sglang.srt.disaggregation.utils import DisaggregationMode
    from sglang.srt.runtime_context import get_exec, get_parallel, get_schedule

    if scheduler.disaggregation_mode != DisaggregationMode.NULL:
        raise RuntimeError("SPECFORGE_SPEC_CAPTURE does not support PD disaggregation")
    model_runner = scheduler.tp_worker.model_runner
    features = get_exec().features
    if not (
        hasattr(type(model_runner), "forward_observer")
        and hasattr(scheduler, "defer_response")
        and hasattr(features, "aux_hidden_state_capture")
    ):
        raise RuntimeError(
            "SPECFORGE_SPEC_CAPTURE needs an SGLang build with forward observers, "
            "deferred outputs and --enable-return-aux-hidden-states"
        )
    # aux_hidden_state_capture is the resolved capture method: SGLang fills it
    # from --enable-return-aux-hidden-states and --speculative-algorithm.
    if features.aux_hidden_state_capture is None:
        raise RuntimeError(
            "SPECFORGE_SPEC_CAPTURE needs --enable-return-aux-hidden-states with "
            "--speculative-algorithm naming the capture convention"
        )
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
    nvlink = nvlink_enabled()
    if nvlink:
        # The sink allocates the arena on the first capture; reject a missing
        # size now rather than failing every capture batch.
        arena_bytes()

    prefix_cache_bytes = int(os.environ.get(PREFIX_CACHE_BYTES_ENV, "0"))
    if prefix_cache_bytes < 0:
        raise ValueError(f"{PREFIX_CACHE_BYTES_ENV} must be nonnegative")
    is_writer = get_parallel().attn_tp_rank == 0
    prefix_cache, prefix_capacity = _configure_prefix_cache(
        scheduler.tree_cache, model_runner, features, prefix_cache_bytes, is_writer
    )
    runtime = CaptureRuntime(
        sink=SpecCaptureSink(features.return_aux_hidden_state_layer_ids),
        is_writer=is_writer,
        gpu_put=gpu_put_enabled(),
        max_pending_batches=int(os.environ.get(MAX_PENDING_BATCHES_ENV, "2")),
        defer_response=scheduler.defer_response,
        prefix_cache=prefix_cache,
        prefix_capacity_tokens=prefix_capacity,
    )
    method = features.aux_hidden_state_capture
    runtime.capture_method = str(getattr(method, "name", method)).lower()
    model_runner.forward_observer = runtime.observer
    scheduler.output_streamer.spec_capture_results = runtime.results
    scheduler._specforge_capture_runtime = runtime
    logger.info(
        "SpecForge spec capture enabled (%s publication, writer=%s)",
        "nvlink" if nvlink else "device" if runtime.gpu_put else "host",
        runtime.is_writer,
    )
    return runtime
