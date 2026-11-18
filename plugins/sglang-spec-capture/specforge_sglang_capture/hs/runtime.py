"""Native capture admission and publication, owned by the scheduler thread.

SG owns KV/SSM admission and TP transaction receipts. This module owns only HS
leases, output allocations and immutable external-attempt publication. Assembly
workers and sink callbacks never read a Req or execute a collective.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future
from typing import Any

import torch

from .attempts import (
    Attempt, AttemptCapacityDeclined, AttemptRegistry, CaptureControlRejected,
    digest_request, scalar_error,
)
from .cache import PrefixFeatureCache
from .config import HSConfig
from .controller import CaptureController
from .protocol import (
    CaptureContractError, CapturedSpan, ExecutionIdentity, FeatureSchema,
    FeatureSpec, OutputReservation, PrefixLease,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _Candidate:
    execution: ExecutionIdentity
    entry: Attempt
    tokens: tuple[int, ...]
    lease: PrefixLease | None = None
    reservation: OutputReservation | None = None
    L: int = 0


@dataclasses.dataclass
class _Publication:
    execution: ExecutionIdentity
    entry: Attempt
    spec: dict
    plan: Any
    materialize: Future
    output: Future
    consumed: bool = False
    sample: Any = None
    sink: Future | None = None
    sink_index: int = 0
    sink_count: int = 0
    cleanup: Future | None = None
    failed: BaseException | None = None
    publish_started: float = 0.0
    diagnostic: dict | None = None


class HSCompletion:
    """Native ResponseCompletion for one immutable execution.

    Cancelling a response fences its execution. The runtime retains the job
    until materialization and any shared sink reader really stop.
    """

    def __init__(self, runtime, job):
        self.runtime, self.job = runtime, job
        self._cancelled = self._resolved = False

    def done(self):
        return self._cancelled or self.job.output.done()

    def result(self):
        if self._cancelled or self._resolved:
            return
        self.runtime._assert_owner()
        if not self.runtime._same_execution(self.job):
            self._resolved = True
            return
        try:
            result = self.job.output.result()
        except Exception as exc:
            result = {"sample_id": self.job.spec["sample_id"], "error": str(exc)}
        if self.job.diagnostic is not None:
            now = time.monotonic_ns()
            self.job.diagnostic["p0_v1_response_callback_ns"] = now
            observed = self.job.diagnostic.get("p0_v1_owner_observed_ns")
            if observed is not None:
                elapsed = (now - observed) / 1e6
                self.job.diagnostic["p0_v1_owner_to_response_callback_ms"] = elapsed
                result.setdefault("timings", {})["p0_v1_owner_to_response_callback_ms"] = elapsed
        self.runtime.capture.results[self.job.execution.request_id] = result
        self._resolved = True

    def cancel(self):
        if self._resolved:
            return False
        if not self._cancelled:
            self.runtime._assert_owner()
            self.runtime._cancel_job(self.job, "deferred-response-cancelled")
            self._cancelled = True
        return True


class HSCaptureRuntime:
    control_namespace = "specforge_hs"

    def __init__(self, scheduler, capture, config: HSConfig, *, manifest=None,
                 schema=None, tp_size=None, tp_rank=0):
        self.s, self.capture, self.config = scheduler, capture, config
        self.capture.sink.require_hard_pin = True
        self.writer, self.enabled = capture.is_writer, config.cache
        self.owner_thread = threading.get_ident()
        self.tp_size, self.tp_rank = tp_size, tp_rank
        if self.tp_size is None:
            from sglang.srt.runtime_context import get_parallel
            parallel = get_parallel()
            # Native capture elects the attention-TP writer from this same
            # resolved topology; keep its TP rank as execution diagnostics.
            self.tp_size, self.tp_rank = parallel.tp_size, parallel.tp_rank
        self.epoch = self._writer_broadcast("startup.epoch", lambda: uuid.uuid4().hex)
        if manifest is None or schema is None:
            manifest, schema = self._local_phase("startup.model", self._model_identity)

        def identify():
            self.schema = schema
            self.schema_wire = [dataclasses.asdict(feature) for feature in schema.features]
            for feature in self.schema_wire:
                feature["layer_ids"] = list(feature["layer_ids"])
            identity = {**manifest, "load_epoch": self.epoch, "schema": self.schema_wire,
                        "chunk_tokens": config.chunk_tokens}
            self.signature = hashlib.sha256(json.dumps(
                identity, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
            self.namespace = "spec-feature-v1:" + self.signature
            self.registry = AttemptRegistry(self.epoch, config.attempt_capacity)
            return self.signature

        signature = self._local_phase("startup.identity", identify)
        self._agree("startup.signature", signature)
        self.cache = self.controller = None

        def create_core():
            if self.writer:
                self.cache = PrefixFeatureCache(schema, config.chunk_tokens,
                                                config.cache_bytes if self.enabled else 0)
                self.controller = CaptureController(
                    self.cache, schema, config.pinned_bytes, config.output_bytes,
                    config.device_bytes, max_workers=config.max_workers, segmented=config.segmented)
                if config.segmented:
                    self.capture.sink._get_manifest_owner()

        try:
            self._local_phase("startup.core", create_core)
        except Exception:
            if self.controller is not None:
                self.controller.shutdown(wait=True)
            raise
        self.candidates: dict[str, _Candidate] = {}
        self.jobs: dict[ExecutionIdentity, _Publication] = {}
        self.history = deque(maxlen=256)
        self._diagnostics = os.environ.get("SPECFORGE_HS_DIAGNOSTICS", "0") == "1"
        self._diagnostic_every = max(1, int(os.environ.get("SPECFORGE_HS_DIAGNOSTICS_EVERY", "1")))
        self._diagnostic_limit = max(0, int(os.environ.get("SPECFORGE_HS_DIAGNOSTICS_MAX", "4096")))
        self._diagnostic_sequence = 0
        self._diagnostic_batch = 0
        self.diagnostics = deque(maxlen=self._diagnostic_limit)
        self._transferred_tokens = []
        self._closed = False
        self.fatal_capture_error = None
        self.quarantined_snapshots = []
        self.quarantined_publications = []

    def _assert_owner(self):
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError("HS state may only be changed on the scheduler thread")

    def _local_phase(self, phase, call):
        """Vote local work BEFORE any subsequent internal collective.

        SG's outer callback phase runs too late to repair divergent nested
        collectives. Labels also fail closed if ranks choose different plugin
        branches that happen to arrive at a gather together. ``call`` must not
        itself contain a collective; poll/lifecycle therefore remain local.
        """
        value, error = None, None
        try:
            value = call()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        votes = self.gather((phase, error))
        if any(not isinstance(vote, tuple) or len(vote) != 2
               or vote[0] != phase for vote in votes):
            raise CaptureControlRejected(f"TP HS phase order differs at {phase}: {votes!r}")
        if any(vote[1] is not None for vote in votes):
            raise CaptureControlRejected(f"TP HS {phase} failed: {votes!r}")
        return value

    def _agree(self, phase, value):
        votes = self.gather((phase, value))
        if any(not isinstance(vote, tuple) or len(vote) != 2
               or vote[0] != phase or vote[1] != value for vote in votes):
            raise CaptureControlRejected(f"TP HS {phase} differs: {votes!r}")

    def _writer_broadcast(self, phase, build):
        payload = self._local_phase(phase + ".build", lambda: build() if self.writer else None)
        return self.broadcast(payload)

    def _broadcast_wire(self, phase, entry):
        wire = self._writer_broadcast(phase, lambda: self._wire(entry))

        def apply():
            if any(wire.get(name) != expected for name, expected in (
                ("server_epoch", self.epoch), ("store_id", entry.identity[0]),
                ("sample_id", entry.identity[1]), ("gen", entry.identity[2]),
                ("attempt_id", entry.attempt_id), ("reservation_token", entry.token))):
                raise CaptureContractError("TP HS wire identity changed")
            entry.state = wire["state"]

        self._local_phase(phase + ".apply", apply)
        return wire

    def broadcast(self, value):
        if self.tp_size == 1:
            return value
        if hasattr(self.s, "prefix_admission_broadcast"):
            return self.s.prefix_admission_broadcast(value)
        import torch.distributed as dist
        values = [value]
        dist.broadcast_object_list(values, src=self.s.tp_group.ranks[0],
                                   group=self.s.tp_cpu_group)
        return values[0]

    def gather(self, value):
        if self.tp_size == 1:
            return [value]
        if hasattr(self.s, "prefix_admission_gather"):
            return self.s.prefix_admission_gather(value)
        import torch.distributed as dist
        values = [None] * self.tp_size
        dist.all_gather_object(values, value, group=self.s.tp_cpu_group)
        return values

    def _model_identity(self):
        from sglang.srt.arg_groups.model_override_base import resolved_view
        from sglang.srt.runtime_context import get_exec, get_parallel, get_schedule
        from specforge_sglang_capture.sink import gpu_put_enabled

        args = resolved_view(self.s.server_args)
        parallel = get_parallel()
        schedule = get_schedule()
        features = get_exec().features
        forbidden = {
            "chunked_prefill": schedule.chunked_prefill_size != -1,
            "PP": parallel.pp_size != 1,
            "DP": parallel.dp_size != 1,
            "attention_DP": parallel.attn_dp_size != 1,
            "CP": parallel.attn_cp_size != 1 or parallel.dcp_size != 1,
            "HiCache": bool(args.enable_hierarchical_cache),
            "LoRA": bool(args.enable_lora),
            "sessions": bool(args.enable_session_radix_cache)
                        or bool(getattr(args, "enable_streaming_session", False)),
            "offload": any(bool(getattr(args, name, False)) for name in (
                "hicache_storage_backend", "enable_lmcache", "enable_flexkv",
                "disaggregation_decode_enable_offload_kvcache")),
            "GPU_publication": gpu_put_enabled(),
            "priority_preemption": bool(self.s.enable_priority_preemption),
            "prefill_delayer": bool(self.s.prefill_delayer)
                               or bool(getattr(self.s, "min_free_slots_delayer", None)),
            "DLLM": self.s.dllm_config is not None,
            "speculative_decode": not self.s.spec_algorithm.is_none(),
            "radix_disabled": self.enabled and bool(args.disable_radix_cache),
            "memory_saver": bool(args.enable_memory_saver),
            "deferred_weights": bool(getattr(args, "wait_for_weights_before_ready", False)),
        }
        if bad := [name for name, present in forbidden.items() if present]:
            raise CaptureContractError("HS capture unsupported configuration: " + ", ".join(bad))
        if self.enabled:
            if not args.enable_deterministic_inference:
                raise CaptureContractError("HS reuse requires --enable-deterministic-inference")
            incompatible = [name for name in (
                "enable_int8_mamba_checkpoint", "enable_mamba_cache_stochastic_rounding",
                "enable_linear_replayssm", "enable_linear_replayssm_spec",
            ) if getattr(args, name, False)]
            if incompatible:
                raise CaptureContractError("HS reuse requires exact recurrent checkpoints: "
                                           + ", ".join(incompatible))
            from sglang.srt.managers.prefix_admission import validate_capture_checkpoint_support
            checkpoint = validate_capture_checkpoint_support(self.s)
        else:
            checkpoint = {"reuse": False}
        if self.config.chunk_tokens % self.s.page_size:
            raise CaptureContractError("HS chunk tokens must be a multiple of the KV page size")
        layers = tuple(features.return_aux_hidden_state_layer_ids or ())
        if not layers:
            raise CaptureContractError("HS capture needs ordered auxiliary layer IDs")
        text = self.s.model_config.hf_text_config
        width = int(text.hidden_size)
        dtype = str(self.s.model_config.dtype).removeprefix("torch.")
        schema = FeatureSchema((FeatureSpec("aux", width * len(layers), dtype, layers),
                                FeatureSpec("last_hidden", width, dtype, ())))
        manifest = {
            "model": os.path.realpath(args.model_path), "revision": args.revision,
            "config": text.to_dict(), "dtype": dtype,
            "quantization": args.quantization, "load_format": str(args.load_format),
            "weights": "immutable-for-load-epoch", "tp_size": parallel.tp_size,
            "deterministic_inference": bool(args.enable_deterministic_inference),
            "kv_dtype": args.kv_cache_dtype, "bf16_gemm_backend": args.bf16_gemm_backend,
            "attention_backend": args.attention_backend,
            "prefill_attention_backend": args.prefill_attention_backend,
            "linear_attention_backend": getattr(args, "linear_attn_backend", None),
            "checkpoint_semantics": checkpoint,
            "capture_method": str(features.aux_hidden_state_capture),
            "aux_semantics": "raw_ordered_capture", "last_hidden_semantics": "post_norm",
            "layout": "contiguous_token_width", "positions": "standard_text_causal",
        }
        return manifest, schema

    def capability(self):
        return {"protocol_version": 1, "server_epoch": self.epoch,
                "begin_batch": True, "published_ownership": "attempt-v1",
                "capture_namespace": self.namespace, "capture_signature": self.signature,
                "schema": self.schema_wire, "cache_enabled": self.enabled,
                "transport": "host", "no_oversize_fallback": True,
                "feature_layouts": (["hs-manifest-v1", "hs-prefix-v2"] if self.config.segmented else ["full-v1"]),
                "segment_hard_pin_required": self.config.segmented,
                "support": {"nonchunked": True, "host_publication": True,
                            "causal_text_only": True, "pp_size": 1, "dp_size": 1,
                            "offload": False, "sessions": False, "lora": False,
                            "weight_updates": False, "complete_schema": True},
                "limits": {name: getattr(self.config, name) for name in (
                    "attempt_capacity", "chunk_tokens", "cache_bytes", "pinned_bytes",
                    "output_bytes", "device_bytes", "max_workers")}}

    def _claim(self, req):
        from specforge_sglang_capture.capture import request_spec
        spec = request_spec(req)
        if not spec or "protocol_version" not in spec:
            return None
        for name, expected in (("protocol_version", 1), ("server_epoch", self.epoch),
                               ("capture_namespace", self.namespace),
                               ("capture_signature", self.signature)):
            if spec.get(name) != expected:
                raise CaptureContractError("capture " + name + " mismatch")
        layouts = ('hs-manifest-v1', 'hs-prefix-v2') if self.config.segmented else ('full-v1',)
        if spec.get('feature_layout', 'full-v1') not in layouts:
            raise CaptureContractError('capture feature layout was not negotiated')
        if spec.get("object_key_layout") != "attempt-v1":
            raise CaptureContractError("HS capture requires negotiated immutable attempt keys")
        if req.sampling_params.max_new_tokens != 0:
            raise CaptureContractError("HS capture requires max_new_tokens=0")
        if any(getattr(req, name, None) is not None for name in (
            "multimodal_inputs", "session", "lora_id", "positional_embed_overrides",
            "cache_salt", "session_id", "input_embeds")):
            raise CaptureContractError("HS capture supports standard causal text only")
        if set(spec.get("features", {})) != {"aux", "last_hidden"}:
            raise CaptureContractError("HS capture requires complete aux and last_hidden schema")
        names = list(spec["features"].values())
        for item in spec.get("passthrough", []):
            if not isinstance(item, dict):
                raise CaptureContractError("passthrough must contain objects")
            names.append(item.get("name"))
            shape = item.get("shape")
            if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
                raise CaptureContractError("invalid passthrough shape")
            count = 1
            for dim in shape:
                count *= dim
            def elements(value):
                return sum(elements(v) for v in value) if isinstance(value, list) else 1
            if elements(item.get("data")) != count:
                raise CaptureContractError("passthrough data does not match its declared shape")
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise CaptureContractError("capture output names must be distinct nonempty strings")
        if spec.get("request_digest", digest_request(req.origin_input_ids, spec)) != digest_request(req.origin_input_ids, spec):
            raise CaptureContractError("capture request digest mismatch")
        entry = getattr(req, "_specforge_hs_attempt", None)
        if entry is None:
            entry = self.registry.claim(spec, req.origin_input_ids)
            entry.request = req
            req._specforge_hs_attempt = entry
        elif (entry.revoked or entry.request is not req
              or self.registry.lookup(spec) is not entry):
            raise CaptureContractError("capture attempt is stale or revoked")
        req.extra_key = self.namespace if self.enabled else "spec-attempt:" + entry.token
        return entry

    def prepare(self, scheduler, req):
        def membership():
            self._assert_owner()
            from specforge_sglang_capture.capture import request_spec
            spec = request_spec(req)
            return req.rid, bool(spec and "protocol_version" in spec)

        member = self._local_phase("prepare.input", membership)
        self._agree("prepare.membership", member)
        if not member[1]:
            return None
        self._sync_transfers("prepare.transfers")
        return self._local_phase("prepare.lease", lambda: self._prepare_local(req))

    def _prepare_local(self, req):
        if self._closed or self.fatal_capture_error:
            raise CaptureContractError(self.fatal_capture_error or "HS runtime is closed")
        entry = self._claim(req)
        execution = ExecutionIdentity(self.epoch, req.rid, entry.attempt_id, entry.execution_epoch)
        if req.rid in self.candidates:
            raise CaptureContractError("HS admission candidate already active")
        candidate = _Candidate(execution, entry, tuple(req.origin_input_ids))
        self.candidates[req.rid] = candidate
        req._specforge_hs_execution = execution
        if not candidate.tokens:
            raise CaptureContractError("HS capture requires a nonempty prompt")
        cap = 0
        if self.writer:
            candidate.lease = self.cache.lookup_and_pin(
                self.namespace, candidate.tokens,
                len(candidate.tokens) - 1 if self.enabled and not getattr(req, "_specforge_hs_force_full", False) else 0)
            cap = candidate.lease.coverage_end
        from sglang.srt.managers.prefix_admission import PrefixAdmissionProposal
        return PrefixAdmissionProposal(cap, dataclasses.astuple(execution), candidate)

    def after_match(self, scheduler, req, proposal, lawful_prefix_len):
        self._assert_owner()
        candidate = self.candidates[req.rid]
        if not 0 <= lawful_prefix_len <= len(candidate.tokens) - 1:
            raise CaptureContractError("native recovery prefix is outside [0,N-1]")
        if self.writer:
            candidate.lease = self.cache.shrink_lease(candidate.lease, lawful_prefix_len)
        candidate.L = lawful_prefix_len

    def reserve(self, scheduler, req, proposal, lawful_prefix_len):
        self._assert_owner()
        candidate = self.candidates[req.rid]
        if candidate.L != lawful_prefix_len:
            raise CaptureContractError("native recovery prefix changed after match")
        if not self.writer:
            return True
        from specforge_sglang_capture.capture import request_spec
        passthrough = 0
        from specforge_sglang_capture.sink import STR_DTYPE
        for item in request_spec(req).get("passthrough", []):
            dtype = STR_DTYPE.get(item.get("dtype", "int64"))
            if dtype is None:
                raise CaptureContractError("unsupported passthrough dtype")
            shape = item.get("shape")
            if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
                raise CaptureContractError("invalid passthrough shape")
            elements = 1
            for dim in shape:
                elements *= dim
            passthrough += elements * torch.empty((), dtype=dtype).element_size()
        candidate.reservation = self.controller.try_reserve_output(
            candidate.execution, len(candidate.tokens), lawful_prefix_len, passthrough)
        return candidate.reservation is not None

    def commit(self, scheduler, req, proposal, lawful_prefix_len):
        self._assert_owner()
        candidate = self.candidates[req.rid]
        if candidate.L != lawful_prefix_len:
            raise CaptureContractError("native recovery prefix changed before commit")
        if self.writer:
            from specforge_sglang_capture.capture import request_spec
            split_tail = request_spec(req).get('feature_layout') == 'hs-prefix-v2'
            req._specforge_hs_plan = self.controller.freeze_plan(
                candidate.execution, self.namespace, candidate.tokens, len(candidate.tokens),
                lawful_prefix_len, candidate.lease, candidate.reservation,
                H=proposal.max_prefix_len if split_tail else None)
        candidate.entry.state = "ADMITTED"
        candidate.entry.last_plan = {"N": len(candidate.tokens), "L": lawful_prefix_len,
            "H": proposal.max_prefix_len, "span": [lawful_prefix_len, len(candidate.tokens)],
            "execution_epoch": candidate.execution.execution_epoch, "rank": self.tp_rank}
        self.history.append({"attempt_id": candidate.entry.attempt_id,
                             **candidate.entry.last_plan})

    def rollback(self, scheduler, req, proposal, reason):
        self._assert_owner()
        candidate = self.candidates.pop(req.rid, None)
        if candidate is None:
            return
        if self.writer:
            if candidate.reservation is not None:
                self.controller.release_output(candidate.reservation)
            if candidate.lease is not None:
                self.cache.release_lease(candidate.lease)
            self.controller.poll_completed()
            self.controller.poll_drained()
        req._specforge_hs_plan = None
        candidate.entry.execution_epoch += 1
        req._specforge_hs_execution = None
        candidate.entry.state = "QUEUED"

    def lifecycle(self, scheduler, req, event, identity):
        self._assert_owner()
        if event == "shutdown":
            self.shutdown()
            return
        if event == "weight_update":
            raise CaptureContractError("HS capture forbids weight updates within a load epoch")
        if req is None:
            return
        entry = getattr(req, "_specforge_hs_attempt", None)
        if entry is None:
            from specforge_sglang_capture.capture import request_spec
            spec = request_spec(req)
            if not spec or "protocol_version" not in spec:
                return
            try:
                entry = self.registry.lookup(spec)
            except (KeyError, ValueError):
                return
            entry.request = req
            req._specforge_hs_attempt = entry
        current = getattr(req, "_specforge_hs_execution", None)
        if identity is not None and identity not in (current, dataclasses.astuple(current) if current else None):
            return
        if event == "finish":
            return  # Native finish precedes consume/deferred publication.
        if entry.transfer_certificate:
            return  # Receiver already owns the immutable published objects.
        candidate = self.candidates.pop(req.rid, None)
        if candidate is not None and self.writer:
            if candidate.reservation is not None:
                entry.drains.append(self.controller.invalidate(candidate.execution, event))
            if candidate.lease is not None:
                self.cache.release_lease(candidate.lease)
        for job in tuple(self.jobs.values()):
            if job.entry is entry and (current is None or job.execution == current):
                self._cancel_job(job, event, revoke=event not in ("retract", "retry"))
        req._specforge_hs_plan = None
        req._specforge_hs_execution = None
        entry.execution_epoch += 1
        if event in ("retract", "retry"):
            req._specforge_hs_force_full = True
            entry.state = "QUEUED"
        else:
            entry.revoked, entry.state = True, "DRAINING"
        self._mark_drained(entry)

    def invalidate(self, entry, reason):
        if entry.transfer_certificate:
            return
        entry.revoked, entry.state = True, "DRAINING"
        if entry.request is not None:
            self.lifecycle(self.s, entry.request, reason,
                           getattr(entry.request, "_specforge_hs_execution", None))
        self._mark_drained(entry)

    def _mark_drained(self, entry):
        if (entry.state != "RETIRED" and entry.revoked and entry.deliveries == 0
                and all(handle.done() for handle in entry.drains)):
            entry.state = "WRITER_DRAINED"

    def _current(self, job):
        return ((not job.entry.revoked or bool(job.entry.transfer_certificate))
                and self._same_execution(job))

    def _same_execution(self, job):
        return (job.entry.execution_epoch == job.execution.execution_epoch
                and job.entry.attempt_id == job.execution.attempt_id)

    def _cancel_job(self, job, reason, *, revoke=True):
        if job.entry.transfer_certificate:
            return
        if revoke:
            job.entry.revoked, job.entry.state = True, "DRAINING"
        if job.failed is None:
            job.failed = RuntimeError(reason)
            job.entry.drains.append(self.controller.invalidate(job.execution, reason))

    def snapshot(self, req, item, aux, last_hidden):
        """Freeze compact owned suffixes on the forward stream exactly once."""
        self._assert_owner()
        plan = req._specforge_hs_plan
        candidate = self.candidates.pop(req.rid)
        if plan.execution != candidate.execution or plan.execution in self.jobs:
            raise CaptureContractError("capture plan execution is stale or already submitted")
        features = {}
        try:
            for name, tensor in (("aux", aux), ("last_hidden", last_hidden)):
                if tensor is None:
                    raise CaptureContractError("capture is missing " + name)
                features[name] = tensor[item.start:item.start + item.length].detach().clone(
                    memory_format=torch.contiguous_format)
            ready = None
            if next(iter(features.values())).is_cuda:
                ready = torch.cuda.Event()
                ready.record()
            future = self.controller.submit(plan, CapturedSpan(
                plan.execution, plan.L, plan.N, features, ready))
        except BaseException as exc:
            try:
                if any(t.is_cuda for t in features.values()):
                    torch.cuda.current_stream(next(iter(features.values())).device).synchronize()
            except BaseException as drain_error:
                retained = Future()
                candidate.entry.drains.append(retained)
                self.quarantined_snapshots.append((plan, features, retained))
                self.fatal_capture_error = "capture snapshot drain unknown: " + str(drain_error)
                raise CaptureContractError(self.fatal_capture_error) from None
            candidate.entry.drains.append(self.controller.invalidate(plan.execution, "snapshot-failed"))
            self.invalidate(candidate.entry, "snapshot-failed")
            raise exc
        output = Future()
        output.set_running_or_notify_cancel()
        from specforge_sglang_capture.capture import request_spec
        job = _Publication(plan.execution, candidate.entry, copy.deepcopy(request_spec(req)),
                           plan, future, output)
        if self._diagnostics:
            sequence = self._diagnostic_sequence
            self._diagnostic_sequence += 1
            if sequence % self._diagnostic_every == 0 and sequence // self._diagnostic_every < self._diagnostic_limit:
                job.diagnostic = {"p0_v1_event": "publication", "server_epoch": self.epoch,
                    "request_id": plan.execution.request_id, "attempt_id": plan.execution.attempt_id,
                    "execution_epoch": plan.execution.execution_epoch, "rank": self.tp_rank,
                    "store_id": job.spec["store_id"], "sample_id": job.spec["sample_id"],
                    "request_digest": job.entry.request_digest, "N": plan.N, "L": plan.L,
                    "suffix_tokens": plan.N - plan.L, "p0_v1_snapshot_ns": time.monotonic_ns()}
        self.jobs[plan.execution] = job
        candidate.entry.deliveries += 1
        return job

    def consume(self, jobs, batch):
        self._assert_owner()
        reqs = {req.rid: req for req in batch.reqs}
        if self._diagnostics:
            self._diagnostic_batch += 1
        for job in jobs:
            if job.diagnostic is not None:
                # One ID per native consume batch; do not sum shared batch work
                # once per sample. This timestamp is NOT a CUDA forward timer.
                job.diagnostic.update(p0_v1_consume_batch_id=self._diagnostic_batch,
                    p0_v1_consume_batch_samples=len(jobs),
                    p0_v1_consume_ns=time.monotonic_ns())
            req = reqs.get(job.execution.request_id)
            if req is None or not self._current(job):
                self._cancel_job(job, "capture request disappeared before consume")
                continue
            if not req.finished():
                self._cancel_job(job, "capture requests must finish at prefill")
                self.capture.results[req.rid] = {"sample_id": job.spec["sample_id"],
                                                "error": "capture requests must finish at prefill"}
                continue
            job.consumed = True
            self.capture.defer_response(req, HSCompletion(self, job))
        self.poll()

    def capture_error(self, req, reason):
        self._assert_owner()
        entry = getattr(req, "_specforge_hs_attempt", None)
        if entry is not None:
            self.invalidate(entry, reason)

    def _drop_job(self, job):
        if self.jobs.get(job.execution) is not job:
            return
        self.jobs.pop(job.execution)
        job.entry.deliveries -= 1
        if job.entry.deliveries < 0:
            raise RuntimeError("capture delivery ownership underflow")
        self._mark_drained(job.entry)

    def _finish_error(self, job, error):
        if self._current(job):
            self._cancel_job(job, str(error))
        if job.sample is not None:
            job.sample.release()
            job.sample = None
        if not job.output.done():
            job.output.set_exception(scalar_error(error))
        self._drop_job(job)

    def _validate_publication(self, job, result):
        if not isinstance(result, dict):
            raise CaptureContractError("capture sink result is not an object")
        for name in ("store_id", "sample_id", "gen"):
            if result.get(name) != job.spec[name]:
                raise CaptureContractError("capture sink changed " + name)
        names = [*job.spec["features"].values(),
                 *(item["name"] for item in job.spec.get("passthrough", []))]
        base = (f"{job.spec['store_id']}/{job.spec['sample_id']}/g1/attempt/"
                f"{job.execution.server_epoch}/{job.spec['reservation_token']}")
        keys = {name: base + "/" + name for name in names}
        if set(result.get("features", {})) != set(keys):
            raise CaptureContractError("capture sink returned incomplete feature names")
        if any(result["features"][name].get("key") != key for name, key in keys.items()):
            raise CaptureContractError("capture sink immutable physical keys mismatch")
        for feature in self.schema.features:
            name = job.spec["features"][feature.name]
            meta = result["features"][name]
            if meta.get("shape") != [1, job.plan.N, feature.width] or meta.get("dtype") != feature.dtype:
                raise CaptureContractError("capture sink feature schema mismatch")
        for item in job.spec.get("passthrough", []):
            meta = result["features"][item["name"]]
            if meta.get("shape") != item["shape"] or meta.get("dtype") != item.get("dtype", "int64"):
                raise CaptureContractError("capture sink passthrough schema mismatch")
        if result.get("aux_layer_ids") != list(self.schema.features[0].layer_ids):
            raise CaptureContractError("capture sink auxiliary layer order mismatch")
        return keys

    def poll(self, scheduler=None):
        """Progress independent jobs and batch only currently ready publications."""
        self._assert_owner()
        if not self.writer:
            return
        self.controller.poll_completed()
        self.controller.poll_drained()

        ready = []
        for job in tuple(self.jobs.values()):
            if job.cleanup is not None:
                if not job.cleanup.done():
                    continue
                try:
                    job.cleanup.result()
                except Exception as exc:
                    # Writer has stopped reading, but deletion remains unproven.
                    # Keep this exact attempt live so the client can retry cleanup.
                    logger.error("capture cleanup failed for %s: %s", job.execution, exc)
                self._finish_error(job, job.failed or RuntimeError("capture cancelled"))
                continue
            if job.sink is not None or not job.materialize.done():
                continue
            try:
                if job.sample is None:
                    job.sample = job.materialize.result()
                if job.failed is not None or not self._current(job):
                    raise job.failed or RuntimeError("capture execution invalidated")
                if self.fatal_capture_error:
                    raise RuntimeError(self.fatal_capture_error)
                if job.consumed:
                    ready.append(job)
            except Exception as exc:
                self._finish_error(job, exc)
        if ready:
            if self.config.segmented:
                samples = [(job.spec, job.sample, self.schema, self.namespace) for job in ready]
            else:
                samples = [(job.spec, job.sample.features["aux"],
                            job.sample.features["last_hidden"]) for job in ready]
            enqueue_ns = time.monotonic_ns() if self._diagnostics else 0
            try:
                future = (self.capture.sink.submit_segmented_samples(samples) if self.config.segmented
                          else self.capture.sink.submit_samples(samples))
            except Exception as exc:
                samples.clear()
                future = Future()
                future.set_exception(scalar_error(exc))
            for index, job in enumerate(ready):
                job.sink, job.sink_index, job.sink_count = future, index, len(ready)
                job.publish_started = time.perf_counter()
                if job.diagnostic is not None:
                    job.diagnostic["p0_v1_runtime_sink_enqueue_ns"] = enqueue_ns
                    done_ns = job.sample.timings.get("p0_v1_materialize_done_ns")
                    if done_ns is not None:
                        job.diagnostic["p0_v1_materialize_to_enqueue_ms"] = (enqueue_ns - done_ns) / 1e6
                job.entry.state = "PUBLISHING"
            samples = None  # The sink now owns and clears its argument list.
        ready.clear()
        for job in tuple(self.jobs.values()):
            if job.sink is None or not job.sink.done() or job.cleanup is not None:
                continue
            timing = getattr(job.sink, 'publication_timing', None)
            observed_ns = time.monotonic_ns() if timing or job.diagnostic is not None else 0
            if timing:
                trace, batch = timing
                from specforge.runtime.data_plane.publication_timing import IDENTITY
                trace.emit('observed', batch=batch, index=job.sink_index, ns=observed_ns,
                           identity={k:job.spec.get(k) for k in IDENTITY})
            try:
                results = job.sink.result()
                if len(results) != job.sink_count:
                    raise CaptureContractError("capture sink batch result count mismatch")
                result = results[job.sink_index]
                keys = self._validate_publication(job, result)
                if job.failed is not None or not self._current(job):
                    job.failed = job.failed or RuntimeError("capture cancelled during publication")
                    # Keep the publication owner until the exact-key deletion is
                    # ordered after its already-completed writer.
                    job.cleanup = self.capture.sink.discard_sample(job.spec)
                    continue
                entry = job.entry
                if entry.deliveries != 1 or entry.readers or any(not handle.done() for handle in entry.drains):
                    raise CaptureContractError("capture transfer precedes all execution drains")
                timings = dict(job.sample.timings)
                timings["publication_queue_ms"] = (time.perf_counter() - job.publish_started) * 1000
                if job.diagnostic is not None:
                    observed = observed_ns
                    sink_timing = dict(getattr(job.sink, "p0_v1_timings", {}))
                    job.diagnostic.update(timings)
                    job.diagnostic.update(sink_timing)
                    job.diagnostic["p0_v1_owner_observed_ns"] = observed
                    if "p0_v1_sink_done_ns" in sink_timing:
                        job.diagnostic["p0_v1_sink_done_to_owner_ms"] = (observed - sink_timing["p0_v1_sink_done_ns"]) / 1e6
                        job.diagnostic["p0_v1_sink_queue_ms"] = (sink_timing["p0_v1_sink_start_ns"] - sink_timing["p0_v1_sink_enqueue_ns"]) / 1e6
                    timings.update({key: value for key, value in job.diagnostic.items()
                                    if key.startswith("p0_v1_")})
                    self.diagnostics.append(job.diagnostic)
                entry.last_plan["timings"] = timings
                # Sink has cleared its source refs. Transfer allocation ownership
                # to cache (or retire it), then issue the physical-key certificate.
                job.sample.release()
                job.sample = None
                self._drop_job(job)
                entry.state = "PUBLISHED"
                certificate = {**self._wire(entry), "publication_ownership": "attempt-v1",
                               "object_key_layout": "attempt-v1", "physical_keys": keys,
                               "schema": self.schema_wire,
                               **({name: result[name] for name in ('feature_layout', 'segment_manifest', 'full_object_fallback')}
                                  if self.config.segmented else {})}
                entry.transfer_certificate = {"publication_ownership": "attempt-v1",
                    "physical_keys": keys, "publication_certificate": certificate}
                result.update(self._wire(entry))
                result.update(capture_namespace=self.namespace, capture_signature=self.signature,
                              reused_tokens=job.plan.L, schema=self.schema_wire)
                self.registry.retire(entry)
                self._transferred_tokens.append(entry.token)
                if not job.output.done():
                    job.output.set_result(result)
            except Exception as exc:
                if getattr(exc, "capture_writer_drained", True) is False:
                    self._cancel_job(job, "native publication drain unknown")
                    unknown = Future()
                    job.entry.drains.append(unknown)
                    self.quarantined_publications.append(job)
                    self.fatal_capture_error = "capture publication drain unknown; buffers and budgets retained"
                    if not job.output.done():
                        job.output.set_exception(scalar_error(exc))
                    self._drop_job(job)
                elif self.config.segmented:
                    job.failed = scalar_error(exc)
                    job.cleanup = self.capture.sink.discard_sample(job.spec)
                else:
                    self._finish_error(job, exc)
        for entry in self.registry.active.values():
            self._mark_drained(entry)
        self.controller.poll_completed()
        self.controller.poll_drained()

    def has_pending_work(self):
        # Writer-local drain progress, like native deferred outputs. This must
        # not be used as an unsynchronized TP collective idle predicate.
        return bool(self.jobs or self.quarantined_snapshots or self.quarantined_publications)

    def shutdown(self):
        self._assert_owner()
        if self._closed:
            return
        self._closed = True
        for entry in list(self.registry.active.values()):
            self.invalidate(entry, "server-shutdown")
        while self.jobs:
            self.poll()
            if self.jobs:
                time.sleep(0.001)
        if self.quarantined_snapshots or self.quarantined_publications:
            raise RuntimeError("HS shutdown retains quarantined storage; native drain is unproven")
        if self.controller:
            self.controller.shutdown(wait=True)
        self.capture.sink.close()

    def _wire(self, entry):
        return {"protocol_version": 1, "server_epoch": self.epoch, "store_id": entry.identity[0],
                "sample_id": entry.identity[1], "gen": 1, "attempt_id": entry.attempt_id,
                "reservation_token": entry.token, "issue_sequence": entry.sequence,
                "request_digest": entry.request_digest,
                "cleanup_owner": entry.cleanup_owner, "cleanup_token": entry.cleanup_token,
                "active_readers": len(entry.readers),
                "state": entry.state, "writer_drained": entry.state in ("WRITER_DRAINED", "PUBLISHED", "RETIRED"),
                **entry.last_plan,
                **({"object_key_layout": entry.object_key_layout} if entry.object_key_layout else {}),
                **entry.transfer_certificate}

    def control(self, scheduler, body):
        try:
            return self._control(body)
        except ValueError as exc:
            raise CaptureControlRejected(str(exc)) from None

    def _sync_transfers(self, phase):
        def packet():
            return list(self._transferred_tokens), self.fatal_capture_error

        update = self._writer_broadcast(phase + ".payload", packet)

        def apply():
            transferred, fatal = update
            if fatal:
                self.fatal_capture_error = fatal
            if not self.writer:
                for token in transferred:
                    entry = self.registry.tokens.get(token)
                    if entry is not None:
                        if entry.request is not None:
                            self.candidates.pop(entry.request.rid, None)
                        entry.state = "PUBLISHED"
                        self.registry.retire(entry)

        # A peer retirement can fail after local bookkeeping or resource checks.
        # Do not let its caller proceed to a later broadcast without this vote.
        self._local_phase(phase + ".apply", apply)
        self._transferred_tokens = []

    def _control_input(self, body):
        self._assert_owner()
        if not isinstance(body, dict) or int(body.get("protocol_version", -1)) != 1:
            raise ValueError("unsupported capture protocol")
        action = body.get("action")
        if action not in ("capability", "stats", "begin", "begin_batch", "resolve",
                          "read_acquire", "read_release", "cleanup_acquire", "cleanup_done"):
            raise ValueError("unknown capture control action")
        if action != "capability" and body.get("server_epoch") != self.epoch:
            raise ValueError("capture server epoch changed")
        # HTTP and typed IPC carry JSON. Agree the complete request, including
        # action and fencing tokens, before choosing any nested control branch.
        return action, json.dumps(body, sort_keys=True, separators=(",", ":"))

    def _control(self, body):
        request = self._local_phase("control.input", lambda: self._control_input(body))
        self._agree("control.request", request)
        self._local_phase("control.poll", self.poll)
        self._sync_transfers("control.transfers")
        action = request[0]
        if action == "capability":
            return self._local_phase("control.capability", self.capability)
        if action == "stats":
            def stats():
                return {**self.capability(), "active_attempts": len(self.registry.active),
                        "pending_deliveries": len(self.jobs),
                        "active_readers": sum(len(x.readers) for x in self.registry.active.values()),
                        "plans": list(self.history),
                        **({"p0_v1_diagnostics": list(self.diagnostics),
                            "p0_v1_diagnostic_seen": self._diagnostic_sequence,
                            "p0_v1_diagnostic_every": self._diagnostic_every,
                            "p0_v1_diagnostic_limit": self._diagnostic_limit} if self._diagnostics else {}),
                        "cache": self.cache.stats() if self.cache else {},
                        "controller": self.controller.stats() if self.controller else {},
                        "manifest_owner": self.capture.sink.manifest_stats() if self.writer else {}}
            return self._local_phase("control.stats", stats)
        if action in ("begin", "begin_batch"):
            return self._begin_control(body, action)

        kind, entry = self._local_phase("control.lookup", lambda: self._lookup_control(body))
        key = self._local_phase("control.target-key", lambda: (
            kind, entry.identity, entry.attempt_id, entry.request_digest, entry.token, entry.sequence))
        self._agree("control.target", key)
        if kind == "retired":
            return self._local_phase("control.retired", lambda: self._retired_control(body, entry))

        if action == "resolve":
            def resolve():
                if body.get("cancel"):
                    self.invalidate(entry, "external-cancel")
                    if entry.request is not None:
                        from sglang.srt.managers.io_struct import AbortReq
                        self.s.abort_request(AbortReq(rid=entry.request.rid))
                self._mark_drained(entry)
            self._local_phase("control.resolve.local", resolve)
            return self._broadcast_wire("control.resolve.wire", entry)
        if action == "read_acquire":
            wire = self._broadcast_wire("control.read-acquire.wire", entry)

            def validate_read():
                owner = str(body.get("reader_owner") or "")
                if not owner:
                    raise ValueError("read owner is required")
                existing = entry.readers.get(owner)
                # Lost replies can recover an existing permit after revoke;
                # they cannot create a new reader after its first release.
                if existing is None:
                    if any(old == owner for old, _ in entry.released_readers):
                        raise ValueError("released read owner cannot acquire again")
                    if len(entry.readers) + len(entry.released_readers) >= self.registry.capacity:
                        raise ValueError("attempt read owner history capacity exhausted")
                    if entry.revoked or wire["state"] != "PUBLISHED" or entry.cleanup_token is not None:
                        raise ValueError("attempt is not readable")
                    if sum(len(x.readers) for x in self.registry.active.values()) >= self.registry.capacity:
                        raise ValueError("read permit capacity exhausted")
                return owner, existing

            owner, existing = self._local_phase("control.read-acquire.validate", validate_read)
            self._agree("control.read-acquire.branch", (owner, existing))
            token = existing
            if token is None:
                token = self._writer_broadcast("control.read-acquire.token", lambda: uuid.uuid4().hex)

            def acquire_read():
                entry.readers[owner] = token
                return dict(self._wire(entry), reader_owner=owner, reader_token=token)
            return self._local_phase("control.read-acquire.commit", acquire_read)
        if action == "read_release":
            def release_read():
                owner, token = body.get("reader_owner"), body.get("reader_token")
                if entry.readers.get(owner) == token and token is not None:
                    entry.readers.pop(owner)
                    entry.released_readers.append((owner, token))
                elif (owner, token) not in entry.released_readers:
                    raise ValueError("unknown read permit")
                return dict(self._wire(entry), reader_owner=owner, reader_token=token, reader_released=True)
            return self._local_phase("control.read-release", release_read)
        if action == "cleanup_acquire":
            self._local_phase("control.cleanup-acquire.drain", lambda: self._mark_drained(entry))
            wire = self._broadcast_wire("control.cleanup-acquire.wire", entry)

            def validate_cleanup():
                owner = str(body.get("cleanup_owner") or "")
                if not owner or not wire["writer_drained"] or entry.readers:
                    raise ValueError("writer/readers not drained or cleanup owner absent")
                if entry.cleanup_owner is not None and entry.cleanup_owner != owner:
                    raise ValueError("another store owns sample cleanup")
                return owner, entry.cleanup_token

            owner, existing = self._local_phase("control.cleanup-acquire.validate", validate_cleanup)
            self._agree("control.cleanup-acquire.branch", (owner, existing))
            token = existing
            if token is None:
                token = self._writer_broadcast("control.cleanup-acquire.token", lambda: uuid.uuid4().hex)

            def acquire_cleanup():
                entry.cleanup_owner, entry.cleanup_token = owner, token
                return self._wire(entry)
            return self._local_phase("control.cleanup-acquire.commit", acquire_cleanup)
        if action == "cleanup_done":
            def validate_done():
                if entry.cleanup_token is None or body.get("cleanup_token") != entry.cleanup_token:
                    raise ValueError("cleanup permit mismatch")
                self._mark_drained(entry)
            self._local_phase("control.cleanup-done.validate", validate_done)
            self._broadcast_wire("control.cleanup-done.wire", entry)

            def retire():
                self.registry.retire(entry)
                return self._wire(entry)
            return self._local_phase("control.cleanup-done.retire", retire)
        raise AssertionError("validated control action has no implementation")

    def _begin_control(self, body, action):
        def validate():
            if self.fatal_capture_error:
                raise ValueError(self.fatal_capture_error)
            bodies = body.get("tasks") if action == "begin_batch" else [body]
            if not isinstance(bodies, list) or not bodies or len(bodies) > self.registry.capacity:
                raise ValueError("invalid begin_batch size")
            if any(not isinstance(item, dict) or item.get("server_epoch") != self.epoch for item in bodies):
                raise ValueError("capture server epoch changed")
            return bodies

        bodies = self._local_phase("control.begin.validate", validate)
        tokens = self._writer_broadcast("control.begin.tokens", lambda: [uuid.uuid4().hex for _ in bodies])
        prepared = self._local_phase("control.begin.preflight", lambda: self._begin_trial(bodies, tokens))

        def proposal():
            if prepared is None:
                return "capacity-declined", self.registry.capacity
            _, entries = prepared
            return "ready", tuple((entry.identity, entry.attempt_id, entry.request_digest,
                                    entry.token, entry.sequence, entry.object_key_layout) for entry in entries)

        decision = self._local_phase("control.begin.proposal", proposal)
        self._agree("control.begin.decision", decision)
        if prepared is None:
            # Every rank declined an uncommitted trial. No actual registry has
            # granted anything in this operation, so this is the ONLY typed
            # capacity refusal. Divergent preflights or commit failures are
            # generic fail-closed errors, never proof of an ungranted mutation.
            return {"error": "attempt registry capacity exhausted",
                    "error_code": "capture_capacity_declined", "reservation_granted": False}
        trial, entries = prepared
        replies = self._local_phase("control.begin.reply", lambda: [self._wire(entry) for entry in entries])

        def commit():
            self.registry.active, self.registry.tokens, self.registry.sequence = (
                trial.active, trial.tokens, trial.sequence)
        self._local_phase("control.begin.commit", commit)
        return {"server_epoch": self.epoch, "attempts": replies} if action == "begin_batch" else replies[0]

    def _begin_trial(self, bodies, tokens):
        trial = AttemptRegistry(self.registry.epoch, self.registry.capacity)
        trial.active, trial.tokens, trial.sequence = (
            dict(self.registry.active), dict(self.registry.tokens), self.registry.sequence)
        try:
            entries = trial.begin_batch(bodies, tokens)
        except AttemptCapacityDeclined:
            return None
        return trial, entries

    def _lookup_control(self, body):
        try:
            return "active", self.registry.lookup(body)
        except ValueError:
            identity = self.registry.identity(body)
            entry = next((entry for entry in self.registry.retired
                          if entry.token == body.get("reservation_token") and entry.identity == identity
                          and entry.attempt_id == body.get("attempt_id")), None)
            if entry is None:
                raise
            return "retired", entry

    def _retired_control(self, body, entry):
        action = body["action"]
        if action == "read_release":
            pair = body.get("reader_owner"), body.get("reader_token")
            if pair not in entry.released_readers:
                raise ValueError("unknown retired read permit")
            return dict(self._wire(entry), reader_owner=pair[0], reader_token=pair[1], reader_released=True)
        if action in ("cleanup_acquire", "cleanup_done", "resolve"):
            if action == "cleanup_done" and body.get("cleanup_token") != entry.cleanup_token:
                raise ValueError("cleanup permit mismatch")
            return self._wire(entry)
        raise ValueError("unknown, retired or mismatched reservation token")
