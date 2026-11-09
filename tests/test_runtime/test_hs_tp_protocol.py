"""Threaded CPU contracts for actual HS callbacks and SG's outer phase.

The rendezvous enforces collective operation order and exchanges real Python
payloads between two scheduler-owner threads. It is not CUDA, IPC or actual TP
execution. Point SPECFORGE_HS_TEST_SG_SOURCE at the companion SG checkout when
the interpreter's editable still points at the official baseline.
"""

import copy
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "plugins/sglang-spec-capture"))

from specforge_sglang_capture.capture import CaptureRuntime
from specforge_sglang_capture.hs.attempts import digest_request
from specforge_sglang_capture.hs.config import HSConfig
from specforge_sglang_capture.hs.protocol import FeatureSchema, FeatureSpec
from specforge_sglang_capture.hs.runtime import HSCaptureRuntime


def _sg_module():
    if source := os.environ.get("SPECFORGE_HS_TEST_SG_SOURCE"):
        path = Path(source) / "python/sglang/srt/managers/prefix_admission.py"
    else:
        spec = importlib.util.find_spec("sglang")
        if spec is None or spec.origin is None:
            return None
        path = Path(spec.origin).parent / "srt/managers/prefix_admission.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("_specforge_hs_test_prefix_admission", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SG = _sg_module()


class _Rendezvous:
    def __init__(self):
        self.condition = threading.Condition()
        self.steps, self.trace = {}, [[], []]
        self.failed = None

    def call(self, rank, step, operation, value):
        with self.condition:
            if self.failed:
                raise RuntimeError(self.failed)
            self.trace[rank].append((operation, value[0] if operation == "gather"
                                     and isinstance(value, tuple) and len(value) == 2 else None))
            slot = self.steps.setdefault(step, {"operation": operation, "values": {}})
            if slot["operation"] != operation or rank in slot["values"]:
                self.failed = f"collective order mismatch at step {step}: {slot['operation']} / {operation}"
                self.condition.notify_all()
                raise RuntimeError(self.failed)
            slot["values"][rank] = value
            if len(slot["values"]) == 2:
                slot["result"] = ([slot["values"][r] for r in (0, 1)]
                                  if operation == "gather" else slot["values"][0])
                self.condition.notify_all()
            deadline = time.monotonic() + 5
            while "result" not in slot and not self.failed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.failed = f"rank {rank} timed out at collective {step} ({operation})"
                    self.condition.notify_all()
                    break
                self.condition.wait(remaining)
            if self.failed:
                raise RuntimeError(self.failed)
            return copy.deepcopy(slot["result"])


class _Sink:
    aux_layer_ids = [1, 3]

    def close(self):
        pass


def _raise(message):
    raise RuntimeError(message)


def _schema():
    return FeatureSchema((FeatureSpec("aux", 2, "float32", (1, 3)),
                          FeatureSpec("last_hidden", 2, "float32", ())))


@unittest.skipIf(SG is None, "companion native SG prefix-admission module is unavailable")
class ThreadedTPProtocolTests(unittest.TestCase):
    def pair(self, operation, *, initialize=None, configure=None):
        rendezvous = _Rendezvous()
        results, errors, snapshots = [None, None], [None, None], [None, None]

        def run(rank):
            step = 0

            def collective(operation, value):
                nonlocal step
                current, step = step, step + 1
                return rendezvous.call(rank, current, operation, value)

            scheduler = SimpleNamespace(
                prefix_admission_broadcast=lambda value: collective("broadcast", value),
                prefix_admission_gather=lambda value: collective("gather", value),
                abort_request=lambda req: None)
            coordinator = SG.PrefixAdmissionCoordinator.__new__(SG.PrefixAdmissionCoordinator)
            coordinator.gather = scheduler.prefix_admission_gather
            coordinator.broadcast = scheduler.prefix_admission_broadcast

            def phase(call):
                value, failures = coordinator.phase(call)
                if any(error is not None for error in failures):
                    return None, failures
                return coordinator.broadcast(value), failures

            runtime = None
            try:
                native = CaptureRuntime(sink=_Sink(), is_writer=rank == 0,
                    gpu_put=False, max_pending_batches=2, defer_response=lambda *args: None)
                config = HSConfig(capture=True, cache=True, chunk_tokens=4,
                    cache_bytes=256, pinned_bytes=256, output_bytes=512,
                    device_bytes=256, attempt_capacity=2)
                kwargs = {"manifest": {"model": "threaded-cpu"}, "schema": _schema()}
                if initialize is not None:
                    config, kwargs = initialize(rank, config, kwargs)
                runtime = HSCaptureRuntime(scheduler, native, config, tp_size=2, **kwargs)
                if configure is not None:
                    configure(rank, runtime)
                results[rank] = operation(rank, runtime, phase)
            except BaseException as exc:
                errors[rank] = exc
            finally:
                if runtime is not None:
                    snapshots[rank] = {"samples": set(runtime.registry.active),
                                       "tokens": set(runtime.registry.tokens)}
                    try:
                        # Cleanup is local even when a protocol failure has made
                        # further collectives forbidden. Do not call control.
                        if runtime.controller is not None:
                            runtime.controller.shutdown(wait=True)
                    except Exception as exc:
                        if errors[rank] is None:
                            errors[rank] = exc

        threads = [threading.Thread(target=run, args=(rank,), daemon=True) for rank in (0, 1)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(15)
        self.assertFalse(any(thread.is_alive() for thread in threads), "scheduler CPU contract thread hung")
        self.assertIsNone(rendezvous.failed)
        self.assertEqual([item[0] for item in rendezvous.trace[0]],
                         [item[0] for item in rendezvous.trace[1]])
        self.assertEqual(len(rendezvous.trace[0]), len(rendezvous.trace[1]))
        self.assertFalse(torch.cuda.is_initialized())
        return results, errors, snapshots, rendezvous.trace

    def body(self, runtime, action="begin", sample="held", **values):
        return {"protocol_version": 1, "server_epoch": runtime.epoch, "action": action,
                "store_id": "store", "sample_id": sample, "gen": 1,
                "attempt_id": "attempt-" + sample, "request_digest": "digest",
                "publication_ownership": "attempt-v1", **values}

    def begin(self, runtime, phase):
        reply, failures = phase(lambda: runtime.control(runtime.s, self.body(runtime)))
        self.assertFalse(any(failures))
        return reply, runtime.registry.tokens[reply["reservation_token"]]

    def failed(self, result):
        values, errors, _, _ = result
        self.assertEqual(errors, [None, None])
        for value, failures in values:
            self.assertIsNone(value)
            self.assertTrue(all(isinstance(error, str) for error in failures))

    def test_control_poll_owner_and_writer_payload_failures_vote_before_broadcast(self):
        for fault in ("owner", "poll", "transfer-payload", "capability"):
            with self.subTest(fault=fault):
                def operation(rank, runtime, phase):
                    if fault == "owner" and rank == 1:
                        runtime._assert_owner = lambda: _raise("peer owner failure")
                    elif fault == "poll" and rank == 0:
                        runtime.controller.poll_completed = lambda: _raise("writer poll failure")
                    elif fault == "transfer-payload" and rank == 0:
                        self.fail_build(runtime, "control.transfers.payload")
                    elif fault == "capability" and rank == 0:
                        runtime.capability = lambda: _raise("writer capability failure")
                    return phase(lambda: runtime.control(runtime.s,
                        {"protocol_version": 1, "action": "capability"}))
                self.failed(self.pair(operation))

    @staticmethod
    def fail_build(runtime, target):
        original = runtime._writer_broadcast

        def broadcast(phase, build):
            return original(phase, lambda: _raise("writer payload failure") if phase == target else build())
        runtime._writer_broadcast = broadcast

    def test_transfer_peer_retirement_failure_votes_before_next_control_or_prepare_collective(self):
        for action in ("control", "prepare"):
            with self.subTest(action=action):
                def operation(rank, runtime, phase):
                    reply, entry = self.begin(runtime, phase)
                    if rank == 0:
                        entry.state = "PUBLISHED"
                        runtime.registry.retire(entry)
                        runtime._transferred_tokens.append(entry.token)
                    else:
                        runtime.registry.retire = lambda entry: _raise("peer transfer apply failed")
                    if action == "control":
                        return phase(lambda: runtime.control(runtime.s,
                            {"protocol_version": 1, "action": "capability"}))
                    req = SimpleNamespace(rid="r", sampling_params=SimpleNamespace(
                        custom_params={"spec_capture": {"protocol_version": 1}}))
                    return phase(lambda: runtime.prepare(runtime.s, req))
                self.failed(self.pair(operation))

    def test_control_lookup_and_active_retired_branch_disagreement_fail_together(self):
        for fault in ("missing", "active-retired", "request-action"):
            with self.subTest(fault=fault):
                def operation(rank, runtime, phase):
                    reply, entry = self.begin(runtime, phase)
                    if rank == 1:
                        if fault == "missing":
                            runtime.registry.tokens.pop(entry.token)
                        elif fault == "active-retired":
                            entry.state = "PUBLISHED"
                            runtime.registry.retire(entry)
                    body = {**self.body(runtime, "resolve"), "reservation_token": reply["reservation_token"]}
                    if rank == 1 and fault == "request-action":
                        body["action"] = "capability"
                    return phase(lambda: runtime.control(runtime.s, body))
                self.failed(self.pair(operation))

    def test_resolve_invalidate_abort_and_wire_errors_vote_before_later_broadcast(self):
        for fault in ("invalidate", "abort", "wire"):
            with self.subTest(fault=fault):
                def operation(rank, runtime, phase):
                    reply, entry = self.begin(runtime, phase)
                    req = SimpleNamespace(rid="request", _specforge_hs_attempt=entry)
                    entry.request = req
                    if fault == "invalidate" and rank == 0:
                        runtime.invalidate = lambda *args: _raise("writer invalidation failed")
                    elif fault == "abort" and rank == 1:
                        runtime.s.abort_request = lambda req: _raise("peer abort failed")
                    elif fault == "wire" and rank == 0:
                        runtime._wire = lambda entry: _raise("writer wire failed")
                    body = {**self.body(runtime, "resolve"),
                            "reservation_token": reply["reservation_token"], "cancel": fault != "wire"}
                    return phase(lambda: runtime.control(runtime.s, body))
                with mock.patch.dict(sys.modules, {"sglang.srt.managers.io_struct":
                        SimpleNamespace(AbortReq=lambda **kwargs: SimpleNamespace(**kwargs))}):
                    self.failed(self.pair(operation))

    def test_begin_preflight_never_emits_false_no_grant_on_divergence_or_partial_commit(self):
        for fault in ("tokens", "preflight", "reply", "commit", "capacity-disagreement"):
            with self.subTest(fault=fault):
                def operation(rank, runtime, phase):
                    if fault == "capacity-disagreement":
                        self.begin(runtime, phase)
                        self.control_success(runtime, phase, "other")
                        if rank == 0:
                            runtime.registry.active.clear()
                            runtime.registry.tokens.clear()
                    if fault == "tokens" and rank == 0:
                        self.fail_build(runtime, "control.begin.tokens")
                    elif fault == "preflight" and rank == 1:
                        runtime._begin_trial = lambda *args: _raise("peer preflight failed")
                    elif fault == "reply" and rank == 0:
                        runtime._wire = lambda entry: _raise("writer reply failed")
                    elif fault == "commit" and rank == 1:
                        original = runtime._local_phase
                        def local(phase, call):
                            return original(phase, lambda: _raise("peer commit failed")
                                            if phase == "control.begin.commit" else call())
                        runtime._local_phase = local
                    return phase(lambda: runtime.control(runtime.s, self.body(runtime, sample="new")))
                result = self.pair(operation)
                self.failed(result)
                snapshots = result[2]
                new = ("store", "new", 1)
                if fault == "commit":
                    self.assertIn(new, snapshots[0]["samples"])
                    self.assertNotIn(new, snapshots[1]["samples"])
                else:
                    self.assertTrue(all(new not in snapshot["samples"] for snapshot in snapshots))

    def control_success(self, runtime, phase, sample):
        reply, failures = phase(lambda: runtime.control(runtime.s, self.body(runtime, sample=sample)))
        self.assertFalse(any(failures))
        return reply

    def test_unanimous_capacity_decline_is_atomic_and_typed(self):
        def operation(rank, runtime, phase):
            self.begin(runtime, phase)
            self.control_success(runtime, phase, "other")
            return phase(lambda: runtime.control(runtime.s, self.body(runtime, sample="new")))
        values, errors, snapshots, _ = self.pair(operation)
        self.assertEqual(errors, [None, None])
        for reply, failures in values:
            self.assertFalse(any(failures))
            self.assertEqual(reply["error_code"], "capture_capacity_declined")
            self.assertIs(reply["reservation_granted"], False)
        self.assertTrue(all(len(snapshot["samples"]) == 2 for snapshot in snapshots))

    def test_read_cleanup_validation_and_existing_permit_branches_are_voted(self):
        for fault in ("read-validate", "read-branch", "read-token", "cleanup-validate",
                      "cleanup-branch", "cleanup-token", "cleanup-done", "cleanup-retire", "retired-read"):
            with self.subTest(fault=fault):
                def operation(rank, runtime, phase):
                    reply, entry = self.begin(runtime, phase)
                    body = {**self.body(runtime), "reservation_token": reply["reservation_token"]}
                    if fault.startswith("read"):
                        entry.state = "PUBLISHED"
                        body.update(action="read_acquire", reader_owner="owner")
                        if fault == "read-validate" and rank == 1:
                            entry.released_readers.append(("owner", "released-token"))
                        elif fault == "read-branch" and rank == 1:
                            entry.readers["owner"] = "existing-token"
                        elif fault == "read-token" and rank == 0:
                            self.fail_build(runtime, "control.read-acquire.token")
                    elif fault == "retired-read":
                        entry.state = "PUBLISHED"
                        runtime.registry.retire(entry)
                        if rank == 0:
                            entry.released_readers.append(("owner", "released-token"))
                        body.update(action="read_release", reader_owner="owner", reader_token="released-token")
                    else:
                        entry.revoked, entry.state = True, "WRITER_DRAINED"
                        body.update(action="cleanup_acquire", cleanup_owner="owner")
                        if fault == "cleanup-validate" and rank == 1:
                            entry.readers["reader"] = "reader-token"
                        elif fault == "cleanup-branch" and rank == 1:
                            entry.cleanup_owner, entry.cleanup_token = "owner", "existing-token"
                        elif fault == "cleanup-token" and rank == 0:
                            self.fail_build(runtime, "control.cleanup-acquire.token")
                        elif fault in ("cleanup-done", "cleanup-retire"):
                            entry.cleanup_owner, entry.cleanup_token = "owner", "cleanup-token"
                            body.update(action="cleanup_done", cleanup_token="cleanup-token")
                            if fault == "cleanup-done" and rank == 1:
                                entry.cleanup_token = "different-token"
                            elif fault == "cleanup-retire" and rank == 1:
                                runtime.registry.retire = lambda entry: _raise("peer retirement failed")
                    return phase(lambda: runtime.control(runtime.s, body))
                self.failed(self.pair(operation))

    def test_prepare_agrees_membership_and_votes_local_parse_and_lease_failures(self):
        for fault in ("membership", "parse", "lease"):
            with self.subTest(fault=fault):
                def operation(rank, runtime, phase):
                    tokens = list(range(8))
                    spec = {"features": {"aux": "aux", "last_hidden": "last"}, "passthrough": []}
                    body = self.body(runtime, request_digest=digest_request(tokens, spec))
                    reply, failures = phase(lambda: runtime.control(runtime.s, body))
                    self.assertFalse(any(failures))
                    spec.update({key: reply[key] for key in (
                        "server_epoch", "attempt_id", "reservation_token", "object_key_layout",
                        "store_id", "sample_id", "gen", "request_digest")})
                    spec.update(protocol_version=1, capture_namespace=runtime.namespace,
                                capture_signature=runtime.signature)
                    req = SimpleNamespace(rid="request", origin_input_ids=tokens,
                        sampling_params=SimpleNamespace(custom_params={"spec_capture": spec}, max_new_tokens=0))
                    if fault == "membership" and rank == 1:
                        req.sampling_params.custom_params = None
                    elif fault == "parse" and rank == 1:
                        req.sampling_params = None
                    elif fault == "lease" and rank == 0:
                        runtime.cache.lookup_and_pin = lambda *args: _raise("writer lease lookup failed")
                    outcome = phase(lambda: runtime.prepare(runtime.s, req))
                    runtime.rollback(runtime.s, req, None, "CPU contract teardown")
                    return outcome
                self.failed(self.pair(operation))

    def test_startup_epoch_model_identity_and_core_failures_terminate_both_ranks(self):
        for fault in ("epoch", "model", "identity", "core"):
            with self.subTest(fault=fault):
                original = HSCaptureRuntime._writer_broadcast
                def broadcast(runtime, phase, build):
                    return original(runtime, phase, lambda: _raise("writer epoch failure")
                                    if fault == "epoch" and phase == "startup.epoch" else build())
                def initialize(rank, config, kwargs):
                    if fault == "model":
                        kwargs = {"manifest": None, "schema": None}
                    elif fault == "identity" and rank == 1:
                        kwargs["schema"] = None
                        # Keep the identity-build failure after model resolution.
                        kwargs["schema"] = SimpleNamespace(features=(object(),))
                    elif fault == "core" and rank == 0:
                        config = dataclasses.replace(config, max_workers=0)
                    return config, kwargs
                def model(runtime):
                    if not runtime.writer:
                        _raise("peer backend capability failure")
                    return {"model": "threaded-cpu"}, _schema()
                with mock.patch.object(HSCaptureRuntime, "_writer_broadcast", broadcast), \
                     mock.patch.object(HSCaptureRuntime, "_model_identity", model):
                    values, errors, _, _ = self.pair(
                        lambda rank, runtime, phase: self.fail("failed startup returned a runtime"),
                        initialize=initialize)
                self.assertEqual(values, [None, None])
                self.assertTrue(all(error is not None for error in errors))

    def test_successful_permit_lifecycle_and_native_prepare_keep_phase_order(self):
        def operation(rank, runtime, phase):
            reply, entry = self.begin(runtime, phase)
            wire = {**self.body(runtime), "reservation_token": reply["reservation_token"]}
            entry.state = "PUBLISHED"
            read, failures = phase(lambda: runtime.control(runtime.s,
                {**wire, "action": "read_acquire", "reader_owner": "reader"}))
            self.assertFalse(any(failures))
            released, failures = phase(lambda: runtime.control(runtime.s,
                {**wire, "action": "read_release", "reader_owner": "reader", "reader_token": read["reader_token"]}))
            self.assertFalse(any(failures))
            entry.revoked, entry.state = True, "WRITER_DRAINED"
            cleanup, failures = phase(lambda: runtime.control(runtime.s,
                {**wire, "action": "cleanup_acquire", "cleanup_owner": "cleanup"}))
            self.assertFalse(any(failures))
            retired, failures = phase(lambda: runtime.control(runtime.s,
                {**wire, "action": "cleanup_done", "cleanup_token": cleanup["cleanup_token"]}))
            self.assertFalse(any(failures))
            req = SimpleNamespace(rid="native", sampling_params=SimpleNamespace(custom_params=None))
            value, failures = phase(lambda: runtime.prepare(runtime.s, req))
            self.assertFalse(any(failures))
            self.assertIsNone(value)
            return retired
        values, errors, snapshots, _ = self.pair(operation)
        self.assertEqual(errors, [None, None])
        self.assertTrue(all(value["state"] == "RETIRED" for value in values))
        self.assertTrue(all(not snapshot["tokens"] for snapshot in snapshots))


if __name__ == "__main__":
    unittest.main()
