"""CPU native-output contracts; real DMA/model/TP validation remains separate."""

import ctypes
import dataclasses
import json
import os
import sys
import threading
import time
import tempfile
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "plugins", "sglang-spec-capture"))

from specforge_sglang_capture import capture, sink
from specforge_sglang_capture.hs.attempts import AttemptCapacityDeclined, AttemptRegistry, digest_request
from specforge_sglang_capture.hs.config import HSConfig
from specforge_sglang_capture.hs.protocol import FeatureSchema, FeatureSpec
from specforge_sglang_capture.hs.runtime import HSCaptureRuntime
from specforge.inference.capture_protocol import CaptureCapacityDeclined, CaptureProtocol


@dataclasses.dataclass
class _Proposal:
    max_prefix_len: int
    identity: tuple
    opaque: object = None


class _Store:
    def __init__(self, gate=None, status=0):
        self.values, self.registered = {}, set()
        self.gate, self.status = gate, status
        self.entered = threading.Event()

    def register_buffer(self, ptr, size):
        self.registered.add(ptr)
        return 0

    def unregister_buffer(self, ptr):
        self.registered.remove(ptr)
        return 0

    def batch_put_from(self, keys, pointers, sizes, config):
        self.entered.set()
        if self.gate is not None and not self.gate.wait(5):
            raise TimeoutError("test writer gate did not open")
        for key, ptr, size in zip(keys, pointers, sizes):
            self.values[key] = ctypes.string_at(ptr, size)
        return [self.status] * len(keys)

    def remove(self, key, *, force=False):
        self.values.pop(key, None)
        return 0


def _schema():
    return FeatureSchema((FeatureSpec("aux", 2, "float32", (1, 3)),
                          FeatureSpec("last_hidden", 2, "float32", ())))


class HSPluginTests(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {"SGLANG_SPEC_CAPTURE_GPU_PUT": "0", "MOONCAKE_PROTOCOL": "tcp"})
        self.env.start()
        self.protocol_module = mock.patch.dict(sys.modules, {
            "sglang.srt.managers.prefix_admission": SimpleNamespace(PrefixAdmissionProposal=_Proposal)})
        self.protocol_module.start()
        self.runtimes = []

    def tearDown(self):
        for runtime in self.runtimes:
            if runtime.quarantined_publications or runtime.quarantined_snapshots:
                runtime.capture.sink._executor.shutdown(wait=True)
                runtime.controller._executor.shutdown(wait=True)
            else:
                runtime.shutdown()
        self.protocol_module.stop()
        self.env.stop()

    def runtime(self, *, store=None, cache=True, capacity=16, parallel=None, segmented=False):
        writer = sink.SpecCaptureSink([1, 3])
        writer._store, writer._put_config = store or _Store(), object()
        if segmented:
            from specforge.runtime.data_plane.mooncake_store import _InjectedReplicateConfig
            writer._put_config = _InjectedReplicateConfig()
        deferred = []
        native = capture.CaptureRuntime(sink=writer, is_writer=True, gpu_put=False,
            max_pending_batches=2, defer_response=lambda req, completion: deferred.append((req, completion)))
        scheduler = SimpleNamespace(abort_request=lambda req: None)
        config = HSConfig(capture=True, cache=cache, segmented=segmented, chunk_tokens=4, cache_bytes=4096,
            pinned_bytes=4096, output_bytes=8192, device_bytes=4096, attempt_capacity=capacity)
        kwargs = dict(manifest={"model": "cpu-fixture", "weights": "load-epoch"}, schema=_schema())
        if parallel is None:
            hs = HSCaptureRuntime(scheduler, native, config, tp_size=1, **kwargs)
        else:
            getter = mock.Mock(return_value=parallel)
            with mock.patch.dict(sys.modules, {"sglang.srt.runtime_context": SimpleNamespace(get_parallel=getter)}):
                hs = HSCaptureRuntime(scheduler, native, config, **kwargs)
            getter.assert_called_once_with()
        native.hs = hs
        hs.deferred = deferred
        self.runtimes.append(hs)
        return hs

    def request(self, hs, sample="s", *, tokens=tuple(range(8)), mask=1, layout="hs-manifest-v1"):
        spec = {"store_id": "store", "sample_id": sample, "gen": 1, "replace": True,
                "features": {"aux": "hidden_states", "last_hidden": "target"},
                "passthrough": [{"name": "loss_mask", "data": [[mask] * len(tokens)],
                                  "shape": [1, len(tokens)], "dtype": "int64"}]}
        if hs.config.segmented:
            spec["feature_layout"] = layout
        digest = digest_request(tokens, spec)
        body = {"protocol_version": 1, "action": "begin", "server_epoch": hs.epoch,
                "store_id": "store", "sample_id": sample, "gen": 1,
                "attempt_id": "attempt-" + sample, "request_digest": digest,
                "publication_ownership": "attempt-v1"}
        reply = hs.control(hs.s, body)
        spec.update({key: reply[key] for key in ("server_epoch", "attempt_id", "reservation_token", "object_key_layout")})
        spec.update(protocol_version=1, capture_namespace=hs.namespace,
                    capture_signature=hs.signature, request_digest=digest)
        return SimpleNamespace(rid="req-" + sample, origin_input_ids=list(tokens),
            sampling_params=SimpleNamespace(custom_params={"spec_capture": json.dumps(spec)}, max_new_tokens=0),
            finished=lambda: True)

    def admit(self, hs, req, lawful=None):
        proposal = hs.prepare(hs.s, req)
        L = proposal.max_prefix_len if lawful is None else lawful
        hs.after_match(hs.s, req, proposal, L)
        self.assertTrue(hs.reserve(hs.s, req, proposal, L))
        hs.commit(hs.s, req, proposal, L)
        return proposal, L

    def forward(self, hs, reqs, lengths, *, wait_materialize=False):
        aux, last, prefixes = [], [], []
        for req, length in zip(reqs, lengths):
            L = req._specforge_hs_plan.L
            rows = torch.arange(L * 2, (L + length) * 2, dtype=torch.float32).view(length, 2)
            aux.append(rows)
            last.append(rows + 1000)
            prefixes.append(L)
        batch = SimpleNamespace(reqs=reqs)
        forward = SimpleNamespace(extend_seq_lens_cpu=lengths, extend_prefix_lens_cpu=prefixes,
                                  forward_mode=SimpleNamespace(is_extend=lambda: True))
        logits = SimpleNamespace(hidden_states=torch.cat(aux), last_hidden_states=torch.cat(last),
                                 hidden_states_token_indices=None)
        output = hs.capture.observer.after_forward(batch, forward, logits, can_run_graph=True)
        with mock.patch.object(torch.Tensor, "to", side_effect=AssertionError("native D2H is duplicated")):
            host = output.copy_to_host(lambda tensor: self.fail("HS must not call native copy_tensor"))
        if wait_materialize:
            self.spin(hs, lambda: all(job.materialize.done() for job in output.hs_jobs))
        host.consume(batch, [None] * len(reqs))
        return output

    def spin(self, hs, condition):
        deadline = time.monotonic() + 5
        while not condition():
            hs.poll()
            if time.monotonic() > deadline:
                self.fail("HS pipeline did not make progress")
            time.sleep(0.001)
        hs.poll()

    def test_segmented_runtime_and_explicit_full_fallback_use_owner_and_compact_cache(self):
        from tests.test_runtime.test_mooncake_store import _FakeMooncakeStore
        for force_full in ('0', '1'):
            with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {
                'SPECFORGE_HS_REMOTE_LEDGER_DIR': directory,
                'SPECFORGE_HS_REMOTE_BUDGET_BYTES': str(1 << 24),
                'SPECFORGE_HS_SEGMENT_FORCE_FULL': force_full,
            }):
                backend = _FakeMooncakeStore()
                hs = self.runtime(store=backend, segmented=True)
                req = self.request(hs, 'segment-' + force_full, tokens=tuple(range(9)))
                self.admit(hs, req)
                self.forward(hs, [req], [9])
                self.spin(hs, lambda: hs.deferred[0][1].done())
                hs.deferred[0][1].result()
                result = hs.capture.results[req.rid]
                self.assertEqual(result['feature_layout'], 'hs-manifest-v1')
                self.assertEqual(result['full_object_fallback'], force_full == '1')
                manifest = result['segment_manifest']
                self.assertEqual(len(manifest['features']['hidden_states']), 1 if force_full == '1' else 3)
                self.assertEqual(hs.cache.stats()['cache_adopted_samples'], 0)
                hs.capture.sink._managed_put({manifest['ack_key']: torch.tensor(list(bytes.fromhex(manifest['digest'])), dtype=torch.uint8)})
                owner = hs.capture.sink._manifest_owner
                owner.progress()
                self.assertTrue(backend.is_exist(manifest['receipt_key']))
                backend.remove(manifest['receipt_key'])
                owner.progress()
                self.assertEqual(owner.health()['remote_owner_charged_bytes'], 0)
                self.assertFalse(backend._d)
                hs.shutdown()
                self.runtimes.remove(hs)

    def test_publication_trace_matches_reclamation_and_sibling(self):
        from pathlib import Path
        from tests.test_runtime.test_mooncake_store import _FakeMooncakeStore
        from specforge.runtime.data_plane.publication_timing import IDENTITY

        deleting, unlock, trying, second_put, finish = (threading.Event() for _ in range(5))
        class Backend(_FakeMooncakeStore):
            def remove(self, key, **kwargs):
                if '/seed/' in key and not deleting.is_set():
                    deleting.set()
                    if not unlock.wait(8):
                        raise TimeoutError('fixture cleanup gate')
                return super().remove(key)
            def put_from(self, key, *args):
                if '/second/' in key and not second_put.is_set():
                    second_put.set()
                    if not finish.wait(8):
                        raise TimeoutError('fixture sibling gate')
                return super().put_from(key, *args)
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, {
            'SPECFORGE_HS_REMOTE_LEDGER_DIR':root, 'SHORT_TEACHER_RUN_DIR':root,
            'SPECFORGE_HS_REMOTE_BUDGET_BYTES':str(1<<24), 'SPECFORGE_HS_PUBLISH_TIMING':'1'}):
            backend=Backend()
            hs=self.runtime(store=backend,segmented=True)
            def forward(names):
                reqs=[self.request(hs,n,tokens=tuple(range(9)),layout='hs-prefix-v2') for n in names]
                for req in reqs:
                    self.admit(hs,req,lawful=0)
                self.forward(hs,reqs,[9]*len(reqs),wait_materialize=True)
                hs.poll()
                return reqs
            seed,=forward(['seed'])
            self.spin(hs,lambda:hs.deferred[0][1].done())
            owner=hs.capture.sink._manifest_owner
            real_lock=owner.lock
            class NotifyLock:
                def __enter__(self):
                    if deleting.is_set() and threading.current_thread().name.startswith('spec-capture-batch-put'):
                        trying.set()
                    return real_lock.__enter__()
                def __exit__(self,*args):
                    return real_lock.__exit__(*args)
            owner.lock=NotifyLock()
            def ack(req):
                next(f for r,f in hs.deferred if r is req).result()
                manifest=hs.capture.results[req.rid]['segment_manifest']
                hs.capture.sink._managed_put({manifest['ack_key']:torch.tensor(list(bytes.fromhex(manifest['digest'])),dtype=torch.uint8)})
                return manifest['receipt_key']
            receipts=[ack(seed)]
            try:
                self.assertTrue(deleting.wait(5))
                first,second=forward(['first','second'])
                self.assertTrue(trying.wait(5))
                unlock.set()
                self.assertTrue(second_put.wait(5))
                self.assertFalse(hs.deferred[1][1].done())
                self.assertFalse(hs.deferred[2][1].done())
            finally:
                unlock.set()
                finish.set()
            self.spin(hs,lambda:all(f.done() for _,f in hs.deferred))
            receipts.extend(ack(req) for req in (first,second))
            owner.progress()
            for key in receipts:
                self.assertTrue(backend.is_exist(key))
                backend.remove(key)
            owner.progress()
            hs.shutdown()
            self.runtimes.remove(hs)
            path,=Path(root).glob('publication-timing-*.jsonl')
            packets=[json.loads(line) for line in path.read_text().splitlines()]
            self.assertTrue(packets[-1]['final'])
            self.assertTrue(all(p['dropped']==0 for p in packets))
            rows=[r for p in packets for r in p['rows']]
            holds={r['identity']['sample_id']:r for r in rows if r.get('phase')=='publish'}
            one,two=holds['first'],holds['second']
            batch=one['sample'][0]
            self.assertEqual(one['sample'],[batch,0])
            self.assertEqual(two['sample'],[batch,1])
            for req,h in [(first,one),(second,two)]:
                result=hs.capture.results[req.rid]
                self.assertEqual(h['identity'],{k:result[k] for k in IDENTITY})
                self.assertEqual(h['owner'],result['segment_manifest']['owner_epoch'])
                for name in ('ledger_reserve','ledger_state','native_register','native_put','native_unregister'):
                    self.assertGreater(h['component_ns'][name],0)
                obs,=[r for r in rows if r['event']=='observed' and r['identity']==h['identity']]
                self.assertEqual([obs['batch'],obs['index']],h['sample'])
            self.assertLessEqual(one['end_ns'],two['acquired_ns'])
            returned,=[r['ns'] for r in rows if r['event']=='batch_return' and r['batch']==batch]
            self.assertGreaterEqual(returned,two['end_ns'])
            self.assertTrue(any(r.get('phase')=='progress' and r['owner']==one['owner']
                                and r['acquired_ns']<=one['attempt_ns']<r['end_ns']<=one['acquired_ns']
                                and r['component_ns'].get('native_remove',0)>0
                                and r['component_ns'].get('ledger_forget',0)>0 for r in rows))
            if os.environ.get('CPU_TIMING_EXAMPLE'):
                Path(os.environ['CPU_TIMING_EXAMPLE']).write_bytes(path.read_bytes())

    def test_publication_trace_is_off_by_default(self):
        with mock.patch.dict(os.environ):
            os.environ.pop('SPECFORGE_HS_PUBLISH_TIMING',None)
            writer=sink.SpecCaptureSink([1,3])
            self.assertIsNone(writer._publish_trace)
            writer.close()

    def test_publication_trace_full_buffer_reports_loss(self):
        from specforge.runtime.data_plane.publication_timing import PublicationTrace
        with tempfile.TemporaryDirectory() as root:
            trace=PublicationTrace(root,limit=0)
            trace.emit('enqueue',batch='discarded',count=1,ns=time.monotonic_ns())
            trace.close()
            packet=json.loads(trace.path.read_text())
            self.assertTrue(packet['final'])
            self.assertEqual(packet['dropped'],1)
            self.assertEqual(packet['rows'],[])

    def test_committed_proposal_H_survives_shrink_into_wire_manifest(self):
        from tests.test_runtime.test_mooncake_store import _FakeMooncakeStore
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, {
            'SPECFORGE_HS_REMOTE_LEDGER_DIR':root, 'SPECFORGE_HS_REMOTE_BUDGET_BYTES':str(1<<24),
            'SPECFORGE_HS_SEGMENT_FORCE_FULL':'0'}):
            backend = _FakeMooncakeStore()
            hs = self.runtime(store=backend, segmented=True)
            self.assertIn('hs-prefix-v2', hs.capability()['feature_layouts'])
            for i, (h, l) in enumerate([(0,0), (8,5)]):
                req = self.request(hs, 'v2-'+str(i), tokens=tuple(range(9)), layout='hs-prefix-v2')
                proposal,_ = self.admit(hs, req, lawful=l)
                self.assertEqual(proposal.max_prefix_len,h)
                self.forward(hs,[req],[9-l])
                self.spin(hs,lambda:hs.deferred[i][1].done())
                hs.deferred[i][1].result()
                result = hs.capture.results[req.rid]
                m = result['segment_manifest']
                self.assertEqual((m['H'],m['L'],m['N'],m['layout']),(h,l,9,'hs-prefix-v2'))
                hs.capture.sink._managed_put({m['ack_key']:torch.tensor(list(bytes.fromhex(m['digest'])),dtype=torch.uint8)})
                owner = hs.capture.sink._manifest_owner
                owner.progress()
                backend.remove(m['receipt_key'])
                owner.progress()
                self.assertEqual(owner.health()['remote_owner_charged_bytes'],0)
            hs.shutdown()
            self.runtimes.remove(hs)

    def test_native_complete_n_rows_from_nonaligned_prefix_and_loss_mask_independent_identity(self):
        hs = self.runtime(parallel=SimpleNamespace(tp_size=1, tp_rank=0))
        first = self.request(hs, "cold")
        self.admit(hs, first)
        self.forward(hs, [first], [8])
        self.spin(hs, lambda: hs.deferred[0][1].done())
        hs.deferred[0][1].result()
        cold = hs.capture.results[first.rid]
        self.assertEqual(cold["state"], "PUBLISHED")
        self.assertEqual(cold["rank"], hs.tp_rank)
        self.assertEqual(cold["publication_certificate"]["rank"], hs.tp_rank)
        self.assertTrue(cold["writer_drained"])
        self.assertEqual(cold["request_digest"], digest_request(first.origin_input_ids, capture.request_spec(first)))
        second = self.request(hs, "warm", mask=0)
        proposal, L = self.admit(hs, second, lawful=3)
        self.assertEqual(proposal.max_prefix_len, 4)
        self.assertEqual(L, 3)
        self.forward(hs, [second], [5])
        self.spin(hs, lambda: hs.deferred[1][1].done())
        hs.deferred[1][1].result()
        warm = hs.capture.results[second.rid]
        self.assertEqual((warm["N"], warm["L"], warm["span"]), (8, 3, [3, 8]))
        self.assertNotEqual(cold["request_digest"], warm["request_digest"])
        values = hs.capture.sink._store.values
        for name in ("hidden_states", "target"):
            self.assertEqual(values[cold["physical_keys"][name]], values[warm["physical_keys"][name]])
        self.assertEqual(hs.controller.stats()["output_bytes"], 0)
        self.assertGreater(hs.cache.stats()["cache_owned_bytes"], 0)

    def test_cancel_waits_for_shared_sink_and_cannot_delete_sibling_attempt(self):
        gate = threading.Event()
        store = _Store(gate)
        hs = self.runtime(store=store)
        reqs = [self.request(hs, name) for name in ("cancel", "sibling")]
        for req in reqs:
            self.admit(hs, req)
        output = self.forward(hs, reqs, [8, 8], wait_materialize=True)
        first, sibling = output.hs_jobs
        self.assertIs(first.sink, sibling.sink)
        self.assertIsNotNone(first.sink)
        self.assertEqual((first.sink_count, sibling.sink_count), (2, 2))
        self.spin(hs, store.entered.is_set)
        held = first.sample
        hs.deferred[0][1].cancel()
        self.assertTrue(hs.deferred[0][1].done())
        self.assertIs(first.sample, held)
        self.assertFalse(first.sink.done())
        self.assertGreater(hs.controller.stats()["output_bytes"], 0)
        gate.set()
        self.spin(hs, lambda: not hs.jobs)
        hs.deferred[1][1].result()
        result = hs.capture.results[reqs[1].rid]
        self.assertTrue(all(key in store.values for key in result["physical_keys"].values()))
        cancelled = capture.request_spec(reqs[0])
        self.assertFalse(any(cancelled["reservation_token"] in key for key in store.values))
        self.assertEqual(hs.controller.stats()["output_bytes"], 0)

    def test_unknown_native_write_retains_sources_registration_and_output_charge(self):
        store = _Store(status=-800)
        hs = self.runtime(store=store)
        req = self.request(hs)
        self.admit(hs, req)
        self.forward(hs, [req], [8])
        self.spin(hs, lambda: bool(hs.quarantined_publications))
        self.assertGreater(hs.controller.stats()["output_bytes"], 0)
        self.assertTrue(store.registered)
        self.assertFalse(req._specforge_hs_attempt.transfer_certificate)
        self.assertTrue(hs.capture.sink._uncertain_publications)
        with self.assertRaises(RuntimeError):
            hs.shutdown()

    def test_cache_off_preserves_attempt_keys_full_rows_and_zero_reuse(self):
        hs = self.runtime(cache=False)
        for sample in ("a", "b"):
            req = self.request(hs, sample)
            proposal, _ = self.admit(hs, req)
            self.assertEqual(proposal.max_prefix_len, 0)
            self.forward(hs, [req], [8])
            self.spin(hs, lambda: hs.deferred[-1][1].done())
            hs.deferred[-1][1].result()
            self.assertEqual(hs.capture.results[req.rid]["span"], [0, 8])
            self.assertEqual(hs.capture.results[req.rid]["rank"], hs.tp_rank)
        self.assertFalse(hs.capability()["cache_enabled"])
        self.assertEqual(hs.cache.stats()["cache_owned_bytes"], 0)

    def test_rollback_before_and_after_freeze_releases_only_its_candidate(self):
        hs = self.runtime()
        req = self.request(hs)
        proposal = hs.prepare(hs.s, req)
        hs.after_match(hs.s, req, proposal, 0)
        self.assertTrue(hs.reserve(hs.s, req, proposal, 0))
        hs.rollback(hs.s, req, proposal, "native-batch-allocation-failed")
        self.assertEqual(hs.controller.stats()["output_bytes"], 0)
        self.assertEqual(hs.cache.stats()["active_leases"], 0)
        next_proposal, _ = self.admit(hs, req)
        self.assertGreater(next_proposal.identity[3], proposal.identity[3])
        hs.rollback(hs.s, req, next_proposal, "peer-commit-failed")
        self.assertEqual(hs.controller.stats()["output_bytes"], 0)
        self.assertEqual(hs.cache.stats()["active_leases"], 0)

    def test_owner_thread_and_digest_fail_closed_before_reservation(self):
        hs = self.runtime()
        req = self.request(hs)
        req.origin_input_ids[0] = 99
        with self.assertRaises(ValueError):
            hs.prepare(hs.s, req)
        self.assertEqual(hs.controller.stats()["output_bytes"], 0)
        errors = []
        thread = threading.Thread(target=lambda: self._record_error(errors, hs))
        thread.start()
        thread.join()
        self.assertEqual(len(errors), 1)
        self.assertIn("scheduler thread", str(errors[0]))

    @staticmethod
    def _record_error(errors, hs):
        try:
            hs.control(hs.s, {"protocol_version": 1, "action": "capability"})
        except Exception as exc:
            errors.append(exc)

    def test_begin_batch_capacity_is_atomic_and_exposes_only_ungranted_backpressure(self):
        hs = self.runtime(capacity=2)
        self.request(hs, "held")
        task = lambda name: {"server_epoch": hs.epoch, "store_id": "store", "sample_id": name,
            "gen": 1, "attempt_id": "a-" + name, "request_digest": "digest",
            "publication_ownership": "attempt-v1"}
        before = dict(hs.registry.active)
        reply = hs.control(hs.s, {"protocol_version": 1, "action": "begin_batch",
            "server_epoch": hs.epoch, "tasks": [task("new1"), task("new2")]})
        self.assertEqual(reply["error_code"], "capture_capacity_declined")
        self.assertIs(reply["reservation_granted"], False)
        self.assertEqual(hs.registry.active, before)
        protocol = CaptureProtocol("http://fixture", lambda *args, **kwargs: reply, 1)
        with self.assertRaises(CaptureCapacityDeclined):
            protocol.call({"action": "begin_batch"})
        with self.assertRaises(RuntimeError) as failure:
            protocol.call({"action": "resolve"})
        self.assertNotIsInstance(failure.exception, CaptureCapacityDeclined)

    def test_old_attempt_cleanup_never_targets_new_server_keys(self):
        store = _Store()
        first, second = self.runtime(store=store), self.runtime(store=store)
        results = []
        specs = []
        for hs in (first, second):
            req = self.request(hs, "same-sample")
            specs.append(capture.request_spec(req))
            self.admit(hs, req)
            self.forward(hs, [req], [8])
            self.spin(hs, lambda: hs.deferred[0][1].done())
            hs.deferred[0][1].result()
            results.append(hs.capture.results[req.rid])
        first.capture.sink.discard_sample(specs[0]).result(5)
        self.assertTrue(all(key in store.values for key in results[1]["physical_keys"].values()))
        self.assertTrue(all(key not in store.values for key in results[0]["physical_keys"].values()))


if __name__ == "__main__":
    unittest.main()
