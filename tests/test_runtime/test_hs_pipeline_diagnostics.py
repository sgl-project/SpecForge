"""CPU timing-boundary tests; these do not claim CUDA/RDMA performance."""
import os
import threading
import unittest
from unittest import mock

import torch

from tests.test_runtime import test_hs_plugin as fixture

_Store = fixture._Store
from specforge_sglang_capture.sink import SpecCaptureSink


class PipelineDiagnosticsTests(unittest.TestCase):
    setUp = fixture.HSPluginTests.setUp
    tearDown = fixture.HSPluginTests.tearDown
    runtime = fixture.HSPluginTests.runtime
    request = fixture.HSPluginTests.request
    admit = fixture.HSPluginTests.admit
    forward = fixture.HSPluginTests.forward
    spin = fixture.HSPluginTests.spin

    def test_default_off_preserves_response_and_does_not_record(self):
        with mock.patch.dict(os.environ, {"SPECFORGE_HS_DIAGNOSTICS": "0"}):
            hs = self.runtime()
            req = self.request(hs)
            self.admit(hs, req)
            output = self.forward(hs, [req], [8])
            self.spin(hs, lambda: hs.deferred[0][1].done())
            hs.deferred[0][1].result()
            self.assertIsNone(output.hs_jobs[0].diagnostic)
            self.assertFalse(hs.diagnostics)
            result = hs.capture.results[req.rid]
            self.assertFalse(any(k.startswith("p0_v1_") for k in result["timings"]))
            self.assertEqual(hs.controller.stats()["output_bytes"], 0)

    def test_sampled_boundaries_preserve_writer_drain_and_bounded_records(self):
        gate = threading.Event()
        store = _Store(gate=gate)
        with mock.patch.dict(os.environ, {"SPECFORGE_HS_DIAGNOSTICS": "1",
             "SPECFORGE_HS_DIAGNOSTICS_EVERY": "1", "SPECFORGE_HS_DIAGNOSTICS_MAX": "1"}):
            hs = self.runtime(store=store)
            reqs = [self.request(hs, name) for name in ("a", "b")]
            for req in reqs:
                self.admit(hs, req)
            output = self.forward(hs, reqs, [8, 8], wait_materialize=True)
            first, second = output.hs_jobs
            self.assertIs(first.sink, second.sink)
            self.assertTrue(store.entered.wait(2))
            try:
                self.assertFalse(first.sink.done())
                self.assertFalse(first.output.done())
                self.assertGreater(hs.controller.stats()["output_bytes"], 0)
                self.assertTrue(store.registered)
                self.assertNotIn("p0_v1_sink_done_ns", first.sink.p0_v1_timings)
            finally:
                gate.set()
            first.sink.result(timeout=3)
            self.assertFalse(store.registered)
            self.assertFalse(first.output.done())  # Native owner must still observe.
            hs.poll()
            for _, completion in hs.deferred:
                completion.result()
            diag, = hs.diagnostics
            self.assertIs(diag, first.diagnostic)
            self.assertIsNone(second.diagnostic)
            keys = ["p0_v1_sink_enqueue_ns", "p0_v1_sink_start_ns", "p0_v1_sink_done_ns",
                    "p0_v1_owner_observed_ns", "p0_v1_response_callback_ns"]
            self.assertEqual([diag[k] for k in keys], sorted(diag[k] for k in keys))
            self.assertLessEqual(diag["p0_v1_materialize_done_ns"], diag["p0_v1_runtime_sink_enqueue_ns"])
            self.assertGreaterEqual(diag["p0_v1_materialize_to_enqueue_ms"], 0)
            self.assertEqual(diag["p0_v1_sink_batch_samples"], 2)
            self.assertEqual(diag["p0_v1_consume_batch_samples"], 2)
            for name in ("register", "put", "unregister", "queue"):
                self.assertGreaterEqual(diag["p0_v1_sink_" + name + "_ms"], 0)
            self.assertGreater(diag["p0_v1_sink_registered_bytes"], 0)
            self.assertEqual(hs.controller.stats()["output_bytes"], 0)

    def test_sink_queue_is_before_writer_start_and_metadata_has_no_tensors(self):
        with mock.patch.dict(os.environ, {"SPECFORGE_HS_DIAGNOSTICS": "1"}):
            writer = SpecCaptureSink()
        gate, started = threading.Event(), threading.Event()
        self.addCleanup(writer.close)
        def block():
            started.set()
            self.assertTrue(gate.wait(3))
        blocker = writer._executor.submit(block)
        self.assertTrue(started.wait(2))
        writer._store = _Store()
        spec = {"store_id": "test", "sample_id": "a", "features": {"aux": "hidden"}}
        future = writer.submit_samples([(spec, torch.ones(2, 2), None)])
        try:
            self.assertNotIn("p0_v1_sink_start_ns", future.p0_v1_timings)
        finally:
            gate.set()
        blocker.result(timeout=3)
        future.result(timeout=3)
        d = future.p0_v1_timings
        self.assertLessEqual(d["p0_v1_sink_enqueue_ns"], d["p0_v1_sink_start_ns"])
        self.assertLessEqual(d["p0_v1_sink_start_ns"], d["p0_v1_sink_done_ns"])
        self.assertTrue(all(type(value) in (int, float, bool) for value in d.values()))


if __name__ == "__main__":
    unittest.main()
