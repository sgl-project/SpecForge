"""MTP feature travels through the plugin with stable per-request alignment."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tests.test_runtime.test_sglang_capture_plugin import (
    _forward_batch,
    _host_copy,
    _logits_output,
    _req,
    _runtime,
    _spec,
)


class DMTPCaptureTest(unittest.TestCase):
    def test_sink_publishes_mtp_feature_and_alignment(self):
        from specforge_sglang_capture.sink import SpecCaptureSink

        from tests.test_runtime.test_sglang_capture_plugin_sink import _BufferStore

        sink = SpecCaptureSink([1])
        self.addCleanup(sink._executor.shutdown)
        store = _BufferStore()
        # Follow the sink's normal connection seam without a real master.
        spec = _spec("s")
        spec["features"] = {"mtp_hidden": "mtp_hidden"}
        hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        with patch.object(sink, "_connect", return_value=store):
            result = sink.put_samples([(spec, None, None, hidden)])[0]
        self.assertEqual(result["mtp_hidden_alignment"], "anchor_post_norm_v1")
        self.assertEqual(result["features"]["mtp_hidden"]["shape"], [1, 3, 4])
        self.assertEqual(next(iter(store.values.values())), hidden.numpy().tobytes())

    def test_mixed_requests_keep_mtp_rows_and_copy_ownership(self):
        runtime = _runtime()
        calls = []

        def teacher(ids, hidden):
            calls.append(ids.clone())
            return hidden + 100

        runtime.mtp_capture = teacher
        spec = _spec("dmtp")
        spec["features"]["mtp_hidden"] = "mtp_hidden"
        batch = SimpleNamespace(reqs=[_req("normal", 2), _req("dmtp", 3, spec)])
        logits = _logits_output(5)
        output = runtime.observer.after_forward(
            batch, _forward_batch([2, 3]), logits, can_run_graph=True
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].tolist(), [0, 1, 2])
        expected = logits.last_hidden_states[2:].clone() + 100
        logits.last_hidden_states.zero_()
        output.copy_to_host(_host_copy).consume(batch, [])
        samples, _ = runtime.sink.submissions[0]
        torch.testing.assert_close(samples[0][3], expected)

    def test_missing_mtp_capability_fails_request(self):
        runtime = _runtime()
        spec = _spec("dmtp")
        spec["features"]["mtp_hidden"] = "mtp_hidden"
        batch = SimpleNamespace(reqs=[_req("dmtp", 3, spec)])
        output = runtime.observer.after_forward(
            batch, _forward_batch([3]), _logits_output(3), can_run_graph=False
        )
        output.copy_to_host(_host_copy).consume(batch, [])
        self.assertIn("mtp_hidden", str(runtime.results["dmtp"]))
        self.assertEqual(runtime.sink.submissions, [])


if __name__ == "__main__":
    unittest.main()
