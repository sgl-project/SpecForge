"""Attempt-fence integration through real adapter and separate FeatureStores."""

import unittest
from unittest import mock

import torch

from specforge.inference.adapters.server_capture import (
    SGLangServerCaptureAdapter as _NativeAdapter,
)
from specforge.inference.capture_protocol import CaptureProtocol


def SGLangServerCaptureAdapter(*args, **kwargs):
    adapter = _NativeAdapter(*args, **kwargs)
    adapter._capture_protocol = CaptureProtocol(
        adapter.base_url, kwargs["control_fn"], adapter.timeout_s
    )
    return adapter


from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
from tests.test_runtime.capture_fixtures import ProtocolServer
from tests.test_runtime.test_server_capture import (
    _capture_schema,
    _eagle3_contract,
    _FakeMooncakeStore,
    _task,
)


class FenceTests(unittest.TestCase):
    def setUp(self):
        self.backend = _FakeMooncakeStore()
        self.server = ProtocolServer(self.backend)
        self.producer = MooncakeFeatureStore(store=self.backend, store_id="run0")
        self.adapter = SGLangServerCaptureAdapter(
            "http://server",
            self.producer,
            run_id="run0",
            algorithm="eagle3",
            schema=_capture_schema("eagle3"),
            post_fn=self.server,
            control_fn=self.server,
            timeout_s=0.01,
        )

    def produce(self, index=0):
        return self.adapter.produce_refs([_task(index, 5)], capture=_eagle3_contract())[
            0
        ]

    def test_separate_consumer_release_carries_fence_and_retires_registry(self):
        ref = self.produce()
        self.assertIn("capture_fence", ref.metadata)
        consumer = MooncakeFeatureStore(store=self.backend, store_id="run0")
        with mock.patch(
            "specforge.inference.capture_protocol.http_control", self.server
        ):
            tensors, handle = consumer.get(ref)
            expected = self.server.stub.expected[ref.sample_id]
            self.assertEqual(set(tensors), set(expected))
            for name, tensor in expected.items():
                torch.testing.assert_close(tensors[name], tensor, rtol=0, atol=0)
            consumer.release(handle)
        self.assertFalse(self.backend._d)
        self.assertFalse(self.server.active)
        self.assertIn("cleanup_done", self.server.events)
        with mock.patch(
            "specforge.inference.capture_protocol.http_control", self.server
        ):
            with self.assertRaises(KeyError):
                consumer.get(ref)

    def test_timeout_retains_keys_until_writer_drain(self):
        payload = self.adapter._spec_capture_payload(_task(0, 5))
        wire = self.adapter._capture_protocol.begin(
            self.server.capability, payload, [1] * 5, "first"
        )
        self.producer.track_capture_fence(
            wire, ["hidden_state"], control_fn=self.server
        )
        self.producer.track_external_attempt(
            payload["sample_id"], generation=1, feature_names=["hidden_state"]
        )
        self.backend._d[f"run0/{payload['sample_id']}/g1/hidden_state"] = b"old"
        self.server.drained = False
        with self.assertRaises(TimeoutError):
            self.producer.prepare_capture_retry(payload["sample_id"])
        self.assertTrue(self.backend._d)
        self.assertTrue(self.producer._external_provisional)
        self.server.drained = True
        self.producer.prepare_capture_retry(payload["sample_id"])
        self.assertFalse(self.backend._d)
        self.assertFalse(self.producer._capture_fences)

    def test_old_cleanup_and_duplicate_release_cannot_delete_new_g1(self):
        ref = self.produce()
        old_token = ref.metadata["capture_fence"]["reservation_token"]
        values, handle = self.producer.get(ref)
        self.producer.release(handle)
        new_ref = self.produce()
        new_token = new_ref.metadata["capture_fence"]["reservation_token"]
        self.assertNotEqual(old_token, new_token)
        before = dict(self.backend._d)
        self.producer.abort(ref.sample_id, attempt_token=old_token)
        self.producer.release(handle)
        self.assertEqual(before, self.backend._d)
        with self.assertRaises(KeyError):
            self.producer.get(ref)
        values, new_handle = self.producer.get(new_ref)
        self.producer.release(new_handle)
        self.assertFalse(self.backend._d)

    def test_old_producer_gc_cannot_remove_new_cross_process_g1(self):
        ref = self.produce()
        consumer = MooncakeFeatureStore(store=self.backend, store_id="run0")
        with mock.patch(
            "specforge.inference.capture_protocol.http_control", self.server
        ):
            _, handle = consumer.get(ref)
            consumer.release(handle)
        replacement_store = MooncakeFeatureStore(store=self.backend, store_id="run0")
        replacement_adapter = SGLangServerCaptureAdapter(
            "http://server",
            replacement_store,
            run_id="run0",
            algorithm="eagle3",
            schema=_capture_schema("eagle3"),
            post_fn=self.server,
            control_fn=self.server,
        )
        replacement_adapter.produce_refs([_task(0, 5)], capture=_eagle3_contract())
        before = dict(self.backend._d)
        self.producer.max_hold_age_s = 0
        self.producer.gc(now=10**12)
        self.assertEqual(before, self.backend._d)
        self.assertFalse(self.producer._capture_fences)

    def test_lost_begin_response_recovers_same_intent_and_cleans(self):
        original = self.adapter._capture_protocol.post
        lost = [False]

        def transport(url, json_body, timeout):
            reply = original(url, json_body, timeout)
            if json_body.get("action") == "begin" and not lost[0]:
                lost[0] = True
                raise TimeoutError("begin response lost after reservation")
            return reply

        self.adapter._capture_protocol.post = transport
        with self.assertRaises(TimeoutError):
            self.produce()
        self.assertTrue(self.producer._capture_pending_begin)
        self.assertTrue(self.server.active)
        self.producer.discard_external_attempts()
        self.assertFalse(self.producer._capture_pending_begin)
        self.assertFalse(self.server.active)
        self.assertFalse(self.producer._capture_fences)

    def test_lost_read_acquire_reply_recovers_intent_before_cleanup(self):
        ref = self.produce()
        fence = self.producer._capture_fences[ref.sample_id]
        original = fence.post
        lost = [True]

        def transport(url, json_body, timeout):
            reply = original(url, json_body, timeout)
            if json_body.get("action") == "read_acquire" and lost[0]:
                lost[0] = False
                raise TimeoutError("read acquire reply lost")
            return reply

        fence.post = transport
        with self.assertRaises(TimeoutError):
            self.producer.get(ref)
        self.assertEqual(len(self.server.active[ref.sample_id]["readers"]), 1)
        self.assertEqual(self.producer.health()["capture_read_intents"], 1)
        self.producer.drain_pending_removals(max_attempts=2, retry_interval_s=0)
        self.assertEqual(len(self.server.active[ref.sample_id]["readers"]), 0)
        self.assertEqual(self.producer.health()["capture_read_intents"], 0)
        _, handle = self.producer.get(ref)
        self.producer.release(handle)

    def test_live_lease_prevents_same_name_retry(self):
        ref = self.produce()
        _, handle = self.producer.get(ref)
        with self.assertRaisesRegex(RuntimeError, "active consumer"):
            self.producer.prepare_capture_retry(ref.sample_id)
        self.producer.release(handle)

    def test_negotiated_identity_mismatch_fails_without_adopting(self):
        original = self.adapter.post_fn

        def corrupt(url, json_body, timeout):
            result = original(url, json_body, timeout)
            result[0]["meta_info"]["spec_capture"]["server_epoch"] = "wrong"
            return result

        self.adapter.post_fn = corrupt
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            self.produce()
        self.assertTrue(self.producer._external_provisional)
        self.producer.discard_external_attempts()
        self.assertFalse(self.backend._d)


if __name__ == "__main__":
    unittest.main()
