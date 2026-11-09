"""Native opt-in and real coordinator/store CPU ownership contracts.

The transport is an in-memory fixture; these tests do not certify CUDA, actual
Mooncake DMA, scheduler TP agreement or numerical feature equivalence.
"""
import os
import tempfile
import unittest
from unittest import mock

from specforge.config.schema import DisaggregatedDeploymentConfig
from specforge.inference.adapters.server_capture import SGLangServerCaptureAdapter
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel
from tests.test_runtime.test_capture_transfer import TransferServer
from tests.test_runtime.test_disagg_multiserver import _adapter, _build, _prompts
from tests.test_runtime.test_server_capture import (
    _FakeMooncakeStore, _capture_schema, _eagle3_contract, _task,
)


class NativeHSConfiguration(unittest.TestCase):
    def test_reuse_requires_protocol_and_mooncake(self):
        for fields in (dict(hs_prefix_cache=True), dict(hs_capture=True, backend="shared_dir", store_root="/tmp/test"), dict(server_capture="patch")):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                DisaggregatedDeploymentConfig(**{ "control_dir": "/tmp/test", "backend": "mooncake", **fields})
        cfg = DisaggregatedDeploymentConfig(control_dir="/tmp/test", backend="mooncake", hs_capture=True)
        self.assertFalse(cfg.hs_prefix_cache)

    def test_opt_in_rejects_missing_or_different_capability_before_begin(self):
        for capability in (None, dict(published_ownership="attempt-v1", cache_enabled=False)):
            backend = _FakeMooncakeStore()
            store = MooncakeFeatureStore(store=backend, store_id="run0")
            adapter = SGLangServerCaptureAdapter("http://server", store, run_id="run0", algorithm="eagle3", schema=_capture_schema("eagle3"), hs_capture=True, hs_prefix_cache=True)
            with mock.patch.object(adapter._capture_protocol, "capability", return_value=capability), self.assertRaises(RuntimeError):
                adapter.prepare_refs_batches([[_task(0, 5)]], capture=_eagle3_contract())
            self.assertFalse(store._capture_pending_begin)
            self.assertFalse(backend._d)


    def test_negotiated_metadata_cannot_change_digest_dtype_or_feature_names(self):
        for mutation in ("digest", "dtype", "extra"):
            backend = _FakeMooncakeStore()
            store = MooncakeFeatureStore(store=backend, store_id="run0")
            server = TransferServer(backend)
            def post(url, json_body, timeout):
                rows = server(url, json_body, timeout)
                if url.endswith("/generate"):
                    result = rows[0]["meta_info"]["spec_capture"]
                    if mutation == "digest":
                        result["request_digest"] = "changed"
                    elif mutation == "dtype":
                        result["features"]["loss_mask"]["dtype"] = "float16"
                    else:
                        result["features"]["unexpected"] = dict(shape=[1, 1], dtype="int8")
                return rows
            adapter = SGLangServerCaptureAdapter("http://server", store, run_id="run0", algorithm="eagle3", schema=_capture_schema("eagle3"), post_fn=post, control_fn=post, hs_capture=True, hs_prefix_cache=True)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(RuntimeError, "digest|shape/dtype|feature names"):
                adapter.produce_refs([_task(0, 5)], capture=_eagle3_contract())
            self.assertTrue(store._external_provisional)
            store.discard_external_attempts()
            self.assertFalse(backend._d)


class CapacityBackpressure(unittest.TestCase):
    def test_refused_and_new_prepared_groups_can_share_the_next_begin(self):
        from specforge.inference.capture_protocol import CaptureCapacityDeclined
        backend = _FakeMooncakeStore()
        store = MooncakeFeatureStore(store=backend, store_id="run0")
        server = TransferServer(backend)
        refuse = [True]
        def post(url, json_body, timeout):
            if json_body.get("action") == "begin_batch" and refuse[0]:
                refuse[0] = False
                return dict(error="full", error_code="capture_capacity_declined", reservation_granted=False)
            return server(url, json_body, timeout)
        adapter = SGLangServerCaptureAdapter("http://server", store, run_id="run0", algorithm="eagle3", schema=_capture_schema("eagle3"), post_fn=post, control_fn=post, hs_capture=True, hs_prefix_cache=True)
        old = adapter.prepare_refs_batches([[_task(0, 5)]], capture=_eagle3_contract())
        with self.assertRaises(CaptureCapacityDeclined):
            adapter.register_prepared_batches(old)
        attempt = old[0]["payloads"][0]["attempt_id"]
        fresh = adapter.prepare_refs_batches([[_task(1, 5)]], capture=_eagle3_contract())
        adapter.register_prepared_batches(old + fresh)
        self.assertEqual(old[0]["payloads"][0]["attempt_id"], attempt)
        self.assertFalse(store._capture_pending_begin)
        for group in old + fresh:
            self.assertEqual(len(adapter.produce_prepared(group, capture=_eagle3_contract())), 1)

    def test_atomic_refusal_retries_original_lease_without_prompt_failure(self):
        for concurrency in (1, 3):
            with self.subTest(concurrency=concurrency), tempfile.TemporaryDirectory() as root:
                backend = _FakeMooncakeStore()
                store = MooncakeFeatureStore(store=backend, store_id="run0")
                server = TransferServer(backend)
                attempts = []
                refusal_count = 4

                def post(url, json_body, timeout):
                    if json_body.get("action") == "begin_batch":
                        attempts.append(tuple(item["attempt_id"] for item in json_body["tasks"]))
                        if len(attempts) <= refusal_count:
                            return dict(error="full", error_code="capture_capacity_declined", reservation_granted=False)
                    return server(url, json_body, timeout)

                adapter = _adapter(store, post)
                adapter.hs_capture = adapter.hs_prefix_cache = True
                from specforge.inference.capture_protocol import CaptureProtocol
                adapter._capture_protocol = CaptureProtocol(adapter.base_url, post, 1)
                channel = StreamingRefChannel(os.path.join(root, "refs"))
                workers, drive = _build([adapter], _prompts(3), store, channel, lease=1, producer_concurrency=concurrency, max_worker_failures=1, max_prompt_attempts=1, feature_store_max_resident_bytes=1 << 20, backpressure_poll_s=.001, peer_wait_timeout_s=2)
                self.assertEqual(drive(), 3)
                self.assertTrue(all(item == attempts[0] for item in attempts[:refusal_count + 1]))
                self.assertEqual(workers[0].controller.status()["prompts_failed"], 0)
                self.assertEqual(workers[0].health()["in_flight"], 0)
                self.assertFalse(store._capture_pending_begin)
                self.assertEqual(channel.published, 3)


class FencedReceiveDrain(unittest.TestCase):
    def make_ref(self):
        backend = _FakeMooncakeStore()
        producer = MooncakeFeatureStore(store=backend, store_id="run0")
        server = TransferServer(backend)
        adapter = SGLangServerCaptureAdapter("http://server", producer, run_id="run0", algorithm="eagle3", schema=_capture_schema("eagle3"), post_fn=server, control_fn=server, hs_capture=True, hs_prefix_cache=True)
        ref = adapter.produce_refs([_task(0, 5)], capture=_eagle3_contract())[0]
        return backend, ref

    def test_failed_fenced_receive_keeps_buffer_registration_and_read_authority(self):
        for failure in (-800, RuntimeError("late native read")):
            with self.subTest(failure=failure):
                backend, ref = self.make_ref()
                consumer = MooncakeFeatureStore(store=backend, store_id="run0")
                original_keys = set(backend._d)
                effect = failure if isinstance(failure, Exception) else None
                with mock.patch.object(backend, "get_into", side_effect=effect, return_value=failure) as read, mock.patch.object(backend, "unregister_buffer", wraps=backend.unregister_buffer) as unregister:
                    with self.assertRaises((KeyError, RuntimeError)):
                        consumer.get(ref)
                    self.assertEqual(read.call_count, 1)
                    self.assertEqual(unregister.call_count, 0)
                self.assertEqual(consumer.health()["capture_failed_reads"], 1)
                self.assertEqual(len(consumer._quarantined_buffers), 1)
                self.assertEqual(len(consumer._capture_quarantined_registrations), 1)
                with self.assertRaisesRegex(RuntimeError, "read ownership|readers|not drained"):
                    consumer.drain_pending_removals(retry_interval_s=0)
                consumer.close()  # Injected transport is still owned elsewhere.
                self.assertEqual(consumer.health()["capture_failed_reads"], 1)
                self.assertEqual(set(backend._d), original_keys)

    def test_owned_transport_close_is_required_before_releasing_failed_read(self):
        backend, ref = self.make_ref()
        consumer = MooncakeFeatureStore(store=backend, store_id="run0")
        with mock.patch.object(backend, "get_into", return_value=-800), self.assertRaises(KeyError):
            consumer.get(ref)
        consumer._owns_store = True
        with mock.patch.object(backend, "close", create=True, side_effect=RuntimeError("not stopped")):
            with self.assertRaisesRegex(RuntimeError, "not stopped"):
                consumer.close()
        self.assertEqual(consumer.health()["capture_failed_reads"], 1)
        with mock.patch.object(backend, "close", create=True, return_value=0):
            consumer.close()
        self.assertFalse(consumer._capture_read_intents)
        self.assertFalse(consumer._quarantined_buffers)
        self.assertFalse(consumer._capture_quarantined_registrations)


class NativeHSLaunch(unittest.TestCase):
    def test_managed_launch_exports_explicit_protocol_budget_and_keeps_guards(self):
        from tests.test_runtime.test_launch_plan import _managed_raw, _build_managed
        raw = _managed_raw(protocol="tcp")
        config = raw["deployment"]["disaggregated"]
        config.update(hs_capture=True, hs_prefix_cache=True, hs_chunk_tokens=384, hs_output_bytes=987654)
        capture = _build_managed(raw).services[1].command
        self.assertEqual(capture.env["SPECFORGE_HS_CAPTURE"], "1")
        self.assertEqual(capture.env["SPECFORGE_HS_CACHE"], "1")
        self.assertEqual(capture.env["SPECFORGE_HS_CHUNK_TOKENS"], "384")
        self.assertEqual(capture.env["SPECFORGE_HS_OUTPUT_BYTES"], "987654")
        self.assertEqual(capture.env["SGLANG_SPEC_CAPTURE_GPU_PUT"], "0")
        self.assertIsNone(capture.env["FLASHINFER_DISABLE_VERSION_CHECK"])
        config.update(hs_capture=False, hs_prefix_cache=False)
        native = _build_managed(raw).services[1].command
        self.assertEqual(native.env["SPECFORGE_HS_CAPTURE"], "0")
        self.assertEqual(native.env["SPECFORGE_HS_CACHE"], "0")
