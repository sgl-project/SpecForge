"""Immutable published ownership, independent producer/consumer lifecycles."""
import types
import unittest
from unittest import mock

import torch

from specforge.inference.capture_protocol import CaptureProtocol, WriterFence, attempt_keys
from specforge.inference.adapters.server_capture import SGLangServerCaptureAdapter
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
from tests.test_runtime.test_capture_fencing import ProtocolServer
from tests.test_runtime.test_server_capture import _FakeMooncakeStore, _task, _capture_schema, _eagle3_contract


class TransferServer(ProtocolServer):
    def __init__(self, backend):
        super().__init__(backend)
        self.capability.update(begin_batch=True, published_ownership='attempt-v1')
        for feature in self.capability['schema']:
            feature['dtype'] = 'bfloat16'
        self.calls = []

    def __call__(self, url, json_body, timeout):
        action = json_body.get('action', 'generate')
        self.calls.append(action)
        if action == 'begin_batch':
            replies = []
            for body in json_body['tasks']:
                reply = super().__call__(url, dict(body, action='begin'), timeout)
                reply['object_key_layout'] = 'attempt-v1'
                replies.append(reply)
            return dict(attempts=replies)
        if action == 'begin':
            reply = super().__call__(url, json_body, timeout)
            reply['object_key_layout'] = 'attempt-v1'
            return reply
        rows = super().__call__(url, json_body, timeout)
        if action != 'generate':
            return rows
        for row in rows:
            result = row['meta_info']['spec_capture']
            keys = attempt_keys(result, result['features'])
            for name, key in keys.items():
                self.backend._d[key] = self.backend._d.pop(f"{result['store_id']}/{result['sample_id']}/g1/{name}")
                result['features'][name]['key'] = key
            result.update(object_key_layout='attempt-v1', publication_ownership='attempt-v1', physical_keys=keys)
            entry = self.active.pop(result['sample_id'])
            certificate = {k: result[k] for k in ('store_id', 'sample_id', 'gen', 'attempt_id', 'server_epoch',
                'reservation_token', 'object_key_layout', 'publication_ownership', 'physical_keys', 'writer_drained', 'state')}
            entry.update(state='RETIRED', publication_certificate=certificate,
                         publication_ownership='attempt-v1', physical_keys=keys)
            self.retired[entry['reservation_token']] = entry
        return rows


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.backend = _FakeMooncakeStore()
        self.server = TransferServer(self.backend)
        self.producer = MooncakeFeatureStore(store=self.backend, store_id='run0', max_hold_age_s=0)
        self.adapter = SGLangServerCaptureAdapter('http://server', self.producer, run_id='run0',
            algorithm='eagle3', schema=_capture_schema('eagle3'), post_fn=self.server, control_fn=self.server, hs_capture=True, hs_prefix_cache=True)

    def produce(self, *indexes):
        return self.adapter.produce_refs([_task(i, 5) for i in indexes], capture=_eagle3_contract())

    def test_batched_begin_cached_capability_and_zero_consumer_rpc(self):
        refs = self.produce(0, 1)
        self.produce(2)
        self.assertEqual(self.server.calls.count('capability'), 1)
        self.assertEqual(self.server.calls.count('begin_batch'), 2)
        self.assertNotIn('resolve', self.server.calls)
        consumer = MooncakeFeatureStore(store=self.backend, store_id='run0')
        before = list(self.server.calls)
        with mock.patch('specforge.inference.capture_protocol.http_control', side_effect=AssertionError('consumer control RPC')):
            for ref in refs:
                values, handle = consumer.get(ref)
                expected = self.server.stub.expected[ref.sample_id]
                self.assertEqual(set(values), set(expected))
                for name, tensor in expected.items():
                    torch.testing.assert_close(values[name], tensor, rtol=0, atol=0)
                consumer.release(handle)
            consumer.drain_pending_removals(retry_interval_s=0)
        self.assertEqual(before, self.server.calls)

    def test_multiple_readers_defer_local_remove(self):
        ref = self.produce(0)[0]
        _, first = self.producer.get(ref)
        _, second = self.producer.get(ref)
        self.producer.release(first)
        self.assertTrue(self.backend._d)
        self.assertEqual(len(self.producer._active_leases), 1)
        self.producer.release(second)
        self.assertFalse(self.backend._d)

    def test_producer_gc_handoff_and_stale_process_cannot_remove_replacement(self):
        old = self.produce(0)[0]
        consumer = MooncakeFeatureStore(store=self.backend, store_id='run0')
        with mock.patch('specforge.inference.capture_protocol.http_control', side_effect=AssertionError('consumer control RPC')):
            _, handle = consumer.get(old)
            self.producer.gc(now=10**12)
            self.assertTrue(self.backend._d)
            self.producer.relinquish_published(old)
            self.assertFalse(self.producer._generation)
            self.assertTrue(self.backend._d)
            replacement_store = MooncakeFeatureStore(store=self.backend, store_id='run0')
            replacement = SGLangServerCaptureAdapter('http://server', replacement_store, run_id='run0',
                algorithm='eagle3', schema=_capture_schema('eagle3'), post_fn=self.server, control_fn=self.server, hs_capture=True, hs_prefix_cache=True)
            new = replacement.produce_refs([_task(0, 5)], capture=_eagle3_contract())[0]
            self.assertNotEqual(old.feature_keys, new.feature_keys)
            consumer.release(handle)  # Only the old attempt's immutable keys.
            self.assertEqual(set(self.backend._d), set(new.feature_keys.values()))
            self.producer.abort(old.sample_id)
            self.producer.gc(now=10**12)
            self.assertEqual(set(self.backend._d), set(new.feature_keys.values()))
            with self.assertRaises(KeyError):
                self.producer.get(old)

    def test_lost_generate_response_resolves_certificate_then_cleans(self):
        original = self.adapter.post_fn
        def lose(*args, **kwargs):
            original(*args, **kwargs)
            raise TimeoutError('lost after successful publication')
        self.adapter.post_fn = lose
        with self.assertRaises(TimeoutError):
            self.produce(0)
        self.assertTrue(self.backend._d)
        self.producer.discard_external_attempts()
        self.producer.drain_pending_removals(retry_interval_s=0)
        self.assertFalse(self.backend._d)
        self.assertFalse(self.producer._capture_fences)
        self.assertIn('resolve', self.server.calls)
        self.assertNotIn('cleanup_acquire', self.server.calls)

    def test_strict_state_and_exact_keys_fail_closed(self):
        ref = self.produce(0)[0]
        fence = WriterFence(dict(ref.metadata['capture_fence']), tuple(ref.feature_keys))
        for bad in (dict(fence.wire, state='DRAINING'), dict(fence.wire, writer_drained=False),
                    dict(fence.wire, physical_keys={})):
            with self.assertRaises(RuntimeError):
                fence.accept_transfer(bad)
            self.assertFalse(fence.transferred)

    def test_lost_begin_batch_reply_recovers_per_task_intents(self):
        original = self.adapter._capture_protocol.post
        once = [True]
        def lose(url, json_body, timeout):
            result = original(url, json_body, timeout)
            if json_body.get('action') == 'begin_batch' and once[0]:
                once[0] = False
                raise TimeoutError('lost batched reservations')
            return result
        self.adapter._capture_protocol.post = lose
        with self.assertRaises(TimeoutError):
            self.produce(0, 1)
        self.assertEqual(len(self.producer._capture_pending_begin), 2)
        self.producer.discard_external_attempts()
        self.assertFalse(self.server.active)
        self.assertFalse(self.producer._capture_pending_begin)


class CapabilityTests(unittest.TestCase):
    def test_absent_legacy_capability_is_cached(self):
        calls = []
        def post(*args, **kwargs):
            calls.append(1)
            error = RuntimeError('404')
            error.response = types.SimpleNamespace(status_code=404)
            raise error
        protocol = CaptureProtocol('http://server', post, 1)
        self.assertIsNone(protocol.capability())
        self.assertIsNone(protocol.capability())
        self.assertEqual(len(calls), 1)

    def test_409_refreshes_capability_without_replaying_mutation(self):
        calls = []
        def post(url, json_body, timeout):
            calls.append(json_body['action'])
            if json_body['action'] == 'capability':
                return dict(protocol_version=1, server_epoch=str(len(calls)), capture_namespace='n',
                            capture_signature='s', schema=[], cache_enabled=False)
            error = RuntimeError('409')
            error.response = types.SimpleNamespace(status_code=409)
            raise error
        protocol = CaptureProtocol('http://server', post, 1)
        protocol.capability()
        with self.assertRaises(RuntimeError):
            protocol.call(dict(action='begin'))
        self.assertEqual(calls, ['capability', 'begin', 'capability'])


if __name__ == '__main__':
    unittest.main()

class PublicationBoundaryTests(TransferTests):
    def test_visible_fsync_failure_relinquishes_prefix_and_aborts_suffix(self):
        import tempfile
        from specforge.launch import _publish_refs_with_cleanup
        from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel
        refs = self.produce(0, 1)
        with tempfile.TemporaryDirectory() as directory:
            channel = StreamingRefChannel(directory + '/refs.jsonl')
            with mock.patch('specforge.runtime.data_plane.streaming_ref_channel.os.fsync', side_effect=OSError('fsync failed')):
                with self.assertRaises(OSError):
                    _publish_refs_with_cleanup(channel=channel, feature_store=self.producer, refs=refs)
            self.assertEqual(channel.published, 1)
            self.assertEqual([ref.sample_id for ref in channel.poll()], [refs[0].sample_id])
            self.assertEqual(set(self.backend._d), set(refs[0].feature_keys.values()))
            self.assertNotIn(refs[0].sample_id, self.producer._capture_fences)
            self.assertNotIn(refs[1].sample_id, self.producer._capture_fences)

    def test_partial_put_rollback_preserves_same_id_capture(self):
        consumer = MooncakeFeatureStore(store=self.backend, store_id="run0")
        sample_id = "run0:t0"
        legacy_key = f"run0/{sample_id}/g1/input_ids"
        ref = self.produce(0)[0]
        self.assertEqual(ref.sample_id, sample_id)
        self.assertEqual(ref.metadata["generation"], 1)
        capture_keys = set(ref.feature_keys.values())
        put = self.backend.put_from

        def write_then_fail(*args):
            put(*args)
            return -1

        with (
            mock.patch.object(self.backend, "put_from", side_effect=write_then_fail),
            mock.patch.object(self.backend, "remove", return_value=-1),
            self.assertRaises(RuntimeError),
        ):
            consumer.put({"input_ids": torch.tensor([[10]])},
                         sample_id=sample_id, metadata={})
        self.assertEqual(set(self.backend._d), capture_keys | {legacy_key})
        self.assertEqual(consumer.health()["release_pending"], 1)

        consumer.adopt(ref)
        self.assertEqual(consumer.gc()["release_pending"], 0)
        self.assertEqual(set(self.backend._d), capture_keys)
        with self.assertRaisesRegex(RuntimeError, "fenced capture sample"):
            consumer.put({"input_ids": torch.tensor([[20]])},
                         sample_id=sample_id, metadata={})
        values, handle = consumer.get(ref)
        torch.testing.assert_close(values["input_ids"], torch.arange(1, 6).view(1, -1))
        consumer.release(handle)
        self.assertFalse(self.backend._d)

    def test_same_consumer_accepts_new_attempt_and_rejects_old_ref(self):
        old = self.produce(0)[0]
        consumer = MooncakeFeatureStore(store=self.backend, store_id='run0')
        with mock.patch('specforge.inference.capture_protocol.http_control', side_effect=AssertionError('consumer control RPC')):
            _, old_handle = consumer.get(old)
            consumer.release(old_handle)
            other_store = MooncakeFeatureStore(store=self.backend, store_id='run0')
            adapter = SGLangServerCaptureAdapter('http://server', other_store, run_id='run0',
                algorithm='eagle3', schema=_capture_schema('eagle3'), post_fn=self.server, control_fn=self.server, hs_capture=True, hs_prefix_cache=True)
            new = adapter.produce_refs([_task(0, 5)], capture=_eagle3_contract())[0]
            _, new_handle = consumer.get(new)
            with self.assertRaises(KeyError):
                consumer.get(old)
            consumer.release(old_handle)
            self.assertTrue(self.backend._d)
            consumer.release(new_handle)
            self.assertFalse(self.backend._d)

    def test_lost_generate_old_cleanup_cannot_touch_started_new_attempt(self):
        original = self.adapter.post_fn
        def lose(*args, **kwargs):
            original(*args, **kwargs)
            raise TimeoutError('lost old publication')
        self.adapter.post_fn = lose
        with self.assertRaises(TimeoutError):
            self.produce(0)
        other_store = MooncakeFeatureStore(store=self.backend, store_id='run0')
        adapter = SGLangServerCaptureAdapter('http://server', other_store, run_id='run0',
            algorithm='eagle3', schema=_capture_schema('eagle3'), post_fn=self.server, control_fn=self.server, hs_capture=True, hs_prefix_cache=True)
        new = adapter.produce_refs([_task(0, 5)], capture=_eagle3_contract())[0]
        self.producer.discard_external_attempts()
        self.assertEqual(set(self.backend._d), set(new.feature_keys.values()))

class DelayedAbortTests(TransferTests):
    def test_ref_specific_old_abort_does_not_remove_same_store_replacement(self):
        from specforge.runtime.data_plane.feature_store import abort_feature_ref
        old = self.produce(0)[0]
        self.producer.relinquish_published(old)
        new = self.produce(0)[0]
        before = set(self.backend._d)
        abort_feature_ref(self.producer, old, reason='late old publication callback')
        self.assertEqual(set(self.backend._d), before)
        self.assertTrue(set(new.feature_keys.values()).issubset(self.backend._d))

    def test_legacy_ref_cannot_abort_same_id_new_capture(self):
        from dataclasses import replace
        from specforge.runtime.data_plane.feature_store import abort_feature_ref
        new = self.produce(0)[0]
        legacy = replace(new, metadata={k: v for k, v in new.metadata.items() if k != 'capture_fence'})
        before = set(self.backend._d)
        abort_feature_ref(self.producer, legacy, reason='late legacy callback')
        self.assertEqual(set(self.backend._d), before)
