"""Terminal ACKs must use the installed Mooncake configuration type.

Only transport storage/connection is a CPU stub. Its put boundary rejects every
configuration except the actual installed pybind ReplicateConfig, matching the
native API that exposed the unconsumed-manifest failure over RDMA.
"""
from contextlib import contextmanager
import tempfile
import unittest
from unittest import mock

try:
    from mooncake.store import ReplicateConfig
except ImportError:
    ReplicateConfig = None

from specforge.runtime.data_plane import mooncake_store as module
from tests.test_runtime.test_mooncake_store import _FakeMooncakeStore
from tests.test_runtime import test_segment_manifest as segment_fixture


class NativeConfigBackend(_FakeMooncakeStore):
    def __init__(self, data):
        super().__init__()
        self._d = data
        self.configurations = []
        self.registered = {}
        self.close_calls = 0
        self.unknown_ack = None

    def register_buffer(self, pointer, nbytes):
        self.registered[pointer] = nbytes
        return 0

    def unregister_buffer(self, pointer):
        self.registered.pop(pointer)
        return 0

    def put_from(self, key, pointer, nbytes, config=None):
        self.configurations.append((key, config))
        if type(config) is not ReplicateConfig:
            raise TypeError(
                'native put_from requires mooncake.store.ReplicateConfig; got '
                + type(config).__name__
            )
        if key == self.unknown_ack:
            return -800
        return super().put_from(key, pointer, nbytes, config)

    def close(self):
        self.close_calls += 1
        return 0


@unittest.skipIf(ReplicateConfig is None, 'installed Mooncake native config required')
class TerminalNativeConfigTests(unittest.TestCase):
    @contextmanager
    def protocol(self):
        with tempfile.TemporaryDirectory() as directory, segment_fixture.TestSegmentManifest().environment(directory):
            fixture = segment_fixture.Fixture(directory)
            clients = []

            def native_client(replica_num=1):
                backend = NativeConfigBackend(fixture.backend._d)
                # Exercise the owned constructor's real native-config branch.
                # No Mooncake setup, network, or GPU operation is performed.
                with mock.patch.object(module, '_connect_store', return_value=(backend, ReplicateConfig)):
                    store = module.MooncakeFeatureStore(
                        store_id='run0', replica_num=replica_num,
                        hard_pin=True, retain_on_release=True,
                    )
                clients.append(store)
                self.assertIs(type(store._put_config), ReplicateConfig)
                return store, backend

            try:
                yield fixture, native_client
            finally:
                fixture.owner.stop.set()
                fixture.owner.thread.join(5)
                for client in clients:
                    try:
                        client.close()
                    except RuntimeError:
                        # The byte stub submitted no native work in its -800 or
                        # argument-rejection branches. Close only local test
                        # handles; do not fabricate a successful drain/proof.
                        for local in (client._terminal_cleanup_authority, client):
                            if local is None:
                                continue
                            if local._terminal_proofs is not None:
                                local._terminal_proofs.close()
                            if local._node_pool is not None:
                                local._node_pool.close()
                fixture.controller.shutdown()
                fixture.sink.close()
                fixture.owner.ledger.db.close()

    def assert_borrowed_config(self, producer, backend, replicas):
        authority = producer._terminal_cleanup_authority
        self.assertIsNotNone(authority)
        self.assertIs(authority._store, backend)
        self.assertFalse(authority._owns_store)
        self.assertIsNone(authority._node_pool)
        self.assertIs(authority._put_config, producer._put_config)
        self.assertIs(type(authority._put_config), ReplicateConfig)
        self.assertTrue(authority._put_config.with_hard_pin)
        self.assertEqual(authority._put_config.replica_num, replicas)
        return authority

    def test_unconsumed_manifest_uses_native_config_and_repeat_scan_is_idempotent(self):
        for replicas in (1, 2):
            with self.subTest(replicas=replicas), self.protocol() as (fixture, native_client):
                ref = fixture.publish('unconsumed')
                producer, backend = native_client(replicas)
                producer.adopt(ref)
                producer.relinquish_published(ref)
                producer.cleanup_terminal_refs([ref], consumer_done=True, producer_closed=True)
                producer.cleanup_terminal_refs([ref], consumer_done=True, producer_closed=True)
                authority = self.assert_borrowed_config(producer, backend, replicas)
                self.assertEqual(len(backend.configurations), 1)
                self.assertEqual(backend.configurations[0][0], ref.metadata['segment_manifest']['ack_key'])
                self.assertIs(backend.configurations[0][1], producer._put_config)
                self.assertFalse(backend.registered)
                self.assertEqual(backend.close_calls, 0)
                fixture.owner.progress()
                self.assertEqual(fixture.owner.health()['active_manifests'], 0)
                self.assertFalse(backend._d)
                authority.close()
                self.assertEqual(backend.close_calls, 0)
                producer.close()
                self.assertEqual(backend.close_calls, 1)

    def test_consumed_manifest_reuses_proof_without_another_native_ack(self):
        with self.protocol() as (fixture, native_client):
            ref = fixture.publish('consumed')
            producer, backend = native_client(2)
            consumer, consumer_backend = native_client()
            producer.adopt(ref)
            producer.relinquish_published(ref)
            consumer.adopt(ref)
            _, handle = consumer.get(ref)
            consumer.release(handle)
            fixture.drain(consumer, ref)
            consumer.close()
            self.assertEqual(consumer_backend.close_calls, 1)
            producer.cleanup_terminal_refs([ref], consumer_done=True, producer_closed=True)
            producer.cleanup_terminal_refs([ref], consumer_done=True, producer_closed=True)
            authority = self.assert_borrowed_config(producer, backend, 2)
            self.assertEqual(backend.configurations, [])
            self.assertFalse(backend.registered)
            authority.close()
            self.assertEqual(backend.close_calls, 0)
            producer.close()
            self.assertEqual(backend.close_calls, 1)

    def test_unknown_native_ack_completion_retains_claim_buffer_and_registration(self):
        with self.protocol() as (fixture, native_client):
            ref = fixture.publish('unknown')
            producer, backend = native_client(2)
            producer.adopt(ref)
            producer.relinquish_published(ref)
            manifest = ref.metadata['segment_manifest']
            backend.unknown_ack = manifest['ack_key']
            with self.assertRaisesRegex(RuntimeError, 'ACK marker publication failed'):
                producer.cleanup_terminal_refs([ref], consumer_done=True, producer_closed=True)
            authority = self.assert_borrowed_config(producer, backend, 2)
            self.assertEqual(len(backend.configurations), 1)
            self.assertIn(manifest['token'], authority._segment_ack_unsafe)
            self.assertEqual(len(authority._quarantined_buffers), 1)
            pointer = authority._quarantined_buffers[0].data_ptr()
            self.assertIn(pointer, authority._capture_quarantined_registrations)
            self.assertIn(pointer, backend.registered)
            self.assertEqual(authority._terminal_proofs.owned_claims(authority._terminal_proof_owner), 1)
            self.assertEqual(authority._terminal_proofs.health()['terminal_proof_unsafe_claims'], 1)
            with self.assertRaises(RuntimeError):
                producer.cleanup_terminal_refs([ref], consumer_done=True, producer_closed=True)
            self.assertEqual(len(backend.configurations), 1)
            with self.assertRaisesRegex(RuntimeError, 'unresolved terminal ACK claims'):
                producer.close()
            self.assertEqual(backend.close_calls, 0)
            self.assertIn(pointer, backend.registered)
