"""CPU end-to-end manifest publication, node pool, final batching and ACKs."""
import copy
import os
import tempfile
import time
import unittest
from dataclasses import replace
from unittest import mock

import torch

from tests.test_runtime.test_hs_runtime import features, identity, schema
from tests.test_runtime.test_mooncake_store import _FakeMooncakeStore
from specforge_sglang_capture.hs import CaptureController, CapturedSpan, PrefixFeatureCache
from specforge_sglang_capture.sink import SpecCaptureSink, sample_object_key
from specforge.runtime.contracts import FeatureSpec, SampleRef
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
from specforge.runtime.data_plane.feature_dataloader import FeatureDataLoader
from specforge.runtime.data_plane.segment_manifest import (
    RemoteManifestOwner, RemoteCapacityError, digest_manifest, validate_manifest,
    remote_allocation_charge,
)
from specforge.algorithms.common.hidden_states_data import build_collator
from specforge.inference.capture_protocol import attempt_keys


class Fixture:
    def __init__(self, directory, capacity=1 << 24):
        self.directory = directory
        self.backend = _FakeMooncakeStore()
        self.sink = SpecCaptureSink([1, 3])
        self.sink._store = self.backend
        from specforge.runtime.data_plane.mooncake_store import _InjectedReplicateConfig
        self.sink._put_config = _InjectedReplicateConfig()
        self.owner = RemoteManifestOwner(self.backend, self.sink._managed_put,
                                         directory=directory + '/ledger', capacity=capacity, hard_pin=True)
        self.cache = PrefixFeatureCache(schema(), 4, 4096)
        self.controller = CaptureController(self.cache, schema(), 4096, 8192, 4096, segmented=True)
        self.sequence = 0

    def publish(self, name, n=9, l=0, mask=1):
        self.sequence += 1
        execution = identity(self.sequence)
        tokens = tuple(range(n))
        lease = self.cache.lookup_and_pin('epoch-ns', tokens, n)
        lease = self.cache.shrink_lease(lease, l)
        reservation = self.controller.try_reserve_output(execution, n, l, n * 16)
        plan = self.controller.freeze_plan(execution, 'epoch-ns', tokens, n, l, lease, reservation)
        sample = self.controller.submit(plan, CapturedSpan(execution, l, n, features(n-l, l), None)).result(5)
        spec = {'store_id': 'run0', 'sample_id': name, 'gen': 1,
                'server_epoch': 'epoch', 'attempt_id': 'attempt-' + name + '-' + str(self.sequence),
                'reservation_token': 'reservation-' + name + '-' + str(self.sequence), 'object_key_layout': 'attempt-v1',
                'features': {'aux': 'hidden_states', 'last_hidden': 'target_last_hidden_states'},
                'passthrough': [
                    {'name': 'input_ids', 'dtype': 'int64', 'shape': [1,n], 'data': list(range(n))},
                    {'name': 'loss_mask', 'dtype': 'int64', 'shape': [1,n], 'data': [mask] * n}]}
        try:
            result = self.owner.publish(spec, sample, schema(), 'epoch-ns', sample_object_key)
        finally:
            sample.release()
            self.controller.poll_completed()
            self.controller.poll_drained()
        specs = {name: FeatureSpec(name, tuple(item['shape']), item['dtype']) for name,item in result['features'].items()}
        keys = attempt_keys(spec, specs)
        wire = {name: spec[name] for name in ('store_id','sample_id','gen','server_epoch','attempt_id','reservation_token')}
        wire.update(protocol_version=1, object_key_layout='attempt-v1', publication_ownership='attempt-v1',
                    physical_keys=keys, writer_drained=True, state='PUBLISHED',
                    feature_layout='hs-manifest-v1', segment_manifest=result['segment_manifest'])
        return SampleRef(name, 'run', None, 'mooncake://run0/' + name, keys, specs, 'dflash',
                         schema_version=2, num_tokens=n,
                         estimated_bytes=n * 32,
                         metadata={'generation': 1, 'feature_layout': 'hs-manifest-v1',
                                   'segment_manifest': result['segment_manifest'], 'capture_fence': wire})

    def reader(self, **kwargs):
        return MooncakeFeatureStore(store=self.backend, store_id='run0', retain_on_release=True, **kwargs)

    def drain(self, reader, ref):
        reader.abort_ref(ref)
        deadline = time.monotonic() + 5
        while ref.sample_id in reader._release_pending:
            self.owner.progress()
            reader.retry_sample_removals([ref.sample_id])
            if time.monotonic() >= deadline:
                raise RuntimeError('ACK did not drain')
        self.owner.progress()

    def close(self):
        self.owner.progress()
        self.owner.close()
        self.controller.shutdown()
        self.sink.close()


class TestSegmentManifest(unittest.TestCase):
    def environment(self, directory, capacity=1 << 16):
        return mock.patch.dict(os.environ, {
            'SPECFORGE_TRAINER_DIRECT_BATCH': '1',
            'SPECFORGE_TRAINER_FEATURE_DEMAND': '1',
            'SPECFORGE_TRAINER_NODE_POOL': '1',
            'SPECFORGE_TRAINER_NODE_POOL_DIR': directory + '/pool',
            'SPECFORGE_TRAINER_NODE_POOL_BYTES': str(capacity),
        })

    def test_cold_warm_cross_l_tail_padding_mask_and_one_ack_cannot_delete_other(self):
        with tempfile.TemporaryDirectory() as directory, self.environment(directory):
            fixture = Fixture(directory)
            first = fixture.publish('a', n=9)
            second = fixture.publish('b', n=10, l=5, mask=0)
            self.assertGreater(fixture.owner.health()['block_reused'], 0)
            self.assertEqual(fixture.cache.stats()['cache_adopted_samples'], 0)
            reader = fixture.reader()
            data = FeatureDataLoader(reader, refs=[first, second], batch_size=2, strategy='dflash', collate_fn=build_collator())
            actual = next(iter(data))
            expected = build_collator()([
                {'input_ids': torch.arange(n).view(1,n), 'loss_mask': torch.full((1,n), mask),
                 'hidden_states': features(n)['aux'].unsqueeze(0),
                 'target_last_hidden_states': features(n)['last_hidden'].unsqueeze(0)}
                for n,mask in ((9,1),(10,0))])
            for name,value in expected.items():
                torch.testing.assert_close(actual.tensors[name], value)
            self.assertGreater(reader.health()['node_pool_hits'], 0)
            shared = set(item['key'] for item in first.metadata['segment_manifest']['features']['hidden_states']) & set(item['key'] for item in second.metadata['segment_manifest']['features']['hidden_states'])
            self.assertTrue(shared)
            fixture.drain(reader, first)
            self.assertTrue(all(fixture.backend.is_exist(key) for key in shared))
            self.assertGreater(fixture.owner.health()['active_dependencies'], 0)
            # A's receipt must complete while B remains unconsumed/unACKed.
            self.assertEqual(fixture.owner.health()['active_manifests'], 1)
            fixture.drain(reader, second)
            self.assertEqual(fixture.owner.health()['remote_owner_charged_bytes'], 0)
            self.assertFalse(fixture.backend._d)
            reader.close()
            fixture.close()

    def test_pool_restart_and_full_feature_get_share_same_materializer(self):
        with tempfile.TemporaryDirectory() as directory, self.environment(directory):
            fixture = Fixture(directory)
            ref = fixture.publish('a')
            first = fixture.reader()
            first.adopt(ref)
            tensors, handle = first.get(ref)
            first.release(handle)
            first.close()
            second = fixture.reader()
            second.adopt(ref)
            rebuilt, handle = second.get(ref)
            torch.testing.assert_close(tensors['hidden_states'], rebuilt['hidden_states'])
            self.assertGreater(second.health()['node_pool_misses'], 0)
            second.release(handle)
            fixture.drain(second, ref)
            second.close()
            fixture.close()

    def test_non_metric_demand_avoids_last_hidden_network_and_capacity_bypass_progresses(self):
        with tempfile.TemporaryDirectory() as directory, self.environment(directory, 4096):
            fixture = Fixture(directory)
            ref = fixture.publish('a', n=9)
            reader = fixture.reader()
            data = FeatureDataLoader(reader, refs=[ref], batch_size=1, strategy='dflash', collate_fn=build_collator())
            data.configure_consumer_schedule(global_step=0, accumulation_steps=1,
                                             accumulation_remainder=0, log_interval=10, has_logger=True)
            actual = next(iter(data))
            self.assertNotIn('target_last_hidden_states', actual.tensors)
            self.assertEqual(reader.health()['segment_get_bytes'], 9 * 2 * 4)
            fixture.drain(reader, ref)
            reader.close()
            fixture.close()

    def test_rc_zero_deferred_delete_retains_physical_charge_and_blocks_receipt(self):
        with tempfile.TemporaryDirectory() as directory, self.environment(directory):
            fixture = Fixture(directory)
            ref = fixture.publish('a')
            reader = fixture.reader()
            reader.adopt(ref)
            fixture.backend.lease_defer = True
            reader.abort_ref(ref)
            fixture.owner.progress()
            self.assertGreater(fixture.owner.health()['remote_pending_delete_bytes'], 0)
            self.assertEqual(fixture.backend.is_exist(ref.metadata['segment_manifest']['receipt_key']), 0)
            fixture.backend.lease_defer = False
            fixture.drain(reader, ref)
            reader.close()
            fixture.close()

    def test_unknown_layout_wrong_offset_and_allocator_charge(self):
        with tempfile.TemporaryDirectory() as directory, self.environment(directory):
            fixture = Fixture(directory)
            ref = fixture.publish('a')
            bad = copy.deepcopy(ref.metadata['segment_manifest'])
            bad['features']['hidden_states'][1]['start'] = 3
            bad['digest'] = digest_manifest({name:value for name,value in bad.items() if name != 'digest'})
            with self.assertRaisesRegex(ValueError, 'coverage'):
                validate_manifest(bad, ref.feature_specs, 9)
            reader = fixture.reader()
            with self.assertRaisesRegex(ValueError, 'layout'):
                reader.get(replace(ref, metadata={**ref.metadata,'feature_layout':'unknown'}))
            reader.adopt(ref)
            fixture.drain(reader, ref)
            reader.close()
            fixture.close()
        self.assertEqual(remote_allocation_charge(32), 4096)
        self.assertEqual(remote_allocation_charge(9 << 20), 9 << 20)
        self.assertGreaterEqual(remote_allocation_charge((9 << 20) + 1), (9 << 20) + 1)

    def test_partial_publication_failure_detaches_only_its_dependencies(self):
        with tempfile.TemporaryDirectory() as directory, self.environment(directory):
            fixture = Fixture(directory)
            first = fixture.publish('a')
            real_put = fixture.owner.put
            def fail_headers(objects):
                if any('/attempt/' in key for key in objects):
                    raise RuntimeError('partial header failure')
                return real_put(objects)
            with mock.patch.object(fixture.owner, 'put', side_effect=fail_headers):
                with self.assertRaisesRegex(RuntimeError, 'header failure'):
                    fixture.publish('b', n=10, l=5)
            fixture.owner.progress()
            self.assertEqual(fixture.owner.health()['active_manifests'], 1)
            reader = fixture.reader()
            tensors, handle = reader.get(first)
            torch.testing.assert_close(tensors['hidden_states'], features(9)['aux'].unsqueeze(0))
            reader.release(handle)
            fixture.drain(reader, first)
            reader.close()
            fixture.close()

    def test_old_attempt_cleanup_cannot_delete_republished_generation(self):
        with tempfile.TemporaryDirectory() as directory, self.environment(directory):
            fixture = Fixture(directory)
            first = fixture.publish('a')
            reader = fixture.reader()
            reader.adopt(first)
            old_keys = {item['key'] for item in first.metadata['segment_manifest']['features']['hidden_states']}
            fixture.drain(reader, first)
            later = fixture.publish('a', n=9, l=5)
            new_keys = {item['key'] for item in later.metadata['segment_manifest']['features']['hidden_states']}
            self.assertFalse(old_keys & new_keys)
            reader.adopt(later)
            reader.abort_ref(first)
            self.assertTrue(all(fixture.backend.is_exist(key) for key in new_keys))
            fixture.drain(reader, later)
            reader.close()
            fixture.close()

    def test_shared_remote_ledger_reserves_marker_space_before_admission(self):
        from specforge.runtime.data_plane.segment_manifest import RemoteObjectLedger
        with tempfile.TemporaryDirectory() as directory:
            a = RemoteObjectLedger(directory, 16384, 'a')
            b = RemoteObjectLedger(directory, 16384, 'b')
            a.reserve({'block': 1, 'ack': 32, 'receipt': 32})
            self.assertEqual(b.health()['remote_node_charged_bytes'], 12288)
            b.reserve({'other': 1})
            with self.assertRaises(RemoteCapacityError):
                a.reserve({'over': 1})
            self.assertEqual(a.health()['remote_node_peak_charged_bytes'], 16384)
            for key in ('block','ack','receipt'):
                a.forget(key)
            b.forget('other')
            a.close()
            b.close()

    def test_lost_response_certificate_cleans_shared_dependencies_without_a_ref(self):
        with tempfile.TemporaryDirectory() as directory, self.environment(directory):
            fixture = Fixture(directory)
            ref = fixture.publish('a')
            reader = fixture.reader()
            # The producer learned ownership through resolve, but never built a
            # SampleRef. Cleanup still must use its exact manifest certificate.
            reader.track_capture_fence(ref.metadata['capture_fence'], list(ref.feature_keys))
            reader.abort(ref.sample_id, attempt_token=ref.metadata['capture_fence']['reservation_token'])
            fixture.drain(reader, ref)
            self.assertFalse(fixture.backend._d)
            reader.close()
            fixture.close()

    def test_registered_receive_slot_is_leased_directly_into_shared_pool(self):
        with tempfile.TemporaryDirectory() as directory, self.environment(directory):
            fixture = Fixture(directory)
            ref = fixture.publish('a')
            reader = fixture.reader(receive_buffers='pinned', receive_pool_bytes=4096)
            actual, handle = reader.get(ref)
            torch.testing.assert_close(actual['hidden_states'], features(9)['aux'].unsqueeze(0))
            self.assertFalse(reader._receive_pool._active)
            # Only the two ordinary int64 passthrough tensors were cloned out
            # of receive slots. HS cache misses copied the leased slots to mmap.
            self.assertEqual(reader.health()['receive_to_pageable_copy_bytes'], 9 * 16)
            self.assertEqual(reader.health()['node_pool_cpu_copy_bytes'], 9 * 16)
            reader.release(handle)
            fixture.drain(reader, ref)
            reader.close()
            fixture.close()

    def test_receipt_native_failure_retains_manifest_and_quarantined_charge(self):
        from specforge_sglang_capture.sink import CaptureWriterUncertain
        with tempfile.TemporaryDirectory() as directory, self.environment(directory):
            fixture = Fixture(directory)
            # Drive owner progress ourselves so the injected boundary is exact.
            fixture.owner.stop.set()
            fixture.owner.thread.join(5)
            ref = fixture.publish('a')
            reader = fixture.reader()
            reader.adopt(ref)
            reader.abort_ref(ref)
            with mock.patch.object(fixture.owner, 'put', side_effect=CaptureWriterUncertain('receipt drain unknown')):
                with self.assertRaisesRegex(CaptureWriterUncertain, 'receipt drain unknown'):
                    fixture.owner.progress()
            manifest = ref.metadata['segment_manifest']
            rec = fixture.owner.manifests[manifest['token']]
            self.assertEqual(rec['phase'], 'QUARANTINED')
            health = fixture.owner.health()
            self.assertEqual(health['remote_inflight_bytes'], 0)
            self.assertEqual(health['remote_quarantined_bytes'], 4096)
            self.assertEqual(health['remote_owner_charged_bytes'], 4096)
            self.assertEqual(health['active_manifests'], 1)
            with self.assertRaisesRegex(RuntimeError, 'before ACK/receipt drains'):
                fixture.owner.close()
            # The CPU mock raised before submitting any native operation. This
            # test-only proof permits fixture cleanup; production has no implicit
            # recovery of an unknown native writer.
            fixture.owner.ledger.forget(manifest['receipt_key'])
            fixture.owner.manifests.pop(manifest['token'])
            reader._release_pending.clear()
            reader.close()
            fixture.close()
