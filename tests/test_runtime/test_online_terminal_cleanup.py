"""Producer terminal scan reclaims handed-off refs only after proven success."""
import ctypes
import os
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from unittest import mock

import torch

from specforge.config import Config
from specforge.inference.capture_protocol import attempt_keys
from specforge.runtime.contracts import FeatureSpec, SampleRef
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel
from specforge.training.disaggregated import _build_online, _cleanup_online_published_refs
from tests.test_algorithms.fixtures import builtin_algorithm_registry
from tests.test_runtime.test_server_capture import _FakeMooncakeStore


class Backend(_FakeMooncakeStore):
    def __init__(self):
        super().__init__()
        self.close_calls = 0
        self.remove_keys = []
        self.fail_remove = False

    def remove(self, key, **kwargs):
        self.remove_keys.append(key)
        return -1 if self.fail_remove else super().remove(key)

    def close(self):
        self.close_calls += 1


class TerminalCleanupTests(unittest.TestCase):
    def setUp(self):
        self.backend = Backend()
        self.store = MooncakeFeatureStore(store_id='store', store=self.backend)
        self.names = ('hidden', 'target', 'input_ids', 'loss_mask')

    def ref(self, sid, token):
        wire = dict(protocol_version=1, store_id='store', sample_id=sid, gen=1,
                    server_epoch='epoch', attempt_id=token + '-attempt', reservation_token=token,
                    object_key_layout='attempt-v1', publication_ownership='attempt-v1',
                    writer_drained=True, state='PUBLISHED', control_url='http://forbidden/control')
        wire['physical_keys'] = attempt_keys(wire, self.names)
        data = torch.ones(1, 2, dtype=torch.float32)
        raw = ctypes.string_at(data.data_ptr(), data.numel() * data.element_size())
        self.backend._d.update({key: raw for key in wire['physical_keys'].values()})
        return SampleRef(sample_id=sid, run_id='run', source_task_id=sid,
            feature_store_uri='mooncake://store/' + sid, feature_keys=wire['physical_keys'],
            feature_specs={name: FeatureSpec(name=name, shape=(1, 2), dtype='float32') for name in self.names},
            strategy='dflash', schema_version=1, target_model_version='target', tokenizer_version='tok',
            num_tokens=2, estimated_bytes=32, metadata={'generation': 1, 'capture_fence': wire})

    def handoff(self, ref):
        self.store.adopt(ref)
        self.store.relinquish_published(ref)

    def test_terminal_api_rejects_missing_success_or_closed_proof(self):
        ref = self.ref('late', 'late-token')
        self.handoff(ref)
        for done, closed in ((False, False), (True, False), (False, True), (1, True)):
            with self.subTest(done=done, closed=closed), self.assertRaisesRegex(RuntimeError, 'unresolved'):
                self.store.cleanup_terminal_refs([ref], consumer_done=done, producer_closed=closed)
            self.assertEqual(len(self.backend._d), 4)
        self.assertIsNone(self.store._terminal_cleanup_authority)
        self.assertEqual(self.backend.close_calls, 0)

    def test_terminal_scan_preserves_same_id_descendant_not_in_manifest(self):
        old = self.ref('same', 'old')
        self.handoff(old)
        new = self.ref('same', 'new')
        self.store.adopt(new)
        new_keys = set(new.feature_keys.values())
        with tempfile.TemporaryDirectory() as directory:
            channel = StreamingRefChannel(str(Path(directory) / 'refs'))
            channel.publish(old)
            channel.mark_consumer_done()
            channel.close()
            with mock.patch('specforge.inference.capture_protocol.http_control', side_effect=AssertionError('terminal control RPC')):
                _cleanup_online_published_refs(self.store, channel)
                _cleanup_online_published_refs(self.store, channel)  # Repeated scan is idempotent.
        self.assertEqual(set(self.backend._d), new_keys)
        self.assertEqual(self.store._capture_fences['same'].token, 'new')
        self.assertEqual(self.backend.close_calls, 0)

    def test_failure_sentinel_wins_over_done_and_preserves_handoff_refs(self):
        ref = self.ref('late', 'late')
        self.handoff(ref)
        with tempfile.TemporaryDirectory() as directory:
            channel = StreamingRefChannel(str(Path(directory) / 'refs'))
            channel.publish(ref)
            channel.mark_consumer_done()
            channel.mark_consumer_failed('rank may still own a reader')
            channel.close()
            self.assertFalse(channel.consumer_done())
            self.assertTrue(channel.consumer_stopped())
            with self.assertRaisesRegex(RuntimeError, 'unresolved'):
                _cleanup_online_published_refs(self.store, channel)
        self.assertEqual(len(self.backend._d), 4)
        self.assertEqual(self.backend.remove_keys, [])

    def test_terminal_identity_mismatch_never_falls_back_to_legacy_keys(self):
        ref = self.ref('sample', 'token')
        self.handoff(ref)
        for bad in (replace(ref, sample_id='other'),
                    replace(ref, metadata={**ref.metadata, 'generation': 2})):
            with self.subTest(sid=bad.sample_id, generation=bad.metadata['generation']), self.assertRaisesRegex(RuntimeError, 'identity mismatch'):
                self.store.cleanup_terminal_refs([bad], consumer_done=True, producer_closed=True)
        self.assertEqual(self.backend.remove_keys, [])
        self.assertEqual(len(self.backend._d), 4)

    def test_terminal_uri_and_repeated_bad_proofs_are_validated_before_skip(self):
        ref = self.ref('sample', 'token')
        self.handoff(ref)
        bad_uri = replace(ref, feature_store_uri='mooncake://different-store/sample')
        with self.assertRaisesRegex(RuntimeError, 'URI identity mismatch'):
            self.store.cleanup_terminal_refs([bad_uri], consumer_done=True, producer_closed=True)
        self.assertEqual(self.backend.remove_keys, [])
        self.assertIsNone(self.store._terminal_cleanup_authority)
        self.store.cleanup_terminal_refs([ref], consumer_done=True, producer_closed=True)
        removed = list(self.backend.remove_keys)
        bad_generation = replace(ref, metadata={**ref.metadata, 'generation': 2})
        bad_state = replace(ref, metadata={**ref.metadata, 'capture_fence': {
            **ref.metadata['capture_fence'], 'state': 'DRAINING'}})
        for bad in (bad_uri, bad_generation, bad_state):
            with self.assertRaisesRegex(RuntimeError, 'unresolved'):
                self.store.cleanup_terminal_refs([bad], consumer_done=True, producer_closed=True)
        self.assertEqual(self.backend.remove_keys, removed)

    def test_missing_producer_eof_refuses_actual_terminal_scan(self):
        ref = self.ref('late', 'late')
        self.handoff(ref)
        with tempfile.TemporaryDirectory() as directory:
            channel = StreamingRefChannel(str(Path(directory) / 'refs'))
            channel.publish(ref)
            channel.mark_consumer_done()
            with self.assertRaisesRegex(RuntimeError, 'closed producer'):
                _cleanup_online_published_refs(self.store, channel)
        self.assertEqual(self.backend.remove_keys, [])
        self.assertEqual(len(self.backend._d), 4)

    def test_failed_delete_keeps_cleanup_authority_and_retries_exact_keys(self):
        ref = self.ref('late', 'late')
        self.handoff(ref)
        self.backend.fail_remove = True
        drain_samples = MooncakeFeatureStore.drain_sample_removals
        drain_pending = MooncakeFeatureStore.drain_pending_removals
        def bounded_samples(store, ids):
            return drain_samples(store, ids, max_attempts=1, retry_interval_s=0)
        def bounded_pending(store):
            return drain_pending(store, max_attempts=1, retry_interval_s=0)
        with mock.patch.object(MooncakeFeatureStore, 'drain_sample_removals', bounded_samples), mock.patch.object(MooncakeFeatureStore, 'drain_pending_removals', bounded_pending):
            with self.assertRaisesRegex(RuntimeError, 'unresolved'):
                self.store.cleanup_terminal_refs([ref], consumer_done=True, producer_closed=True)
            authority = self.store._terminal_cleanup_authority
            self.assertIn(ref.sample_id, authority._release_pending)
            self.assertEqual(len(self.backend._d), 4)
            self.backend.fail_remove = False
            with self.assertRaises(RuntimeError):
                self.store.cleanup_terminal_refs([ref], consumer_done=False, producer_closed=True)
            self.assertEqual(len(self.backend._d), 4)
            self.store.cleanup_terminal_refs([ref], consumer_done=True, producer_closed=True)
            self.assertIs(self.store._terminal_cleanup_authority, authority)
            self.assertFalse(authority._release_pending)
        self.assertFalse(self.backend._d)
        self.assertEqual(self.backend.close_calls, 0)

    def test_real_produce_terminal_scan_removes_33_late_refs_and_accepts_already_acked(self):
        refs = [self.ref(str(index), 'token-' + str(index)) for index in range(34)]
        for ref in refs:
            self.handoff(ref)
        consumer = MooncakeFeatureStore(store_id='store', store=self.backend)
        with mock.patch('specforge.inference.capture_protocol.http_control', side_effect=AssertionError('normal/terminal control RPC')):
            _, handle = consumer.get(refs[0])
            consumer.release(handle)
            self.assertEqual(len(self.backend._d), 33 * 4)
            with tempfile.TemporaryDirectory() as directory:
                channel_path = str(Path(directory) / 'refs')
                cfg = Config.model_validate({
                    'model': {'target_model_path': 'target'},
                    'draft': 'specforge/config/draft/presets/qwen3-8b-dflash.yaml',
                    'data': {'prompts_path': 'unused'},
                    'training': {'role': 'producer', 'max_steps': 30, 'total_steps': 1197},
                    'deployment': {'mode': 'disaggregated', 'disaggregated': {
                        'control_dir': directory, 'backend': 'mooncake', 'server_urls': ['http://capture']}}})
                def drive(*, should_stop):
                    self.assertFalse(should_stop())
                    channel = StreamingRefChannel(channel_path)
                    channel.publish(refs[0])
                    channel.mark_consumer_done()
                    # Match the observed race: completed captures are published
                    # after consumer_done, then workers finish and EOF is sealed.
                    for ref in refs[1:]:
                        channel.publish(ref)
                    channel.close()
                    return len(refs)
                with (
                    mock.patch.dict(os.environ, {'DISAGG_REF_CHANNEL': channel_path}),
                    mock.patch('specforge.training.disaggregated._mooncake_store', return_value=self.store),
                    mock.patch('specforge.training.assembly._load_input_tools'),
                    mock.patch('specforge.training.model_loading.resolve_draft_config'),
                    mock.patch('specforge.training.disaggregated._producer_capture_metadata', return_value=([1], 2, 4, 4)),
                    mock.patch('specforge.inference.adapters.server_capture.SGLangServerCaptureAdapter'),
                    mock.patch('specforge.launch.build_disagg_online_producer', return_value=([], drive)),
                ):
                    run = _build_online(cfg, algorithm=builtin_algorithm_registry().resolve('dflash'),
                        build_model_bundle=mock.Mock(),
                        prepare_prompts=lambda *args, **kwargs: [{'task_id': 'p', 'payload': {'input_ids': [1, 2]}}],
                        optimizer_factory=mock.Mock(), logger=None)
                    self.assertEqual(run.run(), 34)
        self.assertFalse(self.backend._d)
        self.assertEqual(self.backend.close_calls, 0)
        authority = self.store._terminal_cleanup_authority
        self.assertIsNotNone(authority)
        self.assertFalse(authority._owns_store)
        self.assertFalse(authority._release_pending)
        self.assertFalse(authority._capture_fences)
        with self.assertRaises(KeyError):
            self.store.adopt(refs[-1])  # Normal handoff tombstones were not reopened.


if __name__ == '__main__':
    unittest.main()
