"""CPU store integration: real materialization/trace and owned drain ordering."""
import json
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from specforge.runtime.contracts import FeatureSpec
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
from specforge.runtime.data_plane.node_content_pool import NodeContentPool
from specforge.runtime.data_plane.node_pool_batch import process_gate
from specforge.runtime.data_plane.pool_diagnostics import Diagnostics
from specforge.runtime.data_plane.feature_dataloader import FeatureDataLoader
from specforge.algorithms.common.hidden_states_data import build_collator
from tests.test_runtime.test_mooncake_store import _FakeMooncakeStore
from tests.test_runtime.test_segment_manifest import Fixture


class ReceiveStub:
    """CPU lifetime model, deliberately no CUDA or actual native DMA."""
    def __init__(self, events):
        self.kind = 'pinned'
        self._lock = threading.RLock()
        self._active = []
        self.quarantined = 8
        self.events = events
    def health(self):
        return {'receive_pool_quarantined_bytes': self.quarantined}
    def close(self, *, transport_stopped=False):
        self.events.append('receive-drained' if transport_stopped else 'receive-sealed')
        if self._active:
            raise RuntimeError('active receive')
        if transport_stopped:
            self.quarantined = 0


class TestBatchedSegmentStore(unittest.TestCase):
    def setUp(self):
        self.environment = mock.patch.dict(os.environ, {'SPECFORGE_TRAINER_NODE_POOL_BATCHED':'1',
            'SPECFORGE_TRAINER_NODE_POOL_WINDOW_BYTES':str(64<<20),
            'SPECFORGE_TRAINER_NODE_POOL_WINDOW_BLOCKS':'8',
            'SPECFORGE_TRAINER_NODE_POOL_WINDOW_INFLIGHT':'1'})
        self.environment.start();self.addCleanup(self.environment.stop)
        self.assertIsNone(process_gate().current())

    def store(self, directory):
        backend=_FakeMooncakeStore()
        store=MooncakeFeatureStore(store=backend,store_id='cpu',node_pool_enabled=False,receive_buffers='pinned')
        store._node_pool=NodeContentPool(directory+'/pool',65536)
        return store,backend

    def ref(self, store, size=8):
        segment={'logical_key':'namespace/schema/content/aux','key':'physical',
                 'publication_generation':'publication','nbytes':size,'shape':[1,size,1],
                 'start':0,'end':size}
        store._segment_manifests['sample']={'namespace':'namespace','owner_epoch':'owner',
                                          'schema_signature':'schema','features':{'x':[segment]}}
        return SimpleNamespace(sample_id='sample',feature_specs={'x':FeatureSpec('x',(1,size,1),'uint8')},
                               metadata={'capture_fence':{'server_epoch':'load'}})

    def receive_window(self, directory, *, owned=True, native_rc=0, h2d=False):
        store,backend=self.store(directory);events=[]
        receive=ReceiveStub(events);store._receive_pool=receive;store._owns_store=owned
        backend.close=lambda: events.append('native-close') or native_rc
        window=store._node_pool.acquire_many([('key',8,lambda:torch.ones(8,dtype=torch.uint8))])
        window.__enter__();window.hold_external('receive')
        store._batch_receive_windows[window.id]=(window,backend,receive)
        if h2d:
            window.defer(0)
        window.close()
        return store,backend,receive,window,events

    def test_loader_values_and_trace_lookup_order_and_real_released_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            env={'SPECFORGE_TRAINER_DIRECT_BATCH':'1','SPECFORGE_TRAINER_NODE_POOL':'1',
                 'SPECFORGE_TRAINER_NODE_POOL_DIR':directory+'/pool','SPECFORGE_TRAINER_NODE_POOL_BYTES':str(1<<20)}
            with mock.patch.dict(os.environ,env), mock.patch('specforge.runtime.data_plane.pool_diagnostics.atexit.register'):
                diag=Diagnostics(directory+'/trace');diag.phase_name='measure'
                with mock.patch('specforge.runtime.data_plane.mooncake_store.get_diagnostics',return_value=diag),mock.patch('specforge.runtime.data_plane.node_content_pool.get_diagnostics',return_value=diag):
                    fixture=Fixture(directory);ref=fixture.publish('batch',n=65)
                    reader=fixture.reader(receive_buffers='pinned')
                    loader=FeatureDataLoader(reader,refs=[ref],batch_size=1,strategy='dflash',collate_fn=build_collator(),ack=False,num_workers=1)
                    cold=next(iter(loader));warm=next(iter(loader))
                    for name,value in cold.tensors.items():torch.testing.assert_close(warm.tensors[name],value)
                    self.assertGreater(reader.health()['node_pool_hits'],0)
                    self.assertEqual(reader.health()['node_pool_active_pins'],0)
                    diag.flush()
                    events=[json.loads(line) for line in diag.path.read_text().splitlines()]
                    definitions={r['id']:r for r in events if isinstance(r,dict) and r.get('type')=='key'}
                    lookups=[definitions[r[3]]['logical_key'] for r in events if isinstance(r,list) and r[0]=='lookup']
                    expected=[s['logical_key'] for name in ('hidden_states','target_last_hidden_states') for s in ref.metadata['segment_manifest']['features'][name]]
                    self.assertEqual(lookups,expected*2)
                    finishes=[r for r in events if isinstance(r,list) and r[0]=='finish']
                    self.assertTrue(all(r[6]=={'copy_completed':True,'pin_released':True} for r in finishes))
                    self.assertEqual(len(finishes),len(lookups))
                    loader.close();fixture.drain(reader,ref);reader.close();fixture.close()
                diag.file.close()

    def test_oversize_single_fallback_copies_real_bytes_and_hard_reject_does_not_fetch(self):
        with tempfile.TemporaryDirectory() as directory:
            store,backend=self.store(directory);size=(8<<20)+1;ref=self.ref(store,size)
            backend._d['physical']=bytes([17])*size
            destination=torch.empty((1,size,1),dtype=torch.uint8)
            store._get_segments(ref,'x',destination)
            self.assertTrue(torch.all(destination==17))
            self.assertEqual(store._node_pool.health()['node_pool_misses'],0)
            self.assertEqual(store._node_pool.health()['node_pool_fallbacks'],1)
            self.assertEqual(store._node_pool._batch_high_water['oversize_fallback_bytes'],size)
            ref=self.ref(store,(64<<20)+1)
            with mock.patch.object(store,'_fetch_segment_source',side_effect=AssertionError('oversize native fetch')):
                with self.assertRaisesRegex(ValueError,'hard batch/fallback'):
                    store._get_segments(ref,'x',torch.empty(1))
            store.close()

    def test_incorrect_receive_modes_are_rejected_before_fetch_or_pool_join(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ,{'SPECFORGE_TRAINER_NODE_POOL_DIR':directory+'/rejected'}):
                with self.assertRaisesRegex(ValueError,'requires receive_buffers=pinned'):
                    MooncakeFeatureStore(store=_FakeMooncakeStore(),node_pool_enabled=True,receive_buffers='pageable')
            self.assertFalse((Path(directory)/'rejected/data.mmap').exists())
            store,backend=self.store(directory);ref=self.ref(store);receive=store._receive_pool
            for wrong in (None,SimpleNamespace(kind='cuda')):
                store._receive_pool=wrong
                with mock.patch.object(store,'_fetch_segment_source',side_effect=AssertionError('wrong-mode fetch')):
                    with self.assertRaisesRegex(ValueError,'pinned receive pool before fetch'):
                        store._get_segments(ref,'x',torch.empty((1,8,1),dtype=torch.uint8))
            store._receive_pool=receive;store.close()

    def test_pre_admission_observer_failure_keeps_primary_and_public_close_progresses(self):
        for variant in ('count','observe'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as directory:
                store,backend=self.store(directory);ref=self.ref(store)
                pool=store._node_pool;primary=RuntimeError('observer before admission sentinel')
                original=pool._diag.count
                def observed(name,n=1):
                    if name=='batch_windows':raise primary
                    return original(name,n)
                patch=(mock.patch.object(pool._diag,'count',side_effect=observed) if variant=='count'
                       else mock.patch('specforge.runtime.data_plane.node_pool_batch.BatchWindow.observe',side_effect=primary))
                with patch,mock.patch.object(store,'_fetch_segment_source',side_effect=AssertionError('native before observer settlement')):
                    with self.assertRaises(RuntimeError) as caught:store._get_segments(ref,'x',torch.empty((1,8,1),dtype=torch.uint8))
                self.assertIs(caught.exception,primary)
                self.assertEqual(pool.health()['node_pool_active_pins'],0)
                self.assertEqual(pool.health()['node_pool_inflight_bytes'],0)
                self.assertIsNone(process_gate().current())
                self.assertFalse(pool._batch_windows)
                store.close()
                self.assertTrue(store._closed)

    def test_fetch_exception_before_returning_source_keeps_receive_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            store,backend=self.store(directory);events=[];receive=ReceiveStub(events);store._receive_pool=receive
            ref=self.ref(store);error=RuntimeError('native uncertain sentinel')
            with mock.patch.object(store,'_fetch_segment_source',side_effect=error):
                with self.assertRaises(RuntimeError) as caught:store._get_segments(ref,'x',torch.empty((1,8,1),dtype=torch.uint8))
            self.assertIs(caught.exception,error)
            window=process_gate().current();self.assertIsNotNone(window)
            self.assertTrue(window.receive_only())
            self.assertEqual(store._node_pool.health()['node_pool_inflight_bytes'],0)
            with self.assertRaisesRegex(RuntimeError,'owned transport drain'):store.close()
            receive.close(transport_stopped=True);window.resolve_external('receive');store.close()

    def test_owned_native_and_receive_close_prove_late_cleanup_before_node_close(self):
        with tempfile.TemporaryDirectory() as directory:
            store,backend,receive,window,events=self.receive_window(directory)
            store.close()
            self.assertTrue(store._closed)
            self.assertTrue(store._transport_stopped)
            self.assertEqual(events,['receive-sealed','native-close','receive-drained'])
            self.assertIsNone(process_gate().current())
            self.assertFalse((Path(directory)/'pool/data.mmap').exists())

    def test_borrowed_and_wrong_owner_binding_do_not_prove_drain(self):
        for owned,wrong in [(False,False),(True,True)]:
            with self.subTest(owned=owned),tempfile.TemporaryDirectory() as directory:
                store,backend,receive,window,events=self.receive_window(directory,owned=owned)
                if wrong:store._batch_receive_windows[window.id]=(window,object(),receive)
                with self.assertRaisesRegex(RuntimeError,'owned transport drain|owner binding differs'):store.close()
                self.assertEqual(events,[])
                self.assertIs(process_gate().current(),window)
                store._batch_receive_windows[window.id]=(window,backend,receive)
                receive.close(transport_stopped=True);window.resolve_external('receive');store._owns_store=False;store.close()

    def test_native_close_failure_keeps_gate_and_retry_requires_actual_success(self):
        with tempfile.TemporaryDirectory() as directory:
            store,backend,receive,window,events=self.receive_window(directory,native_rc=-1)
            with self.assertRaisesRegex(RuntimeError,'transport close failed'):store.close()
            self.assertIs(process_gate().current(),window)
            self.assertEqual(receive.quarantined,8)
            self.assertNotIn('receive-drained',events)
            backend.close=lambda:events.append('native-close-success') or 0
            store.close()
            self.assertTrue(store._closed)
            self.assertEqual(receive.quarantined,0)
            self.assertIsNone(process_gate().current())

    def test_public_get_into_release_commit_failure_then_public_close_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            env={'SPECFORGE_TRAINER_NODE_POOL':'1','SPECFORGE_TRAINER_NODE_POOL_DIR':directory+'/pool',
                 'SPECFORGE_TRAINER_NODE_POOL_BYTES':str(1<<20)}
            with mock.patch.dict(os.environ,env):
                fixture=Fixture(directory);ref=fixture.publish('public-close',n=9)
                reader=fixture.reader(receive_buffers='pinned');reader.adopt(ref)
                # Exercise the owned close contract. The CPU backend preserves
                # its remote dictionary for a separate later cleanup client.
                fixture.backend.close=lambda:0
                reader._owns_store=True
                destination=torch.empty(ref.feature_specs['hidden_states'].shape,dtype=torch.float32)
                pool=reader._node_pool;original=pool._transaction;error=RuntimeError('one release commit failure')
                @contextmanager
                def failed_release(scope='unspecified'):
                    with original(scope) as db:
                        yield db
                        if scope=='batch_release':raise error
                with mock.patch.object(pool,'_transaction',failed_release):
                    with self.assertRaises(RuntimeError) as caught:reader.get_into(ref,{'hidden_states':destination})
                self.assertIs(caught.exception,error)
                window=process_gate().current();self.assertIsNotNone(window)
                self.assertTrue(window.closing)
                self.assertGreater(pool.health()['node_pool_active_pins'],0)
                # The previous close implementation never retried this ended
                # window. It must still refuse if that retry is disabled.
                with mock.patch.object(pool,'retry_batch_cleanup',return_value=None):
                    with self.assertRaisesRegex(RuntimeError,'unsettled batch windows'):reader.close()
                reader.close()
                self.assertTrue(reader._closed)
                self.assertIsNone(process_gate().current())
                self.assertFalse((Path(directory)/'pool/data.mmap').exists())
                cleanup=fixture.reader(node_pool_enabled=False)
                cleanup.adopt(ref);fixture.drain(cleanup,ref);cleanup.close()
                fixture.close()

    def test_proved_db_survives_failed_close_then_release_recovery_and_close_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            env={'SPECFORGE_TRAINER_NODE_POOL':'1','SPECFORGE_TRAINER_NODE_POOL_DIR':directory+'/pool',
                 'SPECFORGE_TRAINER_NODE_POOL_BYTES':str(1<<20)}
            with mock.patch.dict(os.environ,env):
                fixture=Fixture(directory)
                acknowledged=fixture.publish('already-proved',n=9)
                reader=fixture.reader(receive_buffers='pinned');reader.adopt(acknowledged)
                fixture.drain(reader,acknowledged)
                proofs=reader._terminal_proofs
                self.assertEqual(proofs.db.execute("SELECT COUNT(*) FROM proofs WHERE phase='PROVED'").fetchone()[0],1)
                self.assertFalse(proofs.closed)
                ref=fixture.publish('close-retry-with-proof',n=9);reader.adopt(ref)
                native_close=[];fixture.backend.close=lambda:native_close.append(1) or 0
                reader._owns_store=True
                pool=reader._node_pool;original=pool._transaction
                failed=RuntimeError('release remains unavailable through first close')
                @contextmanager
                def failing_release(scope='unspecified'):
                    with original(scope) as db:
                        yield db
                        if scope=='batch_release':raise failed
                destination=torch.empty(ref.feature_specs['hidden_states'].shape,dtype=torch.float32)
                with mock.patch.object(pool,'_transaction',failing_release):
                    with self.assertRaises(RuntimeError) as caught:reader.get_into(ref,{'hidden_states':destination})
                    self.assertIs(caught.exception,failed)
                    with self.assertRaises(RuntimeError) as caught:reader.close()
                    self.assertIs(caught.exception,failed)
                self.assertTrue(proofs.closed)
                cached_health=dict(reader._terminal_proof_health)
                self.assertTrue(cached_health)
                self.assertEqual(native_close,[])
                self.assertIsNotNone(process_gate().current())
                reader.close()
                self.assertTrue(reader._closed)
                self.assertEqual(reader._terminal_proof_health,cached_health)
                self.assertEqual(native_close,[1])
                self.assertIsNone(process_gate().current())
                self.assertFalse((Path(directory)/'pool/data.mmap').exists())
                reader.close()
                self.assertEqual(native_close,[1])
                cleanup=fixture.reader(node_pool_enabled=False)
                cleanup.adopt(ref);fixture.drain(cleanup,ref);cleanup.close();fixture.close()

    def test_h2d_pending_cannot_use_native_shutdown_as_cuda_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            store,backend,receive,window,events=self.receive_window(directory,h2d=True)
            ready=threading.Event()
            class Stream:
                def synchronize(self):
                    if not ready.is_set():raise RuntimeError('CUDA completion still unknown')
            pin=window.pins[0]
            store._h2d_quarantined.append({'destination':torch.empty(8,dtype=torch.uint8),
                'source':pin.tensor,'stream':Stream(),'pin':pin,'batch_window':window,'batch_index':0})
            token=pin.token
            with self.assertRaisesRegex(RuntimeError,'completion still unknown'):store.close()
            self.assertEqual(pin.token,token)
            self.assertEqual(events,[])
            self.assertIs(process_gate().current(),window)
            ready.set()
            original=store._node_pool._transaction
            @contextmanager
            def failed_release(scope='unspecified'):
                with original(scope) as db:
                    yield db
                    if scope=='batch_release':raise RuntimeError('late pin delete failed')
            with mock.patch.object(store._node_pool,'_transaction',failed_release):
                with self.assertRaisesRegex(RuntimeError,'late pin delete failed'):store.close()
            self.assertEqual(pin.token,token)
            self.assertEqual(events,[])
            self.assertIs(process_gate().current(),window)
            store.close()
            self.assertIsNone(pin.token)
            self.assertTrue(store._closed)
            self.assertIsNone(process_gate().current())


if __name__=='__main__':unittest.main()
