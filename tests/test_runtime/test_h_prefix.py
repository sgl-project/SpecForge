"""Committed H governs physical object ranges and bounded Trainer admissions."""
import copy
import os
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch
from specforge.runtime.data_plane.segment_manifest import digest_manifest, validate_manifest, validate_attempt
from specforge.runtime.data_plane.node_pool_batch import process_gate
from specforge_sglang_capture.hs import CaptureController, PrefixFeatureCache
from tests.test_runtime.test_segment_manifest import Fixture, features, schema
from dataclasses import replace


class HPrefixTests(unittest.TestCase):
    def env(self, root):
        return patch.dict(os.environ, {'SPECFORGE_TRAINER_NODE_POOL':'1',
            'SPECFORGE_TRAINER_NODE_POOL_DIR':root+'/pool','SPECFORGE_TRAINER_NODE_POOL_BYTES':str(1<<20),
            'SPECFORGE_TRAINER_NODE_POOL_BATCHED':'1'})

    def test_ranges_reads_reuse_and_proofs(self):
        with tempfile.TemporaryDirectory() as root, self.env(root):
            f = Fixture(root)
            refs = [f.publish('cold',h_prefix=True)]
            refs += [f.publish('hot-'+str(l), l=l, h_prefix=True) for l in (8,5,0)]
            reader = f.reader(receive_buffers='pinned')
            other = f.reader(receive_buffers='pinned')
            for i, ref in enumerate(refs):
                r = other if i >= 2 else reader
                r.adopt(ref)
                m = ref.metadata['segment_manifest']
                self.assertEqual((m['N'],m['H'],m['L']), (9, 8 if i else 0, (0,8,5,0)[i]))
                wanted = features(9)
                targets = {n:torch.empty(ref.feature_specs[n].shape) for n in ('hidden_states','target_last_hidden_states')}
                h0 = r.health()
                sql = []
                r._node_pool._db.set_trace_callback(sql.append)
                _,handle = r.get_into(ref,targets)
                r._node_pool._db.set_trace_callback(None)
                r.release(handle)
                h1 = r.health()
                self.assertEqual(h1['tail_get_calls']-h0['tail_get_calls'],2)
                self.assertEqual(h1['prefix_get_calls']-h0['prefix_get_calls'],4 if i==1 else 0)
                if i==0:self.assertEqual(sql,[])
                self.assertTrue(torch.equal(targets['hidden_states'],wanted['aux'].unsqueeze(0)))
                self.assertTrue(torch.equal(targets['target_last_hidden_states'],wanted['last_hidden'].unsqueeze(0)))
                for segs in m['features'].values():
                    self.assertEqual([s['kind'] for s in segs], ['prefix','prefix','tail'] if i else ['tail'])
                f.drain(r,ref)
            forged = copy.deepcopy(m)
            forged['features']['hidden_states'][-1]['kind'] = 'prefix'
            forged['digest'] = digest_manifest({k:v for k,v in forged.items() if k!='digest'})
            with self.assertRaises(ValueError):validate_manifest(forged,ref.feature_specs,9)
            with self.assertRaises(ValueError):validate_attempt(m,refs[0].metadata['capture_fence'])
            reader.close(); other.close(); f.close()

    def test_crossing_block_reads_fresh_forward_rows(self):
        for l in (0,5):
            with self.subTest(L=l), tempfile.TemporaryDirectory() as root, self.env(root):
                f=Fixture(root)
                cold=f.publish('seed',h_prefix=True)
                suffix={k:v+5000 for k,v in features(9-l,l).items()}
                ref=f.publish('fresh',l=l,h_prefix=True,suffix_values=suffix)
                r=f.reader(receive_buffers='pinned')
                r.adopt(cold); f.drain(r,cold); r.adopt(ref)
                targets={n:torch.empty(ref.feature_specs[n].shape) for n in ('hidden_states','target_last_hidden_states')}
                _,h=r.get_into(ref,targets)
                for name,role in [('hidden_states','aux'),('target_last_hidden_states','last_hidden')]:
                    expected=torch.cat([features(l)[role],suffix[role]])
                    self.assertTrue(torch.equal(targets[name],expected.unsqueeze(0)))
                r.release(h); f.drain(r,ref); r.close(); f.close()

    def test_multislice_and_unknown_native_tail(self):
        with tempfile.TemporaryDirectory() as root, self.env(root), ExitStack() as stack:
            f=Fixture(root,capacity=192<<20)
            f.controller.shutdown()
            base=schema()
            wide=replace(base, features=(replace(base.features[0],width=262144),base.features[1]))
            stack.enter_context(patch('tests.test_runtime.test_segment_manifest.schema',return_value=wide))
            f.cache=PrefixFeatureCache(wide,4,0)
            f.controller=CaptureController(f.cache,wide,128<<20,128<<20,128<<20,segmented=True)
            values={'aux':torch.arange(65*262144,dtype=torch.float32).view(65,-1),'last_hidden':features(65)['last_hidden']}
            ref=f.publish('big',n=65,h_prefix=True,suffix_values=values)
            seg=ref.metadata['segment_manifest']['features']['hidden_states']
            self.assertEqual([x['nbytes'] for x in seg],[64<<20,1<<20])
            r=f.reader(receive_buffers='pinned')
            r.adopt(ref)
            out=torch.empty_like(values['aux']).unsqueeze(0)
            _,h=r.get_into(ref,{'hidden_states':out})
            self.assertTrue(torch.equal(out,values['aux'].unsqueeze(0)))
            r.release(h)
            self.assertEqual(r.health()['tail_get_calls'],2)
            self.assertEqual(r.health()['node_pool_misses'],0)
            r._owns_store=True
            with patch.object(f.backend,'get_into',side_effect=RuntimeError('uncertain native')):
                with self.assertRaisesRegex(RuntimeError,'uncertain native'):r.get_into(ref,{'hidden_states':out})
            self.assertTrue(process_gate().current().receive_only())
            self.assertGreater(r.health()['receive_pool_quarantined_bytes'],0)
            with patch.object(f.backend,'close',return_value=0,create=True):r.close()
            cleanup=f.reader(node_pool_enabled=False);cleanup.adopt(ref);f.drain(cleanup,ref)
            cleanup.close();f.close()
