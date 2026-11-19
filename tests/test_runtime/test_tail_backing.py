"""Teacher CPU backing identity, physical charging, fallbacks and borrower lifetime."""
import gc
import os
import unittest
import weakref
from unittest.mock import patch

import torch

from specforge_sglang_capture.hs import (CaptureController, CapturedSpan, FeatureSchema,
                                         FeatureSpec, PrefixFeatureCache)
from tests.test_runtime.test_hs_runtime import schema,identity,features


class TailBackingTests(unittest.TestCase):
    def test_single_block_charges_both_full_parents_until_borrower_and_writer_retire(self):
        layout=FeatureSchema((FeatureSpec('aux',16384,'bfloat16',(1,)),FeatureSpec('last_hidden',2048,'bfloat16',())))
        cache=PrefixFeatureCache(layout,256,128<<20)
        tokens=tuple(range(256))
        key,=cache.block_keys('ns',tokens)
        pieces={s.name:(torch.zeros((rows,s.width),dtype=torch.bfloat16),)
                for rows,s in zip((2048,16384),layout.features)}
        refs=[weakref.ref(p[0].untyped_storage()) for p in pieces.values()]
        fill=cache.reserve_fill(key,identity())
        self.assertEqual(cache._publish_tail_views(fill,pieces,0,64<<20),'shared')
        stats=cache.stats()
        self.assertEqual(stats['cache_bytes'],128<<20)
        self.assertEqual(stats['cache_logical_bytes'],9<<20)
        lease=cache.lookup_and_pin('ns',tokens,256)
        blocks,release=cache.borrow_prefix_blocks(lease,256)
        cache.release_lease(lease)
        self.assertEqual(blocks[0][1]['aux'].untyped_storage().data_ptr(),pieces['aux'][0].data_ptr())
        replacement,=cache.block_keys('other',tokens)
        self.assertIsNone(cache.reserve_fill(replacement,identity(1)))
        del blocks
        release()
        newer=cache.reserve_fill(replacement,identity(2))
        self.assertIsNotNone(newer)
        self.assertEqual(cache.stats()['evictions'],1)
        self.assertEqual(cache.stats()['tail_backing_cache_bytes'],0)
        self.assertTrue(all(ref() is not None for ref in refs))  # still held by writer
        pieces.clear();gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        cache.abort_fill(newer)
        self.assertEqual(cache.stats()['cache_bytes'],0)

    def make(self,cap=4096,output=4096):
        cache=PrefixFeatureCache(schema(),4,cap)
        with patch.dict(os.environ,{'SPECFORGE_HS_SHARE_TAIL_BACKING':'1'}):
            ctrl=CaptureController(cache,schema(),4096,output,4096,segmented=True)
        return ctrl,cache

    def sample(self,ctrl,cache,i,n,h,l):
        execution=identity(i);tokens=tuple(range(n))
        lease=cache.lookup_and_pin('ns',tokens,n-1)
        self.assertEqual(lease.coverage_end,h)
        lease=cache.shrink_lease(lease,l)
        reservation=ctrl.try_reserve_output(execution,n,l,0)
        plan=ctrl.freeze_plan(execution,'ns',tokens,n,l,lease,reservation,H=h)
        source=features(n-l,l)
        sample=ctrl.submit(plan,CapturedSpan(execution,l,n,source,None)).result(10)
        ctrl.poll_completed()
        return sample,source

    def test_cross_slice_and_cache_budget_use_compact_copy(self):
        for cap,reason in [(4096,'boundary'),(80,'budget')]:
            with self.subTest(reason=reason),patch('specforge.runtime.data_plane.segment_manifest.TAIL_BYTES',48):
                ctrl,cache=self.make(cap)
                s,source=self.sample(ctrl,cache,0,9,0,0)
                self.assertGreater(ctrl.stats()['tail_fill_'+reason+'_blocks'],0)
                for t in source.values():t.zero_()
                s.release()
                # At the smaller capacity the ordinary LRU may evict the first block.
                for entry in cache._entries.values():
                    expected=features(4,entry.key.start)
                    for name,t in entry.features.items():self.assertTrue(torch.equal(t,expected[name]))
                self.assertLessEqual(cache.stats()['cache_peak_bytes'],cap)
                ctrl.shutdown()

    def test_prefix_reregistration_or_compact_fallback_and_cancellation(self):
        for pressure in (False,True):
            with self.subTest(pressure=pressure):
                ctrl,cache=self.make(output=512)
                seed,source=self.sample(ctrl,cache,0,17,0,0)
                parent=weakref.ref(seed.tails['aux'][0].untyped_storage())
                for t in source.values():t.zero_()
                seed.release()
                self.assertIsNotNone(parent())
                hold=ctrl.try_reserve_output(identity(2),20,0,0) if pressure else None
                hot,_=self.sample(ctrl,cache,1,5,4,4)
                aux=hot.segments[0].features['aux']
                self.assertTrue(torch.equal(aux,features(4)['aux']))
                before=ctrl.stats()
                if pressure:
                    self.assertEqual(aux.untyped_storage().nbytes(),32)
                    self.assertEqual(before['prefix_registration_fallback_copy_bytes'],64)
                else:
                    self.assertIs(aux.untyped_storage(),parent())
                    self.assertEqual(before['output_bytes'],80+(17-4)*16)
                self.assertLessEqual(before['output_peak_bytes'],512)
                drain=ctrl.invalidate(identity(1),'cancel with sink reader')
                self.assertFalse(drain.done())
                self.assertEqual(ctrl.stats()['output_bytes'],before['output_bytes'])
                del aux
                hot.release();self.assertTrue(drain.done())
                if hold:ctrl.release_output(hold)
                self.assertEqual(ctrl.stats()['output_bytes'],0)
                self.assertGreater(cache.stats()['tail_backing_cache_bytes'],0)
                ctrl.shutdown()

