# coding=utf-8
"""LocalFeatureStore: atomic put, get, idempotent release, abort, file mode (CPU).

The ``TestFeatureStoreGC`` class covers the M5 reclamation surface: feature-lease
max-hold force-free, the leak gate, and release-pending reconciliation.
"""

import logging
import os
import tempfile
import unittest

import torch

from specforge.runtime.data_plane.feature_store import LocalFeatureStore
from specforge.runtime.data_plane.offline_reader import OfflineManifestReader


class TestLocalFeatureStore(unittest.TestCase):
    def test_put_returns_ref_with_no_tensors(self):
        store = LocalFeatureStore("st")
        tensors = {
            "input_ids": torch.arange(8).view(1, 8),
            "hidden_state": torch.randn(1, 8, 4),
        }
        ref = store.put(
            tensors, sample_id="s0", metadata={"run_id": "r", "num_tokens": 8}
        )
        self.assertEqual(ref.sample_id, "s0")
        self.assertTrue(ref.feature_store_uri.startswith("mem://"))
        self.assertEqual(set(ref.feature_specs), {"input_ids", "hidden_state"})
        self.assertEqual(ref.feature_specs["hidden_state"].shape, (1, 8, 4))
        self.assertGreater(ref.estimated_bytes, 0)

    def test_get_returns_tensors_and_handle(self):
        store = LocalFeatureStore("st")
        t = torch.randn(1, 4, 2)
        ref = store.put({"x": t}, sample_id="s0", metadata={})
        out, handle = store.get(ref)
        self.assertTrue(torch.equal(out["x"], t))
        self.assertEqual(handle.sample_id, "s0")

    def test_release_idempotent_and_stale_safe(self):
        store = LocalFeatureStore("st")
        ref = store.put({"x": torch.randn(1, 4)}, sample_id="s0", metadata={})
        _, h = store.get(ref)
        store.release(h)
        store.release(h)  # idempotent: must not raise
        # re-put bumps generation; old handle release is a no-op
        ref2 = store.put({"x": torch.randn(1, 4)}, sample_id="s0", metadata={})
        _, h2 = store.get(ref2)
        store.release(h)  # stale generation -> no-op
        out, _ = store.get(ref2)
        self.assertIn("x", out)
        _ = h2

    def test_release_frees_mem_on_last_lease(self):
        # mem:// is consume-once: the last release must physically free tensors,
        # otherwise an online put->get->release loop grows _mem unboundedly.
        store = LocalFeatureStore("st")
        ref = store.put({"x": torch.randn(1, 4)}, sample_id="s0", metadata={})
        self.assertEqual(store.health()["resident_samples"], 1)
        _, h = store.get(ref)
        store.release(h)
        self.assertEqual(store.health()["resident_samples"], 0)
        self.assertEqual(store.health()["resident_bytes"], 0)
        with self.assertRaises(KeyError):
            store.get(ref)  # consumed -> gone

    def test_release_keeps_mem_while_other_lease_active(self):
        # refcount: only the LAST lease frees the sample.
        store = LocalFeatureStore("st")
        ref = store.put({"x": torch.randn(1, 4)}, sample_id="s0", metadata={})
        _, h1 = store.get(ref)
        _, h2 = store.get(ref)
        store.release(h1)
        self.assertEqual(store.health()["resident_samples"], 1)  # h2 still holds it
        store.release(h2)
        self.assertEqual(store.health()["resident_samples"], 0)  # last lease -> freed

    def test_online_put_get_release_loop_is_bounded(self):
        # The regression this fix targets: many unique samples, consumed and
        # released, must not accumulate in the store.
        store = LocalFeatureStore("st")
        for i in range(50):
            ref = store.put({"x": torch.randn(1, 8)}, sample_id=f"s{i}", metadata={})
            _, h = store.get(ref)
            store.release(h)
        self.assertEqual(store.health()["resident_samples"], 0)
        self.assertEqual(store.health()["resident_bytes"], 0)

    def test_max_resident_bytes_raises_when_consumer_is_behind(self):
        # one float32 (1,8) sample = 32 bytes; cap at 40 admits one, rejects two.
        store = LocalFeatureStore("st", max_resident_bytes=40)
        ref0 = store.put(
            {"x": torch.zeros(1, 8, dtype=torch.float32)}, sample_id="s0", metadata={}
        )
        with self.assertRaises(MemoryError):
            store.put(
                {"x": torch.zeros(1, 8, dtype=torch.float32)},
                sample_id="s1",
                metadata={},
            )
        # once the first is consumed+released, there is room for the next
        _, h = store.get(ref0)
        store.release(h)
        store.put(
            {"x": torch.zeros(1, 8, dtype=torch.float32)}, sample_id="s1", metadata={}
        )
        self.assertEqual(store.health()["resident_samples"], 1)

    def test_abort_evicts(self):
        store = LocalFeatureStore("st")
        ref = store.put({"x": torch.randn(1, 4)}, sample_id="s0", metadata={})
        store.abort("s0", reason="test")
        with self.assertRaises(KeyError):
            store.get(ref)

    def test_abort_returns_bytes_to_baseline(self):
        # M5 leak gate: residency must return to exactly baseline after aborts,
        # including the per-sample bookkeeping (no _generation/_put_time leak).
        store = LocalFeatureStore("st")
        baseline = store.health()["resident_bytes"]
        for i in range(20):
            store.put({"x": torch.randn(4, 16)}, sample_id=f"s{i}", metadata={})
        self.assertGreater(store.health()["resident_bytes"], baseline)
        for i in range(20):
            store.abort(f"s{i}", reason="aborted")
        h = store.health()
        self.assertEqual(h["resident_bytes"], baseline)
        self.assertEqual(h["resident_samples"], 0)
        self.assertEqual(h["oldest_age_s"], 0.0)  # no put_time entries left

    def test_health(self):
        store = LocalFeatureStore("st")
        store.put({"x": torch.randn(1, 4)}, sample_id="s0", metadata={})
        h = store.health()
        self.assertEqual(h["resident_samples"], 1)
        self.assertGreater(h["resident_bytes"], 0)

    def test_estimate_bytes(self):
        store = LocalFeatureStore("st")
        ref = store.put(
            {"x": torch.zeros(1, 10, dtype=torch.float32)}, sample_id="s0", metadata={}
        )
        self.assertEqual(store.estimate_bytes(ref.feature_specs), 10 * 4)

    def test_disk_dump_tap(self):
        with tempfile.TemporaryDirectory() as d:
            store = LocalFeatureStore("st", dump_dir=d)
            store.put({"x": torch.randn(1, 4)}, sample_id="s0", metadata={})
            self.assertTrue(os.path.exists(os.path.join(d, "s0.ckpt")))

    def test_file_mode_get_matches_offline_format(self):
        # write an offline-style .ckpt and read it back through the store + reader
        with tempfile.TemporaryDirectory() as d:
            raw = {
                "input_ids": torch.arange(8),
                "loss_mask": torch.ones(8, dtype=torch.long),
                "hidden_state": torch.randn(1, 8, 4),
                "aux_hidden_state": torch.randn(1, 8, 12),
            }
            torch.save(raw, os.path.join(d, "000.ckpt"))
            refs = OfflineManifestReader(d, run_id="off").read()
            self.assertEqual(len(refs), 1)
            self.assertTrue(refs[0].feature_store_uri.startswith("file://"))
            self.assertEqual(set(refs[0].feature_specs), set(raw))
            self.assertEqual(refs[0].feature_specs["hidden_state"].dtype, "float32")
            self.assertEqual(refs[0].num_tokens, 8)
            store = LocalFeatureStore("st")
            out, handle = store.get(refs[0])
            self.assertEqual(set(out), set(raw))
            self.assertTrue(
                torch.equal(out["aux_hidden_state"], raw["aux_hidden_state"])
            )
            store.release(handle)

    def test_offline_reader_rejects_missing_required_key(self):
        with tempfile.TemporaryDirectory() as d:
            torch.save({"input_ids": torch.arange(4)}, os.path.join(d, "bad.ckpt"))
            with self.assertRaises(KeyError):
                OfflineManifestReader(d, run_id="off").read()

    def test_mem_ref_carries_generation_and_rejects_stale_ref(self):
        # The mem:// ref carries the generation it was minted for; once a sample
        # is reclaimed and a new generation is published under the same id, the
        # stale ref must be rejected rather than silently aliasing fresh data.
        store = LocalFeatureStore("st")
        ref1 = store.put({"x": torch.randn(1, 8)}, sample_id="s0", metadata={})
        self.assertIn("generation=", ref1.feature_store_uri)
        _, h = store.get(ref1)
        store.release(h)  # gen1 reclaimed
        store.put({"x": torch.randn(1, 8)}, sample_id="s0", metadata={})  # gen2
        with self.assertRaises(KeyError):
            store.get(ref1)  # stale gen1 ref -> rejected

    def test_release_after_reput_while_leased_does_not_leak(self):
        # Re-put a sample_id while an older generation is still leased, then drop
        # the newest handle and finally the stale old handle. Freeing is keyed on
        # the CURRENT generation's last lease, so the current generation must be
        # freed and nothing leaks.
        store = LocalFeatureStore("st")
        ref1 = store.put({"x": torch.randn(1, 8)}, sample_id="s0", metadata={})
        _, h1 = store.get(ref1)  # lease on gen1
        ref2 = store.put({"x": torch.randn(1, 8)}, sample_id="s0", metadata={})  # gen2
        _, h2 = store.get(ref2)  # lease on gen2
        store.release(h2)  # newest released first (stale gen1 lease still active)
        store.release(h1)  # stale gen1 handle released last
        h = store.health()
        self.assertEqual(h["resident_samples"], 0)
        self.assertEqual(h["resident_bytes"], 0)

    def test_dump_failure_does_not_abort_publish(self):
        # The disk dump is a best-effort capture/replay tap; mem is authoritative.
        # A dump failure must not undo an otherwise successful in-memory publish.
        class DumpFails(LocalFeatureStore):
            def _dump(self, sample_id, tensors):
                raise RuntimeError("disk full")

        with tempfile.TemporaryDirectory() as d:
            store = DumpFails("st", dump_dir=d)
            logging.disable(logging.CRITICAL)  # silence the expected warning
            try:
                ref = store.put({"x": torch.randn(1, 4)}, sample_id="s0", metadata={})
            finally:
                logging.disable(logging.NOTSET)
            out, h = store.get(ref)  # mem publish survived
            self.assertIn("x", out)
            store.release(h)


class _FakeClock:
    """Deterministic monotonic clock for GC/age tests (no real sleeping)."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class TestFeatureStoreGC(unittest.TestCase):
    def test_gc_force_frees_past_max_hold_age(self):
        clock = _FakeClock()
        store = LocalFeatureStore("st", max_hold_age_s=10.0, clock=clock)
        store.put({"x": torch.randn(1, 8)}, sample_id="old", metadata={})
        clock.advance(20.0)  # "old" is now past max-hold
        store.put({"x": torch.randn(1, 8)}, sample_id="new", metadata={})
        report = store.gc()
        self.assertEqual(report["force_freed"], 1)
        self.assertGreater(report["force_freed_bytes"], 0)
        self.assertEqual(store.health()["resident_samples"], 1)  # only "new" left

    def test_gc_spares_leased_sample_even_if_old(self):
        # A sample still leased by a (slow) trainer must NOT be force-freed, even
        # past max-hold — that would be a use-after-free for the holder.
        clock = _FakeClock()
        store = LocalFeatureStore("st", max_hold_age_s=10.0, clock=clock)
        ref = store.put({"x": torch.randn(1, 8)}, sample_id="held", metadata={})
        _, h = store.get(ref)  # active lease
        clock.advance(100.0)
        report = store.gc()
        self.assertEqual(report["force_freed"], 0)
        self.assertEqual(store.health()["resident_samples"], 1)
        store.release(h)  # once released, normal consume-once free applies
        self.assertEqual(store.health()["resident_samples"], 0)

    def test_gc_no_max_hold_is_noop(self):
        clock = _FakeClock()
        store = LocalFeatureStore("st", clock=clock)  # max_hold_age_s=None
        store.put({"x": torch.randn(1, 8)}, sample_id="s0", metadata={})
        clock.advance(10_000.0)
        self.assertEqual(store.gc()["force_freed"], 0)
        self.assertEqual(store.health()["resident_samples"], 1)

    def test_release_after_reput_while_leased_does_not_leak(self):
        # Regression for the consume-once leak: re-put a sample_id while an older
        # lease is still outstanding, then release the newest handle and finally
        # the stale old handle. The current generation MUST be freed.
        store = LocalFeatureStore("st")
        ref1 = store.put({"x": torch.randn(1, 8)}, sample_id="s0", metadata={})
        _, h1 = store.get(ref1)  # lease on gen1
        ref2 = store.put({"x": torch.randn(1, 8)}, sample_id="s0", metadata={})  # gen2
        _, h2 = store.get(ref2)  # lease on gen2
        store.release(h2)  # newest released first (h1/gen1 still active)
        store.release(h1)  # stale gen1 handle released last
        h = store.health()
        self.assertEqual(h["resident_samples"], 0)  # was leaking before the fix
        self.assertEqual(h["resident_bytes"], 0)

    def test_stale_release_does_not_free_unconsumed_reput(self):
        # The reverse guard: a stale double-release must NOT free a freshly
        # re-put sample that nobody has consumed (leased) yet.
        store = LocalFeatureStore("st")
        ref1 = store.put({"x": torch.randn(1, 8)}, sample_id="s0", metadata={})
        _, h1 = store.get(ref1)
        store.release(h1)  # frees gen1
        store.put({"x": torch.randn(1, 8)}, sample_id="s0", metadata={})  # gen2, no get
        store.release(h1)  # stale double-release of the gen1 handle
        self.assertEqual(store.health()["resident_samples"], 1)  # gen2 preserved

    def test_release_pending_reconciled_by_gc(self):
        # A backend whose physical free fails parks the sample release-pending;
        # gc() retries within the bounded window, then force-frees.
        class FlakyFreeStore(LocalFeatureStore):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.fail_remaining = 5  # always fail retry -> hit force-free

            def _try_physical_free(self, sample_id):
                if self.fail_remaining > 0:
                    self.fail_remaining -= 1
                    return False
                return super()._try_physical_free(sample_id)

        store = FlakyFreeStore("st", max_release_attempts=3)
        ref = store.put({"x": torch.randn(1, 8)}, sample_id="s0", metadata={})
        _, h = store.get(ref)
        store.release(h)  # free fails -> parked release-pending
        self.assertEqual(store.health()["release_pending"], 1)
        self.assertEqual(store.health()["resident_samples"], 1)  # still resident
        store.gc()  # attempt 1
        self.assertEqual(store.health()["release_pending"], 1)
        store.gc()  # attempt 2
        store.gc()  # attempt 3 -> bounded window exhausted -> force-free
        self.assertEqual(store.health()["release_pending"], 0)
        self.assertEqual(store.health()["resident_samples"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
