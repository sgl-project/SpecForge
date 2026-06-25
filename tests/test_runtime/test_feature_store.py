# coding=utf-8
"""LocalFeatureStore: atomic put, get, idempotent release, abort, file mode (CPU)."""

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
