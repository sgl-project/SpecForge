# coding=utf-8
"""FeatureDataLoader: queue + store -> TrainBatch, with injected transform/collate (CPU)."""

import os
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from functools import partial
from unittest import mock

import torch

from specforge.data.preprocessing import process_offline_eagle3_sample
from specforge.runtime.data_plane.feature_dataloader import FeatureDataLoader
from specforge.runtime.data_plane.feature_store import LocalFeatureStore
from specforge.runtime.data_plane.offline_reader import OfflineManifestReader
from specforge.runtime.data_plane.sample_ref_queue import SampleRefQueue
from specforge.runtime.data_plane.streaming_ref_channel import (
    StreamingRefChannel,
    StreamingRefQueue,
)

_OFFLINE_EAGLE3_TRANSFORM = partial(process_offline_eagle3_sample, max_len=2048)


def _simple_collate(features):
    keys = features[0].keys()
    return {k: torch.cat([f[k] for f in features], dim=0) for k in keys}


class _CountingStore(LocalFeatureStore):
    """LocalFeatureStore that counts get() calls (seek must not materialize)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gets = 0

    def get(self, sample_ref, *, device="cpu", names=None):
        self.gets += 1
        return super().get(sample_ref, device=device, names=names)


class _AdoptingLocalStore(LocalFeatureStore):
    """Local cleanup behavior plus Mooncake-style adopt observability."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cleanup_events = []

    def adopt(self, ref):
        self.cleanup_events.append(("adopt", ref.sample_id))

    def abort(self, sample_id, *, reason="aborted"):
        self.cleanup_events.append(("abort", sample_id))
        super().abort(sample_id, reason=reason)


class _DurableQueue(SampleRefQueue):
    """Queue fake whose close cleanup must never advance its ack counter."""

    loader_close_retryable = True

    def __init__(self):
        super().__init__()
        self.consumed = 0
        self.failures = []

    def get_interruptible(self, n, *, stop_event):
        while not stop_event.is_set():
            refs = super().get(n, timeout_s=0.01)
            if refs:
                return refs
        return []

    def ack_durable(self, refs):
        super().ack(refs)
        self.consumed += len(refs)

    def fail(self, refs, reason, retryable):
        self.failures.append(([ref.sample_id for ref in refs], reason, retryable))
        super().fail(refs, reason=reason, retryable=retryable)


class _CloneFailure:
    def numel(self):
        return 1

    def element_size(self):
        return 4

    def clone(self):
        raise RuntimeError("clone failed")


class _LeaseTrackingStore:
    def __init__(self):
        self.active_leases = 0
        self.releases = 0

    def get(self, _ref, *, device="cpu"):
        del device
        self.active_leases += 1
        return {"x": _CloneFailure()}, object()

    def release(self, _handle, *, reason):
        del reason
        self.active_leases -= 1
        self.releases += 1


class _BlockingStore:
    """Wrap a real store while holding get() until the test releases it."""

    def __init__(self, inner):
        self.inner = inner
        self.entered = threading.Event()
        self.resume = threading.Event()

    def get(self, ref, *, device="cpu"):
        self.entered.set()
        self.resume.wait()
        return self.inner.get(ref, device=device)

    def release(self, handle, *, reason):
        self.inner.release(handle, reason=reason)

    def gc(self):
        return self.inner.gc()

    def health(self):
        return self.inner.health()


class TestFeatureDataLoader(unittest.TestCase):
    def _write_offline_files(self, d, n=4, seq=8, h=4, aux=12):
        for i in range(n):
            torch.save(
                {
                    "input_ids": torch.arange(seq) + i,
                    "loss_mask": torch.ones(seq, dtype=torch.long),
                    "hidden_state": torch.randn(1, seq, h),
                    "aux_hidden_state": torch.randn(1, seq, aux),
                },
                os.path.join(d, f"{i:03d}.ckpt"),
            )

    def test_offline_loader_emits_trainbatch(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_offline_files(d, n=4)
            # data-plane unit test: drive the loader from the queue primitive
            # directly (no control plane needed here).
            q = SampleRefQueue()
            q.put(OfflineManifestReader(d, run_id="run").read())
            store = LocalFeatureStore("st")
            loader = FeatureDataLoader(
                store,
                q,
                batch_size=2,
                collate_fn=_simple_collate,
                per_sample_transform=_OFFLINE_EAGLE3_TRANSFORM,
            )
            batches = list(loader)
            self.assertEqual(len(batches), 2)  # 4 samples / batch 2
            b = batches[0]
            self.assertEqual(len(b.sample_ids), 2)
            self.assertEqual(b.tensors["input_ids"].shape, (2, 8))
            self.assertEqual(b.tensors["target"].shape, (2, 8, 4))
            self.assertEqual(b.tensors["hidden_state"].shape, (2, 8, 12))
            # aux<->target swap preserved
            self.assertEqual(b.metadata["target_repr"], "hidden_state")
            # all refs acked
            self.assertEqual(q.in_flight(), 0)
            self.assertEqual(q.depth(), 0)

    def test_drop_last(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_offline_files(d, n=3)
            refs = OfflineManifestReader(d, run_id="run").read()
            store = LocalFeatureStore("st")
            loader = FeatureDataLoader(
                store,
                refs=refs,
                batch_size=2,
                collate_fn=_simple_collate,
                per_sample_transform=_OFFLINE_EAGLE3_TRANSFORM,
                drop_last=True,
            )
            batches = list(loader)
            self.assertEqual(len(batches), 1)  # 3 samples, drop the trailing 1

    def test_queue_partial_batch_drops_terminally_without_failing_run(self):
        with tempfile.TemporaryDirectory() as d:
            store = _AdoptingLocalStore("st")
            refs = [
                store.put(
                    {"x": torch.tensor([[float(index)]])},
                    sample_id=f"s{index}",
                    metadata={"run_id": "run", "target_repr": "hidden_state"},
                )
                for index in range(3)
            ]
            producer = StreamingRefChannel(os.path.join(d, "refs.jsonl"))
            producer.publish_many(refs)
            producer.close()
            q = StreamingRefQueue(StreamingRefChannel(producer.path))
            loader = FeatureDataLoader(
                store,
                q,
                batch_size=2,
                drop_last=True,
            )
            batches = list(loader)
            self.assertEqual(len(batches), 1)
            self.assertEqual(producer.consumed_remote(), 3)
            self.assertEqual(store.health()["resident_samples"], 0)
            self.assertEqual(
                store.cleanup_events,
                [("adopt", "s2"), ("abort", "s2")],
            )

    def test_prefetch_queue_partial_batch_drops_and_closes_cleanly(self):
        with tempfile.TemporaryDirectory() as d:
            store = LocalFeatureStore("st")
            refs = [
                store.put(
                    {"x": torch.tensor([[float(index)]])},
                    sample_id=f"s{index}",
                    metadata={"run_id": "run", "target_repr": "hidden_state"},
                )
                for index in range(3)
            ]
            q = _DurableQueue()
            q.put(refs)
            loader = FeatureDataLoader(
                store,
                q,
                batch_size=2,
                drop_last=True,
                num_workers=1,
            )

            batches = list(loader)
            loader.close()

        self.assertEqual(len(batches), 1)
        self.assertEqual(q.depth(), 0)
        self.assertEqual(q.in_flight(), 0)
        self.assertEqual(len(q.failures), 1)
        self.assertFalse(q.failures[0][2])
        self.assertEqual(store.health()["resident_samples"], 0)
        self.assertIsNone(loader._prefetch_state)

    def test_queue_partial_batch_cleanup_failure_stays_loud(self):
        store = _AdoptingLocalStore("st")
        ref = store.put(
            {"x": torch.tensor([[1.0]])},
            sample_id="s0",
            metadata={"run_id": "run", "target_repr": "hidden_state"},
        )
        q = SampleRefQueue()
        q.put([ref])
        loader = FeatureDataLoader(store, q, batch_size=2, drop_last=True)

        with mock.patch.object(
            store,
            "abort",
            side_effect=RuntimeError("injected abort failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "cleanup errors"):
                list(loader)

        self.assertEqual(q.depth(), 0)
        self.assertEqual(q.in_flight(), 0)
        self.assertEqual(store.cleanup_events, [("adopt", "s0")])

    def test_mixed_target_repr_fails_and_releases_refs(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_offline_files(d, n=2)
            refs = OfflineManifestReader(d, run_id="run").read()
            refs[1] = replace(
                refs[1],
                metadata={**refs[1].metadata, "target_repr": "logits"},
            )
            q = SampleRefQueue()
            q.put(refs)
            loader = FeatureDataLoader(
                LocalFeatureStore("st"),
                q,
                batch_size=2,
                collate_fn=_simple_collate,
                per_sample_transform=_OFFLINE_EAGLE3_TRANSFORM,
            )
            with self.assertRaises(ValueError):
                list(loader)
            self.assertEqual(q.in_flight(), 0)
            self.assertEqual(q.depth(), 0)

    def test_refs_mode_is_reiterable_across_epochs(self):
        # offline: a fixed ref set must re-iterate every epoch (no epoch-drain).
        with tempfile.TemporaryDirectory() as d:
            self._write_offline_files(d, n=4)
            refs = OfflineManifestReader(d, run_id="run").read()
            loader = FeatureDataLoader(
                LocalFeatureStore("st"),
                refs=refs,
                batch_size=2,
                collate_fn=_simple_collate,
                per_sample_transform=_OFFLINE_EAGLE3_TRANSFORM,
            )
            epoch1 = [b.sample_ids for b in loader]
            epoch2 = [b.sample_ids for b in loader]
            self.assertEqual(len(epoch1), 2)
            self.assertEqual(epoch1, epoch2)  # same fixed set each epoch

    def test_seek_repositions_next_pass_only(self):
        # resume: seek(k) skips the first k batches of the NEXT pass without
        # materializing them, then later epochs iterate in full again.
        with tempfile.TemporaryDirectory() as d:
            self._write_offline_files(d, n=6)
            refs = OfflineManifestReader(d, run_id="run").read()
            loader = FeatureDataLoader(
                LocalFeatureStore("st"),
                refs=refs,
                batch_size=2,
                collate_fn=_simple_collate,
                per_sample_transform=_OFFLINE_EAGLE3_TRANSFORM,
            )
            full = [b.sample_ids for b in loader]
            self.assertEqual(len(full), 3)
            loader.seek(2)
            self.assertEqual([b.sample_ids for b in loader], full[2:])
            # one-shot: consumed by that pass, the next epoch is full again
            self.assertEqual([b.sample_ids for b in loader], full)
            # a queue stream has no position to restore
            q = SampleRefQueue()
            qloader = FeatureDataLoader(
                LocalFeatureStore("st2"), q, collate_fn=_simple_collate
            )
            with self.assertRaises(ValueError):
                qloader.seek(1)

    def test_seek_past_end_raises(self):
        # a resume position beyond the dataset must fail loudly, not yield a
        # silent empty epoch; skip == available (fully consumed epoch) is fine.
        with tempfile.TemporaryDirectory() as d:
            self._write_offline_files(d, n=6)
            refs = OfflineManifestReader(d, run_id="run").read()
            loader = FeatureDataLoader(
                LocalFeatureStore("st"),
                refs=refs,
                batch_size=2,
                collate_fn=_simple_collate,
                per_sample_transform=_OFFLINE_EAGLE3_TRANSFORM,
            )
            with self.assertRaisesRegex(ValueError, "skips past the end of the data"):
                loader.seek(4)  # 6 refs / batch 2 = 3 batches
            loader.seek(3)  # boundary: allowed, next pass yields nothing
            self.assertEqual(list(loader), [])
            self.assertEqual(len(list(loader)), 3)  # one-shot, epoch after is full

    def test_seek_bound_counts_partial_batch_without_drop_last(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_offline_files(d, n=5)
            refs = OfflineManifestReader(d, run_id="run").read()
            loader = FeatureDataLoader(
                LocalFeatureStore("st"),
                refs=refs,
                batch_size=2,
                collate_fn=_simple_collate,
                per_sample_transform=_OFFLINE_EAGLE3_TRANSFORM,
                drop_last=False,
            )
            loader.seek(3)  # ceil(5/2) = 3: the trailing partial batch counts
            with self.assertRaisesRegex(ValueError, "skips past the end of the data"):
                loader.seek(4)

    def test_seek_does_not_materialize_skipped_refs(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_offline_files(d, n=6)
            refs = OfflineManifestReader(d, run_id="run").read()
            store = _CountingStore("st")
            transform_calls = []

            def counting_transform(raw):
                transform_calls.append(1)
                return _OFFLINE_EAGLE3_TRANSFORM(raw)

            loader = FeatureDataLoader(
                store,
                refs=refs,
                batch_size=2,
                collate_fn=_simple_collate,
                per_sample_transform=counting_transform,
            )
            loader.seek(2)
            batches = list(loader)
        self.assertEqual(len(batches), 1)
        # only the yielded batch's 2 refs are fetched/transformed
        self.assertEqual(store.gets, 2)
        self.assertEqual(len(transform_calls), 2)

    def test_offline_transform_does_not_mutate_raw_loss_mask(self):
        raw = {
            "input_ids": torch.arange(4),
            "loss_mask": torch.ones(4, dtype=torch.long),
            "hidden_state": torch.randn(1, 4, 2),
            "aux_hidden_state": torch.randn(1, 4, 6),
        }
        original_loss_mask = raw["loss_mask"].clone()

        transformed = process_offline_eagle3_sample(raw, max_len=3)

        self.assertTrue(torch.equal(raw["loss_mask"], original_loss_mask))
        self.assertEqual(transformed["input_ids"].shape, (1, 3))
        self.assertEqual(transformed["hidden_state"].shape, (1, 3, 6))
        self.assertEqual(transformed["target"].shape, (1, 3, 2))
        self.assertEqual(transformed["loss_mask"].tolist(), [[1, 1, 0]])

    def test_offline_workers_prefetch_without_reordering_batches(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_offline_files(d, n=8)
            refs = OfflineManifestReader(d, run_id="run").read()
            worker_names = []

            def record_worker(raw):
                worker_names.append(threading.current_thread().name)
                return _OFFLINE_EAGLE3_TRANSFORM(raw)

            loader = FeatureDataLoader(
                LocalFeatureStore("st"),
                refs=refs,
                batch_size=2,
                collate_fn=_simple_collate,
                per_sample_transform=record_worker,
                num_workers=2,
            )
            batches = [batch.sample_ids for batch in loader]

        self.assertEqual(
            batches,
            [
                [ref.sample_id for ref in refs[index : index + 2]]
                for index in range(0, 8, 2)
            ],
        )
        self.assertEqual(len(worker_names), 8)
        self.assertTrue(all(name.startswith("feature-loader") for name in worker_names))

    def test_materialize_releases_handle_when_clone_raises(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_offline_files(d, n=1)
            ref = OfflineManifestReader(d, run_id="run").read()[0]
            store = _LeaseTrackingStore()
            loader = FeatureDataLoader(store, refs=[ref])

            with self.assertRaisesRegex(RuntimeError, "clone failed"):
                loader._materialize(ref)

        self.assertEqual(store.active_leases, 0)
        self.assertEqual(store.releases, 1)

    def test_prefetch_materialization_exception_leaves_no_lease_or_thread(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_offline_files(d, n=1)
            refs = OfflineManifestReader(d, run_id="run").read()
            queue = SampleRefQueue()
            queue.put(refs)
            store = LocalFeatureStore("st")

            def fail_transform(_raw):
                raise RuntimeError("transform failed")

            loader = FeatureDataLoader(
                store,
                queue,
                batch_size=1,
                per_sample_transform=fail_transform,
                num_workers=1,
            )
            with self.assertRaisesRegex(RuntimeError, "transform failed"):
                list(loader)
            loader.close()
            loader.close()

        self.assertEqual(queue.in_flight(), 0)
        self.assertEqual(store.health()["active_leases"], 0)
        self.assertIsNone(loader._prefetch_state)

    def test_prefetch_close_requeues_only_never_yielded_durable_refs(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_offline_files(d, n=4)
            refs = OfflineManifestReader(d, run_id="run").read()
            by_id = {ref.sample_id: ref for ref in refs}
            queue = _DurableQueue()
            queue.put(refs)
            store = LocalFeatureStore("st")
            loader = FeatureDataLoader(
                store,
                queue,
                batch_size=1,
                collate_fn=_simple_collate,
                per_sample_transform=_OFFLINE_EAGLE3_TRANSFORM,
                ack=False,
                num_workers=2,
            )

            iterator = iter(loader)
            yielded = next(iterator)
            deadline = time.monotonic() + 2.0
            while queue.in_flight() < 3 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertGreaterEqual(queue.in_flight(), 3)
            queue.ack_durable([by_id[yielded.sample_ids[0]]])
            state = loader._prefetch_state
            loader.close()
            loader.close()  # idempotent
            iterator.close()

        self.assertEqual(queue.consumed, 1)
        self.assertEqual(queue.in_flight(), 0)
        self.assertEqual(queue.depth(), 3)
        self.assertTrue(queue.failures)
        self.assertTrue(all(retryable for _, _, retryable in queue.failures))
        failed_ids = {sid for ids, _, _ in queue.failures for sid in ids}
        self.assertNotIn(yielded.sample_ids[0], failed_ids)
        self.assertEqual(store.health()["active_leases"], 0)
        self.assertFalse(state.thread.is_alive())
        self.assertIsNone(loader._prefetch_state)

    def test_prefetch_close_is_bounded_if_store_get_stalls(self):
        import specforge.runtime.data_plane.feature_dataloader as loader_module

        with tempfile.TemporaryDirectory() as d:
            self._write_offline_files(d, n=1)
            refs = OfflineManifestReader(d, run_id="run").read()
            queue = _DurableQueue()
            queue.put(refs)
            store = _BlockingStore(LocalFeatureStore("st"))
            loader = FeatureDataLoader(
                store,
                queue,
                batch_size=1,
                collate_fn=_simple_collate,
                per_sample_transform=_OFFLINE_EAGLE3_TRANSFORM,
                ack=False,
                num_workers=1,
            )
            iterator = iter(loader)
            consumer_errors = []

            def consume_one():
                try:
                    next(iterator)
                except BaseException as exc:
                    consumer_errors.append(exc)

            consumer = threading.Thread(target=consume_one, daemon=True)
            consumer.start()
            self.assertTrue(store.entered.wait(timeout=1.0))
            state = loader._prefetch_state
            with mock.patch.object(loader_module, "_PREFETCH_JOIN_TIMEOUT_S", 0.05):
                started = time.monotonic()
                with self.assertRaisesRegex(RuntimeError, "did not stop"):
                    loader.close()
                self.assertLess(time.monotonic() - started, 0.5)
            self.assertEqual(state.outstanding_ids(), [refs[0].sample_id])

            store.resume.set()
            state.thread.join(timeout=1.0)
            loader.close()
            consumer.join(timeout=1.0)
            iterator.close()

        self.assertFalse(state.thread.is_alive())
        self.assertEqual(queue.in_flight(), 0)
        self.assertEqual(queue.depth(), 1)
        self.assertEqual(store.health()["active_leases"], 0)
        self.assertIsNone(loader._prefetch_state)

    def test_requires_exactly_one_source(self):
        store = LocalFeatureStore("st")
        with self.assertRaises(ValueError):
            FeatureDataLoader(store)  # neither queue nor refs
        with self.assertRaises(ValueError):
            FeatureDataLoader(store, SampleRefQueue(), refs=[])  # both


if __name__ == "__main__":
    unittest.main(verbosity=2)
