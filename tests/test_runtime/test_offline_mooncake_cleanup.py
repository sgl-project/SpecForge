"""Fault-injection coverage for offline Mooncake attempt cleanup."""

from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.runtime.data_plane.disagg_ingest import ingest_offline_features
from specforge.training.disaggregated import _build_offline

ALGORITHM = builtin_algorithm_registry().resolve("eagle3")


class _RecordingStore:
    def __init__(self, *, drain_error: Exception | None = None):
        self.abort_calls = []
        self.drain_calls = 0
        self.drain_error = drain_error

    def abort(self, sample_id, *, reason):
        self.abort_calls.append((sample_id, reason))

    def drain_pending_removals(self, **_kwargs):
        self.drain_calls += 1
        if self.drain_error is not None:
            raise self.drain_error
        return {"removed": len(self.abort_calls), "release_pending": 0}


class OfflineMooncakeCleanupTest(unittest.TestCase):
    def _config(self):
        return SimpleNamespace(
            run_id="offline-cleanup",
            model=SimpleNamespace(input_modality="text"),
            training=SimpleNamespace(
                role="producer",
                strategy="eagle3",
                ttt_length=7,
            ),
            data=SimpleNamespace(
                hidden_states_path="/features",
                max_length=2048,
            ),
        )

    @contextmanager
    def _producer_run(
        self,
        directory,
        store,
        ingest,
        *,
        backend="mooncake",
        hold_side_effect=None,
    ):
        manifest = str(Path(directory) / "manifest.json")
        environment = {
            "DISAGG_BACKEND": backend,
            "DISAGG_MANIFEST": manifest,
        }
        with (
            patch.dict(os.environ, environment),
            patch(
                "specforge.training.disaggregated._offline_store",
                return_value=store,
            ),
            patch(
                "specforge.runtime.data_plane.disagg_ingest.ingest_offline_features",
                side_effect=ingest,
            ),
            patch(
                "specforge.runtime.data_plane.disagg_ingest.write_ref_manifest"
            ) as write_manifest,
            patch(
                "specforge.training.disaggregated._hold_mooncake_producer",
                side_effect=hold_side_effect,
            ),
        ):
            run = _build_offline(
                self._config(),
                algorithm=ALGORITHM,
                build_model_bundle=lambda *_args, **_kwargs: None,
                optimizer_factory=lambda *_args, **_kwargs: None,
                logger=lambda *_args, **_kwargs: None,
            )
            yield run, write_manifest, manifest

    @staticmethod
    def _successful_ingest(refs):
        def ingest(_store, _path, *, on_ref, **_kwargs):
            for ref in refs:
                on_ref(ref)
            return list(refs)

        return ingest

    def test_consumer_success_aborts_all_refs_and_drains(self):
        refs = [SimpleNamespace(sample_id=f"sample-{index}") for index in range(3)]
        store = _RecordingStore()

        with (
            tempfile.TemporaryDirectory() as tmp,
            self._producer_run(tmp, store, self._successful_ingest(refs)) as (
                run,
                write_manifest,
                manifest,
            ),
        ):
            self.assertEqual(3, run.run())
            write_manifest.assert_called_once_with(refs, manifest)
            self.assertFalse(Path(manifest + ".failed").exists())

        self.assertEqual(
            [(ref.sample_id, "offline-attempt-finished") for ref in refs],
            store.abort_calls,
        )
        self.assertEqual(1, store.drain_calls)

    def test_consumer_failure_and_producer_timeout_cleanup(self):
        scenarios = (
            RuntimeError("remote consumer failed"),
            TimeoutError("producer hold timed out"),
        )
        for remote_error in scenarios:
            with self.subTest(error=type(remote_error).__name__):
                refs = [SimpleNamespace(sample_id="sample-0")]
                store = _RecordingStore()
                with (
                    tempfile.TemporaryDirectory() as tmp,
                    self._producer_run(
                        tmp,
                        store,
                        self._successful_ingest(refs),
                        hold_side_effect=remote_error,
                    ) as (run, _write_manifest, manifest),
                ):
                    with self.assertRaisesRegex(type(remote_error), str(remote_error)):
                        run.run()
                    self.assertTrue(Path(manifest + ".failed").exists())

                self.assertEqual(
                    [("sample-0", "offline-attempt-failed")],
                    store.abort_calls,
                )
                self.assertEqual(1, store.drain_calls)

    def test_failure_sentinel_is_published_before_the_cleanup_sweep(self):
        # The abort sweep can take minutes over thousands of refs and the
        # process may be SIGKILLed at supervisor-grace expiry mid-sweep. The
        # remote consumer's .failed wait is unbounded by default, so the
        # sentinel must be durable BEFORE the first abort, not after the last.
        class _SentinelOrderingStore(_RecordingStore):
            def __init__(self, manifest_holder):
                super().__init__()
                self.manifest_holder = manifest_holder
                self.sentinel_seen_before_abort = []

            def abort(self, sample_id, *, reason):
                self.sentinel_seen_before_abort.append(
                    Path(self.manifest_holder["manifest"] + ".failed").exists()
                )
                super().abort(sample_id, reason=reason)

        holder: dict = {}
        refs = [SimpleNamespace(sample_id=f"sample-{index}") for index in range(2)]
        store = _SentinelOrderingStore(holder)
        with (
            tempfile.TemporaryDirectory() as tmp,
            self._producer_run(
                tmp,
                store,
                self._successful_ingest(refs),
                hold_side_effect=TimeoutError("producer hold timed out"),
            ) as (run, _write_manifest, manifest),
        ):
            holder["manifest"] = manifest
            with self.assertRaisesRegex(TimeoutError, "hold timed out"):
                run.run()

        self.assertEqual(len(store.abort_calls), 2)
        self.assertEqual(store.sentinel_seen_before_abort, [True, True])

    def test_partial_ingestion_failure_cleans_every_tracked_ref(self):
        refs = [SimpleNamespace(sample_id=f"sample-{index}") for index in range(2)]
        store = _RecordingStore()

        def partial_ingest(_store, _path, *, on_ref, **_kwargs):
            for ref in refs:
                on_ref(ref)
            raise ValueError("corrupt third feature")

        with (
            tempfile.TemporaryDirectory() as tmp,
            self._producer_run(tmp, store, partial_ingest) as (
                run,
                write_manifest,
                _manifest,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "corrupt third feature"):
                run.run()
            write_manifest.assert_not_called()

        self.assertEqual(
            [(ref.sample_id, "offline-attempt-failed") for ref in refs],
            store.abort_calls,
        )
        self.assertEqual(1, store.drain_calls)

    def test_drain_failure_is_loud_after_all_aborts(self):
        refs = [SimpleNamespace(sample_id=f"sample-{index}") for index in range(2)]
        store = _RecordingStore(drain_error=RuntimeError("remove RPC stuck"))

        with (
            tempfile.TemporaryDirectory() as tmp,
            self._producer_run(tmp, store, self._successful_ingest(refs)) as (
                run,
                _write_manifest,
                manifest,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "offline Mooncake cleanup.*remove RPC stuck"
            ):
                run.run()
            self.assertIn("remove RPC stuck", Path(manifest + ".failed").read_text())

        self.assertEqual(
            [(ref.sample_id, "offline-attempt-finished") for ref in refs],
            store.abort_calls,
        )
        self.assertEqual(1, store.drain_calls)

    def test_shared_directory_attempt_keeps_existing_lifetime(self):
        refs = [SimpleNamespace(sample_id="sample-0")]
        store = _RecordingStore()

        with (
            tempfile.TemporaryDirectory() as tmp,
            self._producer_run(
                tmp,
                store,
                self._successful_ingest(refs),
                backend="shared_dir",
            ) as (run, _write_manifest, _manifest),
        ):
            self.assertEqual(1, run.run())

        self.assertEqual([], store.abort_calls)
        self.assertEqual(0, store.drain_calls)


class OfflineIngestTrackingTest(unittest.TestCase):
    def test_successful_refs_are_reported_before_a_later_put_fails(self):
        reader = SimpleNamespace(
            feature_keys=("input_ids",),
            target_repr="hidden_state",
        )
        tracked = []

        class FailingStore:
            def put(self, _tensors, *, sample_id, metadata):
                del metadata
                if sample_id.endswith("1"):
                    raise RuntimeError("second put failed")
                return SimpleNamespace(sample_id=sample_id)

        raw = {"input_ids": SimpleNamespace(numel=lambda: 4)}
        with (
            patch(
                "specforge.runtime.data_plane.disagg_ingest.list_feature_files",
                return_value=["feature-0", "feature-1"],
            ),
            patch(
                "specforge.runtime.data_plane.disagg_ingest.load_feature_file",
                return_value=raw,
            ),
            self.assertRaisesRegex(RuntimeError, "second put failed"),
        ):
            ingest_offline_features(
                FailingStore(),
                "/features",
                algorithm_name="eagle3",
                build_reader=lambda *_args, **_kwargs: reader,
                run_id="partial",
                on_ref=tracked.append,
            )

        self.assertEqual(["partial:00000000"], [ref.sample_id for ref in tracked])


if __name__ == "__main__":
    unittest.main()
