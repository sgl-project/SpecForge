# coding=utf-8
"""OfflineManifestReader must not read feature files during assembly.

Filling ``SampleRef.feature_specs`` costs a full read of every file — tensor
pages for a ``.ckpt``, a whole-stream decompression for the ``.ckpt.gz`` that
``prepare_hidden_states --compress`` writes, since gzip has no random access.
Paid during assembly that is one serial pass over the dataset per rank before
the first step. So the reader lists filenames and nothing else, and the loader's
prefetch workers read tensors lazily, overlapped with compute.

Refs without specs stay usable: the store resolves tensors through
``feature_keys``, and ``FeatureDataLoader._validate_refs`` skips spec comparison
for them.
"""

import contextlib
import gzip
import os
import tempfile
import unittest
from unittest import mock

import torch

from specforge.runtime.data_plane.feature_dataloader import FeatureDataLoader
from specforge.runtime.data_plane.feature_store import LocalFeatureStore
from specforge.runtime.data_plane.offline_reader import OfflineManifestReader

_KEYS = ("input_ids", "loss_mask", "hidden_states")
_SEQ = 8


def _write_features(directory: str, *, compress: bool, n: int = 3):
    os.makedirs(directory, exist_ok=True)
    for index in range(n):
        payload = {
            "input_ids": torch.arange(_SEQ) + index,
            "loss_mask": torch.ones(_SEQ, dtype=torch.long),
            "hidden_states": torch.randn(_SEQ, 4).to(torch.bfloat16),
        }
        suffix = ".ckpt.gz" if compress else ".ckpt"
        path = os.path.join(directory, f"data_{index}{suffix}")
        if compress:
            with gzip.open(path, "wb") as stream:
                torch.save(payload, stream)
        else:
            torch.save(payload, path)
    return directory


def _features(**kwargs):
    directory = os.path.join(tempfile.mkdtemp(prefix="offline_lazy_"), "features")
    return _write_features(directory, **kwargs)


def _reader(path: str, **kwargs) -> OfflineManifestReader:
    return OfflineManifestReader(
        path,
        run_id="lazy",
        strategy="dflash",
        feature_keys=_KEYS,
        target_repr=None,
        max_len=_SEQ,
        **kwargs,
    )


@contextlib.contextmanager
def _no_file_reads():
    """Fail if anything opens a feature file through either read path."""
    boom = AssertionError("assembly must not read feature files")
    with (
        mock.patch("torch.load", side_effect=boom),
        mock.patch("gzip.open", side_effect=boom),
    ):
        yield


class TestOfflineReaderIsLazy(unittest.TestCase):
    def test_assembly_reads_no_uncompressed_file(self):
        directory = _features(compress=False)
        with _no_file_reads():
            refs = _reader(directory).read()

        self.assertEqual(len(refs), 3)
        for ref in refs:
            self.assertTrue(ref.feature_store_uri.startswith("file://"))
            self.assertEqual(set(ref.feature_keys), set(_KEYS))
            self.assertEqual(ref.feature_specs, {})
            self.assertEqual(ref.num_tokens, 0)

    def test_assembly_reads_no_compressed_file(self):
        directory = _features(compress=True)
        with _no_file_reads():
            refs = _reader(directory).read()

        self.assertEqual(len(refs), 3)
        for ref in refs:
            self.assertEqual(ref.feature_specs, {})


class TestSpecLessRefsRemainUsable(unittest.TestCase):
    def _assert_materializes(self, directory: str):
        refs = _reader(directory).read()
        store = LocalFeatureStore("lazy")

        for index, ref in enumerate(refs):
            tensors, handle = store.get(ref)
            try:
                self.assertEqual(set(tensors), set(_KEYS))
                self.assertTrue(
                    torch.equal(tensors["input_ids"], torch.arange(_SEQ) + index)
                )
                self.assertEqual(tensors["hidden_states"].shape, (_SEQ, 4))
            finally:
                store.release(handle, reason="test")

    def test_uncompressed_refs_materialize(self):
        self._assert_materializes(_features(compress=False))

    def test_compressed_refs_materialize(self):
        self._assert_materializes(_features(compress=True))

    def test_dataloader_accepts_spec_less_refs(self):
        refs = _reader(_features(compress=True, n=4)).read()
        loader = FeatureDataLoader(
            LocalFeatureStore("lazy"),
            refs=refs,
            batch_size=2,
            collate_fn=lambda samples: {
                "input_ids": torch.stack([s["input_ids"] for s in samples])
            },
            strategy="dflash",
        )

        batches = list(loader)
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0].tensors["input_ids"].shape, (2, _SEQ))

    def test_missing_key_surfaces_on_access(self):
        directory = os.path.join(tempfile.mkdtemp(prefix="offline_bad_"), "features")
        os.makedirs(directory)
        torch.save({"input_ids": torch.arange(_SEQ)}, os.path.join(directory, "b.ckpt"))
        ref = _reader(directory).read()[0]

        # Assembly accepted the incomplete file; the loss lands here instead.
        with self.assertRaises(KeyError):
            LocalFeatureStore("lazy").get(ref)


class TestEagerValidationIsOptIn(unittest.TestCase):
    def test_explicit_flag_fills_specs(self):
        refs = _reader(_features(compress=False), validate_files=True).read()

        for ref in refs:
            self.assertEqual(set(ref.feature_specs), set(_KEYS))
            self.assertEqual(ref.num_tokens, _SEQ)
            self.assertGreater(ref.estimated_bytes, 0)

    def test_env_var_fills_specs(self):
        directory = _features(compress=False)
        with mock.patch.dict(os.environ, {"SPECFORGE_VALIDATE_OFFLINE_FEATURES": "1"}):
            refs = _reader(directory).read()

        for ref in refs:
            self.assertEqual(set(ref.feature_specs), set(_KEYS))

    def test_env_var_default_stays_lazy(self):
        directory = _features(compress=False)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SPECFORGE_VALIDATE_OFFLINE_FEATURES", None)
            with _no_file_reads():
                refs = _reader(directory).read()

        self.assertEqual([ref.feature_specs for ref in refs], [{}] * 3)


if __name__ == "__main__":
    unittest.main()
