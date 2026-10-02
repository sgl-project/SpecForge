"""Length indexing reads each file revision once without eager tensor residency."""

import gzip
import os
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from specforge.data.offline_lengths import ensure_offline_lengths
from specforge.runtime.data_plane.offline_reader import OfflineManifestReader


def _write(path, length):
    raw = {"input_ids": torch.arange(length)}
    if str(path).endswith(".gz"):
        with gzip.open(path, "wb") as stream:
            torch.save(raw, stream)
    else:
        torch.save(raw, path)


@pytest.mark.parametrize("suffix", [".ckpt", ".ckpt.gz"])
def test_indexes_once_and_invalidates_changed_file(tmp_path, suffix):
    features = tmp_path / "features"
    features.mkdir()
    path = features / ("sample" + suffix)
    _write(path, 7)
    refs = OfflineManifestReader(str(features)).read()
    assert refs[0].num_tokens == 0
    cache = str(tmp_path / "cache")
    indexed = ensure_offline_lengths(refs, cache_dir=cache)
    assert indexed[0].num_tokens == 7
    assert indexed[0].feature_specs == {}  # tensor loading stays lazy
    assert refs[0].num_tokens == 0
    with patch(
        "specforge.runtime.data_plane.feature_store.load_feature_file",
        side_effect=AssertionError("warm cache must not reopen feature tensors"),
    ):
        assert ensure_offline_lengths(refs, cache_dir=cache) == indexed
    _write(path, 13)
    assert ensure_offline_lengths(refs, cache_dir=cache)[0].num_tokens == 13
    assert sorted(os.listdir(features)) == ["sample" + suffix]


@pytest.mark.parametrize("cached", ["{invalid", '{"version":1}', "[]"])
def test_corrupt_cache_is_rebuilt(tmp_path, cached):
    _write(tmp_path / "sample.ckpt", 9)
    refs = OfflineManifestReader(str(tmp_path)).read()
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "offline-token-lengths.json").write_text(cached)
    assert ensure_offline_lengths(refs, cache_dir=str(cache))[0].num_tokens == 9


def test_remote_refs_need_metadata_and_known_refs_are_not_loaded(tmp_path):
    _write(tmp_path / "sample.ckpt", 9)
    ref = OfflineManifestReader(str(tmp_path)).read()[0]
    remote = replace(ref, feature_store_uri="mooncake://store/sample")
    with pytest.raises(ValueError, match="token-length metadata"):
        ensure_offline_lengths([remote], cache_dir=str(tmp_path / "cache"))
    known = replace(remote, num_tokens=9)
    with patch(
        "specforge.runtime.data_plane.feature_store.load_feature_file",
        side_effect=AssertionError("metadata suffices"),
    ):
        assert ensure_offline_lengths([known], cache_dir=str(tmp_path / "cache")) == [
            known
        ]


def test_cache_invalidates_changed_input_ids_mapping(tmp_path):
    path = tmp_path / "sample.ckpt"
    torch.save({"short_ids": torch.arange(3), "long_ids": torch.arange(11)}, path)
    ref = OfflineManifestReader(str(tmp_path)).read()[0]
    cache = str(tmp_path / "cache")
    for key, length in [("short_ids", 3), ("long_ids", 11), ("short_ids", 3)]:
        mapped = replace(ref, feature_keys={"input_ids": key})
        assert ensure_offline_lengths([mapped], cache_dir=cache)[0].num_tokens == length


def test_empty_or_batched_features_are_rejected(tmp_path):
    path = tmp_path / "sample.ckpt"
    for shape in [(0,), (2, 7)]:
        torch.save({"input_ids": torch.zeros(shape, dtype=torch.long)}, path)
        refs = OfflineManifestReader(str(tmp_path)).read()
        with pytest.raises(ValueError):
            ensure_offline_lengths(refs, cache_dir=str(tmp_path / "cache"))
