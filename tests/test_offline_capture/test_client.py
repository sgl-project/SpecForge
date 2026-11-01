"""Real plugin sink -> transport -> offline files, without a GPU server."""

import gzip
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from scripts.prepare_hidden_states import HiddenStatesGenerator, sample_range
from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.offline_capture import OfflineCaptureClient
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
from tests.test_runtime.test_server_capture import _FakeMooncakeStore, _request_specs

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "plugins/sglang-spec-capture")
)
from specforge_sglang_capture.sink import SpecCaptureSink


class OfflineCaptureTest(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(
            "os.environ",
            {"MOONCAKE_PROTOCOL": "tcp", "SGLANG_SPEC_CAPTURE_GPU_PUT": "0"},
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.backend = _FakeMooncakeStore()
        self.sink = SpecCaptureSink(aux_layer_ids=[1, 3])
        self.sink._store = self.backend
        self.addCleanup(self.sink._executor.shutdown)
        self.store = MooncakeFeatureStore(
            store=self.backend,
            store_id="offline-test",
            receive_buffers="pageable",
            retain_on_release=False,
        )
        self.calls = []
        self.lost_response = False

    def client(self, strategy="eagle3", **overrides):
        self.layout = (
            builtin_algorithm_registry()
            .resolve(strategy)
            .providers.offline_for("text")
            .capture_layout
        )
        info = {
            "model_path": "target",
            "internal_states": [
                {
                    "spec_capture_backend": "plugin",
                    "spec_capture_method": self.layout.capture_method,
                    "spec_capture_aux_layer_ids": [1, 3],
                }
            ],
        }
        kwargs = dict(
            target_model_path="target",
            capture_layers=[1, 3],
            layout=self.layout,
            target_hidden_size=4,
            store=self.store,
            server_info_fn=lambda *_: info,
            post_fn=self.post,
        )
        kwargs.update(overrides)
        client = OfflineCaptureClient(["http://capture"], **kwargs)
        self.addCleanup(client.close)
        return client

    def post(self, url, json_body, timeout):
        body = json_body
        self.calls.append(body)
        samples = []
        for ids, spec in zip(body["input_ids"], _request_specs(body)):
            samples.append(
                (
                    spec,
                    torch.arange(len(ids) * 8, dtype=torch.float32).reshape(
                        len(ids), 8
                    ),
                    torch.ones(len(ids), 4),
                )
            )
        results = self.sink.put_samples(samples)
        if self.lost_response:
            raise TimeoutError("response lost after publication")
        return [{"meta_info": {"spec_capture": [result]}} for result in results]

    @staticmethod
    def rows():
        return [
            {
                "input_ids": [1, 2, 3, 0],
                "attention_mask": [1, 1, 1, 0],
                "loss_mask": [0, 1, 1, 0],
            },
            {"input_ids": [4, 5], "attention_mask": [1, 1], "loss_mask": [1, 1]},
        ]

    def test_algorithms_round_trip_and_resume(self):
        for strategy in ("eagle3", "dflash", "domino", "dspark"):
            for compress in (False, True):
                with (
                    self.subTest(strategy=strategy, compress=compress),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    client = self.client(strategy)
                    with HiddenStatesGenerator(
                        client, self.layout, compress=compress, io_queue_size=1
                    ) as generator:
                        generator.generate([self.rows()], directory, start_idx=7)
                        calls = len(self.calls)
                        generator.generate([self.rows()], directory, start_idx=7)
                        self.assertEqual(len(self.calls), calls)
                    self.assertEqual(self.calls[-1]["input_ids"], [[1, 2, 3], [4, 5]])
                    files = sorted(Path(directory).rglob("data_*"))
                    self.assertEqual(len(files), 2)
                    for path, length in zip(files, (3, 2)):
                        with (
                            gzip.open(path, "rb") if compress else path.open("rb")
                        ) as source:
                            record = torch.load(source, weights_only=True)
                        self.assertEqual(set(record), set(self.layout.output_names))
                        self.assertEqual(record["input_ids"].numel(), length)
                        self.assertEqual(
                            record[self.layout.aux_feature].shape, (1, length, 8)
                        )
                    self.assertFalse(self.backend._d)

    def test_features_live_until_writer_finishes(self):
        client = self.client()
        with (
            tempfile.TemporaryDirectory() as directory,
            HiddenStatesGenerator(client, self.layout) as generator,
        ):
            original = generator._save_tensor_sync

            def save(record, path):
                self.assertTrue(self.backend._d)
                original(record, path)

            with mock.patch.object(generator, "_save_tensor_sync", side_effect=save):
                generator.generate([self.rows()], directory)
        self.assertFalse(self.backend._d)

    def test_disk_failure_leaves_no_completed_file_and_releases_features(self):
        client = self.client()
        with tempfile.TemporaryDirectory() as directory:
            with (
                self.assertRaisesRegex(OSError, "disk full"),
                HiddenStatesGenerator(client, self.layout) as generator,
            ):
                with mock.patch("torch.save", side_effect=OSError("disk full")):
                    generator.generate([self.rows()], directory)
            self.assertFalse(list(Path(directory).rglob("*.ckpt")))
            self.assertFalse(list(Path(directory).rglob("*.tmp")))
            self.assertFalse(self.backend._d)

    def test_lost_response_cleans_published_keys_and_retry_is_isolated(self):
        client = self.client()
        self.lost_response = True
        with (
            self.assertRaises(TimeoutError),
            client.capture_rows([[1, 2]], sample_ids=[9]),
        ):
            pass
        self.assertFalse(self.backend._d)
        self.lost_response = False
        with client.capture_rows([[1, 2]], sample_ids=[9]) as rows:
            self.assertEqual(len(rows), 1)
        self.assertNotEqual(
            _request_specs(self.calls[0])[0]["sample_id"],
            _request_specs(self.calls[1])[0]["sample_id"],
        )
        self.assertFalse(self.backend._d)

    def test_read_failure_releases_already_adopted_batch(self):
        client = self.client()
        with mock.patch.object(self.backend, "get_into", return_value=-1):
            with (
                self.assertRaises(Exception),
                client.capture_rows([[1, 2], [3]], sample_ids=[0, 1]),
            ):
                pass
        self.assertFalse(self.backend._d)

    def test_bad_second_response_cleans_first_adoption_and_provisional_keys(self):
        def corrupt(url, json_body, timeout):
            rows = self.post(url, json_body, timeout)
            rows[1]["meta_info"]["spec_capture"][0]["aux_layer_ids"] = [7, 8]
            return rows

        client = self.client(post_fn=corrupt)
        with (
            self.assertRaises(Exception),
            client.capture_rows([[1, 2], [3]], sample_ids=[0, 1]),
        ):
            pass
        self.assertFalse(self.backend._d)

    def test_rejects_mismatched_model_method_or_layer_order_before_capture(self):
        for field, value in (
            ("model_path", "other"),
            ("spec_capture_method", "dflash"),
            ("spec_capture_aux_layer_ids", [3, 1]),
            ("spec_capture_method", None),
        ):
            info = {
                "model_path": "target",
                "internal_states": [
                    {
                        "spec_capture_backend": "plugin",
                        "spec_capture_method": "eagle3",
                        "spec_capture_aux_layer_ids": [1, 3],
                    }
                ],
            }
            (info if field == "model_path" else info["internal_states"][0])[
                field
            ] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.client(server_info_fn=lambda *_: info)
        self.assertFalse(self.calls)

    @unittest.skipUnless(
        os.environ.get("SPECFORGE_RUN_OFFLINE_MOONCAKE_TESTS") == "1",
        "requires a running Mooncake master and SPECFORGE_RUN_OFFLINE_MOONCAKE_TESTS=1",
    )
    def test_real_mooncake_cpu_round_trip(self):
        # The plugin sink uses a real server-owned storage segment. The client
        # opens its own connection with zero storage/receive-buffer reservations.
        self.sink._store = None
        with mock.patch.dict(
            "os.environ",
            {
                "MOONCAKE_GLOBAL_SEGMENT_SIZE": str(64 << 20),
                "MOONCAKE_LOCAL_BUFFER_SIZE": str(16 << 20),
            },
        ):
            client = self.client(store=None)
            with tempfile.TemporaryDirectory() as directory:
                with HiddenStatesGenerator(client, self.layout) as generator:
                    generator.generate([self.rows()], directory)
                files = sorted(Path(directory).rglob("*.ckpt"))
                self.assertEqual(len(files), 2)
                record = torch.load(files[0], weights_only=True)
                torch.testing.assert_close(
                    record["aux_hidden_state"],
                    torch.arange(24, dtype=torch.float32).reshape(1, 3, 8),
                )
            client.close()
            for spec in _request_specs(self.calls[0]):
                for name in spec["features"].values():
                    key = (
                        f"{spec['store_id']}/{spec['sample_id']}/g{spec['gen']}/{name}"
                    )
                    self.assertNotEqual(self.sink._store.is_exist(key), 1)

    def test_shards_cover_each_sample_once_including_empty_workers(self):
        for total in (1, 7, 16):
            for workers in (1, 3, 8):
                ids = [
                    i
                    for rank in range(workers)
                    for i in range(*sample_range(total, rank, workers))
                ]
                self.assertEqual(ids, list(range(total)))


if __name__ == "__main__":
    unittest.main()
