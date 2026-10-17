"""The vLLM capture backend against a fake extract_hidden_states server.

The fake mimics vLLM's ExampleHiddenStatesConnector: each /v1/completions
request writes ``hidden_states [T, K, H]`` and ``token_ids [T]`` to a
safetensors file next to a ``.lock`` file and returns the path in
``kv_transfer_params.hidden_states_path``. Slot ``k`` of token ``t`` holds
``1000 * t + 10 * k + h`` so every value says where it came from.
"""

import fcntl
import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors.torch import save_file

from scripts.prepare_hidden_states import HiddenStatesGenerator, build_target_model
from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.algorithms.common.hidden_states_data import (
    normalize_dspark_offline_sample,
)
from specforge.offline_capture.sglang import OfflineSGLangCapture
from specforge.offline_capture.vllm_backend import (
    OfflineVLLMCaptureBackend,
    load_capture_file,
    parse_path_map,
)

WIDTH = 4
TAPS = [3, 15, 23, 35, 43]  # Flash-Next's DSpark taps, SpecForge numbering
LAYERS = 48


def expected_slots(token_ids, slots):
    t = torch.arange(len(token_ids), dtype=torch.float32)[:, None, None]
    k = torch.arange(slots, dtype=torch.float32)[None, :, None]
    h = torch.arange(WIDTH, dtype=torch.float32)[None, None, :]
    return 1000 * t + 10 * k + h


class FakeServer:
    """A threaded HTTP server; ``mode`` injects one kind of fault."""

    def __init__(self, storage: Path, slots: int = len(TAPS) + 1):
        self.storage = storage
        self.slots = slots
        self.mode = "ok"
        self.requests = []
        self._lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with server._lock:
                    server.requests.append(body)
                    number = len(server.requests)
                reply = server.reply(body, number)
                data = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def reply(self, body, number):
        ids = list(body["prompt"])
        hidden = expected_slots(ids, self.slots)
        saved_ids = torch.tensor(ids, dtype=torch.int64)
        if self.mode == "nan":
            hidden[0, 0, 0] = float("nan")
        if self.mode == "wrong_saved_ids":
            saved_ids = saved_ids + 1
        path = self.storage / f"req-{number}.safetensors"
        save_file({"hidden_states": hidden, "token_ids": saved_ids}, str(path))
        Path(str(path) + ".lock").touch()
        echoed = [i + 1 for i in ids] if self.mode == "wrong_echo" else ids
        reply = {"choices": [{"prompt_token_ids": echoed}]}
        if self.mode != "no_path":
            reply["kv_transfer_params"] = {"hidden_states_path": str(path)}
        return reply

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def target_dir(root: Path) -> str:
    path = root / "target"
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"text_config": {"num_hidden_layers": LAYERS}})
    )
    return str(path)


class VLLMBackendTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.storage = self.root / "captures"
        self.storage.mkdir()
        self.server = FakeServer(self.storage)
        self.backend = OfflineVLLMCaptureBackend.build(
            target_dir(self.root), url=self.server.url, served_model_name="m"
        )
        self.backend.set_capture_layers(TAPS, capture_method="dspark")

    def tearDown(self):
        self.server.close()
        self.tmp.cleanup()

    def test_server_layer_ids_are_vllm_numbering_plus_the_final_state(self):
        self.assertEqual(self.backend.server_layer_ids(), [4, 16, 24, 36, 44, 48])

    def test_capture_splits_taps_in_order_and_the_final_state(self):
        ids = [7, 8, 9, 10, 11]
        aux, last = self.backend.capture_ids(ids)
        slots = expected_slots(ids, len(TAPS) + 1)
        self.assertEqual(tuple(aux.shape), (5, len(TAPS) * WIDTH))
        for k in range(len(TAPS)):
            self.assertTrue(
                torch.equal(aux[:, k * WIDTH : (k + 1) * WIDTH], slots[:, k]), k
            )
        self.assertTrue(torch.equal(last, slots[:, len(TAPS)]))
        request = self.server.requests[-1]
        self.assertEqual(request["prompt"], ids)
        self.assertEqual(request["max_tokens"], 1)
        self.assertEqual(request["model"], "m")
        self.assertEqual(list(self.storage.iterdir()), [])

    def test_batch_rows_are_sent_unpadded_and_padded_back(self):
        input_ids = torch.tensor([[5, 6, 7, 8], [9, 10, 0, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
        loss_mask = torch.tensor([[0, 1, 1, 1], [0, 1, 0, 0]])
        data, aux_rows, last_rows = self.backend.capture(
            input_ids=input_ids, attention_mask=attention_mask, loss_mask=loss_mask
        )
        self.assertEqual(
            sorted(r["prompt"] for r in self.server.requests), [[5, 6, 7, 8], [9, 10]]
        )
        self.assertEqual(tuple(aux_rows[1].shape), (4, len(TAPS) * WIDTH))
        self.assertTrue(torch.equal(aux_rows[1][2:], torch.zeros(2, 20)))
        self.assertTrue(
            torch.equal(last_rows[1][:2], expected_slots([9, 10], 6)[:, len(TAPS)])
        )
        self.assertTrue(torch.equal(data[1][0], input_ids[1:2]))

    def test_left_padding_is_refused(self):
        with self.assertRaisesRegex(ValueError, "right-padded"):
            self.backend.capture(
                input_ids=torch.tensor([[0, 5, 6]]),
                attention_mask=torch.tensor([[0, 1, 1]]),
                loss_mask=torch.tensor([[0, 1, 1]]),
            )

    def test_faults_are_reported(self):
        cases = {
            "wrong_echo": "prompt_token_ids differ",
            "wrong_saved_ids": "token_ids differ",
            "no_path": "hidden_states_path",
            "nan": "NaN",
        }
        for mode, message in cases.items():
            with self.subTest(mode=mode):
                self.server.mode = mode
                with self.assertRaisesRegex(RuntimeError, message):
                    self.backend.capture_ids([1, 2, 3])

    def test_a_server_with_other_layers_is_refused_with_the_right_launch_value(self):
        self.server.slots = 4
        with self.assertRaisesRegex(RuntimeError, r"\[4, 16, 24, 36, 44, 48\]"):
            self.backend.capture_ids([1, 2, 3])

    def test_capture_layers_must_ascend_and_have_a_next_layer(self):
        with self.assertRaisesRegex(ValueError, "ascending"):
            self.backend.set_capture_layers([15, 3], capture_method="dspark")
        with self.assertRaisesRegex(ValueError, "no following layer"):
            self.backend.set_capture_layers([3, 47], capture_method="dspark")
        with self.assertRaisesRegex(ValueError, "DFlash-family"):
            self.backend.set_capture_layers(TAPS, capture_method="eagle3")

    def test_keep_captures_leaves_the_files(self):
        self.backend.keep_captures = True
        self.backend.capture_ids([1, 2])
        self.assertEqual(len(list(self.storage.glob("*.safetensors"))), 1)

    def test_path_map_rewrites_the_server_prefix(self):
        local = self.root / "mirror"
        os.symlink(self.storage, local)
        self.backend.path_map = parse_path_map(f"{self.storage}={local}")
        aux, _ = self.backend.capture_ids([1, 2])
        self.assertEqual(aux.shape[0], 2)
        with self.assertRaisesRegex(ValueError, "SERVER_PREFIX=LOCAL_PREFIX"):
            parse_path_map("no-equals-sign")


class LockTest(unittest.TestCase):
    def test_a_capture_without_its_lock_file_is_refused(self):
        # With use_synchronization_lock off the connector still writes
        # asynchronously, so a file without its lock may be incomplete.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "req.safetensors")
            save_file(
                {
                    "hidden_states": torch.zeros(2, 3, WIDTH),
                    "token_ids": torch.tensor([1, 2]),
                },
                path,
            )
            with self.assertRaisesRegex(RuntimeError, "use_synchronization_lock"):
                load_capture_file(path)

    def test_reading_waits_for_the_writer_to_release_its_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.safetensors")
            save_file({"hidden_states": torch.ones(1, 1, 1)}, path)
            done = threading.Event()
            with open(path + ".lock", "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                reader = threading.Thread(
                    target=lambda: (load_capture_file(path), done.set())
                )
                reader.start()
                time.sleep(0.2)
                self.assertFalse(done.is_set())
            reader.join(timeout=5)
            self.assertTrue(done.is_set())


class PrepareWithVLLMBackendTest(unittest.TestCase):
    def test_build_target_model_forwards_the_vllm_options(self):
        args = SimpleNamespace(
            target_model_path="target",
            trust_remote_code=False,
            target_backend="vllm",
            vllm_url="http://h:1",
            vllm_served_model_name="m",
            vllm_concurrency=4,
            vllm_timeout=60.0,
            vllm_keep_captures=False,
            vllm_path_map=None,
        )
        target = mock.Mock()
        with mock.patch(
            "scripts.prepare_hidden_states.load_offline_capture", return_value=target
        ) as load:
            build_target_model(
                args, SimpleNamespace(dtype=None), TAPS, capture_method="dspark"
            )
        kwargs = load.call_args.kwargs
        self.assertEqual(kwargs["backend"], "vllm")
        self.assertEqual(kwargs["url"], "http://h:1")
        self.assertEqual(kwargs["concurrency"], 4)
        target.set_capture_layers.assert_called_once_with(TAPS, capture_method="dspark")

    def test_build_target_model_accepts_a_raw_config_without_dtype(self):
        from specforge.modeling.target.target_utils import _RawConfigShim

        args = SimpleNamespace(
            target_model_path="target",
            trust_remote_code=False,
            target_backend="vllm",
            vllm_url="http://h:1",
            vllm_served_model_name="m",
            vllm_concurrency=4,
            vllm_timeout=60.0,
            vllm_keep_captures=False,
            vllm_path_map=None,
        )
        with mock.patch(
            "scripts.prepare_hidden_states.load_offline_capture",
            return_value=mock.Mock(),
        ) as load:
            build_target_model(
                args,
                _RawConfigShim({"text_config": {"num_hidden_layers": LAYERS}}),
                TAPS,
                capture_method="dspark",
            )
        self.assertIsNone(load.call_args.kwargs["torch_dtype"])

    def test_entry_reads_a_target_config_transformers_does_not_know(self):
        """The vllm client needs only config.json, not a Transformers release
        that registers the target's model type."""

        import scripts.prepare_hidden_states as prepare

        class Reached(Exception):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "target"
            target.mkdir()
            (target / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["NotRegisteredForConditionalGeneration"],
                        "model_type": "specforge_not_registered",
                        "text_config": {
                            "num_hidden_layers": LAYERS,
                            "hidden_size": 2560,
                            "vocab_size": 248320,
                        },
                    }
                )
            )
            argv = [
                "prepare_hidden_states.py",
                "--target-model-path",
                str(target),
                "--data-path",
                str(Path(tmp) / "data.jsonl"),
                "--output-path",
                str(Path(tmp) / "out"),
                "--strategy",
                "dspark",
                "--draft-model-config",
                "configs/qwen3.8-flash-next-dspark-moe64.json",
                "--target-backend",
                "vllm",
            ]
            with (
                mock.patch("sys.argv", argv),
                mock.patch.object(prepare, "init_distributed", side_effect=Reached),
            ):
                # Reaching distributed setup means the config was read and the
                # DSpark capture plan resolved from it.
                with self.assertRaises(Reached):
                    prepare.main()

    def test_generate_saves_trainable_records_at_their_real_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            storage = root / "captures"
            storage.mkdir()
            server = FakeServer(storage)
            try:
                backend = OfflineVLLMCaptureBackend.build(
                    target_dir(root), url=server.url, served_model_name="m"
                )
                model = OfflineSGLangCapture(backend)
                model.set_capture_layers(TAPS, capture_method="dspark")
                layout = (
                    builtin_algorithm_registry()
                    .resolve("dspark")
                    .providers.offline_for("text")
                    .capture_layout
                )
                generator = HiddenStatesGenerator.__new__(HiddenStatesGenerator)
                generator.model = model
                generator.capture_layout = layout
                generator.show_progress = False
                generator.file_group_size = 2000
                generator.compress = False
                generator.compression_level = 1
                generator.file_extension = ".ckpt"
                generator.num_io_threads = 2
                generator.pending_futures = []
                generator._save_tensor_async = generator._save_tensor_sync

                batch = {
                    "input_ids": torch.tensor([[5, 6, 7, 8, 9], [9, 10, 11, 0, 0]]),
                    "attention_mask": torch.tensor([[1] * 5, [1, 1, 1, 0, 0]]),
                    "loss_mask": torch.tensor([[0, 1, 1, 1, 1], [0, 1, 1, 0, 0]]),
                }
                fake_dist = mock.Mock()
                fake_dist.get_process_group_ranks.return_value = [0]
                fake_dist.get_rank.return_value = 0
                module = "scripts.prepare_hidden_states"
                out = root / "features"
                with (
                    mock.patch(f"{module}.dist", fake_dist),
                    mock.patch(f"{module}.get_tp_group"),
                    mock.patch(f"{module}.get_dp_group"),
                    mock.patch(f"{module}.is_tp_rank_0", return_value=True),
                    mock.patch(f"{module}.get_device_type", return_value="cpu"),
                ):
                    generator.generate([batch], str(out), start_idx=0, samples_per_dp=2)
            finally:
                server.close()

            files = sorted(out.rglob("*.ckpt"))
            self.assertEqual(len(files), 2)
            records = [torch.load(f, weights_only=True) for f in files]
            short = min(records, key=lambda r: r["input_ids"].numel())
            self.assertEqual(short["input_ids"].tolist(), [9, 10, 11])
            self.assertEqual(short["loss_mask"].tolist(), [0, 1, 1])
            self.assertEqual(tuple(short["hidden_states"].shape), (1, 3, 20))
            self.assertEqual(tuple(short["target_last_hidden_states"].shape), (1, 3, 4))
            # The saved tensors own only the kept rows: a view of the padded
            # batch row would serialize all of its storage.
            for name in ("hidden_states", "target_last_hidden_states"):
                tensor = short[name]
                self.assertEqual(
                    tensor.untyped_storage().nbytes(),
                    tensor.numel() * tensor.element_size(),
                    name,
                )
            slots = expected_slots([9, 10, 11], len(TAPS) + 1)
            self.assertTrue(
                torch.equal(
                    short["hidden_states"][0], slots[:, : len(TAPS)].reshape(3, 20)
                )
            )
            self.assertTrue(
                torch.equal(short["target_last_hidden_states"][0], slots[:, -1])
            )
            for record in records:
                normalized = normalize_dspark_offline_sample(record, max_len=16)
                self.assertEqual(normalized["hidden_states"].dim(), 3)
                self.assertTrue(
                    torch.equal(normalized["hidden_states"], record["hidden_states"])
                )


if __name__ == "__main__":
    unittest.main()
