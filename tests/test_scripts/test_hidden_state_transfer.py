import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from scripts.prepare_hidden_states import HiddenStatesGenerator, parse_args


class HiddenStateTransferTest(unittest.TestCase):
    def generator(self, mode="async", queue_size=2):
        with (
            mock.patch("scripts.prepare_hidden_states.get_tp_group"),
            mock.patch("scripts.prepare_hidden_states.dist.get_rank", return_value=0),
        ):
            result = HiddenStatesGenerator(
                None,
                hidden_state_transfer=mode,
                num_io_threads=2,
                io_queue_size=queue_size,
            )
        result.show_progress = False
        return result

    def test_cli_default_and_sync_override(self):
        argv = [
            "prepare_hidden_states.py",
            "--target-model-path",
            "m",
            "--data-path",
            "d",
        ]
        with mock.patch("sys.argv", argv):
            self.assertEqual(parse_args().hidden_state_transfer, "async")
        with mock.patch("sys.argv", argv + ["--hidden-state-transfer", "sync"]):
            self.assertEqual(parse_args().hidden_state_transfer, "sync")
        with self.assertRaises(ValueError):
            self.generator(queue_size=0)

    def test_cpu_sources_are_independent_and_none_is_supported(self):
        for mode in ("async", "sync"):
            source = torch.arange(24).reshape(4, 6).t()
            expected = source.clone().unsqueeze(0)
            (copied, absent), events = self.generator(mode)._stage_hidden_states(
                source, None
            )
            source.zero_()
            self.assertTrue(torch.equal(copied, expected))
            self.assertIsNone(absent)
            self.assertEqual(events, [])

    def test_writer_waits_before_inspecting_record(self):
        generator = self.generator()
        reached, release = threading.Event(), threading.Event()

        def wait():
            reached.set()
            if not release.wait(5):
                raise TimeoutError("test did not release transfer event")

        event = SimpleNamespace(synchronize=wait)
        with (
            mock.patch.object(generator, "_save_tensor_sync") as save,
            ThreadPoolExecutor(1) as pool,
        ):
            future = pool.submit(generator._save_after_transfer, {}, "unused", [event])
            try:
                self.assertTrue(reached.wait(5))
                save.assert_not_called()
            finally:
                release.set()
            future.result(timeout=5)
            save.assert_called_once_with({}, "unused")

    def test_failed_write_is_reported_and_executor_is_closed(self):
        generator = self.generator()
        with (
            mock.patch("scripts.prepare_hidden_states.is_tp_rank_0", return_value=True),
            mock.patch("scripts.prepare_hidden_states.dist.barrier") as barrier,
        ):
            with self.assertRaisesRegex(OSError, "disk full"):
                with generator:
                    future = generator.io_executor.submit(
                        mock.Mock(side_effect=OSError("disk full"))
                    )
                    generator.pending_futures.append(future)
                    with self.assertRaises(OSError):
                        future.result()
                    generator._wait_for_io_slot()
            self.assertIsNone(generator.io_executor)
            self.assertEqual(generator.pending_futures, [])
            barrier.assert_not_called()

    def test_queue_backpressure_waits_for_oldest_write(self):
        generator = self.generator(queue_size=1)
        started, release = threading.Event(), threading.Event()

        def write():
            started.set()
            if not release.wait(5):
                raise TimeoutError("test did not release writer")

        with ThreadPoolExecutor(2) as pool:
            generator.pending_futures = [pool.submit(write)]
            self.assertTrue(started.wait(5))
            wait = pool.submit(generator._wait_for_io_slot)
            try:
                with self.assertRaises(TimeoutError):
                    wait.result(timeout=0.05)
            finally:
                release.set()
            wait.result(timeout=5)
            self.assertEqual(generator.pending_futures, [])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_generate_matches_sync_and_async_feature_records(self):
        class Model:
            def capture(self, **batch):
                n = batch["input_ids"].shape[1]
                return SimpleNamespace(
                    hidden_states=torch.arange(
                        n * 8, device="cuda", dtype=torch.float32
                    ).reshape(1, n, 8),
                    last_hidden_states=torch.arange(
                        n * 2, device="cuda", dtype=torch.float32
                    ).reshape(1, n, 2),
                )

        outputs = []
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch("scripts.prepare_hidden_states.is_tp_rank_0", return_value=True),
            mock.patch("scripts.prepare_hidden_states.get_tp_group"),
            mock.patch("scripts.prepare_hidden_states.get_dp_group"),
            mock.patch("scripts.prepare_hidden_states.dist.get_rank", return_value=0),
            mock.patch(
                "scripts.prepare_hidden_states.dist.get_process_group_ranks",
                return_value=[0],
            ),
            mock.patch("scripts.prepare_hidden_states.dist.broadcast"),
            mock.patch("scripts.prepare_hidden_states.dist.barrier"),
        ):
            for mode in ("sync", "async"):
                generator = self.generator(mode)
                generator.model = Model()
                batches = [
                    {
                        "input_ids": torch.arange(31 + i).unsqueeze(0),
                        "loss_mask": torch.ones(1, 31 + i, dtype=torch.long),
                        "attention_mask": torch.ones(1, 31 + i, dtype=torch.long),
                    }
                    for i in range(6)
                ]
                output = Path(directory) / mode
                with (
                    mock.patch(
                        "scripts.prepare_hidden_states.torch.cuda.empty_cache"
                    ) as clear,
                    mock.patch("scripts.prepare_hidden_states.gc.collect") as collect,
                ):
                    with generator:
                        generator.generate(batches, str(output), samples_per_dp=6)
                    self.assertEqual(clear.called, mode == "sync")
                    self.assertEqual(collect.called, mode == "sync")
                outputs.append(
                    [
                        torch.load(
                            generator._get_file_path(str(output), i), weights_only=True
                        )
                        for i in range(6)
                    ]
                )
            for baseline, optimized in zip(*outputs):
                self.assertEqual(baseline.keys(), optimized.keys())
                for key in baseline:
                    self.assertTrue(torch.equal(baseline[key], optimized[key]), key)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_copies_survive_source_release_and_match_serialization(self):
        generator = self.generator()
        producer = torch.cuda.Stream()
        copy_stream = torch.cuda.Stream()
        generator._copy_streams[torch.device("cuda", torch.cuda.current_device())] = (
            copy_stream
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch("scripts.prepare_hidden_states.is_tp_rank_0", return_value=True),
            mock.patch("scripts.prepare_hidden_states.dist.barrier"),
        ):
            with generator:
                for i in range(16):
                    generator._wait_for_io_slot()
                    expected = (
                        (torch.arange((33 + i) * 32).reshape(33 + i, 32) % 97)
                        .to(torch.bfloat16)
                        .t()
                    )
                    # Keep D2H pending while the producer drops and reallocates
                    # the source; without record_stream this can reuse its storage.
                    with torch.cuda.stream(copy_stream):
                        torch.cuda._sleep(5_000_000)
                    with torch.cuda.stream(producer):
                        source = torch.empty(
                            (33 + i, 32), device="cuda", dtype=torch.bfloat16
                        )
                        source.copy_(expected.t())
                        source = source.t()
                        (copied,), events = generator._stage_hidden_states(source)
                        del source
                        torch.empty(
                            (33 + i, 32), device="cuda", dtype=torch.bfloat16
                        ).fill_(-1)
                    generator._save_tensor_async(
                        {"hidden_state": copied},
                        str(Path(directory) / f"{i}.pt"),
                        events,
                    )
                generator._wait_all_saves()
            for i in range(16):
                expected = (
                    (torch.arange((33 + i) * 32).reshape(33 + i, 32) % 97)
                    .to(torch.bfloat16)
                    .t()
                    .unsqueeze(0)
                )
                actual = torch.load(Path(directory) / f"{i}.pt", weights_only=True)
                self.assertTrue(torch.equal(actual["hidden_state"], expected))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_only_allocation_failures_receive_memory_hint(self):
        generator = self.generator()
        source = torch.ones((4, 8), device="cuda")
        for failure, hinted in [
            (MemoryError("cannot allocate memory"), True),
            (torch.cuda.OutOfMemoryError("out of memory"), True),
            (RuntimeError("invalid argument"), False),
        ]:
            with (
                self.subTest(failure=type(failure).__name__),
                mock.patch(
                    "scripts.prepare_hidden_states.torch.empty", side_effect=failure
                ),
            ):
                with self.assertRaises(type(failure)) as raised:
                    generator._stage_hidden_states(source)
                self.assertEqual(
                    "--hidden-state-transfer sync" in str(raised.exception), hinted
                )
        for stream in generator._copy_streams.values():
            stream.synchronize()


if __name__ == "__main__":
    unittest.main()
