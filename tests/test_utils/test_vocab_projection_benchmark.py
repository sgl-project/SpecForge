"""CPU-only contract tests for the standalone synthetic projection benchmark."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "vocab_projection_benchmark",
    ROOT / "benchmarks" / "bench_vocab_projection.py",
)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class VocabProjectionBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        self.rng = torch.random.fork_rng(devices=[])
        self.rng.__enter__()

    def tearDown(self):
        self.rng.__exit__(None, None, None)
        torch.set_num_threads(self.threads)

    def args(self, *extra):
        return bench.make_parser().parse_args(
            [
                "--hidden-size",
                "8",
                "--vocab-size",
                "17",
                "--warmup",
                "1",
                "--iterations",
                "2",
                "--repeats",
                "1",
                *extra,
            ]
        )

    def test_output_gradient_dtype_layout_and_rng(self):
        for dtype in ("float64", "float32", "bfloat16"):
            for batch in (1, 2):
                for layout in ("strided", "contiguous"):
                    with self.subTest(dtype=dtype, batch=batch, layout=layout):
                        args = self.args(
                            "--dtype", dtype, "--batch", str(batch), "--layout", layout
                        )
                        case = bench.make_case(args, torch)
                        before = case.head.weight.detach().clone()
                        self.assertEqual(len(bench.numerical_checks(case)), 4)
                        self.assertTrue(torch.equal(before, case.head.weight))
                        self.assertIsNone(case.head.weight.grad)
        for batch in (1, 2):
            with self.subTest(autocast="bfloat16", batch=batch):
                args = self.args(
                    "--dtype",
                    "float32",
                    "--batch",
                    str(batch),
                    "--autocast",
                    "bfloat16",
                )
                self.assertEqual(
                    len(bench.numerical_checks(bench.make_case(args, torch))), 4
                )

    def test_head_hook_shape_and_copy_boundary(self):
        case = bench.make_case(self.args(), torch)
        seen = []
        hook = case.head.register_forward_pre_hook(
            lambda module, inputs: seen.append(
                (tuple(inputs[0].shape), inputs[0].is_contiguous())
            )
        )
        try:
            for variant in bench.VARIANTS:
                bench.invoke(case, variant, "teacher")
        finally:
            hook.remove()
        self.assertEqual(
            seen,
            [((2, 21, 8), False), ((2, 21, 8), True), ((42, 8), True)],
        )

    def test_cpu_abba_order_without_cuda_memory_queries(self):
        case = bench.make_case(self.args(), torch)
        with mock.patch.object(
            torch.cuda,
            "reset_peak_memory_stats",
            side_effect=AssertionError("CUDA queried"),
        ):
            report = bench.measure(case)
        self.assertEqual(len(report["blocks"]), 16)
        for start in range(0, 16, 4):
            rows = report["blocks"][start : start + 4]
            candidate = rows[0]["candidate"]
            self.assertEqual(
                [row["variant"] for row in rows],
                ["native", candidate, candidate, "native"],
            )
            self.assertTrue(all(row["wall_ms_per_iteration"] > 0 for row in rows))
        self.assertEqual(len(report["summary"]), 4)

    def test_cuda_events_use_tensor_device_stream_without_a_gpu(self):
        device = "cuda:1"
        stream = object()
        start, end = mock.Mock(), mock.Mock()
        start.elapsed_time.return_value = 10.0
        cuda = mock.Mock()
        cuda.Event.side_effect = [start, end]
        cuda.current_stream.return_value = stream
        cuda.memory_allocated.return_value = 100
        cuda.memory_reserved.return_value = 200
        cuda.max_memory_allocated.return_value = 300
        cuda.max_memory_reserved.return_value = 400
        case = SimpleNamespace(
            hidden=SimpleNamespace(is_cuda=True, device=device, grad=None),
            args=SimpleNamespace(iterations=2),
            torch=SimpleNamespace(cuda=cuda),
        )
        with mock.patch.object(bench, "invoke") as invoke:
            result = bench.timed_block(case, "contiguous", "draft")
        cuda.current_stream.assert_called_once_with(device)
        cuda.synchronize.assert_called_once_with(device)
        cuda.reset_peak_memory_stats.assert_called_once_with(device)
        cuda.memory_allocated.assert_called_once_with(device)
        cuda.memory_reserved.assert_called_once_with(device)
        cuda.max_memory_allocated.assert_called_once_with(device)
        cuda.max_memory_reserved.assert_called_once_with(device)
        start.record.assert_called_once_with(stream)
        end.record.assert_called_once_with(stream)
        end.synchronize.assert_called_once_with()
        self.assertEqual(invoke.call_count, 2)
        self.assertEqual(result["cuda_ms_per_iteration"], 5.0)

    def test_numeric_cli_limits_reject_nonfinite_or_invalid_values(self):
        for field in ("rtol", "atol", "max_grad_relative_l2"):
            for value in (float("nan"), float("inf"), -1.0):
                with self.subTest(field=field, value=value):
                    args = self.args()
                    setattr(args, field, value)
                    with self.assertRaises(ValueError):
                        bench.validate_args(args)
        for value in (float("nan"), float("inf"), -1.1, 1.1):
            with self.subTest(min_grad_cosine=value):
                args = self.args()
                args.min_grad_cosine = value
                with self.assertRaises(ValueError):
                    bench.validate_args(args)
        args = self.args("--dtype", "float64", "--autocast", "bfloat16")
        with self.assertRaisesRegex(ValueError, "autocast does not lower float64"):
            bench.validate_args(args)
        args = self.args()
        args.rtol = args.atol = args.max_grad_relative_l2 = 0.0
        args.min_grad_cosine = -1.0
        bench.validate_args(args)

    def test_compare_reports_differences_and_rejects_dtype_changes(self):
        with self.assertRaises(AssertionError):
            bench.compare(torch.ones(3), torch.ones(3, dtype=torch.float64), 1e-5, 1e-6)
        info = bench.compare(torch.ones(3), torch.zeros(3), 1e-5, 1e-6)
        self.assertEqual(info["elementwise_violations"], 3)
        self.assertEqual(info["relative_l2"], 1.0)


if __name__ == "__main__":
    unittest.main()
