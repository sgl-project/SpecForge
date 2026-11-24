"""Canonical cached execution matches full-sequence loss and gradients under USP."""

import subprocess
import sys
import unittest
from pathlib import Path

import torch


class SequenceParallelTest(unittest.TestCase):
    def run_ranks(self, count):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={count}",
                "-m",
                "tests.test_algorithms.helpers.check_eagle_usp",
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            timeout=180,
        )

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
    def test_ulysses_and_ring(self):
        self.run_ranks(2)

    @unittest.skipUnless(torch.cuda.device_count() >= 4, "requires four CUDA devices")
    def test_mixed_ulysses_ring(self):
        self.run_ranks(4)


if __name__ == "__main__":
    unittest.main()
