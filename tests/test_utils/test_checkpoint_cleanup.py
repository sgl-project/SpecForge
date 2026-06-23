"""Tests for checkpoint cleanup utility."""

import os
import tempfile
import unittest

from specforge.utils import cleanup_checkpoints


def create_checkpoint_dir(output_dir, epoch, step):
    """Create a mock checkpoint directory."""
    dirname = f"epoch_{epoch}_step_{step}"
    path = os.path.join(output_dir, dirname)
    os.makedirs(path, exist_ok=True)
    # Add a dummy file so the dir is non-empty
    with open(os.path.join(path, "config.json"), "w") as f:
        f.write("{}")
    return path


class TestCleanupCheckpoints(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_no_cleanup_when_limit_none(self):
        """No checkpoints removed when save_total_limit is None."""
        for i in range(5):
            create_checkpoint_dir(self.tmpdir, 0, (i + 1) * 1000)
        removed = cleanup_checkpoints(self.tmpdir, None)
        self.assertEqual(removed, 0)
        self.assertEqual(len(os.listdir(self.tmpdir)), 5)

    def test_no_cleanup_when_limit_zero(self):
        """No checkpoints removed when save_total_limit is 0."""
        for i in range(5):
            create_checkpoint_dir(self.tmpdir, 0, (i + 1) * 1000)
        removed = cleanup_checkpoints(self.tmpdir, 0)
        self.assertEqual(removed, 0)
        self.assertEqual(len(os.listdir(self.tmpdir)), 5)

    def test_no_cleanup_when_limit_negative(self):
        """No checkpoints removed when save_total_limit is negative."""
        for i in range(5):
            create_checkpoint_dir(self.tmpdir, 0, (i + 1) * 1000)
        removed = cleanup_checkpoints(self.tmpdir, -1)
        self.assertEqual(removed, 0)

    def test_removes_oldest_checkpoints(self):
        """Older checkpoints removed when limit exceeded."""
        for i in range(5):
            create_checkpoint_dir(self.tmpdir, 0, (i + 1) * 1000)
        removed = cleanup_checkpoints(self.tmpdir, 2)
        self.assertEqual(removed, 3)
        remaining = sorted(os.listdir(self.tmpdir))
        self.assertEqual(remaining, ["epoch_0_step_4000", "epoch_0_step_5000"])

    def test_no_removal_when_under_limit(self):
        """No checkpoints removed when count <= limit."""
        for i in range(3):
            create_checkpoint_dir(self.tmpdir, 0, (i + 1) * 1000)
        removed = cleanup_checkpoints(self.tmpdir, 5)
        self.assertEqual(removed, 0)
        self.assertEqual(len(os.listdir(self.tmpdir)), 3)

    def test_removes_oldest_across_epochs(self):
        """Correct removal ordering across multiple epochs."""
        steps = [
            (0, 1000),
            (0, 2000),
            (1, 3000),
            (1, 4000),
            (2, 5000),
        ]
        for epoch, step in steps:
            create_checkpoint_dir(self.tmpdir, epoch, step)
        removed = cleanup_checkpoints(self.tmpdir, 2)
        self.assertEqual(removed, 3)
        remaining = sorted(os.listdir(self.tmpdir))
        self.assertEqual(remaining, ["epoch_1_step_4000", "epoch_2_step_5000"])

    def test_nonexistent_dir(self):
        """No error when output_dir does not exist."""
        removed = cleanup_checkpoints("/nonexistent/path", 3)
        self.assertEqual(removed, 0)

    def test_empty_dir(self):
        """No error when output_dir is empty."""
        removed = cleanup_checkpoints(self.tmpdir, 3)
        self.assertEqual(removed, 0)

    def test_ignores_non_checkpoint_dirs(self):
        """Non-checkpoint directories are not affected."""
        create_checkpoint_dir(self.tmpdir, 0, 1000)
        create_checkpoint_dir(self.tmpdir, 0, 2000)
        create_checkpoint_dir(self.tmpdir, 0, 3000)
        # Create non-checkpoint dirs/files
        os.makedirs(os.path.join(self.tmpdir, "some_other_dir"), exist_ok=True)
        with open(os.path.join(self.tmpdir, "random_file.txt"), "w") as f:
            f.write("test")
        removed = cleanup_checkpoints(self.tmpdir, 1)
        self.assertEqual(removed, 2)
        remaining = sorted(os.listdir(self.tmpdir))
        self.assertIn("epoch_0_step_3000", remaining)
        self.assertIn("some_other_dir", remaining)
        self.assertIn("random_file.txt", remaining)


if __name__ == "__main__":
    unittest.main()
