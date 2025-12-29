import unittest
from pathlib import Path

from sglang.utils import execute_shell_command

CACHE_DIR = Path(__file__).parent.parent.parent.joinpath("cache")


class TestPrepareData(unittest.TestCase):

    def test_prepare_sharegpt(self):
        sharegpt_train_path = CACHE_DIR.joinpath("dataset", "sharegpt_train.jsonl")

        if sharegpt_train_path.exists():
            # delete the file
            sharegpt_train_path.unlink()
        process = execute_shell_command(
            "python scripts/prepare_data.py --dataset sharegpt"
        )
        process.wait()
        self.assertEqual(process.returncode, 0)
        self.assertTrue(sharegpt_train_path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
