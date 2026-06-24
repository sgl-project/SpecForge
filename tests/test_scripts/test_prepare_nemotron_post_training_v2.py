import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "prepare_nemotron_post_training_v2_under_test",
    ROOT / "scripts" / "prepare_nemotron_post_training_v2.py",
)
prepare_nemotron = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_nemotron)


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows
        self.last_split_kwargs = None

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)

    def shuffle(self, seed):
        return self

    def select(self, indices):
        return FakeDataset([self.rows[i] for i in indices])

    def train_test_split(self, test_size, seed, shuffle):
        self.last_split_kwargs = {
            "test_size": test_size,
            "seed": seed,
            "shuffle": shuffle,
        }
        test_count = int(round(len(self.rows) * test_size))
        train_count = len(self.rows) - test_count
        return {
            "train": FakeDataset(self.rows[:train_count]),
            "test": FakeDataset(self.rows[train_count:]),
        }


class TestNemotronPreparation(unittest.TestCase):
    def test_parse_args_defaults_to_core_splits(self):
        with patch.object(sys, "argv", ["prepare_nemotron_post_training_v2.py"]):
            args = prepare_nemotron.parse_args()

        self.assertEqual(args.splits, ["stem", "chat", "math", "code"])

    def test_validates_supported_splits_and_dedupes(self):
        self.assertEqual(
            prepare_nemotron.validate_splits(["stem", "code", "stem"]),
            ["stem", "code"],
        )

        with self.assertRaisesRegex(ValueError, "multilingual_ja"):
            prepare_nemotron.validate_splits(["stem", "multilingual_ja"])

    def test_converts_nemotron_messages_to_specforge_conversations(self):
        row = {
            "uuid": "row-1",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        }

        self.assertEqual(
            prepare_nemotron.row_to_specforge_conversation(row),
            {
                "id": "row-1",
                "conversations": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ],
            },
        )

    def test_partition_dataset_uses_eval_ratio(self):
        dataset = FakeDataset([{"id": str(i)} for i in range(10)])

        train_dataset, eval_dataset = prepare_nemotron.partition_dataset(
            dataset,
            eval_ratio=0.1,
            seed=123,
        )

        self.assertEqual(
            dataset.last_split_kwargs, {"test_size": 0.1, "seed": 123, "shuffle": True}
        )
        self.assertEqual(len(train_dataset), 9)
        self.assertEqual(len(eval_dataset), 1)

    def test_writes_jsonl(self):
        dataset = FakeDataset(
            [
                {
                    "uuid": "row-1",
                    "messages": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "a"},
                    ],
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "out.jsonl"
            count = prepare_nemotron.write_jsonl(dataset, output_path)

            self.assertEqual(count, 1)
            self.assertIn('"conversations"', output_path.read_text())


if __name__ == "__main__":
    unittest.main()
