import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from specforge.data.prompt_builder import prepare_prompt_tasks


class _FakeDataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)

    def select(self, indices):
        return _FakeDataset(self.rows[index] for index in indices)


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")


class TestPreparePromptTasks(unittest.TestCase):
    def test_json_array_is_accepted(self):
        records = [
            {"input_ids": [1, 2], "loss_mask": [0, 1]},
            {"input_ids": [3, 4], "loss_mask": [1, 1]},
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "prompts.json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(records, stream, indent=2)
            prompts = prepare_prompt_tasks(
                path,
                tokenizer=None,
                chat_template=None,
                max_length=8,
                is_preformatted=False,
                train_only_last_turn=False,
                cache_dir=None,
                cache_key=None,
                num_proc=1,
            )

        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[1]["payload"]["input_ids"], [3, 4])

    def test_pre_tokenized_path_truncates_filters_and_caps(self):
        records = [
            {"input_ids": [[1, 2, 3, 4]], "loss_mask": [[0, 1, 1, 1]]},
            {"input_ids": [5, 6], "loss_mask": [0, 0]},
            {"input_ids": [7, 8], "loss_mask": [1, 1]},
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "prompts.jsonl")
            _write_jsonl(path, records)

            prompts = prepare_prompt_tasks(
                path,
                tokenizer=object(),
                chat_template=None,
                max_length=3,
                is_preformatted=False,
                train_only_last_turn=False,
                cache_dir=None,
                cache_key=None,
                num_proc=1,
                min_loss_tokens=2,
                max_prompts=2,
            )

        self.assertEqual(
            prompts,
            [
                {"payload": {"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1]}},
                {"payload": {"input_ids": [7, 8], "loss_mask": [1, 1]}},
            ],
        )

    def test_raw_conversations_use_lazy_dataset_preprocessing(self):
        raw_rows = [
            {"conversations": [{"role": "user", "content": "one"}]},
            {"conversations": [{"role": "user", "content": "two"}]},
        ]
        loaded_dataset = _FakeDataset(raw_rows)
        processed_dataset = _FakeDataset(
            [
                {
                    "input_ids": [[10, 11, 12]],
                    "loss_mask": [[0, 1, 1]],
                    "attention_mask": [[1, 1, 1]],
                }
            ]
        )
        load_calls = []
        build_calls = []

        def fake_load_dataset(*args, **kwargs):
            load_calls.append((args, kwargs))
            return loaded_dataset

        def fake_build_eagle3_dataset(**kwargs):
            build_calls.append(kwargs)
            return processed_dataset

        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = fake_load_dataset
        fake_preprocessing = types.ModuleType("specforge.data.preprocessing")
        fake_preprocessing.build_eagle3_dataset = fake_build_eagle3_dataset

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "raw.jsonl")
            _write_jsonl(path, raw_rows)
            with patch.dict(
                sys.modules,
                {
                    "datasets": fake_datasets,
                    "specforge.data.preprocessing": fake_preprocessing,
                },
            ):
                prompts = prepare_prompt_tasks(
                    path,
                    tokenizer="tokenizer",
                    chat_template="qwen",
                    max_length=128,
                    is_preformatted=False,
                    train_only_last_turn=True,
                    cache_dir=tmp_dir,
                    cache_key="cache-key",
                    num_proc=3,
                    min_loss_tokens=2,
                    max_prompts=1,
                )

        self.assertEqual(
            prompts,
            [
                {
                    "payload": {
                        "input_ids": [10, 11, 12],
                        "loss_mask": [0, 1, 1],
                    }
                }
            ],
        )
        self.assertEqual(
            load_calls,
            [(("json",), {"data_files": path, "split": "train"})],
        )
        self.assertEqual(len(build_calls), 1)
        self.assertEqual(len(build_calls[0]["dataset"]), 1)
        self.assertEqual(build_calls[0]["tokenizer"], "tokenizer")
        self.assertEqual(build_calls[0]["chat_template"], "qwen")
        self.assertEqual(build_calls[0]["minimum_valid_tokens"], 2)
        self.assertTrue(build_calls[0]["train_only_last_turn"])

    def test_rejects_incomplete_or_mismatched_token_records(self):
        cases = [
            ({"input_ids": [1, 2]}, "missing loss_mask"),
            (
                {"input_ids": [1, 2], "loss_mask": [1]},
                "mismatched input_ids/loss_mask lengths",
            ),
        ]
        for record, expected_message in cases:
            with self.subTest(record=record):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    path = os.path.join(tmp_dir, "invalid.jsonl")
                    _write_jsonl(path, [record])
                    with self.assertRaisesRegex(ValueError, expected_message):
                        prepare_prompt_tasks(
                            path,
                            tokenizer=None,
                            chat_template=None,
                            max_length=8,
                            is_preformatted=False,
                            train_only_last_turn=False,
                            cache_dir=None,
                            cache_key=None,
                            num_proc=1,
                        )

    def test_rejects_partial_cache_configuration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "prompts.jsonl")
            _write_jsonl(path, [{"input_ids": [1], "loss_mask": [1]}])
            with self.assertRaisesRegex(
                ValueError, "cache_dir and cache_key must be provided together"
            ):
                prepare_prompt_tasks(
                    path,
                    tokenizer=None,
                    chat_template=None,
                    max_length=8,
                    is_preformatted=False,
                    train_only_last_turn=False,
                    cache_dir=tmp_dir,
                    cache_key=None,
                    num_proc=1,
                )


if __name__ == "__main__":
    unittest.main()
