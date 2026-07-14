import io
import json
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts import prepare_data


class TestPrepareData(unittest.TestCase):
    def test_parser_exposes_only_canonical_presets(self):
        parser = prepare_data.build_parser()
        dataset_action = next(
            action for action in parser._actions if action.dest == "dataset"
        )

        self.assertEqual(
            ("ultrachat", "sharegpt"),
            tuple(dataset_action.choices),
        )

    def test_custom_data_path_is_limited_to_sharegpt_json(self):
        for suffix in (".json", ".jsonl"):
            with self.subTest(suffix=suffix):
                args = prepare_data.parse_args(
                    [
                        "--dataset",
                        "sharegpt",
                        "--data-path",
                        f"custom{suffix}",
                    ]
                )
                self.assertEqual(Path(f"custom{suffix}"), args.data_path)

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            prepare_data.parse_args(
                ["--dataset", "ultrachat", "--data-path", "custom.jsonl"]
            )
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            prepare_data.parse_args(
                ["--dataset", "sharegpt", "--data-path", "custom.csv"]
            )

    def test_custom_json_and_jsonl_use_the_hugging_face_json_loader(self):
        sentinel_dataset = object()
        for suffix in (".json", ".jsonl"):
            data_path = Path(f"custom{suffix}")
            with (
                self.subTest(suffix=suffix),
                patch.object(
                    prepare_data,
                    "_load_hf_dataset",
                    return_value=sentinel_dataset,
                ) as load_dataset,
            ):
                dataset = prepare_data.load_dataset_from_path(data_path)

                self.assertIs(sentinel_dataset, dataset)
                load_dataset.assert_called_once_with(
                    "json",
                    data_files=str(data_path),
                    split="train",
                )

    def test_sharegpt_conversion_normalizes_roles(self):
        row, skipped_count = prepare_data.process_sharegpt_row(
            {
                "id": 7,
                "conversations": [
                    {"from": "system", "value": "ignored"},
                    {"from": "human", "value": "question"},
                    {"from": "gpt", "value": "answer"},
                ],
            }
        )

        self.assertEqual(1, skipped_count)
        self.assertEqual(
            {
                "id": "7",
                "conversations": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ],
            },
            row,
        )

    def test_conversion_writes_only_the_training_jsonl(self):
        with TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            output_path = prepare_data.process_and_save_dataset(
                [
                    {
                        "prompt_id": "prompt-1",
                        "messages": [
                            {"role": "user", "content": "question"},
                            {"role": "assistant", "content": "answer"},
                        ],
                    }
                ],
                output_directory,
                prepare_data.process_ultrachat_row,
                "ultrachat",
            )

            self.assertEqual(output_directory / "ultrachat_train.jsonl", output_path)
            self.assertEqual([output_path], list(output_directory.iterdir()))
            self.assertEqual(
                {
                    "id": "prompt-1",
                    "conversations": [
                        {"role": "user", "content": "question"},
                        {"role": "assistant", "content": "answer"},
                    ],
                },
                json.loads(output_path.read_text()),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
