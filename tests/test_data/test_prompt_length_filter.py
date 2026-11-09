"""Whole-record filtering at the native loader's public boundary."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from specforge.config.schema import DataConfig
from specforge.data.loss_mask import has_consecutive_supervised_tokens
from specforge.data.prompt_builder import prepare_prompt_tasks


class TestPromptLengthFilter(unittest.TestCase):
    def load(self, records, **options):
        report = {}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "data.jsonl")
            path.write_text("".join(json.dumps(r) + "\n" for r in records))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                prompts = prepare_prompt_tasks(
                    path,
                    tokenizer=None,
                    chat_template=None,
                    max_length=65536,
                    is_preformatted=False,
                    train_only_last_turn=False,
                    cache_dir=None,
                    cache_key=None,
                    num_proc=1,
                    min_loss_tokens=2,
                    loss_mask_filter=has_consecutive_supervised_tokens,
                    overlength_policy="drop",
                    filter_report=report,
                    **options,
                )
        self.assertIn("[prompt-filter]", output.getvalue())
        return prompts, report

    def test_boundary_and_original_ids_masks_survive(self):
        records = [
            dict(id=str(n), input_ids=list(range(n)), loss_mask=[0] * (n - 2) + [1, 1])
            for n in (65535, 65536, 65537)
        ]
        prompts, report = self.load(records)
        self.assertEqual(
            [p["payload"] for p in prompts],
            [{k: r[k] for k in ("input_ids", "loss_mask")} for r in records[:2]],
        )
        self.assertEqual(report["accepted"], 2)
        self.assertEqual(report["rows_seen"], 3)
        self.assertEqual(report["dropped_other"], 0)
        self.assertEqual(report["dropped_overlength"], 1)
        self.assertEqual(report["kept_input_tokens"], 131071)
        self.assertEqual(report["kept_loss_tokens"], 4)
        self.assertEqual(report["dropped_samples"][0]["id"], "65537")

    def test_limit_counts_accepted_after_filter(self):
        records = [
            dict(id="long", input_ids=[1] * 65537, loss_mask=[1] * 65537),
            dict(id="invalid_supervision", input_ids=[2, 3, 4], loss_mask=[1, 0, 1]),
            dict(id="keep", input_ids=[[4, 5]], loss_mask=[[1, 1]]),
            dict(id="unread", input_ids=[6, 7], loss_mask=[1, 1]),
        ]
        prompts, report = self.load(records, max_prompts=1)
        self.assertEqual(
            prompts, [{"payload": {"input_ids": [4, 5], "loss_mask": [1, 1]}}]
        )
        self.assertEqual(report["rows_seen"], 3)
        self.assertEqual(report["dropped_other"], 1)
        self.assertEqual(report["dropped_overlength"], 1)

    def test_zero_and_none_limits_keep_all_accepted_records(self):
        records = [
            dict(id="long", input_ids=[1] * 65537, loss_mask=[1] * 65537),
            dict(id="first", input_ids=[2, 3], loss_mask=[1, 1]),
            dict(id="second", input_ids=[4, 5, 6], loss_mask=[0, 1, 1]),
        ]
        for limit in (None, 0):
            with self.subTest(limit=limit):
                prompts, report = self.load(records, max_prompts=limit)
                self.assertEqual(
                    [p["payload"]["input_ids"] for p in prompts], [[2, 3], [4, 5, 6]]
                )
                self.assertEqual(report["rows_seen"], 3)
                self.assertEqual(report["accepted"], 2)
                self.assertEqual(report["dropped_overlength"], 1)

    def test_malformed_overlength_still_fails_validation(self):
        with self.assertRaisesRegex(ValueError, "mismatched"):
            self.load([dict(input_ids=[1] * 65537, loss_mask=[1, 1])])
        with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
            self.load([dict(input_ids=[1] * 65537, loss_mask=[1] * 65536 + [2])])

    def test_raw_conversations_cannot_silently_truncate(self):
        with self.assertRaisesRegex(ValueError, "requires pre-tokenized"):
            self.load([{"conversations": [{"role": "user", "content": "hello"}]}])

    def test_schema_requires_native_source_and_defaults_to_legacy(self):
        self.assertEqual(
            DataConfig(prompts_path="data.jsonl").overlength_policy, "truncate"
        )
        self.assertEqual(
            DataConfig(
                prompts_path="data.jsonl", overlength_policy="drop"
            ).overlength_policy,
            "drop",
        )
        with self.assertRaisesRegex(
            ValidationError, "requires native data.prompts_path"
        ):
            DataConfig(train_data_path="raw.jsonl", overlength_policy="drop")
        with self.assertRaises(ValidationError):
            DataConfig(prompts_path="data.jsonl", overlength_policy="invalid")


if __name__ == "__main__":
    unittest.main()
