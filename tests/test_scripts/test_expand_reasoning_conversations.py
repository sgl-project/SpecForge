import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from scripts.expand_reasoning_conversations import (
    expand_row,
    main,
    skipped_path,
    validate_conversation,
)


def _load_preprocessing_stack():
    repo_root = Path(__file__).resolve().parents[2]
    package_name = "_specforge_data_test"
    root_package = types.ModuleType(package_name)
    root_package.__path__ = []
    sys.modules[package_name] = root_package

    data_package_name = f"{package_name}.data"
    data_package = types.ModuleType(data_package_name)
    data_package.__path__ = []
    sys.modules[data_package_name] = data_package

    for module_name in ("template", "parse", "preprocessing"):
        full_name = f"{data_package_name}.{module_name}"
        module_path = repo_root / "specforge" / "data" / f"{module_name}.py"
        spec = importlib.util.spec_from_file_location(full_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)

    template_module = sys.modules[f"{data_package_name}.template"]
    preprocessing_module = sys.modules[f"{data_package_name}.preprocessing"]
    return (
        preprocessing_module.preprocess_conversations,
        template_module.TEMPLATE_REGISTRY,
    )


class _CharEncoding:
    def __init__(self, text: str, max_length: int | None = None):
        ids = [ord(char) for char in text]
        if max_length is not None:
            ids = ids[:max_length]
        self.input_ids = torch.tensor([ids], dtype=torch.long)


class _CharTokenizer:
    pad_token_id = 0
    unk_token_id = 0

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
        tools=None,
        **kwargs,
    ):
        del tokenize, add_generation_prompt, add_special_tokens, tools, kwargs
        rendered = []
        for message in messages:
            content = message.get("content", "")
            if message["role"] == "assistant" and "reasoning_content" in message:
                content = (
                    f"<think>\n{message['reasoning_content']}</think>\n\n" f"{content}"
                )
            rendered.append(f"<|im_start|>{message['role']}\n{content}<|im_end|>\n")
        return "".join(rendered)

    def __call__(
        self,
        text,
        max_length=None,
        truncation=True,
        return_tensors=None,
        add_special_tokens=False,
        **kwargs,
    ):
        del truncation, return_tensors, add_special_tokens, kwargs
        return _CharEncoding(text, max_length=max_length)

    def encode(
        self,
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=None,
    ):
        del add_special_tokens, truncation
        ids = [ord(char) for char in text]
        return ids[:max_length] if max_length is not None else ids

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(int(token_id)) for token_id in ids)


class TestExpandGenerationEvents(unittest.TestCase):
    def test_expand_reasoning_conversation_strips_history_reasoning(self):
        row = {
            "id": "abc",
            "status": "success",
            "conversations": [
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": "u1"},
                {
                    "role": "assistant",
                    "reasoning_content": "r1",
                    "content": "a1",
                },
                {"role": "user", "content": "u2"},
                {
                    "role": "assistant",
                    "reasoning_content": "r2",
                    "content": "a2",
                },
            ],
        }

        events = expand_row(row, source_row_index=7)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["id"], "abc#turn0")
        self.assertEqual(events[0]["source_id"], "abc")
        self.assertEqual(events[0]["source_row_index"], 7)
        self.assertEqual(events[0]["assistant_turn_index"], 0)
        self.assertEqual(
            events[0]["conversations"][-1],
            {
                "role": "assistant",
                "reasoning_content": "r1",
                "content": "a1",
            },
        )

        second_messages = events[1]["conversations"]
        self.assertEqual(
            second_messages[0], {"role": "system", "content": "be concise"}
        )
        self.assertEqual(
            second_messages,
            [
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"},
                {
                    "role": "assistant",
                    "reasoning_content": "r2",
                    "content": "a2",
                },
            ],
        )

    def test_validate_rejects_trailing_user(self):
        reason = validate_conversation(
            [
                {"role": "user", "content": "u1"},
                {
                    "role": "assistant",
                    "content": "a1",
                    "reasoning_content": "r1",
                },
                {"role": "user", "content": "u2"},
            ]
        )

        self.assertEqual(reason, "Conversation ends with a user message")

    def test_validate_rejects_missing_or_invalid_content(self):
        for content in (None, "", []):
            with self.subTest(content=content):
                reason = validate_conversation(
                    [
                        {"role": "user", "content": content},
                        {
                            "role": "assistant",
                            "content": "answer",
                            "reasoning_content": "reasoning",
                        },
                    ]
                )

                self.assertIn("expected non-empty string", reason)

    def test_validate_rejects_missing_or_invalid_reasoning(self):
        for reasoning in (None, "", []):
            with self.subTest(reasoning=reasoning):
                reason = validate_conversation(
                    [
                        {"role": "user", "content": "question"},
                        {
                            "role": "assistant",
                            "content": "answer",
                            "reasoning_content": reasoning,
                        },
                    ]
                )

                self.assertIn("assistant reasoning_content", reason)
                self.assertIn("expected non-empty string", reason)

    def test_main_writes_events_and_skipped_rows(self):
        with TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            output_path = Path(tmpdir) / "events.jsonl"
            skipped_path = Path(tmpdir) / "skipped.jsonl"
            rows = [
                {
                    "id": "ok",
                    "status": "success",
                    "conversations": [
                        {"role": "user", "content": "u1"},
                        {
                            "role": "assistant",
                            "reasoning_content": "r1",
                            "content": "a1",
                        },
                        {"role": "user", "content": "u2"},
                        {
                            "role": "assistant",
                            "reasoning_content": "r2",
                            "content": "a2",
                        },
                    ],
                },
                {
                    "id": "bad",
                    "status": "success",
                    "conversations": [
                        {"role": "assistant", "content": "starts wrong"},
                    ],
                },
            ]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            argv = [
                "expand_reasoning_conversations.py",
                "--input-file-path",
                str(input_path),
                "--output-file-path",
                str(output_path),
                "--skipped-file-path",
                str(skipped_path),
            ]
            with patch("sys.argv", argv):
                main()

            output_rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            skipped_rows = [
                json.loads(line)
                for line in skipped_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(len(output_rows), 2)
            self.assertEqual(output_rows[1]["id"], "ok#turn1")
            self.assertNotIn("reasoning_content", output_rows[1]["conversations"][1])
            self.assertEqual(len(skipped_rows), 1)
            self.assertEqual(skipped_rows[0]["status"], "skipped")
            self.assertIn("expected user", skipped_rows[0]["error"])

    def test_expand_rejects_non_success_or_invalid_id(self):
        conversations = [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "reasoning",
            },
        ]
        invalid_rows = [
            {"id": "row", "status": "error", "conversations": conversations},
            {"id": "", "status": "success", "conversations": conversations},
            {"id": None, "status": "success", "conversations": conversations},
        ]

        for row in invalid_rows:
            with self.subTest(row=row):
                with self.assertRaises(ValueError):
                    expand_row(row, source_row_index=0)

    def test_main_records_non_object_row_as_skipped(self):
        with TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            output_path = Path(tmpdir) / "events.jsonl"
            input_path.write_text("[]\n", encoding="utf-8")

            argv = [
                "expand_reasoning_conversations.py",
                "--input-file-path",
                str(input_path),
                "--output-file-path",
                str(output_path),
            ]
            with patch("sys.argv", argv):
                main()

            skipped_rows = [
                json.loads(line)
                for line in Path(skipped_path(str(output_path)))
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(skipped_rows[0]["data"], [])
            self.assertEqual(skipped_rows[0]["status"], "skipped")
            self.assertIn("JSON object", skipped_rows[0]["error"])

    def test_main_rejects_input_output_collision_without_truncating_input(self):
        with TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            original = '{"id": "keep"}\n'
            input_path.write_text(original, encoding="utf-8")

            argv = [
                "expand_reasoning_conversations.py",
                "--input-file-path",
                str(input_path),
                "--output-file-path",
                str(input_path),
            ]
            with patch("sys.argv", argv):
                with self.assertRaisesRegex(SystemExit, "must be distinct"):
                    main()

            self.assertEqual(input_path.read_text(encoding="utf-8"), original)

    def test_main_refuses_existing_output(self):
        with TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            output_path = Path(tmpdir) / "events.jsonl"
            input_path.write_text("{}\n", encoding="utf-8")
            output_path.write_text("keep\n", encoding="utf-8")

            argv = [
                "expand_reasoning_conversations.py",
                "--input-file-path",
                str(input_path),
                "--output-file-path",
                str(output_path),
            ]
            with patch("sys.argv", argv):
                with self.assertRaisesRegex(SystemExit, "refusing to overwrite"):
                    main()

            self.assertEqual(output_path.read_text(encoding="utf-8"), "keep\n")

    def test_skipped_path_requires_jsonl_output(self):
        with self.assertRaisesRegex(ValueError, "must end in .jsonl"):
            skipped_path("events.txt")

    def test_expanded_event_qwen35_masks_only_current_generation(self):
        preprocess_conversations, template_registry = _load_preprocessing_stack()

        row = {
            "id": "abc",
            "status": "success",
            "conversations": [
                {"role": "user", "content": "u1"},
                {
                    "role": "assistant",
                    "reasoning_content": "r1",
                    "content": "a1",
                },
                {"role": "user", "content": "u2"},
                {
                    "role": "assistant",
                    "reasoning_content": "r2",
                    "content": "a2",
                },
            ],
        }
        second_event = expand_row(row, source_row_index=0)[1]
        tokenizer = _CharTokenizer()
        template = template_registry.get("qwen3.5")

        all_turns = preprocess_conversations(
            tokenizer,
            [second_event["conversations"]],
            template,
            max_length=1024,
            train_only_last_turn=False,
        )
        last_turn = preprocess_conversations(
            tokenizer,
            [second_event["conversations"]],
            template,
            max_length=1024,
            train_only_last_turn=True,
        )

        def masked_text(processed):
            input_ids = processed["input_ids"][0].view(-1)
            mask = processed["loss_mask"][0].view(-1).bool()
            return tokenizer.decode(input_ids[mask].tolist())

        all_turns_text = masked_text(all_turns)
        current_only_text = masked_text(last_turn)
        self.assertEqual(all_turns_text, current_only_text)
        self.assertNotIn("a1", all_turns_text)
        self.assertIn("r2", all_turns_text)
        self.assertIn("a2", all_turns_text)
        self.assertNotIn("a1", current_only_text)
        self.assertIn("r2", current_only_text)
        self.assertIn("a2", current_only_text)

    def test_non_expanded_last_turn_matches_last_event_only_after_stripping_history_reasoning(
        self,
    ):
        preprocess_conversations, template_registry = _load_preprocessing_stack()

        row = {
            "id": "abc",
            "status": "success",
            "conversations": [
                {"role": "user", "content": "u1"},
                {
                    "role": "assistant",
                    "reasoning_content": "r1",
                    "content": "a1",
                },
                {"role": "user", "content": "u2"},
                {
                    "role": "assistant",
                    "reasoning_content": "r2",
                    "content": "a2",
                },
                {"role": "user", "content": "u3"},
                {
                    "role": "assistant",
                    "reasoning_content": "r3",
                    "content": "a3",
                },
            ],
        }
        last_event = expand_row(row, source_row_index=0)[-1]
        tokenizer = _CharTokenizer()
        template = template_registry.get("qwen3.5")

        def render(messages):
            processed = preprocess_conversations(
                tokenizer,
                [messages],
                template,
                max_length=1024,
                train_only_last_turn=True,
            )
            input_ids = processed["input_ids"][0].view(-1)
            mask = processed["loss_mask"][0].view(-1).bool()
            return (
                tokenizer.decode(input_ids.tolist()),
                tokenizer.decode(input_ids[mask].tolist()),
            )

        original_input_text, original_loss_text = render(row["conversations"])
        event_input_text, event_loss_text = render(last_event["conversations"])

        self.assertEqual(original_loss_text, event_loss_text)
        self.assertIn("r3", event_loss_text)
        self.assertIn("a3", event_loss_text)
        self.assertNotEqual(original_input_text, event_input_text)
        self.assertIn("r1", original_input_text)
        self.assertIn("r2", original_input_text)
        self.assertNotIn("r1", event_input_text)
        self.assertNotIn("r2", event_input_text)

        stripped_history = last_event["conversations"]
        stripped_input_text, stripped_loss_text = render(stripped_history)

        self.assertEqual(stripped_input_text, event_input_text)
        self.assertEqual(stripped_loss_text, event_loss_text)


if __name__ == "__main__":
    unittest.main()
