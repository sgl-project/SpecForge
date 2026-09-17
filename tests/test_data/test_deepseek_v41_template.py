"""DeepSeek-V4.1 Python encoder registration and rendering contract."""

import hashlib
import importlib
import json
import os
import tempfile
import unittest

import torch

from specforge.data.encoders import deepseek_v41 as encoder
from specforge.data.parse import DeepSeekV4Parser, DeepSeekV41Parser
from specforge.data.preprocessing import preprocess_conversations
from specforge.data.template import TEMPLATE_REGISTRY

GOLDEN_CASES = os.path.join(
    os.path.dirname(__file__),
    "test_references",
    "deepseek-v4.1_official-encoding-cases.json",
)

# sha256 of encoding/encoding.py in deepseek-ai/DeepSeek-V4.1-Flash at
# revision dba1be0a40aa45a94ad051997016db3960a90277, i.e. the vendored file
# without its license/vendoring header. Bump it together with the revision
# note in the header when re-vendoring.
VENDORED_SHA256 = "502bdaec8a3fd88ebc24c4721a7038fbe42f2063c664638127056107920035c1"

EFFORT_LINE = (
    "Reasoning Effort: {budget} "
    "(range 1-100, the higher the value, the more thorough the reasoning)\n\n"
)


class CharacterTokenizer:
    pad_token_id = 0
    unk_token_id = 0

    @staticmethod
    def _ids(text, max_length=None):
        values = [ord(character) for character in text]
        if max_length is not None:
            values = values[:max_length]
        return torch.tensor(values, dtype=torch.long)

    def __call__(
        self,
        text,
        max_length,
        truncation,
        return_tensors,
        add_special_tokens,
    ):
        class Encoding:
            pass

        encoding = Encoding()
        encoding.input_ids = self._ids(text, max_length)[None, :]
        return encoding

    def encode(self, text, add_special_tokens=False, truncation=True, max_length=None):
        return self._ids(text, max_length).tolist()


class TestDeepseekV41Template(unittest.TestCase):
    @staticmethod
    def _render(messages, tools=None, template="deepseek-v4.1", **kwargs):
        parser = DeepSeekV41Parser(None, TEMPLATE_REGISTRY.get(template))
        return parser.apply_chat_template(messages, tool=tools, **kwargs)

    def test_registered_with_python_encoder(self):
        for name, thinking in (
            ("deepseek-v4.1", False),
            ("deepseek-v4.1-thinking", True),
        ):
            with self.subTest(template=name):
                template = TEMPLATE_REGISTRY.get(name)
                self.assertIsNotNone(template)
                self.assertEqual(template.parser_type, "deepseek-v4.1")
                self.assertEqual(template.enable_thinking, thinking)
                self.assertEqual(template.assistant_header, "<｜Assistant｜>")
                self.assertEqual(template.user_header, "<｜User｜>")
                self.assertEqual(template.end_of_turn_token, "<｜end▁of▁sentence｜>")

    def test_parser_is_a_v4_parser_with_the_v41_encoder(self):
        self.assertTrue(issubclass(DeepSeekV41Parser, DeepSeekV4Parser))
        self.assertIs(DeepSeekV41Parser._encode_messages, encoder.encode_messages)
        self.assertIsNot(DeepSeekV4Parser._encode_messages, encoder.encode_messages)

    def test_vendored_encoder_is_pinned_to_the_model_release(self):
        with open(encoder.__file__, encoding="utf-8") as handle:
            source = handle.read()
        marker = "# fmt: off\n\n"
        self.assertIn(marker, source)
        body = source.split(marker, 1)[1]
        self.assertEqual(
            hashlib.sha256(body.encode("utf-8")).hexdigest(), VENDORED_SHA256
        )

    def test_python_encoder_basic_chat(self):
        rendered = self._render(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2 + 2 = 4."},
            ]
        )
        # Unlike V4, the system prompt sits behind the <｜System｜> token.
        self.assertEqual(
            rendered,
            "<｜begin▁of▁sentence｜><｜System｜>You are a helpful assistant."
            "<｜User｜>What is 2+2?"
            "<｜Assistant｜></think>2 + 2 = 4.<｜end▁of▁sentence｜>",
        )

    def test_chat_mode_drops_reasoning_content(self):
        rendered = self._render(
            [
                {"role": "user", "content": "Question"},
                {
                    "role": "assistant",
                    "reasoning_content": "must not be encoded",
                    "content": "Answer",
                },
            ]
        )
        self.assertEqual(
            rendered,
            "<｜begin▁of▁sentence｜><｜User｜>Question"
            "<｜Assistant｜></think>Answer<｜end▁of▁sentence｜>",
        )

    def test_thinking_mode_keeps_only_latest_turn_reasoning(self):
        rendered = self._render(
            [
                {"role": "user", "content": "First"},
                {
                    "role": "assistant",
                    "reasoning_content": "old reasoning",
                    "content": "First answer",
                },
                {"role": "user", "content": "Second"},
                {
                    "role": "assistant",
                    "reasoning_content": "latest reasoning",
                    "content": "Second answer",
                },
            ],
            enable_thinking=True,
        )
        # Thinking mode opens with the reasoning-effort prefix behind the
        # <｜System｜> token; the release default is "high" -> 75.
        self.assertEqual(
            rendered,
            "<｜begin▁of▁sentence｜><｜System｜>"
            + EFFORT_LINE.format(budget=75)
            + "<｜User｜>First"
            "<｜Assistant｜></think>First answer<｜end▁of▁sentence｜>"
            "<｜User｜>Second"
            "<｜Assistant｜><think>latest reasoning</think>"
            "Second answer<｜end▁of▁sentence｜>",
        )

    def test_reasoning_effort_accepts_an_integer_budget(self):
        rendered = self._render(
            [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "reasoning_content": "r", "content": "A"},
            ],
            enable_thinking=True,
            reasoning_effort=30,
        )
        self.assertTrue(
            rendered.startswith(
                "<｜begin▁of▁sentence｜><｜System｜>" + EFFORT_LINE.format(budget=30)
            )
        )

    def test_generation_prompt_uses_selected_mode(self):
        messages = [{"role": "user", "content": "Question"}]
        self.assertEqual(
            self._render(messages),
            "<｜begin▁of▁sentence｜><｜User｜>Question<｜Assistant｜></think>",
        )
        self.assertEqual(
            self._render(messages, enable_thinking=True),
            "<｜begin▁of▁sentence｜><｜System｜>"
            + EFFORT_LINE.format(budget=75)
            + "<｜User｜>Question<｜Assistant｜><think>",
        )

    def test_mid_conversation_system_message_uses_system_token(self):
        rendered = self._render(
            [
                {"role": "system", "content": "S"},
                {"role": "user", "content": "U1"},
                {"role": "assistant", "content": "A1"},
                {"role": "system", "content": "R"},
                {"role": "user", "content": "U2"},
                {"role": "assistant", "content": "A2"},
            ]
        )
        self.assertEqual(
            rendered,
            "<｜begin▁of▁sentence｜><｜System｜>S<｜User｜>U1"
            "<｜Assistant｜></think>A1<｜end▁of▁sentence｜>"
            "<｜System｜>R<｜User｜>U2"
            "<｜Assistant｜></think>A2<｜end▁of▁sentence｜>",
        )

    def test_external_tools_use_spaced_dsml_tags(self):
        rendered = self._render(
            [
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                },
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "description": "Weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )
        self.assertTrue(
            rendered.startswith("<｜begin▁of▁sentence｜><｜System｜>\n\n## Tools")
        )
        self.assertIn("### Available Tool Schemas", rendered)
        self.assertIn('<｜DSML｜ invoke name="weather">', rendered)
        self.assertIn(
            '<｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>',
            rendered,
        )
        # V4's unspaced tag names must be gone.
        self.assertNotIn("<｜DSML｜invoke", rendered)
        self.assertNotIn("<｜DSML｜tool_calls>", rendered)

    def test_official_golden_cases(self):
        # encoding/tests/test_input_N.json -> test_output_N.txt as shipped with
        # the model, stored in one JSON file so formatting hooks cannot touch the
        # expected bytes. The model's own test encodes each case in chat mode and
        # lets the case's thinking_mode/reasoning_effort override it.
        with open(GOLDEN_CASES, encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertEqual(len(document["cases"]), 5)
        for entry in document["cases"]:
            with self.subTest(case=entry["case_id"]):
                with tempfile.NamedTemporaryFile(
                    "w", suffix=".json", encoding="utf-8", delete=False
                ) as handle:
                    json.dump(entry["input"], handle, ensure_ascii=False)
                try:
                    case = encoder.load_cases(handle.name)[0]
                finally:
                    os.unlink(handle.name)
                prompt, _ = encoder.encode_case(case, thinking_mode="chat")
                self.assertEqual(prompt, entry["expected_prompt"])

    def test_preprocessing_masks_assistant_tokens(self):
        conversation = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
        cases = {
            "deepseek-v4.1": "</think>Answer<｜end▁of▁sentence｜>",
            "deepseek-v4.1-thinking": "<think></think>Answer<｜end▁of▁sentence｜>",
        }
        for name, supervised_text in cases.items():
            with self.subTest(template=name):
                processed = preprocess_conversations(
                    CharacterTokenizer(),
                    [conversation],
                    TEMPLATE_REGISTRY.get(name),
                    max_length=512,
                    tools=[[]],
                )
                self.assertEqual(
                    processed["loss_mask"][0].sum().item(), len(supervised_text)
                )
                self.assertTrue(torch.all(processed["attention_mask"][0] == 1))

    def test_preprocessing_masks_only_assistant_turns_around_mid_system(self):
        processed = preprocess_conversations(
            CharacterTokenizer(),
            [
                [
                    {"role": "system", "content": "S"},
                    {"role": "user", "content": "U1"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "system", "content": "R"},
                    {"role": "user", "content": "U2"},
                    {"role": "assistant", "content": "A2"},
                ]
            ],
            TEMPLATE_REGISTRY.get("deepseek-v4.1"),
            max_length=512,
            tools=[[]],
        )
        supervised = "</think>A1<｜end▁of▁sentence｜>" "</think>A2<｜end▁of▁sentence｜>"
        self.assertEqual(processed["loss_mask"][0].sum().item(), len(supervised))

    def test_encoder_errors_do_not_fall_back_to_generic_rendering(self):
        # GeneralParser would swallow a TypeError from the template and fall
        # back to header/footer rendering; the encoder path must surface it.
        parser = DeepSeekV41Parser(
            CharacterTokenizer(), TEMPLATE_REGISTRY.get("deepseek-v4.1")
        )
        with self.assertRaises(TypeError):
            parser.parse(
                [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ],
                max_length=512,
                tool=[
                    {
                        "type": "function",
                        "function": {"name": "weather", "parameters": object()},
                    }
                ],
            )


def _sglang_v41_encoder():
    try:
        return importlib.import_module("sglang.srt.entrypoints.openai.encoding_dsv41")
    except Exception:  # pragma: no cover - sglang absent or too old
        return None


@unittest.skipUnless(
    _sglang_v41_encoder(), "sglang with the V4.1 encoder is not installed"
)
class TestDeepseekV41ServingParity(unittest.TestCase):
    """Training renders what an SGLang server renders for the same conversation.

    Chat mode and integer reasoning budgets are compared verbatim. The named
    tiers (low/high/max) are deliberately not compared here: SGLang's copy of
    the encoder still carries the pre-release tier budgets
    (sgl-project/sglang#39909), while the vendored file follows the release.
    """

    CONVERSATIONS = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2 + 2 = 4."},
        ],
        [
            {"role": "user", "content": "First"},
            {"role": "assistant", "reasoning_content": "old", "content": "A1"},
            {"role": "system", "content": "Reminder"},
            {"role": "user", "content": "Second"},
            {"role": "assistant", "reasoning_content": "new", "content": "A2"},
        ],
    ]

    def test_renders_match_sglang(self):
        sgl = _sglang_v41_encoder()
        for index, messages in enumerate(self.CONVERSATIONS):
            for thinking, effort in ((False, None), (True, 40)):
                with self.subTest(conversation=index, thinking=thinking):
                    ours = DeepSeekV41Parser(
                        None, TEMPLATE_REGISTRY.get("deepseek-v4.1")
                    ).apply_chat_template(
                        messages,
                        tool=[],
                        enable_thinking=thinking,
                        reasoning_effort=effort,
                    )
                    theirs = sgl.encode_messages(
                        [dict(m) for m in messages],
                        thinking_mode="thinking" if thinking else "chat",
                        reasoning_effort=effort,
                    )
                    self.assertEqual(ours, theirs)


if __name__ == "__main__":
    unittest.main()
