"""Training renders Qwen3.8 conversations the way serving prompts them.

Qwen3.8 recipes use the ``qwen3.5`` template entry (the turn markers did not
change), which renders a whole conversation once, with
``enable_thinking=True`` and every other template variable at its default, and
supervises each assistant span. That is right only if, for every assistant
turn, the prompt a server builds for that turn is a prefix of the training
text, and the supervised span is exactly what the model then generates.

These tests render the released Jinja (``data/qwen3.8_chat_template.jinja``,
byte-identical in Qwen3.8-27B and Qwen3.8-Flash-Next) both ways and check that,
for single-turn, multi-turn and tool-calling conversations -- first as text,
then through ``ThinkingParser.parse`` itself, whose loss mask must cover each
turn's generated tokens and nothing of any prompt. They also pin the
serving settings under which it fails, because the corpus and the evaluation
must avoid them:

* ``reasoning_effort`` other than ``xhigh``: the effort is a system-prompt
  sentence, and training always renders the default, ``xhigh``.
* ``preserve_thinking=False``: earlier turns lose their reasoning in the
  prompt but keep it in the training text.
* ``enable_thinking=False``: the prompt pre-fills an empty thought and drops
  the effort sentence; the training text supervises that empty thought.
"""

import hashlib
import os
import re
import unittest

from specforge.data.parse import ThinkingParser
from specforge.data.template import TEMPLATE_REGISTRY

TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "qwen3.8_chat_template.jinja"
)
# sha256 of chat_template.jinja in Qwen/Qwen3.8-Flash-Next and Qwen/Qwen3.8-27B.
RELEASED_TEMPLATE_SHA256 = (
    "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
)

with open(TEMPLATE_PATH, encoding="utf-8") as _stream:
    TEMPLATE = _stream.read()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


class _JinjaTokenizer:
    """Just enough tokenizer for ThinkingParser.apply_chat_template."""

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
        del tokenize, add_special_tokens
        return render(
            messages, tools=tools, add_generation_prompt=add_generation_prompt, **kwargs
        )


def render(messages, *, tools=None, add_generation_prompt=False, **kwargs):
    from transformers.utils.chat_template_utils import render_jinja_template

    rendered, _ = render_jinja_template(
        [messages],
        tools=tools,
        chat_template=TEMPLATE,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )
    return rendered[0]


def generation(message):
    """The text the model emits for an assistant turn, after the prompt's <think>\\n."""

    text = message.get("reasoning_content", "").strip() + "\n</think>\n\n"
    content = message.get("content") or ""
    text += content
    for index, call in enumerate(message.get("tool_calls") or []):
        function = call["function"]
        if index == 0:
            text += "\n\n<tool_call>\n" if content.strip() else "<tool_call>\n"
        else:
            text += "\n<tool_call>\n"
        text += f"<function={function['name']}>\n"
        for name, value in function["arguments"].items():
            text += f"<parameter={name}>\n{value}\n</parameter>\n"
        text += "</function>\n</tool_call>"
    return text + "<|im_end|>"


SINGLE_TURN = [
    {"role": "user", "content": "What is 2 + 2?"},
    {
        "role": "assistant",
        "reasoning_content": "Two plus two is four.",
        "content": "4",
    },
]

MULTI_TURN = [
    {"role": "user", "content": "Name a prime."},
    {"role": "assistant", "reasoning_content": "2 is prime.", "content": "2"},
    {"role": "user", "content": "And an even one?"},
    {"role": "assistant", "reasoning_content": "2 is even too.", "content": "2 again"},
    {"role": "user", "content": "Thanks."},
    {
        "role": "assistant",
        "reasoning_content": "Be polite.",
        "content": "You're welcome.",
    },
]

TOOL_TURNS = [
    {"role": "system", "content": "You are a weather bot."},
    {"role": "user", "content": "Weather in Paris?"},
    {
        "role": "assistant",
        "reasoning_content": "I should call the tool.",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
            }
        ],
    },
    {"role": "tool", "content": "Sunny, 21C"},
    {
        "role": "assistant",
        "reasoning_content": "The tool says sunny.",
        "content": "It is sunny and 21C in Paris.",
    },
]


class Qwen38TemplateServingTest(unittest.TestCase):
    def setUp(self):
        self.parser = ThinkingParser(
            _JinjaTokenizer(), TEMPLATE_REGISTRY.get("qwen3.5")
        )

    def training_render(self, messages, tools=None):
        # What ThinkingParser.parse renders: enable_thinking=True, nothing else.
        return self.parser.apply_chat_template(messages, tools, enable_thinking=True)

    def supervised_spans(self, text):
        return [
            match.group(1)
            for match in re.finditer(self.parser.assistant_pattern, text, re.DOTALL)
        ]

    def assert_serving_consistent(self, messages, tools=None, **serving_kwargs):
        text = self.training_render(messages, tools)
        spans = self.supervised_spans(text)
        assistant_turns = [
            i for i, m in enumerate(messages) if m["role"] == "assistant"
        ]
        self.assertEqual(len(spans), len(assistant_turns))
        for span, turn in zip(spans, assistant_turns):
            with self.subTest(turn=turn):
                prompt = render(
                    messages[:turn],
                    tools=tools,
                    add_generation_prompt=True,
                    **serving_kwargs,
                )
                self.assertTrue(text.startswith(prompt))
                # The span is what the model generates, plus the newline the
                # template writes after <|im_end|>, which generation stops at.
                self.assertEqual(span, generation(messages[turn]) + "\n")
                self.assertEqual(text[len(prompt) :].index(span), 0)

    def test_fixture_is_the_released_template(self):
        digest = hashlib.sha256(TEMPLATE.encode("utf-8")).hexdigest()
        self.assertEqual(digest, RELEASED_TEMPLATE_SHA256)

    def test_single_turn(self):
        self.assert_serving_consistent(SINGLE_TURN)

    def test_every_turn_of_a_multi_turn_dialogue(self):
        self.assert_serving_consistent(MULTI_TURN)

    def test_tool_calling_trajectory(self):
        self.assert_serving_consistent(TOOL_TURNS, tools=TOOLS)

    def test_serving_defaults_are_what_training_renders(self):
        # Spelling the defaults out changes nothing: xhigh effort, preserved
        # thinking, thinking on.
        explicit = dict(
            enable_thinking=True, reasoning_effort="xhigh", preserve_thinking=True
        )
        self.assert_serving_consistent(MULTI_TURN, **explicit)
        self.assert_serving_consistent(TOOL_TURNS, tools=TOOLS, **explicit)

    def test_other_reasoning_efforts_change_the_system_prompt(self):
        text = self.training_render(SINGLE_TURN)
        self.assertIn("Reasoning effort is set to xhigh.", text)
        for effort in ("medium", "low"):
            with self.subTest(effort=effort):
                prompt = render(
                    SINGLE_TURN[:1], add_generation_prompt=True, reasoning_effort=effort
                )
                self.assertFalse(text.startswith(prompt))

    def test_dropping_preserved_thinking_breaks_later_turns(self):
        text = self.training_render(MULTI_TURN)
        assistant_turns = [
            i for i, m in enumerate(MULTI_TURN) if m["role"] == "assistant"
        ]
        consistent = [
            text.startswith(
                render(
                    MULTI_TURN[:turn],
                    add_generation_prompt=True,
                    preserve_thinking=False,
                )
            )
            for turn in assistant_turns
        ]
        # Only the first turn has no earlier reasoning to drop.
        self.assertEqual(consistent, [True, False, False])

    def test_non_thinking_requests_do_not_match(self):
        row = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "reasoning_content": "", "content": "Hello!"},
        ]
        text = self.training_render(row)
        prompt = render(row[:1], add_generation_prompt=True, enable_thinking=False)
        # No effort sentence and a pre-filled empty thought in the prompt ...
        self.assertTrue(
            prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        )
        self.assertFalse(text.startswith(prompt))
        # ... while training supervises that empty thought as if generated.
        (span,) = self.supervised_spans(text)
        self.assertTrue(span.startswith("\n</think>\n\nHello!"))


@unittest.skipUnless(
    os.environ.get("QWEN38_TOKENIZER_PATH"),
    "set QWEN38_TOKENIZER_PATH to a Qwen3.8 tokenizer directory",
)
class Qwen38TokenServingTest(unittest.TestCase):
    """The same check one level down, in tokens.

    Training tokenizes the whole rendered text; serving tokenizes the prompt and
    appends what the model generates. A merge across that boundary would shift
    every supervised token, so each turn's prompt tokens must prefix the
    training tokens and the generation must tokenize the same alone as in
    context.
    """

    def test_turn_boundaries_survive_tokenization(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(os.environ["QWEN38_TOKENIZER_PATH"])

        def ids(text):
            return tokenizer(text, add_special_tokens=False).input_ids

        for name, messages, tools in (
            ("single", SINGLE_TURN, None),
            ("multi", MULTI_TURN, None),
            ("tools", TOOL_TURNS, TOOLS),
        ):
            full = ids(render(messages, tools=tools, enable_thinking=True))
            for turn, message in enumerate(messages):
                if message["role"] != "assistant":
                    continue
                with self.subTest(conversation=name, turn=turn):
                    prompt = ids(
                        render(messages[:turn], tools=tools, add_generation_prompt=True)
                    )
                    generated = ids(generation(message))
                    self.assertEqual(full[: len(prompt)], prompt)
                    self.assertEqual(
                        full[len(prompt) : len(prompt) + len(generated)], generated
                    )


# Qwen3.8's added tokens that the template and the corpus emit; each is one id.
_SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
_ADDED = [
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
]


def byte_tokenizer():
    """A byte-level tokenizer with Qwen3.8's added tokens and released template.

    Every byte is its own token, so the test needs no download, while the added
    tokens keep the turn markers single tokens as in the real vocabulary.
    """

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(
        models.BPE(vocab={char: i for i, char in enumerate(alphabet)}, merges=[])
    )
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=_SPECIAL,
    )
    tokenizer.add_tokens(_ADDED)
    tokenizer.chat_template = TEMPLATE
    return tokenizer


class ParsedLossMaskMixin:
    """Run the training parser and compare its loss mask with serving, per turn.

    For each assistant turn: the turn's serving prompt tokens prefix the parsed
    input ids, the generated tokens follow them, and the loss mask is exactly
    those generated tokens plus the newline the template writes after
    <|im_end|> -- nothing from any prompt.
    """

    tokenizer = None

    def check(self, messages, tools=None):
        parser = ThinkingParser(self.tokenizer, TEMPLATE_REGISTRY.get("qwen3.5"))
        input_ids, loss_mask = parser.parse(
            [dict(message) for message in messages],
            max_length=100_000,
            tool=tools or [],
        )
        input_ids, loss_mask = input_ids.tolist(), loss_mask.tolist()

        def ids(text):
            return self.tokenizer.encode(text, add_special_tokens=False)

        def same(actual, wanted, what):
            # Positions, not a diff: difflib recurses too deep on long sequences.
            wrong = [i for i, (a, b) in enumerate(zip(actual, wanted)) if a != b]
            self.assertEqual(len(actual), len(wanted), f"{what}: length")
            self.assertFalse(
                wrong, f"{what}: {len(wrong)} positions differ, first {wrong[:8]}"
            )

        expected = [0] * len(input_ids)
        for turn, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            with self.subTest(turn=turn):
                prompt = ids(
                    render(messages[:turn], tools=tools, add_generation_prompt=True)
                )
                generated = ids(generation(message))
                end = len(prompt) + len(generated)
                same(input_ids[: len(prompt)], prompt, "prompt tokens")
                same(input_ids[len(prompt) : end], generated, "generated tokens")
                same(input_ids[end : end + 1], ids("\n"), "newline after <|im_end|>")
                for index in range(len(prompt), end + 1):
                    expected[index] = 1
        self.assertGreater(sum(expected), 0)
        same(loss_mask, expected, "loss mask")

    def test_single_turn(self):
        self.check(SINGLE_TURN)

    def test_multi_turn(self):
        self.check(MULTI_TURN)

    def test_tool_calling_trajectory(self):
        self.check(TOOL_TURNS, tools=TOOLS)


class Qwen38ParsedLossMaskTest(ParsedLossMaskMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = byte_tokenizer()


@unittest.skipUnless(
    os.environ.get("QWEN38_TOKENIZER_PATH"),
    "set QWEN38_TOKENIZER_PATH to a Qwen3.8 tokenizer directory",
)
class Qwen38ParsedLossMaskRealTokenizerTest(ParsedLossMaskMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(
            os.environ["QWEN38_TOKENIZER_PATH"]
        )


if __name__ == "__main__":
    unittest.main()
