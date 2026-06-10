import os
import tempfile
import unittest

import torch
from transformers import AutoTokenizer

from datasets import Dataset
from specforge.data.preprocessing import build_eagle3_dataset
from specforge.utils import safe_conversations_generator

# ANSI color codes
RED = "\033[91m"
RESET = "\033[0m"


def print_with_loss_mask(tokenizer, input_ids, loss_mask, title=""):
    """Print text with loss_mask=1 (assistant) parts in RED."""
    input_ids = input_ids.flatten()
    loss_mask = loss_mask.flatten()

    print(f"\n{'=' * 60}")
    print(f"{title}")
    print("=" * 60)

    # Group consecutive tokens by loss_mask value
    current_mask = loss_mask[0].item()
    current_ids = [input_ids[0].item()]

    for i in range(1, len(input_ids)):
        if loss_mask[i].item() == current_mask:
            current_ids.append(input_ids[i].item())
        else:
            # Decode and print current group
            text = tokenizer.decode(current_ids, skip_special_tokens=False)
            if current_mask == 1:
                print(f"{RED}{text}{RESET}", end="")
            else:
                print(text, end="")
            current_ids = [input_ids[i].item()]
            current_mask = loss_mask[i].item()

    # Print remaining tokens
    if current_ids:
        text = tokenizer.decode(current_ids, skip_special_tokens=False)
        if current_mask == 1:
            print(f"{RED}{text}{RESET}")
        else:
            print(text)

    print("=" * 60)


# Tools definition from specforge/data/tools.py
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit of temperature",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

# 1 sample from test_parsers.py: tool_use_messages
TOOL_USE_CONVERSATION = [
    {"role": "user", "content": "我想知道今天北京和上海的天气怎么样？"},
    {
        "role": "assistant",
        "content": "我来帮您查询北京和上海的天气情况。",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": {"location": "北京", "date": "today"},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": {"location": "上海", "date": "today"},
                },
            },
        ],
    },
    {
        "role": "tool",
        "content": '{"location": "北京", "temperature": 25, "condition": "晴朗", "humidity": "45%"}',
    },
    {
        "role": "tool",
        "content": '{"location": "上海", "temperature": 28, "condition": "多云", "humidity": "65%"}',
    },
    {
        "role": "assistant",
        "content": "根据查询结果，北京今天晴朗，25°C；上海多云，28°C。两地都比较适合出行。",
    },
]


class TestBuildEagle3Dataset(unittest.TestCase):
    """Test for build_eagle3_dataset with tools from specforge/data/tools.py."""

    @classmethod
    def setUpClass(cls):
        cls.model_name = "Qwen/Qwen3.5-35B-A3B"
        cls.template_key = "qwen3.5"
        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model_name, trust_remote_code=True
        )
        cls.max_length = 65535

    def test_build_eagle3_dataset_basic(self):
        """Test build_eagle3_dataset with 1 tool_use conversation sample."""
        # Create a HF Dataset with 1 sample
        data_file = os.path.join(
            os.path.dirname(__file__), "data", "tool_use_conversation.jsonl"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = Dataset.from_generator(
                generator=safe_conversations_generator,
                gen_kwargs={"file_path": data_file},
                cache_dir=tmp_dir,
                keep_in_memory=True,
            )
            result_dataset = build_eagle3_dataset(
                dataset=dataset,
                tokenizer=self.tokenizer,
                chat_template=self.template_key,
                max_length=self.max_length,
                shuffle_seed=42,
                num_proc=1,
                cache_dir=None,
                cache_key=None,
            )

            # Verify the dataset has the expected columns
            self.assertIn("input_ids", result_dataset.column_names)
            self.assertIn("loss_mask", result_dataset.column_names)
            self.assertIn("attention_mask", result_dataset.column_names)
            self.assertEqual(len(result_dataset), 1)

            # Decode input_ids to text
            input_ids = result_dataset[0]["input_ids"].squeeze()
            loss_mask = result_dataset[0]["loss_mask"].squeeze()

            # Print full text with loss_mask=1 in RED
            print_with_loss_mask(
                self.tokenizer,
                input_ids,
                loss_mask,
                title="[build_eagle3_dataset] Full text (RED = loss_mask=1):",
            )

            # Verify assistant tokens exist
            assistant_indices = torch.where(loss_mask == 1)[0]
            self.assertTrue(len(assistant_indices) > 0, "No assistant tokens found")


if __name__ == "__main__":
    unittest.main(verbosity=2)
