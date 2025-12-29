import unittest

import torch
from transformers import AutoTokenizer

from specforge.data.preprocessing import preprocess_conversations
from specforge.data.template import TEMPLATE_REGISTRY


# Utility function for visual debugging
def visualize_loss_mask(tokenizer, input_ids, loss_mask):
    """Utility function to visualize which tokens contribute to loss."""
    RED = "\033[91m"  # Non-assistant tokens (loss_mask = 0)
    GREEN = "\033[92m"  # Assistant tokens (loss_mask = 1)
    RESET = "\033[0m"

    print("\nLoss Mask Visualization:")
    print("RED = Non-assistant tokens (loss_mask = 0)")
    print("GREEN = Assistant tokens (loss_mask = 1)")
    print("-" * 50)

    # Handle both 1D and 2D tensors - flatten if needed
    if input_ids.dim() > 1:
        input_ids = input_ids.flatten()
    if loss_mask.dim() > 1:
        loss_mask = loss_mask.flatten()

    if len(input_ids) == 0 or len(loss_mask) == 0:
        print("Empty input")
        return

    current_mask = loss_mask[0].item()
    current_ids = []

    for i in range(len(input_ids)):
        if current_mask == loss_mask[i].item():
            current_ids.append(input_ids[i].item())
        else:
            if hasattr(tokenizer, "decode"):
                decoded_text = tokenizer.decode(current_ids, skip_special_tokens=False)
            else:
                decoded_text = " ".join([f"token_{id}" for id in current_ids])
            if current_mask == 0:
                print(f"{RED}{decoded_text}{RESET}", end="")
            else:
                print(f"{GREEN}{decoded_text}{RESET}", end="")
            current_ids = [input_ids[i].item()]
            current_mask = loss_mask[i].item()

    # Print remaining tokens
    if current_ids:
        if hasattr(tokenizer, "decode"):
            decoded_text = tokenizer.decode(current_ids, skip_special_tokens=False)
        else:
            decoded_text = " ".join([f"token_{id}" for id in current_ids])
        if current_mask == 0:
            print(f"{RED}{decoded_text}{RESET}")
        else:
            print(f"{GREEN}{decoded_text}{RESET}")
    print("\n" + "-" * 50)


class TestPreprocessing(unittest.TestCase):
    """Test suite for conversation preprocessing and loss mask generation."""

    def setUp(self):
        """Set up test fixtures with Qwen3-8B tokenizer and template."""
        self.model_path = "Qwen/Qwen3-8B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.chat_template = TEMPLATE_REGISTRY.get("qwen")
        self.max_length = 512

    def test_conversation_preprocessing_basic(self):
        """Test basic conversation preprocessing with assistant response identification."""
        conversations = [
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "The answer is 4."},
            ]
        ]

        results = preprocess_conversations(
            tokenizer=self.tokenizer,
            conversations=conversations,
            chat_template=self.chat_template,
            max_length=self.max_length,
            is_preformatted=False,
        )

        # Check structure
        self.assertIn("input_ids", results)
        self.assertIn("loss_mask", results)
        self.assertIn("attention_mask", results)
        self.assertEqual(len(results["input_ids"]), 1)
        self.assertEqual(len(results["loss_mask"]), 1)
        self.assertEqual(len(results["attention_mask"]), 1)

        # Verify tensor shapes match
        input_ids = results["input_ids"][0].squeeze()
        loss_mask = results["loss_mask"][0].squeeze()
        attention_mask = results["attention_mask"][0].squeeze()

        self.assertEqual(input_ids.shape, loss_mask.shape)
        self.assertEqual(input_ids.shape, attention_mask.shape)

        # Check that some tokens are marked for loss (assistant response)
        self.assertTrue(
            torch.any(loss_mask == 1), "No tokens marked for loss computation"
        )

        # Check that some tokens are not marked for loss (system/user parts)
        self.assertTrue(
            torch.any(loss_mask == 0), "All tokens marked for loss computation"
        )

        # Verify the complete assistant response is captured in the loss mask
        assistant_token_indices = torch.where(loss_mask == 1)[0]
        if len(assistant_token_indices) > 0:
            assistant_tokens = input_ids[assistant_token_indices]
            assistant_text = self.tokenizer.decode(
                assistant_tokens, skip_special_tokens=False
            )
            expected_assistant_text = (
                "<think>\n\n</think>\n\nThe answer is 4.<|im_end|>\n"
            )
            self.assertEqual(
                assistant_text,
                expected_assistant_text,
                f"Assistant text does not match exactly. Expected: {repr(expected_assistant_text)}, Got: {repr(assistant_text)}",
            )

    def test_multiple_turns_conversation(self):
        """Test conversation with multiple user-assistant turns."""
        conversations = [
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "The answer is 4."},
                {"role": "user", "content": "Are you sure?"},
                {"role": "assistant", "content": "Yes, I'm certain."},
            ]
        ]

        results = preprocess_conversations(
            tokenizer=self.tokenizer,
            conversations=conversations,
            chat_template=self.chat_template,
            max_length=self.max_length,
            is_preformatted=False,
        )

        input_ids = results["input_ids"][0].squeeze()
        loss_mask = results["loss_mask"][0].squeeze()

        # Get all assistant response tokens
        assistant_token_indices = torch.where(loss_mask == 1)[0]
        self.assertTrue(
            len(assistant_token_indices) > 0, "No assistant tokens identified"
        )

        # Decode assistant tokens to verify both responses are captured
        assistant_tokens = input_ids[assistant_token_indices]
        assistant_text = self.tokenizer.decode(
            assistant_tokens, skip_special_tokens=False
        )

        # Exact match for the complete assistant text from both turns
        expected_assistant_text = "The answer is 4.<|im_end|>\n<think>\n\n</think>\n\nYes, I'm certain.<|im_end|>\n"
        self.assertEqual(
            assistant_text,
            expected_assistant_text,
            f"Assistant text does not match exactly. Expected: {repr(expected_assistant_text)}, Got: {repr(assistant_text)}",
        )

    def test_preformatted_conversation(self):
        """Test preprocessing of pre-formatted conversation strings."""
        preformatted_conversations = [
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is Python?<|im_end|>\n<|im_start|>assistant\nPython is a programming language.<|im_end|>\n"
        ]

        results = preprocess_conversations(
            tokenizer=self.tokenizer,
            conversations=preformatted_conversations,
            chat_template=self.chat_template,
            max_length=self.max_length,
            is_preformatted=True,
        )

        # Check basic structure
        self.assertEqual(len(results["input_ids"]), 1)

        input_ids = results["input_ids"][0].squeeze()
        loss_mask = results["loss_mask"][0].squeeze()

        # Verify assistant response is identified
        self.assertTrue(
            torch.any(loss_mask == 1),
            "No assistant tokens marked in preformatted input",
        )

        # Extract and verify assistant content
        assistant_token_indices = torch.where(loss_mask == 1)[0]
        assistant_tokens = input_ids[assistant_token_indices]
        assistant_text = self.tokenizer.decode(
            assistant_tokens, skip_special_tokens=False
        )

        # Check for exact match of the expected assistant response
        expected_assistant_text = "Python is a programming language.<|im_end|>\n"
        self.assertEqual(
            assistant_text,
            expected_assistant_text,
            f"Assistant text does not match exactly. Expected: {repr(expected_assistant_text)}, Got: {repr(assistant_text)}",
        )

    def test_assistant_span_boundaries(self):
        """Test that assistant span boundaries are correctly identified without truncation."""
        test_cases = [
            {
                "name": "Short response",
                "conversation": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                ],
                "expected_assistant_text": "<think>\n\n</think>\n\nHello!<|im_end|>\n",
            },
            {
                "name": "Response with punctuation",
                "conversation": [
                    {"role": "user", "content": "What's your name?"},
                    {"role": "assistant", "content": "I'm Claude, an AI assistant."},
                ],
                "expected_assistant_text": "<think>\n\n</think>\n\nI'm Claude, an AI assistant.<|im_end|>\n",
            },
            {
                "name": "Multi-sentence response",
                "conversation": [
                    {"role": "user", "content": "Tell me about Python."},
                    {
                        "role": "assistant",
                        "content": "Python is a programming language. It's very popular for AI.",
                    },
                ],
                "expected_assistant_text": "<think>\n\n</think>\n\nPython is a programming language. It's very popular for AI.<|im_end|>\n",
            },
            {
                "name": "Response with special characters",
                "conversation": [
                    {"role": "user", "content": "Show me math."},
                    {
                        "role": "assistant",
                        "content": "Sure! Here's an example: 2 + 2 = 4, and π ≈ 3.14159.",
                    },
                ],
                "expected_assistant_text": "<think>\n\n</think>\n\nSure! Here's an example: 2 + 2 = 4, and π ≈ 3.14159.<|im_end|>\n",
            },
        ]

        for test_case in test_cases:
            with self.subTest(test_case["name"]):
                conversations = [test_case["conversation"]]

                results = preprocess_conversations(
                    tokenizer=self.tokenizer,
                    conversations=conversations,
                    chat_template=self.chat_template,
                    max_length=self.max_length,
                    is_preformatted=False,
                )

                input_ids = results["input_ids"][0].squeeze()
                loss_mask = results["loss_mask"][0].squeeze()

                # Extract assistant tokens
                assistant_token_indices = torch.where(loss_mask == 1)[0]
                self.assertTrue(
                    len(assistant_token_indices) > 0,
                    f"No assistant tokens found for test case: {test_case['name']}",
                )

                assistant_tokens = input_ids[assistant_token_indices]
                assistant_text = self.tokenizer.decode(
                    assistant_tokens, skip_special_tokens=False
                )

                # Verify exact match of the expected assistant text
                expected_assistant_text = test_case["expected_assistant_text"]
                self.assertEqual(
                    assistant_text,
                    expected_assistant_text,
                    f"Assistant text does not match exactly for test case '{test_case['name']}'. Expected: {repr(expected_assistant_text)}, Got: {repr(assistant_text)}",
                )

                # Additional check: ensure no user content leaked into assistant spans
                user_content = test_case["conversation"][0]["content"]
                # Check if user content appears in assistant text (should not happen with exact matching)
                self.assertNotIn(
                    user_content,
                    assistant_text,
                    f"User content '{user_content}' found in assistant spans for test case '{test_case['name']}': '{assistant_text}'",
                )


if __name__ == "__main__":
    suite = unittest.TestSuite()

    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestPreprocessing))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

    # Commented-out example for using visualize_loss_mask function directly
    """
    # Example usage of visualize_loss_mask for debugging/visualization
    model_path = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    chat_template = TEMPLATE_REGISTRY.get("qwen")

    # Using conversations list
    conversations = [
        [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."},
            {"role": "user", "content": "I don't think that's right"},
            {"role": "assistant", "content": "I'm pretty sure it's 4."},
        ],
        [
            {"role": "user", "content": "How do you boil water?"},
            {"role": "assistant", "content": "Use a stove."},
        ],
    ]
    results = preprocess_conversations(
        tokenizer=tokenizer,
        conversations=conversations,
        chat_template=chat_template,
        max_length=512,
        is_preformatted=False,
    )
    for i in range(len(results["input_ids"])):
        visualize_loss_mask(tokenizer, results["input_ids"][i], results["loss_mask"][i])

    # Using preformatted conversation
    preformatted_conversations = [
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\nThe answer is 4.<|im_end|>\n<|im_start|>user\nI don't think that's right<|im_end|>\n<|im_start|>assistant\n<think>\nI know 2+2 is 4</think>\n\nI'm pretty sure it's 4.<|im_end|>\n",
    ]
    results = preprocess_conversations(
        tokenizer=tokenizer,
        conversations=preformatted_conversations,
        chat_template=chat_template,
        max_length=512,
        is_preformatted=True,
    )
    for i in range(len(results["input_ids"])):
        visualize_loss_mask(tokenizer, results["input_ids"][i], results["loss_mask"][i])
    """
