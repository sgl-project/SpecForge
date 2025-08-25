import os
import unittest
from unittest.mock import Mock, patch

import pytest
import torch
from transformers import AutoTokenizer

from specforge.data.preprocessing import preprocess_conversations
from specforge.data.template import TEMPLATE_REGISTRY, ChatTemplate


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


if __name__ == "__main__":
    # Visualize loss mask for conversations
    model_path = "Qwen/Qwen3-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    chat_template = TEMPLATE_REGISTRY.get("qwen")

    ## Using conversations list
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

    ## Using preformatted conversation
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
