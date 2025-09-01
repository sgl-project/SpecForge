#!/usr/bin/env python3
"""
Simple script to generate responses using local SGLang API from JSONL file.

Data: https://huggingface.co/datasets/mlabonne/open-perfectblend
Environment variables:
     # optional, default: http://localhost:30000
Usage:
    step 1: data splitting
        ```
        #!/bin/bash
        input="your_file.txt"
        lines_per_file=20000
        prefix="shard"
        ext=".json"
        total=$(($(wc -l < "$input" + lines_per_file - 1) / lines_per_file))
        split -l $lines_per_file -d -a 4 "$input" tmp_shard_
        i=0
        for f in tmp_shard_*; do
            shard_num=$((i+1))
            mv "$f" "${prefix}_${shard_num}_of_${total}${ext}"
            i=$((i+1))
        done
        ```
    step 2: python3 -m sglang.launch_server --model-path openai/gpt-oss-20b --tp 8
    step 3: python gen_data.py <shared>
    Example: python gen_data.py 9
"""
import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import requests
from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
    ToolDescription,
    load_harmony_encoding,
)
from tqdm.auto import tqdm

# Configuration
BASE_URL = os.getenv("SGLANG_BASE_URL", "http://localhost:30000/v1/completions")
HEADERS = {"Content-Type": "application/json"}

MODEL = "openai/gpt-oss-20b"
MAX_TOKENS = 2048
BATCH_SIZE = 128
TEMPERATURE = 0.7

# Load harmony encoding once at module level to avoid repeated loading
_harmony_encoding = None


def get_random_reasoning_effort() -> ReasoningEffort:
    """Get a random reasoning effort level for the model with weighted probabilities."""
    # Reasoning effort levels with weights: LOW(7), MEDIUM(2), HIGH(1)
    reasoning_efforts = [
        ReasoningEffort.LOW,
        ReasoningEffort.MEDIUM,
        ReasoningEffort.HIGH,
    ]
    weights = [7, 2, 1]  # 7:2:1 probability ratio
    return random.choices(reasoning_efforts, weights=weights, k=1)[0]


def get_harmony_encoding():
    """Get the harmony encoding, loading it only once."""
    global _harmony_encoding
    if _harmony_encoding is None:
        _harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _harmony_encoding


def build_prompt(user_msg: str, reasoning_effort) -> str:
    """Embed user message into the required prompt template."""
    system_message = (
        SystemContent.new()
        .with_model_identity(
            "You are ChatGPT, a large language model trained by OpenAI."
        )
        .with_reasoning_effort(reasoning_effort)
        .with_conversation_start_date("2025-06-28")
        .with_knowledge_cutoff("2024-06")
        .with_required_channels(["analysis", "commentary", "final"])
    )
    convo = []
    convo.append(Message.from_role_and_content(Role.SYSTEM, system_message))
    convo.append(Message.from_role_and_content(Role.USER, user_msg))
    convo = Conversation.from_messages(convo)
    enc = get_harmony_encoding()  # Use cached encoding
    tokens = enc.render_conversation_for_completion(convo, Role.ASSISTANT)
    prompt_text = enc.decode_utf8(tokens)
    return prompt_text


def build_prompt_batch_parallel(
    batch_data: List[tuple], max_workers: int = 8
) -> List[tuple]:
    """
    Build prompts in parallel for a batch of data.

    Args:
        batch_data: List of (item, human_msg) tuples
        max_workers: Maximum number of worker threads

    Returns:
        List of (item, human_msg, reasoning_effort, prompt) tuples for successful builds
    """

    def build_single_prompt(item_data):
        item, human_msg = item_data
        try:
            reasoning_effort = get_random_reasoning_effort()
            prompt = build_prompt(human_msg, reasoning_effort)
            return (item, human_msg, reasoning_effort, prompt, None)
        except Exception as e:
            return (item, human_msg, None, None, str(e))

    results = []

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_data = {
            executor.submit(build_single_prompt, item_data): item_data
            for item_data in batch_data
        }

        # Collect results as they complete
        for future in as_completed(future_to_data):
            item, human_msg, reasoning_effort, prompt, error = future.result()
            if error:
                print(f"Error building prompt: {error}")
            else:
                results.append((item, human_msg, reasoning_effort, prompt))

    return results


def call_sglang_batch(prompts: List[str]) -> List[str]:
    """Send a batch of prompts to sglang /v1/completions."""
    payload = {
        "model": MODEL,
        "prompt": prompts,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "skip_special_tokens": False,
    }

    resp = requests.post(BASE_URL, headers=HEADERS, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    return [choice["text"].strip() for choice in data["choices"]]


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_human_message(item: Dict[str, Any]) -> str:
    """Extract human message from data item."""
    # Try common formats
    if "conversations" in item:
        conv = item["conversations"]
        if isinstance(conv, list) and len(conv) > 0:
            return conv[0].get("value", conv[0].get("content", ""))

    # Try other common fields
    for field in ["message", "instruction", "question", "input", "text"]:
        if field in item:
            return item[field]

    return str(item)


def parse_channel_output(output: str) -> Dict[str, Optional[str]]:
    """Parse the channel-based output format into analysis and final parts."""
    result = {"analysis": None, "final": None}

    # Find analysis channel
    analysis_start = output.find("<|channel|>analysis<|message|>")
    if analysis_start != -1:
        analysis_start += len("<|channel|>analysis<|message|>")
        analysis_end = output.find("<|end|>", analysis_start)
        if analysis_end != -1:
            result["analysis"] = output[analysis_start:analysis_end].strip()

    # Find final channel
    final_start = output.find("<|channel|>final<|message|>")
    if final_start != -1:
        final_start += len("<|channel|>final<|message|>")
        # Final content goes to the end of the string
        result["final"] = output[final_start:].strip()

    return result


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate GPT-OSS data from JSONL files"
    )
    parser.add_argument("shared", type=int, help="Starting shard number")
    parser.add_argument(
        "--input-dir", default="/data/", help="Input directory path (default: /data/)"
    )
    parser.add_argument(
        "--output-dir", default="/data/", help="Output directory path (default: /data/)"
    )
    parser.add_argument(
        "--shard-step",
        type=int,
        default=5,
        help="Process every Nth shard; step size (default: 5)",
    )

    args = parser.parse_args()

    start_shared = args.shared
    max_shared = 72  # Based on the filename pattern shard_X_of_72
    shard_step = max(1, args.shard_step)

    for shared in range(start_shared, max_shared + 1, shard_step):
        input_file = os.path.join(args.input_dir, f"shard_{shared}_of_72.json")
        output_file = os.path.join(args.output_dir, f"shard_{shared}_of_72.json")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Input file not found: {input_file}")
            print(f"Stopping at shard {shared}")
            break
        try:
            data = load_jsonl(input_file)
            print(f"Loaded {len(data)} items")

            if not data:
                print("No data found in input file, skipping.")
                continue

            # Process data in batches
            total_saved = 0

            # Prepare all valid data first
            valid_items = []
            for item in data:
                human_msg = extract_human_message(item)
                if human_msg.strip():
                    valid_items.append((item, human_msg))

            # Open output file once and write each batch result immediately
            with open(output_file, "w", encoding="utf-8") as f:
                # Process in batches
                for i in tqdm(
                    range(0, len(valid_items), BATCH_SIZE),
                    desc=f"Processing shard {shared}",
                ):
                    batch = valid_items[i : i + BATCH_SIZE]

                    # Build prompts in parallel for the entire batch
                    try:
                        batch_results = build_prompt_batch_parallel(
                            batch, max_workers=8
                        )

                        if not batch_results:
                            continue

                        batch_prompts = []
                        batch_items = []

                        for item, human_msg, reasoning_effort, prompt in batch_results:
                            batch_prompts.append(prompt)
                            batch_items.append((item, human_msg, reasoning_effort))

                    except Exception as e:
                        print(f"Error in parallel prompt building: {e}")
                        continue

                    if not batch_prompts:
                        continue

                    try:
                        # Process entire batch at once
                        outputs = call_sglang_batch(batch_prompts)

                        # Process each response in the batch
                        for j, output in enumerate(outputs):
                            if (
                                j < len(batch_items) and output
                            ):  # Check bounds and valid response
                                item, human_msg, reasoning_effort = batch_items[j]

                                # Parse the channel-based output
                                parsed_output = parse_channel_output(output)

                                row = {
                                    "conversations": [
                                        {"role": "human", "content": human_msg},
                                        {"role": "assistant", "content": output},
                                        {
                                            "role": "assistant_analysis",
                                            "content": parsed_output["analysis"],
                                        },
                                        {
                                            "role": "assistant_final",
                                            "content": parsed_output["final"],
                                        },
                                        {
                                            "role": "assistant_reasoning_effort",
                                            "content": reasoning_effort.value,
                                        },
                                    ],
                                }
                                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                                total_saved += 1
                            else:
                                print(f"Warning: Empty response for batch item {j}")

                        f.flush()  # Ensure data is written to disk after each batch

                    except Exception as e:
                        print(f"Error processing batch starting at index {i}: {e}")
                        continue
            # Show results for this shard
            if total_saved > 0:
                print(f"✅ Saved {total_saved} responses to {output_file}")
                print(
                    f"Success rate: {total_saved}/{len(data)} ({total_saved/len(data)*100:.1f}%)"
                )
            else:
                print("No responses were generated for this shard.")
        except Exception as e:
            print(f"Error processing shard {shared}: {e}")
            print("Continuing to next shard...")
            continue

    print(f"\n{'='*60}")
    print(
        f"Completed processing shards starting from {start_shared} (every {shard_step}th shard)"
    )
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
