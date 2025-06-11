### Filter data
"""
sharegpt style is like:
[
    {
        "conversations": [
            {
                "from": "human",
                "value": "What time is it in London?",
            },
            {
                "from": "gpt",
                "value": "It is 10:00 AM in London.",
            },
        ],
    },
    {
        "conversations": [
            ...
        ],
    },
    ...,
]
OpenAI style is like:
{
    "messages": [
        {
            "role": <system|user|assistant>,
            "content": [
                {
                    "type": "text",
                    "text": "What'\''s in this image?",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": <url>,
                    },
                },
        },
        ...
    ]
}
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import concatenate_datasets, load_dataset

from sglang.api import set_default_backend
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import dump_state_text


def load_and_prepare_data(
    dataset_name: str,
    dataset_path: Optional[str],
    data_files: Optional[str],
    dataset_splits: List[str],
    start: int,
    end: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Load and prepare dataset from various sources"""
    print("Loading dataset...")

    if data_files:
        print(f"Loading from JSON files: {data_files}")
        dataset = load_dataset("json", data_files=data_files)
        combined_data = dataset["train"]
    elif dataset_path:
        print(f"Loading from local path: {dataset_path}")
        dataset = load_dataset(dataset_path)
        datasets_to_combine = [
            dataset[split] for split in dataset_splits if split in dataset
        ]
        combined_data = (
            concatenate_datasets(datasets_to_combine)
            if len(datasets_to_combine) > 1
            else datasets_to_combine[0]
        )
    else:
        print(f"Loading from HuggingFace Hub: {dataset_name}")
        dataset = load_dataset(dataset_name)
        datasets_to_combine = [
            dataset[split] for split in dataset_splits if split in dataset
        ]
        combined_data = (
            concatenate_datasets(datasets_to_combine)
            if len(datasets_to_combine) > 1
            else datasets_to_combine[0]
        )

    # Shuffle and select data
    combined_data = combined_data.shuffle(seed=seed)

    total_length = len(combined_data)
    if end == -1:
        actual_end = total_length
        print(
            f"Processing all data from index {start} to end (total: {total_length} samples)"
        )
    else:
        actual_end = min(end, total_length)
        print(
            f"Processing data from index {start} to {actual_end} (total: {total_length} samples)"
        )

    if start >= total_length:
        raise ValueError(f"Start index {start} is >= dataset length {total_length}")
    if start >= actual_end:
        raise ValueError(f"Start index {start} is >= end index {actual_end}")

    selected_data = combined_data.select(range(start, actual_end))
    print(f"Selected {len(selected_data)} samples")

    return list(selected_data)


### MODIFIED ###
# This function is updated to handle both old and new OpenAI-style data formats.
def extract_conversations(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract conversation content from messages, supporting both string and list content formats.
    It also extracts the system prompt if present.
    """
    messages = data.get("messages", data.get("conversations", []))

    if not messages:
        if "prompt" in data and "response" in data:
            return {
                "system_prompt": None,
                "queries": [data["prompt"]],
                "responses": [data["response"]],
            }
        else:
            return {"system_prompt": None, "queries": [], "responses": []}

    system_prompt = None
    queries = []
    responses = []

    # In some datasets, the first message is a user prompt, but in others it might be a system prompt.
    # We will now explicitly look for system prompts.

    for message in messages:
        role = message.get("role", message.get("from", "")).lower()
        raw_content = message.get("content", message.get("value", ""))

        # ### NEW ###
        # Process content that might be a list (OpenAI format) or a simple string.
        content = ""
        if isinstance(raw_content, list):
            # It's the new format with a list of content blocks.
            # We only care about text blocks.
            text_parts = []
            for item in raw_content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            content = "\n".join(text_parts)
        elif isinstance(raw_content, str):
            # It's the old format with a string content.
            content = raw_content

        if not content.strip():
            continue

        if role == "system":
            # ### NEW ### Handle system prompt
            system_prompt = content
        elif role in ["user", "human"]:
            queries.append(content)
        elif role in ["assistant", "gpt", "bot"]:
            responses.append(content)

    # Ensure user and assistant turns are balanced, assuming conversation starts with a user.
    # This logic might need adjustment based on specific dataset assumptions.
    # For now, we assume a user query is always followed by an assistant response.
    if len(queries) != len(responses):
        # This can happen if the conversation ends on a user query.
        # We will keep all queries and available responses.
        pass

    return {"system_prompt": system_prompt, "queries": queries, "responses": responses}


### MODIFIED ###
# This function is updated to handle the system prompt.
def process_conversations(raw_data: List[Dict]) -> tuple[List[Dict], List[Dict]]:
    """Process conversations and prepare arguments for generation"""
    print("Processing conversations...")

    processed_conversations = []
    arguments = []

    for i, data in enumerate(raw_data):
        conv = extract_conversations(data)

        if not conv["queries"]:
            continue

        processed_conversations.append(
            {
                "id": i,
                "queries": conv["queries"],
                "responses": conv["responses"],
                "conversations": [],
                "system_prompt": conv[
                    "system_prompt"
                ],  # ### NEW ### Store system prompt
            }
        )

        # Prepare arguments for each query in the conversation
        conversation_history = []
        # ### NEW ### Add system prompt to the beginning of the history if it exists
        if conv["system_prompt"]:
            conversation_history.append(
                {"role": "system", "content": conv["system_prompt"]}
            )

        for j, query in enumerate(conv["queries"]):
            # Add user message to history
            conversation_history.append({"role": "user", "content": query})

            arguments.append(
                {
                    "conversation_id": len(processed_conversations) - 1,
                    "query_round": j,
                    "conversation_history": conversation_history.copy(),
                    "original_response": conv["responses"][j]
                    if j < len(conv["responses"])
                    else "",
                    "current_query": query,
                }
            )

            # Add assistant response to history for next round (if available)
            if j < len(conv["responses"]):
                conversation_history.append(
                    {"role": "assistant", "content": conv["responses"][j]}
                )

    print(f"Processed {len(processed_conversations)} conversations")
    print(f"Generated {len(arguments)} requests")

    return processed_conversations, arguments


def main(args):
    # Select backend
    set_default_backend(select_sglang_backend(args))

    # Load and process data
    raw_data = load_and_prepare_data(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        data_files=args.data_files,
        dataset_splits=args.dataset_split,
        start=args.start,
        end=args.end,
        seed=args.seed,
    )

    processed_conversations, arguments = process_conversations(raw_data)

    if not arguments:
        print("No data to process!")
        return

    #####################################
    ######### SGL Program Begin #########
    #####################################

    import sglang as sgl

    if args.backend.startswith("gpt-"):

        @sgl.function
        def conversation_generation(s, conversation_history, **kwargs):
            # ### MODIFIED ### Handle the 'system' role
            for msg in conversation_history:
                if msg["role"] == "system":
                    s += sgl.system(msg["content"])
                elif msg["role"] == "user":
                    s += sgl.user(msg["content"])
                elif msg["role"] == "assistant":
                    s += sgl.assistant(msg["content"])
            s += sgl.assistant(sgl.gen("response", max_tokens=args.max_tokens))

    else:

        @sgl.function
        def conversation_generation(s, conversation_history, **kwargs):
            # ### MODIFIED ### Handle the 'system' role for generic models
            for msg in conversation_history:
                if msg["role"] == "system":
                    s += f"System: {msg['content']}\n\n"
                elif msg["role"] == "user":
                    s += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    s += f"Assistant: {msg['content']}\n"
            s += "Assistant: " + sgl.gen("response", max_tokens=args.max_tokens)

    #####################################
    ########## SGL Program End ##########
    #####################################

    # Run requests
    print("Starting batch generation...")
    tic = time.perf_counter()
    states = conversation_generation.run_batch(
        arguments,
        temperature=args.temperature,
        top_p=args.top_p,
        num_threads=args.parallel,
        progress_bar=True,
    )
    latency = time.perf_counter() - tic

    # Process results
    print("Processing generation results...")
    for i, (state, arg) in enumerate(zip(states, arguments)):
        conv_id = arg["conversation_id"]
        generated_response = state["response"].strip()

        # Add to conversation record
        processed_conversations[conv_id]["conversations"].extend(
            [
                {"from": "human", "value": arg["current_query"]},
                {"from": "dataset", "value": arg["original_response"]},
                {"from": "gpt", "value": generated_response},
            ]
        )

    # Calculate statistics
    try:
        num_output_tokens = sum(
            s.get_meta_info("response")["completion_tokens"] for s in states
        )
        output_throughput = num_output_tokens / latency
    except Exception:
        # Fallback if meta info is not available
        num_output_tokens = 0
        output_throughput = 0

    # Print results
    print(f"Latency: {latency:.3f} s")
    print(f"Output throughput: {output_throughput:.3f} token/s")
    print(
        f"Generated {len(arguments)} responses for {len(processed_conversations)} conversations"
    )

    # Save results
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.end == -1:
        filename = f"{timestamp}_{args.index}_data_start{args.start}_all.json"
    else:
        filename = f"{timestamp}_{args.index}_data_start{args.start}_end{args.end}.json"

    output_path = Path(args.outdir) / filename

    # ### MODIFIED ### Use an outer dictionary to store results for clarity
    output_data = {
        "metadata": {
            "source_dataset": args.dataset_name or args.dataset_path or args.data_files,
            "generation_backend": args.backend,
            "timestamp": timestamp,
        },
        "results": processed_conversations,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {output_path}")

    # Dump state text for debugging
    debug_file = Path(args.outdir) / f"debug_{args.backend}_{args.index}.txt"
    dump_state_text(str(debug_file), states)
    print(f"Debug output saved to {debug_file}")

    # Write summary results
    result_file = Path(args.outdir) / "results.jsonl"
    with open(result_file, "a") as fout:
        value = {
            "task": "data_cleaning",
            "backend": args.backend,
            "latency": round(latency, 3),
            "output_throughput": round(output_throughput, 3),
            "num_requests": len(arguments),
            "num_conversations": len(processed_conversations),
            "timestamp": timestamp,
            "other": {
                "start": args.start,
                "end": args.end,
                "parallel": args.parallel,
                "dataset_name": args.dataset_name,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
            },
        }
        fout.write(json.dumps(value) + "\n")

    print(f"Summary saved to {result_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dataset processing and generation with SGLang"
    )

    # Data processing arguments
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument(
        "--end", type=int, default=-1, help="End index (-1 means process all data)"
    )
    parser.add_argument("--index", type=int, default=1, help="Job index")
    parser.add_argument("--outdir", type=str, default="debug", help="Output directory")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="HuggingFaceH4/ultrachat_200k",
        help="Dataset name or path",
    )
    parser.add_argument(
        "--dataset-path", type=str, default=None, help="Local dataset path"
    )
    parser.add_argument(
        "--data-files", type=str, default=None, help="Data files path for JSON datasets"
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        nargs="+",
        default=["train_sft"],
        help="Dataset splits to use",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling"
    )
    parser.add_argument(
        "--max-query-len", type=int, default=4000, help="Max query length"
    )

    # Generation parameters
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument(
        "--max-tokens", type=int, default=2048, help="Max generation tokens"
    )

    args = add_common_sglang_args_and_parse(parser)
    main(args)
