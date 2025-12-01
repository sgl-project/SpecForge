"""
This script will re-generate the dataset from target model,
which better aligns the draft model with the target model's output distribution.

Usage:
1. Set up one or more SGLang servers for the target model.

python3 -m sglang.launch_server \
	--model meta-llama/Llama-3.1-8B-Instruct \
	--mem-fraction-static 0.75 \
	--cuda-graph-max-bs 128 \
	--tp 1 \
	--trust-remote-code \
	--host 0.0.0.0 \
	--port 30000 \
	--dtype bfloat16


2. Regenerate the dataset using the `regenerate_train_data.py` script.
python scripts/regenerate_train_data.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --concurrency 128 \
    --max-tokens 4096 \
    --server-address localhost:30000 \
    --temperature 0.8 \
    --input-file-path ./cache/dataset/sharegpt_train.jsonl \
    --output-file-path ./cache/dataset/sharegpt_train_regen.jsonl
"""

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from openai import OpenAI
from tqdm import tqdm

CONTEXT_TOKEN_SUM = 0
CONTEXT_TOKEN_MIN = None
CONTEXT_TOKEN_MAX = 0
CONNECTION_ERROR_KEYWORDS = [
    "ConnectionError",
    "Timeout",
    "timed out",
    "ECONNREFUSED",
    "Connection refused",
    "RemoteDisconnected",
    "SSLError",
    "ReadTimeout",
]


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Re-generate training data using sglang model server"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sglang model server",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling top_p",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling value sent via extra_body",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Mapped to presence_penalty in the OpenAI API",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens (default: 4096)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="The number of requests to send to a single server concurrently, the total number of concurrent requests is concurrency * number of server addresses",
    )
    parser.add_argument(
        "--input-file-path", type=str, required=True, help="Path to the input file"
    )
    parser.add_argument(
        "--output-file-path", type=str, required=True, help="Path to the output file"
    )
    parser.add_argument(
        "--server-address",
        type=str,
        nargs="+",
        help="Server address and port for sglang model server",
    )
    parser.add_argument(
        "--is-reasoning-model",
        action="store_true",
        help="Whether the model is a reasoning model",
    )
    parser.add_argument(
        "--is-gpt-oss",
        action="store_true",
        help="Whether the model is a GPT-OSS model",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="The number of samples to regenerate, if not provided, all samples will be regenerated",
    )
    parser.add_argument(
        "--sample-max-retries",
        type=int,
        default=5,
        help="The maximum number of retries for a sample",
    )
    return parser.parse_args()


def get_random_reasoning_effort() -> str:
    """Get a random reasoning effort level for the model with weighted probabilities."""
    # usage example: https://huggingface.co/openai/gpt-oss-20b/discussions/28
    # Reasoning effort levels with weights: LOW(4), MEDIUM(4), HIGH(2)
    reasoning_efforts = [
        "low",
        "medium",
        "high",
    ]
    weights = [4, 4, 2]
    return random.choices(reasoning_efforts, weights=weights, k=1)[0]


def build_query_kwargs(args, messages, max_tokens=None):
    effective_max_tokens = max_tokens if max_tokens is not None else args.max_tokens

    query_kwargs = dict(
        model=args.model,
        messages=messages,
        max_tokens=effective_max_tokens,
        temperature=args.temperature,
        stream=False,
    )
    if args.top_p is not None:
        query_kwargs["top_p"] = args.top_p
    if args.repetition_penalty is not None:
        query_kwargs["presence_penalty"] = args.repetition_penalty
    extra_body = {}
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k
    if extra_body:
        query_kwargs["extra_body"] = extra_body
    if args.is_gpt_oss:
        query_kwargs["reasoning_effort"] = get_random_reasoning_effort()
    return query_kwargs


def calculate_metrics(context_length: int):
    global CONTEXT_TOKEN_SUM, CONTEXT_TOKEN_MIN, CONTEXT_TOKEN_MAX
    CONTEXT_TOKEN_SUM += context_length
    if CONTEXT_TOKEN_MIN is None:
        CONTEXT_TOKEN_MIN = context_length
    else:
        CONTEXT_TOKEN_MIN = min(CONTEXT_TOKEN_MIN, context_length)
    CONTEXT_TOKEN_MAX = max(CONTEXT_TOKEN_MAX, context_length)


def is_connection_error(error_msg: str) -> bool:
    if not error_msg:
        return False
    return any(keyword in error_msg for keyword in CONNECTION_ERROR_KEYWORDS)


def call_sglang(
    args,
    server_address: str,
    data: Dict[str, Any],
    max_tokens=None,
) -> Dict[str, Any]:
    """Send a batch of prompts to sglang /v1/completions."""
    client = OpenAI(base_url=f"http://{server_address}/v1", api_key="None")

    messages = data["conversations"]
    regenerated_messages = []

    # ignore data which starts with an assistant message
    if messages[0]["role"] == "assistant":
        data["status"] = "error"
        data["error"] = "Data starts with an assistant message"
        return data

    for message in messages:
        if message["role"] == "system":
            regenerated_messages.append(message)
        elif message["role"] == "assistant":
            continue
        elif message["role"] == "user":
            regenerated_messages.append(message)

            query_kwargs = build_query_kwargs(args, regenerated_messages, max_tokens)

            try:
                resp = client.chat.completions.create(**query_kwargs)
            except Exception as e:
                data["status"] = "error"
                data["error"] = str(e)
                return data

            data["output_tokens"] = resp.usage.completion_tokens
            data["input_tokens"] = resp.usage.prompt_tokens
            data["context_length"] = data["input_tokens"] + data["output_tokens"]
            response_text = resp.choices[0].message.content
            resp_msg = {
                "role": "assistant",
                "content": response_text,
            }
            if args.is_reasoning_model:
                resp_msg["thinking"] = resp.choices[0].message.reasoning_content
            regenerated_messages.append(resp_msg)
        else:
            data["status"] = "error"
            data["error"] = f"Invalid message role: {message['role']}"
            return data
    data["conversations"] = regenerated_messages
    data["status"] = "success"
    return data


def health_check_server(args, server_address: str) -> bool:
    dummy_data = {
        "conversations": [{"role": "user", "content": "Hello, how are you?"}],
    }
    try:
        result = call_sglang(args, server_address, dummy_data, max_tokens=1)
    except Exception:
        return False
    if result is None:
        return False
    status = result.get("status")
    if status == "error":
        error_msg = str(result.get("error", ""))
        return not is_connection_error(error_msg)
    return True


def wait_for_healthy_servers(args) -> List[str]:
    while True:
        valid_server_addresses = []
        for server_address in args.server_address:
            if health_check_server(args, server_address):
                valid_server_addresses.append(server_address)
            else:
                print(f"Server {server_address} is not available")

        if valid_server_addresses:
            print(
                f"Using {len(valid_server_addresses)} server addresses: {valid_server_addresses}"
            )
            print("-" * 50)
            return valid_server_addresses

        print("No valid server available, waiting for servers to become healthy...")
        time.sleep(5)


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Validate parameters
    if not (0.0 <= args.temperature <= 1.0):
        raise ValueError("Temperature must be between 0.0 and 1.0")

    if args.max_tokens <= 0:
        raise ValueError("Max tokens must be greater than 0")

    print(f"Configuration:")
    print(f"  Model path: {args.model}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Temperature: {args.temperature}")
    print(f"  API URL: {args.server_address}")
    print(f"  Input file: {args.input_file_path}")
    print(f"  Output file: {args.output_file_path}")
    print("-" * 50)
    total_lines = sum(1 for _ in open(args.input_file_path))

    valid_server_addresses = wait_for_healthy_servers(args)

    # create error file path
    error_file_path = args.output_file_path.replace(".jsonl", "_error.jsonl")
    print(
        f"Regenerating dataset and saving the output to {args.output_file_path} and error log to {error_file_path}"
    )
    print("-" * 50)
    processed_ids = set()
    success_samples = 0
    error_samples = 0
    sample_max_retries = args.sample_max_retries
    retry_counts = {}

    # Load existing outputs for checkpointing
    output_file_exists = os.path.exists(args.output_file_path)
    error_file_exists = os.path.exists(error_file_path)

    if output_file_exists:
        print(f"Found existing output file at {args.output_file_path}, resuming.")
        with open(args.output_file_path, "r") as existing_output_file:
            for line in existing_output_file:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                data_id = record.get("data_id")
                if isinstance(data_id, int):
                    processed_ids.add(data_id)
                conversations = record.get("conversations")
                if conversations is not None:
                    calculate_metrics(record.get("context_length"))
                    success_samples += 1

    if processed_ids:
        print(
            f"Detected {len(processed_ids)} existing successful samples in output file. "
            "Previously processed data_ids will be skipped."
        )

    if args.num_samples is not None and success_samples >= args.num_samples:
        print(
            f"num_samples={args.num_samples} already satisfied by existing outputs "
            f"({success_samples} successes). Nothing to do."
        )
        return

    output_open_mode = "a" if output_file_exists else "w"

    def run_pass(
        data_batch: List[Dict[str, Any]],
        valid_server_addresses: List[str],
        output_file_handle,
        error_file_handle,
    ) -> List[Dict[str, Any]]:
        nonlocal success_samples, error_samples, retry_counts

        waiting_queue: Dict[str, List] = {addr: [] for addr in valid_server_addresses}
        pbar = tqdm(total=len(data_batch), desc="Processing")
        start_server_index = 0
        retry_records: List[Dict[str, Any]] = []

        def ensure_servers_available(
            valid_addrs: List[str], waiting: Dict[str, List]
        ) -> List[str]:
            if valid_addrs:
                return valid_addrs
            print("No valid server available, waiting for servers to become healthy...")
            new_addrs = wait_for_healthy_servers(args)
            waiting.clear()
            for addr in new_addrs:
                waiting[addr] = []
            return new_addrs

        def handle_finished_future(req_future):
            nonlocal valid_server_addresses, success_samples, error_samples, retry_counts
            regen_data, server_addr = req_future.result()

            if regen_data.get("status") == "error":
                if is_connection_error(str(regen_data.get("error", ""))):
                    if server_addr in valid_server_addresses:
                        print(
                            f"Removing unhealthy server {server_addr} from valid list"
                        )
                        valid_server_addresses.remove(server_addr)

                data_id = regen_data.get("data_id")
                if isinstance(data_id, int):
                    retry_counts[data_id] = retry_counts.get(data_id, 0) + 1
                    if retry_counts[data_id] >= sample_max_retries:
                        error_file_handle.write(
                            json.dumps(regen_data, ensure_ascii=False) + "\n"
                        )
                        error_samples += 1
                    else:
                        retry_records.append(regen_data)
                else:
                    error_file_handle.write(
                        json.dumps(regen_data, ensure_ascii=False) + "\n"
                    )
                    error_samples += 1
            else:
                calculate_metrics(regen_data.get("context_length"))
                output_file_handle.write(
                    json.dumps(regen_data, ensure_ascii=False) + "\n"
                )
                success_samples += 1

            if server_addr in waiting_queue:
                waiting_queue[server_addr].remove(req_future)

        with ThreadPoolExecutor(
            max_workers=args.concurrency * max(1, len(valid_server_addresses))
        ) as executor:
            for data in data_batch:
                valid_server_addresses = ensure_servers_available(
                    valid_server_addresses, waiting_queue
                )
                if not valid_server_addresses:
                    continue

                server_address = valid_server_addresses[start_server_index]
                start_server_index = (start_server_index + 1) % len(
                    valid_server_addresses
                )

                while len(waiting_queue[server_address]) >= args.concurrency:
                    finished_on_request = False
                    for req_future in list(waiting_queue[server_address]):
                        if req_future.done():
                            handle_finished_future(req_future)
                            finished_on_request = True
                    if finished_on_request:
                        break

                future = executor.submit(
                    lambda addr, payload: (call_sglang(args, addr, payload), addr),
                    server_address,
                    data,
                )
                if server_address not in waiting_queue:
                    waiting_queue[server_address] = []
                waiting_queue[server_address].append(future)
                pbar.update(1)

            for server_address, waiting_queue_items in waiting_queue.items():
                for req_future in list(waiting_queue_items):
                    if req_future.done():
                        handle_finished_future(req_future)
                    else:
                        handle_finished_future(req_future)

        return retry_records

    remaining_data: List[Dict[str, Any]] = []
    with open(args.input_file_path, "r") as input_file:
        data_id = 0
        for line in input_file:
            if (
                args.num_samples is not None
                and success_samples + error_samples >= args.num_samples
            ):
                break
            if data_id in processed_ids:
                data_id += 1
                continue
            data = json.loads(line.strip())
            data["data_id"] = data_id
            remaining_data.append(data)
            data_id += 1

    with open(error_file_path, "w") as error_file_handle:
        while remaining_data:
            with open(args.output_file_path, output_open_mode) as output_file_handle:
                retry_records = run_pass(
                    remaining_data,
                    valid_server_addresses,
                    output_file_handle,
                    error_file_handle,
                )

            if not retry_records:
                break

            remaining_data = []
            for record in retry_records:
                record.pop("status", None)
                record.pop("error", None)
                remaining_data.append(record)
            output_open_mode = "a"

    print(f"\nProcessing completed!")
    if success_samples > 0:
        avg_len = CONTEXT_TOKEN_SUM / success_samples
        print("Context length statistics (token count over conversations):")
        print(f"Number of successful examples: {success_samples}")
        print(f"Shortest context length: {CONTEXT_TOKEN_MIN}")
        print(f"Longest context length: {CONTEXT_TOKEN_MAX}")
        print(f"Average context length: {avg_len:.2f}")
    else:
        print("No successful examples to compute context length statistics.")

    print(
        f"\nProcessing completed! {success_samples} samples regenerated, {error_samples} samples failed."
    )


if __name__ == "__main__":
    main()
