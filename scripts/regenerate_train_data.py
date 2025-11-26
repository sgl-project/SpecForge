"""
This script will re-generate the dataset from target model,
which better aligns the draft model with the target model’s output distribution.

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
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from openai import OpenAI
from tqdm import tqdm


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Re-generate training data using sglang model server"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens (default: 4096)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sglang model server",
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


def call_sglang(
    model: str,
    max_tokens: int,
    temperature: float,
    server_address: str,
    data: List[Dict[str, Any]],
    is_reasoning_model: bool,
    is_gpt_oss: bool,
) -> str:
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

            query_kwargs = dict(
                model=model,
                messages=regenerated_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            )
            if is_gpt_oss:
                query_kwargs["reasoning_effort"] = get_random_reasoning_effort()

            try:
                resp = client.chat.completions.create(**query_kwargs)
            except Exception as e:
                data["status"] = "error"
                data["error"] = str(e)
                return data
            response_text = resp.choices[0].message.content
            resp_msg = {
                "role": "assistant",
                "content": response_text,
            }
            if is_reasoning_model:
                resp_msg["thinking"] = resp.choices[0].message.reasoning_content
            regenerated_messages.append(resp_msg)
        else:
            data["status"] = "error"
            data["error"] = f"Invalid message role: {message['role']}"
            return data
    data["conversations"] = regenerated_messages
    data["status"] = "success"
    return data


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

    # test all server addresses
    valid_server_addresses = []
    for server_address in args.server_address:
        dummy_data = dict(
            conversations=[{"role": "user", "content": "Hello, how are you?"}]
        )
        result = call_sglang(
            args.model,
            1,
            args.temperature,
            server_address,
            dummy_data,
            args.is_reasoning_model,
            args.is_gpt_oss,
        )
        if result is not None:
            valid_server_addresses.append(server_address)
        else:
            print(f"Server {server_address} is not available")

    if len(valid_server_addresses) == 0:
        raise ValueError("No server address is available")
    print(
        f"Using {len(valid_server_addresses)} server addresses: {valid_server_addresses}"
    )
    print("-" * 50)

    # create error file path if not exists
    error_file_path = args.output_file_path.replace(".jsonl", "_error.jsonl")
    print(
        f"Regenerating dataset and saving the output to {args.output_file_path} and error log to {error_file_path}"
    )
    print("-" * 50)

    success_samples = 0
    error_samples = 0

    # Create progress bar
    with open(args.input_file_path, "r") as input_file, open(
        args.output_file_path, "w"
    ) as output_file_handle, open(error_file_path, "w") as error_file_handle:
        executor = ThreadPoolExecutor(
            max_workers=args.concurrency * len(valid_server_addresses)
        )
        waiting_queue = {
            server_address: [] for server_address in valid_server_addresses
        }
        pbar = tqdm(total=total_lines, desc="Processing")
        start_server_index = 0

        for line in input_file:
            if (
                args.num_samples is not None
                and success_samples + error_samples >= args.num_samples
            ):
                break

            data = json.loads(line.strip())

            # find server address with the least waiting requests
            server_address = valid_server_addresses[start_server_index]
            start_server_index = (start_server_index + 1) % len(valid_server_addresses)

            # submit prompt to sglang
            while len(waiting_queue[server_address]) >= args.concurrency:
                finished_on_request = False
                # check if any future is done, if so, write the result to the output file
                for req_future in waiting_queue[server_address]:
                    if req_future.done():
                        regen_data = req_future.result()

                        if regen_data["status"] == "error":
                            error_file_handle.write(
                                json.dumps(regen_data, ensure_ascii=False) + "\n"
                            )
                            error_samples += 1
                        else:
                            output_file_handle.write(
                                json.dumps(regen_data, ensure_ascii=False) + "\n"
                            )
                            success_samples += 1
                        waiting_queue[server_address].remove(req_future)
                        finished_on_request = True

                if finished_on_request:
                    break

            req_future = executor.submit(
                call_sglang,
                args.model,
                args.max_tokens,
                args.temperature,
                server_address,
                data,
                args.is_reasoning_model,
                args.is_gpt_oss,
            )
            waiting_queue[server_address].append(req_future)
            pbar.update(1)

        # deal with all the remaining requests
        for server_address, waiting_queue_items in waiting_queue.items():
            for req_future in waiting_queue_items:
                regen_data = req_future.result()
                if regen_data["status"] == "error":
                    error_file_handle.write(
                        json.dumps(regen_data, ensure_ascii=False) + "\n"
                    )
                    error_samples += 1
                else:
                    output_file_handle.write(
                        json.dumps(regen_data, ensure_ascii=False) + "\n"
                    )
                    success_samples += 1

    print(
        f"\nProcessing completed! {success_samples} samples regenerated, {error_samples} samples failed."
    )


if __name__ == "__main__":
    main()
