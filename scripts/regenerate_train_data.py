"""
This script will re-generate the dataset from target model,
which better aligns the draft model with the target model’s output distribution.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import requests
from tqdm import tqdm
from transformers import AutoTokenizer


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
    return parser.parse_args()


def call_sglang(
    model: str, max_tokens: int, temperature: float, server_address: str, prompt: str
) -> str:
    """Send a batch of prompts to sglang /v1/completions."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "skip_special_tokens": False,
    }
    headers = {"Content-Type": "application/json"}
    url = f"http://{server_address}/v1/completions"
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["text"].strip()
    except:
        return None


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

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    total_lines = sum(1 for _ in open(args.input_file_path))

    # test all server addresses
    valid_server_addresses = []
    for server_address in args.server_address:
        result = call_sglang(
            args.model, 1, args.temperature, server_address, "Hello, how are you?"
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

    # Create progress bar
    with (
        open(args.input_file_path, "r") as input_file,
        open(args.output_file_path, "w") as output_file_handle,
    ):

        executor = ThreadPoolExecutor(
            max_workers=args.concurrency * len(valid_server_addresses)
        )
        waiting_queue = {
            server_address: [] for server_address in valid_server_addresses
        }
        pbar = tqdm(total=total_lines, desc="Processing")
        start_server_index = 0

        for line in input_file:
            data = json.loads(line.strip())
            messages = data["conversations"]

            # Remove original last assistant message
            if messages[-1]["role"] == "assistant":
                messages.pop()
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # find server address with the least waiting requests
            server_address = valid_server_addresses[start_server_index]
            start_server_index = (start_server_index + 1) % len(valid_server_addresses)

            # submit prompt to sglang
            while len(waiting_queue[server_address]) >= args.concurrency:
                finished_on_request = False
                # check if any future is done, if so, write the result to the output file
                for req_data, req_future in waiting_queue[server_address]:
                    if req_future.done():
                        req_output = req_future.result()

                        if req_output is None:
                            pass
                        else:
                            req_data["conversations"].append(
                                {"role": "assistant", "content": req_output}
                            )
                            output_file_handle.write(
                                json.dumps(req_data, ensure_ascii=False) + "\n"
                            )
                        waiting_queue[server_address].remove((req_data, req_future))
                        finished_on_request = True

                if finished_on_request:
                    break

            req_future = executor.submit(
                call_sglang,
                args.model,
                args.max_tokens,
                args.temperature,
                server_address,
                prompt,
            )
            waiting_queue[server_address].append((data, req_future))
            pbar.update(1)

        # deal with all the remaining requests
        for server_address, waiting_queue_items in waiting_queue.items():
            for req_data, req_future in waiting_queue_items:
                req_future.wait()
                req_output = req_future.result()
                if req_output is not None:
                    req_data["conversations"].append(
                        {"role": "assistant", "content": req_output}
                    )
                    output_file_handle.write(
                        json.dumps(req_data, ensure_ascii=False) + "\n"
                    )

    print(f"\nProcessing completed!")


if __name__ == "__main__":
    main()
