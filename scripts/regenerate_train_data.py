#!/usr/bin/env python3
import argparse
import json
import signal
import socket
import subprocess
import sys
import time
from typing import List

import requests
from tqdm import tqdm
from transformers import AutoTokenizer

# Global variables will be initialized in main function
MODEL = None
MAX_TOKENS = None
BATCH_SIZE = None
TEMPERATURE = None
BASE_URL = None
HEADERS = {"Content-Type": "application/json"}
SERVER_PROCESS = None


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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--input-file-path", type=str, required=True)
    parser.add_argument("--output-file-path", type=str, required=True)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--max-running-requests", type=int, default=128)
    parser.add_argument(
        "--auto-launch-server",
        action="store_true",
        help="Automatically launch sglang server if port is available",
    )
    parser.add_argument("--num-samples", type=int, default=None)

    return parser.parse_args()


def is_port_in_use(port: int) -> bool:
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("localhost", port))
            return False
        except OSError:
            return True


def launch_sglang_server(
    model_path: str,
    port: int,
    tp_size: int,
    dp_size: int,
    mem_fraction_static: float,
    max_running_requests: int,
) -> subprocess.Popen:
    """Launch sglang server"""
    cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model",
        model_path,
        "--trust-remote-code",
        "--tp-size",
        str(tp_size),
        "--dp-size",
        str(dp_size),
        "--enable-cache-report",
        "--dtype",
        "bfloat16",
        "--log-level",
        "info",
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--port",
        str(port),
        "--max-running-requests",
        str(max_running_requests),
    ]

    print(f"Launching sglang server with command:")
    print(" ".join(cmd))

    # Start the server process
    process = subprocess.Popen(cmd)
    return process


def wait_for_server_ready(port: int, timeout: int = 3600) -> bool:
    """Wait for server to be ready"""
    print(f"Waiting for server to be ready at localhost:{port}...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        if is_port_in_use(int(port)):
            # Port is in use, try to make a simple request
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=5)
                if response.status_code == 200:
                    print("Server is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
        time.sleep(5)

    print(f"Server failed to start within {timeout} seconds")
    return False


def cleanup_server():
    """Clean up server process"""
    global SERVER_PROCESS
    if SERVER_PROCESS and SERVER_PROCESS.poll() is None:
        print("Shutting down sglang server...")
        SERVER_PROCESS.terminate()
        try:
            SERVER_PROCESS.wait(timeout=30)
        except subprocess.TimeoutExpired:
            SERVER_PROCESS.kill()
        print("Server shutdown complete")


def signal_handler(sig, frame):
    """Handle interrupt signals"""
    print("\nReceived interrupt signal, cleaning up...")
    cleanup_server()
    sys.exit(0)


def call_sglang_batch(prompts: List[str]) -> List[str]:
    """Send a batch of prompts to sglang /v1/completions."""
    global MODEL, MAX_TOKENS, TEMPERATURE, BASE_URL, HEADERS

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


def main():
    global MODEL, MAX_TOKENS, BATCH_SIZE, TEMPERATURE, BASE_URL, SERVER_PROCESS

    # Parse command line arguments
    args = parse_arguments()

    # Set global variables
    MODEL = args.model
    MAX_TOKENS = args.max_tokens
    BATCH_SIZE = args.batch_size
    TEMPERATURE = args.temperature
    BASE_URL = f"http://localhost:{args.port}/v1/completions"
    input_file_path = args.input_file_path
    output_file_path = args.output_file_path

    # Validate parameters
    if not (0.0 <= TEMPERATURE <= 1.0):
        raise ValueError("Temperature must be between 0.0 and 1.0")

    if MAX_TOKENS <= 0:
        raise ValueError("Max tokens must be greater than 0")

    if BATCH_SIZE <= 0:
        raise ValueError("Batch size must be greater than 0")

    # Check if server needs to be launched
    if args.auto_launch_server:
        port = args.port
        if not is_port_in_use(port):
            print(f"Port {port} is available, launching sglang server...")
            try:
                SERVER_PROCESS = launch_sglang_server(
                    model_path=args.model,
                    port=port,
                    tp_size=args.tp_size,
                    dp_size=args.dp_size,
                    mem_fraction_static=args.mem_fraction_static,
                    max_running_requests=args.max_running_requests,
                )

                # Wait for server to be ready
                if not wait_for_server_ready(port):
                    cleanup_server()
                    raise RuntimeError("Failed to start server")

                print("Server launched successfully!")
            except Exception as e:
                print(f"Failed to launch server: {e}")
                sys.exit(1)
        else:
            print(f"Port {port} is already in use, assuming server is running")
    else:
        port = args.port
        if not is_port_in_use(port):
            print(
                f"Warning: Port {port} is not in use. Please ensure sglang server is running."
            )

    # Set up signal handlers for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"Configuration:")
    print(f"  Model path: {MODEL}")
    print(f"  Max tokens: {MAX_TOKENS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  API URL: {BASE_URL}")
    print(f"  Input file: {input_file_path}")
    print(f"  Output file: {output_file_path}")
    print("-" * 50)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Variables for batch processing
    batch_prompts = []
    batch_data = []

    # Count total lines for progress bar
    print("Counting total lines in file...")
    with open(input_file_path, "r") as f:
        total_lines = sum(1 for _ in f)
    total_lines = (
        min(args.num_samples, total_lines) if args.num_samples else total_lines
    )
    print(f"Total {total_lines} lines to process")

    # Create progress bar
    pbar = tqdm(total=total_lines, desc="Processing", unit="item")

    processed_count = 0

    try:
        with open(input_file_path, "r") as input_file, open(
            output_file_path, "w"
        ) as output_file_handle:

            for _, line in zip(range(total_lines), input_file):
                data = json.loads(line)
                messages = data["conversations"]

                # Remove original last assistant message
                if messages[-1]["role"] == "assistant":
                    messages = messages[:-1]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                # Add to batch
                batch_prompts.append(prompt)
                batch_data.append(data)

                # Process when batch reaches specified size
                if len(batch_prompts) == BATCH_SIZE:
                    # Generate outputs
                    outputs = call_sglang_batch(batch_prompts)

                    # Process each output
                    for i, output in enumerate(outputs):
                        # Create assistant message
                        assistant_message = {"role": "assistant", "content": output}

                        # Add assistant message to original conversations
                        batch_data[i]["conversations"].append(assistant_message)

                        # Write to output file
                        output_file_handle.write(
                            json.dumps(batch_data[i], ensure_ascii=False) + "\n"
                        )

                        processed_count += 1
                        pbar.update(1)

                    # Update progress bar description
                    pbar.set_postfix(
                        {
                            "Processed": processed_count,
                            "Current batch": len(batch_prompts),
                        }
                    )

                    # Clear batch
                    batch_prompts = []
                    batch_data = []

            # Process remaining data that doesn't fill a complete batch
            if batch_prompts:
                outputs = call_sglang_batch(batch_prompts)

                # Process each output
                for i, output in enumerate(outputs):
                    assistant_message = {"role": "assistant", "content": output}

                    batch_data[i]["conversations"].append(assistant_message)
                    output_file_handle.write(
                        json.dumps(batch_data[i], ensure_ascii=False) + "\n"
                    )

                    # Update processing count and progress bar
                    processed_count += 1
                    pbar.update(1)

                # Update progress bar description
                pbar.set_postfix(
                    {"Processed": processed_count, "Last batch": len(batch_prompts)}
                )

        # Close progress bar
        pbar.close()
        print(f"\nProcessing completed! Total {processed_count} lines processed")

    except Exception as e:
        print(f"Error during processing: {e}")
        raise
    finally:
        # Clean up server if we launched it
        cleanup_server()


if __name__ == "__main__":
    main()
