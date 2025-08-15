"""
Usage:
config_list=(
    "1,0,0,0"
    "1,1,1,2"
    "1,2,1,3"
    "1,2,4,4"
    "1,3,1,4"
    "1,3,2,6"
    "1,3,4,4"
    "1,4,1,5"
    "1,5,1,6"
    "1,5,8,16"
    "1,5,8,32"
    "1,6,1,7"
    "1,7,1,8"
    "1,8,1,9"
    "1,8,8,32"
)
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run_mtbench_chatapi.py \
    --model-path lmsys/gpt-oss-120b-bf16 \
    --speculative-draft-model-path zhuyksir/EAGLE3-gpt-oss-120b-bf16 \
    --trust-remote-code \
    --mem-fraction-static 0.8 \
    --num-prompts 80 \
    --tp-size 4 \
    --config-list "${config_list[@]}" \
    --output mtbench_120b_eagle_tune_result.jsonl

CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server \
  --model lmsys/gpt-oss-20b-bf16 \
  --cuda-graph-max-bs 1 \
  --context-length 4096 \
  --dtype bfloat16 --mem-frac=0.8 --port 30001 &

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server \
  --model lmsys/gpt-oss-120b-bf16 \
  --cuda-graph-max-bs 1 \
  --context-length 4096 \
  --dtype bfloat16 --mem-frac=0.8 --tp 4 \
  --speculative-algo EAGLE3 \
  --speculative-draft zhuyksir/EAGLE3-gpt-oss-120b-bf16 \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 --port 30002 &
"""

import argparse
import asyncio
import json
import os
import time
from types import SimpleNamespace
from typing import List
import numpy as np
import requests

from sglang.bench_serving import DatasetRow, benchmark, set_global_args
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    kill_process_tree,
    popen_launch_server,
)
from sglang.utils import download_and_cache_file, read_jsonl

def get_eval_prompts():
    url = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
    download_and_cache_file(url, filename="mtbench.jsonl")
    questions = list(read_jsonl("mtbench.jsonl"))
    prompts = [q["turns"][0] for q in questions]
    return prompts

def parse_args():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parser.add_argument("--num-prompts", type=int, default=80)
    parser.add_argument("--output", type=str, default="output.jsonl")
    parser.add_argument("--config-list", type=str, nargs="+", default=["1,0,0,0", "1,3,1,4"])
    return parser.parse_args()

class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        return []

def send_one_batch(base_url, prompts: List[str], batch_size):
    # format: (prompt, input_len, output len). We set input_len as a dummy value 0.
    input_requests: List[DatasetRow] = [DatasetRow(p, 0, 512) for p in prompts]

    # We need to set some dummy values in order to call `benchmark` below.
    args = SimpleNamespace(
        disable_ignore_eos=False,
        disable_stream=False,
        return_logprob=False,
        backend="sglang-oai-chat",
        dataset_name="custom",
        num_prompts=None,
        sharegpt_output_len=None,
        random_input_len=None,
        random_output_len=None,
        random_range_ratio=None,
        output_file=None,
        warmup_requests=1,
        output_details=False,
    )
    set_global_args(args)
    tokenizer = FakeTokenizer()

    # Run benchmark
    results = asyncio.run(
        benchmark(
            backend="sglang-oai-chat",
            api_url=f"{base_url}/v1/chat/completions",
            base_url=base_url,
            model_id="default",
            tokenizer=tokenizer,
            input_requests=input_requests,
            request_rate=float("inf"),
            max_concurrency=batch_size,
            disable_tqdm=False,
            lora_names=None,
            extra_request_body={},
            profile=None,
        )
    )

    # assert results["completed"] == len(input_requests)
    acc_length = results["accept_length"] or 1.0
    avg_output_token = results["total_output_tokens"] / results["completed"]

    server_info = requests.get(base_url + "/get_server_info").json()
    # We use 20% percentile instead of median on purpose
    step_time = np.percentile(
        server_info["internal_states"][0]["step_time_dict"][str(batch_size)], 20
    )
    speed = 1 / step_time * acc_length

    return (
        round(acc_length, 3),
        round(step_time, 5),
        round(speed, 3),
        avg_output_token,
    )

def build_server_args(batch_size, steps, topk, num_draft_tokens, server_args):
    if steps == 0:
        # non-speculative decoding
        other_args = []
    else:
        other_args = [
            "--speculative-algorithm", "EAGLE3",
            "--speculative-num-steps", steps,
            "--speculative-eagle-topk", topk,
            "--speculative-num-draft-tokens", num_draft_tokens,
        ]
        if server_args.speculative_draft_model_path is not None:
            other_args.extend(
                [
                    "--speculative-draft-model-path",
                    server_args.speculative_draft_model_path,
                ]
            )

    other_args.extend(
        [
            "--cuda-graph-max-bs", batch_size,
            "--mem-fraction-static", server_args.mem_fraction_static,
            "--tp-size", server_args.tp_size,
            "--max-running-requests", batch_size,
        ]
    )

    if server_args.disable_cuda_graph:
        other_args.extend(["--disable-cuda-graph"])

    if server_args.trust_remote_code:
        other_args.extend(["--trust-remote-code"])

    if server_args.enable_ep_moe:
        other_args.extend(["--enable-ep-moe"])

    if server_args.attention_backend:
        other_args.extend(["--attention-backend", server_args.attention_backend])

    if server_args.quantization:
        other_args.extend(["--quantization", server_args.quantization])
    return other_args

def main(args, server_args):
    base_url = "http://127.0.0.1:20000"
    # batch_size, steps, topk, num_draft_tokens
    print(args.config_list)
    configs = [tuple(map(int, config.split(","))) for config in args.config_list]
    prompts = get_eval_prompts()
    prompts = prompts[: args.num_prompts]

    for batch_size, steps, topk, num_draft_tokens in configs:
        print(f"Start {batch_size=}, {steps=}, {topk=}, {num_draft_tokens=}")
        # Create an LLM.
        other_args = build_server_args(batch_size, steps, topk, num_draft_tokens, server_args)
        process = popen_launch_server(
            args.model_path,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env={
                "SGLANG_RECORD_STEP_TIME": "1",
                **os.environ,
            },
        )

        try:
            # Warmup
            send_one_batch(base_url, prompts[:batch_size], batch_size)

            # Benchmark
            acc_length, step_time, speed, completion_tokens = send_one_batch(base_url, prompts, batch_size)
        finally:
            kill_process_tree(process.pid)

        print(
            f"Finish {batch_size=}, {steps=}, {topk=}, {num_draft_tokens=}, {speed=:.2f} token/s, step_time={step_time * 1000:.2f} ms"
        )

        record = {
            "batch_size": batch_size,
            "steps": steps,
            "topk": topk,
            "num_draft_tokens": num_draft_tokens,
            "acc_length": acc_length,
            "step_time": step_time,
            "speed": speed,
            "completion_tokens": completion_tokens,
        }

        with open(args.output, "a") as fout:
            fout.write(json.dumps(record) + "\n")

        # Wait for the server to shutdown
        time.sleep(5)


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    args = parse_args()
    server_args: ServerArgs = ServerArgs.from_cli_args(args)
    main(args, server_args)