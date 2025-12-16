#!/usr/bin/env python3
"""
Usage:

# if you want to run benchmarks directly
# mtbench:20 means only run 20 samples in the dataset
python bench_eagle3.py \
    --model meta-llama/Llama-3.1-8B-Instruct   \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
    --port 30000 \
    --config-list 1,0,0,0 1,3,1,4 \
    --benchmark-list mtbench:20 \
    --dtype bfloat16


or if you want run sglang alone.

# launch sglang
python3 -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct   \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --mem-fraction-static 0.75 \
    --cuda-graph-max-bs 1 \
    --tp 1 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16

# then run benchmarks
python bench_eagle3.py \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 30000 \
    --config-list 1,0,0,0 \
    --benchmark-list mtbench:80 \
    --dtype bfloat16 \
    --skip-launch-server
"""
import argparse
import json
import os
import time
from dataclasses import asdict
from typing import List

import requests
from benchmarker import BENCHMARKS
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    kill_process_tree,
    popen_launch_server,
)
from sglang.utils import wait_for_server


def parse_args():
    parser = argparse.ArgumentParser()
    sglang_group = parser.add_argument_group("sglang")
    ServerArgs.add_cli_args(sglang_group)

    # make the follow args a group
    benchmark_group = parser.add_argument_group("benchmark")
    benchmark_group.add_argument(
        "--skip-launch-server", action="store_true", default=False
    )
    benchmark_group.add_argument("--num-prompts", type=int, default=80)
    benchmark_group.add_argument("--output-dir", type=str, default="./results")
    benchmark_group.add_argument(
        "--config-list", type=str, nargs="+", default=["1,0,0,0", "1,3,1,4"]
    )
    benchmark_group.add_argument(
        "--name",
        type=str,
        default=None,
        help="name of this benchmark run, if provided, will be added to the output file name",
    )
    benchmark_group.add_argument(
        "--benchmark-list",
        type=str,
        nargs="+",
        default=[
            "mtbench:80",
            "gsm8k:200",
            "humaneval:200",
            "math500:200",
            "ceval:200",
        ],
        help=f"The list of benchmarks to run. The format is <benchmark-name>:<num-prompts>:<subset>,<subset>. We support the following benchmarks: {', '.join(BENCHMARKS.benchmarks.keys())}",
    )
    benchmark_group.add_argument(
        "--enable-multi-turn-conversation",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def launch_sglang_server(
    server_args: ServerArgs,
    base_url: str,
    batch_size: int,
    steps: int,
    topk: int,
    num_draft_tokens: int,
):
    """
    This function launches the SGLang server with the given server arguments.
    """
    sglang_args: List[str] = []
    if steps > 0:
        sglang_args.extend(
            [
                "--speculative-algorithm",
                "EAGLE3",
                "--speculative-num-steps",
                str(steps),
                "--speculative-eagle-topk",
                str(topk),
                "--speculative-num-draft-tokens",
                str(num_draft_tokens),
                "--speculative-draft-model-path",
                server_args.speculative_draft_model_path,
            ]
        )

    sglang_args.extend(
        [
            "--cuda-graph-max-bs",
            str(batch_size),
            "--mem-fraction-static",
            str(server_args.mem_fraction_static),
            "--tp-size",
            str(server_args.tp_size),
            "--max-running-requests",
            str(batch_size),
        ]
    )

    if server_args.trust_remote_code:
        sglang_args.extend(["--trust-remote-code"])

    if server_args.disable_radix_cache:
        sglang_args.extend(["--disable-radix-cache"])

    if server_args.ep_size:
        sglang_args.extend(["--ep-size", str(server_args.ep_size)])

    if server_args.attention_backend:
        sglang_args.extend(["--attention-backend", server_args.attention_backend])

    if server_args.quantization:
        sglang_args.extend(["--quantization", server_args.quantization])

    if server_args.dtype:
        sglang_args.extend(["--dtype", server_args.dtype])

    process = popen_launch_server(
        server_args.model_path,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=sglang_args,
        env={
            "SGLANG_RECORD_STEP_TIME": "1",
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
            **os.environ,
        },
    )
    return process


def send_flush_cache_request(base_url: str):
    requests.post(base_url + "/flush_cache")


def main():
    args = parse_args()
    server_args: ServerArgs = ServerArgs.from_cli_args(args)
    configs = [tuple(map(int, config.split(","))) for config in args.config_list]

    # split the arg into list of (bench_name, num_prompts)
    benchmark_list = []
    for item in args.benchmark_list:
        splits = item.split(":")
        if len(splits) == 1:
            bench_name = splits[0]
            num_prompts = None
            subset = None
        elif len(splits) == 2:
            bench_name, num_prompts = splits
            subset = None
        elif len(splits) == 3:
            bench_name, num_prompts, subset = splits
            subset = subset.split(",")
        else:
            raise ValueError(f"Invalid benchmark list format: {item}")
        benchmark_list.append((bench_name, num_prompts, subset))
    assert len(benchmark_list) != 0, "the number of benchmark list is 0"

    base_url = f"http://localhost:{args.port}"
    results = {}

    def run_benchmarks(batch_size: int, steps: int, topk: int, num_draft_tokens: int):
        for benchmark_name, num_prompts, subset in benchmark_list:
            print(
                f"Running benchmark {benchmark_name} with {num_prompts} prompts, batch size {batch_size}, steps {steps}, topk {topk}, num_draft_tokens {num_draft_tokens}, subset {subset}"
            )
            benchmarkder_cls = BENCHMARKS.get(benchmark_name)
            num_prompts = int(num_prompts) if num_prompts is not None else None
            if subset is None:
                benchmarker = benchmarkder_cls(num_samples=num_prompts)
            else:
                benchmarker = benchmarkder_cls(num_samples=num_prompts, subset=subset)
            metrics_list = benchmarker.run(
                host=args.host, port=args.port, batch_size=batch_size
            )
            send_flush_cache_request(base_url)
            if benchmark_name not in results:
                results[benchmark_name] = []
            results[benchmark_name].append(
                dict(
                    batch_size=batch_size,
                    steps=steps,
                    topk=topk,
                    num_draft_tokens=num_draft_tokens,
                    metrics=[asdict(metric) for metric in metrics_list],
                    num_samples=num_prompts,
                )
            )

    if args.skip_launch_server:
        batch_size = configs[0][0] if len(configs) > 0 else 8
        run_benchmarks(batch_size, None, None, None)
    else:
        # we itearate over each config from args
        for batch_size, steps, topk, num_draft_tokens in configs:
            process = launch_sglang_server(
                server_args, base_url, batch_size, steps, topk, num_draft_tokens
            )
            wait_for_server(base_url)
            run_benchmarks(batch_size, steps, topk, num_draft_tokens)
            kill_process_tree(process.pid)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(
        args.output_dir,
        f"{args.name + '_' if args.name else ''}results_{timestamp}.jsonl",
    )
    with open(result_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {result_file}")


if __name__ == "__main__":
    main()
