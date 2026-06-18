#!/usr/bin/env python3
"""
Usage:

# if you want to run benchmarks directly
# mmstar:20 means only run 20 samples in the dataset
python bench_dflash.py \
    --model-path Qwen/Qwen3-8B \
    --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 \
    --port 30000 \
    --config-list 1,8 1,16 \
    --benchmark-list mmstar:20 \
    --dtype bfloat16


or if you want run sglang alone.

# launch sglang
python3 -m sglang.launch_server \
    --model Qwen/Qwen3-8B \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 \
    --speculative-dflash-block-size 16 \
    --mem-fraction-static 0.75 \
    --cuda-graph-max-bs 1 \
    --tp 1 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16

# then run benchmarks
python bench_dflash.py \
    --model-path Qwen/Qwen3-8B \
    --port 30000 \
    --config-list 1,16 \
    --benchmark-list mmstar:80 \
    --dtype bfloat16 \
    --skip-launch-server
"""
import argparse
import json
import os
import time
from dataclasses import asdict
from typing import List, Optional, Tuple

import requests
from benchmarker import BENCHMARKS
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import kill_process_tree, popen_launch_server
from sglang.utils import wait_for_server


BenchmarkConfig = Tuple[int, Optional[int]]


def parse_args():
    parser = argparse.ArgumentParser()
    sglang_group = parser.add_argument_group("sglang")
    ServerArgs.add_cli_args(sglang_group)

    benchmark_group = parser.add_argument_group("benchmark")
    benchmark_group.add_argument(
        "--skip-launch-server", action="store_true", default=False
    )
    benchmark_group.add_argument("--timeout-for-server-launch", type=int, default=600)
    benchmark_group.add_argument("--num-prompts", type=int, default=80)
    benchmark_group.add_argument("--output-dir", type=str, default="./results")
    benchmark_group.add_argument(
        "--config-list",
        type=str,
        nargs="+",
        default=["1,8", "1,16"],
        help=(
            "The list of DFlash configurations to test. Preferred format is "
            "<batch-size>,<dflash-block-size>. "
            "For convenience, EAGLE-style <batch-size>,<steps>,<topk>,"
            "<num-draft-tokens> is also accepted, with steps/topk ignored and "
            "num-draft-tokens used as DFlash block size."
        ),
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
        required=True,
        help=f"The list of benchmarks to run. The format is <benchmark-name>:<num-prompts>:<subset>,<subset>. We support the following benchmarks: {', '.join(BENCHMARKS.benchmarks.keys())}",
    )
    benchmark_group.add_argument(
        "--enable-multi-turn-conversation",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def parse_config(config: str) -> BenchmarkConfig:
    values = tuple(map(int, config.split(",")))
    if len(values) == 2:
        batch_size, dflash_block_size = values
    elif len(values) == 4:
        batch_size, _steps, _topk, dflash_block_size = values
    else:
        raise ValueError(
            f"Invalid config format: {config}. Expected "
            "<batch-size>,<dflash-block-size> or "
            "<batch-size>,<steps>,<topk>,<num-draft-tokens>."
        )

    dflash_block_size = None if dflash_block_size <= 0 else dflash_block_size
    return batch_size, dflash_block_size


def build_benchmarker(benchmark_name: str, num_prompts, subset):
    benchmarker_cls = BENCHMARKS.get(benchmark_name)
    num_samples = int(num_prompts) if num_prompts is not None else None
    kwargs = {"num_samples": num_samples}
    if subset is not None:
        kwargs["subset"] = subset
    return benchmarker_cls(**kwargs)


def launch_sglang_server(
    server_args: ServerArgs,
    base_url: str,
    batch_size: int,
    dflash_block_size: Optional[int],
    timeout: int,
):
    """
    This function launches the SGLang server with the given server arguments.
    """
    sglang_args: List[str] = []
    if dflash_block_size is not None:
        if server_args.speculative_draft_model_path is None:
            raise ValueError(
                "--speculative-draft-model-path is required when DFlash is enabled"
            )
        sglang_args.extend(
            [
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-dflash-block-size",
                str(dflash_block_size),
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

    if getattr(server_args, "speculative_draft_attention_backend", None):
        sglang_args.extend(
            [
                "--speculative-draft-attention-backend",
                server_args.speculative_draft_attention_backend,
            ]
        )

    if getattr(server_args, "speculative_draft_model_quantization", None):
        sglang_args.extend(
            [
                "--speculative-draft-model-quantization",
                server_args.speculative_draft_model_quantization,
            ]
        )

    if getattr(server_args, "speculative_draft_model_revision", None):
        sglang_args.extend(
            [
                "--speculative-draft-model-revision",
                server_args.speculative_draft_model_revision,
            ]
        )

    if getattr(server_args, "speculative_draft_load_format", None):
        sglang_args.extend(
            [
                "--speculative-draft-load-format",
                server_args.speculative_draft_load_format,
            ]
        )

    if getattr(server_args, "mamba_scheduler_strategy", None):
        sglang_args.extend(
            ["--mamba-scheduler-strategy", server_args.mamba_scheduler_strategy]
        )

    process = popen_launch_server(
        server_args.model_path,
        base_url,
        timeout=timeout,
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
    configs = [parse_config(config) for config in args.config_list]

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
    results["model"] = server_args.speculative_draft_model_path

    def run_benchmarks(
        batch_size: int,
        dflash_block_size: Optional[int],
    ):
        for benchmark_name, num_prompts, subset in benchmark_list:
            print(
                f"Running benchmark {benchmark_name} with {num_prompts} prompts, "
                f"batch size {batch_size}, dflash_block_size {dflash_block_size}, "
                f"subset {subset}"
            )
            benchmarker = build_benchmarker(benchmark_name, num_prompts, subset)
            num_prompts = int(num_prompts) if num_prompts is not None else None
            metrics_list = benchmarker.run(
                host=args.host, port=args.port, batch_size=batch_size
            )
            send_flush_cache_request(base_url)
            if benchmark_name not in results:
                results[benchmark_name] = []
            results[benchmark_name].append(
                dict(
                    batch_size=batch_size,
                    dflash_block_size=dflash_block_size,
                    metrics=[asdict(metric) for metric in metrics_list],
                    num_samples=num_prompts,
                )
            )

    if args.skip_launch_server:
        batch_size = configs[0][0] if len(configs) > 0 else 8
        run_benchmarks(batch_size, None)
    else:
        for batch_size, dflash_block_size in configs:
            process = launch_sglang_server(
                server_args,
                base_url,
                batch_size,
                dflash_block_size,
                args.timeout_for_server_launch,
            )
            wait_for_server(base_url)
            run_benchmarks(batch_size, dflash_block_size)
            kill_process_tree(process.pid)
            process.wait()

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
