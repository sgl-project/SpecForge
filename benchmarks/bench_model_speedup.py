#!/usr/bin/env python3
"""
Usage:
config_list=(
    "1,0,0,0"
    "1,1,1,2"
    "1,2,1,3"
    "1,2,4,4"
    "1,3,1,4"
    "1,3,2,6"
    "1,4,1,5"
    "1,5,1,6"
    "1,5,8,16"
    "1,6,1,7"
    "1,7,1,8"
    "1,8,1,9"
)
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 bench_model_speedup.py \
    --model-path lmsys/gpt-oss-120b-bf16 \
    --speculative-draft-model-path zhuyksir/EAGLE3-gpt-oss-120b-bf16 \
    --port 20001 \
    --trust-remote-code \
    --mem-fraction-static 0.8 \
    --tp-size 4 \
    --attention-backend fa3 \
    --config-list "${config_list[@]}" \
    --benchmark-list mtbench:80 gsm8k:200 humaneval:200 math500:200 ceval:200 cmmlu:200 \
    --output lmsys_gpt-oss-120b_Eagle3_result.jsonl
"""
import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests
from datasets import load_dataset
from openai import AsyncOpenAI
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    kill_process_tree,
    popen_launch_server,
)
from sglang.utils import download_and_cache_file, read_jsonl
from tqdm.asyncio import tqdm

SYSTEM_PROMPT = None
# SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."


def parse_args():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parser.add_argument("--skip-launch-server", action="store_true", default=False)
    parser.add_argument("--num-prompts", type=int, default=80)
    parser.add_argument("--output", type=str, default="output.jsonl")
    parser.add_argument(
        "--config-list", type=str, nargs="+", default=["1,0,0,0", "1,3,1,4"]
    )
    parser.add_argument(
        "--benchmark-list",
        type=str,
        nargs="+",
        default=[
            "mtbench:80",
            "gsm8k:200",
            "humaneval:200",
            "math500:200",
            "ceval:200",
            "cmmlu:200",
        ],
    )
    parser.add_argument(
        "--split-category",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--enable-multi-turn-conversation",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def get_mtbench_conversations(
    num_prompts: int,
    split_category: bool = True,
    use_multi_turn_conversation: bool = False,
):
    url = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
    download_and_cache_file(url, filename="mtbench.jsonl")
    questions = list(read_jsonl("mtbench.jsonl"))[:num_prompts]
    bench_conversations, bench_name = {}, "mtbench"
    for q in questions:
        conversation = []
        conversation.append({"role": "user", "content": q["turns"][0]})
        if use_multi_turn_conversation:
            conversation.append({"role": "user", "content": q["turns"][1]})
        sub_bench_name = (
            f"{bench_name}-{q['category']}" if split_category else bench_name
        )
        if sub_bench_name not in bench_conversations:
            bench_conversations[sub_bench_name] = []
        bench_conversations[sub_bench_name].append(conversation)
    return bench_conversations


def get_gsm8k_conversations(num_prompts: int, num_shots: int = 5):
    def get_one_example(questions, i, include_answer):
        ret = "Question: " + questions[i]["question"] + "\nAnswer:"
        if include_answer:
            ret += " " + questions[i]["answer"]
        return ret

    def get_few_shot_examples(questions, k):
        ret = ""
        for i in range(k):
            ret += get_one_example(questions, i, True) + "\n\n"
        return ret

    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    download_and_cache_file(url, filename="gsm8k.jsonl")
    questions = list(read_jsonl("gsm8k.jsonl"))[:num_prompts]
    few_shot_examples = get_few_shot_examples(questions, num_shots)
    bench_name = "gsm8k"
    bench_conversations = {bench_name: []}
    for i in range(len(questions)):
        conversation = []
        conversation.append(
            {
                "role": "user",
                "content": few_shot_examples + get_one_example(questions, i, False),
            }
        )
        bench_conversations[bench_name].append(conversation)
    return bench_conversations


def get_humaneval_conversations(num_prompts: int):
    dataset = load_dataset("openai/openai_humaneval")["test"]
    prompts = [q["prompt"] for q in dataset][:num_prompts]
    bench_name = "humaneval"
    bench_conversations = {bench_name: []}
    for i in range(len(prompts)):
        bench_conversations[bench_name].append(
            [{"role": "user", "content": prompts[i]}]
        )
    return bench_conversations


def get_math500_conversations(num_prompts: int):
    dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
    prompts = [q["problem"] for q in dataset][:num_prompts]
    bench_name = "math500"
    bench_conversations = {bench_name: []}
    for i in range(len(prompts)):
        bench_conversations[bench_name].append(
            [{"role": "user", "content": prompts[i]}]
        )
    return bench_conversations


def get_ceval_conversations(num_prompts: int):
    dataset = load_dataset("zhaode/ceval")["train"]
    prompts = [q["instruction"] for q in dataset][:num_prompts]
    bench_name = "ceval"
    bench_conversations = {bench_name: []}
    for i in range(len(prompts)):
        bench_conversations[bench_name].append(
            [{"role": "user", "content": prompts[i]}]
        )
    return bench_conversations


def get_cmmlu_conversations(num_prompts: int):
    dataset = load_dataset("zhaode/cmmlu")["train"]
    prompts = [q["instruction"] for q in dataset][:num_prompts]
    bench_name = "ceval"
    bench_conversations = {bench_name: []}
    for i in range(len(prompts)):
        bench_conversations[bench_name].append(
            [{"role": "user", "content": prompts[i]}]
        )
    return bench_conversations


def launch_sglang_server(
    server_args: ServerArgs,
    base_url: str,
    batch_size: int,
    steps: int,
    topk: int,
    num_draft_tokens: int,
):
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
    process = popen_launch_server(
        server_args.model_path,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=sglang_args,
        env={
            "SGLANG_RECORD_STEP_TIME": "1",
            **os.environ,
        },
    )
    return process


@dataclass
class RequestFuncObject:
    conversation_id: str
    input_conversations: List[Dict[str, str]]
    model_name: str
    system_prompt: Optional[str]
    temperature: float = 0.0
    max_tokens: int = 2048
    output_conversations: Optional[List[Dict[str, str]]] = None
    output_tokens: int = 0
    error: Optional[str] = None


async def build_conversation(
    req_obj: RequestFuncObject, client: AsyncOpenAI, pbar: Optional[tqdm] = None
) -> str:
    messages = []
    if req_obj.system_prompt is not None and len(req_obj.system_prompt) > 0:
        messages.append({"role": "system", "content": req_obj.system_prompt})
    req_obj.output_tokens = 0
    for conversation in req_obj.input_conversations:
        if conversation["role"] == "assistant":
            continue
        if conversation["role"] == "user":
            messages.append({"role": "user", "content": conversation["content"]})
            try:
                response = await client.chat.completions.create(
                    model=req_obj.model_name,
                    messages=messages,
                    max_tokens=req_obj.max_tokens,
                    temperature=req_obj.temperature,
                    stream=False,
                )
                response_text = response.choices[0].message.content
                req_obj.output_tokens += response.usage.completion_tokens
            except Exception as e:
                req_obj.error = str(e)
                break
            messages.append({"role": "assistant", "content": response_text})

    if pbar is not None:
        pbar.update(1)
    req_obj.output_conversations = messages
    return req_obj


async def limited_build_conversation(
    req_obj: RequestFuncObject,
    client: AsyncOpenAI,
    semaphore: Optional[asyncio.Semaphore] = None,
    pbar: Optional[tqdm] = None,
):
    if semaphore is None:
        return await build_conversation(req_obj=req_obj, client=client, pbar=pbar)
    async with semaphore:
        return await build_conversation(req_obj=req_obj, client=client, pbar=pbar)


async def run_benchmark(
    conversation_list,
    server_args: ServerArgs,
    client: AsyncOpenAI,
    semaphore: Optional[asyncio.Semaphore] = None,
):
    pbar = tqdm(total=len(conversation_list))
    tasks = []
    for i, conversation in enumerate(conversation_list):
        req_obj = RequestFuncObject(
            conversation_id=str(i),
            input_conversations=conversation,
            model_name=server_args.model_path,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=512,
        )
        tasks.append(
            asyncio.create_task(
                limited_build_conversation(
                    req_obj=req_obj, client=client, semaphore=semaphore, pbar=pbar
                )
            )
        )

    outputs = await asyncio.gather(*tasks)
    return outputs


def send_flush_cache_request(base_url: str):
    requests.post(base_url + "/flush_cache")


def send_get_accept_length_request(base_url: str):
    try:
        server_info = requests.get(base_url + "/get_server_info")
    except Exception as e:
        print(f"Error sending get_server_info request: {e}")
        return None
    accept_length = None
    if server_info.status_code == 200:
        server_info_json = server_info.json()
        if "decode" in server_info_json:
            server_info_json = server_info_json["decode"][0]
        accept_length = server_info_json["internal_states"][0].get(
            "avg_spec_accept_length", None
        )
    return accept_length


def main():
    args = parse_args()
    server_args: ServerArgs = ServerArgs.from_cli_args(args)
    bench_conversations = {}
    configs = [tuple(map(int, config.split(","))) for config in args.config_list]
    # max_batch_size = max(batch_size for batch_size, _, _, _ in configs)

    benchmark_list = [tuple(b.split(":")) for b in args.benchmark_list]
    bench_conversations = {}
    for bench_name, num_prompts in benchmark_list:
        num_prompts = int(num_prompts)
        if bench_name == "mtbench":
            bench_conversations.update(
                get_mtbench_conversations(
                    num_prompts=num_prompts,
                    split_category=args.split_category,
                    use_multi_turn_conversation=args.enable_multi_turn_conversation,
                )
            )
        elif bench_name == "gsm8k":
            bench_conversations.update(get_gsm8k_conversations(num_prompts))
        elif bench_name == "humaneval":
            bench_conversations.update(get_humaneval_conversations(num_prompts))
        elif bench_name == "math500":
            bench_conversations.update(get_math500_conversations(num_prompts))
        elif bench_name == "ceval":
            bench_conversations.update(get_ceval_conversations(num_prompts))
        elif bench_name == "cmmlu":
            bench_conversations.update(get_cmmlu_conversations(num_prompts))
        else:
            print(f"{bench_name} is not supported yet, skip ... ")
            continue

    if len(bench_conversations) == 0:
        print(
            "no prompt is set, please check whether --benchmark-list is set correctly!"
        )
        exit()
    base_url = f"http://localhost:{args.port}"
    client = AsyncOpenAI(base_url=f"{base_url}/v1", api_key="None")
    if args.skip_launch_server:
        batch_size = configs[0][0] if len(configs) > 0 else 8
        for bench_name, conversation_list in bench_conversations.items():
            semaphore = asyncio.Semaphore(batch_size)
            start_timestamp = time.perf_counter()
            outputs = asyncio.run(
                run_benchmark(conversation_list, server_args, client, semaphore)
            )
            duration = time.perf_counter() - start_timestamp
            completion_tokens = sum(req_obj.output_tokens for req_obj in outputs)
            acc_length = send_get_accept_length_request(base_url)
            send_flush_cache_request(base_url)
            throughput = completion_tokens / duration
            print(
                f"bs=8:{bench_name=} {completion_tokens=} {acc_length=} {duration=}s {throughput=} token/s"
            )
        exit()

    for batch_size, steps, topk, num_draft_tokens in configs:
        process = launch_sglang_server(
            server_args, base_url, batch_size, steps, topk, num_draft_tokens
        )
        for bench_name, conversation_list in bench_conversations.items():
            semaphore = asyncio.Semaphore(batch_size)
            start_timestamp = time.perf_counter()
            outputs = asyncio.run(
                run_benchmark(conversation_list, server_args, client, semaphore)
            )
            duration = time.perf_counter() - start_timestamp
            completion_tokens = sum(req_obj.output_tokens for req_obj in outputs)
            time.sleep(3)
            if steps > 0:
                acc_length = send_get_accept_length_request(base_url)
            else:
                # steps == 0 means no speculative algorithm is used
                acc_length = 1.0
            record = {
                "batch_size": batch_size,
                "steps": steps,
                "topk": topk,
                "num_draft_tokens": num_draft_tokens,
                "acc_length": (
                    float(f"{acc_length:.2f}") if acc_length is not None else None
                ),
                "duration": float(f"{duration:.2f}"),
                "throughput": float(f"{completion_tokens / duration:.2f}"),
                "completion_tokens": completion_tokens,
                "benchmark": bench_name,
            }
            send_flush_cache_request(base_url)
            with open(args.output, "a") as fout:
                fout.write(json.dumps(record) + "\n")
        kill_process_tree(process.pid)


if __name__ == "__main__":
    main()
