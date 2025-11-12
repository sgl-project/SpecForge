"""
Adapted from https://github.com/chromecast56/sglang/blob/6f145d2eadb93a116134f703358ce76f15381045/benchmark/mtbench/bench_sglang.py

Benchmark SGLang EAGLE/EAGLE3 Speculative Decoding

Usage:
python3 benchmark/mtbench/bench_sglang_eagle.py --num-questions 80 --parallel 1
"""

import argparse
import json
import os
import time
import uuid

import numpy as np
import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import download_and_cache_file, read_jsonl


def load_questions(filename):
    questions = []
    with open(filename, "r") as fin:
        for line in fin:
            obj = json.loads(line)
            questions.append(obj)
    return questions


def write_answers(filename, model_id, questions, answers):
    with open(os.path.expanduser(filename), "w") as fout:
        for i in range(len(answers)):
            ans_json = {
                "question_id": questions[i]["question_id"],
                "answer_id": uuid.uuid4().hex,
                "model_id": model_id,
                "choices": {
                    "index": 0,
                    "turns": [answers[i][0], answers[i][1]],
                },
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


@sgl.function
def answer_mt_bench(s, question_1, question_2):
    s += sgl.system(
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    )
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1"))
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2"))


def main(args):
    # Construct prompts
    url = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
    download_and_cache_file(url, filename="mtbench.jsonl")
    questions = list(read_jsonl("mtbench.jsonl"))
    questions = questions[: args.num_questions]
    arguments = [
        {"question_1": q["turns"][0], "question_2": q["turns"][1]} for q in questions
    ]

    # Select backend
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    latency_list = []
    output_throughput_list = []
    accept_length_list = []

    for _ in range(args.num_runs):
        # Run requests
        tic = time.perf_counter()
        rets = answer_mt_bench.run_batch(
            arguments,
            temperature=0,
            max_new_tokens=2048,
            num_threads=args.parallel,
            progress_bar=True,
        )

        latency = time.perf_counter() - tic
        num_output_tokens = sum(
            s.get_meta_info("answer_1")["completion_tokens"]
            + s.get_meta_info("answer_2")["completion_tokens"]
            for s in rets
        )

        output_throughput = num_output_tokens / latency

        has_verify = "spec_verify_ct" in rets[0].get_meta_info("answer_1")
        if has_verify:
            num_verify_tokens = sum(
                s.get_meta_info("answer_1")["spec_verify_ct"]
                + s.get_meta_info("answer_2")["spec_verify_ct"]
                for s in rets
            )
            if num_verify_tokens == 0:
                accept_length = 1.0
            else:
                accept_length = num_output_tokens / num_verify_tokens
        else:
            accept_length = 1.0

        latency_list.append(latency)
        output_throughput_list.append(output_throughput)
        accept_length_list.append(accept_length)

    print(f"Average Latency: {np.mean(latency_list):.3f} s")
    print(f"Average Output throughput: {np.mean(output_throughput_list):.3f} token/s")
    print(f"Average Accept length: {np.mean(accept_length_list):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-questions", type=int, default=80)
    parser.add_argument("--num-runs", type=int, default=1)
    args = add_common_sglang_args_and_parse(parser)
    main(args)
