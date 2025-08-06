import argparse
import ast
import re
import time

import numpy as np
import openai
from sglang import set_default_backend
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import download_and_cache_file, read_jsonl
from tqdm import tqdm

INVALID = -9999999


def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def main(args):
    # Read data
    data_path = "gsm8k.jsonl"
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    data_path = download_and_cache_file(url)
    lines = list(read_jsonl(data_path))
    lines = lines[: args.num_questions]
    client = openai.Client(base_url=f"http://127.0.0.1:30000/v1", api_key="None")

    # Run requests
    tic = time.perf_counter()

    ret = []
    for line in tqdm(lines):
        question = line["question"]
        assert isinstance(question, str)
        response = client.chat.completions.create(
            model="",
            messages=[
                {"role": "user", "content": question},
            ],
            temperature=0,
            max_tokens=2048,
            seed=42,
        )
        ret.append(response)

    total_completion_tokens = sum(r.usage.completion_tokens for r in ret)
    num_verify_tokens = sum(
        r.usage.spec_verify_ct if r.usage.spec_verify_ct is not None else 0 for r in ret
    )

    if num_verify_tokens == 0:
        accept_length = 1.0
    else:
        accept_length = total_completion_tokens / num_verify_tokens

    latency = time.perf_counter() - tic

    # Compute speed
    output_throughput = total_completion_tokens / latency

    # Print results
    print(f"Latency: {latency:.3f} s")
    print(f"Output throughput: {output_throughput:.3f} token/s")
    print(f"Accept length: {accept_length:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shots", type=int, default=1)
    parser.add_argument("--num-questions", type=int, default=200)
    args = add_common_sglang_args_and_parse(parser)
    main(args)
