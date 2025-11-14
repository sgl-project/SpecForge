import argparse
import time

import numpy as np
from datasets import load_dataset
from sglang import set_default_backend
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)


def main(args):
    # Select backend
    set_default_backend(select_sglang_backend(args))

    # Read data
    dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
    questions = [{"question": q["problem"]} for q in dataset]

    # Construct prompts
    questions = questions[: args.num_questions]

    #####################################
    ######### SGL Program Begin #########
    #####################################
    import sglang as sgl

    @sgl.function
    def get_humaneval_answer(s, question):
        s += sgl.user(question)
        s += sgl.assistant(sgl.gen("answer"))

    #####################################
    ########## SGL Program End ##########
    #####################################

    latency_list = []
    output_throughput_list = []
    accept_length_list = []
    for _ in range(args.num_runs):
        # Run requests
        tic = time.perf_counter()
        states = get_humaneval_answer.run_batch(
            questions,
            temperature=0,
            max_new_tokens=2048,
            num_threads=args.parallel,
            progress_bar=True,
        )
        latency = time.perf_counter() - tic

        # Compute speed
        num_output_tokens = sum(
            s.get_meta_info("answer")["completion_tokens"] for s in states
        )
        output_throughput = num_output_tokens / latency

        has_verify = "spec_verify_ct" in states[0].get_meta_info("answer")
        if has_verify:
            num_verify_tokens = sum(
                s.get_meta_info("answer")["spec_verify_ct"] for s in states
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

    # Print results
    print(f"Average Latency: {np.mean(latency_list):.3f} s")
    print(f"Average Output throughput: {np.mean(output_throughput_list):.3f} token/s")
    print(f"Average Accept length: {np.mean(accept_length_list):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-questions", type=int, default=200)
    parser.add_argument("--num-runs", type=int, default=1)
    args = add_common_sglang_args_and_parse(parser)
    main(args)
