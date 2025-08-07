import argparse
import time

from datasets import load_dataset
from sglang.api import set_default_backend
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)


def main(args):
    # Select backend
    set_default_backend(select_sglang_backend(args))

    # Read data
    dataset = load_dataset("Maxwell-Jia/AIME_2024")["train"]
    questions = [{"question": q["Problem"]} for q in dataset]

    # Construct prompts
    questions = questions[: args.num_questions]

    #####################################
    ######### SGL Program Begin #########
    #####################################
    import sglang as sgl

    @sgl.function
    def reasoning_gen(s, question: str):
        s += sgl.user(
            question
            + "\nPlease reason step by step, and put your final answer within \boxed{}."
        )
        s += sgl.assistant(
            sgl.gen(
                "answer",
            )
        )

    #####################################
    ########## SGL Program End ##########
    #####################################

    # Run requests
    tic = time.perf_counter()
    states = reasoning_gen.run_batch(
        questions,
        temperature=0,
        max_new_tokens=32768,
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

    # Print results
    print(f"Latency: {latency:.3f} s")
    print(f"Output throughput: {output_throughput:.3f} token/s")
    print(f"Accept length: {accept_length:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-questions", type=int, default=2)
    args = add_common_sglang_args_and_parse(parser)
    main(args)
