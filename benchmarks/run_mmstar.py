import argparse
import ast
import os
import re
import shutil
import time

from datasets import load_dataset
from sglang import set_default_backend
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)


def main(args):
    # Select backend
    set_default_backend(select_sglang_backend(args))

    cache_dir = os.path.join(".cache", "mmstar_specforge")
    image_dir = os.path.join(cache_dir, "images")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    print(f"Created temporary image directory: {cache_dir}")

    # Read data
    dataset = load_dataset("Lin-Chen/MMStar")["validation"]
    questions = []
    for idx, q in enumerate(dataset):
        if idx >= args.num_questions:
            break
        image = q["image"]
        image_path = os.path.join(cache_dir, q["meta_info"]["image_path"])
        image.convert("RGB").save(image_path, "JPEG")
        item = [
            {
                "image_path": image_path,
                "question": q["question"].split("Options:", 1)[0].strip(),
            }
        ]
        questions.append(item)

    #####################################
    ######### SGL Program Begin #########
    #####################################
    import sglang as sgl

    @sgl.function
    def get_mmstar_answer(s, question):
        s += sgl.user(sgl.image(question["image_path"]) + question["question"])
        s += sgl.assistant(sgl.gen("answer"))

    #####################################
    ########## SGL Program End ##########
    #####################################

    # Run requests
    tic = time.perf_counter()
    states = get_mmstar_answer.run_batch(
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

    # Print results
    print(f"Latency: {latency:.3f} s")
    print(f"Output throughput: {output_throughput:.3f} token/s")
    print(f"Accept length: {accept_length:.3f}")

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Deleted temporary directory: {cache_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-questions", type=int, default=20)
    args = add_common_sglang_args_and_parse(parser)
    main(args)
