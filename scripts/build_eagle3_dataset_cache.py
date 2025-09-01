"""
Preprocess data for dataset generation. This runs faster without c10d comms.
"""

import argparse
import hashlib
import os

from accelerate.utils import set_seed
from datasets import load_dataset
from transformers import AutoTokenizer

from specforge import AutoDraftModelConfig
from specforge.data import build_eagle3_dataset, generate_vocab_mapping_file
from specforge.data.template import TEMPLATE_REGISTRY


def parse_args():
    parser = argparse.ArgumentParser(description="Train Eagle3 with online data")

    # add model-related arguments
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--draft-model-config", type=str, required=True)

    # add training-related arguments
    parser.add_argument("--train-data-path", type=str, required=True)
    parser.add_argument("--eval-data-path", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=2048)

    # data processing type
    parser.add_argument("--chat-template", type=str, required=True)

    # other args
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--view-train-data", type=int, nargs="+", default=[])
    args = parser.parse_args()
    return args


def main():
    # initialize
    args = parse_args()
    assert (
        args.chat_template in TEMPLATE_REGISTRY.get_all_template_names()
    ), f"Chat template {args.chat_template} not found in TEMPLATE_REGISTRY, support templates: {TEMPLATE_REGISTRY.get_all_template_names()}"
    set_seed(args.seed)

    # build dataloaders
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)
    if args.eval_data_path is not None:
        test_ds = load_dataset("json", data_files=args.eval_data_path)["train"]
        test_cache_params_string = (
            f"{args.eval_data_path}-"
            f"{args.max_length}-"
            f"{args.chat_template}-"
            f"{args.target_model_path}"  # Tokenizer may also different
        )
        test_cache_key = hashlib.md5(test_cache_params_string.encode()).hexdigest()
        print(
            f"test cache key: {test_cache_key}, test output jsonl path: {args.eval_data_path}"
        )
        test_eagle3_dataset = build_eagle3_dataset(
            dataset=test_ds,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
            cache_key=test_cache_key,
        )
        RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
        for idx in args.view_train_data:
            input_ids = test_eagle3_dataset["input_ids"][idx].view(-1)
            loss_mask = test_eagle3_dataset["loss_mask"][idx].view(-1).tolist()
            print(f"Loss mask sum: {sum(loss_mask)}")
            current_mask = input_ids[0]
            current_ids = []
            for i in range(len(input_ids)):
                if current_mask == loss_mask[i]:
                    current_ids.append(input_ids[i])
                else:
                    decoded_text = tokenizer.decode(
                        current_ids, skip_special_tokens=False
                    )
                    if current_mask == 0:
                        print(f"{RED}{decoded_text}{RESET}", end="")
                    else:
                        print(f"{GREEN}{decoded_text}{RESET}", end="")
                    current_ids = [input_ids[i]]
                    current_mask = loss_mask[i]
            print(
                f"{GREEN}{tokenizer.decode(current_ids, skip_special_tokens=False)}{RESET}"
            )
    train_ds = load_dataset("json", data_files=args.train_data_path)["train"]
    train_cache_params_string = (
        f"{args.train_data_path}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}"  # Tokenizer may also different
    )
    train_cache_key = hashlib.md5(train_cache_params_string.encode()).hexdigest()
    print(
        f"train cache key: {train_cache_key}, train output jsonl path: {args.train_data_path}"
    )
    train_eagle3_dataset = build_eagle3_dataset(
        dataset=train_ds,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
        cache_key=train_cache_key,
    )
    RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
    for idx in args.view_train_data:
        input_ids = train_eagle3_dataset["input_ids"][idx].view(-1)
        loss_mask = train_eagle3_dataset["loss_mask"][idx].view(-1).tolist()
        print(f"Loss mask sum: {sum(loss_mask)}")
        current_mask = input_ids[0]
        current_ids = []
        for i in range(len(input_ids)):
            if current_mask == loss_mask[i]:
                current_ids.append(input_ids[i])
            else:
                decoded_text = tokenizer.decode(current_ids, skip_special_tokens=False)
                if current_mask == 0:
                    print(f"{RED}{decoded_text}{RESET}", end="")
                else:
                    print(f"{GREEN}{decoded_text}{RESET}", end="")
                current_ids = [input_ids[i]]
                current_mask = loss_mask[i]
        print(
            f"{GREEN}{tokenizer.decode(current_ids, skip_special_tokens=False)}{RESET}"
        )

    vocab_mapping_path = generate_vocab_mapping_file(
        dataset=train_eagle3_dataset,
        target_vocab_size=draft_model_config.vocab_size,
        draft_vocab_size=draft_model_config.draft_vocab_size,
        cache_dir=os.path.join(args.cache_dir, "vocab_mapping"),
        cache_key=train_cache_key,
    )


if __name__ == "__main__":
    main()
