#!/usr/bin/env python3
"""Prepare DFlash activation cache from rollout JSONL.

Run with TP-only torchrun for the target model, for example:

torchrun --standalone --nproc_per_node 8 scripts/prepare_dflash_hidden_states.py \
  --target-model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --data-path cache/dataset/perfectblend_qwen3_30b_rollouts.jsonl \
  --output-path /data/dflash_cache/qwen3_30b_a3b_12layers \
  --tp-size 8 --batch-size 4 --max-length 4096
"""

import argparse
import gc
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from datasets import Dataset
from specforge.args import SGLangBackendArgs
from specforge.data import build_eagle3_dataset
from specforge.data.dflash_cache import (
    DEFAULT_DFLASH_CAPTURE_LAYERS,
    DFlashCacheManifest,
    cache_file_path,
    layer_stack_from_concatenated,
    manifest_dict,
    parse_layer_ids,
    save_cache_record,
)
from specforge.data.utils import DataCollatorWithPadding
from specforge.distributed import destroy_distributed, get_dp_group, get_tp_group, init_distributed
from specforge.modeling.target.dflash_target_model import get_dflash_target_model
from specforge.utils import print_with_rank, rank_0_priority, safe_conversations_generator


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare DFlash hidden-state cache")
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--target-model-revision", type=str, default=None)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--tokenizer-revision", type=str, default=None)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--chat-template", type=str, default="qwen")
    parser.add_argument("--is-preformatted", action="store_true")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--build-dataset-num-proc", type=int, default=8)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--dist-timeout", type=int, default=2000)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument(
        "--capture-layer-ids",
        type=str,
        default=",".join(str(x) for x in DEFAULT_DFLASH_CAPTURE_LAYERS),
    )
    parser.add_argument("--file-group-size", type=int, default=2000)
    parser.add_argument("--compress", action="store_true")
    parser.add_argument(
        "--target-model-backend",
        choices=["sglang", "hf"],
        default="sglang",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    sglang_group = parser.add_argument_group("sglang")
    SGLangBackendArgs.add_args(sglang_group)
    return parser.parse_args()


def write_manifest(args, config, layer_ids):
    manifest = DFlashCacheManifest(
        target_model_path=args.target_model_path,
        target_model_revision=args.target_model_revision,
        tokenizer_path=args.tokenizer_path or args.target_model_path,
        tokenizer_revision=args.tokenizer_revision,
        selected_layer_ids=layer_ids,
        hidden_size=config.hidden_size,
        max_length=args.max_length,
        chat_template=args.chat_template,
        source_data_path=args.data_path,
    )
    path = Path(args.output_path) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(manifest_dict(manifest), f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def main():
    args = parse_args()
    layer_ids = parse_layer_ids(args.capture_layer_ids)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    tp_rank = dist.get_rank(get_tp_group())
    dp_size = dist.get_world_size(get_dp_group())
    if dp_size != 1:
        raise ValueError(
            "prepare_dflash_hidden_states currently expects TP-only collection "
            f"(dp_size=1), got dp_size={dp_size}."
        )

    config = AutoConfig.from_pretrained(
        args.target_model_path,
        revision=args.target_model_revision,
        trust_remote_code=args.trust_remote_code,
    )
    with rank_0_priority():
        if tp_rank == 0:
            write_manifest(args, config, layer_ids)

    target_kwargs = {}
    if args.target_model_backend == "sglang":
        target_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()
    target_model = get_dflash_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=torch.bfloat16,
        device="cuda" if args.target_model_backend == "hf" else None,
        trust_remote_code=args.trust_remote_code,
        **target_kwargs,
    )
    target_model.set_capture_layers(layer_ids)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or args.target_model_path,
        revision=args.tokenizer_revision,
        trust_remote_code=args.trust_remote_code,
    )

    with rank_0_priority():
        dataset = Dataset.from_generator(
            generator=safe_conversations_generator,
            gen_kwargs={"file_path": args.data_path},
            cache_dir=os.path.join(args.cache_dir, "hf_dataset"),
            num_proc=min(args.build_dataset_num_proc, 32),
        )
    if args.num_samples is not None:
        dataset = dataset.select(range(args.num_samples))

    with rank_0_priority():
        processed = build_eagle3_dataset(
            dataset=dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            is_preformatted=args.is_preformatted,
            cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
            cache_key=(
                f"dflash-{Path(args.data_path).name}-{args.max_length}-"
                f"{args.target_model_path}-{','.join(map(str, layer_ids))}"
            ),
            num_proc=args.build_dataset_num_proc,
        )

    dataloader = DataLoader(
        processed,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=DataCollatorWithPadding(),
        drop_last=False,
    )

    show_progress = tp_rank == 0
    pbar = tqdm(total=len(processed), desc="dflash activations", disable=not show_progress)
    written = 0
    skipped = 0
    sample_index = 0
    try:
        for batch in dataloader:
            batch_size = batch["input_ids"].shape[0]
            batch_start_index = sample_index
            file_paths = [
                cache_file_path(args.output_path, batch_start_index + i, args.file_group_size)
                for i in range(batch_size)
            ]
            exists = [path.exists() or path.with_suffix(path.suffix + ".gz").exists() for path in file_paths]
            sample_index += batch_size

            if all(exists):
                skipped += batch_size
                if show_progress:
                    pbar.update(batch_size)
                continue

            gpu_batch = {
                "input_ids": batch["input_ids"].cuda(non_blocking=True),
                "attention_mask": batch["attention_mask"].cuda(non_blocking=True),
                "loss_mask": batch["loss_mask"].cuda(non_blocking=True),
            }
            output = target_model.generate_dflash_data(**gpu_batch)
            hidden_stack = layer_stack_from_concatenated(
                output.hidden_states,
                num_layers=len(layer_ids),
                hidden_size=config.hidden_size,
            )

            if tp_rank == 0:
                for i, path in enumerate(file_paths):
                    if exists[i]:
                        skipped += 1
                        continue
                    seq_len = int(batch["attention_mask"][i].sum().item())
                    record = {
                        "input_ids": batch["input_ids"][i, :seq_len].cpu().clone(),
                        "attention_mask": batch["attention_mask"][i, :seq_len].cpu().clone(),
                        "loss_mask": batch["loss_mask"][i, :seq_len].cpu().clone(),
                        "selected_hidden_states": hidden_stack[i, :seq_len].cpu().clone(),
                        "selected_layer_ids": layer_ids,
                        "hidden_size": config.hidden_size,
                        "source_index": batch_start_index + i,
                    }
                    save_cache_record(record, path, compress=args.compress)
                    written += 1

            del gpu_batch, output, hidden_stack
            torch.cuda.empty_cache()
            gc.collect()
            if show_progress:
                pbar.update(batch_size)
                pbar.set_postfix({"written": written, "skipped": skipped})
    finally:
        if show_progress:
            pbar.close()
            print(f"Activation cache done: written={written}, skipped={skipped}")
        destroy_distributed()


if __name__ == "__main__":
    main()
