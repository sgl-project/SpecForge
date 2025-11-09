"""
This script will generate the hidden states for the dataset use transformer as the target model backend.
By generating hidden states in advance, we can avoid:
- the memory overhead of loading target model
- the latency overhead of generating hidden states for each request.

Optimized for lower memory usage and higher efficiency.
"""

import argparse
import gc
import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from datasets import load_dataset
from rich.logging import RichHandler
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from specforge.data import build_eagle3_dataset, prepare_dp_dataloaders
from specforge.distributed import (
    destroy_distributed,
    get_target_dp_group,
    get_target_dp_rank,
    get_target_dp_size,
    get_target_tp_group,
    get_target_tp_rank,
    get_target_tp_size,
    init_distributed,
)
from specforge.modeling.target import Eagle3TargetModel, get_eagle3_target_model
from specforge.utils import print_with_rank, rank_0_priority


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--chat-template", type=str, default="llama3")
    parser.add_argument("--target-tp-size", type=int, default=1)
    parser.add_argument(
        "--target-tp-size",
        "--tp",
        dest="target_tp_size",
        type=int,
        default=1,
        help="Tensor parallel size (alias: --tp)",
    )

    parser.add_argument("--target-batch-size", type=int, default=32)
    parser.add_argument("--target-micro-batch-size", type=int, default=1)
    parser.add_argument(
        "--is-vlm", action="store_true", help="Whether the target model is a VLM"
    )
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--enable-aux-hidden-states", action="store_true")
    parser.add_argument("--aux-hidden-states-layers", type=str, default=None)
    parser.add_argument("--build-dataset-num-proc", type=int, default=8)
    parser.add_argument(
        "--dist-timeout",
        type=int,
        default=2000,
        help="Timeout for collective communication in minutes",
    )
    parser.add_argument(
        "--num-io-threads",
        type=int,
        default=4,
        help="Number of threads for async I/O operations",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers for DataLoader"
    )
    parser.add_argument(
        "--io-queue-size",
        type=int,
        default=50,
        help="Max number of pending I/O futures.",
    )
    parser.add_argument(
        "--file-group-size",
        type=int,
        default=2000,
        help="Number of files per subdirectory.",
    )
    parser.add_argument(
        "--tp-sync-interval",
        type=int,
        default=10,
        help="Batch interval for TP group barrier synchronization.",
    )
    return parser.parse_args()


def build_target_model(
    args: argparse.Namespace, model_config: AutoConfig
) -> Tuple[Eagle3TargetModel, Optional[AutoProcessor]]:
    """
    Build the target model according to the arguments.

    For VLM models (Qwen2.5-VL) without TP, load directly from transformers.
    Otherwise, use the Eagle3 target model wrapper.
    """
    if (
        args.is_vlm
        and model_config.model_type == "qwen2_5_vl"
        and get_target_tp_size() == 1
    ):
        from transformers import Qwen2_5_VLForConditionalGeneration

        target_model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=args.target_model_path,
                torch_dtype=torch.bfloat16,
            )
            .eval()
            .cuda()
        )
    else:
        target_model = get_eagle3_target_model(
            pretrained_model_name_or_path=args.target_model_path,
            backend="hf",
            torch_dtype=torch.bfloat16,
            device="cuda",
            cache_dir=args.cache_dir,
        )

    # Set auxiliary hidden states layers if specified
    if args.aux_hidden_states_layers is not None:
        target_model.set_aux_hidden_states_layers(args.aux_hidden_states_layers)
    else:
        target_model.set_aux_hidden_states_layers()

    if args.is_vlm:
        processor = AutoProcessor.from_pretrained(args.target_model_path)
    else:
        processor = None

    return target_model, processor


class HiddenStatesGenerator:
    """
    Generator for creating and saving hidden states from target model.

    (Refactored Version)
    - Fixes a potential deadlock in TP > 1 scenarios when a batch is skipped.
    - Implements a context manager (`with` statement) for robust resource handling.
    - Makes internal settings (like queue sizes, group sizes) configurable.
    - Centralizes resource cleanup logic.
    """

    def __init__(
        self,
        target_model,
        enable_aux_hidden_states: bool = True,
        num_io_threads: int = 4,
        io_queue_size: int = 50,
        file_group_size: int = 2000,
        tp_sync_interval: int = 10,
    ):
        """
        Args:
            target_model: The model for inference.
            enable_aux_hidden_states: Whether to save auxiliary hidden states.
            num_io_threads: Number of threads for async I/O.
            io_queue_size: Max number of pending I/O futures before cleanup.
            file_group_size: Number of files per subdirectory.
            tp_sync_interval: How often (in batches) to synchronize TP group with a barrier.
        """
        self.model = target_model
        self.enable_aux_hidden_states = enable_aux_hidden_states

        # --- REFACTOR: Configurable parameters ---
        self.num_io_threads = num_io_threads
        self.io_queue_size = io_queue_size
        self.file_group_size = file_group_size
        self.tp_sync_interval = tp_sync_interval

        # Determine if this rank should show progress (DP rank 0)
        dp_group = get_target_dp_group()
        self.show_progress = get_target_dp_rank() == 0 if dp_group else True

        # --- REFACTOR: Thread pool is now managed by __enter__ and __exit__ ---
        self.io_executor = None
        self.pending_futures = []

    def __enter__(self):
        """Initializes resources when entering a 'with' block."""
        self.io_executor = ThreadPoolExecutor(max_workers=self.num_io_threads)
        self.pending_futures = []
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleans up resources when exiting a 'with' block."""
        if self.io_executor is not None:
            if self.show_progress:
                print("\nWaiting for all async I/O operations to complete...")
            self._wait_all_saves()
            self.io_executor.shutdown(wait=True)
            self.io_executor = None  # Reset for safety

        # Final barrier to ensure all processes exit generate() cleanly
        if dist.is_initialized() and get_target_tp_group():
            dist.barrier(group=get_target_tp_group())

    def _save_tensor_sync(self, data_point: Dict[str, torch.Tensor], output_file: str):
        if "hidden_state" in data_point and torch.any(
            torch.isnan(data_point["hidden_state"])
        ):
            print(
                f"Warning: NaN found in hidden_state for {output_file}. Skipping save."
            )
            return
        if "aux_hidden_state" in data_point and torch.any(
            torch.isnan(data_point["aux_hidden_state"])
        ):
            print(
                f"Warning: NaN found in aux_hidden_state for {output_file}. Skipping save."
            )
            return
        torch.save(data_point, output_file)

    def _save_tensor_async(self, data_point: Dict[str, torch.Tensor], output_file: str):
        # If the queue of pending save operations is full, we must wait.
        if len(self.pending_futures) >= self.io_queue_size:
            # First, try to clear any futures that have already finished without waiting.
            self.pending_futures = [f for f in self.pending_futures if not f.done()]
            # If the queue is *still* full, it means all I/O threads are busy and we have
            # a backlog. We must now block the main generation loop and wait for the
            # oldest I/O operation to complete before proceeding.
            if len(self.pending_futures) >= self.io_queue_size:
                self.pending_futures.pop(0).result()
        future = self.io_executor.submit(
            self._save_tensor_sync, data_point, output_file
        )
        self.pending_futures.append(future)

    def _wait_all_saves(self):
        if self.pending_futures:
            for future in tqdm(
                self.pending_futures,
                desc="Finalizing Writes",
                disable=not self.show_progress,
            ):
                future.result()  # Wait and raise exception if any
            self.pending_futures.clear()

    def _prepare_output_dirs(self, output_path: Path, total_samples: int):
        if total_samples == 0:
            return
        start_group = 0
        end_idx = total_samples - 1
        end_group = (end_idx // self.file_group_size) * self.file_group_size
        for group_idx in range(start_group, end_group + 1, self.file_group_size):
            grouped_subdir = f"dp_{get_target_dp_rank()}_rows_{group_idx}-{group_idx + self.file_group_size}"
            output_dir = os.path.join(output_path, grouped_subdir)
            os.makedirs(output_dir, exist_ok=True)

    def _check_existing_files_batch(
        self, output_path: Path, global_indices: List[int]
    ) -> bool:
        # only skip file generation if all files exist
        return all(
            self._get_file_path(output_path, idx).exists() for idx in global_indices
        )

    def _get_file_path(self, output_path: Path, idx: int) -> Path:
        idx = idx * get_target_tp_size() + get_target_tp_rank()
        group_idx = (idx // self.file_group_size) * self.file_group_size
        grouped_subdir = f"dp_{get_target_dp_rank()}_rows_{group_idx}-{group_idx + self.file_group_size}"
        return output_path / grouped_subdir / f"data_{idx}.ckpt"

    @torch.no_grad()
    def generate(
        self,
        data_loader: torch.utils.data.DataLoader,
        output_path: str,
    ):
        self._prepare_output_dirs(
            output_path, len(data_loader) * data_loader.batch_size
        )

        tp_group = get_target_tp_group()
        tp_size = get_target_tp_size()
        global_idx = 0

        progress_bar = tqdm(
            data_loader,
            disable=(not self.show_progress),
            desc="Generating Hidden States",
            position=0,
            leave=True,
        )

        for batch_idx, batch in enumerate(progress_bar):
            batch_size = batch["input_ids"].size(0)
            current_batch_indices = list(range(global_idx, global_idx + batch_size))

            # Step 1: TP rank 0 checks which samples need processing
            exists_list = self._check_existing_files_batch(
                output_path, current_batch_indices
            )
            if exists_list:
                global_idx += batch_size
                continue
            eagle3_data_list = self.model.generate_eagle3_data(
                input_ids=batch["input_ids"].cuda(),
                attention_mask=batch["attention_mask"].cuda(),
                loss_mask=batch["loss_mask"].cuda(),
            )
            for eagle3_data in eagle3_data_list:
                seq_lengths = eagle3_data.attention_mask.sum(dim=1).tolist()

                for i, (current_global_idx, seq_len) in enumerate(
                    zip(current_batch_indices, seq_lengths)
                ):

                    # Process ONE sample at a time to minimize CPU RAM footprint
                    # 1. Transfer only the required slice for one sample to CPU
                    hidden_state_cpu = (
                        eagle3_data.target[i, :seq_len, :].cpu().clone().unsqueeze(0)
                    )

                    aux_hidden_state_cpu = None
                    if self.enable_aux_hidden_states and hasattr(
                        eagle3_data, "hidden_states"
                    ):
                        aux_hidden_state_cpu = (
                            eagle3_data.hidden_states[i, :seq_len, :]
                            .cpu()
                            .clone()
                            .unsqueeze(0)
                        )

                    # 2. Prepare the data point for this single sample
                    data_point = {
                        "input_ids": batch["input_ids"][i, :seq_len].clone(),
                        "loss_mask": batch["loss_mask"][i, :seq_len].clone(),
                        "hidden_state": hidden_states_cpu,
                    }
                    if aux_hidden_state_cpu is not None:
                        data_point["aux_hidden_state"] = aux_hidden_state_cpu

                    # 3. Save asynchronously (the backpressure logic is still crucial)
                    output_file = self._get_file_path(output_path, current_global_idx)
                    self._save_tensor_async(data_point, output_file)

                del hidden_states_cpu, aux_hidden_states_cpu

            del eagle3_data_list, batch

            if batch_idx % 5 == 0:  # Make GC and cache clearing more frequent
                torch.cuda.empty_cache()
                gc.collect()

            if tp_size > 1 and batch_idx % self.tp_sync_interval == 0:
                dist.barrier(group=tp_group)

            global_idx += batch_size
            if self.show_progress:
                progress_bar.set_postfix(
                    {
                        "processed": global_idx,
                        "pending_io": len(self.pending_futures),
                    }
                )

        # --- REFACTOR: Cleanup is now handled by __exit__ ---
        print_with_rank(f"Generation loop finished. Processed: {global_idx}")


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[RichHandler()],
        force=True,
    )
    if args.aux_hidden_states_layers is not None:
        args.aux_hidden_states_layers = [
            int(x) for x in args.aux_hidden_states_layers.split(",")
        ]

    # Initialize distributed environment (TP + DP)
    init_distributed(
        timeout=args.dist_timeout, target_tp_size=args.target_tp_size, draft_tp_size=1
    )
    if dist.get_rank() == 0:
        Path(args.output_path).mkdir(parents=True, exist_ok=True)

    # Build target model (with TP)
    target_model_config = AutoConfig.from_pretrained(args.target_model_path)
    target_model, processor = build_target_model(args, target_model_config)

    print_with_rank(
        f"DP Rank {get_target_dp_rank()}, TP Rank {get_target_tp_rank()}, "
        f"DP Size {get_target_dp_size()}, TP Size {get_target_tp_size()}"
    )

    # Load complete dataset
    assert Path(
        args.data_path
    ).exists(), f"Dataset path {args.data_path} does not exist"
    dataset = load_dataset("json", data_files=args.data_path)["train"]
    if args.num_samples is not None:
        dataset = dataset.select(range(args.num_samples))

    # Tokenizer and cache key
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path, trust_remote_code=True
    )
    cache_params_string = f"{args.data_path}-{args.max_length}-{args.chat_template}-{args.target_model_path}-{args.num_samples}"
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()

    # Preprocess on complete, un-sharded dataset
    with rank_0_priority():
        print_with_rank("Main process is building the dataset cache...")
        eagle3_dataset = build_eagle3_dataset(
            dataset=dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
            cache_key=cache_key,
            is_vlm=args.is_vlm,
            processor=processor,
            num_proc=args.build_dataset_num_proc,
        )

    print_with_rank(f"Dataset prepared with {len(eagle3_dataset)} samples.")

    # Create DP-sharded dataloader
    data_loader = prepare_dp_dataloaders(
        dataset=eagle3_dataset,
        batch_size=args.target_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        process_group=get_target_dp_group(),
        is_vlm=args.is_vlm,
    )

    print_with_rank(
        f"DataLoader created for DP Rank {get_target_dp_rank()}. "
        f"Number of batches: {len(data_loader)}"
    )

    # Pass configurable arguments from args if needed
    with HiddenStatesGenerator(
        target_model,
        args.enable_aux_hidden_states,
        num_io_threads=args.num_io_threads,
        io_queue_size=args.io_queue_size,
        file_group_size=args.file_group_size,
        tp_sync_interval=args.tp_sync_interval,
        # Other params like io_queue_size can also be added to argparse
    ) as hidden_states_generator:

        # Generate hidden states
        hidden_states_generator.generate(
            data_loader,
            output_path=Path(args.output_path),
        )

    destroy_distributed()


if __name__ == "__main__":
    main()
