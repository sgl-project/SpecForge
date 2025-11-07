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
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from specforge.data import build_eagle3_dataset, prepare_dp_dataloaders
from specforge.distributed import (
    destroy_distributed,
    get_dp_group,
    get_tp_group,
    init_distributed,
    is_tp_rank_0,
)
from specforge.modeling.target import Eagle3TargetModel, get_eagle3_target_model
from specforge.utils import print_with_rank, rank_0_priority


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--chat-template", type=str, default="llama3")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
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
    if args.is_vlm and model_config.model_type == "qwen2_5_vl" and args.tp_size == 1:
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
        dp_group = get_dp_group()
        self.show_progress = dist.get_rank(dp_group) == 0 if dp_group else True

        # --- REFACTOR: Thread pool is now managed by __enter__ and __exit__ ---
        self.io_executor = None
        self.pending_futures = []

    def __enter__(self):
        """Initializes resources when entering a 'with' block."""
        if is_tp_rank_0():
            self.io_executor = ThreadPoolExecutor(max_workers=self.num_io_threads)
        self.pending_futures = []
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleans up resources when exiting a 'with' block."""
        if is_tp_rank_0() and self.io_executor is not None:
            if self.show_progress:
                print("\nWaiting for all async I/O operations to complete...")
            self._wait_all_saves()
            self.io_executor.shutdown(wait=True)
            self.io_executor = None  # Reset for safety

        # Final barrier to ensure all processes exit generate() cleanly
        if dist.is_initialized() and get_tp_group():
            dist.barrier(group=get_tp_group())

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
        assert is_tp_rank_0(), "Only tp_rank=0 should call _save_tensor_async"
        future = self.io_executor.submit(
            self._save_tensor_sync, data_point, output_file
        )
        self.pending_futures.append(future)
        if len(self.pending_futures) > self.io_queue_size:
            self.pending_futures = [f for f in self.pending_futures if not f.done()]

    def _wait_all_saves(self):
        if is_tp_rank_0() and self.pending_futures:
            for future in tqdm(
                self.pending_futures,
                desc="Finalizing Writes",
                disable=not self.show_progress,
            ):
                future.result()  # Wait and raise exception if any
            self.pending_futures.clear()

    def _prepare_output_dirs(
        self, output_path: str, start_idx: int, total_samples: int
    ):
        if not is_tp_rank_0() or total_samples == 0:
            return
        start_group = (start_idx // self.file_group_size) * self.file_group_size
        end_idx = start_idx + total_samples - 1
        end_group = (end_idx // self.file_group_size) * self.file_group_size
        for group_idx in range(start_group, end_group + 1, self.file_group_size):
            grouped_subdir = f"rows_{group_idx}-{group_idx + self.file_group_size}"
            output_dir = os.path.join(output_path, grouped_subdir)
            os.makedirs(output_dir, exist_ok=True)

    def _check_existing_files_batch(
        self, output_path: str, global_indices: List[int]
    ) -> List[bool]:
        if not is_tp_rank_0():
            return [False] * len(global_indices)

        def check_single_file(idx):
            return os.path.exists(self._get_file_path(output_path, idx))

        # Parallel file existence check
        with ThreadPoolExecutor(max_workers=self.num_io_threads) as executor:
            exists = list(executor.map(check_single_file, global_indices))
        return exists

    def _get_file_path(self, output_path: str, idx: int) -> str:
        group_idx = (idx // self.file_group_size) * self.file_group_size
        grouped_subdir = f"rows_{group_idx}-{group_idx + self.file_group_size}"
        return os.path.join(output_path, grouped_subdir, f"data_{idx}.ckpt")

    @torch.no_grad()
    def generate(
        self,
        data_loader: torch.utils.data.DataLoader,
        output_path: str,
        start_idx: int = 0,
        samples_per_dp: int = 0,
    ):
        self._prepare_output_dirs(output_path, start_idx, samples_per_dp)

        tp_group = get_tp_group()
        tp_size = dist.get_world_size(tp_group)
        tp_group_ranks = dist.get_process_group_ranks(tp_group)
        tp_rank_0_global = tp_group_ranks[0]
        global_idx = start_idx

        progress_bar = tqdm(
            data_loader,
            disable=(not self.show_progress),
            desc="Generating Hidden States",
            position=0,
            leave=True,
        )

        total_skipped, total_processed = 0, 0

        for batch_idx, batch in enumerate(progress_bar):
            batch_size = batch["input_ids"].size(0)
            current_batch_indices = list(range(global_idx, global_idx + batch_size))

            # Step 1: TP rank 0 checks which samples need processing
            if is_tp_rank_0():
                exists_list = self._check_existing_files_batch(
                    output_path, current_batch_indices
                )
                valid_indices_in_batch = [
                    i for i, exists in enumerate(exists_list) if not exists
                ]
                sample_global_indices = [
                    current_batch_indices[i] for i in valid_indices_in_batch
                ]
                num_valid = len(valid_indices_in_batch)
                total_skipped += batch_size - num_valid
            else:
                num_valid = 0
                sample_global_indices = []

            global_idx += batch_size

            # Step 1: Synchronize valid indices across TP group
            if tp_size > 1:
                if is_tp_rank_0():
                    # Use -1 as sentinel for empty batch
                    if num_valid == 0:
                        indices_to_broadcast = torch.tensor(
                            [-1], dtype=torch.long, device="cuda"
                        )
                    else:
                        indices_to_broadcast = torch.tensor(
                            valid_indices_in_batch, dtype=torch.long, device="cuda"
                        )
                    length_tensor = torch.tensor(
                        [indices_to_broadcast.size(0)], dtype=torch.long, device="cuda"
                    )
                else:
                    length_tensor = torch.zeros(1, dtype=torch.long, device="cuda")

                # Broadcast length first
                dist.broadcast(length_tensor, src=tp_rank_0_global, group=tp_group)

                # Create receive buffer on other ranks
                if not is_tp_rank_0():
                    indices_to_broadcast = torch.zeros(
                        length_tensor.item(), dtype=torch.long, device="cuda"
                    )

                # Broadcast indices
                dist.broadcast(
                    indices_to_broadcast, src=tp_rank_0_global, group=tp_group
                )

                # Check for empty batch sentinel
                if indices_to_broadcast[0].item() == -1:
                    del length_tensor, indices_to_broadcast
                    continue

                # Convert to list on non-rank-0
                if not is_tp_rank_0():
                    valid_indices_in_batch = indices_to_broadcast.tolist()

                del length_tensor, indices_to_broadcast
            else:
                if num_valid == 0:
                    continue

            # Step 2: Filter batch before moving to GPU to save memory
            filtered_batch = {
                "input_ids": batch["input_ids"][valid_indices_in_batch],
                "attention_mask": batch["attention_mask"][valid_indices_in_batch],
                "loss_mask": batch["loss_mask"][valid_indices_in_batch],
            }
            filtered_batch_gpu = {
                k: v.cuda(non_blocking=True) for k, v in filtered_batch.items()
            }

            eagle3_data = self.model.generate_eagle3_data(**filtered_batch_gpu)
            del filtered_batch_gpu

            if is_tp_rank_0():
                seq_lengths = eagle3_data.attention_mask.sum(dim=1).tolist()
                hidden_states_cpu = eagle3_data.hidden_states.cpu()
                aux_hidden_states_cpu = None
                if self.enable_aux_hidden_states and hasattr(
                    eagle3_data, "aux_hidden_states"
                ):
                    aux_hidden_states_cpu = eagle3_data.aux_hidden_states.cpu()

                for i, (current_global_idx, seq_len) in enumerate(
                    zip(sample_global_indices, seq_lengths)
                ):
                    data_point = {
                        "input_ids": filtered_batch["input_ids"][i].clone(),
                        "loss_mask": filtered_batch["loss_mask"][i].clone(),
                        "hidden_state": hidden_states_cpu[i, :seq_len, :]
                        .clone()
                        .unsqueeze(0),
                    }
                    if aux_hidden_states_cpu is not None:
                        data_point["aux_hidden_state"] = (
                            aux_hidden_states_cpu[i, :seq_len, :].clone().unsqueeze(0)
                        )

                    output_file = self._get_file_path(output_path, current_global_idx)
                    self._save_tensor_async(data_point, output_file)

                del hidden_states_cpu, aux_hidden_states_cpu
                total_processed += len(sample_global_indices)

            del eagle3_data, filtered_batch

            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()
            if batch_idx % 20 == 0:
                gc.collect()

            if tp_size > 1 and batch_idx % self.tp_sync_interval == 0:
                dist.barrier(group=tp_group)

            if self.show_progress:
                progress_bar.set_postfix(
                    {
                        "processed": total_processed,
                        "skipped": total_skipped,
                        "pending_io": (
                            len(self.pending_futures) if is_tp_rank_0() else 0
                        ),
                    }
                )

        # --- REFACTOR: Cleanup is now handled by __exit__ ---
        if self.show_progress:
            print(
                f"\nGeneration loop finished. Processed: {total_processed}, Skipped: {total_skipped}"
            )


def main():
    args = parse_args()
    if args.aux_hidden_states_layers is not None:
        args.aux_hidden_states_layers = [
            int(x) for x in args.aux_hidden_states_layers.split(",")
        ]

    # Initialize distributed environment (TP + DP)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    args.dp_size = dist.get_world_size() // args.tp_size

    # Build target model (with TP)
    target_model_config = AutoConfig.from_pretrained(args.target_model_path)
    target_model, processor = build_target_model(args, target_model_config)

    print_with_rank(
        f"DP Rank {dist.get_rank(get_dp_group())}, TP Rank {dist.get_rank(get_tp_group())}, "
        f"DP Size {dist.get_world_size(get_dp_group())}, TP Size {dist.get_world_size(get_tp_group())}"
    )

    if args.output_path is None:
        args.output_path = os.path.join(
            Path(__file__).parent.parent, "cache", "hidden_states"
        )

    # Load complete dataset
    assert os.path.exists(
        args.data_path
    ), f"Dataset path {args.data_path} does not exist"
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
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        process_group=get_dp_group(),
        is_vlm=args.is_vlm,
    )

    print_with_rank(
        f"DataLoader created for DP Rank {dist.get_rank(get_dp_group())}. "
        f"Number of batches: {len(data_loader)}"
    )

    # Calculate starting index and sample count for current DP rank
    total = len(eagle3_dataset)
    dp_rank = dist.get_rank(get_dp_group())
    dp_size = dist.get_world_size(get_dp_group())

    # Calculate samples per DP rank (handle non-divisible case)
    samples_per_dp = total // dp_size
    remainder = total % dp_size

    # Earlier ranks handle one extra sample if there's a remainder
    if dp_rank < remainder:
        samples_per_dp += 1
        start_idx = dp_rank * samples_per_dp
    else:
        start_idx = dp_rank * samples_per_dp + remainder

    print_with_rank(
        f"DP Rank {dp_rank} will process {samples_per_dp} samples, "
        f"starting from index {start_idx}"
    )

    # Generate hidden states
    try:
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
                output_path=args.output_path,
                start_idx=start_idx,
                samples_per_dp=samples_per_dp,
            )

    finally:
        # The finally block ensures destroy_distributed is always called
        print_with_rank("All hidden states generated or job finished.")
        destroy_distributed()


if __name__ == "__main__":
    main()
