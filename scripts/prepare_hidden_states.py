"""
This script will generate the hidden states for the dataset use transformer as the target model backend.
By generating hidden states in advance, we can avoid:
- the memory overhead of loading target model
- the latency overhead of generating hidden states for each request.

Optimized for lower memory usage and higher efficiency.

Usage:
torchrun --nproc_per_node=8 \
    scripts/prepare_hidden_states.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-aux-hidden-states \
    --data-path ./cache/dataset/sharegpt_train.jsonl \
    --output-path ./cache/hidden_states/sharegpt_train_Llama-3.1-8B-Instruct \
    --chat-template llama3 \
    --max-length 2048 \
    --tp-size 1 \
    --batch-size 32 \
    --num-samples 1000 \
    --output-path ./cache/hidden_states

For pre-formatted data (with chat template already applied), add --is-preformatted:
torchrun --nproc_per_node=8 \
    scripts/prepare_hidden_states.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-aux-hidden-states \
    --data-path ./cache/dataset/preformatted_data.jsonl \
    --output-path ./cache/hidden_states \
    --chat-template llama3 \
    --is-preformatted \
    --max-length 2048
"""

import argparse
import gc
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from specforge.args import SGLangBackendArgs
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


@dataclass
class DataPoint:
    input_ids: torch.Tensor
    loss_mask: torch.Tensor
    hidden_state: torch.Tensor
    aux_hidden_state: Optional[torch.Tensor] = None


def parse_args():
    parser = argparse.ArgumentParser()

    # model-related arguments
    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument(
        "--is-vlm", action="store_true", help="Whether the target model is a VLM"
    )
    model_group.add_argument("--enable-aux-hidden-states", action="store_true")
    model_group.add_argument("--aux-hidden-states-layers", type=str, default=None)

    data_group = parser.add_argument_group("data")
    data_group.add_argument("--data-path", type=str, required=True)
    data_group.add_argument("--max-length", type=int, default=2048)
    data_group.add_argument("--chat-template", type=str, default="llama3")
    data_group.add_argument(
        "--is-preformatted",
        action="store_true",
        help="Whether the input data is preformatted text with the chat template already applied to the conversation messages.",
    )
    data_group.add_argument("--num-samples", type=int, default=None)
    data_group.add_argument("--build-dataset-num-proc", type=int, default=8)

    inference_group = parser.add_argument_group("inference")
    inference_group.add_argument("--tp-size", type=int, default=1)
    inference_group.add_argument("--batch-size", type=int, default=32)

    others_group = parser.add_argument_group("others")
    others_group.add_argument("--cache-dir", type=str, default="./cache")
    others_group.add_argument("--output-path", type=str, default=None)
    others_group.add_argument(
        "--model-download-dir",
        type=str,
        default=None,
        help="The directory to download the target model to",
    )
    others_group.add_argument(
        "--dist-timeout",
        type=int,
        default=2000,
        help="Timeout for collective communication in minutes, default to 2000 so that it does not go timeout",
    )
    others_group.add_argument(
        "--num-io-threads",
        type=int,
        default=4,
        help="Number of threads for async I/O operations",
    )
    others_group.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers for DataLoader"
    )
    others_group.add_argument(
        "--io-queue-size",
        type=int,
        default=50,
        help="Max number of pending I/O futures.",
    )
    others_group.add_argument(
        "--file-group-size",
        type=int,
        default=2000,
        help="Number of files per subdirectory.",
    )

    sglang_group = parser.add_argument_group("sglang")
    SGLangBackendArgs.add_args(sglang_group)
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
        # TODO: replace with sglang
        from transformers import Qwen2_5_VLForConditionalGeneration

        target_model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=args.target_model_path,
                torch_dtype=(
                    model_config.dtype
                    if hasattr(model_config, "dtype")
                    else model_config.torch_dtype
                ),
            )
            .eval()
            .cuda()
        )
    else:
        target_model_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()
        target_model = get_eagle3_target_model(
            pretrained_model_name_or_path=args.target_model_path,
            backend="sglang",  # we set this as the default backend to minimize precision mismatch in training and serving
            torch_dtype=(
                model_config.dtype
                if hasattr(model_config, "dtype")
                else model_config.torch_dtype
            ),
            device="cuda",
            cache_dir=args.model_download_dir,
            **target_model_kwargs,
        )
    # Set auxiliary hidden states layers if specified
    target_model.set_aux_hidden_states_layers(args.aux_hidden_states_layers)

    if args.is_vlm:
        processor = AutoProcessor.from_pretrained(args.target_model_path)
    else:
        processor = None

    return target_model, processor


class HiddenStatesGenerator:
    """
    This is a generator for creating and saving the hidden states based on the target model.
    It includes the following features:
        1. Fixes a potential deadlock in TP > 1 scenarios when a batch is skipped.
        2. Implements a context manager (`with` statement) for robust resource handling.
        3. Makes internal settings (like queue sizes, group sizes) configurable.
        4. Centralizes resource cleanup logic.
    """

    def __init__(
        self,
        target_model,
        enable_aux_hidden_states: bool = True,
        num_io_threads: int = 4,
        io_queue_size: int = 50,
        file_group_size: int = 2000,
    ):
        """
        Args:
            target_model: The model for inference.
            enable_aux_hidden_states: Whether to save auxiliary hidden states.
            num_io_threads: Number of threads for async I/O.
            io_queue_size: Max number of pending I/O futures before cleanup.
            file_group_size: Number of files per subdirectory.
        """
        self.model = target_model
        self.enable_aux_hidden_states = enable_aux_hidden_states

        # --- Configurable parameters ---
        self.num_io_threads = num_io_threads
        self.io_queue_size = io_queue_size
        self.file_group_size = file_group_size

        # progress bar should only shown on TP rank = 0
        self.show_progress = dist.get_rank(get_tp_group()) == 0

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
        dist.barrier()

    def _save_tensor_sync(self, data_point: DataPoint, output_file: str) -> None:
        """
        Save a data point to a file synchronously. If there is any NaN value in the data, this datapoint will be skipped.

        Args:
            data_point (DataPoint): The data point to save.
            output_file (str): The path to the output file.
        """
        if data_point.hidden_state is not None and torch.any(
            torch.isnan(data_point.hidden_state)
        ):
            print(
                f"Warning: NaN found in hidden_state for {output_file}. Skipping save."
            )
            return

        if data_point.aux_hidden_state is not None and torch.any(
            torch.isnan(data_point.aux_hidden_state)
        ):
            print(
                f"Warning: NaN found in aux_hidden_state for {output_file}. Skipping save."
            )
            return

        torch.save(asdict(data_point), output_file)

    def _save_tensor_async(self, data_point: DataPoint, output_file: str) -> None:
        """
        Submit a job to the io_executor to save the data point asynchronously.

        Args:
            data_point (DataPoint): The data point to save.
            output_file (str): The path to the output file.
        """
        assert is_tp_rank_0(), "Only tp_rank=0 should call _save_tensor_async"
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
        """
        This method is to ensure that all submitted jobs are completed.
        """
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
    ) -> None:
        """
        The dataset is organized into groups of files, each group has a folder which contains the files for this group. For example, if the
        file_group_size is 2000, the 0-1999 samples will be saved in the folder "rows_0-2000", the 2000-3999 samples will be saved in the folder "rows_2000-4000", etc.

        Args:
            output_path (str): The path to the output directory.
            start_idx (int): The starting index of the samples to save.
            total_samples (int): The total number of samples to save.

        Returns:
            None
        """
        if not is_tp_rank_0() or total_samples == 0:
            return
        start_group = (start_idx // self.file_group_size) * self.file_group_size
        end_sample_idx = start_idx + total_samples - 1
        end_group = (end_sample_idx // self.file_group_size) * self.file_group_size
        for group_start_idx in range(start_group, end_group + 1, self.file_group_size):
            grouped_subdir = (
                f"rows_{group_start_idx}-{group_start_idx + self.file_group_size}"
            )
            output_dir = os.path.join(output_path, grouped_subdir)
            os.makedirs(output_dir, exist_ok=True)

    def _check_existing_files_batch(
        self, output_path: str, global_indices: List[int]
    ) -> List[bool]:
        """
        A helper function to check if the files for the given global indices exist.

        Args:
            output_path (str): The path to the output directory.
            global_indices (List[int]): The global indices of the samples to check.

        Returns:
            List[bool]: A list of booleans indicating if the files for the given global indices exist.
        """
        if not is_tp_rank_0():
            return [False] * len(global_indices)

        def check_single_file(idx):
            return os.path.exists(self._get_file_path(output_path, idx))

        # Parallel file existence check
        with ThreadPoolExecutor(max_workers=self.num_io_threads) as executor:
            exists = list(executor.map(check_single_file, global_indices))
        return exists

    def _get_file_path(self, output_path: str, idx: int) -> str:
        """
        A helper function to get the standard file path for the data point with the given index.

        Args:
            output_path (str): The path to the output directory.
            idx (int): The global index of the data point.

        Returns:
            str: The file path for the data point.
        """
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
        """
        This version prioritizes minimal CPU RAM usage above all else, even at the cost of performance.
        - It processes samples one-by-one within the tp_rank_0 process.
        - It avoids batching GPU-to-CPU transfers.
        - It ensures only one sample's data is in RAM for I/O at any given time.
        """
        self._prepare_output_dirs(output_path, start_idx, samples_per_dp)

        tp_group = get_tp_group()
        tp_group_ranks = dist.get_process_group_ranks(tp_group)
        tp_rank_0_global = tp_group_ranks[0]
        global_idx = start_idx

        progress_bar = tqdm(
            data_loader,
            disable=(not self.show_progress),
            desc="Generating Hidden States",
            position=dist.get_rank(get_dp_group()),
            leave=True,
        )

        total_skipped, total_processed = 0, 0

        for batch_idx, batch in enumerate(progress_bar):
            batch_size = batch["input_ids"].size(0)
            current_batch_indices = list(range(global_idx, global_idx + batch_size))

            # # Step 1: Synchronize valid indices across TP group
            # we check which files already exist and sync this info across TP ranks
            # if exists, we will skip these samples
            if is_tp_rank_0():
                exists_list = self._check_existing_files_batch(
                    output_path, current_batch_indices
                )
                exists_tensor = torch.tensor(
                    exists_list, dtype=torch.bool, device="cuda"
                )
            else:
                exists_tensor = torch.tensor(
                    [False] * batch_size, dtype=torch.bool, device="cuda"
                )
            dist.broadcast(exists_tensor, src=tp_rank_0_global, group=tp_group)

            # Step 1: TP rank 0 checks which samples need processing
            valid_indices_in_batch = [
                i for i, exists in enumerate(exists_tensor) if not exists
            ]
            sample_global_indices = [
                current_batch_indices[i] for i in valid_indices_in_batch
            ]
            num_valid = len(valid_indices_in_batch)
            total_skipped += batch_size - num_valid

            # Step 2: Filter batch before moving to GPU to save memory
            global_idx += batch_size
            filtered_batch = {
                "input_ids": batch["input_ids"][valid_indices_in_batch],
                "attention_mask": batch["attention_mask"][valid_indices_in_batch],
                "loss_mask": batch["loss_mask"][valid_indices_in_batch],
            }
            del batch
            if num_valid == 0:
                # Data has already been generated, no sample processing, update progress bar.
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
                continue

            filtered_batch_gpu = {
                k: v.cuda(non_blocking=True) for k, v in filtered_batch.items()
            }

            _, _, aux_hidden_states_list, last_hidden_states_list = self.model.extend(
                **filtered_batch_gpu,
                return_last_hidden_states=True,
                return_logits=False,
            )

            del filtered_batch_gpu

            if is_tp_rank_0():
                for i, (
                    current_global_idx,
                    aux_hidden_states,
                    last_hidden_states,
                ) in enumerate(
                    zip(
                        sample_global_indices,
                        aux_hidden_states_list,
                        last_hidden_states_list,
                    )
                ):

                    # Process ONE sample at a time to minimize CPU RAM footprint
                    # 1. Transfer only the required slice for one sample to CPU
                    aux_hidden_states = (
                        aux_hidden_states.cpu().clone().unsqueeze(0)
                        if aux_hidden_states is not None
                        else None
                    )
                    last_hidden_states = (
                        last_hidden_states.cpu().clone().unsqueeze(0)
                        if last_hidden_states is not None
                        else None
                    )
                    data_point = DataPoint(
                        input_ids=filtered_batch["input_ids"][i].clone(),
                        loss_mask=filtered_batch["loss_mask"][i].clone(),
                        hidden_state=last_hidden_states,
                        aux_hidden_state=aux_hidden_states,
                    )

                    # 3. Save asynchronously (the backpressure logic is still crucial)
                    output_file = self._get_file_path(output_path, current_global_idx)
                    self._save_tensor_async(data_point, output_file)

                    # 4. Immediately clean up the single-sample CPU tensors
                    del last_hidden_states, aux_hidden_states

                total_processed += len(sample_global_indices)

            # Clean up the large GPU and CPU batch data
            del aux_hidden_states_list, last_hidden_states_list, filtered_batch

            if batch_idx % 5 == 0:  # Make GC and cache clearing more frequent
                torch.cuda.empty_cache()
                gc.collect()

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

        if self.show_progress:
            print(
                f"\nGeneration loop finished. Processed: {total_processed}, Skipped: {total_skipped}"
            )
        dist.barrier()


def main():
    args = parse_args()
    if args.aux_hidden_states_layers is not None:
        args.aux_hidden_states_layers = [
            int(x) for x in args.aux_hidden_states_layers.split(",")
        ]

    # Initialize distributed environment (TP + DP)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)

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
    cache_params_string = f"{args.data_path}-{args.max_length}-{args.chat_template}-{args.target_model_path}-{args.num_samples}-{args.is_preformatted}"
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
            is_preformatted=args.is_preformatted,
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
