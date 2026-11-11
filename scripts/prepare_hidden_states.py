"""Prepare offline features using a SGLang capture plugin server and Mooncake.

Start the capture service separately (see examples/offline/README.md), then:

    python scripts/prepare_hidden_states.py \
        --server-urls http://127.0.0.1:30000 \
        --target-model-path Qwen/Qwen3-8B \
        --data-path ./cache/dataset/sharegpt_train.jsonl \
        --output-path ./cache/hidden_states/sharegpt_qwen3-8b \
        --chat-template qwen --strategy eagle3

Optional torchrun workers shard samples on CPU; model TP belongs to the server.
"""

import argparse
import gzip
import hashlib
import os
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Callable, Mapping, Optional

import torch
import torch.distributed as dist
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig

from specforge.algorithms.common.providers import OfflineCaptureLayout
from specforge.application import resolve_offline_capture
from specforge.config import Config
from specforge.data.preprocessing import (
    build_eagle3_dataset,
    generate_vocab_mapping_file,
)
from specforge.offline_capture import OfflineCaptureClient
from specforge.utils import (
    load_tokenizer,
    print_args_with_dots,
    safe_conversations_generator,
)


@dataclass(frozen=True)
class OfflineCapturePlan:
    """Resolved algorithm-owned feature schema for one preparation run."""

    strategy: str
    draft_config: object
    capture_method: str
    capture_layers: tuple[int, ...]
    layout: OfflineCaptureLayout
    loss_mask_filter: Optional[Callable[[object], bool]]


def parse_args():
    parser = argparse.ArgumentParser()

    # model-related arguments
    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument(
        "--strategy",
        type=str,
        default="eagle3",
        help="Offline draft strategy (default: eagle3)",
    )
    draft_group = model_group.add_mutually_exclusive_group()
    draft_group.add_argument(
        "--draft", help="Canonical draft YAML file (including presets)"
    )
    draft_group.add_argument(
        "--draft-model-config",
        type=str,
        default=None,
        help=(
            "Draft config used to resolve capture layers; required by strategies "
            "without target-derived defaults"
        ),
    )
    model_group.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading models",
    )
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
    inference_group.add_argument("--server-urls", nargs="+", required=True)
    inference_group.add_argument("--capture-timeout", type=float, default=300.0)
    inference_group.add_argument("--batch-size", type=int, default=32)

    others_group = parser.add_argument_group("others")
    others_group.add_argument("--cache-dir", type=str, default="./cache")
    others_group.add_argument("--output-path", type=str, default=None)
    others_group.add_argument(
        "--dist-timeout",
        type=int,
        default=2000,
        help="Timeout for collective communication in minutes, default to 2000 so that it does not go timeout",
    )
    others_group.add_argument(
        "--num-io-threads",
        type=int,
        default=None,
        help="Number of threads for async I/O operations (default: all of CPU cores).",
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
    others_group.add_argument(
        "--compress",
        action="store_true",
        help="Compress hidden state files on disk (gzip).",
    )
    others_group.add_argument(
        "--compression-level",
        type=int,
        default=6,
        help="Gzip compression level (1-9).",
    )

    args = parser.parse_args()
    for name in ("batch_size", "io_queue_size", "file_group_size", "capture_timeout"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def _resolve_draft_vocab_size(draft_config: object) -> int:
    """Read the vocabulary size from the canonical resolved draft config."""

    missing = object()
    value = getattr(draft_config, "draft_vocab_size", missing)
    if value is missing:
        value = getattr(draft_config, "vocab_size", None)

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            "resolved draft model config must define a positive "
            f"draft_vocab_size (or vocab_size fallback), got {value!r}"
        )
    return value


def _generate_shared_vocab_mapping(
    dataset,
    *,
    output_path: str,
    target_vocab_size: int,
    draft_vocab_size: int,
) -> str:
    """Generate one mapping on global rank 0 and propagate failures to peers."""

    mapping_dir = os.path.join(output_path, "vocab_mapping")
    mapping_path = os.path.join(mapping_dir, "vocab_mapping.pt")
    result = [None]
    if _rank() == 0:
        temporary_path = None
        try:
            temporary_key = f".vocab_mapping.{uuid.uuid4().hex}"
            temporary_path = generate_vocab_mapping_file(
                dataset=dataset,
                target_vocab_size=target_vocab_size,
                draft_vocab_size=draft_vocab_size,
                cache_dir=mapping_dir,
                cache_key=temporary_key,
            )
            mapping = torch.load(
                temporary_path,
                map_location="cpu",
                weights_only=True,
            )
            d2t = mapping.get("d2t")
            t2d = mapping.get("t2d")
            if not isinstance(d2t, torch.Tensor) or tuple(d2t.shape) != (
                draft_vocab_size,
            ):
                raise ValueError(
                    f"{temporary_path} has invalid d2t shape; expected "
                    f"({draft_vocab_size},), got {getattr(d2t, 'shape', None)}"
                )
            if not isinstance(t2d, torch.Tensor) or tuple(t2d.shape) != (
                target_vocab_size,
            ):
                raise ValueError(
                    f"{temporary_path} has invalid t2d shape; expected "
                    f"({target_vocab_size},), got {getattr(t2d, 'shape', None)}"
                )
            os.replace(temporary_path, mapping_path)
            temporary_path = None
            result[0] = {"path": mapping_path}
        except BaseException as exc:
            result[0] = {
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            if temporary_path is not None:
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass

    if dist.is_initialized():
        dist.broadcast_object_list(result, src=0)
    if result[0] is None or "error" in result[0]:
        reason = "rank 0 did not publish a result"
        if result[0] is not None:
            reason = result[0]["error"]
        raise RuntimeError(f"failed to generate vocabulary mapping: {reason}")
    if result[0]["path"] != mapping_path:
        raise RuntimeError(
            "vocabulary mapping generator returned an unexpected path: "
            f"{result[0]['path']} != {mapping_path}"
        )
    return mapping_path


def resolve_offline_capture_plan(
    args: argparse.Namespace,
    target_config=None,
) -> OfflineCapturePlan:
    """Resolve capture layers and output keys through training composition."""

    strategy = getattr(args, "strategy", "eagle3")
    model = {
        "target_model_path": args.target_model_path,
        "draft_model_config": getattr(args, "draft_model_config", None),
        "trust_remote_code": args.trust_remote_code,
        "cache_dir": getattr(args, "cache_dir", None),
    }
    cfg = Config(
        model=model,
        data={
            "hidden_states_path": args.output_path or "__offline_capture__",
            "max_length": args.max_length,
        },
        draft=getattr(args, "draft", None),
        training=(
            {"attention_backend": "eager"}
            if getattr(args, "draft", None)
            else {"strategy": strategy}
        ),
    )
    resolved = resolve_offline_capture(cfg, target_config=target_config)
    return OfflineCapturePlan(
        strategy=resolved.run.algorithm.name,
        draft_config=resolved.draft_config,
        capture_method=resolved.capture_method,
        capture_layers=resolved.capture_layers,
        layout=resolved.layout,
        loss_mask_filter=resolved.loss_mask_filter,
    )


def build_capture_client(args, model_config, plan):
    text_config = getattr(model_config, "text_config", model_config)
    hidden_size = (
        text_config["hidden_size"]
        if isinstance(text_config, dict)
        else text_config.hidden_size
    )
    return OfflineCaptureClient(
        args.server_urls,
        target_model_path=args.target_model_path,
        capture_layers=plan.capture_layers,
        layout=plan.layout,
        target_hidden_size=hidden_size,
        timeout_s=args.capture_timeout,
    )


def _rank():
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


@contextmanager
def rank_0_priority():
    if dist.is_initialized() and _rank() != 0:
        dist.barrier()
    try:
        yield
    finally:
        if dist.is_initialized() and _rank() == 0:
            dist.barrier()


def print_with_rank(message):
    print(f"[capture worker {_rank()}] {message}", flush=True)


def sample_range(total, rank, world_size):
    """Contiguous, non-overlapping ranges, without sampler padding/duplicates."""
    count, remainder = divmod(total, world_size)
    start = rank * count + min(rank, remainder)
    return start, start + count + (rank < remainder)


def _collate_rows(rows):
    # Keep requests unpadded and independent of client batch/worker counts.
    return rows


def _capture_input(row):
    ids = torch.as_tensor(row["input_ids"]).reshape(-1)
    mask = torch.as_tensor(row["attention_mask"]).reshape(-1)
    loss = torch.as_tensor(row["loss_mask"]).reshape(-1)
    if ids.numel() != mask.numel() or ids.numel() != loss.numel():
        raise ValueError("input_ids, attention_mask and loss_mask must align")
    length = int(mask.sum())
    if (
        length < 1
        or not bool((mask[:length] == 1).all())
        or not bool((mask[length:] == 0).all())
    ):
        raise ValueError(
            "offline capture requires a nonempty prompt with right padding only"
        )
    return {"input_ids": ids[:length].clone(), "loss_mask": loss[:length].clone()}


class HiddenStatesGenerator:
    """Read plugin features on CPU and atomically persist the existing schema."""

    def __init__(
        self,
        capture_client,
        capture_layout=None,
        *,
        num_io_threads=None,
        io_queue_size=50,
        file_group_size=2000,
        compress=False,
        compression_level=6,
    ):
        if io_queue_size < 1 or file_group_size < 1:
            raise ValueError("I/O queue and file group sizes must be positive")
        self.client = capture_client
        self.capture_layout = capture_layout or OfflineCaptureLayout(
            capture_method="eagle3",
            aux_feature="aux_hidden_state",
            last_hidden_feature="hidden_state",
            passthrough=(("input_ids", "input_ids"), ("loss_mask", "loss_mask")),
        )
        self.num_io_threads = num_io_threads
        self.io_queue_size = io_queue_size
        self.file_group_size = file_group_size
        self.compress = compress
        self.compression_level = compression_level
        self.file_extension = ".ckpt.gz" if compress else ".ckpt"
        self.show_progress = _rank() == 0
        self.io_executor = None
        self.pending_futures = []

    def __enter__(self):
        self.io_executor = ThreadPoolExecutor(max_workers=self.num_io_threads)
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self._wait_all_saves()
        finally:
            self.io_executor.shutdown(wait=True)
            self.io_executor = None
            self.pending_futures.clear()

    def _wait_all_saves(self):
        pending, self.pending_futures = self.pending_futures, []
        if pending:
            # All writes must stop before the capture context releases features,
            # including when an earlier future failed.
            wait(pending)
            for future in pending:
                future.result()

    def _wait_for_io_slot(self):
        if len(self.pending_futures) >= self.io_queue_size:
            self.pending_futures.pop(0).result()

    def _save_tensor_sync(self, record: Mapping[str, torch.Tensor], output_file):
        persisted = dict(record)
        for name, tensor in persisted.items():
            if not torch.is_tensor(tensor):
                raise TypeError(f"offline feature {name!r} must be a tensor")
            if (torch.is_floating_point(tensor) or torch.is_complex(tensor)) and bool(
                torch.isnan(tensor).any()
            ):
                print(f"Warning: NaN in {name} for {output_file}; skipping save.")
                return
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
            ) as output:
                temporary = output.name
                if self.compress:
                    with gzip.GzipFile(
                        fileobj=output, mode="wb", compresslevel=self.compression_level
                    ) as zipped:
                        torch.save(persisted, zipped)
                else:
                    torch.save(persisted, output)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                os.unlink(temporary)

    def _save_tensor_async(self, record, output_file):
        self._wait_for_io_slot()
        self.pending_futures.append(
            self.io_executor.submit(self._save_tensor_sync, record, output_file)
        )

    def _get_file_path(self, output_path, idx, extension=None):
        group = (idx // self.file_group_size) * self.file_group_size
        return str(
            Path(output_path)
            / f"rows_{group}-{group + self.file_group_size}"
            / f"data_{idx}{extension or self.file_extension}"
        )

    def _check_existing_files_batch(self, output_path, indices):
        return [
            any(
                Path(self._get_file_path(output_path, idx, extension=ext)).is_file()
                for ext in (".ckpt", ".ckpt.gz")
            )
            for idx in indices
        ]

    def generate(self, data_loader, output_path, start_idx=0):
        global_idx = start_idx
        for batch in tqdm(
            data_loader,
            disable=not self.show_progress,
            desc="Capturing offline features",
        ):
            indices = list(range(global_idx, global_idx + len(batch)))
            global_idx += len(batch)
            exists = self._check_existing_files_batch(output_path, indices)
            selected = [
                (idx, row) for idx, row, done in zip(indices, batch, exists) if not done
            ]
            if not selected:
                continue
            sample_ids = [idx for idx, _ in selected]
            inputs = [_capture_input(row) for _, row in selected]
            with self.client.capture_rows(
                [row["input_ids"].tolist() for row in inputs], sample_ids=sample_ids
            ) as captured:
                if len(captured) != len(inputs):
                    raise RuntimeError(
                        "capture server returned an unexpected row count"
                    )
                try:
                    for idx, source, tensors in zip(sample_ids, inputs, captured):
                        record = self.capture_layout.materialize({**source, **tensors})
                        self._save_tensor_async(
                            record, self._get_file_path(output_path, idx)
                        )
                finally:
                    self._wait_all_saves()


def main():
    args = parse_args()
    if args.num_io_threads is None:
        cpu_cores = os.cpu_count() or 1
        args.num_io_threads = max(1, cpu_cores)
    if args.output_path is None:
        args.output_path = os.path.join(
            Path(__file__).parent.parent, "cache", "hidden_states"
        )

    if args.draft:
        from specforge.checkpoint.resolve import read_target_config

        target_model_config = read_target_config(
            args.target_model_path, cache_dir=args.cache_dir
        )
    else:
        target_model_config = AutoConfig.from_pretrained(
            args.target_model_path,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
        )
    capture_plan = resolve_offline_capture_plan(args, target_model_config)
    draft_vocab_size = _resolve_draft_vocab_size(capture_plan.draft_config)

    # Client workers use CPU collectives; target TP is owned by SGLang.
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("gloo", timeout=timedelta(minutes=args.dist_timeout))
    try:
        print_args_with_dots(args)
        if _rank() == 0:
            print("Resolved draft model config:")
            if args.draft:
                import json

                from specforge.config.draft.codec import encode_config

                print(json.dumps(encode_config(capture_plan.draft_config), indent=2))
            else:
                print(capture_plan.draft_config.to_json_string(use_diff=False))
        print_with_rank(
            f"Resolved {capture_plan.strategy} offline "
            f"capture method {capture_plan.capture_method!r}, layers: "
            f"{list(capture_plan.capture_layers)}"
        )

        target_text_config = getattr(
            target_model_config, "text_config", target_model_config
        )
        target_vocab_size = int(
            target_text_config["vocab_size"]
            if isinstance(target_text_config, dict)
            else target_text_config.vocab_size
        )
        if draft_vocab_size > target_vocab_size:
            raise ValueError(
                f"draft_vocab_size={draft_vocab_size} exceeds target "
                f"vocab_size={target_vocab_size}"
            )

        # Load complete dataset
        assert os.path.exists(
            args.data_path
        ), f"Dataset path {args.data_path} does not exist"

        with rank_0_priority():
            print_with_rank("Loading/building dataset cache...")
            dataset = Dataset.from_generator(
                generator=safe_conversations_generator,
                gen_kwargs={"file_path": args.data_path},
                cache_dir=os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "cache",
                    "hf_dataset",
                ),
                num_proc=min(args.build_dataset_num_proc, 32),
            )
        if args.num_samples is not None and capture_plan.loss_mask_filter is None:
            dataset = dataset.select(range(min(args.num_samples, len(dataset))))
        # Tokenizer and cache key
        tokenizer = load_tokenizer(
            args.target_model_path, trust_remote_code=args.trust_remote_code
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
                is_preformatted=args.is_preformatted,
                num_proc=args.build_dataset_num_proc,
                loss_mask_filter=capture_plan.loss_mask_filter,
            )
        if capture_plan.loss_mask_filter is not None and args.num_samples is not None:
            eagle3_dataset = eagle3_dataset.select(
                range(min(args.num_samples, len(eagle3_dataset)))
            )
        if not len(eagle3_dataset):
            raise ValueError(
                f"no samples satisfy {capture_plan.strategy} training eligibility"
            )
        print_with_rank(f"Dataset prepared with {len(eagle3_dataset)} samples.")

        vocab_mapping_path = _generate_shared_vocab_mapping(
            eagle3_dataset,
            output_path=args.output_path,
            target_vocab_size=target_vocab_size,
            draft_vocab_size=draft_vocab_size,
        )
        if _rank() == 0:
            print(f"Vocabulary mapping ready at {vocab_mapping_path}")

        start_idx, end_idx = sample_range(len(eagle3_dataset), _rank(), _world_size())
        data_loader = DataLoader(
            eagle3_dataset.select(range(start_idx, end_idx)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            collate_fn=_collate_rows,
            multiprocessing_context="spawn" if args.num_workers > 0 else None,
        )
        print_with_rank(f"Capturing samples [{start_idx}, {end_idx})")

        with (
            build_capture_client(args, target_model_config, capture_plan) as client,
            HiddenStatesGenerator(
                client,
                capture_layout=capture_plan.layout,
                num_io_threads=args.num_io_threads,
                io_queue_size=args.io_queue_size,
                file_group_size=args.file_group_size,
                compress=args.compress,
                compression_level=args.compression_level,
            ) as hidden_states_generator,
        ):

            # Generate hidden states
            hidden_states_generator.generate(
                data_loader,
                output_path=args.output_path,
                start_idx=start_idx,
            )

    finally:
        print_with_rank("All hidden states generated or job finished.")
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
