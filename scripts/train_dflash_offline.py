#!/usr/bin/env python3
# coding=utf-8
"""Offline DFlash training from cached target hidden states.

This is the offline counterpart to scripts/train_dflash.py for the DFlash
activation caches produced by scripts/prepare_dflash_hidden_states.py. It runs
single-process training, logs to W&B, saves Hugging Face-compatible draft
checkpoints, and can upload checkpoint folders to the Hub.
"""

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from huggingface_hub import HfApi, create_repo, upload_folder
from specforge.core.dflash import OnlineDFlashModel
from specforge.data.preprocessing import list_local_files
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead


def setup_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def reduce_mean(value: float, device: torch.device, world_size: int) -> float:
    if world_size == 1:
        return value
    tensor = torch.tensor(value, device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float((tensor / world_size).detach().cpu())


def should_skip_batch(local_skip: bool, device: torch.device, world_size: int) -> bool:
    if world_size == 1:
        return local_skip
    flag = torch.tensor(1 if local_skip else 0, device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.SUM)
    return bool(flag.item())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    model = parser.add_argument_group("model")
    model.add_argument("--target-model-path", required=True)
    model.add_argument("--num-draft-layers", type=int, default=1)
    model.add_argument("--block-size", type=int, default=16)
    model.add_argument("--num-anchors", type=int, default=512)
    model.add_argument(
        "--attention-backend",
        choices=["eager", "sdpa", "flex_attention"],
        default="sdpa",
    )
    model.add_argument("--mask-token-id", type=int, default=None)
    model.add_argument("--embedding-key", type=str, default=None)
    model.add_argument("--lm-head-key", type=str, default=None)
    model.add_argument("--trust-remote-code", action="store_true")

    data = parser.add_argument_group("data")
    data.add_argument("--train-hidden-states-path", action="append", required=True)
    data.add_argument("--eval-hidden-states-path", action="append", default=[])
    data.add_argument("--max-train-samples", type=int, default=None)
    data.add_argument("--max-eval-samples", type=int, default=256)
    data.add_argument("--max-length", type=int, default=4096)
    data.add_argument("--shuffle-files", action="store_true")
    data.add_argument("--num-workers", type=int, default=0)
    data.add_argument("--prefetch-factor", type=int, default=2)
    data.add_argument("--pin-memory", action="store_true")

    training = parser.add_argument_group("training")
    training.add_argument("--max-steps", type=int, default=1000)
    training.add_argument("--batch-size", type=int, default=1)
    training.add_argument("--learning-rate", type=float, default=1e-4)
    training.add_argument("--weight-decay", type=float, default=0.0)
    training.add_argument("--max-grad-norm", type=float, default=1.0)
    training.add_argument("--gradient-accumulation-steps", type=int, default=1)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--loss-type", default="dflash")
    training.add_argument("--dpace-alpha", type=float, default=0.5)
    training.add_argument("--loss-decay-gamma", type=float, default=None)

    output = parser.add_argument_group("output")
    output.add_argument("--output-dir", required=True)
    output.add_argument("--save-interval", type=int, default=100)
    output.add_argument("--eval-interval", type=int, default=100)
    output.add_argument("--log-interval", type=int, default=1)
    output.add_argument("--resume-from", type=str, default=None)

    wandb_group = parser.add_argument_group("wandb")
    wandb_group.add_argument("--wandb-project", default=None)
    wandb_group.add_argument("--wandb-entity", default=None)
    wandb_group.add_argument("--wandb-name", default=None)
    wandb_group.add_argument("--wandb-mode", default="online")

    hub = parser.add_argument_group("huggingface hub")
    hub.add_argument("--hf-repo-id", default=None)
    hub.add_argument("--hf-private", action="store_true")
    hub.add_argument("--push-to-hub", action="store_true")
    hub.add_argument("--push-every-save", action="store_true")

    return parser.parse_args()


def _read_manifest(cache_dir: str | Path) -> dict:
    manifest_path = Path(cache_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text())


def _collect_files(
    paths: Iterable[str],
    max_samples: Optional[int],
    seed: int,
    shuffle_files: bool,
) -> list[str]:
    files: list[str] = []
    for path in paths:
        files.extend(list_local_files(path))
    if not files:
        raise FileNotFoundError(f"no .ckpt files under {list(paths)}")
    files = sorted(files)
    if shuffle_files:
        random.Random(seed).shuffle(files)
    if max_samples is not None:
        files = files[:max_samples]
    return files


class DFlashHiddenStateDataset(Dataset):
    def __init__(self, files: list[str], max_length: int):
        self.files = files
        self.max_length = max_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        record = torch.load(self.files[index], map_location="cpu", mmap=True)
        input_ids = record["input_ids"][: self.max_length].long()
        loss_mask = record["loss_mask"][: self.max_length].float().clone()
        hidden = record["selected_hidden_states"][: self.max_length].to(torch.bfloat16)
        if hidden.ndim != 3:
            raise ValueError(
                "selected_hidden_states must be [seq, layers, hidden], "
                f"got {tuple(hidden.shape)}"
            )
        seq_len = min(input_ids.shape[0], loss_mask.shape[0], hidden.shape[0])
        input_ids = input_ids[:seq_len]
        loss_mask = loss_mask[:seq_len]
        hidden = hidden[:seq_len]
        if seq_len > 0:
            loss_mask[-1] = 0
        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "hidden_states": hidden.flatten(start_dim=1),
            "source_path": self.files[index],
        }


def collate_batch(features):
    max_len = max(item["input_ids"].shape[0] for item in features)
    hidden_width = features[0]["hidden_states"].shape[-1]
    input_ids = torch.zeros(len(features), max_len, dtype=torch.long)
    loss_mask = torch.zeros(len(features), max_len, dtype=torch.float32)
    hidden_states = torch.zeros(
        len(features), max_len, hidden_width, dtype=torch.bfloat16
    )
    source_paths = []
    for i, item in enumerate(features):
        seq_len = item["input_ids"].shape[0]
        input_ids[i, :seq_len] = item["input_ids"]
        loss_mask[i, :seq_len] = item["loss_mask"]
        hidden_states[i, :seq_len] = item["hidden_states"]
        source_paths.append(item["source_path"])
    return {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "hidden_states": hidden_states,
        "source_paths": source_paths,
    }


def build_draft_config(args, target_config, selected_layer_ids: list[int]) -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=target_config.vocab_size,
        hidden_size=target_config.hidden_size,
        intermediate_size=getattr(
            target_config, "intermediate_size", 4 * target_config.hidden_size
        ),
        num_hidden_layers=args.num_draft_layers,
        num_attention_heads=target_config.num_attention_heads,
        num_key_value_heads=target_config.num_key_value_heads,
        head_dim=getattr(
            target_config,
            "head_dim",
            target_config.hidden_size // target_config.num_attention_heads,
        ),
        hidden_act=getattr(target_config, "hidden_act", "silu"),
        max_position_embeddings=getattr(target_config, "max_position_embeddings", 4096),
        rms_norm_eps=getattr(target_config, "rms_norm_eps", 1e-6),
        rope_theta=getattr(target_config, "rope_theta", 1000000.0),
        attention_bias=getattr(target_config, "attention_bias", False),
        attention_dropout=getattr(target_config, "attention_dropout", 0.0),
        bos_token_id=getattr(target_config, "bos_token_id", None),
        eos_token_id=getattr(target_config, "eos_token_id", None),
        pad_token_id=getattr(target_config, "pad_token_id", None),
        tie_word_embeddings=getattr(target_config, "tie_word_embeddings", False),
        use_cache=False,
    )
    config.block_size = args.block_size
    config.num_target_layers = getattr(target_config, "num_hidden_layers", None)
    config.layer_types = ["full_attention"] * args.num_draft_layers
    config.max_window_layers = args.num_draft_layers
    config.use_sliding_window = False
    config.sliding_window = None
    config.dflash_config = {
        "mask_token_id": args.mask_token_id,
        "target_layer_ids": selected_layer_ids,
    }
    config._attn_implementation = args.attention_backend
    return config


def validate_manifests(paths: list[str]) -> tuple[list[int], int]:
    manifests = [_read_manifest(path) for path in paths]
    selected_layer_ids = manifests[0]["selected_layer_ids"]
    hidden_size = manifests[0]["hidden_size"]
    for manifest in manifests[1:]:
        if manifest["selected_layer_ids"] != selected_layer_ids:
            raise ValueError("all caches must use the same selected_layer_ids")
        if manifest["hidden_size"] != hidden_size:
            raise ValueError("all caches must use the same hidden_size")
    return selected_layer_ids, hidden_size


def batch_to_device(batch, device):
    return (
        batch["input_ids"].to(device, non_blocking=True),
        batch["hidden_states"].to(device, non_blocking=True),
        batch["loss_mask"].to(device, non_blocking=True),
    )


@torch.no_grad()
def evaluate(model, dataloader, args, device, max_batches: Optional[int] = None):
    model.eval()
    losses = []
    accuracies = []
    loss_tokens = []
    skipped = 0
    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        if batch["loss_mask"].sum().item() < 2 * args.block_size:
            skipped += 1
            continue
        input_ids, hidden_states, loss_mask = batch_to_device(batch, device)
        loss, accuracy = model(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
        )
        losses.append(float(loss.detach().cpu()))
        accuracies.append(float(accuracy.detach().cpu()))
        loss_tokens.append(float(loss_mask.sum().detach().cpu()))
    model.train()
    if not losses:
        return {"eval/skipped": skipped, "eval/batches": 0}
    return {
        "eval/loss": sum(losses) / len(losses),
        "eval/accuracy": sum(accuracies) / len(accuracies),
        "eval/loss_tokens": sum(loss_tokens),
        "eval/skipped": skipped,
        "eval/batches": len(losses),
    }


def copy_modeling_file(save_dir: Path):
    src = Path(__file__).resolve().parent.parent / "specforge" / "modeling" / "draft" / "dflash.py"
    if src.exists():
        shutil.copy(src, save_dir / "dflash.py")


def save_checkpoint(
    save_dir: Path,
    model: OnlineDFlashModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    args,
    metrics: dict,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    unwrap_model(model).draft_model.save_pretrained(save_dir)
    copy_modeling_file(save_dir)
    torch.save(
        {
            "global_step": step,
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "metrics": metrics,
        },
        save_dir / "training_state.pt",
    )
    (save_dir / "offline_train_metrics.json").write_text(json.dumps(metrics, indent=2))


def maybe_push_checkpoint(path: Path, args, step: int):
    if not args.push_to_hub or not args.hf_repo_id:
        return None
    create_repo(args.hf_repo_id, private=args.hf_private, exist_ok=True)
    commit = upload_folder(
        repo_id=args.hf_repo_id,
        folder_path=str(path),
        path_in_repo=path.name,
        commit_message=f"offline dflash checkpoint step {step}",
    )
    return getattr(commit, "commit_url", None)


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    is_main = is_main_process(rank)
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    all_cache_paths = args.train_hidden_states_path + args.eval_hidden_states_path
    selected_layer_ids, hidden_size = validate_manifests(all_cache_paths)

    target_config = AutoConfig.from_pretrained(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    if target_config.hidden_size != hidden_size:
        raise ValueError(
            f"cache hidden_size={hidden_size} but target config hidden_size={target_config.hidden_size}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    if args.mask_token_id is None:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        args.mask_token_id = tokenizer.mask_token_id
    if args.mask_token_id is None or args.mask_token_id >= target_config.vocab_size:
        raise ValueError(
            f"mask_token_id={args.mask_token_id} is not inside vocab_size={target_config.vocab_size}"
        )

    train_files = _collect_files(
        args.train_hidden_states_path,
        args.max_train_samples,
        args.seed,
        args.shuffle_files,
    )
    eval_files = []
    if args.eval_hidden_states_path:
        eval_files = _collect_files(
            args.eval_hidden_states_path,
            args.max_eval_samples,
            args.seed + 1,
            args.shuffle_files,
        )

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "collate_fn": collate_batch,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = True

    train_dataset = DFlashHiddenStateDataset(train_files, args.max_length)
    train_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **loader_kwargs,
    )
    eval_loader = None
    if eval_files and is_main:
        eval_loader = DataLoader(
            DFlashHiddenStateDataset(eval_files, args.max_length),
            batch_size=args.batch_size,
            shuffle=False,
            **loader_kwargs,
        )

    device = torch.device(
        f"cuda:{local_rank}" if world_size > 1 else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type != "cuda":
        raise RuntimeError("offline DFlash training expects a CUDA GPU")

    draft_config = build_draft_config(args, target_config, selected_layer_ids)
    draft_model = DFlashDraftModel(draft_config).to(device=device, dtype=torch.bfloat16)
    draft_model.mask_token_id = args.mask_token_id
    if args.resume_from:
        loaded = DFlashDraftModel.from_pretrained(
            args.resume_from, torch_dtype=torch.bfloat16
        ).to(device)
        draft_model.load_state_dict(loaded.state_dict())
        del loaded

    target_components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path,
        embed_key=args.embedding_key,
        lm_head_key=args.lm_head_key,
        device=str(device),
        dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
    )

    model = OnlineDFlashModel(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        block_size=args.block_size,
        mask_token_id=args.mask_token_id,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
        loss_type=args.loss_type,
        dpace_alpha=args.dpace_alpha,
    ).to(device)
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    run = None
    if args.wandb_project and is_main:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            mode=args.wandb_mode,
            config={
                **vars(args),
                "selected_layer_ids": selected_layer_ids,
                "hidden_size": hidden_size,
                "train_files": len(train_files),
                "eval_files": len(eval_files),
                "world_size": world_size,
                "global_batch_size": args.batch_size * world_size,
            },
        )
        print(f"wandb_url={run.url}", flush=True)

    if args.push_to_hub and args.hf_repo_id and is_main:
        api = HfApi()
        whoami = api.whoami()
        print(f"hf_user={whoami.get('name')}", flush=True)
        create_repo(args.hf_repo_id, private=args.hf_private, exist_ok=True)
        print(f"hf_repo=https://huggingface.co/{args.hf_repo_id}", flush=True)

    metrics_history = []
    step = 0
    skipped = 0
    accum_loss = 0.0
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    while step < args.max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(step)
        for batch in train_loader:
            if step >= args.max_steps:
                break
            local_skip = batch["loss_mask"].sum().item() < 2 * args.block_size
            if should_skip_batch(local_skip, device, world_size):
                skipped += 1
                continue

            micro_step = step + 1
            input_ids, hidden_states, loss_mask = batch_to_device(batch, device)
            loss, accuracy = model(
                input_ids=input_ids,
                hidden_states=hidden_states,
                loss_mask=loss_mask,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {micro_step}: {loss.item()}")

            (loss / args.gradient_accumulation_steps).backward()
            accum_loss += float(loss.detach().cpu())
            grad_norm = None

            if micro_step % args.gradient_accumulation_steps == 0:
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )
                grad_norm = float(grad_norm_tensor.detach().cpu())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            elapsed = max(time.time() - start_time, 1e-6)
            log = {
                "train/loss": reduce_mean(float(loss.detach().cpu()), device, world_size),
                "train/accuracy": reduce_mean(float(accuracy.detach().cpu()), device, world_size),
                "train/loss_tokens": float(loss_mask.sum().detach().cpu()),
                "train/seq_len": int(input_ids.shape[1]),
                "train/skipped": skipped,
                "train/steps_per_second": micro_step / elapsed,
                "train/lr": optimizer.param_groups[0]["lr"],
            }
            if grad_norm is not None:
                log["train/grad_norm"] = reduce_mean(grad_norm, device, world_size)

            if run is not None and is_main and micro_step % args.log_interval == 0:
                run.log(log, step=micro_step)
            if is_main:
                print(json.dumps({"step": micro_step, **log}), flush=True)

            checkpoint_metrics = dict(log)
            if micro_step % args.eval_interval == 0:
                eval_metrics = {}
                if eval_loader is not None and is_main:
                    eval_metrics = evaluate(unwrap_model(model), eval_loader, args, device)
                if world_size > 1:
                    dist.barrier()
                checkpoint_metrics.update(eval_metrics)
                if run is not None and is_main:
                    run.log(eval_metrics, step=micro_step)
                if eval_metrics and is_main:
                    print(json.dumps({"step": micro_step, **eval_metrics}), flush=True)

            if is_main and (micro_step % args.save_interval == 0 or micro_step == args.max_steps):
                ckpt_dir = output_dir / f"step_{micro_step:08d}"
                save_checkpoint(
                    ckpt_dir,
                    model,
                    optimizer,
                    micro_step,
                    args,
                    checkpoint_metrics,
                )
                commit_url = None
                if args.push_every_save or micro_step == args.max_steps:
                    commit_url = maybe_push_checkpoint(ckpt_dir, args, micro_step)
                save_log = {
                    "checkpoint/path": str(ckpt_dir),
                    "checkpoint/pushed": commit_url is not None,
                }
                if commit_url:
                    save_log["checkpoint/commit_url"] = commit_url
                if run is not None:
                    run.log(save_log, step=micro_step)
                print(json.dumps({"step": micro_step, **save_log}), flush=True)

            if is_main:
                metrics_history.append({"step": micro_step, **checkpoint_metrics})
            step = micro_step

    if is_main:
        summary = {
            "completed_steps": step,
            "skipped_batches": skipped,
            "train_files": len(train_files),
            "eval_files": len(eval_files),
            "selected_layer_ids": selected_layer_ids,
            "hidden_size": hidden_size,
            "mask_token_id": args.mask_token_id,
            "world_size": world_size,
            "global_batch_size": args.batch_size * world_size,
            "wandb_url": run.url if run is not None else None,
            "hf_repo": f"https://huggingface.co/{args.hf_repo_id}" if args.hf_repo_id else None,
            "metrics": metrics_history,
        }
        (output_dir / "offline_train_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"summary_path={output_dir / 'offline_train_summary.json'}", flush=True)
        if run is not None:
            run.summary.update(summary)
            run.finish()
    cleanup_distributed()


if __name__ == "__main__":
    main()
