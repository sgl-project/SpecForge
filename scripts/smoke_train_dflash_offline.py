#!/usr/bin/env python3
# coding=utf-8
"""One-GPU DFlash training smoke test from cached hidden states.

This intentionally avoids the online target forward used by train_dflash.py.
It loads the DFlash activation cache written by prepare_dflash_hidden_states.py,
flattens the preserved layer axis, and runs a few real DFlash loss/optimizer
steps. The goal is fast validation of cache compatibility and training plumbing,
not a production training loop.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from specforge.core.dflash import OnlineDFlashModel
from specforge.data.preprocessing import list_local_files
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument(
        "--hidden-states-path",
        action="append",
        required=True,
        help="Activation cache directory. May be passed multiple times.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-draft-layers", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-anchors", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--attention-backend",
        choices=["eager", "sdpa", "flex_attention"],
        default="sdpa",
    )
    parser.add_argument("--loss-type", default="dflash")
    parser.add_argument("--dpace-alpha", type=float, default=0.5)
    parser.add_argument("--loss-decay-gamma", type=float, default=None)
    parser.add_argument("--mask-token-id", type=int, default=None)
    parser.add_argument("--embedding-key", type=str, default=None)
    parser.add_argument("--lm-head-key", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def _read_manifest(cache_dir: str | Path) -> dict:
    manifest_path = Path(cache_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text())


def _collect_files(paths: Iterable[str], max_samples: int, seed: int) -> list[str]:
    files: list[str] = []
    for path in paths:
        files.extend(list_local_files(path))
    if not files:
        raise FileNotFoundError(f"no .ckpt files under {list(paths)}")
    rng = random.Random(seed)
    rng.shuffle(files)
    return files[:max_samples]


class DFlashHiddenStateDataset(Dataset):
    def __init__(self, files: list[str], max_length: int, block_size: int):
        self.files = files
        self.max_length = max_length
        self.block_size = block_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        record = torch.load(self.files[index], map_location="cpu", mmap=True)
        input_ids = record["input_ids"][: self.max_length].long()
        loss_mask = record["loss_mask"][: self.max_length].float().clone()
        hidden = record["selected_hidden_states"][: self.max_length].to(torch.bfloat16)
        if hidden.ndim != 3:
            raise ValueError(
                f"selected_hidden_states must be [seq, layers, hidden], got {tuple(hidden.shape)}"
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
        intermediate_size=getattr(target_config, "intermediate_size", 4 * target_config.hidden_size),
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


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    manifests = [_read_manifest(path) for path in args.hidden_states_path]
    selected_layer_ids = manifests[0]["selected_layer_ids"]
    hidden_size = manifests[0]["hidden_size"]
    for manifest in manifests[1:]:
        if manifest["selected_layer_ids"] != selected_layer_ids:
            raise ValueError("all caches must use the same selected_layer_ids")
        if manifest["hidden_size"] != hidden_size:
            raise ValueError("all caches must use the same hidden_size")

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

    files = _collect_files(args.hidden_states_path, args.max_samples, args.seed)
    dataset = DFlashHiddenStateDataset(files, args.max_length, args.block_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("DFlash smoke training expects a CUDA GPU")

    draft_config = build_draft_config(args, target_config, selected_layer_ids)
    draft_model = DFlashDraftModel(draft_config).to(device=device, dtype=torch.bfloat16)
    draft_model.mask_token_id = args.mask_token_id

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
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate
    )

    step = 0
    skipped = 0
    metrics = []
    for batch in dataloader:
        if step >= args.max_steps:
            break
        if batch["loss_mask"].sum().item() < 2 * args.block_size:
            skipped += 1
            continue

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        loss_mask = batch["loss_mask"].to(device, non_blocking=True)
        hidden_states = batch["hidden_states"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss, accuracy = model(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        row = {
            "step": step + 1,
            "loss": float(loss.detach().cpu()),
            "accuracy": float(accuracy.detach().cpu()),
            "grad_norm": float(grad_norm.detach().cpu()),
            "seq_len": int(input_ids.shape[1]),
            "loss_tokens": float(loss_mask.sum().detach().cpu()),
        }
        print(json.dumps(row), flush=True)
        metrics.append(row)
        step += 1

    if step == 0:
        raise RuntimeError(f"no training steps completed; skipped={skipped}")

    summary = {
        "completed_steps": step,
        "skipped_batches": skipped,
        "num_files": len(files),
        "selected_layer_ids": selected_layer_ids,
        "hidden_size": hidden_size,
        "mask_token_id": args.mask_token_id,
        "metrics": metrics,
    }
    summary_path = Path(args.output_dir) / "offline_dflash_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
