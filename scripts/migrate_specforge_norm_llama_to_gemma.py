#!/usr/bin/env python3
# coding=utf-8
"""
migrate_specforge_norm_llama_to_gemma.py
========================================

One-shot migration: rewrite a SpecForge MTP *draft* checkpoint that was
trained under the legacy **Llama-style** RMSNorm parametrization
(``y = w * x_normed``, weight initialised to 1) so that it is loadable by
the new draft code that uses :class:`GemmaRMSNorm`
(``y = (1 + w) * x_normed``, weight initialised to 0).

Why this is needed
------------------
Before commit X (when the draft used ``LlamaRMSNorm``) the stored value
``w_llama`` directly multiplied ``x_normed``.  After switching the draft to
``GemmaRMSNorm`` the inference forward computes ``(1 + w) * x_normed``,
so the weight that yields the same output is::

    w_gemma = w_llama - 1.0

This script rewrites every RMSNorm weight in the draft checkpoint by
``W -= 1.0`` (in fp32, then cast back to the original dtype), preserving
all other tensors verbatim.  After migration the resulting checkpoint
can be:

* `from_pretrained()`-loaded by the new draft code (which expects Gemma
  parametrization) and used to **resume training** without numerical
  discontinuity, OR
* fed directly to the new (default) ``merge_mtp_draft_into_target.py``
  (which assumes both sides are already in Gemma parametrization, i.e.
  no ``-1`` correction needed at merge time).

Usage
-----
    python scripts/migrate_specforge_norm_llama_to_gemma.py \\
        --src    /path/to/old_specforge_draft/epoch_3_step_28000 \\
        --dst    /path/to/new_specforge_draft/epoch_3_step_28000_gemma \\
        --link-mode symlink

    # Dry-run: print what would be rewritten without touching disk.
    python scripts/migrate_specforge_norm_llama_to_gemma.py \\
        --src ... --dst ... --dry-run

Output layout (mirrors merge_mtp_draft_into_target.py)
------------------------------------------------------
    dst/
        model-XXXXX-of-YYYYY.safetensors   # rewritten (real file)
        model-ZZZZZ-of-YYYYY.safetensors -> /abs/path/to/src/.../...  # symlink
        model.safetensors.index.json       # copied verbatim
        config.json, *.json, training_state.pt, ...                  # symlinked

Idempotency
-----------
This script is **NOT idempotent**: running it twice on the same checkpoint
will subtract 1.0 twice and silently corrupt the model.  As a guard the
script refuses to run on a directory whose ``config.json`` already
contains ``"_specforge_norm_convention": "gemma"``, and on success it
writes that marker into the destination ``config.json`` so a second run
is rejected.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# Same set as in merge_mtp_draft_into_target.py.  Keep these two lists in
# sync (or import; we keep a copy here so the migration script has zero
# dependency on the merge script's import path).
NORM_DRAFT_KEYS = frozenset([
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
    "norm.weight",
    "midlayer.input_layernorm.weight",
    "midlayer.post_attention_layernorm.weight",
    "midlayer.self_attn.q_norm.weight",
    "midlayer.self_attn.k_norm.weight",
])

NORM_CONVENTION_MARKER = "_specforge_norm_convention"


def _convert_llama_to_gemma(t: torch.Tensor) -> torch.Tensor:
    """Subtract 1.0 in fp32 then cast back to the original dtype."""
    out = t.to(torch.float32) - 1.0
    return out.to(t.dtype).contiguous()


def load_index(model_dir: Path):
    """Return (weight_map, index_filename_or_None).

    Mirrors ``merge_mtp_draft_into_target.load_index`` so the two scripts
    behave identically on both single-file and sharded checkpoints.
    """
    idx_files = sorted(model_dir.glob("*.index.json"))
    if not idx_files:
        single = model_dir / "model.safetensors"
        if not single.exists():
            raise FileNotFoundError(
                f"No *.index.json and no model.safetensors in {model_dir}"
            )
        with safe_open(single, framework="pt") as f:
            wm = {k: "model.safetensors" for k in f.keys()}
        return wm, None
    if len(idx_files) > 1:
        raise RuntimeError(f"Multiple *.index.json files in {model_dir}: {idx_files}")
    with idx_files[0].open() as f:
        idx = json.load(f)
    return idx["weight_map"], idx_files[0].name


def _place(src: Path, dst: Path, mode: str):
    """Place src at dst using symlink / hardlink / copy."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            os.symlink(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown link mode: {mode}")


def _read_config(src_dir: Path):
    cfg_path = src_dir / "config.json"
    if not cfg_path.exists():
        return None, None
    with cfg_path.open() as f:
        cfg = json.load(f)
    return cfg, cfg_path


def _check_already_gemma(src_dir: Path) -> bool:
    cfg, _ = _read_config(src_dir)
    if cfg is None:
        return False
    return cfg.get(NORM_CONVENTION_MARKER) == "gemma"


def _norm_tensor_stats(t: torch.Tensor):
    f = t.detach().to(torch.float32).flatten()
    return float(f.mean().item()), float(f.std().item()), \
           float(f.min().item()), float(f.max().item())


def main():
    p = argparse.ArgumentParser(
        description=("Rewrite a legacy LlamaRMSNorm SpecForge draft "
                     "checkpoint into the new GemmaRMSNorm parametrization."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--src", required=True,
                   help="Path to the legacy (LlamaRMSNorm) draft checkpoint dir.")
    p.add_argument("--dst", required=True,
                   help="Output dir for the migrated (GemmaRMSNorm) checkpoint.")
    p.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"),
                   default="symlink",
                   help="How to place shards/files NOT touched by the migration "
                        "into the output dir (default: symlink).")
    p.add_argument("--dry-run", action="store_true",
                   help="Only print the migration plan, do not write anything.")
    p.add_argument("--force", action="store_true",
                   help="Bypass the 'already-migrated' guard. NOT recommended.")
    args = p.parse_args()

    src_dir = Path(args.src).resolve()
    dst_dir = Path(args.dst).resolve()
    if src_dir == dst_dir:
        p.error("--src and --dst must differ (this script is not in-place safe).")
    if not src_dir.exists():
        p.error(f"--src does not exist: {src_dir}")

    # ---- Idempotency guard ----
    if _check_already_gemma(src_dir) and not args.force:
        print(f"[migrate] !! {src_dir / 'config.json'} already declares "
              f"{NORM_CONVENTION_MARKER}=gemma; refusing to subtract 1.0 again. "
              f"Pass --force to override.", file=sys.stderr)
        sys.exit(1)

    weight_map, index_filename = load_index(src_dir)
    print(f"[migrate] src = {src_dir}")
    print(f"[migrate] dst = {dst_dir}")
    print(f"[migrate] {len(weight_map)} tensor(s) across "
          f"{len(set(weight_map.values()))} shard(s).")

    # ---- Plan ----
    keys_in_norm_set = sorted(k for k in weight_map.keys() if k in NORM_DRAFT_KEYS)
    if not keys_in_norm_set:
        print("[migrate] !! No keys matching NORM_DRAFT_KEYS were found in the "
              "checkpoint. Nothing to do; aborting to avoid producing a "
              "useless copy.", file=sys.stderr)
        for k in sorted(NORM_DRAFT_KEYS):
            print(f"   - expected: {k}", file=sys.stderr)
        sys.exit(1)

    print(f"[migrate] Will rewrite {len(keys_in_norm_set)} RMSNorm tensor(s):")
    for k in keys_in_norm_set:
        print(f"   - {k}    (shard: {weight_map[k]})")

    shard_to_norm_keys = defaultdict(list)
    for k in keys_in_norm_set:
        shard_to_norm_keys[weight_map[k]].append(k)
    touched_shards = set(shard_to_norm_keys.keys())
    all_shards = set(weight_map.values())
    untouched_shards = sorted(all_shards - touched_shards)
    print(f"[migrate] {len(touched_shards)} shard(s) will be rewritten, "
          f"{len(untouched_shards)} shard(s) will be {args.link_mode}ed.")

    if args.dry_run:
        print("[migrate] --dry-run: not writing anything.")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) Untouched shards: link/copy ----
    for shard in untouched_shards:
        _place((src_dir / shard).resolve(), dst_dir / shard, mode=args.link_mode)
    if untouched_shards:
        print(f"[migrate] linked {len(untouched_shards)} untouched shard(s) "
              f"(mode={args.link_mode}).")

    # ---- 2) Touched shards: read, patch, save ----
    n_total_norm_rewritten = 0
    for shard in sorted(touched_shards):
        src_path = src_dir / shard
        out_path = dst_dir / shard
        tmp_path = out_path.with_name(out_path.name + ".tmp")

        if out_path.exists() or out_path.is_symlink():
            out_path.unlink()

        tensors = {}
        with safe_open(src_path, framework="pt") as f:
            metadata = f.metadata() or {}
            for k in f.keys():
                tensors[k] = f.get_tensor(k)

        n_norm = 0
        for k in shard_to_norm_keys[shard]:
            old_stats = _norm_tensor_stats(tensors[k])
            tensors[k] = _convert_llama_to_gemma(tensors[k])
            new_stats = _norm_tensor_stats(tensors[k])
            print(f"   {k}:  mean {old_stats[0]:+.4f} -> {new_stats[0]:+.4f}  "
                  f"(min {old_stats[2]:+.4f}->{new_stats[2]:+.4f}, "
                  f"max {old_stats[3]:+.4f}->{new_stats[3]:+.4f})")
            n_norm += 1

        save_kwargs = {"metadata": metadata} if metadata else {}
        save_file(tensors, str(tmp_path), **save_kwargs)
        os.replace(tmp_path, out_path)
        n_total_norm_rewritten += n_norm
        print(f"[migrate] {shard}: rewrote {n_norm} norm tensor(s) -> {out_path}")

    # ---- 3) Auxiliary files (config.json + tokenizer.* + index.json + ...) ----
    cfg_obj, cfg_src_path = _read_config(src_dir)

    for entry in sorted(src_dir.iterdir()):
        name = entry.name
        if name.endswith(".safetensors"):
            continue
        dst = dst_dir / name
        if dst.exists() or dst.is_symlink():
            continue
        if name == "config.json":
            # Will be rewritten with the marker further down.
            continue
        if name.endswith(".index.json"):
            # Index is unchanged (key set / shard layout unchanged); copy a
            # real file so the user can freely inspect/edit it.
            shutil.copy2(entry, dst)
        elif entry.is_dir():
            os.symlink(entry.resolve(), dst)
        else:
            _place(entry.resolve(), dst, mode=args.link_mode)

    # ---- 4) config.json: copy + add the convention marker ----
    if cfg_obj is None:
        # Hard to imagine for a save_pretrained() output, but be defensive.
        print("[migrate] WARNING: src has no config.json; cannot write the "
              "idempotency marker. Future re-runs will not be guarded.",
              file=sys.stderr)
    else:
        cfg_obj[NORM_CONVENTION_MARKER] = "gemma"
        cfg_obj.setdefault("_specforge_migration_notes", []).append(
            "Migrated from LlamaRMSNorm (y=w*x) to GemmaRMSNorm "
            "(y=(1+w)*x) by subtracting 1.0 from every RMSNorm.weight."
        )
        with (dst_dir / "config.json").open("w") as f:
            json.dump(cfg_obj, f, indent=2, ensure_ascii=False)
        print(f"[migrate] Wrote {dst_dir / 'config.json'} with "
              f"{NORM_CONVENTION_MARKER}=gemma marker.")

    print()
    print(f"[migrate] DONE. Rewrote {n_total_norm_rewritten} RMSNorm tensor(s) "
          f"across {len(touched_shards)} shard(s).")
    print(f"[migrate] Migrated checkpoint: {dst_dir}")
    print(f"[migrate] Original checkpoint untouched at: {src_dir}")


if __name__ == "__main__":
    main()
