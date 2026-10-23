"""Checkpoint-aligned replay of an immutable online prompt plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ledger_path(cfg) -> Path:
    deployment = cfg.deployment.disaggregated
    return Path(
        os.environ.get("DISAGG_DB")
        or Path(deployment.consumer_state_dir or deployment.control_dir)
        / "consumer.sqlite"
    )


def _read_ledger(path: Path):
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        # Keep the marker and ack set in one SQLite read snapshot.
        db.execute("BEGIN")
        marker = dict(db.execute("SELECT k, v FROM marker"))
        ids = {row[0] for row in db.execute("SELECT sample_id FROM acked")}
        committed = db.execute("SELECT count(*) FROM committed").fetchone()[0]
    return marker, ids, committed


def validate_replay_ids(
    sample_ids, *, run_id, prompt_count, prompt_epochs, prompt_epoch_offset
):
    if prompt_epochs == 1 and prompt_epoch_offset == 0:
        raise ValueError(
            "replay requires epoch-tagged IDs; legacy single-epoch plans are unsupported"
        )
    task_ids = set()
    prefix = run_id + ":"
    for sample_id in sample_ids:
        task_id = sample_id.removeprefix(prefix)
        match = re.fullmatch(r"epoch(\d{4,})-prompt(\d{12,})", task_id)
        if (
            not sample_id.startswith(prefix)
            or match is None
            or not prompt_epoch_offset
            <= int(match[1])
            < prompt_epoch_offset + prompt_epochs
            or not 0 <= int(match[2]) < prompt_count
            or task_id != f"epoch{int(match[1]):04d}-prompt{int(match[2]):012d}"
        ):
            raise ValueError(
                f"replay exclusion is outside the immutable plan: {sample_id}"
            )
        task_ids.add(task_id)
    return task_ids


def _plan_contract(cfg):
    if cfg.deployment.mode != "disaggregated" or not cfg.data.prompts_index_path:
        raise ValueError(
            "replay requires disaggregated training with data.prompts_index_path"
        )
    if (
        cfg.training.tp_size != 1
        or cfg.training.sp_ulysses_size != 1
        or cfg.training.sp_ring_size != 1
    ):
        raise ValueError("replay currently requires pure data parallel training")
    index_path = Path(cfg.data.prompts_index_path)
    index = json.loads(index_path.read_text())
    count = index["records"]
    if cfg.data.max_prompts:
        count = min(count, cfg.data.max_prompts)
    trainer = cfg.deployment.trainer
    return {
        "version": 1,
        "run_id": cfg.run_id,
        "corpus_records": count,
        "corpus_sha256": index["source_sha256"],
        "index_sha256": _sha256(index_path),
        "prompt_epochs": cfg.training.num_epochs,
        "prompt_epoch_offset": cfg.training.prompt_epoch_offset,
        "prompt_seed": (
            cfg.training.seed
            if cfg.training.prompt_seed is None
            else cfg.training.prompt_seed
        ),
        "global_batch": cfg.training.batch_size
        * cfg.training.accumulation_steps
        * trainer.nnodes
        * trainer.nproc_per_node,
    }


def _validate_alignment(manifest, marker, ids):
    step = manifest["checkpoint_step"]
    if type(step) is not int or step < 0:
        raise ValueError("replay checkpoint_step must be a nonnegative integer")
    if marker != {"global_step": str(step), "optimizer_durable": "true"}:
        raise ValueError(
            "consumer ledger is not aligned with the durable checkpoint step"
        )
    if (
        len(ids) != manifest["acked_count"]
        or len(ids) != step * manifest["global_batch"]
    ):
        raise ValueError(
            "acknowledged sample count does not match checkpoint step and global batch"
        )
    validate_replay_ids(
        ids,
        run_id=manifest["run_id"],
        prompt_count=manifest["corpus_records"],
        prompt_epochs=manifest["prompt_epochs"],
        prompt_epoch_offset=manifest["prompt_epoch_offset"],
    )


def prepare_replay_snapshot(cfg, checkpoint, destination):
    """Copy a stopped, checkpoint-aligned ledger and discard stale feature refs.

    The supplied configuration must describe the original immutable prompt plan.
    This does not roll a ledger back or repartition optimizer/RNG state.
    """
    import torch

    from specforge.training.checkpoint import CheckpointManager

    manifest = _plan_contract(cfg)
    checkpoint = Path(CheckpointManager.resolve_resume_dir(str(checkpoint))).resolve()
    state = torch.load(
        checkpoint / "training_state.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    trainer = cfg.deployment.trainer
    for key, expected in {
        "run_id": cfg.run_id,
        "world_size": trainer.nnodes * trainer.nproc_per_node,
        "batch_size": cfg.training.batch_size,
        "accumulation_steps": cfg.training.accumulation_steps,
    }.items():
        if state.get(key) != expected:
            raise ValueError(
                f"checkpoint {key} does not match the supplied configuration"
            )
    checkpoint_files = ["training_state.pt"] + [
        f"training_state_rank{rank}.pt" for rank in range(state["world_size"])
    ]
    checkpoint_hashes = {name: _sha256(checkpoint / name) for name in checkpoint_files}
    for name in checkpoint_files[1:]:
        rank_state = torch.load(
            checkpoint / name, map_location="cpu", weights_only=False, mmap=True
        )
        if not rank_state.get("rng") or (
            rank_state.get("optimizer") is None
            and state.get("replicated_optimizer_state") is None
        ):
            raise ValueError(f"checkpoint is missing optimizer or RNG state: {name}")
    source = _ledger_path(cfg)
    marker, ids, _ = _read_ledger(source)
    manifest.update(
        checkpoint_step=state["global_step"],
        acked_count=len(ids),
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=checkpoint_hashes,
    )
    _validate_alignment(manifest, marker, ids)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    target_path = destination / "consumer.sqlite"
    with closing(
        sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)
    ) as old:
        with closing(sqlite3.connect(target_path)) as target:
            old.backup(target)
            target.execute("DELETE FROM committed")
            target.commit()
    copied_marker, copied_ids, _ = _read_ledger(target_path)
    _validate_alignment(manifest, copied_marker, copied_ids)
    if copied_ids != ids:
        raise ValueError(
            "source ledger changed during snapshot; stop the run before retrying"
        )
    payload = ("".join(sample_id + "\n" for sample_id in sorted(ids))).encode()
    (destination / "acked-ids.txt").write_bytes(payload)
    manifest.update(
        acked_ids_path="acked-ids.txt",
        acked_ids_sha256=hashlib.sha256(payload).hexdigest(),
    )
    manifest_path = destination / "replay.json"
    temporary_manifest = destination / "replay.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary_manifest.replace(manifest_path)
    return manifest_path


def load_replay_exclusions(path, cfg, prompt_count=None):
    """Validate the replay manifest, checkpoint files, and fresh consumer ledger."""
    from specforge.training.checkpoint import CheckpointManager

    path = Path(path)
    manifest = json.loads(path.read_text())
    for key, expected in _plan_contract(cfg).items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"replay manifest {key} does not match the current configuration"
            )
    if prompt_count is not None and prompt_count != manifest["corpus_records"]:
        raise ValueError("prepared prompt count differs from the immutable replay plan")
    if cfg.training.role == "consumer" and not cfg.training.resume_from:
        raise ValueError(
            "consumer replay requires training.resume_from with full optimizer state"
        )
    checkpoint = Path(
        CheckpointManager.resolve_resume_dir(
            cfg.training.resume_from or manifest["checkpoint_path"]
        )
    )
    for name, expected in manifest["checkpoint_sha256"].items():
        if Path(name).name != name or _sha256(checkpoint / name) != expected:
            raise ValueError(f"replay checkpoint digest mismatch: {name}")
    payload = (path.parent / manifest["acked_ids_path"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["acked_ids_sha256"]:
        raise ValueError("replay acknowledged-ID digest mismatch")
    lines = payload.decode().splitlines()
    ids = set(lines)
    if len(lines) != len(ids):
        raise ValueError("replay acknowledged IDs contain duplicates")
    marker, ledger_ids, committed = _read_ledger(_ledger_path(cfg))
    _validate_alignment(manifest, marker, ids)
    if ledger_ids != ids or committed:
        raise ValueError(
            "replay requires the exact acknowledged IDs and a fresh ledger without committed refs"
        )
    return ids


def main():
    from specforge.config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    print(
        prepare_replay_snapshot(
            load_config(args.config), args.checkpoint, args.destination
        )
    )


if __name__ == "__main__":
    main()
