#!/usr/bin/env python3
# coding=utf-8
"""Upload completed DFlash checkpoint directories to the Hugging Face Hub."""

import argparse
import json
import time
from pathlib import Path

from huggingface_hub import create_repo, upload_folder


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def is_complete_checkpoint(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "config.json").exists()
        and (path / "training_state.pt").exists()
        and (path / "offline_train_metrics.json").exists()
    )


def load_uploaded(uploaded_path: Path) -> set[str]:
    if not uploaded_path.exists():
        return set()
    return set(json.loads(uploaded_path.read_text()))


def save_uploaded(uploaded_path: Path, uploaded: set[str]):
    uploaded_path.write_text(json.dumps(sorted(uploaded), indent=2))


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    uploaded_path = run_dir / ".hf_uploaded_checkpoints.json"
    create_repo(args.repo_id, private=args.private, exist_ok=True)

    while True:
        uploaded = load_uploaded(uploaded_path)
        for ckpt_dir in sorted(run_dir.glob("step_*")):
            if ckpt_dir.name in uploaded or not is_complete_checkpoint(ckpt_dir):
                continue
            commit = upload_folder(
                repo_id=args.repo_id,
                folder_path=str(ckpt_dir),
                path_in_repo=ckpt_dir.name,
                commit_message=f"offline dflash checkpoint {ckpt_dir.name}",
            )
            uploaded.add(ckpt_dir.name)
            save_uploaded(uploaded_path, uploaded)
            print(
                json.dumps(
                    {
                        "checkpoint": str(ckpt_dir),
                        "commit_url": getattr(commit, "commit_url", None),
                    }
                ),
                flush=True,
            )
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
