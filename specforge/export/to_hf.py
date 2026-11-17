# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Publish canonical checkpoints with standalone Transformers loading code."""

from __future__ import annotations

import argparse
from typing import Optional


def export_to_hf(
    checkpoint_path: str,
    output_dir: str,
    *,
    target_model: Optional[str] = None,
) -> str:
    """Publish canonical config, exact owned weights and standalone HF code.

    Target embeddings stay external; ``target_model`` only validates bindings.
    """
    from .model import export_draft

    return export_draft(
        checkpoint_path,
        output_dir,
        export_mode="standalone",
        target_model=target_model,
    )


def write_hf(checkpoint, state, model_class, output_dir):
    from specforge.checkpoint.draft_dir import write_artifact
    from specforge.checkpoint.formats.hf import bundle_code
    from specforge.modeling.draft.hf_config import SpecForgeDraftConfig

    config = SpecForgeDraftConfig(checkpoint.config, **checkpoint.hf_metadata).to_dict()

    def write_code(directory):
        config["auto_map"] = bundle_code(directory, model_class)

    write_artifact(output_dir, config, state, write_code=write_code)
    return output_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=export_to_hf.__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-model", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    out = export_to_hf(args.checkpoint, args.output_dir, target_model=args.target_model)
    print(f"exported HF draft to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
