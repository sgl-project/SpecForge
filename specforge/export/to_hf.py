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
    draft_config_path: str,
    output_dir: str,
    *,
    vocab_mapping_path: Optional[str] = None,
    embedding_source: Optional[str] = None,
    embedding_key: str = "model.embed_tokens.weight",
    target_model: Optional[str] = None,
) -> str:
    """Publish canonical config, exact owned weights and standalone HF code."""
    from specforge.checkpoint.draft_dir import write_artifact
    from specforge.checkpoint.formats.hf import bundle_code
    from specforge.modeling.draft.hf_config import SpecForgeDraftConfig

    from .canonical import read_export

    if embedding_key != "model.embed_tokens.weight":
        raise ValueError("custom embedding keys are not part of canonical export")
    if embedding_source is not None:
        raise ValueError(
            "embedding_source is not supported for canonical export; use target_model "
            "(--target-model). Target embeddings remain external."
        )
    checkpoint, state, model_class = read_export(
        checkpoint_path,
        draft_config=draft_config_path,
        target=target_model,
        vocab_mapping=vocab_mapping_path,
    )
    config = SpecForgeDraftConfig(checkpoint.config, **checkpoint.hf_metadata).to_dict()

    def write_code(directory):
        config["auto_map"] = bundle_code(directory, model_class)

    write_artifact(output_dir, config, state, write_code=write_code)
    return output_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=export_to_hf.__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--draft-config", default=None)
    parser.add_argument("--target-model", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocab-mapping", default=None)
    parser.add_argument(
        "--embedding-source",
        default=None,
        help="Unsupported for canonical export; use --target-model instead.",
    )
    parser.add_argument("--embedding-key", default="model.embed_tokens.weight")
    args = parser.parse_args(argv)
    out = export_to_hf(
        args.checkpoint,
        args.draft_config,
        args.output_dir,
        vocab_mapping_path=args.vocab_mapping,
        embedding_source=args.embedding_source,
        embedding_key=args.embedding_key,
        target_model=args.target_model,
    )
    print(f"exported HF draft to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
