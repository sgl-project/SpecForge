# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Export a DataFlow training checkpoint to an HF-format draft directory.

The output loads back through ``AutoEagle3DraftModel.from_pretrained`` (e.g. to
finetune from it). Embeddings are absent by design — training reloads them from
the target via ``load_embedding``.
"""

from __future__ import annotations

import argparse
from typing import Optional

from specforge.export.checkpoint_io import materialize_draft, resolve_training_state


def export_to_hf(
    checkpoint_path: str,
    draft_config_path: str,
    output_dir: str,
    *,
    vocab_mapping_path: Optional[str] = None,
    embedding_source: Optional[str] = None,
    embedding_key: str = "model.embed_tokens.weight",
) -> str:
    """Write the checkpoint's draft as an HF directory; returns ``output_dir``.

    The exported directory is SELF-CONTAINED (reloads via ``from_pretrained``
    with no missing keys). Draft checkpoints deliberately exclude the frozen
    embedding, so it must come from somewhere real: pass ``embedding_source``
    (the target model path/dir — the embedding is deterministic from it) unless
    the checkpoint itself carries ``embed_tokens.weight``. A randomly
    initialized embedding would silently break serving, so its absence raises.
    """
    state = resolve_training_state(checkpoint_path)
    model = materialize_draft(
        state, draft_config_path, vocab_mapping_path=vocab_mapping_path
    )
    if "embed_tokens.weight" not in state["draft_state_dict"]:
        if not embedding_source:
            raise ValueError(
                "checkpoint has no embed_tokens.weight (draft checkpoints "
                "exclude the frozen embedding); pass embedding_source=<target "
                "model path> so the export ships the real embedding"
            )
        model.load_embedding(embedding_source, embedding_key=embedding_key)
    full_state = dict(model.state_dict())
    full_state.update(state["draft_state_dict"])  # trained keys win
    model.save_pretrained(output_dir, state_dict=full_state)
    return output_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=export_to_hf.__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--draft-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocab-mapping", default=None)
    parser.add_argument(
        "--embedding-source",
        default=None,
        help="target model path/dir supplying the frozen embedding "
        "(required unless the checkpoint carries embed_tokens.weight)",
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
    )
    print(f"exported HF draft to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
