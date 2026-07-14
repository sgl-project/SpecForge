# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Export a DataFlow training checkpoint to an SGLang spec-decoder draft directory.

Weight-name compatibility is the silent-failure risk here: a key the serving
loader does not expect is skipped (or zero-filled) without an error. The
per-architecture trainer-key -> serving-key map therefore lives in ONE
documented place — ``WEIGHT_MAPS`` below plus ``docs/export_weight_map_mla.md``
— and this exporter validates the produced state against it instead of hoping.

For ``LlamaForCausalLMEagle3`` the map is the identity: SpecForge's draft module
names (``midlayer.*`` / ``fc`` / ``norm`` / ``lm_head`` + the ``t2d``/``d2t``
buffers) are exactly what sglang's EAGLE3 draft loader reads. The MLA draft adds
real renames when it lands (see the doc).
"""

from __future__ import annotations

import argparse
from typing import Dict, Optional

from specforge.export.checkpoint_io import materialize_draft, resolve_training_state

#: per-architecture trainer-key -> serving-key renames ({} = identity).
WEIGHT_MAPS: Dict[str, Dict[str, str]] = {
    "LlamaForCausalLMEagle3": {},
}

#: keys the sglang EAGLE3 spec-decoder loader requires in a draft checkpoint.
_REQUIRED_SERVING_KEYS = ("fc.weight", "norm.weight", "lm_head.weight", "t2d", "d2t")


def _serving_state(state_dict, weight_map: Dict[str, str]):
    out = {weight_map.get(k, k): v for k, v in state_dict.items()}
    bad_prefix = [k for k in out if k.startswith("draft_model.")]
    if bad_prefix:
        raise ValueError(
            f"serving state still carries trainer prefixes: {sorted(bad_prefix)}"
        )
    missing = [k for k in _REQUIRED_SERVING_KEYS if weight_map.get(k, k) not in out]
    if missing:
        raise ValueError(
            f"serving state is missing required keys {missing}; the sglang loader "
            f"would silently produce a broken draft. Present: {sorted(out)[:8]}..."
        )
    return out


def export_to_sglang(
    checkpoint_path: str,
    draft_config_path: str,
    output_dir: str,
    *,
    vocab_mapping_path: Optional[str] = None,
    weight_map: Optional[Dict[str, str]] = None,
) -> str:
    """Write an sglang-loadable draft directory; returns ``output_dir``.

    ``weight_map`` overrides the per-architecture entry in :data:`WEIGHT_MAPS`.
    ``vocab_mapping_path`` refreshes the ``t2d``/``d2t`` buffers when the
    checkpoint predates them.
    """
    state = resolve_training_state(checkpoint_path)
    if state.get("strategy") != "eagle3":
        raise ValueError(
            "the specialized SGLang exporter currently supports EAGLE3 "
            f"checkpoints only, got strategy={state.get('strategy')!r}; use "
            "--to hf for DFlash-family and P-EAGLE draft model directories"
        )
    model = materialize_draft(
        state, draft_config_path, vocab_mapping_path=vocab_mapping_path
    )
    if weight_map is None:
        weight_map = WEIGHT_MAPS.get(type(model).__name__, {})
    # the model's state dict includes any refreshed t2d/d2t buffers; drop the
    # embeddings exactly as the trainer-side checkpoint filter does.
    full = {k: v for k, v in model.state_dict().items() if "embed" not in k.lower()}
    model.save_pretrained(output_dir, state_dict=_serving_state(full, weight_map))
    return output_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=export_to_sglang.__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--draft-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocab-mapping", default=None)
    args = parser.parse_args(argv)
    out = export_to_sglang(
        args.checkpoint,
        args.draft_config,
        args.output_dir,
        vocab_mapping_path=args.vocab_mapping,
    )
    print(f"exported sglang draft to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
