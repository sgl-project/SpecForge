# coding=utf-8
"""Merge a trained MTP draft checkpoint back into the base target checkpoint.

This module owns the MTP-specific merge policy (native key prefix handling,
shared-embedding backfill, config patching).  The model-independent merge
machinery — copying non-weight files, replacing keys by prefix, shard/index
writing — lives in ``specforge/modeling/target/checkpoint.py``.
Architecture-specific knowledge (target-side embed/lm_head key candidates,
native key prefix) comes from the registered MTP draft class — see
``specforge/modeling/draft/mtp/``.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
from typing import Dict, List, Optional, Tuple

import torch

from specforge.modeling.target.checkpoint import (
    load_selected_tensors,
    load_tensors_by_keys,
    merge_state_into_checkpoint,
)


def _default_key_candidates() -> Tuple[List[str], List[str], str]:
    """Base-class defaults, imported lazily to keep this module import-light."""

    from specforge.modeling.draft.mtp.base import MTPDraftModel

    return (
        list(MTPDraftModel.TARGET_EMBED_KEY_CANDIDATES),
        list(MTPDraftModel.TARGET_HEAD_KEY_CANDIDATES),
        MTPDraftModel.NATIVE_KEY_PREFIX,
    )


def _resolve_key_candidates(
    draft_config_source: str,
) -> Tuple[List[str], List[str], str]:
    """Return (embed candidates, head candidates, native prefix) for the draft.

    Reads the draft ``config.json`` and resolves its
    ``architectures[0]`` through the draft registry, so each MTP family can
    override its target-side key candidates on the draft class.
    """

    embed, head, prefix = _default_key_candidates()
    config_path = (
        draft_config_source
        if os.path.isfile(draft_config_source)
        else os.path.join(draft_config_source, "config.json")
    )
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                architectures = json.load(f).get("architectures") or []
            if architectures:
                from specforge.modeling.draft.registry import DRAFT_REGISTRY

                draft_cls = DRAFT_REGISTRY.get(architectures[0])
                if draft_cls is not None:
                    embed = list(
                        getattr(draft_cls, "TARGET_EMBED_KEY_CANDIDATES", embed)
                    )
                    head = list(getattr(draft_cls, "TARGET_HEAD_KEY_CANDIDATES", head))
                    prefix = getattr(draft_cls, "NATIVE_KEY_PREFIX", prefix)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  warning: could not resolve draft key candidates: {exc}")
    return embed, head, prefix


def convert_mtp_keys(
    state_dict: Dict[str, torch.Tensor], fmt: str, prefix: str = "mtp."
) -> Dict[str, torch.Tensor]:
    """Convert MTP weight keys to the requested output format.

    Training already saves the flat native layout that both SGLang and
    HF/vLLM MTP modules expect, so ``sglang`` and ``hf`` both return it
    unchanged; ``fmt`` is kept for backward compatibility. A legacy nested
    layout (``mtp.model.layers.0.*``) is normalized to flat.
    """

    converted = {}
    for k, v in state_dict.items():
        # Normalize legacy nested keys (mtp.model.layers.* -> mtp.layers.*).
        if k.startswith(f"{prefix}model.layers."):
            new_k = k.replace(f"{prefix}model.layers.", f"{prefix}layers.", 1)
        elif k == f"{prefix}model.norm.weight":
            new_k = f"{prefix}norm.weight"
        # Promote bare embed_tokens / lm_head saved by the training script to the
        # native namespace expected by vLLM/SGLang.
        elif k == "embed_tokens.weight":
            new_k = f"{prefix}embed_tokens.weight"
        elif k == "lm_head.weight":
            new_k = f"{prefix}lm_head.weight"
        else:
            new_k = k
        converted[new_k] = v
    return _unshare_storage(converted)


def _unshare_storage(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Clone tensors whose storage is aliased under another key.

    A tied target shares one Parameter between ``embed_tokens.weight`` and
    ``mtp.lm_head.weight``; after promotion both keys keep aliasing the same
    storage, which safetensors' ``save_file`` rejects ("tensors share
    memory").  Cloning the later alias preserves the values while giving every
    key its own storage.
    """

    seen: set[int] = set()
    out: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        ptr = value.untyped_storage().data_ptr()
        if ptr in seen:
            value = value.clone()
        else:
            seen.add(ptr)
        out[key] = value
    return out


def _find_base_key(state_dict: Dict[str, torch.Tensor], *candidates: str) -> str | None:
    """Return the first candidate key that exists in ``state_dict``."""

    for key in candidates:
        if key in state_dict:
            return key
    return None


def _copy_shared_embeddings(
    base_state: Dict[str, torch.Tensor],
    mtp_state: Dict[str, torch.Tensor],
    tie_word_embeddings: bool,
    embed_key_candidates: List[str],
    head_key_candidates: List[str],
    prefix: str,
) -> Dict[str, torch.Tensor]:
    """Copy base embed_tokens/lm_head into the MTP state if they are missing.

    During training the draft model typically shares ``embed_tokens`` and
    ``lm_head`` with the target model, so the saved MTP checkpoint does not
    contain those tensors.  vLLM/SGLang, however, instantiate their own
    ``mtp.embed_tokens`` (and a separate ``lm_head`` when weights are not tied),
    and expect them in the checkpoint.  Copying them from the base model keeps
    the merged checkpoint self-contained and avoids random-initialization of the
    MTP input/output embeddings at serving time.
    """

    embed_target = f"{prefix}embed_tokens.weight"
    head_target = f"{prefix}lm_head.weight"

    if embed_target not in mtp_state:
        embed_key = _find_base_key(base_state, *embed_key_candidates)
        if embed_key:
            mtp_state[embed_target] = base_state[embed_key]
            print(f"  copied {embed_key} -> {embed_target}")
        else:
            print(
                "  warning: base embed_tokens.weight not found; "
                f"{embed_target} will be randomly initialized"
            )

    if not tie_word_embeddings and head_target not in mtp_state:
        lm_head_key = _find_base_key(base_state, *head_key_candidates)
        if lm_head_key:
            mtp_state[head_target] = base_state[lm_head_key]
            print(f"  copied {lm_head_key} -> {head_target}")
        else:
            print(
                "  warning: base lm_head.weight not found; "
                f"{head_target} will be randomly initialized"
            )

    return mtp_state


def _patch_text_config(base_config: dict, draft_config: dict) -> dict:
    """Ensure base text_config contains MTP-critical dims from the draft config.

    Some Qwen3.5 base checkpoints omit ``head_dim`` in ``text_config``; vLLM's
    ``Qwen3_5TextConfig`` then falls back to its default (``head_dim=256``),
    which mismatches the trained MTP weights (e.g. q_norm/k_norm shape 128).
    Only the structural dims that must agree between base and draft are synced.
    """

    keys_to_sync = [
        "head_dim",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
    ]

    target = base_config
    if "text_config" in base_config:
        target = base_config["text_config"]

    source = draft_config
    if "text_config" in draft_config:
        source = draft_config["text_config"]

    for key in keys_to_sync:
        if key not in source:
            continue
        old = target.get(key)
        new = source[key]
        if old != new:
            target[key] = new
            print(f"  overriding text_config.{key}: {old} -> {new}")

    return base_config


def _load_first_checkpoint(checkpoint_dir: str) -> Dict[str, torch.Tensor]:
    """Load every tensor of a single-file checkpoint directory."""

    safetensors = glob.glob(os.path.join(checkpoint_dir, "*.safetensors"))
    bins = glob.glob(os.path.join(checkpoint_dir, "*.bin"))
    if safetensors:
        return load_selected_tensors(checkpoint_dir, lambda _key: True)
    if bins:
        return torch.load(bins[0], map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No safetensors/bin weights found in {checkpoint_dir}")


def _has_model_weights(path: str) -> bool:
    """Return whether ``path`` is already an exported model directory."""

    if not os.path.isdir(path):
        return False
    patterns = ("model*.safetensors", "pytorch_model*.bin")
    return any(glob.glob(os.path.join(path, pattern)) for pattern in patterns)


def _load_mtp_source(
    checkpoint_path: str,
    draft_config_path: Optional[str],
) -> Tuple[Dict[str, torch.Tensor], str, Optional[str]]:
    """Load MTP weights from either runtime state or an exported draft.

    Returns ``(state_dict, config_source, model_source_dir)``. The last item is
    set only for an exported model directory, where companion modeling files
    may also need to be copied.
    """

    path = checkpoint_path
    if path.startswith("file://"):
        path = path[len("file://") :]
    if _has_model_weights(path):
        return _load_first_checkpoint(path), path, path

    from specforge.export.checkpoint_io import resolve_training_state

    state = resolve_training_state(checkpoint_path)
    if state.get("strategy") != "mtp":
        raise ValueError(
            "MTP merge requires a training checkpoint written by strategy='mtp'; "
            f"got strategy={state.get('strategy')!r}"
        )
    draft_state = state.get("draft_state_dict")
    if not isinstance(draft_state, dict):
        raise ValueError("MTP training checkpoint has no draft_state_dict")
    if not draft_config_path:
        raise ValueError(
            "draft_config_path is required when merging a runtime training "
            "checkpoint"
        )
    if not os.path.isfile(draft_config_path):
        raise FileNotFoundError(f"draft config not found: {draft_config_path}")
    return dict(draft_state), draft_config_path, None


def merge_mtp_into_base(
    base_model_path: str,
    mtp_checkpoint_path: str,
    output_path: str,
    key_format: str = "sglang",
    *,
    draft_config_path: Optional[str] = None,
) -> None:
    """Merge trained MTP weights into a copy of the base checkpoint.

    The output directory is a self-contained HF checkpoint loadable directly by
    SGLang's native MTP modules (no separate draft-model path). Runtime
    checkpoints require ``draft_config_path``; an exported HF draft supplies its
    own ``config.json``.
    """

    mtp_state, config_source, model_source_dir = _load_mtp_source(
        mtp_checkpoint_path, draft_config_path
    )
    embed_key_candidates, head_key_candidates, prefix = _resolve_key_candidates(
        config_source
    )
    os.makedirs(output_path, exist_ok=True)

    mtp_state = convert_mtp_keys(mtp_state, key_format, prefix)

    # Determine whether word embeddings are tied to decide whether a separate
    # lm_head must be materialized for the MTP module.
    tie_word_embeddings = True
    base_config_path = os.path.join(base_model_path, "config.json")
    if os.path.exists(base_config_path):
        with open(base_config_path, "r") as f:
            base_cfg = json.load(f)
        # VLM checkpoints nest text config under "text_config".
        text_cfg = base_cfg.get("text_config", base_cfg)
        tie_word_embeddings = text_cfg.get("tie_word_embeddings", True)

    # If the trained checkpoint did not save shared embeddings, copy them from
    # the base checkpoint so vLLM/SGLang can initialise the MTP embed_tokens/
    # lm_head from the merged checkpoint.
    embed_target = f"{prefix}embed_tokens.weight"
    head_target = f"{prefix}lm_head.weight"
    if embed_target not in mtp_state or (
        not tie_word_embeddings and head_target not in mtp_state
    ):
        base_state = load_tensors_by_keys(
            base_model_path, embed_key_candidates + head_key_candidates
        )
        mtp_state = _copy_shared_embeddings(
            base_state,
            mtp_state,
            tie_word_embeddings,
            embed_key_candidates,
            head_key_candidates,
            prefix,
        )

    # The generic merge machinery (copy, prefix-key replacement, shard/index
    # writing) lives in modeling/target/checkpoint.py.
    merge_state_into_checkpoint(
        base_model_path,
        mtp_state,
        output_path,
        shard_name="mtp-merged.safetensors",
        drop_prefixes=(prefix,),
    )

    # Ensure the merged config exposes the MTP structural dims.  vLLM/SGLang
    # use these values to build the MTP module; if the base config omits
    # ``head_dim`` (common for some Qwen3.5 checkpoints), the loader will use
    # its default and fail with a shape mismatch.
    resolved_draft_config_path = (
        config_source
        if os.path.isfile(config_source)
        else os.path.join(config_source, "config.json")
    )
    output_config_path = os.path.join(output_path, "config.json")
    if os.path.exists(resolved_draft_config_path) and os.path.exists(
        output_config_path
    ):
        with open(resolved_draft_config_path, "r") as f:
            draft_config = json.load(f)
        with open(output_config_path, "r") as f:
            base_config = json.load(f)
        patched_config = _patch_text_config(base_config, draft_config)
        with open(output_config_path, "w") as f:
            json.dump(patched_config, f, indent=2)

    # Copy over the MTP modeling file if present; some loaders need it for
    # trust_remote_code / auto_map resolution.
    if model_source_dir is not None:
        mtp_py_src = os.path.join(model_source_dir, "mtp.py")
        if os.path.exists(mtp_py_src):
            shutil.copy2(mtp_py_src, os.path.join(output_path, "mtp.py"))

    print(f"Merged checkpoint saved to {output_path}")
    print(f"  key format: {key_format}")
    print(f"  MTP tensors merged: {len(mtp_state)}")


__all__ = ["convert_mtp_keys", "merge_mtp_into_base"]
