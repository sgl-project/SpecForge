#!/usr/bin/env python3
# coding=utf-8
"""
merge_mtp_draft_into_target.py
==============================

Merge the trained MTP draft model produced by SpecForge-MTP back into the
**original** target model's MTP head, producing a new complete model whose MTP
slots have been refreshed by the draft training.

Why this script exists:
    The draft is structurally identical to the target's native MTP head (see
    ``specforge/modeling/draft/qwen3_moe_mtp.py``), but it stores its tensors
    under draft-style keys (``midlayer.*``, ``fc.weight``, ...). To use the
    refreshed MTP at inference time you need to write those tensors back under
    the target-style keys (``mtp.layers.{i}.*``, ``mtp.fc.weight``, ...).

RMSNorm convention (IMPORTANT)
    By default this script assumes BOTH the draft and the target store
    RMSNorm.weight in the **Gemma** parametrization
    ``y = (1 + w) * x_normed``  (weight init = 0).
    This is the case for any draft trained by the new SpecForge code that
    uses :class:`GemmaRMSNorm` (commit X and later) merged into a Qwen3.5 /
    Qwen3-MoE / Gemma / DeepSeek-V3 family target.  In this default mode
    every RMSNorm tensor is copied verbatim.

    If your draft was trained by the LEGACY code that used ``LlamaRMSNorm``
    (``y = w * x_normed``, weight init = 1) -- e.g. the
    ``epoch_3_step_28000`` checkpoint produced before the convention switch
    -- you MUST either:
      (a) pre-process the draft once with
          ``scripts/migrate_specforge_norm_llama_to_gemma.py`` (recommended;
          the migrated checkpoint is also resume-trainable), OR
      (b) pass ``--legacy-llama-norm`` to this script, which restores the
          old behaviour:  every RMSNorm weight in the merge plan is
          rewritten as ``W_target = W_draft - 1.0`` (Llama -> Gemma), and
          for any Gemma-style RMSNorm slot that the legacy draft did NOT
          save, the target's own value is also rewritten in-place via
          ``W -= 1.0`` ("self-convert").  Without this flag a legacy draft
          will silently produce an MTP head whose every norm scale is off
          by 1.0, which destroys speculative-decoding accept length.

Usage:
    # Default mode -- new GemmaRMSNorm draft, GemmaRMSNorm target.
    # DOES NOT modify the source target model.
    # Untouched shards in the output dir are symlinks to the originals to save
    # disk space.
    python scripts/merge_mtp_draft_into_target.py \
        --draft  outputs/<new_specforge_run>/epoch_X_step_Y \
        --target /mnt/tidal-alsh01/dataset/xiaowen/model/qwen35-122b \
        --output /mnt/tidal-alsh01/dataset/xiaowen/model/qwen35-122b-mtp-merged-stepY

    # Legacy mode -- LlamaRMSNorm draft (pre-convention-switch ckpt).
    python scripts/merge_mtp_draft_into_target.py \
        --draft  outputs/qwen3.5-122b-a10b-mtp-offline-lr-2e-5/epoch_3_step_28000 \
        --target /mnt/tidal-alsh01/dataset/xiaowen/model/qwen35-122b \
        --output /mnt/tidal-alsh01/dataset/xiaowen/model/qwen35-122b-mtp-merged-step28000-v2 \
        --legacy-llama-norm

    # Dry-run -- only print the validation report.
    python scripts/merge_mtp_draft_into_target.py \
        --draft  ... --target ... --output ... --dry-run

    # In-place mode -- USE WITH CAUTION, mutates the target safetensors.
    python scripts/merge_mtp_draft_into_target.py \
        --draft  ... --target ... --in-place

What this script checks BEFORE writing anything:
    1. Every (target MTP slot) -> (draft trained tensor) pair has matching
       shapes.
    2. Reports draft keys that are unmapped (e.g. spec-decode buffers
       ``t2d`` / ``d2t``) so you know they are intentionally ignored.
    3. Reports MTP slots that the draft did NOT save (e.g. ``embed_tokens``,
       ``lm_head`` when the draft was trained on a vocab subset). Those slots
       in the target model are kept as-is.
    4. Reports vocab-mapping diagnostics: if ``t2d`` is present and it is NOT
       all-True, that means the draft trained on a draft-vocab subset and its
       ``lm_head`` (if saved) is NOT compatible with the target's full vocab
       ``shared_head.head.weight``. In that case the script will refuse to copy
       ``lm_head.weight`` and will keep the target's original head.
    5. (Default mode) Sanity-checks that every draft-side RMSNorm weight has
       ``|mean| < 0.5``, which is the expected centred-at-0 distribution for
       a Gemma-trained tensor.  A norm with ``|mean| >= 0.5`` strongly
       suggests the draft was actually trained with LlamaRMSNorm, in which
       case the script aborts and tells you to use ``--legacy-llama-norm``.

Output layout:
    output_dir/
        # SAFETENSORS -- only the shards that contain at least one MTP slot
        # are rewritten; the rest are symlinks to the originals.
        model-XXXXX-of-YYYYY.safetensors    # rewritten (real file)
        model-ZZZZZ-of-YYYYY.safetensors -> /abs/path/to/target/model-ZZZZZ-of-YYYYY.safetensors
        model.safetensors.index.json        # copied verbatim (we never change the key set)
        # Everything else (config.json, tokenizer*, processor_config, *.txt, *.jinja,
        # iter_* checkpoint dirs, ...) is symlinked.
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


# ----------------------------------------------------------------------------
# RMSNorm parametrization handling.
#
# The target model (`Qwen3_5MoeRMSNorm`, transformers) and sglang's runtime
# both use the **Gemma-style** RMSNorm:
#
#     y = (1 + w_gemma) * x_normed       # weight init = 0 (1-centered)
#
# Two regimes are supported by this script:
#
# (1) DEFAULT (post-convention-switch SpecForge):
#     The draft uses :class:`GemmaRMSNorm` and stores weights in the SAME
#     Gemma parametrization as the target.  No numeric conversion is
#     needed -- every RMSNorm tensor is copied verbatim.  As a defensive
#     safety check we also assert the draft's norm tensors are roughly
#     centred at 0 (|mean| < 0.5); a tensor that looks ~1-centred is a
#     strong signal that the user actually has a legacy checkpoint and
#     forgot to either migrate it or pass --legacy-llama-norm.
#
# (2) LEGACY (`--legacy-llama-norm`, pre-convention-switch SpecForge):
#     The draft used :class:`LlamaRMSNorm`
#     (`y = w_llama * x_normed`, weight init = 1) so the stored weight
#     is the ~1-centred effective scale that was applied at training
#     time.  To reproduce that scale at inference time on the Gemma side
#     we store
#         w_target_slot = w_draft - 1.0
#     so that the Gemma forward at inference recovers
#         (1 + w_target_slot) * x = (1 + (w_draft - 1)) * x = w_draft * x.
#     For Gemma-style RMSNorm slots that the legacy draft did NOT save
#     (e.g. the historical `pre_fc_norm_embedding` filter bug) the
#     target's own value must also be rewritten in-place via W -= 1.0
#     ("self-convert"), because at training time the draft was implicitly
#     applying that target weight as a Llama-style scale.
#
# Without the proper handling, every RMSNorm in the MTP block ends up with
# a scale that differs from training by ~1.0; for `input_layernorm` (whose
# stored weight is close to 0) this is a >20x mismatch with a sign flip,
# which silently destroys speculative-decoding accept-length even though
# token-level top-1 training accuracy looks healthy.
# ----------------------------------------------------------------------------
NORM_DRAFT_KEYS = frozenset([
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
    "norm.weight",
    "midlayer.input_layernorm.weight",
    "midlayer.post_attention_layernorm.weight",
    "midlayer.self_attn.q_norm.weight",
    "midlayer.self_attn.k_norm.weight",
])

# Sanity safety-net for norm.weight magnitude.  This is intentionally very
# loose: we used to enforce |mean|>=0.5 to detect a "Llama-shaped" tensor,
# but that judgement turns out to be unreliable -- the real qwen35 MTP
# target ships `mtp.norm.weight` with mean=+1.585 (and many other Gemma
# weights drift well above 1.0 after training).  We now only flag tensors
# whose |mean| is catastrophically far from any plausible regime, which
# almost always indicates ckpt corruption rather than a wrong convention.
_NORM_ABS_MEAN_PANIC = 5.0


def _convert_llama_to_gemma(t: torch.Tensor) -> torch.Tensor:
    """Subtract 1.0 in fp32 then cast back to the original dtype."""
    out = t.to(torch.float32) - 1.0
    return out.to(t.dtype).contiguous()


def _read_norm_mean_abs(model_dir: Path, weight_map, key: str) -> float:
    """Return |mean| of a stored norm.weight tensor, computed in fp32."""
    shard = weight_map[key]
    with safe_open(model_dir / shard, framework="pt") as f:
        t = f.get_tensor(key)
    return float(t.to(torch.float32).mean().abs().item())


def _read_norm_convention_marker(draft_dir: Path):
    """Read the `_specforge_norm_convention` marker from draft's config.json.

    Written by `scripts/migrate_specforge_norm_llama_to_gemma.py` after a
    successful Llama->Gemma migration.  Returns one of {"llama","gemma"}
    if present and recognised, otherwise None.

    Brand-new ckpts produced by the GemmaRMSNorm-based training code do
    NOT carry this marker (the trainer is unaware of it), so a missing
    marker is treated as "trust the user" and only emits an INFO note.
    """
    cfg_path = draft_dir / "config.json"
    if not cfg_path.exists():
        return None
    try:
        with cfg_path.open() as f:
            cfg = json.load(f)
    except Exception:
        return None
    marker = cfg.get("_specforge_norm_convention")
    if isinstance(marker, str) and marker.lower() in ("llama", "gemma"):
        return marker.lower()
    return None


# ----------------------------------------------------------------------------
# Mapping from draft model state-dict keys to target MTP head keys.
# Ordering matters only for log readability.
# ----------------------------------------------------------------------------
def build_mapping(num_experts: int, mtp_layer_idx: int):
    mtp = "mtp"
    layer = f"{mtp}.layers.{mtp_layer_idx}"
    moe = f"{layer}.mlp"

    pairs = [
        # ---- MTP-level shared modules (named "mtp.<name>" in target) ----
        ("fc.weight",                                f"{mtp}.fc.weight"),
        ("pre_fc_norm_embedding.weight",             f"{mtp}.pre_fc_norm_embedding.weight"),
        ("pre_fc_norm_hidden.weight",                f"{mtp}.pre_fc_norm_hidden.weight"),
        ("norm.weight",                              f"{mtp}.norm.weight"),
        # ---- Per-MTP-layer norms ----
        ("midlayer.input_layernorm.weight",          f"{layer}.input_layernorm.weight"),
        ("midlayer.post_attention_layernorm.weight", f"{layer}.post_attention_layernorm.weight"),
        # ---- Self-attention ----
        ("midlayer.self_attn.q_proj.weight",         f"{layer}.self_attn.q_proj.weight"),
        ("midlayer.self_attn.k_proj.weight",         f"{layer}.self_attn.k_proj.weight"),
        ("midlayer.self_attn.v_proj.weight",         f"{layer}.self_attn.v_proj.weight"),
        ("midlayer.self_attn.o_proj.weight",         f"{layer}.self_attn.o_proj.weight"),
        ("midlayer.self_attn.q_norm.weight",         f"{layer}.self_attn.q_norm.weight"),
        ("midlayer.self_attn.k_norm.weight",         f"{layer}.self_attn.k_norm.weight"),
        # ---- MoE router & shared expert ----
        ("midlayer.mlp.gate.weight",                 f"{moe}.gate.weight"),
        ("midlayer.mlp.shared_expert.gate_proj.weight", f"{moe}.shared_expert.gate_proj.weight"),
        ("midlayer.mlp.shared_expert.up_proj.weight",   f"{moe}.shared_expert.up_proj.weight"),
        ("midlayer.mlp.shared_expert.down_proj.weight", f"{moe}.shared_expert.down_proj.weight"),
        ("midlayer.mlp.shared_expert_gate.weight",   f"{moe}.shared_expert_gate.weight"),
        # ---- Optional: only present in draft if explicitly saved ----
        ("embed_tokens.weight",                      f"{layer}.embed_tokens.weight"),
        ("lm_head.weight",                           f"{layer}.shared_head.head.weight"),
    ]
    # ---- MoE experts ----
    for i in range(num_experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            pairs.append(
                (f"midlayer.mlp.experts.{i}.{proj}.weight",
                 f"{moe}.experts.{i}.{proj}.weight")
            )
    return pairs


# ----------------------------------------------------------------------------
# Index helpers
# ----------------------------------------------------------------------------
def load_index(model_dir: Path):
    """Return (weight_map, index_filename_or_None)."""
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


def get_meta(model_dir: Path, weight_map, key: str):
    """Return (shape, dtype) of a tensor without materializing it."""
    shard = weight_map[key]
    with safe_open(model_dir / shard, framework="pt") as f:
        sl = f.get_slice(key)
        return tuple(sl.get_shape()), sl.get_dtype()


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
def validate(draft_dir: Path, target_dir: Path, mtp_layer_idx: int,
             num_experts: int, legacy_llama_norm: bool = False):
    """Build the merge plan and report any issues.

    Args:
        legacy_llama_norm: If True, the draft is assumed to use the legacy
            LlamaRMSNorm parametrization (weights ~1-centred). The script
            will subtract 1.0 from every draft norm.weight at copy time AND
            rewrite any unsaved norm slot in the target via W -= 1.0
            (`norm_self_convert`).  If False (default), both sides are
            assumed to already be in Gemma parametrization (weights
            ~0-centred); norms are copied verbatim and `norm_self_convert`
            is always empty.

    Returns:
        (plan, draft_map, target_map, vocab_diag, fatal, lm_head_safe,
         norm_self_convert, norm_convention_violations)

        norm_convention_violations is a list of (key, abs_mean) for any
        draft norm.weight whose distribution looks inconsistent with the
        chosen mode (~1-centred under default mode, or ~0-centred under
        --legacy-llama-norm).  Always empty if there are no norm tensors
        in the plan.
    """
    print(f"[validate] draft  = {draft_dir}")
    print(f"[validate] target = {target_dir}")
    print(f"[validate] mtp_layer_idx = {mtp_layer_idx}, num_experts = {num_experts}")
    print(f"[validate] mode   = "
          f"{'LEGACY (Llama->Gemma -1.0 conversion)' if legacy_llama_norm else 'DEFAULT (verbatim copy, both sides Gemma)'}")
    _marker_preview = _read_norm_convention_marker(draft_dir)
    print(f"[validate] draft norm marker = {_marker_preview!r} "
          f"(from {draft_dir}/config.json:_specforge_norm_convention)")
    if _marker_preview is None:
        print(f"[validate]    INFO: no marker present -- new training ckpts and "
              f"pre-migration ckpts both lack it; trusting --legacy-llama-norm flag.")

    draft_map, _ = load_index(draft_dir)
    target_map, _ = load_index(target_dir)

    pairs = build_mapping(num_experts, mtp_layer_idx)
    pairs_dict = dict(pairs)

    draft_keys = set(draft_map.keys())
    target_keys = set(target_map.keys())

    plan = []
    skipped_draft_missing = []
    target_missing = []
    shape_mismatch = []
    extra_draft_keys = sorted(draft_keys - set(pairs_dict.keys()))
    norm_self_convert = []

    for dk, tk in pairs:
        in_d = dk in draft_keys
        in_t = tk in target_keys
        if not in_d:
            skipped_draft_missing.append((dk, tk, in_t))
            # Self-convert is meaningful ONLY in legacy mode: in default
            # mode the target weight is already in the correct (Gemma)
            # convention and must be preserved verbatim.
            if legacy_llama_norm and in_t and dk in NORM_DRAFT_KEYS:
                norm_self_convert.append(tk)
            continue
        if not in_t:
            target_missing.append((dk, tk))
            continue
        d_shape, d_dtype = get_meta(draft_dir, draft_map, dk)
        t_shape, t_dtype = get_meta(target_dir, target_map, tk)
        if d_shape != t_shape:
            shape_mismatch.append((dk, tk, d_shape, t_shape, str(d_dtype), str(t_dtype)))
            continue
        plan.append((dk, tk, d_shape))

    # Vocab-mapping diagnostics + decide whether lm_head copy is safe.
    vocab_diag = {}
    lm_head_safe = True
    if "t2d" in draft_keys:
        with safe_open(draft_dir / draft_map["t2d"], framework="pt") as f:
            t2d = f.get_tensor("t2d")
        n_total = int(t2d.numel())
        n_true = int(t2d.sum().item())
        vocab_diag["t2d_total"] = n_total
        vocab_diag["t2d_true"] = n_true
        vocab_diag["t2d_full_vocab"] = (n_true == n_total)
        if n_true != n_total:
            lm_head_safe = False
    if "d2t" in draft_keys:
        with safe_open(draft_dir / draft_map["d2t"], framework="pt") as f:
            d2t = f.get_tensor("d2t")
        vocab_diag["d2t_size"] = int(d2t.numel())
        vocab_diag["d2t_all_zero"] = (int(d2t.abs().sum().item()) == 0)

    # Filter lm_head out of plan if it would be unsafe
    if not lm_head_safe:
        before = len(plan)
        plan = [(dk, tk, sh) for (dk, tk, sh) in plan if dk != "lm_head.weight"]
        if len(plan) != before:
            print("[validate] !! draft trained on a vocab subset (t2d not all-True);")
            print("[validate]    refusing to overwrite target's shared_head.head.weight.")
            print("[validate]    Pass --force-lm-head to override.")

    # ---- Norm-convention sanity check on the draft side ----
    # We rely on an explicit marker in `draft_dir/config.json` written by
    # `scripts/migrate_specforge_norm_llama_to_gemma.py`.  Pure |mean|-based
    # heuristics turn out to be unreliable -- the official qwen35 MTP
    # target itself ships `mtp.norm.weight` with mean=+1.585, which is
    # well within the regime that a naive heuristic would mis-classify
    # as "Llama-shaped".  See sglang `GemmaRMSNorm.forward_native`
    # (`/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/layernorm.py:402`):
    # the formula is `y = (1 + w) * x_normed`, so weights can drift far
    # from 0 during training.
    #
    # Decision table:
    #   marker == "gemma"  + flag == False  -> ok (default path)
    #   marker == "gemma"  + flag == True   -> ERROR: do not pass flag
    #   marker == "llama"  + flag == True   -> ok (legacy path)
    #   marker == "llama"  + flag == False  -> ERROR: pass flag or migrate
    #   marker missing                       -> INFO only (legacy ckpts and
    #                                            brand-new training ckpts
    #                                            both lack the marker)
    # An additional cheap "panic" magnitude check is kept as a safety
    # net for catastrophically bad weights.
    norm_convention_violations = []
    norm_convention_marker = _read_norm_convention_marker(draft_dir)
    if norm_convention_marker == "gemma" and legacy_llama_norm:
        norm_convention_violations.append(
            ("config.json", -1.0,
             "draft is marked _specforge_norm_convention=gemma but "
             "--legacy-llama-norm IS set (drop the flag)"))
    elif norm_convention_marker == "llama" and not legacy_llama_norm:
        norm_convention_violations.append(
            ("config.json", -1.0,
             "draft is marked _specforge_norm_convention=llama but "
             "--legacy-llama-norm is NOT set (add the flag, or run "
             "scripts/migrate_specforge_norm_llama_to_gemma.py first)"))
    norm_keys_in_plan = [(dk, tk) for dk, tk, _ in plan if dk in NORM_DRAFT_KEYS]
    for dk, _tk in norm_keys_in_plan:
        m = _read_norm_mean_abs(draft_dir, draft_map, dk)
        if m >= _NORM_ABS_MEAN_PANIC:
            norm_convention_violations.append(
                (dk, m,
                 f"|mean|={m:.3f} is unreasonably large (>={_NORM_ABS_MEAN_PANIC}); "
                 "ckpt may be corrupted"))

    # ---- Count how many pairs in plan are RMSNorm tensors ----
    n_norm_in_plan = len(norm_keys_in_plan)

    # ---- Print summary ----
    print()
    print(f"[validate] mapped & shape-OK : {len(plan)} / {len(pairs)}")
    if legacy_llama_norm:
        print(f"[validate] RMSNorm slots needing Llama->Gemma (-1.0) conversion: "
              f"{n_norm_in_plan} from plan + {len(norm_self_convert)} self-convert "
              f"(target-side only)")
    else:
        print(f"[validate] RMSNorm slots in plan (will be copied VERBATIM, no "
              f"conversion): {n_norm_in_plan}")

    if norm_self_convert:
        print(f"[validate] norm SELF-CONVERT ({len(norm_self_convert)}): "
              f"draft did not save these RMSNorm slots, but target value still "
              f"needs in-place (W -> W - 1.0) conversion for parametrization "
              f"consistency:")
        for tk in norm_self_convert:
            print(f"   - {tk}")

    if skipped_draft_missing:
        print(f"[validate] SKIPPED ({len(skipped_draft_missing)}): "
              f"draft did not save these trainable slots, target value preserved:")
        for dk, tk, in_t in skipped_draft_missing:
            mark = "ok-in-target" if in_t else "ALSO-MISSING-IN-TARGET"
            extra = ""
            if dk in NORM_DRAFT_KEYS and in_t and legacy_llama_norm:
                extra = "  [will SELF-CONVERT (-1.0)]"
            print(f"   - {dk:55s} -> {tk}   [{mark}]{extra}")

    if extra_draft_keys:
        print(f"[validate] EXTRA in draft ({len(extra_draft_keys)}): "
              f"not in mapping, will be ignored (e.g. d2t/t2d, training buffers):")
        for dk in extra_draft_keys:
            print(f"   - {dk}")

    if target_missing:
        print(f"[validate] !! TARGET MISSING ({len(target_missing)}): "
              f"these MTP slots do not exist in target model:")
        for dk, tk in target_missing:
            print(f"   - {dk} -> {tk}")

    if shape_mismatch:
        print(f"[validate] !! SHAPE MISMATCH ({len(shape_mismatch)}):")
        for dk, tk, ds, ts, ddt, tdt in shape_mismatch:
            print(f"   - {dk}: draft={ds}/{ddt}  vs  target={ts}/{tdt}")

    if vocab_diag:
        print(f"[validate] vocab-mapping diagnostics:")
        for k, v in vocab_diag.items():
            print(f"   - {k} = {v}")

    if norm_convention_violations:
        print(f"[validate] !! NORM CONVENTION VIOLATIONS ({len(norm_convention_violations)}):")
        for dk, m, reason in norm_convention_violations:
            if m < 0:
                print(f"   - {dk}: {reason}")
            else:
                print(f"   - {dk}: |mean|={m:.4f}   {reason}")
        print(f"[validate]    Either fix the --legacy-llama-norm flag, or run "
              f"scripts/migrate_specforge_norm_llama_to_gemma.py first.")

    fatal = bool(target_missing or shape_mismatch or norm_convention_violations)
    return (plan, draft_map, target_map, vocab_diag, fatal, lm_head_safe,
            norm_self_convert, norm_convention_violations)


# ----------------------------------------------------------------------------
# Apply
# ----------------------------------------------------------------------------
def apply_merge(plan, draft_map, target_map,
                draft_dir: Path, target_dir: Path, output_dir: Path,
                in_place: bool, link_mode: str = "symlink",
                norm_self_convert=None,
                legacy_llama_norm: bool = False):
    """
    plan: list of (draft_key, target_key, shape) -- copied from draft.
        In legacy mode (legacy_llama_norm=True) draft RMSNorm tensors are
        rewritten as W_target = W_draft - 1.0; otherwise they are copied
        verbatim.
    norm_self_convert: list of target_keys -- RMSNorm slots NOT saved by the
        draft, but whose target-side value must still be rewritten in-place
        (W -> W - 1.0) to match the Llama-style forward used during legacy
        training.  Only used when legacy_llama_norm=True; in default mode
        the validator always returns an empty list.
    legacy_llama_norm: see module docstring.
    link_mode: 'symlink' | 'hardlink' | 'copy'  (only for shards that are NOT rewritten)
    """
    norm_self_convert = list(norm_self_convert or [])
    if not legacy_llama_norm and norm_self_convert:
        # Defensive: validator should never produce self_convert entries
        # in default mode. If somehow they leaked through, refuse to apply
        # them (mutating target weights without an explicit user opt-in
        # would be very surprising).
        raise RuntimeError(
            "apply_merge: legacy_llama_norm=False but norm_self_convert is "
            f"non-empty ({norm_self_convert}); this would silently mutate "
            "target RMSNorm weights. Refusing to proceed."
        )

    shard_to_pairs = defaultdict(list)
    for dk, tk, _shape in plan:
        shard_to_pairs[target_map[tk]].append((dk, tk))
    # Also include self-convert target keys -- their shard becomes "touched".
    shard_to_self_convert = defaultdict(list)
    for tk in norm_self_convert:
        if tk not in target_map:
            # Should not happen -- validate filters these, but be defensive.
            continue
        shard_to_self_convert[target_map[tk]].append(tk)

    if not in_place:
        output_dir.mkdir(parents=True, exist_ok=True)

    target_shards = set(target_map.values())
    touched_shards = set(shard_to_pairs.keys()) | set(shard_to_self_convert.keys())
    untouched_shards = sorted(target_shards - touched_shards)

    # ---- 1) Untouched shards -- link / copy into output_dir ----
    if not in_place:
        for shard in untouched_shards:
            src = (target_dir / shard).resolve()
            dst = output_dir / shard
            _place(src, dst, mode=link_mode)
        print(f"[apply] linked {len(untouched_shards)} untouched shard(s) "
              f"(mode={link_mode}).")

    # ---- 2) Touched shards -- read, patch, save ----
    for shard in sorted(touched_shards):
        src_path = target_dir / shard
        out_path = (target_dir if in_place else output_dir) / shard
        tmp_path = out_path.with_name(out_path.name + ".tmp")

        # Drop a stale link/file before writing
        if not in_place and (out_path.exists() or out_path.is_symlink()):
            out_path.unlink()

        # Load full shard tensors + metadata
        tensors = {}
        with safe_open(src_path, framework="pt") as f:
            metadata = f.metadata() or {}
            for k in f.keys():
                tensors[k] = f.get_tensor(k)

        # Patch with draft tensors (with Llama->Gemma -1.0 conversion only
        # in legacy mode).
        n_patched = 0
        n_norm_converted = 0
        n_norm_verbatim = 0
        for dk, tk in shard_to_pairs[shard]:
            d_shard = draft_map[dk]
            with safe_open(draft_dir / d_shard, framework="pt") as f:
                t_new = f.get_tensor(dk)
            t_old = tensors[tk]
            if t_new.dtype != t_old.dtype:
                t_new = t_new.to(t_old.dtype)
            if t_new.shape != t_old.shape:
                raise RuntimeError(
                    f"Apply-time shape mismatch: {dk} {tuple(t_new.shape)} "
                    f"-> {tk} {tuple(t_old.shape)}"
                )
            if dk in NORM_DRAFT_KEYS:
                if legacy_llama_norm:
                    # Llama-style (training) -> Gemma-style (inference/sglang).
                    t_new = _convert_llama_to_gemma(t_new)
                    n_norm_converted += 1
                else:
                    # Both sides Gemma; verbatim copy.
                    n_norm_verbatim += 1
            tensors[tk] = t_new.contiguous()
            n_patched += 1

        # Self-convert RMSNorm slots that the draft did NOT save: the value
        # currently in `tensors[tk]` is the original target weight (Gemma) but
        # it was actually applied as a Llama-style scale at training time, so
        # we must rewrite it to (W - 1.0) here for inference-time consistency.
        # In default (non-legacy) mode this list is always empty (enforced by
        # validate() and asserted at the top of this function).
        n_self_converted = 0
        for tk in shard_to_self_convert.get(shard, ()):
            tensors[tk] = _convert_llama_to_gemma(tensors[tk])
            n_self_converted += 1

        # Some safetensors versions reject metadata=None / empty dict; only
        # forward it when non-empty so we don't accidentally inject keys.
        save_kwargs = {"metadata": metadata} if metadata else {}
        save_file(tensors, str(tmp_path), **save_kwargs)
        os.replace(tmp_path, out_path)
        if legacy_llama_norm:
            print(f"[apply] {shard}: patched {n_patched} tensor(s) "
                  f"(of which {n_norm_converted} RMSNorm Llama->Gemma) "
                  f"+ {n_self_converted} self-converted RMSNorm -> {out_path}")
        else:
            print(f"[apply] {shard}: patched {n_patched} tensor(s) "
                  f"(of which {n_norm_verbatim} RMSNorm copied verbatim, "
                  f"no conversion) -> {out_path}")

    # ---- 3) Auxiliary files (config.json, tokenizer.*, *.jinja, ...) ----
    if not in_place:
        for entry in sorted(target_dir.iterdir()):
            name = entry.name
            if name.endswith(".safetensors"):
                continue
            dst = output_dir / name
            if dst.exists() or dst.is_symlink():
                # Already placed (e.g. previous run); leave it.
                continue
            if name.endswith(".index.json"):
                # We never alter the key set, so the original index is correct.
                # Copy it (not symlink) so the user can freely inspect/edit.
                shutil.copy2(entry, dst)
            elif entry.is_dir():
                # E.g. iter_0000913/ -- symlink the whole dir.
                os.symlink(entry.resolve(), dst)
            else:
                _place(entry.resolve(), dst, mode=link_mode)


def _place(src: Path, dst: Path, mode: str):
    """Place src at dst using the requested strategy."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            # Cross-device fallback to symlink.
            os.symlink(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown link mode: {mode}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Merge a SpecForge MTP draft model back into the target's MTP head.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--draft",  required=True,
                   help="Path to draft (specforge MTP) checkpoint dir.")
    p.add_argument("--target", required=True,
                   help="Path to ORIGINAL target model dir (untouched unless --in-place).")
    p.add_argument("--output", default=None,
                   help="Output dir for the merged model. Required unless --in-place.")
    p.add_argument("--mtp-layer-idx", type=int, default=0,
                   help="Which MTP block to overwrite (default: 0).")
    p.add_argument("--num-experts", type=int, default=None,
                   help="Override num_experts (default: read draft config.json).")
    p.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"),
                   default="symlink",
                   help="How to place shards/files NOT touched by the merge "
                        "into the output dir (default: symlink).")
    p.add_argument("--in-place", action="store_true",
                   help="Patch the target safetensors in-place. POTENTIALLY DESTRUCTIVE.")
    p.add_argument("--dry-run", action="store_true",
                   help="Only run validation, do not write anything.")
    p.add_argument("--force", action="store_true",
                   help="Proceed even if validation reports fatal issues "
                        "(missing slots / shape mismatch). NOT recommended.")
    p.add_argument("--force-lm-head", action="store_true",
                   help="Force-copy lm_head.weight even when t2d signals a "
                        "vocab subset. ONLY use when you know the draft was "
                        "actually trained on the FULL vocab.")
    p.add_argument("--legacy-llama-norm", action="store_true",
                   help="Treat the draft as having been trained with the "
                        "legacy LlamaRMSNorm parametrization (y=w*x, weight "
                        "init=1).  Every RMSNorm weight will be rewritten "
                        "as W_target = W_draft - 1.0 so that the Gemma "
                        "inference forward (1+W)*x reproduces the training "
                        "scale.  For Gemma-style RMSNorm slots NOT saved by "
                        "the legacy draft, the target's own value is also "
                        "rewritten in-place via W -= 1.0.  USE ONLY for "
                        "checkpoints produced before the GemmaRMSNorm "
                        "switch; for new checkpoints the default mode "
                        "(verbatim copy) is correct.  Mutually-exclusive "
                        "with running migrate_specforge_norm_llama_to_gemma.py "
                        "first (you should do exactly one of the two).")
    args = p.parse_args()

    draft_dir = Path(args.draft).resolve()
    target_dir = Path(args.target).resolve()

    if args.in_place:
        if args.output:
            print("[main] --in-place set; ignoring --output.", file=sys.stderr)
        output_dir = target_dir
    else:
        if not args.output:
            p.error("--output is required unless --in-place is given.")
        output_dir = Path(args.output).resolve()
        if output_dir == target_dir:
            p.error("--output must differ from --target (or pass --in-place).")

    # Resolve num_experts from draft config if not overridden.
    if args.num_experts is None:
        cfg_path = draft_dir / "config.json"
        if not cfg_path.exists():
            p.error(f"--num-experts not given and no config.json at {cfg_path}")
        with cfg_path.open() as f:
            cfg = json.load(f)
        num_experts = cfg["num_experts"]
    else:
        num_experts = args.num_experts

    (plan, draft_map, target_map, _diag, fatal, lm_head_safe,
     norm_self_convert, _violations) = validate(
        draft_dir, target_dir, args.mtp_layer_idx, num_experts,
        legacy_llama_norm=args.legacy_llama_norm,
    )

    # Re-add lm_head if forced.
    if args.force_lm_head and not lm_head_safe and "lm_head.weight" in draft_map:
        tk = f"mtp.layers.{args.mtp_layer_idx}.shared_head.head.weight"
        if tk in target_map:
            d_shape, _ = get_meta(draft_dir, draft_map, "lm_head.weight")
            t_shape, _ = get_meta(target_dir, target_map, tk)
            if d_shape == t_shape:
                plan.append(("lm_head.weight", tk, d_shape))
                print("[main] --force-lm-head: re-added lm_head.weight to plan.")
            else:
                print(f"[main] --force-lm-head: shape mismatch "
                      f"draft={d_shape} target={t_shape}; refused.",
                      file=sys.stderr)

    if fatal and not args.force:
        print("\n[main] Validation FAILED; aborting. Pass --force to override "
              "(NOT recommended).", file=sys.stderr)
        sys.exit(1)

    print()
    print(f"[main] Plan: {len(plan)} tensor(s) will be overwritten in target MTP.")
    if args.dry_run:
        print("[main] --dry-run: not writing anything.")
        return

    if args.in_place:
        confirm = os.environ.get("MTP_MERGE_CONFIRM_INPLACE", "")
        if confirm != "yes":
            print("[main] --in-place safety guard: re-run with environment variable "
                  "MTP_MERGE_CONFIRM_INPLACE=yes to actually mutate the target.",
                  file=sys.stderr)
            sys.exit(2)

    apply_merge(
        plan, draft_map, target_map,
        draft_dir, target_dir, output_dir,
        in_place=args.in_place, link_mode=args.link_mode,
        norm_self_convert=norm_self_convert,
        legacy_llama_norm=args.legacy_llama_norm,
    )

    print()
    print(f"[main] DONE. Merged model is at: {output_dir}")
    if not args.in_place:
        n_links = sum(1 for s in set(target_map.values())
                      if s not in {target_map[tk] for _, tk, _ in plan})
        print(f"[main] {n_links} shard(s) reused from the original target via "
              f"{args.link_mode}; the source target model was NOT modified.")


if __name__ == "__main__":
    main()
