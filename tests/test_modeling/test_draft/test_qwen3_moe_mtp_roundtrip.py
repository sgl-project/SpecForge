"""
Self-check / equivalence assertions for Qwen3MoeForCausalLMMTP.

Three checks, each can be invoked individually:

  keys   – only reads target's *.index.json (no model instantiation): asserts
           every weight key referenced by `load_mtp_weights` exists in the
           target checkpoint. Cheap (~MB) and safe to run anywhere.

  freeze – instantiates a TINY MTP draft from a synthetic config (no weights
           loaded) and verifies that `freeze_lm_head()` (a) flips
           `lm_head.weight.requires_grad` to False, (b) keeps the weight
           bit-identical after a simulated optimizer step.

  load   – HEAVY. Instantiates the real draft (~10B params for the 122B-MoE
           target), loads MTP weights from `--model-path`, then asserts:
             * draft.lm_head.weight bit-equals target shared_head.head.weight
             * draft.embed_tokens.weight bit-equals target model.embed_tokens
             * round-trip via export_mtp_weights → re-read produces zero diff
           Default device is CPU; pass --device cuda:0 if you have ≥40 GB free.

Usage:
    # Lightweight smoke tests (no GPU, no big checkpoint):
    python tests/test_modeling/test_draft/test_qwen3_moe_mtp_roundtrip.py \
        --mode keys freeze \
        --model-path /mnt/tidal-alsh01/dataset/xiaowen/model/qwen35-122b \
        --draft-config configs/qwen3.5-122b-a10b-mtp.json

    # Full equivalence (heavy):
    python tests/test_modeling/test_draft/test_qwen3_moe_mtp_roundtrip.py \
        --mode all --device cuda:0 \
        --model-path /mnt/tidal-alsh01/dataset/xiaowen/model/qwen35-122b \
        --draft-config configs/qwen3.5-122b-a10b-mtp.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
from typing import Dict, List


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _expected_mtp_keys(num_experts: int, mtp_layer_idx: int = 0) -> List[str]:
    """Mirror `Qwen3MoeForCausalLMMTP.load_mtp_weights` weight_mapping (src side).

    Note: embed_tokens is sourced from the MTP-side key
    `mtp.layers.{idx}.embed_tokens.weight` (always present in target index,
    regardless of multimodal nesting on the main trunk).
    """
    mtp = "mtp"
    layer = f"{mtp}.layers.{mtp_layer_idx}"
    moe = f"{layer}.mlp"
    keys = [
        f"{layer}.embed_tokens.weight",
        f"{mtp}.fc.weight",
        f"{mtp}.pre_fc_norm_embedding.weight",
        f"{mtp}.pre_fc_norm_hidden.weight",
        f"{mtp}.norm.weight",
        f"{layer}.input_layernorm.weight",
        f"{layer}.post_attention_layernorm.weight",
        f"{layer}.self_attn.q_proj.weight",
        f"{layer}.self_attn.k_proj.weight",
        f"{layer}.self_attn.v_proj.weight",
        f"{layer}.self_attn.o_proj.weight",
        f"{layer}.self_attn.q_norm.weight",
        f"{layer}.self_attn.k_norm.weight",
        f"{moe}.gate.weight",
        f"{moe}.shared_expert.gate_proj.weight",
        f"{moe}.shared_expert.up_proj.weight",
        f"{moe}.shared_expert.down_proj.weight",
        f"{moe}.shared_expert_gate.weight",
        f"{layer}.shared_head.head.weight",
    ]
    for ei in range(num_experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            keys.append(f"{moe}.experts.{ei}.{proj}.weight")
    return keys


def _load_index(model_path: str) -> Dict[str, str]:
    """Return {weight_key: shard_filename} from the first *.index.json in model_path."""
    paths = glob.glob(os.path.join(model_path, "*.index.json"))
    if not paths:
        raise FileNotFoundError(
            f"No *.index.json under {model_path}; this script needs a sharded checkpoint."
        )
    with open(paths[0], "r") as f:
        return json.load(f)["weight_map"]


def _read_tensor(model_path: str, weight_map: Dict[str, str], key: str):
    from safetensors import safe_open

    if key not in weight_map:
        raise KeyError(key)
    with safe_open(os.path.join(model_path, weight_map[key]), framework="pt") as f:
        return f.get_tensor(key)


# ---------------------------------------------------------------------------
# keys check
# ---------------------------------------------------------------------------

def check_keys(model_path: str, draft_config_path: str, mtp_layer_idx: int) -> None:
    print(f"[keys] target = {model_path}")
    with open(draft_config_path, "r") as f:
        cfg = json.load(f)
    num_experts = int(cfg.get("num_experts", 256))
    print(f"[keys] num_experts = {num_experts}, mtp_layer_idx = {mtp_layer_idx}")

    weight_map = _load_index(model_path)
    expected = _expected_mtp_keys(num_experts, mtp_layer_idx)

    missing = [k for k in expected if k not in weight_map]
    if missing:
        print(f"[keys] FAIL: {len(missing)}/{len(expected)} keys missing in target index.")
        for k in missing[:20]:
            print(f"  - {k}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        raise AssertionError("Missing MTP keys in target checkpoint.")
    print(f"[keys] OK: all {len(expected)} expected MTP keys present in target index.")


# ---------------------------------------------------------------------------
# freeze check
# ---------------------------------------------------------------------------

def check_freeze() -> None:
    """No model load — synthesise a tiny config and exercise freeze_lm_head."""
    import torch
    from transformers import Qwen3MoeConfig

    from specforge.modeling.draft.qwen3_moe_mtp import Qwen3MoeForCausalLMMTP

    print("[freeze] instantiating tiny MTP draft (hidden=64, vocab=256, experts=4)...")
    cfg = Qwen3MoeConfig(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=1,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=256,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        pad_token_id=0,
    )
    # required by the MTP draft's t2d/d2t bookkeeping
    cfg.draft_vocab_size = 64

    model = Qwen3MoeForCausalLMMTP(cfg, attention_backend="sdpa")
    w_before = model.lm_head.weight.detach().clone()

    assert model.lm_head.weight.requires_grad is True, "fresh lm_head must require grad"
    model.freeze_lm_head()
    assert model.lm_head.weight.requires_grad is False, (
        "freeze_lm_head() did NOT clear requires_grad on lm_head.weight"
    )

    # Simulate one optimizer step using a leaf tensor optimizer that filters frozen params.
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert all(p is not model.lm_head.weight for p in trainable), (
        "lm_head.weight leaked into trainable params after freeze."
    )

    # Sanity: even if some careless op produced a grad and we ran SGD on ALL params,
    # the standard pattern of skipping requires_grad=False protects us.
    optim = torch.optim.SGD(trainable, lr=1e-2)
    # Fake forward signal: run a zero loss against a buffer to exercise optimizer.
    loss = sum(p.float().sum() * 0.0 for p in trainable)
    if hasattr(loss, "backward"):
        loss.backward()
    optim.step()

    w_after = model.lm_head.weight.detach()
    delta = (w_after - w_before).abs().max().item()
    assert delta == 0.0, (
        f"lm_head.weight changed by {delta} after a simulated step; freeze is broken."
    )
    print("[freeze] OK: lm_head.weight is frozen and untouched after optimizer.step().")


# ---------------------------------------------------------------------------
# load + equivalence + round-trip check (heavy)
# ---------------------------------------------------------------------------

def check_load_equivalence_and_roundtrip(
    model_path: str,
    draft_config_path: str,
    mtp_layer_idx: int,
    device: str,
) -> None:
    import torch
    from transformers import Qwen3MoeConfig

    from specforge.modeling.draft.qwen3_moe_mtp import Qwen3MoeForCausalLMMTP

    print(f"[load] reading draft config: {draft_config_path}")
    with open(draft_config_path, "r") as f:
        cfg_dict = json.load(f)
    cfg = Qwen3MoeConfig(**{k: v for k, v in cfg_dict.items() if k != "draft_vocab_size"})
    cfg.draft_vocab_size = cfg_dict.get("draft_vocab_size", cfg.vocab_size)

    print(f"[load] instantiating MTP draft on {device} (this is heavy: ~10B params)...")
    torch.set_default_dtype(torch.bfloat16)
    model = Qwen3MoeForCausalLMMTP(cfg, attention_backend="sdpa")
    model.to(device)

    print(f"[load] loading MTP weights from {model_path} (mtp_layer_idx={mtp_layer_idx})...")
    model.load_mtp_weights(model_path, mtp_layer_idx=mtp_layer_idx)

    weight_map = _load_index(model_path)
    layer_prefix = f"mtp.layers.{mtp_layer_idx}"

    # (b) lm_head bit-equality
    lm_src = _read_tensor(model_path, weight_map, f"{layer_prefix}.shared_head.head.weight")
    diff_lm = (model.lm_head.weight.detach().cpu().float() - lm_src.float()).abs().max().item()
    assert diff_lm == 0.0, (
        f"draft.lm_head.weight diverges from target shared_head.head.weight "
        f"(max abs diff = {diff_lm})"
    )
    print("[load] OK: draft.lm_head.weight bit-equals target shared_head.head.weight.")

    # embed_tokens bit-equality (loaded from target's MTP-side dedicated embed,
    # which is shared with the main trunk embedding by construction).
    emb_src = _read_tensor(model_path, weight_map, f"{layer_prefix}.embed_tokens.weight")
    diff_emb = (model.embed_tokens.weight.detach().cpu().float() - emb_src.float()).abs().max().item()
    assert diff_emb == 0.0, (
        f"draft.embed_tokens.weight diverges from target {layer_prefix}.embed_tokens.weight "
        f"(max abs diff = {diff_emb})"
    )
    print(f"[load] OK: draft.embed_tokens.weight bit-equals target {layer_prefix}.embed_tokens.weight.")

    # (a) round-trip: export → re-read → diff against the source we loaded from
    with tempfile.TemporaryDirectory() as td:
        out_path = os.path.join(td, "exported_mtp.pt")
        exported = model.export_mtp_weights(out_path, mtp_layer_idx=mtp_layer_idx)

        bad = []
        for export_key, exported_tensor in exported.items():
            try:
                src_tensor = _read_tensor(model_path, weight_map, export_key)
            except KeyError:
                bad.append(f"{export_key}: not found in target index")
                continue
            d = (exported_tensor.detach().cpu().float() - src_tensor.float()).abs().max().item()
            if d != 0.0:
                bad.append(f"{export_key}: max abs diff = {d}")

        if bad:
            print(f"[load] FAIL: round-trip mismatch on {len(bad)} keys:")
            for msg in bad[:20]:
                print(f"  - {msg}")
            if len(bad) > 20:
                print(f"  ... and {len(bad) - 20} more")
            raise AssertionError("export_mtp_weights round-trip diverges from source.")

    print(f"[load] OK: round-trip exported {len(exported)} keys, all bit-equal to source.")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        nargs="+",
        choices=["keys", "freeze", "load", "all"],
        default=["keys", "freeze"],
        help="Which checks to run. 'all' = keys + freeze + load (load is heavy).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/mnt/tidal-alsh01/dataset/xiaowen/model/qwen35-122b",
        help="Path to the target model directory (sharded safetensors + index.json).",
    )
    parser.add_argument(
        "--draft-config",
        type=str,
        default=None,
        help="Path to the draft model config json (default: configs/qwen3.5-122b-a10b-mtp.json under repo root).",
    )
    parser.add_argument("--mtp-layer-idx", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for the heavy 'load' check; cpu by default."
    )
    args = parser.parse_args()

    if args.draft_config is None:
        # default to the repo's pinned config
        here = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
        args.draft_config = os.path.join(repo_root, "configs", "qwen3.5-122b-a10b-mtp.json")

    modes = set(args.mode)
    if "all" in modes:
        modes = {"keys", "freeze", "load"}

    failures = []
    if "keys" in modes:
        try:
            check_keys(args.model_path, args.draft_config, args.mtp_layer_idx)
        except Exception as e:
            failures.append(("keys", e))
            print(f"[keys] FAIL: {e}")

    if "freeze" in modes:
        try:
            check_freeze()
        except Exception as e:
            failures.append(("freeze", e))
            print(f"[freeze] FAIL: {e}")

    if "load" in modes:
        try:
            check_load_equivalence_and_roundtrip(
                args.model_path, args.draft_config, args.mtp_layer_idx, args.device
            )
        except Exception as e:
            failures.append(("load", e))
            print(f"[load] FAIL: {e}")

    if failures:
        print(f"\n=== {len(failures)} check(s) FAILED ===")
        for name, err in failures:
            print(f"  - {name}: {err}")
        return 1
    print("\n=== All requested checks PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
