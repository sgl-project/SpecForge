#!/usr/bin/env python3
"""Build a DSpark MoE drafter export for SGLang serving-speed tests.

Takes a dense DSpark drafter export (e.g. ``RadixArk/Qwen3.8-27B-DSpark``),
keeps every non-FFN tensor (attention, norms, fc projector, Markov and
confidence heads) and replaces each layer's dense MLP with randomly
initialised MoE tensors in the Qwen naming that SGLang's ``Qwen3MoeDSparkModel``
(``specforge/serving/sglang_models/dflash_moe.py``) loads::

    <layer>.mlp.gate.weight                 [E, hidden]  fp32
    <layer>.mlp.gate.bias                   [E]          fp32   (zeros, --router-bias)
    <layer>.mlp.experts.{i}.gate_proj.weight [w, hidden]
    <layer>.mlp.experts.{i}.up_proj.weight   [w, hidden]
    <layer>.mlp.experts.{i}.down_proj.weight [hidden, w]
    <layer>.mlp.shared_expert.{gate,up}_proj.weight [s, hidden]
    <layer>.mlp.shared_expert.down_proj.weight      [hidden, s]
    <layer>.mlp.shared_expert_gate.weight           [1, hidden]

The experts are random, so the drafter's accept length is meaningless; run
the server with ``SGLANG_SIMULATE_ACC_LEN`` to compare step time against the
dense drafter at a fixed accept length (see speed_test_moe_draft.sh).

Example::

    python scripts/moe_speed/make_moe_draft_from_dense.py \
        --src RadixArk/Qwen3.8-27B-DSpark --out exports/qwen38-dspark-moe-16x1024-rand \
        --experts 16 --width 1024 --topk 2 --shared 2048
    # same weights, different top-k (config.json only, safetensors symlinked):
    python scripts/moe_speed/make_moe_draft_from_dense.py \
        --src RadixArk/Qwen3.8-27B-DSpark --out exports/qwen38-dspark-moe-16x1024-rand-top4 \
        --experts 16 --width 1024 --topk 4 --shared 2048 \
        --reuse-weights-from exports/qwen38-dspark-moe-16x1024-rand
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

_MLP_KEY = re.compile(
    r"^(?P<layer>(?:model\.)?layers\.(?P<idx>\d+)\.)mlp\.(?P<proj>gate_proj|up_proj|down_proj|gate_up_proj)\.weight$"
)


def resolve_src(src: str) -> Path:
    p = Path(src)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(src, allow_patterns=["*.json", "*.safetensors", "*.py"])
    )


def load_src_tensors(src: Path) -> "OrderedDict[str, torch.Tensor]":
    files = sorted(src.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"no safetensors in {src}")
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for f in files:
        out.update(load_file(str(f)))
    return out


def build_moe_tensors(
    layer_prefix: str,
    hidden: int,
    experts: int,
    width: int,
    shared: int,
    std: float,
    dtype: torch.dtype,
    gen: torch.Generator,
    router_bias: bool,
) -> "OrderedDict[str, torch.Tensor]":
    def normal(*shape):
        return (torch.randn(*shape, generator=gen) * std).to(dtype)

    t: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    base = f"{layer_prefix}mlp."
    # Router in fp32 (the serving block computes logits in fp32).
    t[base + "gate.weight"] = (torch.randn(experts, hidden, generator=gen) * std).float()
    if router_bias:
        t[base + "gate.bias"] = torch.zeros(experts, dtype=torch.float32)
    for i in range(experts):
        t[f"{base}experts.{i}.gate_proj.weight"] = normal(width, hidden)
        t[f"{base}experts.{i}.up_proj.weight"] = normal(width, hidden)
        t[f"{base}experts.{i}.down_proj.weight"] = normal(hidden, width)
    if shared > 0:
        t[base + "shared_expert.gate_proj.weight"] = normal(shared, hidden)
        t[base + "shared_expert.up_proj.weight"] = normal(shared, hidden)
        t[base + "shared_expert.down_proj.weight"] = normal(hidden, shared)
        t[base + "shared_expert_gate.weight"] = normal(1, hidden)
    return t


def write_sharded(tensors: "OrderedDict[str, torch.Tensor]", out: Path, layers: list[str]) -> None:
    """One shard for the non-layer tensors, one per layer, plus the HF index."""
    groups: list[tuple[str, OrderedDict]] = []
    rest = OrderedDict((k, v) for k, v in tensors.items() if not any(k.startswith(p) for p in layers))
    groups.append(("misc", rest))
    for p in layers:
        groups.append((p.rstrip(".").replace(".", "_"), OrderedDict((k, v) for k, v in tensors.items() if k.startswith(p))))
    n = len(groups)
    weight_map = {}
    total = 0
    for i, (name, group) in enumerate(groups, 1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        save_file({k: v.contiguous() for k, v in group.items()}, str(out / fname), metadata={"format": "pt"})
        for k, v in group.items():
            weight_map[k] = fname
            total += v.numel() * v.element_size()
    with open(out / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="dense DSpark drafter: HF repo id or local dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--width", type=int, default=1024, help="moe_intermediate_size")
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--shared", type=int, default=2048, help="shared_expert_intermediate_size (0 = none)")
    ap.add_argument("--std", type=float, default=None, help="init std (default: config initializer_range)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-router-bias", action="store_true", help="omit gate.bias / moe_router_bias=false")
    ap.add_argument("--reuse-weights-from", default=None, help="symlink safetensors from this export; write config only")
    args = ap.parse_args()

    src = resolve_src(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(src / "config.json"))
    hidden = int(cfg["hidden_size"])
    std = args.std if args.std is not None else float(cfg.get("initializer_range", 0.02))
    router_bias = not args.no_router_bias

    if args.reuse_weights_from:
        ref = Path(args.reuse_weights_from).resolve()
        for f in list(ref.glob("*.safetensors")) + [ref / "model.safetensors.index.json"]:
            dst = out / f.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(f, dst)
    else:
        tensors = load_src_tensors(src)
        dtype = next(v.dtype for k, v in tensors.items() if k.endswith("self_attn.q_proj.weight"))
        layers: list[str] = []
        for k in list(tensors):
            m = _MLP_KEY.match(k)
            if m is None:
                continue
            if m["layer"] not in layers:
                layers.append(m["layer"])
            del tensors[k]
        if not layers:
            raise SystemExit("no dense MLP tensors found; is --src a dense DSpark/DFlash export?")
        gen = torch.Generator().manual_seed(args.seed)
        for p in layers:
            tensors.update(
                build_moe_tensors(p, hidden, args.experts, args.width, args.shared, std, dtype, gen, router_bias)
            )
        layers_sorted = sorted(layers, key=lambda s: int(re.search(r"(\d+)\.$", s).group(1)))
        write_sharded(tensors, out, layers_sorted)
        total = sum(v.numel() for v in tensors.values())
        print(f"wrote {len(tensors)} tensors, {total/1e9:.2f}B params, {len(layers)} MoE layers -> {out}")

    # config.json: dense DSpark config + the MoE keys the serving block reads.
    for k in ("auto_map",):
        cfg.pop(k, None)
    cfg["architectures"] = ["Qwen3MoeDSparkModel"]
    cfg["moe_preset"] = "qwen3_5_moe"
    cfg["n_routed_experts"] = args.experts
    cfg["num_experts"] = args.experts
    cfg["num_experts_per_tok"] = args.topk
    cfg["moe_intermediate_size"] = args.width
    cfg["n_shared_experts"] = 1 if args.shared > 0 else 0
    cfg["shared_expert_intermediate_size"] = args.shared
    cfg["shared_expert_gate"] = "sigmoid" if args.shared > 0 else "none"
    cfg["scoring_func"] = "softmax"
    cfg["norm_topk_prob"] = True
    cfg["routed_scaling_factor"] = 1.0
    cfg["n_group"] = 1
    cfg["topk_group"] = 1
    cfg["topk_method"] = "greedy"
    cfg["moe_router_bias"] = router_bias
    with open(out / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    for extra in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"):
        if (src / extra).exists() and not (out / extra).exists():
            shutil.copy(src / extra, out / extra)
    print(f"config: E={args.experts} w={args.width} top-{args.topk} shared={args.shared} router_bias={router_bias} -> {out/'config.json'}")


if __name__ == "__main__":
    main()
