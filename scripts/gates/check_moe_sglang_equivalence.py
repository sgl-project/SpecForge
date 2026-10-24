"""Served vs trained MoE FFN: SGLang's sparse block against SpecForge's ``MoELayer``.

Loads every layer's FFN from one normalized HF export twice: into the block
``specforge.serving.sglang_models`` serves for the export's ``moe_preset``
(its router, ``FusedMoE`` and the shared expert) and into the training layer of
that preset in eval mode; a trained ``gate.expert_bias`` loads into both.
Feeds both the same inputs and compares the chosen expert sets and the outputs.
Needs one GPU and the SGLang the draft will be served with.

    python scripts/gates/check_moe_sglang_equivalence.py --export <normalized export dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch


def sglang_runtime() -> None:
    """Server args plus a TP1 group, the way a scheduler process has them."""
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.layers.moe.utils import initialize_moe_config
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    server_args = ServerArgs(model_path="dummy", attention_backend="triton")
    publish(server_args, role="scheduler")
    set_global_server_args_for_scheduler(server_args)
    initialize_moe_config(server_args)
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method="tcp://127.0.0.1:29599",
        local_rank=0,
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", required=True)
    parser.add_argument("--tokens", type=int, default=4096)
    args = parser.parse_args()

    sglang_runtime()
    from safetensors.torch import load_file
    from transformers import AutoConfig

    from specforge.modeling.draft.moe.families import FAMILIES
    from specforge.serving.sglang_models.dflash_moe import SPARSE_BLOCKS

    with open(os.path.join(args.export, "config.json"), encoding="utf-8") as handle:
        fields = json.load(handle)
    # The export names the serving class; the model_type's config class fills defaults.
    config = AutoConfig.for_model(fields.pop("model_type", "qwen3"), **fields)
    preset = getattr(config, "moe_preset", None)
    if preset not in FAMILIES or preset not in SPARSE_BLOCKS:
        print(f"RESULT: FAIL (moe_preset={preset!r} is not both trained and served)")
        return 1
    build_moe, sparse_block = FAMILIES[preset], SPARSE_BLOCKS[preset]

    device, dtype = "cuda", torch.bfloat16
    state = {}
    for name in sorted(os.listdir(args.export)):
        if name.endswith(".safetensors"):
            state.update(load_file(os.path.join(args.export, name)))

    ok = True
    for layer in range(int(config.num_hidden_layers)):
        prefix = f"layers.{layer}.mlp."
        ffn_state = {
            k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)
        }

        trained = build_moe(config).to(device, dtype).eval()
        trained.load_state_dict(ffn_state, strict=True)

        previous = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            with torch.device(device):
                served = sparse_block(config, layer_id=layer, prefix=prefix[:-1])
        finally:
            torch.set_default_dtype(previous)
        served.load_weights([(k, v.to(device)) for k, v in ffn_state.items()])
        served.check_loaded(f"layer {layer}")
        served.experts.quant_method.process_weights_after_loading(served.experts)
        served.eval()

        generator = torch.Generator(device="cpu").manual_seed(layer)
        norm_weight = state[f"layers.{layer}.post_attention_layernorm.weight"].float()
        x = torch.randn(args.tokens, config.hidden_size, generator=generator)
        x = (x * norm_weight).to(device, dtype)
        with torch.no_grad():
            y_trained = trained(x.unsqueeze(0)).squeeze(0).float()
            y_served = served(x).float()
            picks_trained = trained.gate(x)[1].sort(dim=-1).values
            picks_served = served.selected_experts(x).sort(dim=-1).values
        same = (picks_trained == picks_served).all(dim=-1)
        agree = same.float().mean().item()
        scale = y_trained.norm(dim=-1).clamp_min(1e-6)
        rel = (y_trained - y_served).norm(dim=-1) / scale
        rel_same = rel[same].mean().item() if same.any() else float("nan")
        load = torch.bincount(
            picks_served.flatten(), minlength=served.num_experts
        ).tolist()
        layer_ok = agree >= 0.995 and rel_same < 0.02
        ok &= layer_ok
        print(
            f"layer {layer}: top-{picks_served.shape[-1]} set agreement {agree:.4f}, "
            f"mean relative |dy| on agreeing tokens {rel_same:.2e}, expert load "
            f"(served) {load} -> {'ok' if layer_ok else 'FAIL'}"
        )
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
