"""Full synthetic draft training steps, including projection, loss and optimizer.

No teacher inference or data loading is timed. A zero learning rate keeps all
variants at identical weights while still running the optimizer kernels.
"""

import argparse
import gc
import json
import os
import statistics
import time
from contextlib import nullcontext
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import (
    OnlineDFlashModel,
    OnlineDSparkModel,
)
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.dspark import DSparkDraftModel


def build_model(args, scenario):
    torch.manual_seed(17)
    dspark = scenario.startswith("dspark")
    config = Qwen3Config(
        hidden_size=args.hidden,
        intermediate_size=2 * args.hidden,
        num_attention_heads=args.hidden // 64,
        num_key_value_heads=4,
        num_hidden_layers=2,
        num_target_layers=4,
        head_dim=64,
        vocab_size=args.vocab,
        max_position_embeddings=4096,
        layer_types=["full_attention"] * 2,
        block_size=8,
        dflash_config={
            "block_size": 8,
            "target_layer_ids": [0, 2],
            "mask_token_id": args.vocab - 1,
            "projector_type": "dspark" if dspark else "linear",
            "markov_head_type": "vanilla",
            "markov_rank": 8,
            "enable_confidence_head": False,
        },
        attn_implementation="sdpa",
    )
    draft = DSparkDraftModel(config) if dspark else DFlashDraftModel(config)
    head = nn.Linear(args.hidden, args.vocab, bias=False).requires_grad_(False)
    embed = nn.Embedding(args.vocab, args.hidden).requires_grad_(False)
    kwargs = dict(
        draft_model=draft,
        target_lm_head=head,
        target_embed_tokens=embed,
        mask_token_id=args.vocab - 1,
        block_size=8,
        num_anchors=args.anchors,
        attention_backend="sdpa",
        objective_chunk_blocks=args.chunk,
    )
    if dspark:
        model = OnlineDSparkModel(
            **kwargs,
            dspark_ce_loss_alpha=0.1 if scenario == "dspark-l1" else 1.0,
            dspark_l1_loss_alpha=0.9 if scenario == "dspark-l1" else 0.0,
            dspark_confidence_head_alpha=0.0,
            fused_logits_loss=True,
        )
    else:
        model = OnlineDFlashModel(**kwargs, fused_logits_loss=True)
    return model.to(device="cuda", dtype=torch.bfloat16)


def gradients(model, batch):
    model.zero_grad(set_to_none=True)
    torch.manual_seed(23)
    loss = model(**batch)[0]
    loss.backward()
    return loss.item(), {
        name: p.grad.detach().float().cpu()
        for name, p in model.named_parameters()
        if p.grad is not None
    }


def run(args, scenario):
    model = build_model(args, scenario)
    torch.manual_seed(19)
    batch = dict(
        input_ids=torch.randint(args.vocab - 1, (1, 512), device="cuda"),
        hidden_states=torch.randn(
            1, 512, 2 * args.hidden, device="cuda", dtype=torch.bfloat16
        ),
        loss_mask=torch.ones(1, 512, device="cuda"),
        target_last_hidden_states=torch.randn(
            1, 512, args.hidden, device="cuda", dtype=torch.bfloat16
        ),
        collect_detailed_metrics=False,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.0, foreach=True
    )
    original_ce = F.cross_entropy

    def fp32_ce(logits, *positional, **keyword):
        return original_ce(logits.float(), *positional, **keyword)

    model.fused_logits_loss = False
    with (
        patch.dict(os.environ, {"SPECFORGE_DSPARK_FUSED_LOSS": "0"}),
        patch.object(F, "cross_entropy", fp32_ce),
    ):
        reference_loss, reference_grads = gradients(model, batch)
    for variant in args.variants:
        model.fused_logits_loss = variant == "fused"
        context = (
            patch.object(F, "cross_entropy", fp32_ce)
            if variant == "fp32"
            else nullcontext()
        )
        with context, patch.dict(os.environ, {"SPECFORGE_DSPARK_FUSED_LOSS": "0"}):
            loss, grads = gradients(model, batch)
            squared_error = sum(
                (grads[n] - g).square().sum().item() for n, g in reference_grads.items()
            )
            squared_norm = sum(
                g.square().sum().item() for g in reference_grads.values()
            )
            max_error = max(
                (grads[n] - g).abs().max().item() for n, g in reference_grads.items()
            )
            del grads

            def step():
                optimizer.zero_grad(set_to_none=True)
                model(**batch)[0].backward()
                optimizer.step()

            for _ in range(args.warmup):
                step()
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            resident = torch.cuda.memory_allocated()
            timings = []
            for _ in range(3):
                start = time.perf_counter()
                for _ in range(args.steps):
                    step()
                torch.cuda.synchronize()
                timings.append((time.perf_counter() - start) / args.steps)
            seconds = statistics.median(timings)
            print(
                json.dumps(
                    dict(
                        scenario=scenario,
                        variant=variant,
                        hidden=args.hidden,
                        vocab=args.vocab,
                        anchors=args.anchors,
                        chunk=args.chunk,
                        step_ms=seconds * 1000,
                        steps_per_second=1 / seconds,
                        objective_slots_per_second=args.anchors
                        * (7 if scenario == "dflash-ce" else 8)
                        / seconds,
                        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                        activation_peak_mib=(
                            torch.cuda.max_memory_allocated() - resident
                        )
                        / 2**20,
                        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                        loss=loss,
                        fp32_reference_loss=reference_loss,
                        gradient_relative_l2=(squared_error / max(squared_norm, 1e-30))
                        ** 0.5,
                        gradient_max_abs=max_error,
                    )
                ),
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--vocab", type=int, default=151936)
    parser.add_argument("--anchors", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--variants", nargs="+", choices=["fp32", "fused"], default=["fp32", "fused"]
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=["dflash-ce", "dspark-ce", "dspark-l1"]
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(
        json.dumps(
            dict(
                torch=torch.__version__,
                cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(),
                scope="synthetic full draft train step",
            )
        ),
        flush=True,
    )
    for scenario in args.scenarios:
        run(args, scenario)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
