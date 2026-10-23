"""Compare current DFlash-family losses with fusion and checkpointing on/off.

Example: python -m specforge.benchmarks.benchmark_dflash_objective \
    --method dflash --draft-config configs/qwen3-8b-dflash.json

BF16 forward, FP32 losses, synthetic teacher features and frozen projections.
Times the full draft forward/backward and FP32-master AdamW step (lr=0).
Excludes teacher inference, data loading, DDP and backbone checkpointing.
"""

import argparse
import gc
import json
import math
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from transformers import Qwen3Config

from specforge.algorithms.common import dflash_family_model as family
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.dflash2 import DFlash2DraftModel
from specforge.modeling.draft.domino import DominoDraftModel
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.optimizer import BF16Optimizer


def build_model(args):
    torch.manual_seed(17)
    config = Qwen3Config.from_json_file(args.draft_config)
    config._attn_implementation = "sdpa"
    draft_class = {
        "dflash": DFlashDraftModel,
        "dspark": DSparkDraftModel,
        "domino": DominoDraftModel,
        "dflash2": DFlash2DraftModel,
    }[args.method]
    config.architectures = [draft_class.__name__]
    draft = draft_class(config)
    head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    embed = nn.Embedding(config.vocab_size, config.hidden_size)
    head.requires_grad_(False)
    embed.requires_grad_(False)
    # Concentrate synthetic candidate hits and exercise nonzero selector gradients.
    with torch.no_grad():
        head.weight[:128].mul_(4)
        if args.method == "dflash2":
            draft.candidate_selector.successor_codebook.normal_(std=0.02)
    common = dict(
        draft_model=draft,
        target_lm_head=head,
        target_embed_tokens=embed,
        mask_token_id=config.dflash_config["mask_token_id"],
        block_size=draft.block_size,
        num_anchors=args.anchors,
        objective_chunk_blocks=args.chunk,
        objective_checkpointing=False,
        attention_backend="sdpa",
    )
    if args.method == "dspark":
        model = family.OnlineDSparkModel(**common, fused_logits_loss=False)
    elif args.method == "domino":
        model = family.OnlineDominoModel(
            **common, shift_label=draft.shift_label, fused_logits_loss=False
        )
    else:
        model = family.OnlineDFlashModel(**common, fused_logits_loss=False)
    model.to(device="cuda", dtype=torch.bfloat16).train()
    assert not draft.is_gradient_checkpointing
    return model, config


def set_policy(model, *, checkpoint, fused):
    model.objective_checkpointing = checkpoint
    model.fused_logits_loss = fused
    probe = torch.empty(1, 128, device="cuda", dtype=torch.bfloat16)
    assert model._logits_loss_backend(probe) == ("fused" if fused else "eager")


def forward_backward(model, batch):
    model.zero_grad(set_to_none=True)
    torch.manual_seed(23)
    loss = model(**batch)[0]
    loss.backward()
    return loss.detach()


def gradient_snapshot(model):
    return {
        name: p.grad.detach().float().cpu()
        for name, p in model.named_parameters()
        if p.requires_grad and p.grad is not None
    }


def gradient_error(model, reference):
    error = norm = max_abs = 0.0
    names = set()
    selector_error = selector_norm = 0.0
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            names.add(name)
            value = p.grad.detach().float().cpu()
            diff = value - reference[name]
            sq = diff.square().sum().item()
            ref_sq = reference[name].square().sum().item()
            error += sq
            norm += ref_sq
            max_abs = max(max_abs, diff.abs().max().item())
            if "candidate_selector" in name:
                selector_error += sq
                selector_norm += ref_sq
    assert names == reference.keys(), "Different gradient connectivity"
    return dict(
        gradient_relative_l2=(error / max(norm, 1e-30)) ** 0.5,
        gradient_max_abs=max_abs,
        selector_gradient_relative_l2=(selector_error / max(selector_norm, 1e-30))
        ** 0.5,
        selector_reference_gradient_norm=selector_norm**0.5,
    )


def measure(args, model, batch, optimizer, reference_loss, reference_grads):
    loss = forward_backward(model, batch)
    parity = gradient_error(model, reference_grads)
    parity.update(
        loss=loss.item(),
        reference_loss=reference_loss,
        loss_abs_error=abs(loss.item() - reference_loss),
    )

    def step():
        forward_backward(model, batch)
        optimizer.step()

    for _ in range(args.warmup):
        step()
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    resident = torch.cuda.memory_allocated()
    timings = []
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.steps):
            step()
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000 / args.steps)
    return dict(
        step_ms=statistics.median(timings),
        repeat_step_ms=timings,
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
        transient_peak_mib=(torch.cuda.max_memory_allocated() - resident) / 2**20,
        **parity,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=["dflash", "dspark", "domino", "dflash2"], required=True
    )
    parser.add_argument("--draft-config", required=True, help="Draft model JSON config")
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--anchors", type=int, default=512)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--artifacts", default=".benchmark-results/dflash-objective")
    args = parser.parse_args()
    if (
        args.sequence < 2
        or min(args.anchors, args.warmup, args.steps, args.repeats) < 1
        or args.chunk < 0
    ):
        parser.error(
            "sequence >= 2, anchors/warmup/steps/repeats >= 1 and chunk >= 0 required"
        )
    torch.set_num_threads(4)
    output = Path(args.artifacts)
    output.mkdir(parents=True, exist_ok=True)
    model, config = build_model(args)
    print(
        json.dumps(
            dict(
                settings=vars(args),
                config=config.to_dict(),
                torch=torch.__version__,
                cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(),
            )
        ),
        flush=True,
    )
    torch.manual_seed(19)
    batch = dict(
        input_ids=torch.randint(
            min(128, config.vocab_size), (1, args.sequence), device="cuda"
        ),
        hidden_states=torch.randn(
            1,
            args.sequence,
            len(model.draft_model.target_layer_ids) * config.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        ),
        loss_mask=torch.ones(1, args.sequence, device="cuda"),
        max_valid_anchors=args.sequence - 1,
        collect_detailed_metrics=False,
    )
    if args.method == "dspark":
        batch["target_last_hidden_states"] = torch.randn(
            1, args.sequence, config.hidden_size, device="cuda", dtype=torch.bfloat16
        )
    if args.method == "domino":
        # Exercise both base and corrected CE gradients during the schedule.
        batch["lambda_base"] = 0.5
    optimizer = BF16Optimizer(
        model, lr=0.0, max_grad_norm=1.0, warmup_ratio=0.0, lr_scheduler="constant"
    )
    set_policy(model, checkpoint=False, fused=False)
    reference_loss = forward_backward(model, batch)
    reference_loss = reference_loss.item()
    reference_grads = gradient_snapshot(model)
    assert math.isfinite(reference_loss)
    assert all(torch.isfinite(grad).all() for grad in reference_grads.values())
    print("Reference ready", flush=True)
    for checkpoint, fused in (
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ):
        set_policy(model, checkpoint=checkpoint, fused=fused)
        row = dict(
            method=args.method,
            checkpoint=checkpoint,
            fused=fused,
            sequence=args.sequence,
            anchors=args.anchors,
            chunk=args.chunk,
            block=model.block_size,
            hidden=config.hidden_size,
            layers=config.num_hidden_layers,
            vocab=config.vocab_size,
            trainable_parameters=sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        )
        print(f"Measuring checkpoint={checkpoint}, fused={fused}", flush=True)
        row.update(
            measure(args, model, batch, optimizer, reference_loss, reference_grads)
        )
        assert all(
            math.isfinite(row[key])
            for key in ("loss", "gradient_relative_l2", "gradient_max_abs")
        )
        print(json.dumps(row), flush=True)
        with (output / f"{args.method}.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
