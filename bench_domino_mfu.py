"""Isolated single-GPU MFU / FLOPs / memory benchmark for the domino trainer.

Removes the data-supply path and DP comm so we measure ONLY the per-step trainer
compute: build the real draft (specforge/config/draft/presets/qwen3-8b-domino.yaml shapes) with random
weights on one GPU, feed a realistic batch, time fwd+bwd with CUDA events, count
forward FLOPs with torch's FlopCounterMode, and read peak allocated memory.

Reports achieved TFLOP/s and MFU vs H200 bf16 dense peak (989.5 TFLOP/s) for a
sweep of batch sizes, plus per-sample compute (to show it is ~flat in batch =
compute-latency bound, not FLOPs/memory bound).
"""

from dataclasses import replace
from pathlib import Path

import torch
import yaml
from torch.utils.flop_counter import FlopCounterMode

from specforge.algorithms.objectives import build_objective_model
from specforge.config.draft.codec import parse_draft
from specforge.modeling import build_draft_model

H200_BF16_TFLOPS = 989.5  # SXM, dense
SEQ = 768
NUM_ANCHORS = 256
CFG_PATH = "specforge/config/draft/presets/qwen3-8b-domino.yaml"


def build(dtype, device):
    cfg = parse_draft(yaml.safe_load(Path(CFG_PATH).read_text()))
    cfg = replace(cfg, target_bindings=(), target_logit_transform=None)
    draft = build_draft_model(cfg, backend="sdpa", device=device, dtype=dtype)
    draft.embed_tokens.requires_grad_(False)
    draft.lm_head.requires_grad_(False)
    model = build_objective_model(
        draft, options={"num_anchors": NUM_ANCHORS, "loss_decay_gamma": 7.0}
    )
    n_train = sum(p.numel() for p in draft.parameters() if p.requires_grad)
    return model, cfg, n_train


def make_inputs(cfg, bsz, dtype, device):
    naux = len(cfg.spec_config.target_features.target_layer_ids)
    g = torch.Generator(device="cpu").manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, SEQ), generator=g).to(device)
    hidden = torch.randn(
        bsz, SEQ, naux * cfg.hidden_size, generator=g, dtype=torch.float32
    ).to(device=device, dtype=dtype)
    loss_mask = torch.ones(bsz, SEQ, device=device)
    return dict(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)


def bench(bsz, dtype=torch.bfloat16, device="cuda", iters=10, warmup=3):
    model, cfg, n_train = build(dtype, device)
    inp = make_inputs(cfg, bsz, dtype, device)

    def step():
        for p in model.draft_model.parameters():
            p.grad = None
        loss = model(inp, step=1, total_steps=4, collect_detailed_metrics=False).loss
        loss.backward()
        return loss

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()

    # forward FLOPs (counts mm/bmm/sdpa; flex_attention may be undercounted)
    with torch.no_grad():
        fc = FlopCounterMode(display=False)
        with fc:
            model(inp, step=1, total_steps=4, collect_detailed_metrics=False)
        fwd_flop = fc.get_total_flops()

    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        step()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    # total training FLOP ~ 3x forward (1 fwd + 2 bwd), the standard rule
    total_flop = 3.0 * fwd_flop
    tflops = total_flop / (ms / 1e3) / 1e12
    mfu = 100.0 * tflops / H200_BF16_TFLOPS
    per_sample_ms = ms / bsz
    print(
        f"[bsz={bsz}] step={ms:7.1f}ms  per_sample={per_sample_ms:6.1f}ms  "
        f"fwdFLOP={fwd_flop/1e12:6.2f}T  ~totFLOP={total_flop/1e12:6.2f}T  "
        f"achieved={tflops:6.1f}TFLOP/s  MFU={mfu:4.1f}%  peakmem={peak_gb:6.1f}GB  "
        f"draft_params={n_train/1e9:.2f}B",
        flush=True,
    )
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print(
        f"device={torch.cuda.get_device_name(0)}  peak_ref={H200_BF16_TFLOPS} TFLOP/s bf16 dense"
    )
    for b in (2, 4, 8):
        try:
            bench(b)
        except RuntimeError as e:
            print(f"[bsz={b}] FAILED: {str(e)[:120]}")
            torch.cuda.empty_cache()
