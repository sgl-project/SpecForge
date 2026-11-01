"""Isolated single-GPU MFU / FLOPs / memory benchmark for the domino trainer.

Removes the data-supply path and DP comm so we measure ONLY the per-step trainer
compute: build the draft from the Qwen3-8B Domino training recipe with random
weights on one GPU, feed a realistic batch, time fwd+bwd with CUDA events, count
forward FLOPs with torch's FlopCounterMode, and read peak allocated memory.

Reports achieved TFLOP/s and MFU vs H200 bf16 dense peak (989.5 TFLOP/s) for a
sweep of batch sizes, plus per-sample compute (to show it is ~flat in batch =
compute-latency bound, not FLOPs/memory bound).
"""

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from specforge.algorithms.common.dflash_family_model import OnlineDominoModel
from specforge.config.draft import AutoDraftConfig
from specforge.modeling.auto import AutoDraftModel

H200_BF16_TFLOPS = 989.5  # SXM, dense
SEQ = 768
NUM_ANCHORS = 256
CFG_PATH = "examples/configs/online/disaggregated/external/qwen3-8b-domino.yaml"


def build(dtype, device):
    cfg = AutoDraftConfig.from_yaml(CFG_PATH)
    draft = AutoDraftModel.from_config(cfg).to(device=device, dtype=dtype)
    V, H = cfg.vocab_size, cfg.hidden_size
    lm_head = nn.Linear(H, V, bias=False).to(device=device, dtype=dtype)
    embed = nn.Embedding(V, H).to(device=device, dtype=dtype)
    for p in list(lm_head.parameters()) + list(embed.parameters()):
        p.requires_grad_(False)

    model = OnlineDominoModel(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        attention_backend="sdpa",
        num_anchors=NUM_ANCHORS,
        loss_decay_gamma=7.0,
    )
    n_train = sum(p.numel() for p in draft.parameters() if p.requires_grad)
    return model, cfg, n_train


def make_inputs(cfg, bsz, dtype, device):
    naux = len(cfg.dflash_config["target_layer_ids"])
    g = torch.Generator(device="cpu").manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, SEQ), generator=g).to(device)
    hidden = torch.randn(
        bsz, SEQ, naux * cfg.hidden_size, generator=g, dtype=torch.float32
    ).to(device=device, dtype=dtype)
    loss_mask = torch.ones(bsz, SEQ, device=device)
    return input_ids, hidden, loss_mask


def bench(bsz, dtype=torch.bfloat16, device="cuda", iters=10, warmup=3):
    model, cfg, n_train = build(dtype, device)
    inp = make_inputs(cfg, bsz, dtype, device)

    def step():
        for p in model.draft_model.parameters():
            p.grad = None
        loss, _, _ = model(*inp, lambda_base=0.5)
        loss.backward()
        return loss

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()

    # forward FLOPs (counts mm/bmm/sdpa; flex_attention may be undercounted)
    with torch.no_grad():
        fc = FlopCounterMode(display=False)
        with fc:
            model(*inp, lambda_base=0.5)
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
