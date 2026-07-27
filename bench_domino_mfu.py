"""Isolated single-GPU MFU / FLOPs / memory benchmark for the domino trainer.

Removes the data-supply path and DP comm so we measure ONLY the per-step trainer
compute: build the real draft (configs/qwen3-8b-domino.json shapes) with random
weights on one GPU, feed a realistic batch, time fwd+bwd with CUDA events, count
forward FLOPs with torch's FlopCounterMode, and read peak allocated memory.

Reports achieved TFLOP/s and MFU vs H200 bf16 dense peak (989.5 TFLOP/s) for a
sweep of batch sizes, plus per-sample compute (to show it is ~flat in batch =
compute-latency bound, not FLOPs/memory bound).
"""

import json

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import OnlineDominoModel
from specforge.modeling.draft.dflash import DFlashDraftModel

H200_BF16_TFLOPS = 989.5  # SXM, dense
SEQ = 768
NUM_ANCHORS = 256
CFG_PATH = "configs/qwen3-8b-domino.json"


def build(dtype, device):
    raw = json.load(open(CFG_PATH))
    dfc = raw["dflash_config"]
    cfg = Qwen3Config(
        hidden_size=raw["hidden_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw.get("num_key_value_heads", 8),
        head_dim=raw.get("head_dim", 128),
        intermediate_size=raw["intermediate_size"],
        vocab_size=raw["vocab_size"],
        max_position_embeddings=raw.get("max_position_embeddings", 40960),
        rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
        attention_bias=raw.get("attention_bias", False),
        attention_dropout=0.0,
        rope_theta=raw.get("rope_theta", 1000000.0),
    )
    cfg._attn_implementation = "sdpa"
    cfg.layer_types = ["full_attention"] * raw["num_hidden_layers"]
    cfg.num_target_layers = 36
    cfg.block_size = raw["block_size"]
    cfg.dflash_config = dfc

    draft = DFlashDraftModel(cfg).to(device=device, dtype=dtype)
    V, H = raw["vocab_size"], raw["hidden_size"]
    lm_head = nn.Linear(H, V, bias=False).to(device=device, dtype=dtype)
    embed = nn.Embedding(V, H).to(device=device, dtype=dtype)
    for p in list(lm_head.parameters()) + list(embed.parameters()):
        p.requires_grad_(False)

    model = OnlineDominoModel(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=dfc["mask_token_id"],
        block_size=raw["block_size"],
        attention_backend="sdpa",
        num_anchors=NUM_ANCHORS,
        loss_decay_gamma=7.0,
        shift_label=dfc.get("shift_label", True),
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
