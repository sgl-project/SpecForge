"""Single-GPU fwd+bwd microbench of DSparkV4DraftModel at production shapes.

Isolates pure model compute from FSDP/communication. Usage:
  python3 scripts/bench_dspark_compute.py [--config CONFIG_JSON] [--iters N]
"""
import argparse, sys, time, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/deepseek-v4-flash-dspark-official-b200.json")
    p.add_argument("--iters", type=int, default=4)
    p.add_argument("--seq", type=int, default=30720)
    p.add_argument("--anchors", type=int, default=512)
    p.add_argument("--grouped", choices=("on", "off", "cfg"), default="cfg")
    p.add_argument("--ckpt", choices=("on", "off", "cfg"), default="cfg")
    args = p.parse_args()

    from specforge.training.model_loading import load_draft_config_source
    from specforge.modeling.auto import AutoDraftModel

    cfg = load_draft_config_source(args.config)
    torch.manual_seed(0)
    model = AutoDraftModel.from_config(cfg).to("cuda", torch.bfloat16)
    if args.grouped != "cfg":
        for st in model.mtp:
            st.ffn.grouped_dispatch = args.grouped == "on"
    if args.ckpt != "cfg":
        model.stage_gradient_checkpointing = args.ckpt == "on"
    model.train()
    for st in model.mtp:
        st.ffn.bias_update_rate = 0.0
    # random gate weights route roughly uniformly, like the balanced drafter
    d = cfg.hidden_size
    feats = torch.randn(1, args.seq, 3 * d, device="cuda", dtype=torch.bfloat16)
    noise = torch.randn(1, args.anchors * model.block_size, d, device="cuda", dtype=torch.bfloat16)
    anchors = torch.sort(torch.randint(1, args.seq - model.block_size, (1, args.anchors), device="cuda"), -1).values
    keep = torch.ones(1, args.anchors, dtype=torch.bool, device="cuda")

    def one():
        out = model(noise_embedding=noise, target_hidden=feats,
                    anchor_positions=anchors, block_keep_mask=keep)
        model.pop_confidence_hidden()
        t_f = time.perf_counter()
        loss = out.float().square().mean()
        loss.backward()
        model.zero_grad(set_to_none=True)
        return t_f

    torch.cuda.synchronize(); one(); torch.cuda.synchronize()  # warmup
    fw = bw = 0.0
    for _ in range(args.iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        t_f = one()
        torch.cuda.synchronize(); t1 = time.perf_counter()
        # t_f is CPU-side split; resync for fwd-only timing next iter is
        # overkill — report total only, plus a synced fwd measure below.
        fw += t1 - t0
    # separate synced forward timing
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(args.iters):
        with torch.no_grad():
            model(noise_embedding=noise, target_hidden=feats,
                  anchor_positions=anchors, block_keep_mask=keep)
            model.pop_confidence_hidden()
    torch.cuda.synchronize(); fwd_only = (time.perf_counter() - t0) / args.iters
    grouped = model.mtp[0].ffn.grouped_dispatch
    print(f"grouped={grouped} ckpt={model.stage_gradient_checkpointing} "
          f"fwd+bwd {fw/args.iters*1000:.0f} ms  fwd(no_grad) {fwd_only*1000:.0f} ms")

if __name__ == "__main__":
    main()
