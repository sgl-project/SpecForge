"""HF-backend EAGLE3 offline hidden-state dumper (experiment substitute).

prepare_hidden_states.py drives the in-process sglang extend path, which is
incompatible with this pod's dev sglang. This dumper produces the same offline
file layout ({input_ids [L], loss_mask [L], hidden_state [1,L,H] post-norm,
aux_hidden_state [1,L,3H]}) from a plain HF forward, reusing train_eagle3's
dataloader so the tokenization/loss-mask path is identical to training.

Env: EXP_HS_PATH (out dir), EXP_NUM_SAMPLES (default 130).
"""

import os

import torch
from accelerate.utils import set_seed
from train_eagle3 import build_dataloaders, build_draft_model, parse_args
from transformers import AutoModelForCausalLM

from specforge.distributed import destroy_distributed, init_distributed

OUT = os.environ.get("EXP_HS_PATH", "/root/exp/dumps/e3-hs")
N = int(os.environ.get("EXP_NUM_SAMPLES", "130"))


def main():
    parser, args = parse_args()
    args.target_batch_size = args.tp_size * args.batch_size
    set_seed(args.seed)
    init_distributed(
        timeout=args.dist_timeout,
        tp_size=args.tp_size,
        sp_ring_size=args.sp_ring_size,
        sp_ulysses_size=args.sp_ulysses_size,
    )
    draft_config, _draft, _ckpt, _resume = build_draft_model(args)
    train_dataloader, _vocab, _eval = build_dataloaders(args, draft_config)

    model = AutoModelForCausalLM.from_pretrained(
        args.target_model_path, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    num_layers = model.config.num_hidden_layers
    # same default rule as Eagle3CapturePolicy.resolve_capture_layers
    aux_ids = [1, num_layers // 2 - 1, num_layers - 4]
    os.makedirs(OUT, exist_ok=True)

    n = 0
    with torch.no_grad():
        for batch in train_dataloader:
            input_ids = batch["input_ids"].cuda()
            attn = batch["attention_mask"].cuda()
            loss_mask = batch["loss_mask"]
            if loss_mask.dim() == 3:  # some collators emit [B, L, 1]
                loss_mask = loss_mask.squeeze(-1)
            seq = int(attn[0].sum().item())
            out = model(
                input_ids=input_ids[:, :seq],
                attention_mask=attn[:, :seq],
                output_hidden_states=True,
                use_cache=False,
            )
            hs = out.hidden_states  # [0]=embed, [i+1]=layer i output, [-1]=post-norm
            aux = torch.cat([hs[i + 1][0] for i in aux_ids], dim=-1)  # [L, 3H]
            last = hs[-1][0]  # [L, H] post-final-norm (TargetHead is lm_head-only)
            record = {
                "input_ids": input_ids[0, :seq].cpu(),
                "loss_mask": loss_mask[0, :seq].to(torch.long).cpu(),
                "hidden_state": last.unsqueeze(0).cpu(),
                "aux_hidden_state": aux.unsqueeze(0).cpu(),
            }
            torch.save(record, os.path.join(OUT, f"sample_{n:08d}.ckpt"))
            n += 1
            if n >= N:
                break
    print(f"[dump] wrote {n} eagle3 offline files to {OUT}", flush=True)
    destroy_distributed()


if __name__ == "__main__":
    main()
