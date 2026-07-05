#!/usr/bin/env python3
# coding=utf-8
"""Measure DFlash speculative-decoding acceptance length for a trained draft.

Runs the real DFlash accept loop against the target, greedy everywhere:

  1. target forward on the accepted prefix -> hidden states -> target_hidden
     (concat of the draft's capture layers, offset +1, == training).
  2. the draft proposes a block of ``block_size - 1`` tokens after the anchor
     (last accepted token), using the SAME attention the trainer uses
     (`create_dflash_sdpa_mask`, same position ids, same noise/context split).
  3. target verifies the block in one forward; accept the longest prefix whose
     draft token equals the target's greedy continuation, then append that
     prefix plus one bonus (corrected) token.
  4. acceptance length for the block = accepted + 1.

The loop is cache-free (recomputes the target each block) — acceptance length
is cache-invariant, and a single harness for BOTH the Qwen3-GQA draft and the
MLA draft makes the comparison exactly fair (only the draft class differs). This
is the acceptance metric, not a throughput benchmark.

Usage:
  python bench_dflash_acceptance.py --target-model-path deepseek-ai/DeepSeek-V2-Lite \
      --draft-checkpoint out/mla/epoch_2_step_300 \
      --eval-data-path nemotron_eval.jsonl --chat-template deepseek-v2 \
      --num-prompts 30 --max-new-tokens 128 --mask-token-id 100002
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_dataset
from specforge.core.dflash import create_dflash_sdpa_mask
from specforge.modeling.draft import DFlashDraftModel  # noqa: F401 (registers drafts)
from specforge.modeling.draft.registry import DRAFT_REGISTRY


def load_draft(checkpoint: str, device, dtype):
    import os

    from safetensors.torch import load_file
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(checkpoint)
    archs = getattr(cfg, "architectures", None) or []
    assert archs and archs[0] in DRAFT_REGISTRY, f"unknown draft arch {archs}"
    cls = DRAFT_REGISTRY[archs[0]]
    cfg._attn_implementation = "sdpa"
    # Manual state-dict load (NOT from_pretrained): the hand-rolled MLA
    # PreTrainedModel subclass silently returns fresh-init weights through
    # transformers 5.x from_pretrained (keys match but values are re-init), so
    # load the safetensors directly — keys match the model exactly.
    model = cls(cfg)
    sd = load_file(os.path.join(checkpoint, "model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (
        f"state-dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}"
    )
    return model.to(device=device, dtype=dtype).eval(), archs[0]


@torch.inference_mode()
def acceptance_for_prompt(
    target, draft, embed, lm_head, input_ids, *, block_size, layer_ids,
    mask_token_id, max_new_tokens, eos_token_id, device,
):
    ids = input_ids.to(device)
    # prime the first token (greedy)
    out = target(input_ids=ids, use_cache=False)
    ids = torch.cat([ids, out.logits[:, -1:].argmax(-1)], dim=1)

    accept_lengths = []
    generated = 1
    while generated < max_new_tokens:
        T = ids.shape[1]
        # (1) context hidden states for the accepted prefix
        ctx = target(input_ids=ids, output_hidden_states=True, use_cache=False)
        target_hidden = torch.cat(
            [ctx.hidden_states[l + 1] for l in layer_ids], dim=-1
        )  # [1, T, W]

        # (2) draft proposes block_size-1 tokens after the anchor (last token)
        anchor_pos = T - 1
        block_ids = torch.full(
            (1, block_size), mask_token_id, dtype=torch.long, device=device
        )
        block_ids[0, 0] = ids[0, -1]
        noise_emb = embed(block_ids)
        pos = torch.cat(
            [
                torch.arange(T, device=device),
                torch.arange(anchor_pos, anchor_pos + block_size, device=device),
            ]
        ).unsqueeze(0)
        mask = create_dflash_sdpa_mask(
            anchor_positions=torch.tensor([[anchor_pos]], device=device),
            block_keep_mask=torch.ones(1, 1, dtype=torch.bool, device=device),
            S=T,
            block_size=block_size,
            device=device,
        )
        h = draft(
            position_ids=pos,
            noise_embedding=noise_emb,
            target_hidden=target_hidden,
            attention_mask=mask,
        )
        draft_logits = lm_head(h)[:, -(block_size - 1):, :]  # [1, bs-1, V]
        proposed = draft_logits.argmax(-1)[0]  # [bs-1]

        # (3) target verifies the proposed block in one forward
        verify_ids = torch.cat([ids, proposed.unsqueeze(0)], dim=1)
        vout = target(input_ids=verify_ids, use_cache=False)
        post = vout.logits[0, anchor_pos:anchor_pos + block_size, :].argmax(-1)  # [bs]

        matches = (proposed == post[: block_size - 1]).long()
        accept = int(torch.cumprod(matches, dim=0).sum().item())
        bonus = post[accept].item()
        new = proposed[:accept].tolist() + [bonus]
        ids = torch.cat(
            [ids, torch.tensor([new], dtype=torch.long, device=device)], dim=1
        )
        accept_lengths.append(accept + 1)
        generated += len(new)
        if eos_token_id is not None and eos_token_id in new:
            break

    return accept_lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-model-path", required=True)
    ap.add_argument("--draft-checkpoint", required=True)
    ap.add_argument("--eval-data-path", required=True)
    ap.add_argument("--chat-template", default="deepseek-v2")
    ap.add_argument("--num-prompts", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--prompt-max-length", type=int, default=256)
    ap.add_argument("--mask-token-id", type=int, required=True)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    tok = AutoTokenizer.from_pretrained(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    target = (
        AutoModelForCausalLM.from_pretrained(
            args.target_model_path,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )
        .to(device)
        .eval()
    )
    embed = target.get_input_embeddings()
    lm_head = target.get_output_embeddings()

    draft, arch = load_draft(args.draft_checkpoint, device, dtype)
    block_size = draft.block_size
    layer_ids = list(draft.target_layer_ids)
    print(f"draft={arch} block_size={block_size} capture_layers={layer_ids}")

    # Build prompts: render conversations, then keep only the FIRST user turn as
    # the decoding prompt (the draft/target generate the assistant reply).
    ds = load_dataset("json", data_files=args.eval_data_path)["train"]
    prompts = []
    for row in ds:
        conv = row.get("conversations") or row.get("messages")
        if not conv:
            continue
        msgs = [m for m in conv if m.get("role") in ("system", "user")]
        # cut at the first user turn (inclusive)
        cut = []
        for m in msgs:
            cut.append(m)
            if m.get("role") == "user":
                break
        if not any(m.get("role") == "user" for m in cut):
            continue
        text = tok.apply_chat_template(
            cut, tokenize=False, add_generation_prompt=True
        )
        enc = tok(
            text, return_tensors="pt", truncation=True,
            max_length=args.prompt_max_length, add_special_tokens=False,
        )
        if enc.input_ids.shape[1] >= 8:
            prompts.append(enc.input_ids)
        if len(prompts) >= args.num_prompts:
            break

    print(f"benchmarking on {len(prompts)} prompts, max_new={args.max_new_tokens}")
    all_lengths = []
    per_prompt = []
    for i, ids in enumerate(prompts):
        lens = acceptance_for_prompt(
            target, draft, embed, lm_head, ids,
            block_size=block_size, layer_ids=layer_ids,
            mask_token_id=args.mask_token_id, max_new_tokens=args.max_new_tokens,
            eos_token_id=tok.eos_token_id, device=device,
        )
        all_lengths.extend(lens)
        m = sum(lens) / len(lens) if lens else 0.0
        per_prompt.append(m)
        print(f"  prompt {i:>3}: {len(lens)} blocks, mean accept {m:.3f}")

    mean_accept = sum(all_lengths) / len(all_lengths) if all_lengths else 0.0
    # distribution
    from collections import Counter

    dist = Counter(all_lengths)
    result = {
        "arch": arch,
        "checkpoint": args.draft_checkpoint,
        "block_size": block_size,
        "n_prompts": len(prompts),
        "n_blocks": len(all_lengths),
        "mean_accept_length": round(mean_accept, 4),
        "max_accept_length": max(all_lengths) if all_lengths else 0,
        "accept_length_hist": {int(k): int(v) for k, v in sorted(dist.items())},
        "per_prompt_mean": [round(x, 3) for x in per_prompt],
    }
    print("\n=== RESULT ===")
    print(json.dumps({k: v for k, v in result.items() if k != "per_prompt_mean"}, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
