"""ROCm-native offline DFlash feature generator (HF forward, no SGLang).

Produces `.ckpt` files consumable by the offline DFlash reader
(`specforge/algorithms/common/dflash_family_data.py`), i.e. each file holds:
    input_ids     : LongTensor [seq]
    loss_mask     : LongTensor [seq]
    hidden_states : FloatTensor [seq, len(target_layer_ids) * hidden_size]

The hidden-state selection mirrors `extract_context_feature` in
`specforge/modeling/draft/dflash.py` exactly: for each layer id in the draft
config's `dflash_config.target_layer_ids`, it takes HF
`output.hidden_states[layer_id + 1]` (offset=1, because index 0 is the
embedding output) and concatenates them along the feature dim.

This is a test/bench utility for platforms without a CUDA SGLang capture stack
(e.g. AMD ROCm). It is single-process and consumes precomputed nothing — it
loads the target model once and runs plain HF forwards.
"""

import argparse
import json
import os

import torch

# The HF `datasets` formatter does `from torchvision.io import VideoReader`
# whenever it tensorizes torch tensors. Some ROCm torchvision builds ship
# without video support, so that symbol is missing and the import raises.
# We never touch video, so provide a harmless stub to satisfy the isinstance
# check the formatter performs.
try:  # pragma: no cover - environment shim
    import torchvision.io as _tvio

    if not hasattr(_tvio, "VideoReader"):
        class _StubVideoReader:  # noqa: D401 - placeholder only
            pass

        _tvio.VideoReader = _StubVideoReader
except Exception:  # torchvision absent entirely
    pass

from datasets import Dataset
from transformers import AutoModelForCausalLM

from specforge.data.preprocessing import build_eagle3_dataset
from specforge.utils import load_tokenizer, safe_conversations_generator


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-model-path", required=True)
    p.add_argument("--draft-config", required=True,
                   help="Draft JSON with dflash_config.target_layer_ids")
    p.add_argument("--data-path", required=True, help="Conversations JSONL")
    p.add_argument("--output-path", required=True)
    p.add_argument("--chat-template", default="qwen")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--cache-dir", default="./cache")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.draft_config) as f:
        draft_cfg = json.load(f)
    target_layer_ids = draft_cfg["dflash_config"]["target_layer_ids"]
    block_size = draft_cfg.get("block_size") or draft_cfg["dflash_config"].get(
        "block_size"
    )
    # extract_context_feature uses hidden_states[layer_id + 1] (offset=1).
    hs_indices = [lid + 1 for lid in target_layer_ids]
    print(f"target_layer_ids={target_layer_ids} -> hidden_states indices={hs_indices}")
    print(f"block_size={block_size}")

    tokenizer = load_tokenizer(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )

    dataset = Dataset.from_generator(
        generator=safe_conversations_generator,
        gen_kwargs={"file_path": args.data_path},
        cache_dir=os.path.join(args.cache_dir, "hf_dataset"),
    )
    if args.num_samples is not None:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))

    eagle3_dataset = build_eagle3_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
        cache_key=f"dflash-hf-{args.max_length}-{args.num_samples}-min{2 * block_size}",
        minimum_valid_tokens=2 * block_size,
    )
    # Read rows as numpy to avoid the datasets torch-formatter importing
    # torchvision (broken on some ROCm torch/torchvision combos).
    eagle3_dataset = eagle3_dataset.with_format("numpy")
    print(f"Dataset prepared with {len(eagle3_dataset)} samples.")

    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.target_model_path,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to("cuda").eval()

    hidden_size = model.config.hidden_size
    expected_width = len(target_layer_ids) * hidden_size
    os.makedirs(args.output_path, exist_ok=True)

    saved = 0
    skipped_anchor = 0
    with torch.no_grad():
        for idx in range(len(eagle3_dataset)):
            row = eagle3_dataset[idx]
            input_ids = torch.as_tensor(row["input_ids"]).view(1, -1).to("cuda")
            loss_mask = torch.as_tensor(row["loss_mask"]).view(-1)
            seq_len = input_ids.shape[1]
            # Mirror _sample_anchor_positions: anchors are drawn from the region
            # [:seq_len - block_size]. Require enough valid tokens there so the
            # trainer never hits "should preprocess the data" (batch_size=1).
            max_anchor = max(seq_len - block_size, 0)
            anchor_valid = int((loss_mask[: max_anchor + 1] > 0.5).sum())
            if anchor_valid < block_size:
                skipped_anchor += 1
                continue
            out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
            selected = [out.hidden_states[i][0] for i in hs_indices]  # each [seq, H]
            hidden_states = torch.cat(selected, dim=-1)  # [seq, width]
            assert hidden_states.shape[-1] == expected_width, (
                f"width {hidden_states.shape[-1]} != expected {expected_width}"
            )
            payload = {
                "input_ids": input_ids[0].to("cpu"),
                "loss_mask": loss_mask.to("cpu"),
                "hidden_states": hidden_states.to(torch.float16).to("cpu"),
            }
            torch.save(payload, os.path.join(args.output_path, f"data_{idx}.ckpt"))
            saved += 1
            if saved % 25 == 0:
                print(f"  saved {saved}/{len(eagle3_dataset)} "
                      f"(seq={input_ids.shape[1]}, width={hidden_states.shape[-1]})")

    print(f"Done. Wrote {saved} feature files to {args.output_path} "
          f"(skipped {skipped_anchor} with too few anchorable tokens)")


if __name__ == "__main__":
    main()
