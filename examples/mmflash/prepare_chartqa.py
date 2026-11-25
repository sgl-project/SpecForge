"""Regenerate a small ChartQA training split with the Qwen target and save held-out prompts."""

import argparse
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory

DATASET = "HuggingFaceM4/ChartQA"
REVISION = "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5"


def main():
    import torch
    from datasets import load_dataset
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", type=int, default=64)
    parser.add_argument("--heldout", type=int, default=12)
    args = parser.parse_args()
    dest = args.out.resolve()
    if dest.exists() or min(args.train, args.heldout) < 1:
        parser.error("choose a new output directory and positive sample counts")
    dest.parent.mkdir(parents=True, exist_ok=True)
    model = (
        AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16)
        .to("cuda")
        .eval()
    )
    processor = AutoProcessor.from_pretrained(args.model)
    used_images = set()
    with TemporaryDirectory(dir=dest.parent) as temporary:
        stage = Path(temporary) / "dataset"
        (stage / "images").mkdir(parents=True)
        for split, count, name in [
            ("train", args.train, "train"),
            ("val", args.heldout, "heldout"),
        ]:
            rows = load_dataset(DATASET, revision=REVISION, split=split, streaming=True)
            written = 0
            with (stage / f"{name}.jsonl").open("w", encoding="utf-8") as out:
                for idx, row in enumerate(rows):
                    image = row["image"].convert("RGB")
                    image.thumbnail((640, 640), Image.Resampling.LANCZOS)
                    encoded = io.BytesIO()
                    image.save(encoded, format="PNG")
                    image_bytes = encoded.getvalue()
                    digest = hashlib.sha256(image_bytes).hexdigest()
                    if digest in used_images:
                        continue
                    question = (
                        row["query"].strip()
                        + "\nDescribe the relevant chart values and explain how they support your answer in a short paragraph."
                    )
                    record = {
                        "id": f"chartqa-{split}-{idx}",
                        "image": str(dest / "images" / f"{digest}.png"),
                        "conversations": [
                            {"role": "user", "content": "<image>\n" + question}
                        ],
                    }
                    if split == "train":
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": question},
                                ],
                            }
                        ]
                        text = processor.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                        batch = processor(
                            text=[text], images=[image], return_tensors="pt"
                        ).to("cuda")
                        with torch.inference_mode():
                            ids = model.generate(
                                **batch, max_new_tokens=128, do_sample=False
                            )
                        reply = processor.tokenizer.decode(
                            ids[0, batch.input_ids.size(1) :], skip_special_tokens=True
                        ).strip()
                        if (
                            len(
                                processor.tokenizer.encode(
                                    reply, add_special_tokens=False
                                )
                            )
                            < 32
                        ):
                            continue
                        record["conversations"].append(
                            {"role": "assistant", "content": reply}
                        )
                    used_images.add(digest)
                    (stage / "images" / f"{digest}.png").write_bytes(image_bytes)
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                    if written == count:
                        break
            if written != count:
                raise ValueError(
                    f"could only prepare {written}/{count} distinct images in {split}"
                )
        stage.rename(dest)


if __name__ == "__main__":
    main()
