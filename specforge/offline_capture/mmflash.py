"""Score responses with/without their image and cache DFlash features on one GPU.

Use the same offline training YAML as the trainer. Paths in input records are
relative to the working directory. The score sidecar includes every eligible
image response, including short responses used for quantile calibration only.
Both output paths must be fresh. Finished feature rows can feed MMFlash and
its matched DFlash control without another target forward.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch

from specforge.data.mmflash_preprocessing import assistant_delimiters, encode_record
from specforge.data.prompt_builder import _iter_records
from specforge.data.visual_kl import capture_text_features, visual_kl_per_token
from specforge.data.visual_score import VisualScoreTable, fingerprint_input_ids


def prepare_features(
    input_path,
    scores_path,
    features_dir,
    *,
    target,
    processor,
    layer_ids,
    block_size,
    max_length,
    last_turn_only,
    device,
):
    scores_path = Path(scores_path).resolve()
    features_dir = Path(features_dir).resolve()
    if scores_path == features_dir or features_dir in scores_path.parents:
        raise ValueError("put the score sidecar outside the feature directory")
    if scores_path.exists() or (features_dir.exists() and any(features_dir.iterdir())):
        raise FileExistsError("score sidecar and feature directory must be fresh")
    if not layer_ids or len(set(layer_ids)) != len(layer_ids) or min(layer_ids) < 0:
        raise ValueError(f"invalid capture layer ids: {layer_ids}")
    if block_size < 2:
        raise ValueError("DFlash requires a block size of at least 2")
    features_dir.parent.mkdir(parents=True, exist_ok=True)
    scores_path.parent.mkdir(parents=True, exist_ok=True)

    delimiters = assistant_delimiters(processor)
    visited, captured = set(), {}
    scored = skipped = 0
    with (
        tempfile.TemporaryDirectory(
            prefix=".mmflash-", dir=features_dir.parent
        ) as staging,
        tempfile.NamedTemporaryFile(
            mode="w+", encoding="utf-8", dir=scores_path.parent
        ) as scores,
    ):
        stage = Path(staging)
        for line, record in _iter_records(str(input_path)):
            key = str(record.get("id", line))
            if key in visited:
                raise ValueError(f"duplicate row id {key!r} at line {line}")
            visited.add(key)
            source = f"{input_path}:{line} (id {key!r})"
            try:
                seen, blind, mask = encode_record(
                    record,
                    processor,
                    delimiters=delimiters,
                    last_turn_only=last_turn_only,
                    source=source,
                )
                ids = seen["input_ids"][0].tolist()
                if len(ids) > max_length or not any(mask):
                    skipped += 1
                    continue
                # Same supervision floor as the published MMFlash trainer.
                # At least two anchors must fit before the final block.
                eligible = (
                    sum(mask) >= 2 * block_size
                    and sum(mask[: len(ids) - block_size + 1]) >= 2
                )
                if not eligible and blind is None:
                    skipped += 1
                    continue
                seen = {k: v.to(device) for k, v in seen.items()}
                if blind is None:
                    hidden = capture_text_features(target, seen, layer_ids)
                else:
                    blind = {k: v.to(device) for k, v in blind.items()}
                    kl, entropy, hidden = visual_kl_per_token(
                        target,
                        seen,
                        blind,
                        mask,
                        target_layer_ids=layer_ids if eligible else (),
                    )
                    row = dict(
                        id=key,
                        n_tokens=len(mask),
                        n_loss=sum(mask),
                        fp=fingerprint_input_ids(ids),
                        mask_fp=fingerprint_input_ids(mask),
                        kl=[round(x, 6) for x in kl.tolist()],
                        entropy=[round(x, 6) for x in entropy.tolist()],
                    )
                    scores.write(json.dumps(row, allow_nan=False) + "\n")
                    scored += 1
                if eligible:
                    name = f"{line:08d}.ckpt"
                    torch.save(
                        {
                            "input_ids": torch.tensor(ids),
                            "loss_mask": torch.tensor(mask),
                            "hidden_states": hidden,
                        },
                        stage / name,
                    )
                    captured[name] = (key, blind is not None)
                else:
                    skipped += 1
            except (ValueError, OSError, RuntimeError) as exc:
                raise ValueError(f"{source}: {exc}") from exc
            if len(visited) % 100 == 0:
                print(
                    f"{len(visited)} rows read, {scored} scored, {len(captured)} cached",
                    flush=True,
                )
        if not any(image for _, image in captured.values()):
            raise ValueError(
                f"no image rows eligible for training (need {2 * block_size} supervised tokens, max_length={max_length})"
            )
        scores.flush()
        table = VisualScoreTable.load(scores.name)
        for name, (key, has_image) in captured.items():
            path = stage / name
            raw = torch.load(path, map_location="cpu", weights_only=True)
            if has_image:
                channel = table.channel(
                    key, raw["input_ids"].tolist(), raw["loss_mask"].tolist()
                )
                raw["visual_score"] = torch.tensor(channel, dtype=torch.long)
            else:
                raw["visual_score"] = torch.full_like(raw["input_ids"], -1)
            torch.save(raw, path)
        # Publish only complete data. The hard link fails if another writer won.
        os.link(scores.name, scores_path)
        try:
            stage.replace(features_dir)
        except OSError:
            scores_path.unlink()
            raise
    print(
        f"Scored {scored} image rows. Saved {len(captured)} training rows. Skipped {skipped} training rows."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="offline MMFlash training YAML")
    parser.add_argument("--input", required=True, help="ShareGPT image/text JSONL")
    parser.add_argument(
        "--scores", required=True, help="output JSONL with raw KL and entropy"
    )
    parser.add_argument("--features-dir", help="defaults to data.hidden_states_path")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    from specforge.config import load_config
    from specforge.training.canonical import resolve_config

    config = load_config(args.config)
    output = args.features_dir or config.data.hidden_states_path
    if (
        not output
        or config.mode != "offline"
        or config.model.input_modality != "multimodal"
    ):
        parser.error(
            "use an offline run with model.input_modality=multimodal and a feature output directory"
        )
    draft = resolve_config(config)
    if draft.architecture != "DFlashDraftModel":
        parser.error("visual KL scoring expects a DFlash draft")
    spec = draft.spec_config
    settings = {
        "cache_dir": config.model.cache_dir,
        "trust_remote_code": config.model.trust_remote_code,
    }
    processor = AutoProcessor.from_pretrained(
        config.model.target_model_path, **settings
    )
    target = (
        AutoModelForImageTextToText.from_pretrained(
            config.model.target_model_path,
            dtype=getattr(torch, config.model.torch_dtype),
            **settings,
        )
        .to(args.device)
        .eval()
    )
    prepare_features(
        args.input,
        args.scores,
        output,
        target=target,
        processor=processor,
        layer_ids=spec.target_features.target_layer_ids,
        block_size=spec.protocol.num_model_slots,
        max_length=config.data.max_length,
        last_turn_only=config.data.train_only_last_turn,
        device=args.device,
    )
