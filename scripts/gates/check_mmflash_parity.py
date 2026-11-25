"""Compare the SpecForge port with a clean checkout of the released MMFlash code.

Without data arguments, use fixed CPU tensors and a toy target. With --input,
independently render and score the same image rows using the authors' processor
and frozen target on CUDA. Feed both training objectives identical logits from
an initialized author draft on the captured features. The gate checks label IDs,
loss masks, raw scores, calibrated channels, scalar loss and logit gradients.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

PIN = "5b3fa3fe28f5bc6d7dfe75934a156c52b865dc8c"
REPO = Path(__file__).resolve().parents[2]


def fixture():
    # Non-contiguous supervision checks causal alignment and masked prefix products.
    tokens = [11, 12, 12, 12, 4, 5, 1, 7, 2, 8, 4, 1, 3, 4]
    mask = [0] * 6 + [1, 1, 0, 1, 1, 1, 1, 1]
    seen = {
        "input_ids": torch.tensor([tokens]),
        "image_grid_thw": torch.tensor([[1, 2, 6]]),
        "pixel_values": torch.tensor([[0.75]]),
    }
    blind = {"input_ids": torch.tensor([[11] + tokens[4:]])}

    class Target(nn.Module):
        def forward(self, input_ids, logits_to_keep, pixel_values=None, **_kwargs):
            positions = input_ids.cumsum(1).float()
            vocab = torch.arange(15, dtype=torch.float32)
            logits = (positions.unsqueeze(-1) / 9 + vocab / 2).sin()
            if pixel_values is not None:
                logits += pixel_values.item() * (vocab * 0.8).cos()
            return SimpleNamespace(logits=logits[:, -logits_to_keep:])

    class Processor:
        def apply_chat_template(self, *_args, **_kwargs):
            return "same prompt and response, image removed"

        def __call__(self, **_kwargs):
            return blind

    record = {
        "id": "fixture",
        "image": "fixture",
        "conversations": [
            {"role": "user", "content": "<image>\nDescribe."},
            {"role": "assistant", "content": "Fixed response."},
        ],
    }
    return record, Target(), Processor(), seen, blind, mask


def get_logits(wrapper, capture, target, draft, block_size, device):
    ids = capture["input_ids"].reshape(1, -1).to(device)
    mask = capture["loss_mask"].reshape_as(ids).to(device)
    aux = capture["hidden_states"].reshape(1, ids.shape[-1], -1).to(device)
    model = wrapper(
        draft,
        target.get_output_embeddings(),
        target.get_input_embeddings(),
        draft.mask_token_id,
        block_size=block_size,
        num_anchors=2,
        attention_backend="sdpa",
        loss_type="mmflash",
    )
    with torch.no_grad():
        anchors, keep, hidden = model._forward_draft_blocks(ids, aux, mask)
        return anchors.cpu(), keep.cpu(), model.lm_head(hidden).float().cpu()


def replay_author(wrapper, sample, *, dtype, device):
    logits = sample["logits"].to(device=device, dtype=dtype).detach().requires_grad_()
    anchors, keep = sample["anchors"].to(device), sample["keep"].to(device)
    block = logits.shape[1] // anchors.shape[1]
    model = wrapper(
        nn.Identity(),
        nn.Identity(),
        nn.Identity(),
        0,
        block_size=block,
        attention_backend="sdpa",
        loss_type="mmflash",
        objective_chunk_blocks=1,
        mmflash_smoothing=0.5,
        visual_alpha=1.0,
    )
    with patch.object(
        model, "_forward_draft_blocks", return_value=(anchors, keep, logits)
    ):
        loss, _, _ = model(
            input_ids=sample["ids"].to(device),
            hidden_states=torch.empty(0, device=device),
            loss_mask=sample["mask"].to(device),
            visual_score=sample["channel"].to(device).float() / 10_000,
        )
    loss.backward()
    return loss.detach().cpu(), logits.grad.detach().cpu()


def replay_canonical(sample, *, dtype, device):
    """Keep the protocol's real alignment and chunk reduction on fixed logits."""
    from specforge.algorithms.objectives import build_objective_model

    logits = sample["logits"].to(device=device, dtype=dtype).detach().requires_grad_()
    anchors, keep = sample["anchors"].to(device), sample["keep"].to(device)
    block_size = logits.size(1) // anchors.size(1)

    class FixedDraft(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(1, 1, device=device, dtype=dtype)
            self.config = SimpleNamespace(
                draft=SimpleNamespace(
                    architecture="DFlashDraftModel",
                    layer_types=(),
                    spec_config=SimpleNamespace(
                        protocol=SimpleNamespace(
                            num_model_slots=block_size,
                            prediction_positions=tuple(range(1, block_size)),
                        ),
                        target_features=SimpleNamespace(mask_token_id=0),
                    ),
                )
            )

        def forward(self, *_args, **_kwargs):
            return SimpleNamespace(hidden_states=logits)

        def project(self, hidden):
            return {"logits": hidden}

    runtime = build_objective_model(
        FixedDraft(), name="mmflash", loss={"execution": {"chunk_size": 1}}
    )
    ids = sample["ids"].to(device)
    batch = dict(
        input_ids=ids,
        loss_mask=sample["mask"].to(device),
        visual_score=sample["channel"].to(device).long(),
        hidden_states=torch.empty((*ids.shape, 1), device=device, dtype=dtype),
    )
    with patch(
        "specforge.algorithms.objectives.inputs.sample_anchors",
        return_value=(anchors, keep),
    ):
        loss = runtime(batch, step=0, total_steps=1).loss
        loss.backward()
    return loss.detach().cpu(), logits.grad.detach().cpu()


def calibrate(table, row_id, ids, mask):
    from specforge.data.visual_score import expand_visual_score

    channel, status = expand_visual_score(mask, table.get(row_id), input_ids=ids)
    if status != "scored":
        raise ValueError(f"Author calibration failed for {row_id}: {status}")
    return torch.tensor(list(channel)).unsqueeze(0)


def capture_reference_layers(stack, target, layer_ids):
    # Observe the decoder blocks themselves on the author's *first* (image)
    # forward. This does not reuse the adaptation's hidden_states[i + 1] index.
    captured = {}

    def capture(index):
        def hook(_module, _inputs, output):
            if index not in captured:
                tensor = output[0] if isinstance(output, tuple) else output
                captured[index] = tensor[0].detach().cpu()

        return hook

    layers = target.model.language_model.layers
    for index in layer_ids:
        handle = layers[index].register_forward_hook(capture(index))
        stack.callback(handle.remove)
    return captured


def reference(args, work):
    # This runs in its own interpreter. Absolute imports inside author code
    # must resolve against its checkout, including chunking and preprocessing.
    sys.path.insert(0, str(args.author_repo))
    from specforge.algorithms.common.mmflash_model import OnlineMMFlashModel
    from specforge.data.visual_kl import visual_kl_per_token
    from specforge.data.visual_score import VisualScoreTable, fingerprint_input_ids

    if not args.input:
        record, target, processor, seen, _, mask = fixture()
        ids = seen["input_ids"][0].tolist()
        kl, entropy = visual_kl_per_token(
            target, processor, record, seen["input_ids"], mask, 128, model_inputs=seen
        )
        compact_kl, compact_h = list(kl.values()), list(entropy.values())
        sidecar = work / "fixture.jsonl"
        sidecar.write_text(
            json.dumps(
                {
                    "id": record["id"],
                    "fp": fingerprint_input_ids(ids),
                    "mask_fp": fingerprint_input_ids(mask),
                    "n_tokens": len(ids),
                    "n_loss": sum(mask),
                    "kl": compact_kl,
                    "entropy": compact_h,
                }
            )
            + "\n"
        )
        table = VisualScoreTable.load(str(sidecar))
        sample = {
            "row_id": record["id"],
            "ids": seen["input_ids"],
            "mask": torch.tensor([mask]),
            "kl": torch.tensor(compact_kl),
            "entropy": torch.tensor(compact_h),
            "channel": calibrate(table, record["id"], ids, mask),
            "anchors": torch.tensor([[6, 9, 0]]),
            "keep": torch.tensor([[True, True, False]]),
            "logits": torch.randn(
                1, 12, 15, generator=torch.Generator().manual_seed(321)
            ),
        }
        write_sample(work, 0, sample, OnlineMMFlashModel, "cpu")
        return

    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor, Qwen3Config

    from specforge.data.mm_preprocessing import (
        _assistant_header_ids,
        _end_token_ids,
        encode_mm_record,
        to_chat_messages,
    )
    from specforge.modeling.draft.mmflash import MMFlashDraftModel

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    processor = AutoProcessor.from_pretrained(args.target)
    header, end = _assistant_header_ids(processor), _end_token_ids(processor)
    target = AutoModelForImageTextToText.from_pretrained(args.target, dtype=dtype)
    target = target.eval().to(device)
    config = Qwen3Config.from_json_file(str(args.draft_config))
    config._attn_implementation = "sdpa"
    torch.manual_seed(321)
    draft = MMFlashDraftModel(config).to(device=device, dtype=dtype).eval()
    # Calibrate using the whole scored corpus even when replaying only a few rows.
    # The author loader caches percentile edges alongside its input.
    sidecar = work / "real.jsonl"
    sidecar.symlink_to(args.scores.resolve())
    table = VisualScoreTable.load(str(sidecar))
    checked = 0
    with args.input.open() as records:
        for line, raw in enumerate(records, start=1):
            if not raw.strip():
                continue
            record = json.loads(raw)
            row_id = str(record.get("id", line))
            feature = args.features_dir / f"{line:08d}.ckpt"
            if not record.get("image") or not feature.exists():
                continue
            encoded = encode_mm_record(
                record,
                processor,
                image_root=args.image_root,
                max_length=args.max_length,
                train_only_last_turn=args.last_turn_only,
                header_ids=header,
                end_ids=end,
            )
            if encoded is None:
                raise ValueError(
                    f"Pinned author encoder rejected candidate row {row_id}"
                )
            with Image.open(encoded["image"]) as handle:
                picture = handle.convert("RGB")
            rendered = processor.apply_chat_template(
                to_chat_messages(record["conversations"]),
                tokenize=False,
                add_generation_prompt=False,
            )
            seen = processor(text=[rendered], images=[picture], return_tensors="pt").to(
                device
            )
            ids, mask = encoded["input_ids"], encoded["loss_mask"]
            with ExitStack() as stack:
                captured = capture_reference_layers(
                    stack, target, draft.target_layer_ids
                )
                result = visual_kl_per_token(
                    target,
                    processor,
                    record,
                    seen["input_ids"],
                    mask,
                    args.max_length,
                    model_inputs=seen,
                )
            positions = [p for p, keep in enumerate(mask) if keep]
            if result is None or set(result[0]) != set(positions):
                raise ValueError(
                    f"Pinned author scorer failed to align every label in row {row_id}"
                )
            kl, entropy = result
            capture = torch.load(feature, map_location="cpu", weights_only=True)
            if (
                capture["input_ids"].flatten().tolist() != ids
                or capture["loss_mask"].flatten().tolist() != mask
            ):
                raise ValueError(
                    f"Author and SpecForge IDs or masks differ for {row_id}"
                )
            reference_hidden = torch.cat(
                [captured[i] for i in draft.target_layer_ids], dim=-1
            )
            saved_hidden = capture["hidden_states"]
            torch.testing.assert_close(
                saved_hidden, reference_hidden, atol=1e-5, rtol=1e-5
            )
            hidden_error = (
                (saved_hidden.float() - reference_hidden.float()).abs().max().item()
            )
            anchors, keep, logits = get_logits(
                OnlineMMFlashModel, capture, target, draft, draft.block_size, device
            )
            sample = {
                "row_id": row_id,
                "line": line,
                "aux_layer_ids": list(draft.target_layer_ids),
                "max_abs_aux_hidden": hidden_error,
                "ids": torch.tensor([ids]),
                "mask": torch.tensor([mask]),
                "logits": logits,
                "anchors": anchors,
                "keep": keep,
                "kl": torch.tensor([kl[p] for p in positions]),
                "entropy": torch.tensor([entropy[p] for p in positions]),
                "channel": calibrate(table, row_id, ids, mask),
            }
            write_sample(work, checked, sample, OnlineMMFlashModel, args.device)
            checked += 1
            if checked == args.rows:
                break
    if checked != args.rows:
        raise ValueError(f"Requested {args.rows} captured image rows, found {checked}")


def write_sample(work, index, sample, wrapper, device):
    for dtype in ("float32", "bfloat16"):
        sample[dtype] = replay_author(
            wrapper, sample, dtype=getattr(torch, dtype), device=device
        )
    torch.save(sample, work / f"sample-{index}.pt")


def compare(args, work):
    sys.path.insert(0, str(REPO))
    from specforge.data.visual_kl import visual_kl_per_token
    from specforge.data.visual_score import VisualScoreTable

    if args.input:
        with args.scores.open() as f:
            rows = {str(row["id"]): row for row in map(json.loads, f)}
    results = []
    for path in sorted(work.glob("sample-*.pt")):
        sample = torch.load(path, weights_only=True)
        key = sample["row_id"]
        if args.input:
            row = rows[key]
            with_feature = args.features_dir / f"{sample['line']:08d}.ckpt"
            stored = torch.load(with_feature, map_location="cpu", weights_only=True)
            channel = stored["visual_score"].reshape(1, -1)
            kl, h = torch.tensor(row["kl"]), torch.tensor(row["entropy"])
        else:
            _, target, _, seen, blind, mask = fixture()
            kl, h, _ = visual_kl_per_token(target, seen, blind, mask)
            table = VisualScoreTable.load(work / "fixture.jsonl")
            channel = torch.tensor(
                [table.channel(key, seen["input_ids"][0].tolist(), mask)]
            )
        torch.testing.assert_close(
            channel.long(), sample["channel"].long(), atol=0, rtol=0
        )
        torch.testing.assert_close(kl, sample["kl"], atol=args.score_atol, rtol=0)
        torch.testing.assert_close(h, sample["entropy"], atol=args.score_atol, rtol=0)
        sample["channel"] = channel
        objectives = {}
        for dtype in ("float32", "bfloat16"):
            loss, grad = replay_canonical(
                sample,
                dtype=getattr(torch, dtype),
                device=args.device if args.input else "cpu",
            )
            ref_loss, ref_grad = sample[dtype]
            torch.testing.assert_close(loss, ref_loss, atol=1e-5, rtol=1e-6)
            torch.testing.assert_close(grad, ref_grad, atol=1e-6, rtol=1e-6)
            objectives[dtype] = {
                "author_loss": ref_loss.item(),
                "specforge_loss": loss.item(),
                "max_abs_gradient": (grad - ref_grad).abs().max().item(),
            }
        block_size = sample["logits"].shape[1] // sample["anchors"].shape[1]
        positions = sample["anchors"][0, :, None] + torch.arange(1, block_size)
        in_bounds = positions < channel.numel()
        safe_positions = positions.clamp(max=channel.numel() - 1)
        active = (channel[0] > 0) & sample["mask"][0].bool()
        active_slots = active[safe_positions] & sample["keep"][0, :, None] & in_bounds
        results.append(
            {
                "row_id": key,
                "scored_tokens": kl.numel(),
                "nonzero_score_tokens": active.sum().item(),
                "nonzero_supervised_block_slots": active_slots.sum().item(),
                "max_abs_kl": (kl - sample["kl"]).abs().max().item(),
                "max_abs_entropy": (h - sample["entropy"]).abs().max().item(),
                "max_abs_wire_score": 0,
                "objectives": objectives,
                **(
                    {k: sample[k] for k in ("aux_layer_ids", "max_abs_aux_hidden")}
                    if args.input
                    else {}
                ),
            }
        )
    if args.require_visual_signal and not any(
        row["nonzero_supervised_block_slots"] for row in results
    ):
        raise ValueError(
            "No replayed block has a nonzero visual weight. Increase --rows and inspect the target scores."
        )
    return {
        "author_commit": PIN,
        "mode": "scored_rows" if args.input else "cpu_fixture",
        "target": args.target if args.input else "deterministic toy target",
        "target_device": args.device if args.input else "cpu",
        "target_dtype": args.dtype if args.input else "float32",
        "objective_input": (
            "Shared logits of the seed-321 initialized author draft, SpecForge "
            "captured hidden states, and frozen target LM head"
            if args.input
            else "Shared seeded fixture logits"
        ),
        "score_atol": args.score_atol,
        "rows": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-repo", type=Path, required=True)
    parser.add_argument(
        "--input", type=Path, help="Same JSONL and working directory used by the scorer"
    )
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--features-dir", type=Path)
    parser.add_argument("--target", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--draft-config", type=Path, help="Author-format draft JSON")
    parser.add_argument(
        "--device", default="cuda", help="Device for the target and draft forwards"
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--image-root", default="")
    parser.add_argument("--last-turn-only", action="store_true")
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--score-atol", type=float, default=1e-5)
    parser.add_argument(
        "--require-visual-signal",
        action="store_true",
        help="Require at least one positively scored supervised slot among replayed blocks",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--reference-worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if len({bool(args.input), bool(args.scores), bool(args.features_dir)}) != 1:
        parser.error("--input, --scores and --features-dir must be supplied together")
    if args.rows < 1 or not 0 <= args.score_atol < float("inf"):
        parser.error("--rows must be positive and --score-atol finite and nonnegative")
    args.author_repo = args.author_repo.resolve()
    if args.draft_config is None:
        args.draft_config = args.author_repo / "configs/qwen3.5-4b-mmflash.json"
    if args.input and not args.draft_config.is_file():
        parser.error(
            f"missing {args.draft_config}; include configs in the author checkout "
            "or pass --draft-config"
        )
    torch.set_num_threads(1)
    if args.reference_worker:
        reference(args, args.reference_worker)
        return
    git = ["git", "-C", str(args.author_repo)]
    if subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip() != PIN:
        parser.error(f"The author checkout must be at {PIN}")
    if subprocess.check_output(
        git + ["status", "--porcelain", "--", "specforge"]
    ).strip():
        parser.error("The author checkout contains modified reference sources")
    with tempfile.TemporaryDirectory(prefix="mmflash-parity-") as temp:
        work = Path(temp)
        subprocess.run(
            [sys.executable, __file__, *sys.argv[1:], "--reference-worker", temp],
            check=True,
        )
        report = compare(args, work)
    payload = json.dumps(report, indent=2, allow_nan=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
