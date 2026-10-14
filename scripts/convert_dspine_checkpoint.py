"""Package an independent DSpine reference checkpoint as a SpecForge draft."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import Qwen3Config

from specforge.modeling.draft.dspine import DSpineDraftModel


def convert_checkpoint(
    model_path: Path, config_path: Path, transfer_path: Path, output_path: Path
) -> None:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    payload = json.loads(config_path.read_text())
    payload = payload.get("config", payload)
    state = load_file(str(model_path))
    transfer = load_file(str(transfer_path))
    layers = payload["num_hidden_layers"]
    hidden = payload["hidden_size"]
    rank = state["injection.readouts.0.weight"].shape[0]
    codes = transfer["codes"]
    if codes.shape != (payload["vocab_size"], rank):
        raise ValueError(
            "Transfer codes do not match the draft vocabulary and readout rank"
        )
    if not torch.isfinite(codes).all() or not all(
        torch.isfinite(tensor).all() for tensor in state.values()
    ):
        raise ValueError("DSpine checkpoint and transfer codes must be finite")
    taps = payload.get("dflash_config", {}).get("target_layer_ids")
    if not taps or state["fc.weight"].shape != (hidden, len(taps) * hidden):
        raise ValueError(
            "Reference config must specify the exact target_layer_ids used for training"
        )
    mask_token_id = payload.get("dflash_config", {}).get("mask_token_id")
    if (
        not isinstance(mask_token_id, int)
        or not 0 <= mask_token_id < payload["vocab_size"]
    ):
        raise ValueError("Reference config must specify a valid mask_token_id")
    # The reference implementation used block16 independently of its source config.
    payload.update(
        architectures=["DSpineDraftModel"],
        block_size=16,
        use_sliding_window=False,
        sliding_window=None,
        layer_types=["full_attention"] * layers,
    )
    payload.pop("auto_map", None)
    payload["dflash_config"] = dict(
        block_size=16, mask_token_id=mask_token_id, target_layer_ids=taps
    )
    payload["dspine_config"] = dict(
        transfer_rank=rank,
        message_size=state["injection.message.weight"].shape[0],
        top_k=16,
    )
    state.update(transfer_codes=codes, transfer_ready=torch.tensor(True))
    with torch.device("meta"):
        model = DSpineDraftModel(Qwen3Config(**payload))
    model.load_state_dict(state, strict=True, assign=True)
    model.save_pretrained(output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert_checkpoint(args.model, args.config, args.transfer, args.output)


if __name__ == "__main__":
    main()
