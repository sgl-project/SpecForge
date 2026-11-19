"""Convert a historical (pre-canonical) draft checkpoint to a canonical directory."""

import argparse
import copy
from pathlib import Path
from types import SimpleNamespace

import torch

from specforge.checkpoint import TensorStore, make_weight_plan
from specforge.checkpoint.draft_dir import read_artifact_metadata, write_artifact
from specforge.checkpoint.initialization import resolve_bindings
from specforge.checkpoint.resolve import (
    read_json,
    read_target_config,
    resolve_checkpoint_path,
    resolve_directory,
)
from specforge.checkpoint.tensors import TensorInfo, TensorRef
from specforge.modeling import build_draft_model
from specforge.modeling.draft.hf_config import SpecForgeDraftConfig

try:
    from scripts.legacy_checkpoint.reader import read_legacy
except ModuleNotFoundError:
    from legacy_checkpoint.reader import read_legacy


def _read_legacy_config(directory, draft_config):
    path = directory / "config.json"
    if path.is_file():
        if draft_config is not None:
            raise ValueError(f"{directory} has a config.json; omit --draft-config")
    elif draft_config is None:
        raise ValueError(
            f"{directory} has no config.json; pass its historical configuration "
            "with --draft-config"
        )
    else:
        path = draft_config
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must hold a configuration object")
    # native and serving artifacts name their own format here and are rejected
    if payload.get("model_type") == "specforge_draft" or read_artifact_metadata(
        payload
    ):
        raise ValueError(f"{path} is already canonical; there is nothing to convert")
    return payload


def _target_tables(config, preview, target_model, target_config, *, read):
    """Reference the bound target tables; by shape alone unless they are read."""
    if read:
        store = TensorStore(resolve_directory(target_model))
        return resolve_bindings(config, store, target_config)
    shapes = SimpleNamespace(
        metadata={
            name: TensorInfo(
                tuple(preview.get_submodule(name).weight.shape), torch.float32
            )
            for name in config.target_bindings
        }
    )
    return {name: TensorRef(shapes, name) for name in config.target_bindings}


def _check_bound_copies(preview, copies, tables):
    """A canonical directory omits bound tables: a saved copy must be the target's."""
    for name, ref in copies.items():
        saved = ref.read()
        # bind as the loader will: cast to the saved dtype, then the logit transform
        module = copy.deepcopy(preview.get_submodule(name)).to_empty(device="cpu")
        module.to(saved.dtype).load_target_weight(tables[name].read())
        if not torch.equal(saved, module.bound_weight()):
            raise ValueError(
                f"saved {ref.key} differs from the target's {tables[name].key}; "
                "convert with the target model this checkpoint was trained against"
            )


def convert_checkpoint(
    source,
    output,
    *,
    target_model,
    draft_config=None,
    mask_token_id=None,
    norm_before_residual=None,
    capture_layers=None,
):
    """Write one historical checkpoint as a new canonical draft directory.

    The source is only read. Weights keep their values and dtypes under canonical
    names; optimizer, scheduler and RNG state are not converted.
    """
    if target_model is None:
        raise ValueError(
            "a target model is required: historical configurations take "
            "target-derived settings from it"
        )
    path = resolve_checkpoint_path(source)
    output = Path(output).resolve()
    if path.resolve() in (output, *output.parents):
        raise ValueError("Output must be outside the source checkpoint")
    if output.exists():
        raise FileExistsError(output)
    payload = _read_legacy_config(path.parent if path.is_file() else path, draft_config)
    target_config = read_target_config(target_model)
    settings = {
        "mask_token_id": mask_token_id,
        "norm_before_residual": norm_before_residual,
        "capture_layers": capture_layers,
    }
    config, weights = read_legacy(
        payload,
        TensorStore(path),
        target_config,
        settings={k: v for k, v in settings.items() if v is not None},
    )
    copies = {
        name: weights.pop(name + ".weight")
        for name in config.target_bindings
        if name + ".weight" in weights
    }
    # Meta construction validates every tensor name, shape and dtype before a read.
    preview = build_draft_model(config, device="meta")
    tables = _target_tables(
        config, preview, target_model, target_config, read=bool(copies)
    )
    plan = make_weight_plan(preview, weights, bindings=tables)
    _check_bound_copies(preview, copies, tables)
    state, seen = {}, set()
    for name, ref in plan.weights:
        # tied tensors are stored once
        if ref not in seen:
            state[name] = ref.read()
            seen.add(ref)
    # The writer stages beside the destination and renames last, so a failed
    # conversion leaves nothing at the output path.
    write_artifact(output, SpecForgeDraftConfig(config).to_dict(), state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="historical checkpoint directory, weights file or run directory",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--target-model",
        required=True,
        help="target the checkpoint was trained against; resolves target-derived "
        "settings and checks saved copies of its embedding / LM head",
    )
    parser.add_argument(
        "--draft-config",
        type=Path,
        help="historical draft config JSON, for a checkpoint without config.json",
    )
    parser.add_argument(
        "--mask-token-id",
        type=int,
        help="P-EAGLE: mask token of the run (the old model.mask_token_id), unless "
        "training_state.pt records it",
    )
    parser.add_argument(
        "--norm-before-residual",
        action=argparse.BooleanOptionalAction,
        help="P-EAGLE: the old training.norm_before_residual, when the config omits it",
    )
    parser.add_argument(
        "--capture-layers",
        type=int,
        nargs="+",
        metavar="LAYER",
        help="EAGLE3 / P-EAGLE: the old model.aux_hidden_state_layer_ids, when "
        "neither the config nor training_state.pt records them",
    )
    args = parser.parse_args()
    convert_checkpoint(
        args.checkpoint,
        args.output_dir,
        target_model=args.target_model,
        draft_config=args.draft_config,
        mask_token_id=args.mask_token_id,
        norm_before_residual=args.norm_before_residual,
        capture_layers=args.capture_layers,
    )


if __name__ == "__main__":
    main()
