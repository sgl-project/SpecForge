"""Convert pre-named-loss canonical checkpoints without changing model or optimizer state."""

import argparse
import shutil
from dataclasses import asdict
from pathlib import Path

import torch

from specforge.algorithms.objectives.registry import objective_record, resolve_objective


def convert_record(state):
    """Return current loss records; all other checkpoint fields are preserved."""
    if "canonical_loss" not in state:
        raise ValueError(
            "Not a canonical training checkpoint: training_state.pt records no "
            "canonical_loss"
        )
    old_loss = state["canonical_loss"]
    if "terms" in old_loss:
        raise ValueError("Checkpoint already uses named loss terms")
    architecture = state["canonical_draft_config"]["architectures"][0]
    options = dict(state["canonical_objective_options"])
    terms = {}
    for old, term in (
        ("selector_loss_alpha", "selector"),
        ("confidence_loss_alpha", "confidence"),
        ("alignment_weight", "alignment"),
    ):
        if old in options:
            terms[term] = {"coefficient": options.pop(old)}

    # DSpark's two weights only configured its default token loss, and the old record
    # stores that loss resolved, so they are already in old_loss["options"].
    for old in ("ce_loss_alpha", "l1_loss_alpha"):
        options.pop(old, None)
    # DSpine's ce_weight still mixes the refinement loss. The token loss is the
    # recorded one: an explicit flat loss never took this weight.
    if "ce_weight" in options:
        weight = options.pop("ce_weight")
        terms["refinement"] = {
            "options": {"ce_weight": weight, "l1_weight": 1 - weight}
        }

    defaults = resolve_objective(
        architecture, name=state["canonical_objective"], options=options
    )
    for name in ("base", "final") if "base" in defaults.losses.terms else ("token",):
        terms[name] = {"kind": old_loss["kind"], "options": dict(old_loss["options"])}
    policy = dict(state.get("canonical_loss_execution", {}))
    fused = policy.pop("fused", False)
    resolved = resolve_objective(
        architecture,
        name=state["canonical_objective"],
        options=options,
        loss={"terms": terms, "fused": fused, "execution": policy},
    )
    implementations = dict(state.get("canonical_loss_implementations", {}))
    if fused and "canonical_loss_implementations" not in state:
        raise ValueError("Fused checkpoint does not record its loss implementations")
    if architecture == "DSpineDraftModel":
        for key, name in list(implementations.items()):
            if key.endswith("/compute/refinement:0"):
                replacements = {"token_ce": "token_ce_l1", "reference": "reference"}
                if name not in replacements:
                    raise ValueError(
                        f"Cannot convert refinement implementation {name!r}"
                    )
                implementations[key] = replacements[name]
                implementations[key.replace("/refinement:", "/alignment:")] = (
                    "reference"
                )
    return {
        **state,
        "canonical_loss": resolved.losses.to_dict(),
        "canonical_objective_options": objective_record(
            state["canonical_objective"], asdict(resolved.objective)
        ),
        "canonical_loss_execution": asdict(resolved.execution),
        "canonical_loss_implementations": implementations,
        "canonical_use_fused_conv": state.get("canonical_use_fused_conv", False),
    }


def convert_checkpoint(source, output):
    """Copy a checkpoint to a new directory and replace only its loss metadata."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError("Output must be outside the source checkpoint")
    if output.exists():
        raise FileExistsError(output)
    state = torch.load(
        source / "training_state.pt", map_location="cpu", weights_only=True
    )
    converted = convert_record(state)
    # Build the copy beside the destination and rename it last, so a failed
    # conversion never leaves an unconverted checkpoint at the output path.
    staging = output.with_name(output.name + ".converting")
    if staging.exists():
        shutil.rmtree(staging)
    try:
        shutil.copytree(source, staging)
        torch.save(converted, staging / "training_state.pt")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    convert_checkpoint(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
