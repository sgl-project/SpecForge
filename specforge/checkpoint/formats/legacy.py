"""Weight names for supported historical draft checkpoints, not native targets."""

import torch

from specforge.config.draft.codec import decode_legacy_config

ARCHITECTURES = {
    "DFlashDraftModel",
    "DFlash2DraftModel",
    "DSparkDraftModel",
    "DSpineDraftModel",
    "DominoDraftModel",
    "PEagleDraftModel",
    "LlamaForCausalLMEagle3",
    "Qwen3_5MTPDraftModel",
}


def _recorded_settings(directory, architecture):
    path = directory / "training_state.pt"
    if not path.is_file():
        return {}
    # Read structural provenance only; never restore optimizer or trainer state.
    state = torch.load(path, map_location="meta", weights_only=True, mmap=True)
    provenance = state.get("specforge_model_provenance")
    try:
        settings = {"capture_layers": dict(provenance or ()).get("capture_layers")}
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid legacy model provenance") from exc
    if architecture == "PEagleDraftModel":
        settings.update(
            mask_token_id=state.get("peagle_mask_token_id"),
            norm_before_residual=state.get("peagle_norm_before_residual"),
        )
    return settings


def read_legacy(payload, store, target_config):
    if target_config is None:
        raise ValueError(
            "legacy import requires target configuration to resolve saved semantics"
        )
    architecture = payload["architectures"][0]
    settings = (
        _recorded_settings(store.root, architecture)
        if architecture in {"LlamaForCausalLMEagle3", "PEagleDraftModel"}
        else {}
    )
    config = decode_legacy_config(payload, target_config=target_config, **settings)
    if architecture == "DFlashDraftModel":
        architecture = config.architecture
    weights = {}
    for key, ref in store.refs().items():
        name = key
        if architecture in {"DSparkDraftModel", "DominoDraftModel"} and name.startswith(
            (
                "logit_head.markov_head.",
                "logit_head.confidence_head.",
                "logit_head.prefix_gru.",
                "logit_head.embed_proj.",
            )
        ):
            name = name.removeprefix("logit_head.")
        if architecture == "Qwen3_5MTPDraftModel":
            name = name.removeprefix("mtp.")
        if name in {n + ".weight" for n in config.target_bindings}:
            continue  # Old formats sometimes stored frozen target token tables.
        for prefix, replacement in (
            ("layers.", "backbone.layers."),
            ("midlayer.", "backbone.layers.0."),
            ("norm.", "backbone.final_norm."),
            ("fc.", "fusion.fc."),
            ("fc_norm.", "fusion.fc_norm."),
            ("hidden_norm.", "fusion.hidden_norm."),
            ("pre_fc_norm_embedding.", "fusion.pre_fc_norm_embedding."),
            ("pre_fc_norm_hidden.", "fusion.pre_fc_norm_hidden."),
        ):
            if name.startswith(prefix):
                name = replacement + name[len(prefix) :]
                break
        if architecture == "DominoDraftModel" and name.startswith(
            ("prefix_gru.", "embed_proj.")
        ):
            name = "gru_head." + name
        if architecture in {"PEagleDraftModel", "LlamaForCausalLMEagle3"} and name in {
            "t2d",
            "d2t",
        }:
            name = "vocabulary." + name
        if name in weights:
            raise ValueError(f"legacy tensors collide at {name}")
        weights[name] = ref
    # A weights-only warm start may explicitly allow a new vocabulary mapping.
    # Continuation still rejects missing buffers through ordinary plan validation.
    return config, weights, {}
