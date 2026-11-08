"""Weight names for supported historical draft checkpoints, not native targets."""

from dataclasses import dataclass

import torch

from specforge.config.draft.codec import decode_legacy_config

from ..tensors import TensorInfo

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


@dataclass(frozen=True)
class _Ready:
    key: str = "vocabulary.ready"
    info: TensorInfo = TensorInfo((), torch.bool)

    def read(self, region=None):
        return torch.tensor(True)


def read_legacy(payload, store, target_config):
    if target_config is None:
        raise ValueError(
            "legacy import requires target configuration to resolve saved semantics"
        )
    config = decode_legacy_config(payload, target_config=target_config)
    architecture = payload["architectures"][0]
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
            (
                "norm.",
                (
                    "logit_norm."
                    if architecture == "LlamaForCausalLMEagle3"
                    and not config.spec_config.norm_output
                    else "backbone.final_norm."
                ),
            ),
            ("fc.", "fusion.fc."),
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
    if {"vocabulary.t2d", "vocabulary.d2t"} <= weights.keys():
        weights["vocabulary.ready"] = _Ready()
    # A weights-only warm start may explicitly allow a new vocabulary mapping.
    # Continuation still rejects missing buffers through ordinary plan validation.
    return config, weights, {}
