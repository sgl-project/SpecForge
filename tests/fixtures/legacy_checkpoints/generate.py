"""Generate the legacy checkpoint manifests on a tree that still reads them.

Run at this path in the last tree whose package holds the legacy models and
reader (transfer stack 07, commit 036ae06d), with that checkout on PYTHONPATH
and an output directory argument. The legacy model classes supply each tensor
layout; that tree's reader supplies the expected configuration and names.
"""

import json
import re
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file

from specforge.config.draft.codec import encode_config
from specforge.export.model import export_draft
from specforge.loading import load_draft, read_draft
from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.draft.config import AutoDraftConfig
from tests.test_checkpoint.test_loading import dflash2_payload, write_target
from tests.test_config.test_draft_legacy_dflash import legacy_payload
from tests.test_config.test_draft_legacy_domino import domino_payload
from tests.test_config.test_draft_legacy_dspark import dspark_payload
from tests.test_config.test_draft_legacy_dspine import dspine_payload
from tests.test_config.test_draft_legacy_eagle3 import eagle_payload
from tests.test_config.test_draft_legacy_peagle import peagle_payload

MTP = Path(__file__).resolve().parents[1] / "legacy_mtp"
SMALL = dict(
    hidden_size=32,
    intermediate_size=64,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    vocab_size=64,
    draft_vocab_size=64,
    bos_token_id=None,
    eos_token_id=None,
    pad_token_id=None,
)


def context(payload, **method):
    payload = payload | SMALL
    payload.update(
        num_hidden_layers=2,
        num_target_layers=128,
        layer_types=["full_attention"] * 2,
    )
    payload["dflash_config"] = payload["dflash_config"] | dict(
        mask_token_id=63, target_layer_ids=[1, 3], **method
    )
    return payload


# name -> (legacy config.json, settings the old route took from the recipe,
# target configuration fields beyond write_target's)
CASES = {
    "dflash": (context(legacy_payload()), {}, {}),
    "dflash2": (context(dflash2_payload()), {}, {}),
    "dspark": (context(dspark_payload(), markov_rank=8), {}, {}),
    "dspine": (context(dspine_payload()), {}, {}),
    "dflash_mla": (
        {
            key: value
            for key, value in context(legacy_payload(), attention_mode="mla").items()
            if key != "head_dim"
        }
        | dict(
            q_lora_rank=None,
            kv_lora_rank=16,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            v_head_dim=8,
            rope_interleave=True,
        ),
        {},
        {},
    ),
    "dspark_mha": (
        context(dspark_payload(), markov_rank=8, attention_mode="mha")
        | {"num_key_value_heads": 4},
        {},
        {},
    ),
    "domino": (
        context(domino_payload(), emb_dim=16, gru_hidden_dim=24) | {"emb_dim": 16},
        {},
        {},
    ),
    "eagle3": (
        eagle_payload()
        | SMALL
        | dict(target_hidden_size=32, draft_vocab_size=32, fc_norm=True),
        {},
        {},
    ),
    "peagle": (
        peagle_payload()
        | SMALL
        | dict(target_hidden_size=32, num_hidden_layers=2, pad_token_id=0),
        {"mask_token_id": 63},
        {},
    ),
    "mtp": (json.loads((MTP / "config.json").read_text()), {}, {}),
    # a shared head is saved with the target's muP division folded in
    "mtp_shared_head": (
        json.loads((MTP / "config.json").read_text())
        | {"mtp_config": {"share_lm_head": True}},
        {},
        {"logits_mup_width_multiplier": 3},
    ),
}


# name -> (base case, changed legacy config fields, changed target configuration
# fields); none changes the base's tensor layout
VARIANTS = {
    "dflash_sliding_window": (
        "dflash",
        dict(
            use_sliding_window=True,
            sliding_window=4,
            layer_types=["sliding_attention", "full_attention"],
        ),
        {},
    ),
    "dflash2_bidirectional_window": (
        "dflash2",
        dict(
            use_sliding_window=True,
            sliding_window=4,
            layer_types=["sliding_attention"] * 2,
            is_causal=False,
            rope_parameters={"rope_theta": 1000000, "rope_type": "default"},
        ),
        {},
    ),
    "dflash_default_layers": (
        "dflash",
        dict(
            use_sliding_window=True,
            sliding_window=4,
            max_window_layers=1,
            layer_types=None,
            dflash_config={"mask_token_id": 63},
        ),
        {},
    ),
    "dspark_yarn_rope": (
        "dspark",
        dict(
            rope_parameters={
                "factor": 16.0,
                "original_max_position_embeddings": 2560,
                "rope_theta": 1000000,
                "rope_type": "yarn",
            }
        ),
        {},
    ),
    "eagle3_llama3_rope": (
        "eagle3",
        dict(
            rope_scaling={
                "factor": 8.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 4096,
                "rope_type": "llama3",
            }
        ),
        {},
    ),
    "eagle3_yarn_rope": (
        "eagle3",
        dict(
            rope_scaling={
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "factor": 40.0,
                "mscale": 0.707,
                "mscale_all_dim": 0.707,
                "original_max_position_embeddings": 1024,
                "rope_type": "yarn",
            }
        ),
        {},
    ),
    "eagle3_mrope": (
        "eagle3",
        dict(rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]}),
        {},
    ),
    "eagle3_capture_layers": (
        "eagle3",
        dict(
            eagle_config={
                "eagle_aux_hidden_state_layer_ids": [57, 1, 29],
                "use_aux_hidden_state": True,
            },
            norm_output=True,
        ),
        {},
    ),
    "mtp_multi_axis_rope": (
        "mtp",
        dict(rope_parameters={"rope_theta": 10000000.0}),
        dict(
            rope_parameters={
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.5,
                "mrope_section": [1, 1, 0],
                "mrope_interleaved": True,
            }
        ),
    ),
    "peagle_unused_rope_scaling": (
        "peagle",
        dict(
            rope_scaling={"rope_type": "linear", "factor": 4},
            norm_before_residual=True,
            eagle_config={"eagle_aux_hidden_state_layer_ids": [3, 1, 2]},
        ),
        {},
    ),
}


def legacy_state(name, payload):
    if name.startswith("mtp"):  # its legacy model class was already removed
        return load_file(MTP / "model.safetensors")
    model = AutoDraftModel.from_config(AutoDraftConfig.from_dict(payload)).float()
    if name == "eagle3":
        model.t2d[32:] = False
    if name == "dspine":
        model.transfer_ready.fill_(True)
    return model.state_dict()


def compact(fixture):
    """Indented JSON with scalar arrays and short leaf objects on one line."""
    text = json.dumps(fixture, indent=1)
    text = re.sub(
        r"\[\s+([^\[\]{}]*?)\s+\]",
        lambda match: "[" + re.sub(r"\s+", " ", match.group(1)) + "]",
        text,
    )

    def leaf(match):
        line = "{" + re.sub(r"\s+", " ", match.group(1)) + "}"
        return line if len(line) <= 72 else match.group(0)

    return re.sub(r"\{\s+([^{}]*?)\s+\}", leaf, text)


def read(payload, settings, target_extra, state):
    """Write the checkpoint and return (state, target config, what this tree read)."""
    with tempfile.TemporaryDirectory() as tmp:
        source, target = Path(tmp) / "draft", Path(tmp) / "target"
        tables = write_target(target)
        target_config = json.loads((target / "config.json").read_text()) | target_extra
        (target / "config.json").write_text(json.dumps(target_config))
        state = dict(state)
        source.mkdir()
        (source / "config.json").write_text(json.dumps(payload))
        torch.save(state, source / "pytorch_model.bin")
        kwargs = dict(target=target, legacy_settings=settings or None)
        # saved copies of target-bound tables must equal the target's
        for module, ref in read_draft(source, **kwargs).bound_copies.items():
            if module == "lm_head":
                state[ref.key] = tables["lm_head.weight"] / target_config.get(
                    "logits_mup_width_multiplier", 1
                )
            else:
                state[ref.key] = tables["model.embed_tokens.weight"].clone()
        torch.save(state, source / "pytorch_model.bin")
        checkpoint = read_draft(source, **kwargs)
        load_draft(source, **kwargs)
        if not settings:  # the old export took no recipe settings
            exported = Path(tmp) / "exported"
            export_draft(source, None, exported, target_model=str(target))
            assert set(load_file(exported / "model.safetensors")) == set(
                checkpoint.weights
            )
        expected = {
            "config": encode_config(checkpoint.config),
            "names": {
                ref.key: canonical
                for canonical, ref in sorted(checkpoint.weights.items())
            },
            "bound_copies": {
                module: ref.key
                for module, ref in sorted(checkpoint.bound_copies.items())
            },
        }
        return state, target_config, expected


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    fixtures, states = {}, {}
    for name, (payload, settings, target_extra) in CASES.items():
        states[name], target_config, expected = read(
            payload, settings, target_extra, legacy_state(name, payload)
        )
        fixtures[name] = {
            "config": payload,
            "settings": settings,
            "target_config": target_config,
            "tensors": {
                key: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype).removeprefix("torch."),
                }
                | (
                    {}
                    if value.is_floating_point()
                    else {"values": value.flatten().tolist()}
                )
                for key, value in states[name].items()
            },
            "expected": expected,
        }
        (output / f"{name}.json").write_text(compact(fixtures[name]) + "\n")
        print(name, len(states[name]), "tensors")
    variants = {}
    for name, (base, changes, target_changes) in VARIANTS.items():
        payload, settings, target_extra = CASES[base]
        _, _, expected = read(
            payload | changes, settings, target_extra | target_changes, states[base]
        )
        recorded = fixtures[base]["expected"]
        assert expected["names"] == recorded["names"]
        assert expected["bound_copies"] == recorded["bound_copies"]
        assert expected["config"].keys() == recorded["config"].keys()
        variants[name] = {
            "base": base,
            "config": changes,
            "target_config": target_changes,
            # the canonical fields that differ from the base's
            "expected": {
                key: value
                for key, value in expected["config"].items()
                if value != recorded["config"][key]
            },
        }
        print(name, sorted(variants[name]["expected"]))
    (output / "variants.json").write_text(compact(variants) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
