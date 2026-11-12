"""Native Qwen3.5 merge-back; remove stale MTP tensors from physical shards."""

import copy
import json
import os
import shutil
import tempfile
from pathlib import Path

from safetensors.torch import save_file

from specforge.checkpoint.initialization import resolve_bindings
from specforge.checkpoint.native import validate_qwen3_5_mtp
from specforge.checkpoint.resolve import resolve_directory
from specforge.checkpoint.tensors import TensorStore
from specforge.checkpoint.weight_plan import _validate_unquantized
from specforge.config.draft.codec.components import decode_component

from . import artifact_metadata
from .converters import require


def export_native(config, fields, state, target, output, *, copy_target=False):
    require(target is not None, "native MTP export requires --target-model")
    source = resolve_directory(target)
    payload = json.loads((source / "config.json").read_text())
    payload = copy.deepcopy(payload)
    text = payload.get("text_config", payload)
    require(
        text.get("model_type") == "qwen3_5_text",
        "native converter supports dense Qwen3.5 targets only",
    )
    require(
        not payload.get("quantization_config") and not text.get("quantization_config"),
        "native merge requires an unquantized target",
    )
    validate_qwen3_5_mtp(config, text)
    for name in (
        "hidden_size",
        "vocab_size",
        "intermediate_size",
        "hidden_act",
        "rms_norm_eps",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "attn_output_gate",
    ):
        require(
            name in text and text[name] == fields[name],
            f"target {name} must be explicit and match draft",
        )
    require(
        text.get("is_causal", True) == fields["is_causal"],
        "target is_causal disagrees with draft",
    )
    attention = config.attention_configs[
        config.layer_configs[config.layer_types[0]].attention
    ]
    expected_rope = attention.rope_parameters or config.rope_parameters
    require(
        text.get("rope_parameters") is not None or text.get("rope_scaling") is not None,
        "target must state rope_parameters or rope_scaling",
    )
    for key in ("rope_parameters", "rope_scaling"):
        if text.get(key) is not None:
            values = dict(text[key])
            for option in ("use_fope", "video_rope", "mrope_interleaved_glm"):
                require(
                    not values.pop(option, False),
                    f"target {key}.{option} is unsupported",
                )
            kind = values.pop("type", "default")
            values.setdefault("rope_type", kind)
            if "mrope_section" in values and values["rope_type"] == "default":
                values["rope_type"] = "mrope"
            if values["rope_type"] == "mrope":
                values.setdefault("mrope_interleaved", False)
            for name in ("rope_theta", "partial_rotary_factor"):
                value = values.get(name, text.get(name))
                require(value is not None, f"target {key}.{name} must be explicit")
                require(
                    name not in text or text[name] == value,
                    f"target {key}.{name} disagrees with top-level {name}",
                )
                values[name] = value
            values.setdefault("interleaved", False)
            require(
                decode_component("rope", values) == expected_rope,
                f"target {key} disagrees with draft",
            )
    store = TensorStore(source)
    require(
        all(path.suffix == ".safetensors" for path in store.files.values()),
        "native export requires safetensors target shards",
    )
    weights = {"mtp." + key: value for key, value in state.items()}
    bindings = resolve_bindings(config, store, text)
    for name, ref in bindings.items():
        _validate_unquantized(name, ref)
        require(
            ref.info.dtype.is_floating_point,
            f"{name}: native binding requires unquantized floating weights",
        )
    shapes = {name: tuple(value.shape) for name, value in weights.items()}
    shapes.update(
        {f"mtp.{name}.weight": ref.info.shape for name, ref in bindings.items()}
    )
    previous = {
        name: info for name, info in store.metadata.items() if name.startswith("mtp.")
    }
    missing = previous.keys() - shapes.keys()
    added = (
        shapes.keys()
        - previous.keys()
        - {"mtp.embed_tokens.weight", "mtp.lm_head.weight"}
    )
    require(
        not missing and not added,
        f"native weight coverage differs: missing={sorted(missing)}, unexpected={sorted(added)}",
    )
    for name, info in previous.items():
        require(
            shapes[name] == info.shape,
            f"native weight shape differs for {name}: {shapes[name]} != {info.shape}",
        )
    for name, ref in bindings.items():
        weights[f"mtp.{name}.weight"] = ref.read()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise FileExistsError(f"checkpoint directory is not empty: {destination}")
    with tempfile.TemporaryDirectory(prefix=".native-", dir=destination.parent) as tmp:
        staging = Path(tmp) / "artifact"
        staging.mkdir()
        weight_map = {}
        # Only shard files in the selected layout are included, never an alternate
        # model.bin/model.safetensors that would override the merged index.
        files = set(store.files.values())
        for index, file in enumerate(sorted(files)):
            names = [name for name, path in store.files.items() if path == file]
            retained = [name for name in names if not name.startswith("mtp.")]
            if not retained:
                continue
            filename = f"target-{index:05d}.safetensors"
            if len(retained) != len(names):
                save_file(
                    {name: store.read(name).contiguous().clone() for name in retained},
                    staging / filename,
                )
            elif copy_target:
                shutil.copy2(file, staging / filename)
            else:
                (staging / filename).symlink_to(file.resolve())
            weight_map.update({name: filename for name in retained})
        save_file(
            {key: tensor.contiguous().clone() for key, tensor in weights.items()},
            staging / "draft.safetensors",
        )
        weight_map.update({name: "draft.safetensors" for name in weights})
        # Preserve target assets, including HF chat templates, but not stale weights.
        for file in source.iterdir():
            if (
                file.is_file()
                and file.name not in {"config.json", "artifact.json"}
                and file.suffix not in {".safetensors", ".bin", ".pt", ".pth"}
                and not file.name.endswith(".index.json")
            ):
                shutil.copy2(file, staging / file.name)
            elif file.is_dir() and file.name == "chat_templates":
                shutil.copytree(file, staging / file.name)
        (staging / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2) + "\n"
        )
        payload["specforge_artifact"] = artifact_metadata()
        (staging / "config.json").write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(staging, destination)
