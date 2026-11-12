"""Dispatch native exports and publish validated replacement weights."""

import json
import os
import shutil
import tempfile
from pathlib import Path

from safetensors.torch import save_file

from specforge.checkpoint.initialization import resolve_bindings
from specforge.checkpoint.native import resolve_native_target
from specforge.checkpoint.resolve import resolve_directory
from specforge.checkpoint.tensors import TensorStore
from specforge.checkpoint.weight_plan import _validate_unquantized


def require(condition, message):
    if not condition:
        raise ValueError(f"Native export: {message}")


def export_native(config, state, target, output, *, copy_target=False, metadata=None):
    require(target is not None, "native export requires --target-model")
    source = resolve_directory(target)
    payload = json.loads((source / "config.json").read_text())
    text = payload.get("text_config", payload)
    native = resolve_native_target(config, text)
    require(
        not payload.get("quantization_config") and not text.get("quantization_config"),
        "native merge requires an unquantized target",
    )
    native.validate(config, text)
    payload = payload | {
        "specforge_artifact": metadata or {"format": "native", "version": 1}
    }
    native.export(config, state, source, payload, output, copy_target=copy_target)


def write_native(
    config,
    weights,
    source,
    payload,
    output,
    *,
    prefix,
    binding_names,
    optional_weights=(),
    copy_target=False,
):
    """Replace a native weight subtree after metadata-only validation.

    Family implementations supply the complete destination names and payload.
    binding_names maps each draft binding to its destination checkpoint key.
    """
    text = payload.get("text_config", payload)
    store = TensorStore(source)
    require(
        all(path.suffix == ".safetensors" for path in store.files.values()),
        "native export requires safetensors target shards",
    )
    bindings = resolve_bindings(config, store, text)
    for name, ref in bindings.items():
        _validate_unquantized(name, ref)
        require(
            ref.info.dtype.is_floating_point,
            f"{name}: native binding requires unquantized floating weights",
        )
    shapes = {name: tuple(value.shape) for name, value in weights.items()}
    shapes.update(
        {binding_names[name]: ref.info.shape for name, ref in bindings.items()}
    )
    previous = {
        name: info for name, info in store.metadata.items() if name.startswith(prefix)
    }
    missing = previous.keys() - shapes.keys()
    added = shapes.keys() - previous.keys() - set(optional_weights)
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
        weights[binding_names[name]] = ref.read()
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
            retained = [name for name in names if not name.startswith(prefix)]
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
        payload.setdefault("specforge_artifact", {"format": "native", "version": 1})
        (staging / "config.json").write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(staging, destination)
