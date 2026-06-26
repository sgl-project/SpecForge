"""Tensor-parallel aware weight loading for the custom backend target models.

The custom backend layers (``VocabParallelEmbedding``, ``ColumnParallelLinear``,
``RowParallelLinear``, ``ParallelLMHead``) shard their parameters across the TP
group and rely on ``_register_load_state_dict_pre_hook(self.shard_state_dict)`` to
shard the full checkpoint weight down to the local shard at load time. Those hooks
expect to be called with a *local* (per-module, un-prefixed) state dict whose keys
are the bare parameter names (``"weight"`` / ``"bias"``).

Up to transformers 4.x, ``from_pretrained`` ultimately routed weights through
``nn.Module._load_from_state_dict`` per submodule, which fired those hooks. In
transformers 5.x the loading path was rewritten to a tensor-by-tensor loader that
never calls ``_load_from_state_dict``; it instead compares the full-checkpoint
shape against the (already sharded) parameter shape, reports a size mismatch, and
raises. As a result every tensor-parallel (``tp_size > 1``) model failed to load.

To stay compatible without depending on transformers internals, ``from_pretrained``
is overridden here. For ``tp_size == 1`` (no sharding) it delegates to the stock
transformers implementation. For ``tp_size > 1`` it builds the sharded model, reads
the full checkpoint, and drives loading per-submodule through
``_load_from_state_dict`` so the existing ``shard_state_dict`` hooks run exactly as
they were designed to.
"""

import json
import os
from typing import Optional

import torch
import torch.distributed as dist
from safetensors.torch import load_file as _safe_load_file
from transformers import AutoConfig

from specforge.distributed import get_tp_group

_SAFE_INDEX = "model.safetensors.index.json"
_SAFE_SINGLE = "model.safetensors"
_BIN_INDEX = "pytorch_model.bin.index.json"
_BIN_SINGLE = "pytorch_model.bin"


def _resolve_checkpoint_dir(
    pretrained_model_name_or_path: str, cache_dir: Optional[str] = None
) -> str:
    """Return a local directory that contains the checkpoint files, downloading
    from the Hugging Face Hub if necessary."""
    if os.path.isdir(pretrained_model_name_or_path):
        return pretrained_model_name_or_path

    from huggingface_hub import snapshot_download

    return snapshot_download(
        pretrained_model_name_or_path,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.bin", "*.json"],
    )


def _load_full_state_dict(
    pretrained_model_name_or_path: str, cache_dir: Optional[str] = None
) -> dict:
    """Read the complete (un-sharded) checkpoint into a single state dict.

    Supports single-file and sharded ``safetensors`` and ``pytorch_model.bin``
    checkpoints.
    """
    ckpt_dir = _resolve_checkpoint_dir(pretrained_model_name_or_path, cache_dir)

    def _load_from_index(index_path: str, loader) -> dict:
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        state_dict = {}
        for shard_file in sorted(set(weight_map.values())):
            state_dict.update(loader(os.path.join(ckpt_dir, shard_file)))
        return state_dict

    def _load_bin(path: str) -> dict:
        return torch.load(path, map_location="cpu", weights_only=True)

    safe_index = os.path.join(ckpt_dir, _SAFE_INDEX)
    safe_single = os.path.join(ckpt_dir, _SAFE_SINGLE)
    bin_index = os.path.join(ckpt_dir, _BIN_INDEX)
    bin_single = os.path.join(ckpt_dir, _BIN_SINGLE)

    if os.path.exists(safe_index):
        return _load_from_index(safe_index, _safe_load_file)
    if os.path.exists(safe_single):
        return _safe_load_file(safe_single)
    if os.path.exists(bin_index):
        return _load_from_index(bin_index, _load_bin)
    if os.path.exists(bin_single):
        return _load_bin(bin_single)

    raise FileNotFoundError(
        f"Could not find a safetensors or pytorch_model.bin checkpoint in {ckpt_dir!r}."
    )


@torch.no_grad()
def _load_sharded_state_dict(model: torch.nn.Module, full_state_dict: dict) -> None:
    """Load ``full_state_dict`` into ``model`` one submodule at a time.

    Each submodule is loaded through ``_load_from_state_dict`` with a local,
    bare-key state dict (prefix ``""``). This fires the layer's
    ``shard_state_dict`` pre-hook, which shards the full checkpoint tensor down to
    the rank-local shard before it is copied into the (already sharded) parameter.
    Submodules without such a hook simply receive their weights unchanged.
    """
    error_msgs: list[str] = []
    loaded_keys: set[str] = set()

    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        local_state_dict = {}
        for param_name, _ in list(module.named_parameters(recurse=False)) + list(
            module.named_buffers(recurse=False)
        ):
            full_key = prefix + param_name
            if full_key in full_state_dict:
                local_state_dict[param_name] = full_state_dict[full_key]
                loaded_keys.add(full_key)

        if not local_state_dict:
            continue

        # prefix="" so the bare-key shard hooks find their tensors; non-recursive
        # so every module is processed exactly once (no double sharding).
        module._load_from_state_dict(
            local_state_dict, "", {}, False, [], [], error_msgs
        )

    if error_msgs:
        raise RuntimeError(
            "Error(s) loading sharded checkpoint:\n\t" + "\n\t".join(error_msgs)
        )


class TPShardedFromPretrainedMixin:
    """Mixin providing a tensor-parallel aware ``from_pretrained``.

    Placed before the transformers ``PreTrainedModel`` base in the MRO so this
    ``from_pretrained`` takes precedence.
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        config=None,
        torch_dtype=None,
        cache_dir=None,
        **kwargs,
    ):
        tp_group = get_tp_group()
        tp_size = (
            dist.get_world_size(tp_group)
            if dist.is_available() and dist.is_initialized() and tp_group is not None
            else 1
        )

        # Without sharding the parameter shapes match the checkpoint, so the stock
        # transformers loader works as-is.
        if tp_size == 1:
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                torch_dtype=torch_dtype,
                cache_dir=cache_dir,
                **kwargs,
            )

        if config is None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, cache_dir=cache_dir
            )

        model = cls(config, *model_args)
        if torch_dtype is not None:
            model = model.to(torch_dtype)

        full_state_dict = _load_full_state_dict(
            pretrained_model_name_or_path, cache_dir=cache_dir
        )
        if torch_dtype is not None:
            full_state_dict = {
                key: value.to(torch_dtype) if value.is_floating_point() else value
                for key, value in full_state_dict.items()
            }

        _load_sharded_state_dict(model, full_state_dict)
        # Re-establish tied weights (e.g. lm_head <- embed_tokens) after loading.
        model.tie_weights()
        model.eval()
        return model
