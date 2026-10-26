"""Warm-start a Qwen MoE DSpark drafter from Qwen3.8-Flash-Next's MTP layer.

The target ships one MTP layer (``mtp.layers.0``) whose attention and MoE have
exactly the drafter's shapes: 24 query / 2 KV heads of 256 with a gated output,
512 routed experts of width 640 behind a softmax top-10 router, one
sigmoid-gated shared expert. Those tensors seed every drafter layer.

What maps, and how:

* ``self_attn.{q,k,v,o}_proj`` -- copied. ``q_proj`` already interleaves
  ``[query | gate]`` per head, the layout ``attn_output_gate`` expects.
* ``self_attn.{q,k}_norm`` -- the target's RMSNorm scales by ``1 + w``; the
  drafter's Qwen3 RMSNorm by ``w``. The copy adds one.
* ``mlp.gate`` and the routed experts -- the stacked ``experts.gate_up_proj``
  ``[E, 2I, H]`` splits into each expert's ``gate_proj`` (first ``I`` rows) and
  ``up_proj``; ``down_proj`` ``[E, H, I]`` maps one to one. A drafter with fewer
  experts takes the listed subset, router rows included.
* ``mlp.shared_expert`` and ``mlp.shared_expert_gate`` -- copied.

What does not map stays freshly initialized: the input projection (``fc``,
``hidden_norm``) -- the MTP layer reads one embedding and one final hidden state,
the drafter five tap states -- the layer norms, which in the target are
hyper-connection reads with no single-stream equivalent, the final norm, and the
Markov and confidence heads. The MTP indexer is dropped: a draft block's
context is short enough for dense attention.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch

MTP_PREFIX = "mtp.layers.0."
_ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")
_NORMS = ("q_norm", "k_norm")


def read_mtp_tensors(model_path: str) -> Dict[str, torch.Tensor]:
    """Read the ``mtp.layers.0.*`` tensors from a local Flash-Next checkpoint.

    Works for the official BF16 release and for checkpoints that keep the MTP
    layer unquantized (the ModelSlim W8A8 export passes it through in BF16).
    Only the shards that hold MTP tensors are opened.
    """

    from safetensors import safe_open

    index_names = [
        name
        for name in sorted(os.listdir(model_path))
        if name.endswith(".safetensors.index.json")
    ]
    if not index_names:
        raise FileNotFoundError(f"no *.safetensors.index.json under {model_path}")
    with open(os.path.join(model_path, index_names[0]), encoding="utf-8") as stream:
        weight_map = json.load(stream)["weight_map"]
    wanted = {
        name: shard for name, shard in weight_map.items() if name.startswith(MTP_PREFIX)
    }
    if not wanted:
        raise ValueError(f"{model_path} has no {MTP_PREFIX}* tensors")
    tensors: Dict[str, torch.Tensor] = {}
    for shard in sorted(set(wanted.values())):
        with safe_open(os.path.join(model_path, shard), framework="pt") as handle:
            for name in handle.keys():
                if name in wanted:
                    tensors[name[len(MTP_PREFIX) :]] = handle.get_tensor(name)
    return tensors


def _require(tensors: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    if name not in tensors:
        raise KeyError(f"MTP checkpoint is missing {MTP_PREFIX}{name}")
    tensor = tensors[name]
    if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(
            f"{MTP_PREFIX}{name} is {tensor.dtype}; warm start needs an "
            "unquantized MTP layer (the BF16 release, or an export that keeps it)"
        )
    return tensor


def mtp_layer_state(
    tensors: Mapping[str, torch.Tensor],
    *,
    expert_ids: Optional[Sequence[int]] = None,
) -> Dict[str, torch.Tensor]:
    """One drafter layer's tensors, keyed relative to ``layers.{i}.``.

    ``expert_ids`` selects, in order, which MTP experts become the drafter's
    experts ``0..len(expert_ids) - 1``; ``None`` takes all of them.
    """

    state: Dict[str, torch.Tensor] = {}
    for name in _ATTENTION:
        state[f"self_attn.{name}.weight"] = _require(
            tensors, f"self_attn.{name}.weight"
        )
    for name in _NORMS:
        weight = _require(tensors, f"self_attn.{name}.weight")
        state[f"self_attn.{name}.weight"] = (weight.float() + 1.0).to(weight.dtype)

    router = _require(tensors, "mlp.gate.weight")
    gate_up = _require(tensors, "mlp.experts.gate_up_proj")
    down = _require(tensors, "mlp.experts.down_proj")
    available = router.shape[0]
    if gate_up.shape[0] != available or down.shape[0] != available:
        raise ValueError(
            f"MTP router has {available} experts but the stacked experts have "
            f"{gate_up.shape[0]} / {down.shape[0]}"
        )
    ids = list(range(available)) if expert_ids is None else [int(i) for i in expert_ids]
    if len(set(ids)) != len(ids) or not all(0 <= i < available for i in ids):
        raise ValueError(
            f"expert_ids must be distinct ids in [0, {available}), got {ids[:16]}..."
        )
    intermediate = gate_up.shape[1] // 2
    state["mlp.gate.weight"] = router[ids]
    # Views into the stacked tensors, not copies: the caller copies them into
    # the drafter, and a second copy of 512 experts is ~5 GB of host memory.
    for local, source in enumerate(ids):
        state[f"mlp.experts.{local}.gate_proj.weight"] = gate_up[source, :intermediate]
        state[f"mlp.experts.{local}.up_proj.weight"] = gate_up[source, intermediate:]
        state[f"mlp.experts.{local}.down_proj.weight"] = down[source]
    for name in ("gate_proj", "up_proj", "down_proj"):
        state[f"mlp.shared_expert.{name}.weight"] = _require(
            tensors, f"mlp.shared_expert.{name}.weight"
        )
    state["mlp.shared_expert_gate.weight"] = _require(
        tensors, "mlp.shared_expert_gate.weight"
    )
    return state


def apply_mtp_warm_start(
    model: torch.nn.Module,
    tensors: Mapping[str, torch.Tensor],
    *,
    layers: Optional[Iterable[int]] = None,
    expert_ids: Optional[Sequence[int]] = None,
) -> List[str]:
    """Copy the MTP layer into ``model``'s drafter layers (all by default).

    Shapes are checked against the model, so a drafter that does not mirror the
    MTP layer fails here rather than training from a silent partial copy.
    Returns the parameter names written.
    """

    num_layers = len(model.layers)
    targets = list(range(num_layers)) if layers is None else sorted(set(layers))
    if not targets or not all(0 <= i < num_layers for i in targets):
        raise ValueError(f"layers must lie in [0, {num_layers}), got {targets}")
    experts = int(getattr(model.config, "num_experts", 0) or 0)
    if expert_ids is None and experts != _require(tensors, "mlp.gate.weight").shape[0]:
        raise ValueError(
            f"the drafter has {experts} experts and the MTP layer "
            f"{_require(tensors, 'mlp.gate.weight').shape[0]}; pass expert_ids "
            "to choose which ones it starts from"
        )
    if expert_ids is not None and len(expert_ids) != experts:
        raise ValueError(
            f"expert_ids lists {len(expert_ids)} experts for a drafter with {experts}"
        )

    layer_state = mtp_layer_state(tensors, expert_ids=expert_ids)
    own = model.state_dict()
    written = []
    with torch.no_grad():
        for index in targets:
            for suffix, tensor in layer_state.items():
                name = f"layers.{index}.{suffix}"
                if name not in own:
                    raise KeyError(f"drafter has no parameter {name}")
                if own[name].shape != tensor.shape:
                    raise ValueError(
                        f"{name}: drafter {tuple(own[name].shape)} vs MTP "
                        f"{tuple(tensor.shape)}"
                    )
                own[name].copy_(tensor.to(own[name].dtype))
                written.append(name)
    return written


__all__ = [
    "MTP_PREFIX",
    "apply_mtp_warm_start",
    "mtp_layer_state",
    "read_mtp_tensors",
]
