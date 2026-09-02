# coding=utf-8
"""Checkpoint-naming boundary for MoE modules.

Checkpoint FILES (trainer checkpoints, warm-start sources, exports) use the
official per-expert naming so SGLang and HF loaders read them unchanged.
Modules may use a different native layout (e.g. stacked ``[E, out, in]``
expert tensors, which FSDP ``use_orig_params`` and grouped GEMMs want).

FSDP's full-state-dict hooks index the gathered dict by the module's own
parameter FQNs, so the rename cannot live inside ``state_dict()``; it lives at
the save/load boundary instead. Every place that reads a model's state for a
file, or loads a file into a model, goes through :func:`to_checkpoint_state_dict`
/ :func:`from_checkpoint_state_dict`: the training backend, warm start, and
the HF/SGLang exporters. Implementations with a native layout register a
converter pair; both directions must be no-ops on dicts already in the other
form, and on dense models.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

Converter = Callable[[Dict[str, object]], Dict[str, object]]

_CONVERTERS: List[Tuple[str, Converter, Converter]] = []


def register_state_dict_converter(
    name: str, *, to_checkpoint: Converter, from_checkpoint: Converter
) -> None:
    for existing, _, _ in _CONVERTERS:
        if existing == name:
            raise ValueError(f"state-dict converter {name!r} is already registered")
    _CONVERTERS.append((name, to_checkpoint, from_checkpoint))


def unregister_state_dict_converter(name: str) -> None:
    _CONVERTERS[:] = [entry for entry in _CONVERTERS if entry[0] != name]


def registered_state_dict_converters() -> List[str]:
    return [name for name, _, _ in _CONVERTERS]


def to_checkpoint_state_dict(state: Dict[str, object]) -> Dict[str, object]:
    """Module-native naming -> official checkpoint naming (identity for dense)."""
    for _, to_checkpoint, _ in _CONVERTERS:
        state = to_checkpoint(state)
    return state


def from_checkpoint_state_dict(state: Dict[str, object]) -> Dict[str, object]:
    """Official checkpoint naming -> module-native naming (identity for dense)."""
    for _, _, from_checkpoint in reversed(_CONVERTERS):
        state = from_checkpoint(state)
    return state
