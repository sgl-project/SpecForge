"""Expand explicit warm-start/target mappings into ordinary tensor assignments."""

import logging
from collections.abc import Mapping

from .weight_plan import make_weight_plan

logger = logging.getLogger(__name__)


def resolve_bindings(config, store, target_config):
    if not config.target_bindings:
        return {}
    if store is None or target_config is None:
        raise ValueError(
            "target checkpoint and configuration are required for target_bindings"
        )
    from .target import resolve_target_tensor

    return {
        name: resolve_target_tensor(store, target_config, name)
        for name in config.target_bindings
    }


def _match(name, prefix):
    parts = prefix.rstrip(".").split(".") if prefix else []
    names, captures = name.split("."), []
    if len(parts) > len(names):
        return None
    for part, value in zip(parts, names):
        if part == "*":
            captures.append(value)
        elif part != value:
            return None
    return captures, ".".join(names[len(parts) :])


def _destination(prefix, captures, suffix):
    parts = prefix.rstrip(".").split(".") if prefix else []
    if parts.count("*") != len(captures):
        raise ValueError(
            "source and destination mappings must have the same wildcard count"
        )
    values = iter(captures)
    return ".".join(
        [next(values) if p == "*" else p for p in parts] + ([suffix] if suffix else [])
    )


def _expand(refs, mapping, *, bound=()):
    if not isinstance(mapping, Mapping):
        raise ValueError("weight mapping must be destination-prefix -> source-prefix")
    result, used = {}, set()
    for destination, entry in mapping.items():
        if not isinstance(destination, str):
            raise ValueError("destination prefixes must be strings")
        if isinstance(entry, str):
            source, renames = entry, {}
        elif isinstance(entry, Mapping) and not entry.keys() - {"from", "rename"}:
            source, renames = entry.get("from"), entry.get("rename", {})
        else:
            raise ValueError(f"invalid mapping for {destination}")
        if (
            not isinstance(source, str)
            or not isinstance(renames, Mapping)
            or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in renames.items()
            )
        ):
            raise ValueError(f"invalid source/renames for {destination}")
        matched = False
        for key, ref in refs.items():
            match = _match(key, source)
            if match is None:
                continue
            matched = True
            captures, suffix = match
            replacements = [
                (old, new, _match(suffix, old))
                for old, new in renames.items()
                if _match(suffix, old) is not None
            ]
            if len(replacements) > 1:
                raise ValueError(f"overlapping renames for {key}")
            if replacements:
                _, new, (inner, rest) = replacements[0]
                suffix = _destination(new, inner, rest)
            name = _destination(destination, captures, suffix)
            if name in {n + ".weight" for n in bound} and destination == "":
                continue  # A root copy explicitly excludes target-bound token tables.
            if name in result:
                raise ValueError(f"overlapping weight mappings at {name}")
            result[name] = ref
            used.add(key)
        if not matched:
            raise ValueError(
                f"mapping {destination} <- {source} selects no source tensors"
            )
    return result, used


def make_initialization_plan(
    model,
    *,
    weights_from=None,
    mapping=None,
    allow_new=(),
    target_store=None,
    target_config=None,
    from_target=None,
    cast_target_copies=(),
):
    """Scratch or weights-only warm start; never used to resume a checkpoint.

    from_target performs one-to-one bit-preserving copies with explicit renames;
    the destinations in cast_target_copies cast to the draft dtype instead.
    Current components store separate parameters; no fused-parameter layout is
    inferred from model-family names. Unexplained target tensors fail validation.
    """
    if weights_from is None and (mapping is not None or allow_new):
        raise ValueError("mapping/allow_new require weights_from")
    if isinstance(allow_new, (str, bytes)) or not all(
        isinstance(p, str) for p in allow_new
    ):
        raise ValueError("allow_new must be a sequence of tensor prefixes")
    state = model.state_dict(keep_vars=True)
    weights, unused = {}, set()
    allowed = {
        name
        for name in state
        if any(_match(name, pattern) is not None for pattern in allow_new)
    }
    for pattern in allow_new:
        if not isinstance(pattern, str) or not any(
            _match(n, pattern) is not None for n in state
        ):
            raise ValueError(f"allow_new selects no draft tensor: {pattern}")
    if weights_from is not None:
        # a legacy source's stored bound tables are ordinary source tensors here
        source = dict(weights_from.weights) | {
            f"{name}.weight": ref for name, ref in weights_from.bound_copies.items()
        }
        if mapping is None:
            weights = {
                name: ref
                for name, ref in source.items()
                if name in state and name not in allowed
            }
            used = weights.keys()
        else:
            weights, used = _expand(source, mapping)
        unused = source.keys() - used
    copies = {}
    if from_target is not None:
        if target_store is None:
            raise ValueError("from_target requires a target checkpoint")
        copies, _ = _expand(
            target_store.refs(), from_target, bound=model.config.draft.target_bindings
        )
        covered = {
            n for n in state if any(_match(n, p) is not None for p in from_target)
        }
        if covered - copies.keys():
            raise ValueError(
                f"target copy missing tensors: {sorted(covered - copies.keys())}"
            )
        overlap = (covered | copies.keys()) & (weights.keys() | allowed)
        if overlap:
            raise ValueError(
                f"from_target overlaps weights_from or allow_new: {sorted(overlap)}"
            )
        unchecked = sorted(
            ref.key
            for ref in copies.values()
            if target_store.files[ref.key].suffix != ".safetensors"
        )
        if unchecked:
            raise ValueError(
                f"strict target copies require safetensors header metadata: {unchecked}"
            )
        logger.info(
            "Target weight mapping: %s", {k: v.key for k, v in sorted(copies.items())}
        )
    cast = set(cast_target_copies)
    if cast - copies.keys():
        raise ValueError(
            f"cast_target_copies names no target copy: {sorted(cast - copies.keys())}"
        )
    weights.update(copies)
    if weights_from is None:
        # by identity: the tied alias of a copied tensor is not new
        assigned = {id(state[name]) for name in weights if name in state}
        allowed = {name for name in state if id(state[name]) not in assigned}
    if unused:
        logger.warning("Unused source draft tensors: %s", sorted(unused))
    return make_weight_plan(
        model,
        weights,
        allow_new=allowed,
        strict_copies=[name for name in copies if name not in cast],
        bindings=resolve_bindings(model.config.draft, target_store, target_config),
        unused=unused,
    )
