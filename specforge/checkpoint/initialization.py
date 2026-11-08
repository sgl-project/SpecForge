"""Expand explicit warm-start/target mappings into ordinary tensor assignments."""

import logging
from collections.abc import Mapping

from .weight_plan import make_weight_plan

logger = logging.getLogger(__name__)
_TARGET_KEYS = {
    "embed_tokens": (
        "model.language_model.embed_tokens.weight",
        "model.embed_tokens.weight",
        "embed_tokens.weight",
    ),
    "lm_head": (
        "model.language_model.lm_head.weight",
        "model.lm_head.weight",
        "lm_head.weight",
    ),
}


def resolve_bindings(config, store, target_config):
    if not config.target_bindings:
        return {}
    if store is None or target_config is None:
        raise ValueError(
            "target checkpoint and configuration are required for target_bindings"
        )
    refs, bindings = store.refs(), {}
    for module in config.target_bindings:
        candidates = [key for key in _TARGET_KEYS[module] if key in refs]
        if (
            module == "lm_head"
            and not candidates
            and target_config.get("tie_word_embeddings") is True
        ):
            candidates = [key for key in _TARGET_KEYS["embed_tokens"] if key in refs]
        if len(candidates) != 1:
            raise ValueError(f"{module}: expected one target key, found {candidates}")
        bindings[module] = refs[candidates[0]]
    return bindings


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
):
    """Scratch or weights-only warm start; never used to resume a checkpoint.

    from_target performs one-to-one bit-preserving copies with explicit renames.
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
        weights, used = _expand(
            weights_from.weights, {"": ""} if mapping is None else mapping
        )
        unused = weights_from.weights.keys() - used
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
        if (covered | copies.keys()) & (weights.keys() | allowed):
            raise ValueError("from_target overlaps weights_from or allow_new")
        if any(
            target_store.files[ref.key].suffix != ".safetensors"
            for ref in copies.values()
        ):
            raise ValueError("strict target copies require safetensors header metadata")
        logger.info(
            "Target weight mapping: %s", {k: v.key for k, v in sorted(copies.items())}
        )
    weights.update(copies)
    if weights_from is None:
        allowed = state.keys() - weights.keys()
    if unused:
        logger.info("Unused source draft tensors: %s", sorted(unused))
    return make_weight_plan(
        model,
        weights,
        allow_new=allowed,
        strict_copies=copies,
        bindings=resolve_bindings(model.config.draft, target_store, target_config),
        unused=unused,
    )
