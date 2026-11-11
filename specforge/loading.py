"""Public draft loading: files and weight plans, never training state."""

import torch

from specforge.checkpoint import apply_weight_plan, make_weight_plan
from specforge.checkpoint.initialization import (
    make_initialization_plan,
    resolve_bindings,
)
from specforge.checkpoint.resolve import (
    read_checkpoint,
    resolve_directory,
    target_configuration,
)
from specforge.checkpoint.tensors import TensorStore
from specforge.checkpoint.weight_plan import _validate_unquantized
from specforge.config.draft import DraftConfig
from specforge.config.draft.codec.target import validate_target
from specforge.modeling import build_draft_model
from specforge.modeling.draft.models.vocabulary import DraftVocabulary


def _target_config(source, cache_dir):
    if source is None:
        return None
    return target_configuration(
        resolve_directory(source, cache_dir=cache_dir, config_only=True)
    )


def _target_store(source, config, from_target, cache_dir):
    if source is None or not (config.target_bindings or from_target):
        return None
    return TensorStore(resolve_directory(source, cache_dir=cache_dir))


def _check_bound_copies(checkpoint, preview, plan):
    """Do not silently replace a legacy table with a different target table."""
    bindings = dict(plan.bindings)
    for name, ref in checkpoint.bound_copies.items():
        _validate_unquantized(name, ref)
        saved = ref.read()
        target = bindings[name].read().to(saved.dtype).clone()
        preview.get_submodule(name)._transform_weight(target)
        if saved.shape != target.shape or not torch.equal(saved, target):
            raise ValueError(
                f"legacy {ref.key} differs from bound target {bindings[name].key}"
            )


def read_draft(
    source, *, target=None, legacy_config=None, adapter=None, cache_dir=None
):
    """Read structure and tensor references without constructing/loading a model.

    target is a checkpoint path/repository, not a target model instance. A source
    without config.json needs an explicit supplementary legacy configuration.
    """
    target_config = _target_config(target, cache_dir)
    return read_checkpoint(
        source,
        target_config=target_config,
        legacy_config=legacy_config,
        adapter=adapter,
        cache_dir=cache_dir,
    )


def _dtype(requested, checkpoint):
    if requested is not None:
        if not isinstance(requested, torch.dtype) or not requested.is_floating_point:
            raise ValueError("dtype must be a floating-point torch.dtype")
        return requested
    types = {
        ref.info.dtype
        for ref in checkpoint.weights.values()
        if isinstance(ref.info.dtype, torch.dtype) and ref.info.dtype.is_floating_point
    }
    if len(types) != 1:
        raise ValueError(
            "checkpoint has mixed/no floating dtypes; pass dtype explicitly"
        )
    return types.pop()


def _validate_vocabulary(model, plan):
    loaded = dict(plan.weights).keys()
    for name, module in model.named_modules():
        if not isinstance(module, DraftVocabulary):
            continue
        keys = {f"{name}.t2d", f"{name}.d2t"}
        if keys <= loaded:
            module.validate_draft_state()
        elif keys & loaded:
            raise ValueError("load t2d and d2t together as one vocabulary mapping")


def load_draft(
    source,
    *,
    target=None,
    draft=None,
    legacy_config=None,
    adapter=None,
    device="cpu",
    dtype=None,
    backend="eager",
    cache_dir=None,
):
    """Restore canonical/legacy draft weights; optional draft is an equality check.

    Constructs on meta for preflight, then constructs the real model once and
    copies into its initialized tensors. Missing weights are never randomized.
    Target-bound weights must be supplied before the returned model can run.
    """
    target_config = _target_config(target, cache_dir)
    checkpoint = read_checkpoint(
        source,
        target_config=target_config,
        legacy_config=legacy_config,
        adapter=adapter,
        cache_dir=cache_dir,
    )
    if draft is not None and (
        type(draft) is not DraftConfig or draft != checkpoint.config
    ):
        raise ValueError(
            "supplied draft does not match checkpoint structure; use weights_from for a new structure"
        )
    dtype = _dtype(dtype, checkpoint)
    preview = build_draft_model(
        checkpoint.config, device="meta", dtype=dtype, backend=backend
    )
    target_store = _target_store(target, checkpoint.config, None, cache_dir)
    plan = make_weight_plan(
        preview,
        checkpoint.weights,
        bindings=resolve_bindings(checkpoint.config, target_store, target_config),
    )
    _check_bound_copies(checkpoint, preview, plan)
    model = build_draft_model(
        checkpoint.config, device=device, dtype=dtype, backend=backend
    )
    for key, value in checkpoint.hf_metadata.items():
        if key not in {"dtype", "torch_dtype"}:
            setattr(model.config, key, value)
    apply_weight_plan(model, plan)
    _validate_vocabulary(model, plan)
    return model.eval()


def initialize_draft(
    draft,
    *,
    target=None,
    weights_from=None,
    mapping=None,
    allow_new=(),
    from_target=None,
    device="cpu",
    dtype=torch.float32,
    backend="eager",
    cache_dir=None,
    legacy_config=None,
):
    """Construct scratch/new-structure weights, optionally copying explicit sources.

    This entry point accepts neither checkpoint continuation nor training resume;
    those use load_draft and the training layer's state restoration respectively.
    """
    if type(draft) is not DraftConfig:
        raise TypeError("draft must be a resolved DraftConfig")
    target_config = _target_config(target, cache_dir)
    if target_config is not None:
        validate_target(draft, target_config)
    checkpoint = (
        None
        if weights_from is None
        else read_checkpoint(
            weights_from,
            check_target=False,
            target_config=target_config,
            legacy_config=legacy_config,
            cache_dir=cache_dir,
        )
    )
    preview = build_draft_model(draft, device="meta", dtype=dtype, backend=backend)
    target_store = _target_store(target, draft, from_target, cache_dir)
    plan = make_initialization_plan(
        preview,
        weights_from=checkpoint,
        mapping=mapping,
        allow_new=allow_new,
        target_store=target_store,
        target_config=target_config,
        from_target=from_target,
    )
    model = build_draft_model(draft, device=device, dtype=dtype, backend=backend)
    apply_weight_plan(model, plan)
    _validate_vocabulary(model, plan)
    model.finalize_initialization(loaded_keys=frozenset(dict(plan.weights)))
    return model


__all__ = ["apply_weight_plan", "initialize_draft", "load_draft", "read_draft"]
