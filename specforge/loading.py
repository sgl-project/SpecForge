"""Public draft loading: files and weight plans, never training state."""

from dataclasses import dataclass

import torch

from specforge.checkpoint import apply_weight_plan, make_weight_plan
from specforge.checkpoint.initialization import (
    make_initialization_plan,
    resolve_bindings,
)
from specforge.checkpoint.resolve import (
    read_checkpoint,
    read_target_config,
    resolve_directory,
)
from specforge.checkpoint.target import TargetWeights
from specforge.checkpoint.tensors import TensorStore
from specforge.config.draft import DraftConfig
from specforge.config.draft.codec.target import validate_target
from specforge.modeling import build_draft_model
from specforge.modeling.draft.models.vocabulary import DraftVocabulary


@dataclass(frozen=True)
class DraftWeights:
    """What a constructed draft still has to receive to be the one requested.

    A weight plan (copies under canonical keys and bound target tables) and
    whether the draft then finishes its own initialization (scratch or new
    structure: derived state such as DSpine's transfer space). Construction
    draws the random initialization; ``load`` writes only what the plan names.
    """

    plan: object
    finishes_initialization: bool

    @property
    def loads_vocabulary_mapping(self) -> bool:
        """Whether the plan writes a vocabulary mapping (``t2d`` and ``d2t``).

        Decided from the values, as the loaded vocabulary decides it: a reduced
        vocabulary saved before it had a mapping (all-false ``t2d``, all-zero
        ``d2t``) loads, but maps nothing.
        """
        weights = dict(self.plan.weights)
        for name, t2d in weights.items():
            if not name.endswith(".t2d"):
                continue
            # half a mapping is refused when the plan is written
            d2t = weights.get(name[: -len("t2d")] + "d2t")
            if d2t is not None and (
                t2d.info.shape == d2t.info.shape
                or t2d.read().any().item()
                or d2t.read().any().item()
            ):
                return True
        return False

    def load(self, model) -> None:
        """Write the plan into ``model``, then let it finish initializing."""
        apply_weight_plan(model, self.plan)
        _validate_vocabulary(model, self.plan)
        if self.finishes_initialization:
            model.finalize_initialization(
                loaded_keys=frozenset(dict(self.plan.weights))
            )


def _target_config(source, cache_dir):
    if isinstance(source, TargetWeights):
        return source.config
    if source is None:
        return None
    return read_target_config(source, cache_dir=cache_dir)


def _target_store(source, config, from_target, cache_dir):
    if source is None or not (config.target_bindings or from_target):
        return None
    return (
        source
        if isinstance(source, TargetWeights)
        else TensorStore(resolve_directory(source, cache_dir=cache_dir))
    )


def read_draft(
    source,
    *,
    target=None,
    adapter=None,
    cache_dir=None,
):
    """Read structure and tensor references without constructing/loading a model.

    target is a checkpoint path/repository, not a target model instance. A source
    must contain a canonical config.json.
    """
    target_config = _target_config(target, cache_dir)
    return read_checkpoint(
        source,
        target_config=target_config,
        adapter=adapter,
        cache_dir=cache_dir,
    )


def _require_real_device(device):
    if device is not None and torch.device(device).type == "meta":
        # to_empty would keep garbage wherever the plan writes nothing (new
        # tensors): construction is what initializes them
        raise ValueError(
            "construct the draft on a real device; a meta draft cannot be loaded"
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


def construct_draft(
    source,
    *,
    target=None,
    draft=None,
    adapter=None,
    device="cpu",
    dtype=None,
    backend="eager",
    cache_dir=None,
):
    """Construct the draft a checkpoint describes; its weights come with ``load``.

    Returns:
        ``(model, DraftWeights)``. The model holds its random initialization
        until ``DraftWeights.load(model)`` copies the checkpoint into it.
    """
    _require_real_device(device)
    target_config = _target_config(target, cache_dir)
    checkpoint = read_checkpoint(
        source,
        target_config=target_config,
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
    model = build_draft_model(
        checkpoint.config, device=device, dtype=dtype, backend=backend
    )
    # dtype is the caller's; the artifact format and auto_map describe the source dir
    skipped = {"dtype", "torch_dtype", "specforge_artifact", "auto_map"}
    for key, value in checkpoint.hf_metadata.items():
        if key not in skipped:
            setattr(model.config, key, value)
    return model, DraftWeights(plan, finishes_initialization=False)


def load_draft(source, **options):
    """Restore canonical draft weights; optional draft is an equality check.

    Constructs on meta for preflight, then constructs the real model once and
    copies into its initialized tensors. Missing weights are never randomized.
    Takes ``construct_draft``'s arguments.
    """
    model, weights = construct_draft(source, **options)
    weights.load(model)
    return model.eval()


def construct_initialized_draft(
    draft,
    *,
    target=None,
    weights_from=None,
    mapping=None,
    allow_new=(),
    require_loaded=(),
    cast_target_copies=(),
    from_target=None,
    device="cpu",
    dtype=torch.float32,
    backend="eager",
    cache_dir=None,
):
    """Construct a scratch/new-structure draft; its weights come with ``load``.

    This entry point accepts neither checkpoint continuation nor training resume;
    those use load_draft and the training layer's state restoration respectively.
    require_loaded names parameter/module prefixes that may not be initialized anew;
    a prefix naming a head the draft disables (None) requires nothing.
    cast_target_copies names from_target destinations that cast to dtype.
    """
    if type(draft) is not DraftConfig:
        raise TypeError("draft must be a resolved DraftConfig")
    _require_real_device(device)
    target_config = _target_config(target, cache_dir)
    if target_config is not None:
        validate_target(draft, target_config)
        from specforge.checkpoint.native import validate_target_initialization

        validate_target_initialization(draft, target_config, from_target)
    checkpoint = (
        None
        if weights_from is None
        else read_checkpoint(
            weights_from,
            check_target=False,
            target_config=target_config,
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
        cast_target_copies=cast_target_copies,
    )
    loaded = set(dict(plan.weights)) | {name + ".weight" for name, _ in plan.bindings}
    parameters = dict(preview.named_parameters(remove_duplicate=False))
    loaded_parameters = {id(parameters[name]) for name in loaded if name in parameters}
    for prefix in require_loaded:
        selected = {
            name: parameter
            for name, parameter in parameters.items()
            if not prefix or name == prefix or name.startswith(prefix + ".")
        }
        if not selected:
            # an optional head this draft disables (None) has nothing to load
            if prefix and hasattr(preview, prefix) and getattr(preview, prefix) is None:
                continue
            raise ValueError(f"unknown required parameter or module: {prefix!r}")
        missing = [
            name
            for name, parameter in selected.items()
            if id(parameter) not in loaded_parameters
        ]
        if missing:
            raise ValueError(f"Frozen parameters require loaded weights: {missing}")
    model = build_draft_model(draft, device=device, dtype=dtype, backend=backend)
    return model, DraftWeights(plan, finishes_initialization=True)


def initialize_draft(draft, **options):
    """Construct scratch/new-structure weights, optionally copying explicit sources.

    Takes ``construct_initialized_draft``'s arguments; the weights are loaded
    and the draft finishes its initialization before it is returned.
    """
    model, weights = construct_initialized_draft(draft, **options)
    weights.load(model)
    return model


__all__ = [
    "DraftWeights",
    "apply_weight_plan",
    "construct_draft",
    "construct_initialized_draft",
    "initialize_draft",
    "load_draft",
    "read_draft",
]
