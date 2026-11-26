"""Frozen target token modules; tensor installation only, without file I/O."""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils._python_dispatch import (
    TorchDispatchMode,
    is_traceable_wrapper_subclass,
)
from torch.utils._pytree import tree_leaves

from specforge.config.draft._validation import positive_int
from specforge.config.draft.logits import TargetLogitSpec

from .layers.logits import TargetLogitTransform
from .layers.registry import build_component

_VALUE_PRESERVING_APPLY_OPS = frozenset(
    {
        torch.ops.aten._pin_memory.default,
        torch.ops.aten._to_copy.default,
        torch.ops.aten.alias.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.contiguous.default,
        torch.ops.aten.detach.default,
        torch.ops.aten.lift_fresh.default,
        torch.ops.aten.to.device,
        torch.ops.aten.to.dtype,
        torch.ops.aten.to.dtype_layout,
        torch.ops.aten.to.other,
    }
)


@dataclass(frozen=True)
class _TensorState:
    storage_key: tuple[int, torch.device]
    offset: int
    shape: torch.Size
    stride: tuple[int, ...]
    dtype: torch.dtype
    mesh: object = None
    placements: tuple | None = None
    storage: object = field(default=None, compare=False, hash=False, repr=False)


def _tensor_state(tensor: Tensor) -> _TensorState | None:
    local = tensor
    mesh = placements = None
    if is_traceable_wrapper_subclass(tensor):
        from torch.distributed.tensor import DTensor

        if not isinstance(tensor, DTensor):
            return None
        local = tensor._local_tensor
        mesh = tensor.device_mesh
        placements = tensor.placements
    if local.is_meta or local.layout != torch.strided:
        return None
    storage = local.untyped_storage()
    return _TensorState(
        storage_key=(storage.data_ptr(), storage.device),
        offset=local.storage_offset(),
        shape=tensor.shape,
        stride=local.stride(),
        dtype=tensor.dtype,
        mesh=mesh,
        placements=placements,
        storage=storage,
    )


class _ValuePreservingApply(TorchDispatchMode):
    def __init__(self, source: Tensor, initial_state: _TensorState):
        self.dtype = source.dtype
        self.shape = source.shape
        self.placements = initial_state.placements
        self.approved: set[_TensorState] = set()
        self.storage_keys: set[tuple[int, torch.device]] = set()
        self.tracked_ids: set[int] = set()
        # Keep tracked tensors alive so Python ids cannot be recycled during fn().
        self._keepalive: list[Tensor] = []
        self._record(source, initial_state)
        self.mutated = False

    def _record(self, tensor: Tensor, state: _TensorState) -> None:
        self.approved.add(state)
        self.storage_keys.add(state.storage_key)
        self.tracked_ids.add(id(tensor))
        self._keepalive.append(tensor)

    def _aliases_tracked(self, tensor: Tensor) -> bool:
        if id(tensor) in self.tracked_ids:
            return True
        state = _tensor_state(tensor)
        return state is not None and state.storage_key in self.storage_keys

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func._schema.is_mutable:
            if any(
                isinstance(leaf, Tensor) and self._aliases_tracked(leaf)
                for leaf in tree_leaves((args, kwargs))
            ):
                self.mutated = True
            return func(*args, **kwargs)
        source_state = (
            _tensor_state(args[0])
            if func in _VALUE_PRESERVING_APPLY_OPS
            and args
            and isinstance(args[0], Tensor)
            else None
        )
        result = func(*args, **kwargs)
        if source_state is None or source_state not in self.approved:
            return result
        result_state = _tensor_state(result)
        if (
            result_state is not None
            and result_state.shape == self.shape
            and result_state.dtype == self.dtype
            and result_state.placements == self.placements
        ):
            self._record(result, result_state)
        return result

    def is_preserved(self, tensor: Tensor) -> bool:
        if self.mutated or not isinstance(tensor, Tensor):
            return False
        state = _tensor_state(tensor)
        return state is not None and state in self.approved


class _TargetWeight(nn.Module):
    """Keep a frozen Parameter for placement/sharding, excluded from draft state.

    The checkpoint loader owns target key resolution and supplies a materialized
    tensor. Constructing or restoring draft state alone never makes this module
    usable. Target tensors must be installed after meta materialization.
    FSDP trainers must ignore these modules: their parameters are deliberately
    absent from state_dict, including full-state-dict saves.
    """

    def __init__(self, vocab_size: int, hidden_size: int, *, device=None, dtype=None):
        super().__init__()
        positive_int("vocab_size", vocab_size)
        positive_int("hidden_size", hidden_size)
        if dtype is not None and not dtype.is_floating_point:
            raise TypeError("target modules require a floating-point dtype")
        self.weight = nn.Parameter(
            torch.empty(vocab_size, hidden_size, device=device, dtype=dtype),
            requires_grad=False,
        )
        self._bound = False

    def reset_draft_parameters(self):
        # Generic HF initialization must never synthesize target weights or
        # overwrite an installed target tensor.
        pass

    @torch.no_grad()
    def load_target_weight(self, source: Tensor) -> None:
        """Install an untransformed target tensor in this module's training dtype.

        Preserve Parameter identity and never mutate/alias the source. Repeated
        installation starts from the supplied tensor, so muP is applied once.
        """
        if (
            not isinstance(source, Tensor)
            or not source.is_floating_point()
            or source.layout != torch.strided
        ):
            raise TypeError("target weight must be a dense floating-point tensor")
        if source.shape != self.weight.shape:
            raise ValueError("target weight shape must match the bound module")
        if source.is_meta or self.weight.is_meta:
            raise ValueError("materialize target and draft weights before binding")
        if source.untyped_storage() is self.weight.untyped_storage():
            raise ValueError(
                "target source must not share storage with the bound weight"
            )
        self._bound = False
        self.weight.copy_(source.detach())
        self._transform_weight(self.weight)
        self.weight.requires_grad_(False)
        self._bound = True

    def _apply(self, fn, recurse=True):
        if not self._bound:
            return super()._apply(fn, recurse=recurse)
        weight = self.weight
        initial_state = _tensor_state(weight)
        self._bound = False
        if initial_state is None:
            return super()._apply(fn, recurse=recurse)
        tracker = None
        preserved = False

        def tracked(tensor):
            nonlocal tracker, preserved
            if tensor is not weight:
                return fn(tensor)
            with _ValuePreservingApply(tensor, initial_state) as tracker:
                applied = fn(tensor)
            preserved = tracker.is_preserved(applied)
            return applied

        result = super()._apply(tracked, recurse=recurse)
        if preserved and tracker.is_preserved(self.weight):
            self._bound = True
        return result

    def bound_weight(self) -> Tensor:
        """Checked access for scoring paths that do not call forward()."""
        self._require_bound()
        return self.weight

    def _transform_weight(self, value: Tensor) -> None:
        pass

    def _require_bound(self) -> None:
        if not self._bound or self.weight.is_meta:
            raise RuntimeError(
                "target weight is not bound; load it from the target before forward"
            )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # Only the canonical configuration records the binding. Target tensors
        # are restored independently and must never enter draft artifacts.
        pass

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # This leaf has no draft state. In particular, loading a draft checkpoint
        # must not accept a target tensor under an ordinary parameter key.
        unexpected_keys.extend(key for key in state_dict if key.startswith(prefix))


class BoundEmbedding(_TargetWeight):
    def forward(self, input_ids: Tensor) -> Tensor:
        return F.embedding(input_ids, self.bound_weight())


class BoundLMHead(_TargetWeight):
    def __init__(
        self,
        config: TargetLogitSpec,
        *,
        vocab_size: int,
        hidden_size: int,
        device=None,
        dtype=None,
    ):
        transform = build_component("logits", config)
        if not isinstance(transform, TargetLogitTransform):
            raise TypeError("logits builder must return TargetLogitTransform")
        if tuple(transform.parameters()) or tuple(transform.buffers()):
            raise ValueError(
                "target logit transforms must not own parameters or buffers"
            )
        super().__init__(vocab_size, hidden_size, device=device, dtype=dtype)
        self.config = config
        self.logit_transform = transform

    def _transform_weight(self, value: Tensor) -> None:
        self.logit_transform.transform_weight_(value)

    def forward(self, hidden: Tensor) -> Tensor:
        return F.linear(hidden, self.bound_weight())

    def candidate_weights(self, token_ids: Tensor) -> Tensor:
        """Gather installed, transformed rows for candidate-only logit scoring."""
        return F.embedding(token_ids, self.bound_weight())
