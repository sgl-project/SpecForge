"""Frozen target token modules; tensor installation only, without file I/O."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from specforge.config.draft._validation import positive_int
from specforge.config.draft.logits import TargetLogitSpec

from .layers.logits import TargetLogitTransform
from .layers.registry import build_component


class _TargetWeight(nn.Module):
    """Keep a frozen Parameter for placement/sharding, excluded from draft state.

    The checkpoint loader owns target key resolution and supplies a materialized
    tensor. Constructing or restoring draft state alone never makes this module
    usable. Target tensors must be installed after meta materialization.
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
        value = source.detach().to(
            device=self.weight.device, dtype=self.weight.dtype, copy=True
        )
        self._transform_weight(value)
        self.weight.copy_(value)
        self.weight.requires_grad_(False)
        self._bound = True

    def _apply(self, fn, recurse=True):
        weight = self.weight
        dtype, storage = weight.dtype, weight.untyped_storage()
        result = super()._apply(fn, recurse=recurse)
        if self.weight.dtype != dtype or self.weight.untyped_storage() is not storage:
            self._bound = False
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
