"""Dense visibility supplied by models, narrowed by attention-node semantics."""

from dataclasses import dataclass

import torch
from torch import Tensor

from specforge.config.draft._validation import boolean, positive_int


@dataclass(frozen=True, eq=False, kw_only=True)
class AttentionLayout:
    """Logical positions [batch, Q/K] and allowed pairs [batch, heads, Q, K].

    Visibility defines padding, block isolation, and context access; it must be
    explicit even for bidirectional attention. Its batch/head axes may broadcast.
    Logical positions determine causality/window distance, independently of RoPE
    coordinates (which may have multiple axes). Do not mutate these tensors while
    their autograd graph is live. Visibility currently uses a boolean tensor;
    backend mask conversion is separate from the layout.
    """

    query_positions: Tensor
    key_positions: Tensor
    visibility: Tensor

    def __post_init__(self):
        q, k, allowed = self.query_positions, self.key_positions, self.visibility
        if any(
            p.ndim != 2 or p.dtype not in (torch.int32, torch.int64) for p in (q, k)
        ):
            raise ValueError("logical positions must be rank-two integer tensors")
        if q.device != k.device:
            raise ValueError("query/key positions must share a device")
        if (
            allowed.ndim != 4
            or allowed.dtype != torch.bool
            or allowed.device != q.device
            or allowed.shape[1] < 1
            or allowed.shape[2:] != (q.shape[1], k.shape[1])
        ):
            raise ValueError(
                "visibility must be boolean [batch, heads, Q, K] on the positions device"
            )
        batches = {value.shape[0] for value in (q, k, allowed)} - {1}
        if len(batches) > 1:
            raise ValueError("positions and visibility must have broadcastable batches")


def dense_attention_mask(
    layout: AttentionLayout, *, is_causal: bool, sliding_window: int | None
) -> Tensor:
    """Intersect visibility with causal/window constraints; never expand access.

    A window W permits distances strictly smaller than W in either direction.
    Causality additionally removes future keys. Thus W=1 permits equal positions
    only, and still respects the model's original isolation/padding constraints.
    """
    if not isinstance(layout, AttentionLayout):
        raise TypeError("dense attention requires AttentionLayout")
    boolean("is_causal", is_causal)
    if sliding_window is not None:
        positive_int("sliding_window", sliding_window)
    allowed = layout.visibility
    if is_causal or sliding_window is not None:
        distance = layout.query_positions.unsqueeze(
            -1
        ) - layout.key_positions.unsqueeze(-2)
        if is_causal:
            allowed = allowed & (distance >= 0).unsqueeze(1)
        if sliding_window is not None:
            allowed = allowed & (distance.abs() < sliding_window).unsqueeze(1)
    return allowed
