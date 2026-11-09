"""Visibility descriptions independent of the attention backend."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import create_mask

from specforge.config.draft._validation import boolean, positive_int

MaskMod = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]


@dataclass(frozen=True, eq=False)
class PaddingMask:
    """Head-independent visibility for padded query/key sequences.

    Pass this callable as AttentionLayout.visibility and query_valid as its
    active_queries. Unlike arbitrary mask_mod callbacks, this representation
    can be unpadded without materializing a quadratic mask.
    """

    query_valid: Tensor
    key_valid: Tensor

    def __post_init__(self):
        if any(
            t.ndim != 2 or t.dtype != torch.bool
            for t in (self.query_valid, self.key_valid)
        ):
            raise ValueError("padding masks must be boolean [batch, sequence] tensors")
        if self.query_valid.device != self.key_valid.device:
            raise ValueError("padding masks must share a device")

    def __call__(self, b, h, q, k):
        return (
            self.query_valid[b % self.query_valid.shape[0], q]
            & self.key_valid[b % self.key_valid.shape[0], k]
        )


@dataclass(frozen=True, eq=False, kw_only=True)
class AttentionLayout:
    """Logical positions and visibility, with optional explicit active queries.

    Positions are [batch, Q/K]. Visibility is either a boolean [batch, heads, Q, K]
    reference mask or mask_mod(b, h, q, k) over model-owned small tensors. Callable
    visibility requires boolean active_queries [batch, Q]; it must not close over
    mutable state. Batch/head singleton axes broadcast. Reuse these objects across
    layers, but never mutate them while the forward/backward graph is live.
    """

    query_positions: Tensor
    key_positions: Tensor
    visibility: Tensor | MaskMod
    active_queries: Tensor | None = None

    def __post_init__(self):
        q, k = self.query_positions, self.key_positions
        if any(
            p.ndim != 2 or p.dtype not in (torch.int32, torch.int64) for p in (q, k)
        ):
            raise ValueError("logical positions must be rank-two integer tensors")
        if q.device != k.device:
            raise ValueError("query/key positions must share a device")
        batches = {q.shape[0], k.shape[0]} - {1}
        if isinstance(self.visibility, Tensor):
            allowed = self.visibility
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
            batches.add(allowed.shape[0])
        elif not callable(self.visibility) or self.active_queries is None:
            raise ValueError("callable visibility requires explicit active_queries")
        if isinstance(self.visibility, PaddingMask):
            for valid, positions in (
                (self.visibility.query_valid, q),
                (self.visibility.key_valid, k),
            ):
                if valid.shape[1] != positions.shape[1] or valid.device != q.device:
                    raise ValueError(
                        "padding masks must match layout lengths and device"
                    )
                batches.add(valid.shape[0])
        if self.active_queries is not None:
            active = self.active_queries
            if (
                active.ndim != 2
                or active.dtype != torch.bool
                or active.device != q.device
                or active.shape[1] != q.shape[1]
            ):
                raise ValueError(
                    "active_queries must be boolean [batch, Q] on the positions device"
                )
            batches.add(active.shape[0])
        if len(batches - {1}) > 1:
            raise ValueError("positions and visibility must have broadcastable batches")

    def validate_dimensions(self, batch_size: int, num_heads: int) -> None:
        positive_int("batch_size", batch_size)
        positive_int("num_heads", num_heads)
        tensors = [self.query_positions, self.key_positions]
        if isinstance(self.visibility, Tensor):
            tensors.append(self.visibility)
            if self.visibility.shape[1] not in (1, num_heads):
                raise ValueError("visibility heads must broadcast to attention heads")
        if isinstance(self.visibility, PaddingMask):
            tensors.extend((self.visibility.query_valid, self.visibility.key_valid))
        if self.active_queries is not None:
            tensors.append(self.active_queries)
        if any(t.shape[0] not in (1, batch_size) for t in tensors):
            raise ValueError("layout batch must broadcast to hidden batch")

    def mask_heads(self, num_heads: int) -> int | None:
        """Share metadata across heads only when visibility guarantees it."""
        if isinstance(self.visibility, PaddingMask):
            return None
        if isinstance(self.visibility, Tensor):
            return None if self.visibility.shape[1] == 1 else num_heads
        return num_heads

    def _resolve_window(self, sliding_window: int | None) -> int | None:
        """Drop a window wider than any distance these positions can express."""
        if sliding_window is None:
            return None
        positive_int("sliding_window", sliding_window)
        dtype = torch.promote_types(
            self.query_positions.dtype, self.key_positions.dtype
        )
        # a Python int beyond the dtype would wrap in tensor comparisons
        return None if sliding_window > torch.iinfo(dtype).max else sliding_window

    def mask_mod(self, *, is_causal: bool, sliding_window: int | None) -> MaskMod:
        boolean("is_causal", is_causal)
        sliding_window = self._resolve_window(sliding_window)
        qpos, kpos, visibility, active = (
            self.query_positions,
            self.key_positions,
            self.visibility,
            self.active_queries,
        )
        qlen, klen = qpos.shape[1], kpos.shape[1]

        def mask(b, h, q, k):
            # Flex evaluates padded block indices too. Keep closure indexing safe.
            valid = (q < qlen) & (k < klen)
            q, k = q.clamp(max=qlen - 1), k.clamp(max=klen - 1)
            if isinstance(visibility, Tensor):
                allowed = visibility[
                    b % visibility.shape[0], h % visibility.shape[1], q, k
                ]
            else:
                allowed = visibility(b, h, q, k)
            if active is not None:
                allowed = allowed & active[b % active.shape[0], q]
            if is_causal or sliding_window is not None:
                qp, kp = qpos[b % qpos.shape[0], q], kpos[b % kpos.shape[0], k]
                if is_causal:
                    allowed = allowed & (qp >= kp)
                if sliding_window is not None:
                    # compare distances, not shifted positions: q +/- window can overflow
                    distance = qp - kp
                    allowed = allowed & (
                        (distance < sliding_window) & (distance > -sliding_window)
                    )
            return valid & allowed

        return mask


def dense_attention_mask(
    layout: AttentionLayout,
    *,
    is_causal: bool,
    sliding_window: int | None,
    batch_size: int | None = None,
    num_heads: int | None = None,
) -> Tensor:
    """Materialize a reference mask once, never widen the layout's visibility."""
    if not isinstance(layout, AttentionLayout):
        raise TypeError("dense attention requires AttentionLayout")
    if isinstance(layout.visibility, Tensor):
        boolean("is_causal", is_causal)
        sliding_window = layout._resolve_window(sliding_window)
        allowed = layout.visibility
        q, k = layout.query_positions.unsqueeze(-1), layout.key_positions.unsqueeze(-2)
        if is_causal:
            allowed = allowed & (q >= k).unsqueeze(1)
        if sliding_window is not None:
            distance = q - k
            allowed = allowed & (
                (distance < sliding_window) & (distance > -sliding_window)
            ).unsqueeze(1)
        if layout.active_queries is not None:
            allowed = allowed & layout.active_queries[:, None, :, None]
        return allowed
    if batch_size is None or num_heads is None:
        raise ValueError(
            "callable visibility needs batch_size and num_heads for dense preparation"
        )
    layout.validate_dimensions(batch_size, num_heads)
    return create_mask(
        layout.mask_mod(is_causal=is_causal, sliding_window=sliding_window),
        batch_size,
        layout.mask_heads(num_heads),
        layout.query_positions.shape[1],
        layout.key_positions.shape[1],
        device=layout.query_positions.device,
    )


def prepare_attention_mask(
    layout, *, is_causal, sliding_window, backend, batch_size=None, num_heads=None
):
    """Keep visibility symbolic for FlexAttention; materialize only for dense kernels."""
    if backend != "flex_attention":
        return dense_attention_mask(
            layout,
            is_causal=is_causal,
            sliding_window=sliding_window,
            batch_size=batch_size,
            num_heads=num_heads,
        )
    if not isinstance(layout, AttentionLayout):
        raise TypeError("attention requires AttentionLayout")
    boolean("is_causal", is_causal)
    if sliding_window is not None:
        positive_int("sliding_window", sliding_window)
    from torch.nn.attention.flex_attention import create_block_mask

    from specforge.modeling.draft.layers.attention.backends.policy import (
        flex_attention_backend,
    )

    q, k = layout.query_positions, layout.key_positions
    visibility = layout.visibility

    def allowed(b, h, qi, ki):
        within = (qi < q.shape[1]) & (ki < k.shape[1])
        qi, ki = qi.clamp(max=q.shape[1] - 1), ki.clamp(max=k.shape[1] - 1)
        if callable(visibility):
            result = visibility(b, h, qi, ki)
        else:
            result = visibility[
                b % visibility.shape[0], h % visibility.shape[1], qi, ki
            ]
        qp, kp = q[b % q.shape[0], qi], k[b % k.shape[0], ki]
        if is_causal:
            result = result & (qp >= kp)
        if sliding_window is not None:
            result = result & ((qp - kp).abs() < sliding_window)
        return within & result

    return create_block_mask(
        allowed,
        B=q.shape[0],
        H=(
            None
            if callable(visibility) or visibility.shape[1] == 1
            else visibility.shape[1]
        ),
        Q_LEN=q.shape[1],
        KV_LEN=k.shape[1],
        device=q.device,
        BLOCK_SIZE=(256, 128) if flex_attention_backend() == "FLASH" else 128,
    )
