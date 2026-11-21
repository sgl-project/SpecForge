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
class ContextBlocks:
    """Independent query blocks, each reading a prefix of the target context.

    Queries are ``context_lengths.shape[1]`` consecutive blocks of ``block_size``
    slots; keys are the context rows followed by those queries. Block ``n`` of
    row ``b`` sees the context rows before ``context_lengths[b, n]`` that are
    ``context_valid``, and every slot of its own block. Inactive blocks see
    nothing. Lengths lie in ``[0, context rows]``.
    """

    block_size: int
    context_lengths: Tensor
    context_valid: Tensor
    active_blocks: Tensor

    def __post_init__(self):
        positive_int("block_size", self.block_size)
        lengths, valid, active = (
            self.context_lengths,
            self.context_valid,
            self.active_blocks,
        )
        if lengths.ndim != 2 or lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("context_lengths must be integer [batch, blocks]")
        if valid.ndim != 2 or valid.dtype != torch.bool:
            raise ValueError("context_valid must be boolean [batch, context]")
        if active.dtype != torch.bool or active.shape != lengths.shape:
            raise ValueError("active_blocks must be boolean [batch, blocks]")
        if valid.shape[0] not in (1, lengths.shape[0]):
            raise ValueError("context_valid batch must broadcast to the blocks")
        if not lengths.device == valid.device == active.device:
            raise ValueError("context blocks must share a device")

    @property
    def num_context_rows(self) -> int:
        return self.context_valid.shape[1]

    @property
    def num_queries(self) -> int:
        return self.context_lengths.shape[1] * self.block_size

    def compute_active_queries(self) -> Tensor:
        return self.active_blocks.repeat_interleave(self.block_size, dim=1)

    def compute_visibility(self) -> Tensor:
        """Boolean [batch, 1, queries, context rows + queries] mask."""
        batch, count = self.context_lengths.shape
        device = self.context_lengths.device
        rows = torch.arange(self.num_context_rows, device=device)
        context_visible = (rows < self.context_lengths[..., None]) & self.context_valid[
            :, None, :
        ]
        context_visible = context_visible.repeat_interleave(self.block_size, dim=1)
        block_ids = torch.arange(count, device=device).repeat_interleave(
            self.block_size
        )
        draft_visible = (block_ids[:, None] == block_ids[None, :]).expand(batch, -1, -1)
        visible = torch.cat((context_visible, draft_visible), dim=-1)
        visible &= self.compute_active_queries()[..., None]
        return visible[:, None]


@dataclass(frozen=True, eq=False, kw_only=True)
class AttentionLayout:
    """Logical positions and visibility, with optional explicit active queries.

    Positions are [batch, Q/K]. Visibility is either a boolean [batch, heads, Q, K]
    reference mask or mask_mod(b, h, q, k) over model-owned small tensors. Callable
    visibility requires boolean active_queries [batch, Q]; it must not close over
    mutable state. Batch/head singleton axes broadcast. Reuse these objects across
    layers, but never mutate them while the forward/backward graph is live.

    Context-then-block layouts pass ``context_blocks`` instead; it determines
    visibility and active queries, and stays readable for components such as
    recurrent attention that consume the block structure itself.
    """

    query_positions: Tensor
    key_positions: Tensor
    visibility: Tensor | MaskMod | None = None
    active_queries: Tensor | None = None
    context_blocks: ContextBlocks | None = None

    def __post_init__(self):
        q, k = self.query_positions, self.key_positions
        if any(
            p.ndim != 2 or p.dtype not in (torch.int32, torch.int64) for p in (q, k)
        ):
            raise ValueError("logical positions must be rank-two integer tensors")
        if q.device != k.device:
            raise ValueError("query/key positions must share a device")
        blocks = self.context_blocks
        if blocks is not None:
            if not isinstance(blocks, ContextBlocks):
                raise TypeError("context_blocks must be ContextBlocks")
            if self.visibility is not None or self.active_queries is not None:
                raise ValueError(
                    "context_blocks determine visibility and active_queries"
                )
            if (
                q.shape[1] != blocks.num_queries
                or k.shape[1] != blocks.num_context_rows + blocks.num_queries
                or blocks.context_lengths.device != q.device
            ):
                raise ValueError(
                    "context_blocks must describe the context-then-query positions"
                )
            object.__setattr__(self, "visibility", blocks.compute_visibility())
            object.__setattr__(self, "active_queries", blocks.compute_active_queries())
        elif self.visibility is None:
            raise ValueError("a layout requires visibility or context_blocks")
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
            valid_q, valid_k = q < qlen, k < klen
            q, k = torch.where(valid_q, q, 0), torch.where(valid_k, k, 0)
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
            return valid_q & valid_k & allowed

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
