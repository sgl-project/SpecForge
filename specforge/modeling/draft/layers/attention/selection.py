"""Logical sparse access plans; no indexer, KV expansion or mask allocation."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class IndexedSelection:
    """Token or fixed-size block indices in one named attention KV source.

    Indices have shape [B, Hs, Q, K]; lengths [B, Hs, Q] count valid leading
    slots. Hs=1 shares selection across heads. Other head mappings require
    explicit backend support. block_size=1 selects tokens; otherwise index i
    selects logical tokens [i * block_size, (i + 1) * block_size).

    Coordinates are source-local logical KV indices, not RoPE positions or
    physical cache slots. Source adapters own that mapping, sequence lengths
    and per-token visibility (including partial blocks). QSA retrieval keys
    must be mapped to attention KV before constructing this selection.

    Producers must provide in-range, unique valid indices and lengths in
    [0, K]. Unused slots may contain any integer and must never be gathered.
    Only metadata is checked here, avoiding device synchronization in forward.
    """

    indices: torch.Tensor
    lengths: torch.Tensor
    source: str = "main"
    block_size: int = 1

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("selection source must be nonempty")
        if (
            isinstance(self.block_size, bool)
            or not isinstance(self.block_size, int)
            or self.block_size <= 0
        ):
            raise ValueError("block_size must be a positive integer")
        if self.indices.ndim != 4 or self.lengths.shape != self.indices.shape[:-1]:
            raise ValueError("indices must be [B, Hs, Q, K] and lengths [B, Hs, Q]")
        if self.indices.dtype not in (
            torch.int32,
            torch.int64,
        ) or self.lengths.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError("indices and lengths must be integer tensors")
        if self.indices.device != self.lengths.device:
            raise ValueError("indices and lengths must be on the same device")


@dataclass(frozen=True)
class SlidingWindowSelection:
    """Select source keys with abs(query_position - key_position) < size.

    Positions are logical token positions, not cache offsets. Global causality
    further restricts this to the current and preceding size-1 positions.
    The source adapter supplies positions and excludes padding/other segments.
    """

    size: int
    source: str = "main"

    def __post_init__(self) -> None:
        if (
            isinstance(self.size, bool)
            or not isinstance(self.size, int)
            or self.size <= 0
        ):
            raise ValueError("window size must be a positive integer")
        if not self.source:
            raise ValueError("window source must be nonempty")


@dataclass(frozen=True)
class SparseSelection:
    """Union of indexed keys and local windows, then intersect visibility.

    An empty plan selects nothing, never full attention. Entries sharing a
    source are a set union: overlapping blocks, tokens and windows count once.
    Different sources identify distinct KV entries (e.g. raw and compressed).
    Their scores share one normalization; summing separately normalized branch
    outputs is not equivalent. Backend support for multiple sources is explicit.

    Context masks, causal rules and AttentionConfig.sliding_window are global
    restrictions applied to the union. For local SWA + distant selected keys,
    put the local window here and leave the global sliding_window unset.
    Indexers must apply visibility before top-k so invalid keys do not consume
    selection budget. A selected block never overrides per-token visibility.
    """

    indexed: tuple[IndexedSelection, ...] = ()
    windows: tuple[SlidingWindowSelection, ...] = ()
