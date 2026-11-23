# Copyright (c) 2026 SGLang Project
# SPDX-License-Identifier: Apache-2.0

"""Hidden rows indexed by radix-owned KV pages, with native KV eviction."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, replace

import torch


@dataclass
class PendingHidden:
    start: int
    end: int
    bytes_per_page: int
    tensors: tuple[torch.Tensor, ...] = ()
    ready: torch.cuda.Event | None = None


class HiddenPrefixCache:
    """Reserve hidden slots only for pages the radix cache publishes.

    Every attention TP rank keeps the same admission metadata. Only the
    capture writer allocates hidden tensors.
    """

    def __init__(
        self, max_bytes: int, page_size: int, *, is_writer: bool, max_tokens=None
    ):
        self.max_bytes = max_bytes
        self.page_size = page_size
        self.is_writer = is_writer
        self.max_pages = max_tokens // page_size if max_tokens is not None else None
        self.bytes_per_page = None
        self.slots: dict[int, int] = {}
        self.free_slots: list[int] = []
        self.next_slot = 0
        self.buffers: tuple[torch.Tensor, ...] = ()

    def capacity(self, bytes_per_page):
        pages = self.max_bytes // bytes_per_page
        return pages if self.max_pages is None else min(pages, self.max_pages)

    def page_heads(self, indices):
        return indices[:: self.page_size].tolist()

    def stage(self, req, start, end, tensors, *, copy):
        end = (end - 1) // self.page_size * self.page_size
        if end <= start or start % self.page_size:
            return
        page_bytes = self.page_size * sum(
            t[0].numel() * t.element_size() for t in tensors
        )
        if not self.capacity(page_bytes) or (
            self.bytes_per_page is not None and page_bytes != self.bytes_per_page
        ):
            return
        self.bytes_per_page = page_bytes
        data = PendingHidden(start, end, page_bytes)
        if self.is_writer:
            rows = tuple(t[: end - start].detach() for t in tensors)
            data.tensors = tuple(t.clone() for t in rows) if copy else rows
            if rows[0].is_cuda:
                data.ready = torch.cuda.Event()
                data.ready.record()
        req._spec_capture_pending_hidden = data

    def covered_pages(self, heads):
        for i, head in enumerate(heads):
            if head not in self.slots:
                return i
        return len(heads)

    def insert(self, heads, pending):
        positions, slots = [], []
        for i, head in enumerate(heads):
            if head in self.slots:
                continue
            if self.free_slots:
                slot = heapq.heappop(self.free_slots)
            else:
                slot = self.next_slot
                self.next_slot += 1
            self.slots[head] = slot
            positions.append(i)
            slots.append(slot)
        if not slots or not self.is_writer:
            return

        if not self.buffers:
            capacity = self.capacity(pending.bytes_per_page)
            self.buffers = tuple(
                t.new_empty((capacity, self.page_size, *t.shape[1:]))
                for t in pending.tensors
            )
        device = self.buffers[0].device
        if pending.ready is not None:
            stream = torch.cuda.current_stream(device)
            stream.wait_event(pending.ready)
            for t in pending.tensors:
                t.record_stream(stream)
        count = len(slots)
        if slots == list(range(slots[0], slots[0] + count)) and positions == list(
            range(positions[0], positions[0] + count)
        ):
            start = positions[0] * self.page_size
            end = start + count * self.page_size
            for buffer, rows in zip(self.buffers, pending.tensors, strict=True):
                buffer[slots[0] : slots[0] + count].flatten(0, 1).copy_(rows[start:end])
            return
        dst = torch.tensor(slots, device=device)
        src = torch.tensor(positions, device=device)
        for buffer, rows in zip(self.buffers, pending.tensors, strict=True):
            pages = rows.unflatten(0, (-1, self.page_size))
            buffer.index_copy_(0, dst, pages.index_select(0, src))

    def read(self, req, length):
        heads = getattr(req, "_spec_capture_kv_page_heads", ())
        count = length // self.page_size
        if (
            not count
            or length % self.page_size
            or len(heads) < count
            or not self.buffers
        ):
            return None
        slots = [self.slots.get(head) for head in heads[:count]]
        if None in slots:
            return None
        if slots == list(range(slots[0], slots[0] + count)):
            return tuple(
                t[slots[0] : slots[0] + count].flatten(0, 1) for t in self.buffers
            )
        index = torch.tensor(slots, device=self.buffers[0].device)
        return tuple(t.index_select(0, index).flatten(0, 1) for t in self.buffers)

    def release(self, indices):
        if self.slots:
            for head in self.page_heads(indices):
                slot = self.slots.pop(head, None)
                if slot is not None:
                    heapq.heappush(self.free_slots, slot)

    def clear(self):
        self.slots.clear()
        self.free_slots.clear()
        self.next_slot = 0
        self.buffers = ()


_UNIFIED_RADIX_CACHE = "sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache"


def register_radix_hooks(registry, hook_type):
    registry.register(
        f"{_UNIFIED_RADIX_CACHE}.match_prefix", match_prefix, hook_type.AROUND
    )
    registry.register(
        f"{_UNIFIED_RADIX_CACHE}.checkpoint_into_tree", checkpoint, hook_type.AROUND
    )
    registry.register(
        f"{_UNIFIED_RADIX_CACHE}._apply_cache_action", release_pages, hook_type.BEFORE
    )
    registry.register(f"{_UNIFIED_RADIX_CACHE}._reset_full", reset, hook_type.BEFORE)
    registry.register(
        f"{_UNIFIED_RADIX_CACHE}.on_release", release_request, hook_type.BEFORE
    )


def _wants_prefix(req):
    from .capture import prefix_eligible, request_spec

    if req is None or not req.extra_key:
        return False
    spec = request_spec(req)
    return spec is not None and prefix_eligible(req, spec)


def match_prefix(original, tree, params):
    req = params.req
    if not _wants_prefix(req):
        return original(tree, params)

    req._spec_capture_kv_page_heads = ()
    hidden = getattr(tree, "_spec_capture_hidden", None)
    if hidden is None or not hidden.slots:
        return original(tree, replace(params, key=params.key[:0]))

    match = original(tree, params)
    if not match.device_prefix_len:
        return match
    heads = hidden.page_heads(
        tree.path_device_indices(match.last_device_node)[: match.device_prefix_len]
    )
    covered = hidden.covered_pages(heads)
    if covered != len(heads):
        # Run the native match with the shorter key so its locks and sliding
        # window checks are anchored to the same prefix the capture can use.
        match = original(
            tree, replace(params, key=params.key[: covered * hidden.page_size])
        )
        heads = heads[: match.device_prefix_len // hidden.page_size]
    req._spec_capture_kv_page_heads = heads
    return match


def checkpoint(original, tree, req, *, up_to, **kwargs):
    if not _wants_prefix(req):
        return original(tree, req, up_to=up_to, **kwargs)

    hidden = getattr(tree, "_spec_capture_hidden", None)
    pending = getattr(req, "_spec_capture_pending_hidden", None)
    req._spec_capture_pending_hidden = None
    if hidden is None or pending is None or not req.finished():
        return original(tree, req, up_to=0, **kwargs)

    from sglang.srt.mem_cache.base_prefix_cache import EvictParams

    page_size = hidden.page_size
    capacity = hidden.capacity(pending.bytes_per_page)
    protected = req.kv.cache_protected_len
    if protected != pending.start:
        return original(tree, req, up_to=0, **kwargs)
    requested = (min(up_to, pending.end) - protected) // page_size
    requested = min(requested, capacity - protected // page_size)
    if requested > capacity - len(hidden.slots):
        # The native walk respects active KV locks. Stop as soon as its free
        # actions yield enough hidden slots, even if it also evicts KV-only pages.
        tree.evict_for_alloc(
            EvictParams(num_tokens=tree.full_evictable_size()),
            allocation_reclaim_satisfied=lambda: requested
            <= capacity - len(hidden.slots),
        )

    admitted = min(requested, capacity - len(hidden.slots))
    if admitted <= 0:
        return original(tree, req, up_to=0, **kwargs)
    limit = protected + admitted * page_size
    result = original(tree, req, up_to=limit, **kwargs)
    published = min(req.kv.cache_protected_len, limit)
    if published > protected:
        indices = tree.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, protected:published
        ]
        hidden.insert(hidden.page_heads(indices), pending)
    return result


def release_pages(tree, action):
    hidden = getattr(tree, "_spec_capture_hidden", None)
    if hidden is None or not hidden.slots:
        return
    from sglang.srt.mem_cache.unified_cache.cache_action import (
        FreeComponentDeviceSlot,
        FreeDeviceKV,
        FreeDeviceKVFullOnly,
    )

    if isinstance(action, (FreeDeviceKV, FreeDeviceKVFullOnly)) or (
        isinstance(action, FreeComponentDeviceSlot) and action.component_type.is_full
    ):
        for indices in action.indices:
            hidden.release(indices)


def reset(tree):
    hidden = getattr(tree, "_spec_capture_hidden", None)
    if hidden is not None:
        hidden.clear()


def release_request(tree, req, *, checkpointed):
    if hasattr(req, "_spec_capture_pending_hidden"):
        req._spec_capture_pending_hidden = None
    if hasattr(req, "_spec_capture_kv_page_heads"):
        req._spec_capture_kv_page_heads = ()
