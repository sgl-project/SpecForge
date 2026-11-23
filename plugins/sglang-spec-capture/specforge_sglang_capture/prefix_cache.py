# Copyright (c) 2026 SGLang Project
# SPDX-License-Identifier: Apache-2.0

"""Bounded device cache for the hidden rows omitted by a radix-cache hit."""

from collections import OrderedDict


class HiddenPrefixCache:
    """Keep exact text prefixes scoped to an isolated capture namespace."""

    def __init__(self, max_bytes: int, page_size: int = 1):
        if max_bytes < 0:
            raise ValueError("prefix cache byte budget must be nonnegative")
        if page_size < 1:
            raise ValueError("radix page size must be positive")
        self.max_bytes = max_bytes
        self.page_size = page_size
        self.used_bytes = 0
        self.entries = OrderedDict()

    def get(self, namespace, tokens):
        for key in reversed(self.entries):
            owner, cached_tokens = key
            if owner == namespace and tokens == cached_tokens[: len(tokens)]:
                tensors, _ = self.entries[key]
                self.entries.move_to_end(key)
                return tuple(tensor[: len(tokens)] for tensor in tensors)
        return None

    def put(self, namespace, tokens, tensors):
        if len(tokens) <= self.page_size:
            return
        # Retain at most one eighth of the budget per entry. A single long
        # prompt can still contribute reusable pages. Radix lookup aligns
        # down after excluding the final prompt token.
        bytes_per_token = sum(
            tensor[0].numel() * tensor.element_size() for tensor in tensors
        )
        length = min(len(tokens) - 1, self.max_bytes // (8 * bytes_per_token))
        length -= length % self.page_size
        if length <= 0:
            return
        key = namespace, tuple(tokens[:length])
        if key in self.entries:
            self.entries.move_to_end(key)
            return
        nbytes = length * bytes_per_token
        while self.used_bytes + nbytes > self.max_bytes:
            _, (_, size) = self.entries.popitem(last=False)
            self.used_bytes -= size
        self.entries[key] = (
            tuple(tensor[:length].detach().clone() for tensor in tensors),
            nbytes,
        )
        self.used_bytes += nbytes

    def clear(self):
        self.entries.clear()
        self.used_bytes = 0
