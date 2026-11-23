"""Bounded startup estimate for radix reuse in online text capture."""

from __future__ import annotations

import logging
import random
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from itertools import takewhile

from .adapters.server_capture import _get_server_info

logger = logging.getLogger(__name__)

_SAMPLE_COUNT = 512
_HISTORY_TOKENS = 64 * 1024
_MIN_REUSE_FRACTION = 0.05
_SERVER_INFO_TIMEOUT_S = 5.0


def _server_geometry(url):
    try:
        info = _get_server_info(
            f"{url.rstrip('/')}/server_info", _SERVER_INFO_TIMEOUT_S
        )
    except Exception as exc:
        logger.warning("Prefix cache auto could not read %s/server_info: %s", url, exc)
        return 0, 1
    states = info.get("internal_states") if isinstance(info, Mapping) else None
    if not isinstance(states, list) or not states:
        return 0, 1
    sizes = [
        (
            state.get("spec_capture_prefix_page_size")
            if isinstance(state, Mapping)
            else None
        )
        for state in states
    ]
    page_size = sizes[0]
    if (
        type(page_size) is not int
        or page_size < 1
        or any(size != page_size for size in sizes)
    ):
        return 0, len(states)
    return page_size, len(states)


def auto_prefix_cache_modes(
    prompts: Sequence[dict], server_urls: Sequence[str], *, lease: int, seed: int
) -> list[bool]:
    """Estimate reusable tokens per endpoint, before starting the rollout pool.

    Sample without replacement to approximate the shuffled prompt stream.
    Leases rotate across endpoints in the estimate and requests rotate across
    each endpoint's schedulers. Each scheduler has a bounded FIFO token history.
    This estimates locality without changing live ordering or routing.
    """
    geometry = [_server_geometry(url) for url in server_urls]
    totals = [0] * len(server_urls)
    reused = totals.copy()
    if any(page_size for page_size, _ in geometry):
        indices = random.Random(seed).sample(
            range(len(prompts)), min(_SAMPLE_COUNT, len(prompts))
        )
        history = defaultdict(deque)
        tokens_in_history = defaultdict(int)
        for i, index in enumerate(indices):
            round_index, endpoint = divmod(i // lease, len(server_urls))
            page_size, replicas = geometry[endpoint]
            if not page_size:
                continue
            ids = prompts[index]["payload"]["input_ids"]
            totals[endpoint] += len(ids)
            # Prefill keeps the last token outside a radix hit.
            page_count = max(len(ids) - 1, 0) // page_size
            if not page_count:
                continue
            pages = tuple(
                tuple(ids[start : start + page_size])
                for start in range(0, page_count * page_size, page_size)
            )
            route = (endpoint, (round_index * lease + i % lease) % replicas)
            previous = history[route]
            shared_pages = max(
                (
                    sum(
                        1
                        for _ in takewhile(
                            lambda pair: pair[0] == pair[1], zip(pages, other)
                        )
                    )
                    for other in previous
                ),
                default=0,
            )
            reused[endpoint] += shared_pages * page_size
            previous.append(pages)
            tokens_in_history[route] += page_count * page_size
            while tokens_in_history[route] > _HISTORY_TOKENS:
                tokens_in_history[route] -= len(previous.popleft()) * page_size

    flags = []
    for url, (page_size, replicas), tokens, hit_tokens in zip(
        server_urls, geometry, totals, reused, strict=True
    ):
        fraction = hit_tokens / tokens if tokens else 0.0
        enabled = fraction >= _MIN_REUSE_FRACTION
        flags.append(enabled)
        logger.info(
            "Prefix cache auto for %s: %s (page size %d, schedulers %d, "
            "estimated reuse %.2f%%, sampled tokens %d)",
            url,
            "enabled" if enabled else "disabled",
            page_size,
            replicas,
            fraction * 100,
            tokens,
        )
    return flags
