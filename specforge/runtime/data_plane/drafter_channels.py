# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""DrafterChannels: one ref channel per run behind the interface of one channel.

An online producer captures each prompt once and publishes one ref per run.
The drive loop keeps talking to a single channel; this class spreads its calls
over every run's :class:`StreamingRefChannel`::

    drive loop -> DrafterChannels -> run A: refs.jsonl -> trainer A
                                  -> run B: refs.jsonl -> trainer B
"""

from __future__ import annotations

from typing import Iterator, Mapping, Optional, Sequence, Tuple

from specforge.runtime.contracts import SampleRef
from specforge.runtime.data_plane.streaming_ref_channel import (
    RefPublishTransaction,
    StreamingRefChannel,
)


class _DrafterChannelsPublishTransaction:
    """Publish each channel's refs and expose everything not handed over."""

    def __init__(self, transactions: Sequence[RefPublishTransaction]) -> None:
        self._transactions = transactions

    @property
    def unpublished_refs(self) -> Tuple[SampleRef, ...]:
        return tuple(
            ref
            for transaction in self._transactions
            for ref in transaction.unpublished_refs
        )

    def commit(self) -> None:
        for transaction in self._transactions:
            transaction.commit()


class DrafterChannels:
    """The ref channels of every run, keyed by ``run_id``."""

    def __init__(self, channels: Mapping[str, StreamingRefChannel]) -> None:
        self._channels = dict(channels)

    def consumer_quantum(self) -> Optional[int]:
        """The largest dispatch quantum, once every trainer has published one."""
        quanta = [channel.consumer_quantum() for channel in self._channels.values()]
        return None if None in quanta else max(quanta)

    def consumer_failure(self) -> Optional[str]:
        for channel in self._channels.values():
            failure = channel.consumer_failure()
            if failure is not None:
                return failure
        return None

    def consumer_stopped(self) -> bool:
        """True when every trainer stopped, or one of them failed."""
        return self.consumer_failure() is not None or all(
            channel.consumer_stopped() for channel in self._channels.values()
        )

    def in_flight_remote(self) -> int:
        """The most unread refs any running trainer has.

        Every run gets every prompt, so the fullest channel belongs to the
        slowest trainer, and pausing on it holds no other trainer back. Example:
        after 30 prompts with A at 20 read and B at 27, this is 10, not 3. Once
        every trainer has stopped, every channel counts.
        """
        channels = list(self._channels.values())
        running = [channel for channel in channels if not channel.consumer_stopped()]
        if not running:
            running = channels
        return max(channel.in_flight_remote() for channel in running)

    def consumed_state_remote(self) -> Tuple[int, Optional[int]]:
        """Refs read by all trainers, and the bytes they freed.

        The bytes are ``None`` until every trainer reports them.
        """
        states = [
            channel.consumed_state_remote() for channel in self._channels.values()
        ]
        released = [num_bytes for _, num_bytes in states]
        return (
            sum(count for count, _ in states),
            None if None in released else sum(released),
        )

    def consumed_remote(self) -> int:
        return self.consumed_state_remote()[0]

    def begin_publish(
        self, refs: Sequence[SampleRef]
    ) -> _DrafterChannelsPublishTransaction:
        """One transaction per channel, each with its run's refs in capture order.

        Example: ``[a:t0, b:t0, a:t1, b:t1]`` gives ``a`` ``[a:t0, a:t1]`` and
        ``b`` ``[b:t0, b:t1]``; ``a`` commits first.
        """
        by_run = {run_id: [] for run_id in self._channels}
        for ref in refs:
            by_run[ref.run_id].append(ref)
        return _DrafterChannelsPublishTransaction(
            [
                self._channels[run_id].begin_publish(run_refs)
                for run_id, run_refs in by_run.items()
            ]
        )

    def fail(self, reason: str) -> None:
        for channel in self._channels.values():
            channel.fail(reason)

    def close(self) -> None:
        """Tell every trainer its data is complete, unless one of them failed."""
        failure = self.consumer_failure()
        if failure is not None:
            raise RuntimeError(f"consumer failed: {failure}")
        for channel in self._channels.values():
            channel.close()

    def published_refs(self) -> Iterator[SampleRef]:
        """Every ref published so far, read back from each channel's file."""
        for channel in self._channels.values():
            reader = StreamingRefChannel(channel.path)
            while True:
                refs = reader.poll(max_n=1024)
                if not refs:
                    break
                yield from refs


__all__ = ["DrafterChannels"]
