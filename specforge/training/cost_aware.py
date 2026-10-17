# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Resolve ``training.cost_aware_*`` into a :class:`SchedulingPolicy`."""

from __future__ import annotations

from typing import Optional

from specforge.runtime.data_plane.window_plan import SchedulingPolicy


def resolve_scheduling_policy(training, algorithm) -> Optional[SchedulingPolicy]:
    """Policy for one run, or None when ``training.cost_aware_scheduling`` is off.

    ``auto`` takes the key the algorithm declares in its capabilities; the
    application layer has already rejected algorithms without one.
    """
    if not getattr(training, "cost_aware_scheduling", False):
        return None
    key = training.cost_aware_key
    if key == "auto":
        key = algorithm.spec.capabilities.scheduling_key
        if key is None:
            raise ValueError(
                f"algorithm {algorithm.name!r} declares no cost-aware "
                "scheduling key; set training.cost_aware_key explicitly"
            )
    return SchedulingPolicy(
        key=key,
        num_anchors=int(training.num_anchors),
        window_rounds=int(training.cost_aware_window),
    )


__all__ = ["resolve_scheduling_policy"]
