# coding=utf-8
"""Package-owned entry points for the production windowed fan-out launcher.

The general CLI continues to expose only canonical SpecForge topologies. This
module owns the specialized multi-consumer composition used by
``scripts/run_dflash_fanout.py`` without expanding ``specforge.launch``'s public
surface.
"""

from __future__ import annotations


def build_windowed_capture_contract(**kwargs):
    from specforge.launch import build_disagg_windowed_capture_contract

    return build_disagg_windowed_capture_contract(**kwargs)


def build_windowed_producer(**kwargs):
    from specforge.launch import build_disagg_online_windowed_producer

    return build_disagg_online_windowed_producer(**kwargs)


def build_windowed_consumer(**kwargs):
    from specforge.launch import build_disagg_online_windowed_consumer

    return build_disagg_online_windowed_consumer(**kwargs)


__all__ = [
    "build_windowed_capture_contract",
    "build_windowed_consumer",
    "build_windowed_producer",
]
