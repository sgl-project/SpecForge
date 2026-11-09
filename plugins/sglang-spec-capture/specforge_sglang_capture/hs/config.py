"""Explicit native-plugin HS configuration, independent of legacy SG flags."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _flag(name: str) -> bool:
    value = os.environ.get(name, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


@dataclass(frozen=True)
class HSConfig:
    capture: bool = False
    cache: bool = False
    chunk_tokens: int = 256
    cache_bytes: int = 1 << 30
    pinned_bytes: int = 256 << 20
    output_bytes: int = 512 << 20
    device_bytes: int = 256 << 20
    max_workers: int = 4
    attempt_capacity: int = 4096

    @classmethod
    def from_env(cls) -> HSConfig:
        values = {"capture": _flag("SPECFORGE_HS_CAPTURE"),
                  "cache": _flag("SPECFORGE_HS_CACHE")}
        for name in ("chunk_tokens", "cache_bytes", "pinned_bytes", "output_bytes",
                     "device_bytes", "max_workers", "attempt_capacity"):
            env = "SPECFORGE_HS_" + name.upper()
            value = int(os.environ.get(env, str(getattr(cls, name))))
            if value <= 0:
                raise ValueError(f"{env} must be positive")
            values[name] = value
        if values["cache"] and not values["capture"]:
            raise ValueError("SPECFORGE_HS_CACHE=1 requires SPECFORGE_HS_CAPTURE=1")
        return cls(**values)
