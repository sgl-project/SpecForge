# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""SGLang plugin: SpecForge server-side spec capture into Mooncake.

Installed into the SGLang environment, SGLang's ``load_plugins()`` calls
:func:`register` in every process (entry point group ``sglang.srt.plugins``).
Capture is enabled only when ``SPECFORGE_SPEC_CAPTURE=1``; the server must also
run with ``--aux-hidden-state-capture``, ``--return-hidden-states-mode full``
and ``--chunked-prefill-size -1`` (checked when the scheduler starts).
"""

from __future__ import annotations

import os

ENABLE_ENV = "SPECFORGE_SPEC_CAPTURE"

_SCHEDULER = "sglang.srt.managers.scheduler.Scheduler"


def enabled() -> bool:
    return os.environ.get(ENABLE_ENV) == "1"


def register() -> None:
    """Register the scheduler hooks; a no-op unless capture is enabled."""
    if not enabled():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        f"{_SCHEDULER}.get_output_streamer_class",
        _capture_streamer_class,
        HookType.AFTER,
    )
    HookRegistry.register(f"{_SCHEDULER}.__init__", _install, HookType.AFTER)


def _capture_streamer_class(streamer_class, scheduler, *args, **kwargs):
    from specforge_sglang_capture.capture import capture_streamer_class

    return capture_streamer_class(streamer_class)


def _install(result, scheduler, *args, **kwargs):
    from specforge_sglang_capture.capture import install

    # Raising here fails scheduler startup, unlike errors in register(), which
    # load_plugins() only logs.
    install(scheduler)
    return result
