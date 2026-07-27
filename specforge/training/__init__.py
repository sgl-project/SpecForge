# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Domain training layer (Phase B): the caller-facing lifecycle over the runtime.

``Trainer`` WRAPS — does not replace — the runtime training seam
(``TrainerController`` / ``TrainerCore`` / ``DraftTrainStrategy`` /
``FSDPTrainingBackend``). ``CheckpointManager`` and the learning-rate schedule
live here on top of the same seam.

Import-light: the ``Trainer`` (which imports the GPU/model-heavy runtime backend)
is imported lazily so ``import specforge.training`` stays cheap.
"""

from __future__ import annotations

__all__ = ["Trainer"]


def __getattr__(name):  # PEP 562 lazy re-export
    if name == "Trainer":
        from .trainer import Trainer

        return Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
