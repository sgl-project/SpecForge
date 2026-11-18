# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""What a canonical objective model declares to the frame and a runner."""

from __future__ import annotations

from specforge.training.contracts import PartitionHints, ProgramSpec


def program_spec(model) -> ProgramSpec:
    """Read the declarations of an objective model from its registration.

    Args:
        model: The unwrapped ``ObjectiveModel``.
    """
    registration = model.registration
    if registration is None:
        raise ValueError(
            f"objective model {model.objective_name!r} was built without a "
            "registration and declares nothing a runner could honour"
        )
    statistic = registration.training.window_statistic
    return ProgramSpec(
        name=model.objective_name,
        inputs=tuple(registration.inputs),
        replay_statistics=() if statistic is None else (statistic.argument_name,),
        evaluation_diagnostics=registration.training.evaluation_diagnostics,
        reduction=registration.training.reduction,
        # A draft class names its transformer block; a draft without one
        # (EAGLE) is a single unit.
        # Sorted: a draft class may hold the names in a set, whose order
        # differs between the ranks' processes.
        partition=PartitionHints(
            blocks=tuple(sorted(model.draft_model._no_split_modules))
        ),
    )


__all__ = ["program_spec"]
