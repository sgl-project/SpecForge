# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Built-in benchmark tasks.

Importing this package registers every built-in task in :data:`TASKS`.  Extra
tasks can live anywhere: decorate a :class:`BenchmarkTask` subclass with
``@TASKS.register`` and list the module under ``plugins`` in the config.
"""

from specforge.benchmarks.tasks import (  # noqa: F401  (registration side effects)
    aime,
    ceval,
    financeqa,
    gpqa,
    gsm8k,
    humaneval,
    livecodebench,
    math500,
    mbpp,
    mmlu,
    mmstar,
    mtbench,
    simpleqa,
)
from specforge.benchmarks.tasks.base import TASKS, BenchmarkTask, Sample, TaskRegistry

__all__ = ["BenchmarkTask", "Sample", "TaskRegistry", "TASKS"]
