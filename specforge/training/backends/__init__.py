# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Runtimes beyond the reference.

``plain``: one process, no wrapper, built on the reference runner's classes;
no configuration key selects it. ``fsdp2``: FSDP2 with fp32 masters, selected
by ``training.runtime: fsdp2``. Only ``Trainer``, which builds the runtime the
run chose, imports this package.
"""
