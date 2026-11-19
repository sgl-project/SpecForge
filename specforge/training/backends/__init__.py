# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Step runners beyond the reference.

Today one module, ``plain``, built on the reference runner's own classes.
Nothing in the training plane imports this package: a runner is chosen by
whoever assembles the trainer, and no configuration key selects one yet.
"""
