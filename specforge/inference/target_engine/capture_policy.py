# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Back-compat shim: this module moved to ``target_capture_policy``.

The rename disambiguates the target-side capture policies (what a draft
algorithm extracts from the frozen target model) from the runtime feature
contract in ``specforge.inference.capture`` (what the rollout must hand the
FeatureStore). Import from ``target_capture_policy`` in new code.
"""

from .target_capture_policy import *  # noqa: F401,F403
from .target_capture_policy import __all__  # noqa: F401
