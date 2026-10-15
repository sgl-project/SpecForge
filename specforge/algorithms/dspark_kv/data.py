"""Compatibility entry point for original DSpark-KV offline features."""

from functools import partial

from specforge.algorithms.common.target_kv_data import (
    FEATURES,
    build_collator,
    build_normalizer,
)
from specforge.algorithms.common.target_kv_data import build_reader as _build_reader
from specforge.algorithms.common.target_kv_data import (
    normalize_sample,
    validate_manifest,
)

NORMALIZER_ID = "dspark_target_kv_v1"
build_reader = partial(
    _build_reader,
    strategy="dspark_kv",
    include_teacher=True,
    target_repr="hidden_state",
)

__all__ = [
    "FEATURES",
    "NORMALIZER_ID",
    "build_collator",
    "build_normalizer",
    "build_reader",
    "normalize_sample",
    "validate_manifest",
]
