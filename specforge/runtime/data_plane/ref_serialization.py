# coding=utf-8
"""Dependency-light JSON serialization for tensor-free ``SampleRef`` values."""

from __future__ import annotations

import dataclasses

from specforge.runtime.contracts import FeatureSpec, SampleRef


def ref_to_dict(ref: SampleRef) -> dict:
    return dataclasses.asdict(ref)


def ref_from_dict(raw: dict) -> SampleRef:
    specs = {
        name: FeatureSpec(**{**spec, "shape": tuple(spec["shape"])})
        for name, spec in raw["feature_specs"].items()
    }
    return SampleRef(**{**raw, "feature_specs": specs})


__all__ = ["ref_from_dict", "ref_to_dict"]
