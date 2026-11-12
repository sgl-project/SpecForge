"""Resolved target-logit semantics; target discovery belongs to the parser."""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class TargetLogitSpec:
    """Settings applied to a bound, frozen target LM head."""
