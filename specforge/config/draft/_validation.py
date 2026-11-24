"""Scalar checks shared by configuration nodes; never coerce user values."""

import math


def positive_int(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def boolean(name: str, value: bool) -> None:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")


def finite_number(name: str, value: float) -> None:
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite number")


def positive_number(name: str, value: float) -> None:
    finite_number(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def nonempty_string(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
