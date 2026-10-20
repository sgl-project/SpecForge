# coding=utf-8
"""Target families, one module each; importing one registers its preset and layout."""

# Adding a family: its module plus one import line here. Core never imports this.

from . import qwen  # noqa: F401  (registers qwen3_5_moe and the qwen layout)
