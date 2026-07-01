# coding=utf-8
"""Evaluation: correct acceptance-length / accuracy metrics for draft training,
plus the eval identity (:class:`EvalConfig`) and the on-disk feature cache
(:class:`EvalCache`) that keeps repeat evals from recomputing hidden states."""

from specforge.eval.cache import EvalCache
from specforge.eval.evaluator import EvalConfig, Evaluator

__all__ = ["EvalCache", "EvalConfig", "Evaluator"]
