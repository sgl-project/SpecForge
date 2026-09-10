# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Answer extraction and comparison helpers shared by scored tasks."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from typing import Optional

REASON_STEP_BY_STEP = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)

_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", re.IGNORECASE)
_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_FUNCTION = re.compile(r"(def\s+\w+\s*\(.*?)(?=\n\S|\Z)", re.DOTALL)


def extract_boxed(text: str) -> Optional[str]:
    """Return the content of the last ``\\boxed{...}``, handling nested braces."""
    start = text.rfind("\\boxed")
    if start < 0:
        return None
    index = start + len("\\boxed")
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text):
        return None
    if text[index] != "{":
        # ``\boxed 42`` without braces: take the next token.
        match = re.match(r"\S+", text[index:])
        return match.group(0) if match else None
    depth = 0
    for end in range(index, len(text)):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                return text[index + 1 : end].strip()
    return None


def extract_last_number(text: str) -> Optional[str]:
    """Return the last number in ``text`` with thousands separators removed."""
    matches = _NUMBER.findall(text.replace(",", ""))
    return matches[-1] if matches else None


def extract_final_number(text: str) -> Optional[str]:
    """Boxed answer when present, otherwise the last number in the text."""
    boxed = extract_boxed(text)
    if boxed is not None:
        return extract_last_number(boxed) or boxed
    return extract_last_number(text)


def numbers_equal(prediction: Optional[str], label: Optional[str]) -> bool:
    """Compare two answers as numbers, falling back to normalized strings."""
    if prediction is None or label is None:
        return False
    pred = prediction.strip().replace(",", "").rstrip(".")
    ref = label.strip().replace(",", "").rstrip(".")
    if pred == ref:
        return True
    try:
        return abs(float(pred) - float(ref)) < 1e-6
    except ValueError:
        return False


def normalize_math(answer: str) -> str:
    """Canonicalize a LaTeX/plain math answer for string comparison."""
    text = answer.strip()
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    text = text.replace("$", "").replace(" ", "")
    text = re.sub(r"\\dfrac|\\tfrac", r"\\frac", text)
    text = text.rstrip(".")
    return text.lower()


def math_answers_equal(prediction: Optional[str], label: Optional[str]) -> bool:
    if prediction is None or label is None:
        return False
    if numbers_equal(prediction, label):
        return True
    return normalize_math(prediction) == normalize_math(label)


def extract_choice(text: str, choices: str = "ABCD") -> Optional[str]:
    """Return the selected multiple-choice letter.

    Explicit ``Answer: X`` style markers win; otherwise the first standalone
    letter from ``choices`` is used.
    """
    letters = re.escape(choices)
    explicit = [
        rf"answer\s*(?:is|:|：)?\s*\(?\s*([{letters}])\b",
        rf"答案\s*(?:是|为|:|：)?\s*\(?\s*([{letters}])\b",
        rf"\\boxed\{{\s*([{letters}])\s*\}}",
    ]
    for pattern in explicit:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    match = re.search(rf"(?<![A-Za-z])([{letters}])(?![A-Za-z])", text.upper())
    return match.group(1) if match else None


def extract_python_code(text: str) -> Optional[str]:
    """Return the first fenced Python block, else the first function, else all."""
    blocks = _CODE_BLOCK.findall(text)
    if blocks:
        return blocks[0].strip() or None
    match = _FUNCTION.search(text)
    if match:
        return match.group(1).strip()
    stripped = text.strip()
    return stripped or None


def run_python_tests(program: str, timeout_seconds: float = 10.0) -> bool:
    """Run ``program`` in a fresh interpreter and report whether it exited cleanly.

    The program is model-generated code plus test assertions.  A subprocess
    isolates the benchmark from infinite loops and crashes, but it is not a
    security sandbox: only run scored code tasks in an isolated environment.
    """
    with tempfile.TemporaryDirectory(prefix="specforge-bench-") as workdir:
        path = os.path.join(workdir, "program.py")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(program)
        try:
            completed = subprocess.run(
                [sys.executable, path],
                cwd=workdir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False
    return completed.returncode == 0


__all__ = [
    "REASON_STEP_BY_STEP",
    "extract_boxed",
    "extract_choice",
    "extract_final_number",
    "extract_last_number",
    "extract_python_code",
    "math_answers_equal",
    "normalize_math",
    "numbers_equal",
    "run_python_tests",
]
