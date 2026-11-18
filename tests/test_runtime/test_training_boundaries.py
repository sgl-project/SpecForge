"""Import rules of the training plane: who may reach what, read from the source.

The parts (program, step frame, run loop, step runner) meet through
``specforge/training/contracts.py``. Each rule below names the modules it
covers, what they may not reach, and the exceptions that exist today. An
exception records why it is there; the list may only shrink: an entry that no
longer matches anything fails, so it is removed with the code it excused.
"""

from __future__ import annotations

import ast
import unittest
import unittest.mock
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Collectives and process-global topology.
DISTRIBUTED = ("torch.distributed", "specforge.distributed")

PROGRAM = (
    "specforge/algorithms/**/*.py",
    "specforge/modeling/**/*.py",
    "specforge/losses/**/*.py",
    "specforge/layers/**/*.py",
)
FRAME = (
    "specforge/training/contracts.py",
    "specforge/training/frame.py",
    "specforge/training/program.py",
)
RUN_LOOP = ("specforge/training/controller.py", "specforge/eval/**/*.py")
RUNNER = ("specforge/training/runner.py", "specforge/training/backend.py")


@dataclass(frozen=True)
class Rule:
    """Modules matching ``scope`` reach nothing under ``forbidden``.

    ``allowed`` maps a module path to why it is an exception today.
    """

    name: str
    scope: tuple[str, ...]
    forbidden: tuple[str, ...]
    allowed: dict[str, str]


RULES = (
    Rule(
        "a collective across samples appears only in a runner",
        PROGRAM + FRAME + RUN_LOOP,
        DISTRIBUTED,
        {
            # A model-parallel axis: ranks of one group share a sequence or a
            # tensor, never samples.
            "specforge/modeling/draft/layers/attention/backends/cached_flash.py": (
                "sequence-parallel axis: the ring and Ulysses exchanges"
            ),
            "specforge/modeling/draft/models/eagle_rollout.py": (
                "sequence-parallel axis: the rollout's cache exchange"
            ),
            "specforge/layers/ring/utils.py": "sequence-parallel axis: ring attention",
            "specforge/layers/linear.py": "tensor-parallel axis",
            "specforge/layers/embedding.py": "tensor-parallel axis",
            "specforge/layers/lm_head.py": "tensor-parallel axis",
            # Not a model-parallel axis: these go when the run loop takes its
            # topology as a value (design delivery step 4).
            "specforge/training/controller.py": (
                "run loop: the progress bar's rank, the broadcast of the "
                "best-checkpoint flag, the world size in throughput"
            ),
        },
    ),
    Rule(
        "the frame meets the other parts through the contracts only",
        FRAME,
        ("specforge",),
        {},
    ),
    Rule(
        "the run loop holds no runner, backend, objective or model",
        RUN_LOOP,
        (
            "specforge.training.runner",
            "specforge.training.backend",
            "specforge.algorithms",
            "specforge.modeling",
        ),
        {},
    ),
    Rule(
        "a runner decides from declarations, never from a registry",
        RUNNER,
        ("specforge.algorithms", "specforge.modeling", "specforge.config"),
        {},
    ),
    Rule(
        "a program does not know how it is trained",
        PROGRAM,
        (
            "specforge.training",
            "specforge.runtime",
            "specforge.launch",
            "specforge.eval",
        ),
        {},
    ),
)

#: What each rule's scope may import from ``specforge`` although the rule
#: forbids the prefix: the contracts the parts are meant to share.
SHARED_CONTRACTS = (
    "specforge.training.contracts",
    "specforge.runtime.contracts",
)


def _package(path: Path) -> str:
    """The package a module's relative imports resolve against."""
    return ".".join(path.relative_to(REPO_ROOT).parts[:-1])


def _type_checking_nodes(tree: ast.Module) -> set[int]:
    """Nodes under ``if TYPE_CHECKING:``, which import nothing at run time."""
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        name = test.id if isinstance(test, ast.Name) else getattr(test, "attr", None)
        if name == "TYPE_CHECKING":
            guarded.update(id(child) for child in ast.walk(node))
    return guarded


def _dotted(node: ast.AST) -> str | None:
    """``a.b.c`` for an attribute chain over a plain name."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return ".".join([node.id, *reversed(parts)])


def reached(source: str, package: str) -> set[str]:
    """Dotted names a module reaches at run time, wherever the statement sits.

    Imports at module level and inside functions, with relative imports
    resolved against ``package``, and attribute chains such as
    ``torch.distributed.get_rank``.
    """
    tree = ast.parse(source)
    guarded = _type_checking_nodes(tree)
    names: set[str] = set()
    for node in ast.walk(tree):
        if id(node) in guarded:
            continue
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parent = package.split(".")
                parent = parent[: len(parent) - (node.level - 1)]
                base = ".".join(parent + ([node.module] if node.module else []))
            names.update(f"{base}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Attribute):
            dotted = _dotted(node)
            if dotted is not None:
                names.add(dotted)
    return names


def _under(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def violations(rule: Rule) -> dict[str, list[str]]:
    """For each module in the rule's scope, the forbidden names it reaches."""
    found: dict[str, list[str]] = {}
    for pattern in rule.scope:
        for path in sorted(REPO_ROOT.glob(pattern)):
            hits = sorted(
                name
                for name in reached(path.read_text(), _package(path))
                if any(_under(name, prefix) for prefix in rule.forbidden)
                and not any(_under(name, shared) for shared in SHARED_CONTRACTS)
            )
            if hits:
                found[str(path.relative_to(REPO_ROOT))] = hits
    return found


class TrainingBoundaryTest(unittest.TestCase):
    def test_every_scope_matches_source(self):
        """A moved or renamed module would otherwise leave a rule checking nothing."""
        for rule in RULES:
            for pattern in rule.scope:
                with self.subTest(rule=rule.name, pattern=pattern):
                    self.assertTrue(list(REPO_ROOT.glob(pattern)))

    def test_rules_hold_outside_their_exceptions(self):
        for rule in RULES:
            with self.subTest(rule=rule.name):
                unexpected = {
                    path: names
                    for path, names in violations(rule).items()
                    if path not in rule.allowed
                }
                self.assertEqual(unexpected, {})

    def test_exceptions_only_shrink(self):
        for rule in RULES:
            with self.subTest(rule=rule.name):
                self.assertEqual(sorted(set(rule.allowed) - set(violations(rule))), [])


class ReachedTest(unittest.TestCase):
    """The scan sees what the rules rely on it seeing."""

    def test_imports_inside_functions_count(self):
        names = reached("def f():\n    import torch.distributed as dist\n", "p")
        self.assertIn("torch.distributed", names)

    def test_attribute_use_counts_without_an_import_of_the_submodule(self):
        names = reached("import torch\nx = torch.distributed.get_rank()\n", "p")
        self.assertIn("torch.distributed.get_rank", names)

    def test_relative_imports_are_resolved(self):
        package = "specforge.training"
        names = reached("from .runner import DataParallelRunner\n", package)
        self.assertIn("specforge.training.runner.DataParallelRunner", names)
        self.assertIn(
            "specforge.eval.Evaluator",
            reached("from ..eval import Evaluator\n", package),
        )
        self.assertIn(
            "specforge.training.frame", reached("from . import frame\n", package)
        )

    def test_type_checking_imports_do_not_count(self):
        names = reached(
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n"
            "    from specforge.training.runner import DataParallelRunner\n",
            "p",
        )
        self.assertEqual(names, {"typing.TYPE_CHECKING"})

    def test_a_violation_is_reported_with_what_it_reached(self):
        rule = Rule("probe", ("specforge/training/frame.py",), ("specforge",), {})
        self.assertEqual(violations(rule), {})
        strict = Rule(
            "probe", ("specforge/training/frame.py",), ("specforge.training",), {}
        )
        with unittest.mock.patch(
            f"{__name__}.SHARED_CONTRACTS", ("specforge.runtime.contracts",)
        ):
            found = violations(strict)
        self.assertEqual(list(found), ["specforge/training/frame.py"])
        self.assertTrue(
            all(
                name.startswith("specforge.training.contracts.")
                for name in found["specforge/training/frame.py"]
            )
        )


if __name__ == "__main__":
    unittest.main()
