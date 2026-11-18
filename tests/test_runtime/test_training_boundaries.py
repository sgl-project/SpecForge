"""Import rules of the training plane: which module may reach what, read from the source.

The parts (program, step frame, run loop, step runner) meet through
``specforge/training/contracts.py``. Each rule below names the modules it
covers and what they may not reach. A module that breaks a rule today is
listed with the reason and with exactly what it reaches: one name more or one
name less fails, so an exception neither grows nor outlives its code.

What the scan sees is which names a module reaches. It does not see which
process group a collective runs on, nor what a method called on a value does.
"""

from __future__ import annotations

import ast
import unittest
import unittest.mock
from dataclasses import dataclass
from functools import lru_cache
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
RUN_LOOP = (
    "specforge/training/controller.py",
    "specforge/eval/evaluator.py",
    "specforge/eval/**/*.py",
)
#: What the run loop calls around a step; only the first rule covers these:
#: ``trainer.py`` assembles the runner and so imports it.
LIFECYCLE = (
    "specforge/training/checkpoint.py",
    "specforge/training/profiling.py",
    "specforge/training/trainer.py",
)
RUNNER = (
    "specforge/training/runner.py",
    "specforge/training/backend.py",
    "specforge/optimizer.py",
)

#: The contracts the parts are meant to share, whatever a rule forbids.
SHARED_CONTRACTS = (
    "specforge.training.contracts",
    "specforge.runtime.contracts",
    "specforge.config.draft",
)
#: Inside a shared package, what is a lookup and not a value type.
NOT_SHARED = ("specforge.config.draft.registry",)


@dataclass(frozen=True)
class Excepted:
    """A module that breaks a rule today: why, and exactly what it reaches."""

    reason: str
    reaches: frozenset[str]


@dataclass(frozen=True)
class Rule:
    """Modules matching ``scope`` reach nothing under ``forbidden``."""

    name: str
    scope: tuple[str, ...]
    forbidden: tuple[str, ...]
    excepted: dict[str, Excepted]


def _excepted(reason: str, *reaches: str) -> Excepted:
    return Excepted(reason, frozenset(reaches))


_TORCH = "torch.distributed"
_SPECFORGE = "specforge.distributed"

RULES = (
    Rule(
        "a collective across samples appears only in a runner",
        PROGRAM + FRAME + RUN_LOOP + LIFECYCLE,
        DISTRIBUTED,
        {
            # A model-parallel axis: ranks of one group share a sequence or a
            # tensor, never samples.
            "specforge/modeling/draft/layers/attention/backends/cached_flash.py": (
                _excepted(
                    "sequence-parallel axis: names the type of the ring group it "
                    "hands to ring attention",
                    _TORCH,
                    f"{_TORCH}.ProcessGroup",
                )
            ),
            "specforge/modeling/draft/models/eagle_rollout.py": _excepted(
                "sequence-parallel axis: asks whether the rollout's ring group "
                "has more than one rank",
                _TORCH,
                f"{_TORCH}.get_world_size",
            ),
            "specforge/layers/ring/utils.py": _excepted(
                "sequence-parallel axis: ring attention's exchange between "
                "neighbours",
                _TORCH,
                f"{_TORCH}.P2POp",
                f"{_TORCH}.ProcessGroup",
                f"{_TORCH}.batch_isend_irecv",
                f"{_TORCH}.get_global_rank",
                f"{_TORCH}.get_rank",
                f"{_TORCH}.get_world_size",
                f"{_TORCH}.irecv",
                f"{_TORCH}.isend",
            ),
            "specforge/layers/embedding.py": _excepted(
                "tensor-parallel axis: an embedding sharded over the group, "
                "summed back",
                _TORCH,
                f"{_TORCH}.ProcessGroup",
                f"{_TORCH}.ReduceOp.SUM",
                f"{_TORCH}.all_reduce",
                f"{_TORCH}.get_rank",
                f"{_TORCH}.get_world_size",
                f"{_SPECFORGE}.get_tp_group",
                f"{_SPECFORGE}.shard_tensor",
            ),
            "specforge/layers/linear.py": _excepted(
                "tensor-parallel axis: a linear layer sharded over the group",
                _TORCH,
                f"{_TORCH}.ProcessGroup",
                f"{_TORCH}.get_rank",
                f"{_TORCH}.get_world_size",
                f"{_SPECFORGE}.get_tp_group",
                f"{_SPECFORGE}.shard_tensor",
            ),
            "specforge/layers/lm_head.py": _excepted(
                "tensor-parallel axis: a head sharded over the group, gathered " "back",
                _TORCH,
                f"{_TORCH}.ProcessGroup",
                f"{_TORCH}.all_gather",
                f"{_TORCH}.get_rank",
                f"{_TORCH}.get_world_size",
                f"{_SPECFORGE}.get_tp_group",
                f"{_SPECFORGE}.shard_tensor",
            ),
            # Not a model-parallel axis. These are the run loop's and its
            # helpers' own uses of the process group, which the design moves
            # behind the runner (`agree`) and a topology value.
            "specforge/training/controller.py": _excepted(
                "run loop: the progress bar's rank; rank 0's verdict on whether "
                "an evaluation runs, broadcast; the world size written into a "
                "checkpoint",
                _TORCH,
                f"{_TORCH}.broadcast_object_list",
                f"{_TORCH}.get_rank",
                f"{_TORCH}.get_world_size",
                f"{_TORCH}.is_available",
                f"{_TORCH}.is_initialized",
            ),
            "specforge/training/checkpoint.py": _excepted(
                "checkpoint sequence: the gather of write errors, the barriers, "
                "the broadcast of the best-checkpoint verdict, the gather of "
                "resume descriptors",
                f"{_TORCH}.all_gather_object",
                f"{_TORCH}.barrier",
                f"{_TORCH}.broadcast_object_list",
                f"{_TORCH}.get_rank",
                f"{_TORCH}.get_world_size",
                f"{_TORCH}.is_available",
                f"{_TORCH}.is_initialized",
            ),
            "specforge/training/profiling.py": _excepted(
                "profiler: the rank in its output's name",
                _TORCH,
                f"{_TORCH}.get_rank",
                f"{_TORCH}.is_available",
                f"{_TORCH}.is_initialized",
            ),
            "specforge/training/trainer.py": _excepted(
                "resume: the world size compared with the checkpoint's",
                _TORCH,
                f"{_TORCH}.get_world_size",
                f"{_TORCH}.is_initialized",
            ),
        },
    ),
    Rule(
        "the frame imports the shared contracts only",
        FRAME,
        ("specforge",),
        {},
    ),
    Rule(
        "the run loop imports no runner, backend, optimizer, objective or model",
        RUN_LOOP,
        (
            "specforge.training.runner",
            "specforge.training.backend",
            "specforge.training.backends",
            "specforge.optimizer",
            "specforge.algorithms",
            "specforge.modeling",
        ),
        {},
    ),
    Rule(
        "a runner decides from declarations and imports no registry",
        RUNNER,
        (
            "specforge.algorithms",
            "specforge.modeling",
            "specforge.config.draft.registry",
        ),
        {},
    ),
    Rule(
        "a program imports nothing that trains it",
        PROGRAM,
        (
            "specforge.training",
            "specforge.runtime",
            "specforge.launch",
            "specforge.eval",
            "specforge.optimizer",
            "torch.nn.parallel",
        ),
        {},
    ),
)


def _under(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def _type_checking(test: ast.expr) -> bool | None:
    """True for ``TYPE_CHECKING``, False for ``not TYPE_CHECKING``, else None."""
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        inner = _type_checking(test.operand)
        return None if inner is None else not inner
    if isinstance(test, ast.Name):
        return True if test.id == "TYPE_CHECKING" else None
    if (
        isinstance(test, ast.Attribute)
        and test.attr == "TYPE_CHECKING"
        and isinstance(test.value, ast.Name)
        and test.value.id == "typing"
    ):
        return True
    return None


def _never_run(tree: ast.AST) -> set[int]:
    """Nodes in the branch of a ``TYPE_CHECKING`` test that never executes."""
    skipped: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        only_types = _type_checking(node.test)
        if only_types is None:
            continue
        for statement in node.body if only_types else node.orelse:
            skipped.update(id(child) for child in ast.walk(statement))
    return skipped


def _dotted(node: ast.AST) -> str | None:
    """``a.b.c`` for an attribute chain over a plain name."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return ".".join([node.id, *reversed(parts)])


def reached(source: str | ast.AST, package: str) -> set[str]:
    """Dotted names a piece of source reaches at run time, wherever the statement sits.

    Imports at module level and inside functions, relative ones resolved
    against ``package``; attribute chains whose root is an imported name,
    through its alias (``import torch as t; t.distributed.barrier()``); and
    ``importlib.import_module`` / ``__import__`` of a literal name. The branch
    of a ``TYPE_CHECKING`` test that never executes does not count.
    """
    tree = ast.parse(source) if isinstance(source, str) else source
    skipped = _never_run(tree)
    live = [node for node in ast.walk(tree) if id(node) not in skipped]
    names: set[str] = set()
    aliases: dict[str, str] = {}
    for node in live:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
                root = alias.name.split(".")[0]
                aliases[alias.asname or root] = alias.name if alias.asname else root
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parent = package.split(".")
                parent = parent[: len(parent) - (node.level - 1)]
                base = ".".join(parent + ([node.module] if node.module else []))
            for alias in node.names:
                names.add(f"{base}.{alias.name}")
                aliases[alias.asname or alias.name] = f"{base}.{alias.name}"
    # Only whole chains: ``a.b`` inside ``a.b.c`` is not a use of its own.
    inner = {id(node.value) for node in live if isinstance(node, ast.Attribute)}

    def resolved(node: ast.AST) -> str | None:
        dotted = _dotted(node)
        if dotted is None:
            return None
        root, _, rest = dotted.partition(".")
        if root not in aliases:
            return None
        return aliases[root] + (f".{rest}" if rest else "")

    for node in live:
        if isinstance(node, ast.Attribute) and id(node) not in inner:
            name = resolved(node)
            if name is not None:
                names.add(name)
        elif isinstance(node, ast.Call) and node.args:
            literal = node.args[0]
            dynamic = (
                isinstance(node.func, ast.Name) and node.func.id == "__import__"
            ) or resolved(node.func) == "importlib.import_module"
            if (
                dynamic
                and isinstance(literal, ast.Constant)
                and isinstance(literal.value, str)
            ):
                names.add(literal.value)
    return names


def _package(path: Path) -> str:
    """The package a module's relative imports resolve against."""
    return ".".join(path.relative_to(REPO_ROOT).parts[:-1])


def _module(path: Path) -> str:
    parts = path.relative_to(REPO_ROOT).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


@dataclass(frozen=True)
class _Source:
    """One module: what it reaches, and what each top-level name brings along."""

    #: Everything the module reaches.
    reached: frozenset[str]
    #: Top-level imports by local name: where the name really comes from.
    imported: dict[str, str]
    #: Top-level functions by name: what their bodies reach.
    defined: dict[str, frozenset[str]]


@lru_cache(maxsize=None)
def _sources() -> dict[str, _Source]:
    """Every module of the package, by dotted name."""
    sources = {}
    for path in sorted((REPO_ROOT / "specforge").rglob("*.py")):
        tree, package = ast.parse(path.read_text()), _package(path)
        imports = [
            statement
            for statement in tree.body
            if isinstance(statement, (ast.Import, ast.ImportFrom))
        ]
        imported: dict[str, str] = {}
        for statement in imports:
            for alias in statement.names:
                if isinstance(statement, ast.Import):
                    root = alias.name.split(".")[0]
                    origin = alias.name if alias.asname else root
                    imported[alias.asname or root] = origin
                else:
                    (origin,) = reached(_only(statement), package)
                    imported[alias.asname or alias.name] = origin
        from_imports = reached(ast.Module(body=imports, type_ignores=[]), package)
        defined = {
            statement.name: frozenset(
                # Through the module's imports, and through its own.
                reached(
                    ast.Module(body=[*imports, statement], type_ignores=[]), package
                )
                - from_imports
            )
            for statement in tree.body
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        sources[_module(path)] = _Source(
            frozenset(reached(tree, package)), imported, defined
        )
    return sources


def _only(statement: ast.ImportFrom) -> ast.Module:
    """One ``from ... import name`` per module, so each name has one origin."""
    return ast.Module(
        body=[
            ast.ImportFrom(
                module=statement.module, names=[alias], level=statement.level
            )
            for alias in statement.names[:1]
        ],
        type_ignores=[],
    )


def brought_along(name: str) -> set[str]:
    """What reaching ``name`` reaches besides: one step through the package.

    A name another module only re-exports is followed to where it comes from.
    A function of the package brings what its own body reaches, so a helper
    that issues a collective counts as that collective. A class does not:
    holding or constructing an object is not calling its methods, and what a
    method call on a value does is beyond a scan of names.
    """
    sources = _sources()
    seen: set[str] = set()
    for _ in range(4):
        module, _, symbol = name.rpartition(".")
        source = sources.get(module)
        if source is None or name in seen:
            break
        seen.add(name)
        if symbol in source.imported:
            name = source.imported[symbol]
            continue
        return seen | set(source.defined.get(symbol, ()))
    return seen | {name}


def violations(rule: Rule) -> dict[str, set[str]]:
    """For each module in the rule's scope, the forbidden names it reaches."""
    found: dict[str, set[str]] = {}
    paths = {path for pattern in rule.scope for path in REPO_ROOT.glob(pattern)}
    for path in sorted(paths):
        names = set()
        for name in reached(path.read_text(), _package(path)):
            names |= {name} | brought_along(name)
        hits = {
            name
            for name in names
            if any(_under(name, prefix) for prefix in rule.forbidden)
            and (
                not any(_under(name, shared) for shared in SHARED_CONTRACTS)
                or any(_under(name, lookup) for lookup in NOT_SHARED)
            )
        }
        if hits:
            found[str(path.relative_to(REPO_ROOT))] = hits
    return found


class TrainingBoundaryTest(unittest.TestCase):
    maxDiff = None

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
                    path: sorted(names)
                    for path, names in violations(rule).items()
                    if path not in rule.excepted
                }
                self.assertEqual(unexpected, {})

    def test_an_exception_covers_exactly_what_it_lists(self):
        """One operation more is a new exception; one less is a stale one."""
        for rule in RULES:
            found = violations(rule)
            for path, excepted in rule.excepted.items():
                with self.subTest(rule=rule.name, path=path):
                    self.assertEqual(
                        sorted(found.get(path, ())), sorted(excepted.reaches)
                    )


class ReachedTest(unittest.TestCase):
    """The scan sees what the rules rely on it seeing."""

    def test_imports_inside_functions_count(self):
        names = reached("def f():\n    import torch.distributed as dist\n", "p")
        self.assertIn("torch.distributed", names)

    def test_use_through_an_alias_is_resolved(self):
        names = reached("import torch as t\nt.distributed.all_reduce(x)\n", "p")
        self.assertIn("torch.distributed.all_reduce", names)
        names = reached("import torch.distributed as dist\ndist.barrier()\n", "p")
        self.assertIn("torch.distributed.barrier", names)
        names = reached("from torch import distributed as d\nd.barrier()\n", "p")
        self.assertIn("torch.distributed.barrier", names)

    def test_a_local_name_is_not_a_module(self):
        names = reached("def f(torch):\n    return torch.distributed.rank\n", "p")
        self.assertEqual(names, set())

    def test_only_whole_chains_count(self):
        source = (
            "import specforge.training.contracts\nspecforge.training.contracts.X()\n"
        )
        self.assertEqual(
            reached(source, "p"),
            {"specforge.training.contracts", "specforge.training.contracts.X"},
        )

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

    def test_only_the_branch_that_never_runs_is_skipped(self):
        types_only = (
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n"
            "    from specforge.training.runner import DataParallelRunner\n"
            "else:\n"
            "    import torch.distributed as dist\n"
        )
        self.assertEqual(
            reached(types_only, "p"), {"typing.TYPE_CHECKING", "torch.distributed"}
        )
        inverted = (
            "import typing\n"
            "if not typing.TYPE_CHECKING:\n"
            "    import torch.distributed\n"
            "else:\n"
            "    import specforge.training.runner\n"
        )
        self.assertEqual(
            reached(inverted, "p"),
            {"typing", "typing.TYPE_CHECKING", "torch.distributed"},
        )
        unrelated = "if settings.TYPE_CHECKING:\n    import torch.distributed\n"
        self.assertEqual(reached(unrelated, "p"), {"torch.distributed"})

    def test_literal_dynamic_imports_count(self):
        names = reached(
            'import importlib\nimportlib.import_module("torch.distributed")\n', "p"
        )
        self.assertIn("torch.distributed", names)
        self.assertIn(
            "torch.distributed", reached('__import__("torch.distributed")', "p")
        )


class BroughtAlongTest(unittest.TestCase):
    """A name is followed one step into the package."""

    def test_helper_that_issues_a_collective_counts_as_it(self):
        self.assertIn(
            "torch.distributed.barrier",
            brought_along("specforge.utils.rank_0_priority"),
        )

    def test_reexport_is_followed_to_its_origin(self):
        self.assertIn("torch.distributed", brought_along("specforge.utils.dist"))
        self.assertIn(
            "specforge.training.runner.DataParallelRunner",
            brought_along("specforge.training.trainer.DataParallelRunner"),
        )

    def test_value_type_brings_nothing_forbidden(self):
        along = brought_along("specforge.training.contracts.StepContext")
        self.assertFalse(any(_under(name, "torch.distributed") for name in along))

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
                name.startswith("specforge.training.contracts")
                for name in found["specforge/training/frame.py"]
            )
        )


if __name__ == "__main__":
    unittest.main()
