"""Import rules of the training plane: which module may reach what, read from the source.

The parts (program, step frame, run loop, step runner) meet through
``specforge/training/contracts.py``. Each rule below names the modules it
covers and what they may not reach. A module that breaks a rule today is
listed with the reason and with every forbidden name it reaches and how many
times: one use more or one use less fails, so an exception neither grows nor
outlives its code.

The scan follows names. It does not see which process group a collective runs
on, what a method called on a value does, or anything reached through a name
computed at run time. An exception list in the repository can also be
extended in the change that needs it; that is for review to see.
"""

from __future__ import annotations

import ast
import importlib.util
import unittest
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

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
SHARED_CONTRACTS = ("specforge.training.contracts", "specforge.runtime.contracts")
#: A package whose own names (its value types) are shared and whose
#: submodules (codecs, validators, the registry) are not.
SHARED_VALUE_TYPES = "specforge.config.draft"

_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


@dataclass(frozen=True)
class Excepted:
    """A module that breaks a rule today: why, and each use of what it reaches."""

    reason: str
    reaches: dict[str, int]


@dataclass(frozen=True)
class Rule:
    """Modules matching ``scope`` reach nothing under ``forbidden``."""

    name: str
    scope: tuple[str, ...]
    forbidden: tuple[str, ...]
    excepted: dict[str, Excepted] = field(default_factory=dict)

    def modules(self) -> list[Path]:
        """The modules the rule covers."""
        return sorted(
            {path for pattern in self.scope for path in REPO_ROOT.glob(pattern)}
        )


_T = "torch.distributed"

#: Modules that reach a process group today. The counts include what a helper
#: of the package brings: every use of a helper counts as the group reads in
#: its body.
EXCEPTED_COLLECTIVES = {
    # A model-parallel axis: ranks of one group share a sequence or a tensor,
    # never samples.
    "specforge/modeling/draft/layers/attention/backends/cached_flash.py": Excepted(
        "sequence-parallel axis: names the type of the ring group it hands to "
        "ring attention",
        {_T: 1, f"{_T}.ProcessGroup": 1},
    ),
    "specforge/modeling/draft/models/eagle_rollout.py": Excepted(
        "sequence-parallel axis: asks whether the rollout's ring group has more "
        "than one rank",
        {_T: 1, f"{_T}.get_world_size": 1},
    ),
    "specforge/layers/ring/utils.py": Excepted(
        "sequence-parallel axis: ring attention's exchange between neighbours",
        {
            _T: 1,
            f"{_T}.P2POp": 2,
            f"{_T}.ProcessGroup": 1,
            f"{_T}.batch_isend_irecv": 1,
            f"{_T}.get_global_rank": 2,
            f"{_T}.get_rank": 1,
            f"{_T}.get_world_size": 1,
            f"{_T}.irecv": 1,
            f"{_T}.isend": 1,
        },
    ),
    # Not a model-parallel axis: the run loop's and its helpers' own uses of
    # the process group, which the design moves to the runner and to a
    # topology value.
    "specforge/training/controller.py": Excepted(
        "run loop: the progress bar's rank; rank 0's verdict on whether an "
        "evaluation runs, broadcast; the world size written into a checkpoint",
        {
            _T: 1,
            f"{_T}.broadcast_object_list": 1,
            f"{_T}.get_rank": 2,
            f"{_T}.get_world_size": 2,
            f"{_T}.is_available": 2,
            f"{_T}.is_initialized": 3,
        },
    ),
    "specforge/training/checkpoint.py": Excepted(
        "checkpoint sequence: the gather of write errors, the barrier, rank 0's "
        "best-checkpoint verdict, broadcast, the gather of resume descriptors",
        {
            f"{_T}.all_gather_object": 2,
            f"{_T}.barrier": 1,
            f"{_T}.broadcast_object_list": 1,
            f"{_T}.get_rank": 3,
            f"{_T}.get_world_size": 3,
            f"{_T}.is_available": 1,
            f"{_T}.is_initialized": 6,
        },
    ),
    "specforge/training/profiling.py": Excepted(
        "profiler: the rank in its output's name",
        {
            _T: 1,
            f"{_T}.get_rank": 1,
            f"{_T}.is_available": 1,
            f"{_T}.is_initialized": 1,
        },
    ),
    "specforge/training/trainer.py": Excepted(
        "resume: the world size compared with the checkpoint's",
        {_T: 1, f"{_T}.get_world_size": 1, f"{_T}.is_initialized": 1},
    ),
}

RULES = (
    Rule(
        "a collective across samples appears only in a runner",
        PROGRAM + FRAME + RUN_LOOP + LIFECYCLE,
        DISTRIBUTED,
        EXCEPTED_COLLECTIVES,
    ),
    Rule(
        "the frame imports the shared contracts only",
        FRAME,
        ("specforge",),
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
            "specforge.layers",
            "specforge.losses",
            "torch.optim",
        ),
    ),
    Rule(
        "a runner decides from declarations and imports no registry",
        RUNNER,
        (
            "specforge.algorithms",
            "specforge.modeling",
            "specforge.config.draft.registry",
        ),
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
            "torch.optim",
            "torch.nn.parallel",
        ),
    ),
)


def _under(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def _chain(node: ast.AST) -> tuple[ast.AST, list[str]]:
    """The innermost value of an attribute chain and the attributes after it."""
    attributes = []
    while isinstance(node, ast.Attribute):
        attributes.append(node.attr)
        node = node.value
    return node, attributes[::-1]


def _own(scope: ast.AST) -> list[ast.AST]:
    """Nodes of one scope: nested functions and classes are named, not entered."""
    found, pending = [], list(ast.iter_child_nodes(scope))
    while pending:
        node = pending.pop()
        found.append(node)
        if not isinstance(node, _SCOPES):
            pending.extend(ast.iter_child_nodes(node))
    return found


def _rebound(scope: ast.AST, own: list[ast.AST]) -> set[str]:
    """Names a scope binds itself other than by import: they hide outer ones."""
    names = {
        node.id
        for node in own
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    names.update(
        handler.name
        for handler in own
        if isinstance(handler, ast.ExceptHandler) and handler.name
    )
    arguments = getattr(scope, "args", None)
    if isinstance(arguments, ast.arguments):
        names.update(
            argument.arg
            for argument in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
                arguments.vararg,
                arguments.kwarg,
            )
            if argument is not None
        )
    return names


class _Scan:
    """Every use of an imported name in one module, by the name it resolves to."""

    def __init__(self, package: str) -> None:
        self.package = package
        self.names: Counter[str] = Counter()
        #: Each top-level function's own uses.
        self.functions: dict[str, Counter[str]] = {}
        #: Top-level imports by local name.
        self.imported: dict[str, str] = {}

    def module(self, tree: ast.Module) -> "_Scan":
        self.imported = self._scope(tree, {}, top_level=True)
        return self

    def _resolve(self, node: ast.AST, aliases: dict[str, str]) -> str | None:
        root, attributes = _chain(node)
        if isinstance(root, ast.Name) and root.id in aliases:
            return ".".join([aliases[root.id], *attributes])
        if isinstance(root, ast.Call) and attributes:
            # ``import_module("a.b").c``: the module named, then the chain.
            module = self._dynamic(root, aliases)
            if module is not None:
                return ".".join([module, *attributes])
        return None

    def _dynamic(self, call: ast.Call, aliases: dict[str, str]) -> str | None:
        """The module a literal ``import_module`` / ``__import__`` call names."""
        function = call.func
        builtin = (
            isinstance(function, ast.Name)
            and function.id == "__import__"
            and "__import__" not in aliases
        )
        if (
            not builtin
            and self._resolve(function, aliases) != "importlib.import_module"
        ):
            return None
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        name, package = (
            value.value if isinstance(value, ast.Constant) else None
            for value in (
                call.args[0] if call.args else keywords.get("name"),
                call.args[1] if len(call.args) > 1 else keywords.get("package"),
            )
        )
        if not isinstance(name, str):
            return None
        if name.startswith("."):
            if not isinstance(package, str):
                return None
            return importlib.util.resolve_name(name, package)
        return name

    def _never_run(self, own: list[ast.AST], aliases: dict[str, str]) -> set[int]:
        """Nodes in the branch of a ``typing.TYPE_CHECKING`` test that never runs."""
        skipped: set[int] = set()
        for node in own:
            if not isinstance(node, ast.If):
                continue
            test, only_types = node.test, True
            if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
                test, only_types = test.operand, False
            if self._resolve(test, aliases) != "typing.TYPE_CHECKING":
                continue
            for statement in node.body if only_types else node.orelse:
                skipped.update(id(child) for child in ast.walk(statement))
        return skipped

    def _imports(self, own: list[ast.AST]) -> dict[str, str]:
        """What the scope's import statements bind, by local name."""
        bound = {}
        for node in own:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    bound[alias.asname or root] = alias.name if alias.asname else root
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parent = self.package.split(".")
                    parent = parent[: len(parent) - (node.level - 1)]
                    base = ".".join(parent + ([node.module] if node.module else []))
                for alias in node.names:
                    bound[alias.asname or alias.name] = f"{base}.{alias.name}"
        return bound

    def _scope(
        self, scope: ast.AST, outer: dict[str, str], *, top_level: bool = False
    ) -> dict[str, str]:
        own = _own(scope)
        # A name the scope binds itself is not the outer import any more.
        rebound = _rebound(scope, own)
        aliases = {
            name: origin for name, origin in outer.items() if name not in rebound
        }
        # Which branches run is decided with every import of the scope in
        # view; the imports of the branches that run are then the scope's.
        skipped = self._never_run(own, {**aliases, **self._imports(own)})
        live = [node for node in own if id(node) not in skipped]
        imports = self._imports(live)
        aliases.update(imports)
        uses: Counter[str] = Counter()
        for node in live:
            if isinstance(node, ast.Import):
                uses.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                uses.update(imports[alias.asname or alias.name] for alias in node.names)
        # Only whole chains: ``a.b`` inside ``a.b.c`` is not a use of its own.
        inner = {id(node.value) for node in live if isinstance(node, ast.Attribute)}
        for node in live:
            if id(node) in inner:
                continue
            if isinstance(node, ast.Attribute) or (
                isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            ):
                name = self._resolve(node, aliases)
            elif isinstance(node, ast.Call):
                name = self._dynamic(node, aliases)
            else:
                continue
            if name is not None:
                uses[name] += 1
        self.names.update(uses)
        for node in live:
            if not isinstance(node, _SCOPES):
                continue
            before = Counter(self.names)
            self._scope(node, aliases)
            if top_level and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions[node.name] = self.names - before
        return imports


def reached(source: str, package: str) -> Counter[str]:
    """Every use of an imported name in ``source``, by the dotted name it resolves to.

    An import statement, at module level or inside a function, with relative
    imports resolved against ``package``; a use of the imported name, through
    its alias and with the attributes that follow
    (``import torch as t; t.distributed.barrier()``); a literal
    ``importlib.import_module`` or ``__import__``. Names are resolved in the
    scope they are used in: a function's own import, parameter or assignment
    hides the module's. The branch of a ``typing.TYPE_CHECKING`` test that
    never executes does not count.
    """
    return _Scan(package).module(ast.parse(source)).names


@dataclass(frozen=True)
class _Source:
    """What a module of the package brings to whoever imports from it."""

    #: Top-level imports by local name: where the name really comes from.
    imported: dict[str, str]
    #: Top-level functions by name: what their bodies use.
    functions: dict[str, Counter[str]]


def index(root: Path, package: str) -> dict[str, _Source]:
    """Every module under ``root / package``, by dotted name."""
    sources = {}
    for path in sorted((root / package).rglob("*.py")):
        parts = path.relative_to(root).with_suffix("").parts
        scan = _Scan(".".join(parts[:-1])).module(ast.parse(path.read_text()))
        name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        sources[name] = _Source(scan.imported, scan.functions)
    return sources


@lru_cache(maxsize=None)
def _sources() -> dict[str, _Source]:
    return index(REPO_ROOT, "specforge")


def brought_along(name: str, sources: dict[str, _Source]) -> set[str]:
    """``name`` and what reaching it reaches besides, through the package.

    A name another module only re-exports is followed to where it comes from,
    with the rest of the chain kept (``utils.dist.barrier`` is
    ``torch.distributed.barrier``). A function of the package brings what its
    own body uses, so a helper that issues a collective counts as that
    collective. A class does not: holding or constructing an object is not
    calling its methods.
    """
    seen: set[str] = set()
    pending = [name]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        parts = current.split(".")
        # The longest prefix that is a module of the package owns the rest.
        for cut in range(len(parts) - 1, 0, -1):
            source = sources.get(".".join(parts[:cut]))
            if source is None:
                continue
            symbol, rest = parts[cut], parts[cut + 1 :]
            if symbol in source.imported:
                pending.append(".".join([source.imported[symbol], *rest]))
            elif not rest:
                pending.extend(source.functions.get(symbol, ()))
            break
    return seen


def _shared(name: str, sources: dict[str, _Source]) -> bool:
    if any(_under(name, contract) for contract in SHARED_CONTRACTS):
        return True
    # A name of the package itself, not one of a module inside it.
    return _under(name, SHARED_VALUE_TYPES) and (
        name == SHARED_VALUE_TYPES
        or (
            name.count(".") == SHARED_VALUE_TYPES.count(".") + 1 and name not in sources
        )
    )


def forbidden_uses(
    uses: Counter[str], forbidden: tuple[str, ...], sources: dict[str, _Source]
) -> Counter[str]:
    """Of a module's uses, those that reach a forbidden name, by that name."""
    hits: Counter[str] = Counter()
    for name, count in uses.items():
        if _shared(name, sources):
            continue
        for brought in brought_along(name, sources):
            if any(_under(brought, prefix) for prefix in forbidden):
                hits[brought] += count
    return hits


def violations(rule: Rule) -> dict[str, dict[str, int]]:
    """For each module in the rule's scope, each forbidden name it reaches."""
    found = {}
    for path in rule.modules():
        package = ".".join(path.relative_to(REPO_ROOT).parts[:-1])
        hits = forbidden_uses(
            reached(path.read_text(), package), rule.forbidden, _sources()
        )
        if hits:
            found[str(path.relative_to(REPO_ROOT))] = dict(sorted(hits.items()))
    return found


class TrainingBoundaryTest(unittest.TestCase):
    maxDiff = None

    def test_every_scope_matches_source(self):
        """A moved or renamed module would otherwise leave a rule checking nothing."""
        for rule in RULES:
            self.assertTrue(rule.scope, rule.name)
            for pattern in rule.scope:
                with self.subTest(rule=rule.name, pattern=pattern):
                    self.assertTrue(list(REPO_ROOT.glob(pattern)))

    def test_rules_hold_outside_their_exceptions(self):
        for rule in RULES:
            with self.subTest(rule=rule.name):
                unexpected = {
                    path: names
                    for path, names in violations(rule).items()
                    if path not in rule.excepted
                }
                self.assertEqual(unexpected, {})

    def test_an_exception_covers_exactly_the_uses_it_lists(self):
        """One use more is a new exception; one less is a stale one."""
        for rule in RULES:
            found = violations(rule)
            for path, excepted in rule.excepted.items():
                with self.subTest(rule=rule.name, path=path):
                    self.assertEqual(
                        found.get(path, {}), dict(sorted(excepted.reaches.items()))
                    )

    def test_no_covered_module_star_imports_from_the_package(self):
        """A star import brings names the scan cannot attribute."""
        covered = {path for rule in RULES for path in rule.modules()}
        for path in sorted(covered):
            package = ".".join(path.relative_to(REPO_ROOT).parts[:-1])
            stars = [
                name
                for name in reached(path.read_text(), package)
                if name.endswith(".*") and _under(name, "specforge")
            ]
            self.assertEqual(stars, [], path)


class ReachedTest(unittest.TestCase):
    """The scan sees what the rules rely on it seeing."""

    def test_imports_inside_functions_count(self):
        names = reached("def f():\n    import torch.distributed as dist\n", "p")
        self.assertEqual(names, {"torch.distributed": 1})

    def test_use_through_an_alias_is_resolved(self):
        names = reached("import torch as t\nt.distributed.all_reduce(x)\n", "p")
        self.assertEqual(names["torch.distributed.all_reduce"], 1)
        names = reached("import torch.distributed as dist\ndist.barrier()\n", "p")
        self.assertEqual(names["torch.distributed.barrier"], 1)
        names = reached("from torch import distributed as d\nd.barrier()\n", "p")
        self.assertEqual(names["torch.distributed.barrier"], 1)
        # The import and the call by the imported name.
        names = reached("from torch.distributed import barrier\nbarrier()\n", "p")
        self.assertEqual(names["torch.distributed.barrier"], 2)

    def test_every_use_is_counted(self):
        source = (
            "import torch\ntorch.distributed.barrier()\ntorch.distributed.barrier()\n"
        )
        self.assertEqual(reached(source, "p")["torch.distributed.barrier"], 2)

    def test_names_are_resolved_in_their_own_scope(self):
        # An import in one function says nothing about a name in another.
        source = (
            "import torch as t\n"
            "def forward():\n"
            "    t.distributed.barrier()\n"
            "def timing():\n"
            "    import time as t\n"
            "    return t.monotonic()\n"
        )
        names = reached(source, "p")
        self.assertEqual(names["torch.distributed.barrier"], 1)
        self.assertEqual(names["time.monotonic"], 1)
        self.assertNotIn("time.distributed.barrier", names)
        # A parameter or an assignment hides the module's import.
        hidden = (
            "import torch\n"
            "def f(torch):\n"
            "    return torch.distributed.rank\n"
            "def g(values):\n"
            "    torch = values\n"
            "    return torch.distributed.rank\n"
        )
        self.assertEqual(reached(hidden, "p"), {"torch": 1})
        # And a name that was never imported is not a module.
        never = "def f(torch):\n    return torch.distributed\n"
        self.assertEqual(reached(never, "p"), {})

    def test_only_whole_chains_count(self):
        source = (
            "import specforge.training.contracts\nspecforge.training.contracts.X()\n"
        )
        self.assertEqual(
            reached(source, "p"),
            {"specforge.training.contracts": 1, "specforge.training.contracts.X": 1},
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
            set(reached(types_only, "p")), {"typing.TYPE_CHECKING", "torch.distributed"}
        )
        inverted = (
            "import typing as t\n"
            "if not t.TYPE_CHECKING:\n"
            "    import torch.distributed\n"
            "else:\n"
            "    import specforge.training.runner\n"
        )
        self.assertEqual(
            set(reached(inverted, "p")),
            {"typing", "typing.TYPE_CHECKING", "torch.distributed"},
        )
        renamed = (
            "from typing import TYPE_CHECKING as TC\n"
            "if TC:\n"
            "    import specforge.training.runner\n"
        )
        self.assertEqual(set(reached(renamed, "p")), {"typing.TYPE_CHECKING"})
        # The guard is typing's, not anything spelled like it.
        for unrelated in (
            "if settings.TYPE_CHECKING:\n    import torch.distributed\n",
            "def f(TYPE_CHECKING=True):\n"
            "    if TYPE_CHECKING:\n"
            "        import torch.distributed\n",
        ):
            self.assertEqual(set(reached(unrelated, "p")), {"torch.distributed"})

    def test_literal_dynamic_imports_count(self):
        for source, expected in (
            (
                'import importlib\nimportlib.import_module("torch.distributed")\n',
                "torch.distributed",
            ),
            ('__import__("torch.distributed")', "torch.distributed"),
            (
                "from importlib import import_module\n"
                'import_module(name="torch.distributed").barrier()\n',
                "torch.distributed.barrier",
            ),
            (
                "import importlib\n"
                'importlib.import_module(".runner", package="specforge.training")\n',
                "specforge.training.runner",
            ),
        ):
            with self.subTest(source=source):
                self.assertEqual(reached(source, "p")[expected], 1)


class BroughtAlongTest(unittest.TestCase):
    """A name is followed through a package, here one written for the test."""

    @classmethod
    def setUpClass(cls):
        files = {
            "pkg/__init__.py": "",
            "pkg/comm.py": (
                "import torch.distributed as dist\n"
                "from torch.distributed import barrier\n"
                "def wait():\n"
                "    dist.barrier()\n"
                "def wait_by_name():\n"
                "    barrier()\n"
                "def harmless():\n"
                "    return 1\n"
                "class Group:\n"
                "    def gather(self):\n"
                "        return dist.all_gather_object\n"
            ),
            "pkg/public.py": "from pkg.comm import Group, wait, harmless as fine\n",
            "pkg/outer.py": "from pkg.public import wait\n",
            "pkg/optional.py": (
                "try:\n"
                "    import torch.distributed as dist\n"
                "except ImportError:\n"
                "    dist = None\n"
                "def wait():\n"
                "    dist.barrier()\n"
            ),
        }
        cls.directory = TemporaryDirectory()
        for name, text in files.items():
            path = Path(cls.directory.name) / name
            path.parent.mkdir(exist_ok=True)
            path.write_text(text)
        cls.sources = index(Path(cls.directory.name), "pkg")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def along(self, name):
        return brought_along(name, self.sources)

    def test_helper_that_issues_a_collective_counts_as_it(self):
        self.assertIn("torch.distributed.barrier", self.along("pkg.comm.wait"))
        # Also where the helper uses the collective by its imported name.
        self.assertIn("torch.distributed.barrier", self.along("pkg.comm.wait_by_name"))
        self.assertEqual(self.along("pkg.comm.harmless"), {"pkg.comm.harmless"})

    def test_reexport_is_followed_to_its_origin_with_the_rest_of_the_chain(self):
        self.assertIn("torch.distributed.barrier", self.along("pkg.comm.dist.barrier"))
        # Through two modules, and each name of one statement to its own origin.
        self.assertIn("torch.distributed.barrier", self.along("pkg.outer.wait"))
        self.assertEqual(
            self.along("pkg.public.fine"), {"pkg.public.fine", "pkg.comm.harmless"}
        )

    def test_class_brings_nothing(self):
        along = self.along("pkg.public.Group")
        self.assertEqual(along, {"pkg.public.Group", "pkg.comm.Group"})

    def test_import_inside_a_module_level_try_is_the_modules(self):
        self.assertIn("torch.distributed.barrier", self.along("pkg.optional.wait"))

    def test_forbidden_uses_are_counted_by_what_they_reach(self):
        uses = reached(
            "from pkg.public import wait\nwait()\nwait()\nfrom pkg.comm import harmless\n",
            "other",
        )
        self.assertEqual(
            forbidden_uses(uses, ("torch.distributed",), self.sources),
            {"torch.distributed.barrier": 3},
        )

    def test_shared_package_shares_its_own_names_and_not_its_modules(self):
        sources = _sources()
        self.assertTrue(_shared("specforge.config.draft.DraftConfig", sources))
        self.assertTrue(_shared("specforge.training.contracts.StepContext", sources))
        self.assertFalse(_shared("specforge.config.draft.registry", sources))
        self.assertFalse(
            _shared("specforge.config.draft.codec.v1.decode_config", sources)
        )


if __name__ == "__main__":
    unittest.main()
