from __future__ import annotations

import ast
import subprocess
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "specforge"


@dataclass(frozen=True, order=True)
class ImportEdge:
    importer: str
    imported: str
    path: str
    line: int


def _module_name(path: Path) -> tuple[str, bool]:
    relative = path.relative_to(REPO_ROOT).with_suffix("")
    parts = list(relative.parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    return ".".join(parts), is_package


KNOWN_INTERNAL_MODULES = frozenset(
    _module_name(path)[0] for path in PACKAGE_ROOT.rglob("*.py")
)


def _relative_module(
    importer: str,
    *,
    is_package: bool,
    level: int,
    module: str | None,
) -> str:
    package = importer if is_package else importer.rpartition(".")[0]
    parts = package.split(".") if package else []
    ascend = level - 1
    if ascend > len(parts):
        return ""
    base = parts[: len(parts) - ascend]
    if module:
        base.extend(module.split("."))
    return ".".join(base)


def _literal_dynamic_import(node: ast.Call) -> str | None:
    if not node.args or not isinstance(node.args[0], ast.Constant):
        return None
    module = node.args[0].value
    if not isinstance(module, str):
        return None
    function = node.func
    if isinstance(function, ast.Name) and function.id == "__import__":
        return module
    if isinstance(function, ast.Name) and function.id == "import_module":
        return module
    if (
        isinstance(function, ast.Attribute)
        and function.attr == "import_module"
        and isinstance(function.value, ast.Name)
        and function.value.id == "importlib"
    ):
        return module
    return None


def _imports(path: Path) -> Iterator[ImportEdge]:
    importer, is_package = _module_name(path)
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    relative_path = str(path.relative_to(REPO_ROOT))
    for node in ast.walk(tree):
        imported_modules: Iterable[str] = ()
        if isinstance(node, ast.Import):
            imported_modules = (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = _relative_module(
                    importer,
                    is_package=is_package,
                    level=node.level,
                    module=node.module,
                )
            else:
                base = node.module or ""
            modules = [base] if base else []
            if base:
                candidates = (
                    f"{base}.{alias.name}" for alias in node.names if alias.name != "*"
                )
                modules.extend(
                    candidate
                    for candidate in candidates
                    if candidate in KNOWN_INTERNAL_MODULES
                )
            imported_modules = modules
        elif isinstance(node, ast.Call):
            dynamic = _literal_dynamic_import(node)
            imported_modules = (dynamic,) if dynamic else ()
        else:
            continue

        for imported in imported_modules:
            if imported:
                yield ImportEdge(
                    importer=importer,
                    imported=imported,
                    path=relative_path,
                    line=node.lineno,
                )


def _all_edges() -> set[ImportEdge]:
    return {
        edge for path in sorted(PACKAGE_ROOT.rglob("*.py")) for edge in _imports(path)
    }


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{path.relative_to(REPO_ROOT)} does not define {name}")


def _type_checking_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    names = set()
    for node in tree.body:
        if not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            continue
        for statement in node.body:
            if isinstance(statement, ast.ImportFrom):
                names.update(alias.asname or alias.name for alias in statement.names)
    return names


def _matches(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


FORBIDDEN_DEPENDENCIES = {
    "specforge.algorithms": {
        "specforge.application",
        "specforge.config",
        "specforge.core",
        "specforge.eval",
        "specforge.inference",
        "specforge.launch",
        "specforge.runtime",
        "specforge.training",
    },
    "specforge.config": {
        "specforge.algorithms",
        "specforge.application",
        "specforge.eval",
        "specforge.inference",
        "specforge.launch",
        "specforge.runtime",
        "specforge.training",
    },
    "specforge.runtime": {
        "specforge.algorithms",
        "specforge.application",
        "specforge.config",
        "specforge.eval",
        "specforge.training",
    },
    "specforge.training": {
        "specforge.application",
        "specforge.config",
        "specforge.launch",
    },
}


# Existing main-branch debt.  The equality assertion makes this list
# exhaustible: removing the import without removing its exception fails CI.
TEMPORARY_CROSS_PLANE_EXCEPTIONS = {
    (
        "specforge.runtime.control_plane.controller",
        "specforge.runtime.data_plane.sample_ref_queue",
    ): "replace the concrete queue import with a runtime contract port",
}


class PackageDependencyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.edges = _all_edges()

    def test_forbidden_reverse_dependencies_are_absent(self):
        violations = []
        for edge in self.edges:
            for importer_prefix, forbidden_prefixes in FORBIDDEN_DEPENDENCIES.items():
                if not _matches(edge.importer, importer_prefix):
                    continue
                if any(
                    _matches(edge.imported, prefix) for prefix in forbidden_prefixes
                ):
                    violations.append(
                        f"{edge.path}:{edge.line}: {edge.importer} -> {edge.imported}"
                    )
        self.assertEqual([], sorted(violations), "\n".join(sorted(violations)))

    def test_application_is_a_top_level_dependency_boundary(self):
        violations = [
            f"{edge.path}:{edge.line}: {edge.importer} -> {edge.imported}"
            for edge in self.edges
            if _matches(edge.imported, "specforge.application")
            and not (
                _matches(edge.importer, "specforge.application")
                or edge.importer == "specforge.cli"
            )
        ]
        self.assertEqual([], sorted(violations), "\n".join(sorted(violations)))

    def test_control_and_data_plane_cross_imports_are_explicit_debt(self):
        observed = {
            (edge.importer, edge.imported)
            for edge in self.edges
            if (
                _matches(edge.importer, "specforge.runtime.control_plane")
                and _matches(edge.imported, "specforge.runtime.data_plane")
            )
            or (
                _matches(edge.importer, "specforge.runtime.data_plane")
                and _matches(edge.imported, "specforge.runtime.control_plane")
            )
        }
        self.assertEqual(
            set(TEMPORARY_CROSS_PLANE_EXCEPTIONS),
            observed,
            "cross-plane imports changed; remove a stale exception or redesign "
            "the new dependency behind a runtime contract",
        )

    def test_algorithm_contract_modules_are_stdlib_only(self):
        checked = {
            "specforge.algorithms",
            "specforge.algorithms.contracts",
            "specforge.algorithms.registry",
        }
        allowed_roots = {
            "__future__",
            "copy",
            "dataclasses",
            "enum",
            "importlib",
            "re",
            "specforge",
            "types",
            "typing",
        }
        violations = []
        for edge in self.edges:
            if edge.importer not in checked:
                continue
            root = edge.imported.split(".", 1)[0]
            allowed_specforge_import = _matches(edge.imported, "specforge.algorithms")
            if root not in allowed_roots or (
                root == "specforge" and not allowed_specforge_import
            ):
                violations.append(
                    f"{edge.path}:{edge.line}: {edge.importer} -> {edge.imported}"
                )
        self.assertEqual([], sorted(violations), "\n".join(sorted(violations)))

    def test_algorithm_contract_import_does_not_initialize_torch(self):
        code = (
            "import sys; import specforge.algorithms; "
            "assert 'torch' not in sys.modules, sorted(sys.modules)"
        )

        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(0, result.returncode, result.stderr or result.stdout)

    def test_lazy_package_facade_preserves_existing_direct_exports(self):
        facade = PACKAGE_ROOT / "__init__.py"
        lazy_exports = _literal_assignment(facade, "_LAZY_EXPORTS")
        core_exports = _literal_assignment(
            PACKAGE_ROOT / "core" / "__init__.py", "__all__"
        )
        modeling_exports = _literal_assignment(
            PACKAGE_ROOT / "modeling" / "__init__.py", "__all__"
        )

        self.assertEqual(
            set(core_exports) | set(modeling_exports),
            set(lazy_exports),
        )
        self.assertEqual(
            set(lazy_exports),
            _type_checking_imports(facade),
            "runtime-lazy exports and TYPE_CHECKING exports must stay in sync",
        )


if __name__ == "__main__":
    unittest.main()
