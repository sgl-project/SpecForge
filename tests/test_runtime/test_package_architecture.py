"""Architecture guards for the post-consolidation package layout."""

from __future__ import annotations

import ast
import tomllib
import tokenize
import unittest
from collections.abc import Iterator
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

# These modules were temporary move-only compatibility shims. Keeping the list
# explicit makes an accidental reintroduction fail without importing torch or
# optional inference backends.
REMOVED_MODULE_FILES = (
    "specforge/modeling/target/base.py",
    "specforge/modeling/target/factory.py",
    "specforge/modeling/target/dflash_target_model.py",
    "specforge/modeling/target/eagle3_target_model.py",
    "specforge/modeling/target/sglang_backend/__init__.py",
    "specforge/runtime/inference/capture.py",
    "specforge/runtime/inference/dflash_adapter.py",
    "specforge/runtime/inference/sglang_adapter.py",
    "specforge/runtime/training/__init__.py",
    "specforge/tracker.py",
)

REMOVED_TRAINING_ENTRY_FILES = (
    "scripts/train_eagle3.py",
    "scripts/train_eagle3_dataflow.py",
    "scripts/train_dflash.py",
    "scripts/train_domino.py",
    "scripts/train_peagle.py",
    "examples/disagg/run_disagg_eagle3.py",
    "examples/disagg/run_disagg_dflash.py",
    "examples/disagg/run_disagg_domino.py",
    "examples/disagg/run_disagg_dspark.py",
)

REMOVED_PACKAGE_DIRECTORIES = (
    "specforge/modeling/target/sglang_backend",
    "specforge/runtime/inference",
    "specforge/runtime/training",
)

REMOVED_MODULE_PREFIXES = (
    "specforge.modeling.target.base",
    "specforge.modeling.target.factory",
    "specforge.modeling.target.dflash_target_model",
    "specforge.modeling.target.eagle3_target_model",
    "specforge.modeling.target.sglang_backend",
    "specforge.runtime.inference",
    "specforge.runtime.training",
    "specforge.tracker",
    "scripts.train_eagle3",
    "scripts.train_eagle3_dataflow",
    "scripts.train_dflash",
    "scripts.train_domino",
    "scripts.train_peagle",
    "train_eagle3",
    "train_eagle3_dataflow",
    "train_dflash",
    "train_domino",
    "train_peagle",
)

SOURCE_ROOTS = (
    REPO_ROOT / "specforge",
    REPO_ROOT / "scripts",
    REPO_ROOT / "examples",
    REPO_ROOT / "tests",
)


def _is_removed_module(module: str) -> bool:
    return any(
        module == prefix or module.startswith(f"{prefix}.")
        for prefix in REMOVED_MODULE_PREFIXES
    )


def _imported_modules(path: Path) -> Iterator[tuple[int, str]]:
    with tokenize.open(path) as source_file:
        tree = ast.parse(source_file.read(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.lineno, node.module


class TestPackageArchitecture(unittest.TestCase):
    def test_package_exposes_one_training_cli(self):
        with open(REPO_ROOT / "pyproject.toml", "rb") as project_file:
            project = tomllib.load(project_file)
        self.assertEqual(
            {"specforge": "specforge.cli:main"},
            project["project"]["scripts"],
        )

    def test_removed_training_telemetry_stays_out_of_runtime_dependencies(self):
        with open(REPO_ROOT / "pyproject.toml", "rb") as project_file:
            dependencies = tomllib.load(project_file)["project"]["dependencies"]
        names = {item.split("[", 1)[0].split("=", 1)[0].lower() for item in dependencies}
        self.assertTrue({"wandb", "tensorboard"}.isdisjoint(names))

    def test_removed_move_shims_stay_deleted(self):
        present = [
            relative_path
            for relative_path in REMOVED_MODULE_FILES + REMOVED_PACKAGE_DIRECTORIES
            if (REPO_ROOT / relative_path).exists()
        ]
        self.assertEqual([], present, f"old-path shims reintroduced: {present}")

    def test_legacy_training_entries_stay_deleted(self):
        present = [
            relative_path
            for relative_path in REMOVED_TRAINING_ENTRY_FILES
            if (REPO_ROOT / relative_path).exists()
        ]
        self.assertEqual([], present, f"legacy training entries reintroduced: {present}")

    def test_repository_does_not_import_removed_modules(self):
        violations = []
        for source_root in SOURCE_ROOTS:
            for path in sorted(source_root.rglob("*.py")):
                for line_number, module in _imported_modules(path):
                    if _is_removed_module(module):
                        relative_path = path.relative_to(REPO_ROOT)
                        violations.append(f"{relative_path}:{line_number}: {module}")

        self.assertEqual(
            [],
            violations,
            "imports must use specforge.inference or specforge.training:\n"
            + "\n".join(violations),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
