"""Bundle canonical model dependencies as ordinary HF relative-import modules."""

import ast
import importlib.util
from pathlib import Path


def bundle_code(directory, model_class):
    """Copy the source dependency closure, without requiring installed SpecForge.

    Flatten module names because Transformers' dynamic-module cache follows
    relative Python files, not package directories. Imports retain their graph;
    each artifact has its own module namespace and registries.
    """
    root = Path(__file__).resolve().parents[2]
    pending = [model_class.__module__, "specforge.modeling.draft.hf_config"]
    visited = set()
    filenames = {}

    def filename(module):
        name = module.replace(".", "_")
        previous = filenames.setdefault(name, module)
        if previous != module:
            raise ValueError(f"bundled filename collision: {previous} and {module}")
        return name

    def source_path(module):
        if not module.startswith("specforge."):
            raise ValueError(f"HF export only bundles specforge modules: {module}")
        path = root.joinpath(*module.split(".")[1:])
        return path / "__init__.py" if path.is_dir() else path.with_suffix(".py")

    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        path = source_path(module)
        if not path.is_file():
            raise ValueError(f"cannot bundle model dependency {module}")
        package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
        tree = ast.parse(path.read_text())
        if module == model_class.__module__ and not any(
            isinstance(node, ast.ClassDef) and node.name == model_class.__name__
            for node in tree.body
        ):
            raise ValueError(f"{model_class.__name__} is not defined in {module}")
        if module == "specforge.config.draft.codec.v1":
            for statement in tree.body:
                if (
                    isinstance(statement, ast.Assign)
                    and isinstance(statement.targets[0], ast.Name)
                    and statement.targets[0].id == "_BUILTIN_PACKAGE"
                ):
                    statement.value = ast.Name(id="__package__", ctx=ast.Load())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(
                a.name.startswith("specforge") for a in node.names
            ):
                raise ValueError(f"unsupported package import in {module}")
            if not isinstance(node, ast.ImportFrom):
                continue
            target = node.module or ""
            if node.level:
                target = importlib.util.resolve_name("." * node.level + target, package)
            if target == "specforge":
                raise ValueError(f"use explicit module imports in {module}")
            if not target.startswith("specforge."):
                continue
            if node.module is None:
                raise ValueError(f"use explicit module imports in {module}")
            pending.append(target)
            node.module, node.level = filename(target), 1
        (directory / f"{filename(module)}.py").write_text(ast.unparse(tree) + "\n")
    (directory / "draft_model.py").write_text(
        "from .specforge_modeling_draft_hf_config import SpecForgeDraftConfig\n"
        f"from .{filename(model_class.__module__)} import {model_class.__name__}\n"
        "SpecForgeDraftConfig.__module__ = __name__\n"
        f"{model_class.__name__}.__module__ = __name__\n"
    )
    return {
        "AutoConfig": "draft_model.SpecForgeDraftConfig",
        "AutoModel": f"draft_model.{model_class.__name__}",
        "AutoModelForCausalLM": f"draft_model.{model_class.__name__}",
    }
