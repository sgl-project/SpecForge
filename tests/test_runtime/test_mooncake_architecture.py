"""Dependency-light guards for Mooncake's single raw-tensor wire path."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

STORE_PATH = (
    Path(__file__).resolve().parents[2]
    / "specforge/runtime/data_plane/mooncake_store.py"
)


def _parse_store():
    tree = ast.parse(STORE_PATH.read_text(), filename=str(STORE_PATH))
    store_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MooncakeFeatureStore"
    )
    return tree, store_class


class TestMooncakeArchitecture(unittest.TestCase):
    def test_store_has_one_raw_tensor_transport(self):
        _, store_class = _parse_store()
        methods = {
            node.name for node in store_class.body if isinstance(node, ast.FunctionDef)
        }
        self.assertTrue(
            {"_key", "_store_put", "_get_pickle", "_get_zero_copy"}.isdisjoint(
                methods
            )
        )

        init = next(
            node
            for node in store_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        parameters = {
            arg.arg for arg in init.args.args + init.args.kwonlyargs
        }
        self.assertNotIn("zero_copy", parameters)
        self.assertFalse(
            any(
                isinstance(node, ast.Attribute) and node.attr == "_zero_copy"
                for node in ast.walk(store_class)
            )
        )

        serialization = []
        store_calls = set()
        for node in ast.walk(store_class):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            receiver = node.func.value
            if isinstance(receiver, ast.Name) and receiver.id == "torch":
                if node.func.attr in {"save", "load"}:
                    serialization.append(node.lineno)
            if (
                isinstance(receiver, ast.Attribute)
                and isinstance(receiver.value, ast.Name)
                and receiver.value.id == "self"
                and receiver.attr == "_store"
            ):
                store_calls.add(node.func.attr)
        self.assertEqual([], serialization)
        self.assertTrue({"put_from", "get_into"}.issubset(store_calls))
        self.assertTrue({"put", "get"}.isdisjoint(store_calls))

    def test_constructor_requires_the_raw_api(self):
        tree, store_class = _parse_store()
        validator = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_require_store_api"
        )
        required_assignment = next(
            node
            for node in validator.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "required"
                for target in node.targets
            )
        )
        self.assertEqual(
            {"is_exist", "remove", "put_from", "get_into"},
            set(ast.literal_eval(required_assignment.value)),
        )

        init = next(
            node
            for node in store_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        calls = {
            node.func.id
            for node in ast.walk(init)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("_require_store_api", calls)


if __name__ == "__main__":
    unittest.main(verbosity=2)
