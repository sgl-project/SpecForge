import ast
import types
import unittest
from pathlib import Path


def _load_resolver():
    source_path = (
        Path(__file__).parents[2] / "examples" / "disagg" / "run_disagg_dflash.py"
    )
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    resolver = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_resolve_mask_token"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[resolver], type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    return namespace["_resolve_mask_token"]


class _Tokenizer:
    def __init__(self, mask_token_id=None):
        self.mask_token_id = mask_token_id
        self.added = False

    def add_special_tokens(self, _tokens):
        self.added = True
        self.mask_token_id = 777


class DisaggDFlashMaskTokenTest(unittest.TestCase):
    resolve = staticmethod(_load_resolver())

    @staticmethod
    def _args(mask_token_id=None):
        return types.SimpleNamespace(mask_token_id=mask_token_id)

    @staticmethod
    def _draft(mask_token_id=None):
        config = types.SimpleNamespace(dflash_config={"mask_token_id": mask_token_id})
        return types.SimpleNamespace(config=config)

    def test_cli_override_wins(self):
        tokenizer = _Tokenizer(999)
        self.assertEqual(self.resolve(self._args(12), tokenizer, self._draft(16)), 12)

    def test_draft_config_wins_over_tokenizer(self):
        tokenizer = _Tokenizer(999)
        self.assertEqual(self.resolve(self._args(), tokenizer, self._draft(16)), 16)

    def test_zero_config_id_is_respected(self):
        tokenizer = _Tokenizer(999)
        self.assertEqual(self.resolve(self._args(), tokenizer, self._draft(0)), 0)

    def test_tokenizer_then_new_token_are_fallbacks(self):
        tokenizer = _Tokenizer(999)
        self.assertEqual(self.resolve(self._args(), tokenizer, self._draft()), 999)
        self.assertFalse(tokenizer.added)

        tokenizer = _Tokenizer()
        self.assertEqual(self.resolve(self._args(), tokenizer, self._draft()), 777)
        self.assertTrue(tokenizer.added)


if __name__ == "__main__":
    unittest.main(verbosity=2)
