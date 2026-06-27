"""Integration tests for the offline compact-teacher path in scripts/train_eagle3.py.

Exercises the real ``build_parser``, ``run_forward``, and
``validate_compact_teacher_args`` with lightweight doubles. Self-skips if torch or the
training script cannot be imported here (e.g. an sglang version skew); see
BL-20260613-specforge-import-isolation.
"""

import importlib.util
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

try:
    import torch

    _HAS_TORCH = True
except Exception as exc:  # pragma: no cover - environment dependent
    torch = None
    _HAS_TORCH = False
    _TORCH_ERROR = exc


def _load_train_module():
    spec = importlib.util.spec_from_file_location(
        "train_eagle3_under_test", ROOT / "scripts" / "train_eagle3.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if _HAS_TORCH:
    try:
        train_mod = _load_train_module()
        IMPORT_ERROR = None
    except Exception as exc:  # pragma: no cover - environment dependent
        train_mod = None
        IMPORT_ERROR = exc
else:
    train_mod = None
    IMPORT_ERROR = _TORCH_ERROR

_SKIP = train_mod is None
_SKIP_MSG = f"train_eagle3 import unavailable: {IMPORT_ERROR}"
_HAS_CUDA = bool(_HAS_TORCH and torch.cuda.is_available())


if _HAS_TORCH:

    class _TargetHeadDouble(torch.nn.Module):
        def __init__(self, hidden_size, vocab_size):
            super().__init__()
            self.fc = torch.nn.Linear(hidden_size, vocab_size, bias=False)
            self.forward_calls = 0

        def preprocess(self, input_ids, target, loss_mask):
            return input_ids, target, loss_mask[..., None]

        def forward(self, x):
            self.forward_calls += 1
            return self.fc(x)

    class _RecordingEagle3Model:
        def __init__(self):
            self.kwargs = None

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            dummy = [torch.tensor(0.0)]
            return (dummy, dummy, dummy, dummy, dummy, dummy, dummy)

    def _mapping(vocab=64, draft=16):
        sel = torch.sort(
            torch.randperm(vocab, generator=torch.Generator().manual_seed(0))[:draft]
        ).values
        t2d = torch.zeros(vocab, dtype=torch.bool)
        t2d[sel] = True
        d2t = sel - torch.arange(draft)
        return t2d, d2t


@unittest.skipIf(_SKIP, _SKIP_MSG)
class TestBuildParser(unittest.TestCase):
    def _base_args(self):
        return [
            "--target-model-path",
            "m",
            "--train-data-path",
            "d",
            "--output-dir",
            "o",
        ]

    def test_compact_teacher_defaults_off(self):
        args = train_mod.build_parser().parse_args(self._base_args())
        self.assertFalse(args.compact_teacher)
        self.assertIsNone(args.compact_teacher_chunk_size)

    def test_compact_teacher_flags_parse(self):
        args = train_mod.build_parser().parse_args(
            self._base_args()
            + ["--compact-teacher", "--compact-teacher-chunk-size", "8"]
        )
        self.assertTrue(args.compact_teacher)
        self.assertEqual(args.compact_teacher_chunk_size, 8)


@unittest.skipIf(_SKIP, _SKIP_MSG)
class TestValidateCompactTeacherArgs(unittest.TestCase):
    def _config(self, vocab=64, draft=16):
        t2d, d2t = _mapping(vocab, draft)
        return types.SimpleNamespace(
            args=types.SimpleNamespace(is_vlm=False, compact_teacher_chunk_size=128),
            target_model=_TargetHeadDouble(8, vocab),
            draft_model=types.SimpleNamespace(t2d=t2d, d2t=d2t),
            cfg=types.SimpleNamespace(vocab_size=vocab, draft_vocab_size=draft),
        )

    def _validate(self, c, is_online=False):
        train_mod.validate_compact_teacher_args(
            c.args, is_online, c.target_model, c.draft_model, c.cfg
        )

    def test_accepts_valid_offline(self):
        self._validate(self._config())

    def test_rejects_online(self):
        with self.assertRaises(ValueError):
            self._validate(self._config(), is_online=True)

    def test_rejects_vlm(self):
        c = self._config()
        c.args.is_vlm = True
        with self.assertRaises(ValueError):
            self._validate(c)

    def test_rejects_nonpositive_chunk(self):
        c = self._config()
        c.args.compact_teacher_chunk_size = 0
        with self.assertRaises(ValueError):
            self._validate(c)

    def test_rejects_bad_mapping(self):
        c = self._config()
        c.draft_model.d2t = c.draft_model.d2t.clone()
        c.draft_model.d2t[0] = c.draft_model.d2t[0] + 1
        with self.assertRaises(ValueError):
            self._validate(c)


@unittest.skipIf(_SKIP, _SKIP_MSG)
@unittest.skipUnless(_HAS_CUDA, "CUDA required for run_forward smoke")
class TestOfflineRunForwardSmoke(unittest.TestCase):
    def _data(self):
        return {
            "input_ids": torch.zeros(1, 4, dtype=torch.long),
            "target": torch.randn(1, 4, 8),
            "loss_mask": torch.ones(1, 4, dtype=torch.int),
            "attention_mask": torch.ones(1, 4, dtype=torch.int),
            "hidden_state": torch.randn(1, 4, 8),
        }

    def _args(self, compact):
        return types.SimpleNamespace(
            is_vlm=False,
            target_model_backend="hf",
            compact_teacher=compact,
            compact_teacher_chunk_size=None,
            shard_target_output=False,
        )

    def test_compact_skips_target_forward(self):
        target_model = _TargetHeadDouble(8, 32).cuda()
        eagle3_model = _RecordingEagle3Model()
        train_mod.run_forward(
            self._args(compact=True), eagle3_model, self._data(), target_model, False
        )
        self.assertEqual(target_model.forward_calls, 0)
        kw = eagle3_model.kwargs
        self.assertIsNone(kw["target"])
        self.assertIn("target_hidden_for_compact", kw)
        self.assertIs(kw["target_head_weight"], target_model.fc.weight)
        from specforge.core.compact_teacher import DEFAULT_VOCAB_CHUNK_SIZE

        self.assertEqual(kw["compact_teacher_chunk_size"], DEFAULT_VOCAB_CHUNK_SIZE)

    def test_legacy_calls_target_forward(self):
        target_model = _TargetHeadDouble(8, 32).cuda()
        eagle3_model = _RecordingEagle3Model()
        train_mod.run_forward(
            self._args(compact=False), eagle3_model, self._data(), target_model, False
        )
        self.assertEqual(target_model.forward_calls, 1)
        kw = eagle3_model.kwargs
        self.assertEqual(tuple(kw["target"].shape), (1, 4, 32))
        self.assertNotIn("target_hidden_for_compact", kw)


@unittest.skipIf(_SKIP, _SKIP_MSG)
class TestExportFilter(unittest.TestCase):
    def test_keeps_draft_drops_embed_and_teacher(self):
        state_dict = {
            "draft_model.lm_head.weight": 1,
            "draft_model.t2d": 2,
            "draft_model.d2t": 3,
            "draft_model.midlayer.norm.weight": 4,
            "draft_model.embed_tokens.weight": 5,
            "target_model.fc.weight": 6,
            "verifier_lm_head.weight": 7,
        }
        out = train_mod.filter_draft_state_dict(state_dict)
        self.assertIn("lm_head.weight", out)
        self.assertIn("t2d", out)
        self.assertIn("d2t", out)
        self.assertNotIn("embed_tokens.weight", out)  # embeddings excluded by design
        self.assertFalse(any("fc" in k for k in out))  # no target head leaks
        self.assertFalse(any(k.startswith("target_model") for k in out))
        self.assertTrue(all("draft_model." not in k for k in out))  # prefix stripped


if __name__ == "__main__":
    unittest.main(verbosity=2)
