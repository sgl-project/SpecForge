"""Tests for parser fallback, preprocessing kwargs, vocab mapping, and 1D loss masks."""

import contextlib
import json
import re
import tempfile
import types
import unittest
import warnings
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import torch
from datasets import Dataset, Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast

from specforge.data.loss_mask import has_consecutive_supervised_tokens
from specforge.data.parse import GeneralParser, ThinkingParser
from specforge.data.preprocessing import (
    build_eagle3_dataset,
    preprocess_conversations,
    process_token_dict_to_mappings,
)
from specforge.data.prompt_builder import prepare_prompt_tasks
from specforge.data.template import TEMPLATE_REGISTRY, ChatTemplate
from tests.test_data.test_parser_normalization import DummyTokenizer

_PLAIN = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]


def _offline_hf_tokenizer(chat_template=None):
    vocab = {"<unk>": 0, "<bos>": 1, **{chr(c): c - 30 for c in range(32, 127)}}
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Split(pattern="", behavior="isolated")
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<bos>",
        pad_token="<unk>",
        chat_template=chat_template,
    )


def _template(parser_type="general"):
    return ChatTemplate(
        assistant_header="<assistant>",
        user_header="<user>",
        system_prompt=None,
        end_of_turn_token="</eot>",
        parser_type=parser_type,
    )


def _decode_supervised(tokenizer, input_ids, loss_mask):
    return "".join(
        tokenizer.decode([tid])
        for tid, keep in zip(input_ids.tolist(), loss_mask.tolist())
        if keep
    )


@contextlib.contextmanager
def _assert_warns(test_case, pattern):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    test_case.assertTrue(
        any(re.search(pattern, str(w.message)) for w in caught),
        f"no warning matching {pattern!r} in {[str(w.message) for w in caught]}",
    )


class GeneralParserFallbackTest(unittest.TestCase):
    def test_template_errors_raise_instead_of_falling_back(self):
        class _FailingDuck(DummyTokenizer):
            def __init__(self, exc):
                super().__init__()
                self._exc = exc

            def apply_chat_template(self, messages, **kwargs):
                raise self._exc

        class _FailingWithTemplate(_FailingDuck):
            chat_template = "{{ messages }}"

        for cls in (_FailingWithTemplate, _FailingDuck):
            for exc in (ValueError("jinja syntax"), TypeError("bad kwarg")):
                with self.subTest(cls=cls.__name__, exc=type(exc)):
                    with self.assertRaises(type(exc)):
                        GeneralParser(cls(exc), _template()).parse(_PLAIN, 256)

        broken_hf = _offline_hf_tokenizer("{{ (messages | length) + 'bad' }}")
        with self.assertRaises(TypeError):
            GeneralParser(broken_hf, _template()).parse(_PLAIN, 256)

    def test_custom_hf_renderers_bypass_fallback(self):
        class _FailingApplyHF(PreTrainedTokenizerFast):
            def apply_chat_template(self, messages, **kwargs):
                raise ValueError("custom renderer failure")

        failing = _offline_hf_tokenizer()
        failing.__class__ = _FailingApplyHF
        with self.assertRaisesRegex(ValueError, "custom renderer failure"):
            GeneralParser(failing, _template()).parse(_PLAIN, 256)

        for bind_method in (False, True):
            with self.subTest(bind_method=bind_method):
                tok = _offline_hf_tokenizer()

                def _custom(self_or_msgs, *args, **kwargs):
                    msgs = args[0] if bind_method else self_or_msgs
                    return "".join(
                        f"<{m['role']}>CUSTOM:{m['content']}</eot>" for m in msgs
                    )

                tok.apply_chat_template = (
                    types.MethodType(_custom, tok) if bind_method else _custom
                )
                ids, mask = GeneralParser(tok, _template()).parse(_PLAIN, 256)
                self.assertEqual(_decode_supervised(tok, ids, mask), "CUSTOM:ok</eot>")

        class _CustomResolverHF(PreTrainedTokenizerFast):
            def get_chat_template(self, chat_template=None, tools=None):
                return "{% for m in messages %}<{{ m['role'] }}>RESOLVED:{{ m['content'] }}</eot>{% endfor %}"

        resolver = _offline_hf_tokenizer()
        resolver.__class__ = _CustomResolverHF
        ids, mask = GeneralParser(resolver, _template()).parse(_PLAIN, 256)
        self.assertEqual(_decode_supervised(resolver, ids, mask), "RESOLVED:ok</eot>")

    def test_fallback_renders_plain_chat_and_rejects_unsupported_features(self):
        tok = _offline_hf_tokenizer()
        general = GeneralParser(tok, _template())
        thinking = ThinkingParser(tok, _template(parser_type="thinking"))

        with _assert_warns(self, "fallback rendering"):
            ids, mask = general.parse(_PLAIN, 256)
        self.assertEqual(_decode_supervised(tok, ids, mask), "ok</eot>")

        tool_def = [{"type": "function", "function": {"name": "w"}}]
        tc = [
            {"id": "c1", "type": "function", "function": {"name": "w", "arguments": {}}}
        ]
        unsupported = (
            ("tools", _PLAIN, tool_def),
            (
                "tool_calls",
                [_PLAIN[0], {"role": "assistant", "content": "", "tool_calls": tc}],
                None,
            ),
            ("tool_role", [*_PLAIN, {"role": "tool", "content": "sunny"}], None),
            (
                "reasoning",
                [
                    _PLAIN[0],
                    {"role": "assistant", "content": "4", "reasoning_content": "add"},
                ],
                None,
            ),
            (
                "system_tools",
                [{"role": "system", "content": "s", "tools": tool_def}, *_PLAIN],
                None,
            ),
        )
        for label, conv, tools in unsupported:
            for name, parser in (("general", general), ("thinking", thinking)):
                with self.subTest(case=label, parser=name), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with self.assertRaises(ValueError):
                        parser.parse(conv, 256, tool=tools)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ids, mask = general.parse(
                [
                    _PLAIN[0],
                    {
                        "role": "assistant",
                        "content": "ok",
                        "reasoning_content": "",
                        "tools": [],
                    },
                ],
                256,
            )
        self.assertEqual(_decode_supervised(tok, ids, mask), "ok</eot>")

        with _assert_warns(self, "Skipping conversation"):
            self.assertIsNone(general.parse([], 256))


class PreprocessingTest(unittest.TestCase):
    def _build(
        self,
        dataset,
        tokenizer,
        *,
        template_name="test-general",
        max_length=128,
        **kwargs,
    ):
        with patch.dict(TEMPLATE_REGISTRY.templates):
            if template_name not in TEMPLATE_REGISTRY.templates:
                TEMPLATE_REGISTRY.register(template_name, _template())
            return build_eagle3_dataset(
                dataset=dataset,
                tokenizer=tokenizer,
                chat_template=template_name,
                max_length=max_length,
                num_proc=None,
                **kwargs,
            )

    def test_preprocess_conversations_tools_and_kwargs_validation(self):
        convs = [_PLAIN, _PLAIN, _PLAIN]
        results = preprocess_conversations(DummyTokenizer(), convs, _template(), 128)
        self.assertEqual(
            [len(results[k]) for k in ("input_ids", "loss_mask", "attention_mask")],
            [3, 3, 3],
        )
        with self.assertRaisesRegex(ValueError, "tools"):
            preprocess_conversations(
                DummyTokenizer(), convs[:2], _template(), 128, tools=[[]]
            )
        with self.assertRaisesRegex(ValueError, "enable_thinking"):
            preprocess_conversations(
                DummyTokenizer(), convs[:2], _template(), 128, enable_thinking=[True]
            )

    def test_build_eagle3_dataset_drops_extra_columns_and_forwards_known_kwargs(self):
        tok = DummyTokenizer()
        ds = Dataset.from_list(
            [
                {
                    "id": "0",
                    "conversations": _PLAIN,
                    "metadata": {"k": "v"},
                    "tokenize": True,
                    "enable_thinking": True,
                    "reasoning_effort": "low",
                }
            ]
        )
        self.assertEqual(len(self._build(ds, tok)), 1)
        self.assertEqual(
            tok.template_kwargs, {"enable_thinking": True, "reasoning_effort": "low"}
        )

    def test_build_eagle3_dataset_forwards_deepseek_v4_and_muse_columns(self):
        class _UnicodeTokenizer(DummyTokenizer):
            def _ids(self, text, max_length=None):
                return torch.tensor(
                    [ord(c) for c in text][:max_length], dtype=torch.long
                )

        tok = _UnicodeTokenizer()
        with_ctx = self._build(
            Dataset.from_list(
                [{"conversations": _PLAIN, "context": _PLAIN, "unused": "x"}]
            ),
            tok,
            template_name="deepseek-v4",
            max_length=512,
        )
        no_ctx = self._build(
            Dataset.from_list([{"conversations": _PLAIN, "unused": "x"}]),
            tok,
            template_name="deepseek-v4",
            max_length=512,
        )
        self.assertNotIn(
            "<｜begin▁of▁sentence｜>",
            "".join(chr(i) for i in with_ctx[0]["input_ids"][0].tolist()),
        )
        self.assertTrue(
            "".join(chr(i) for i in no_ctx[0]["input_ids"][0].tolist()).startswith(
                "<｜begin▁of▁sentence｜>"
            )
        )

        muse_jinja = (
            Path(__file__).parent / "fixtures" / "muse_spark_chat_template.jinja"
        ).read_text(encoding="utf-8")
        muse_tok = _offline_hf_tokenizer(muse_jinja)
        muse_ds = Dataset.from_list(
            [
                {
                    "conversations": _PLAIN,
                    "current_date": "2020-01-01",
                    "cutoff_date": "2019-12-31",
                    "unused": "x",
                }
            ]
        )
        processed = self._build(
            muse_ds, muse_tok, template_name="muse", max_length=4096
        )
        decoded = "".join(
            muse_tok.decode([tid]) for tid in processed[0]["input_ids"][0].tolist()
        )
        self.assertIn("Current date: 2020-01-01.", decoded)
        self.assertIn("Knowledge cutoff: 2019-12-31.", decoded)

    def test_build_eagle3_dataset_does_not_decode_unused_lazy_image_column(self):
        ds = Dataset.from_list(
            [
                {
                    "conversations": _PLAIN,
                    "image": {"path": "/nonexistent/missing.png", "bytes": None},
                }
            ]
        ).cast_column("image", Image())
        with self.assertRaises((FileNotFoundError, ImportError)):
            _ = ds[0]["image"]
        self.assertEqual(len(self._build(ds, DummyTokenizer())), 1)

    def test_build_eagle3_dataset_partial_cache_args_warn_instead_of_crashing(self):
        ds = Dataset.from_list([{"conversations": _PLAIN}])
        with tempfile.TemporaryDirectory() as tmp_dir:
            for cache_dir, cache_key in ((tmp_dir, None), (None, "only-key")):
                with self.subTest(cache_dir=cache_dir, cache_key=cache_key):
                    with _assert_warns(
                        self, "cache_dir and cache_key must be provided together"
                    ):
                        processed = self._build(
                            ds,
                            DummyTokenizer(),
                            cache_dir=cache_dir,
                            cache_key=cache_key,
                        )
                    self.assertEqual(len(processed), 1)

    def test_loss_mask_filter_receives_1d_sequence_on_raw_and_pretokenized_paths(self):
        observed = []

        def supervised_last(mask):
            observed.append(mask)
            self.assertTrue(not mask or isinstance(mask[0], int))
            return bool(mask) and bool(mask[-1]) and sum(mask) >= 2

        ds = Dataset.from_list(
            [{"conversations": _PLAIN}, {"conversations": [_PLAIN[0]]}]
        )
        self.assertEqual(
            len(self._build(ds, DummyTokenizer(), loss_mask_filter=supervised_last)), 1
        )
        self.assertEqual(len(observed), 2)

        observed.clear()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "prompts.jsonl"
            path.write_text(
                json.dumps({"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1]})
                + "\n"
                + json.dumps({"input_ids": [4, 5, 6], "loss_mask": [0, 0, 0]}),
                encoding="utf-8",
            )
            prompts = prepare_prompt_tasks(
                str(path),
                None,
                chat_template=None,
                max_length=16,
                is_preformatted=False,
                train_only_last_turn=False,
                cache_dir=None,
                cache_key=None,
                num_proc=None,
                min_loss_tokens=0,
                loss_mask_filter=supervised_last,
            )
        self.assertEqual(len(prompts), 1)
        self.assertEqual(observed, [[0, 1, 1], [0, 0, 0]])

    def test_has_consecutive_supervised_tokens_shapes(self):
        self.assertTrue(has_consecutive_supervised_tokens([0, 1, 1]))
        self.assertFalse(has_consecutive_supervised_tokens([1, 0, 1]))
        self.assertTrue(has_consecutive_supervised_tokens([[0, 1, 1]]))
        self.assertTrue(has_consecutive_supervised_tokens(torch.tensor([[0, 1, 1]])))
        self.assertFalse(has_consecutive_supervised_tokens(torch.tensor([1, 0, 1])))
        for bad in ([[0, 1], [1, 1]], torch.tensor([[0, 1], [1, 1]]), [[[1, 1]]]):
            with self.assertRaises(ValueError):
                has_consecutive_supervised_tokens(bad)


class VocabMappingTest(unittest.TestCase):
    def test_validates_vocab_sizes_token_ids_and_counts(self):
        bad_cases = (
            (Counter({0: 1}), 10, 5),
            (Counter({0: 1}), 0, 5),
            (Counter({0: 1}), 2, 0),
            (Counter({-1: 5, 1: 2}), 2, 10),
            (Counter({10: 5, 1: 2}), 2, 10),
            ({3: -1}, 1, 4),
            ({0: True}, 1, 4),
            ({1: 1.5}, 1, 4),
            ({2: "4"}, 1, 4),
            ({True: 1}, 1, 4),
        )
        for counts, draft_size, target_size in bad_cases:
            with self.subTest(
                counts=counts, draft_size=draft_size, target_size=target_size
            ):
                with self.assertRaises(ValueError):
                    process_token_dict_to_mappings(counts, draft_size, target_size)

    def test_mapping_bijection_ranking_and_caller_immutability(self):
        target_vocab_size, draft_vocab_size = 4096, 1024
        counts = Counter(
            {target_vocab_size - 1 - i: i + 1 for i in range(draft_vocab_size - 1)}
        )
        original = dict(counts)
        d2t, t2d = process_token_dict_to_mappings(
            counts, draft_vocab_size, target_vocab_size
        )
        self.assertEqual(dict(counts), original)
        self.assertEqual((d2t.shape, d2t.dtype), ((draft_vocab_size,), torch.long))
        self.assertEqual((t2d.shape, t2d.dtype), ((target_vocab_size,), torch.bool))
        selected = torch.nonzero(t2d, as_tuple=False).squeeze(-1)
        self.assertTrue(
            torch.equal(
                torch.arange(draft_vocab_size, dtype=torch.long) + d2t, selected
            )
        )
        self.assertEqual(selected.tolist(), sorted([0, *original.keys()]))

        d2t_ranked, t2d_ranked = process_token_dict_to_mappings(
            Counter({1: 5, 3: 50, 5: 1, 7: 20}), 2, 10
        )
        self.assertEqual(
            torch.nonzero(t2d_ranked, as_tuple=False).squeeze(-1).tolist(), [3, 7]
        )
        self.assertEqual(
            (torch.arange(2, dtype=torch.long) + d2t_ranked).tolist(), [3, 7]
        )


if __name__ == "__main__":
    unittest.main()
