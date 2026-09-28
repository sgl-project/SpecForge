from array import array
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

from specforge.data.loss_mask import has_consecutive_supervised_tokens
from specforge.data.prompt_builder import prepare_prompt_tasks


class TestIndexedPrompts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.source = self.root / 'prompts.jsonl'
        rows = [{'input_ids': list(range(i, i + 7)), 'loss_mask': [0, 0, 1, 1, 1, 1, 1]}
                for i in range(20)]
        lines = [(json.dumps(row) + '\n').encode() for row in rows]
        data = b''.join(lines)
        self.source.write_bytes(data)
        offsets = array('Q', [0])
        for line in lines:
            offsets.append(offsets[-1] + len(line))
        if sys.byteorder != 'little':
            offsets.byteswap()
        packed = offsets.tobytes()
        (self.root / 'offsets.u64').write_bytes(packed)
        self.manifest = self.root / 'index.json'
        self.manifest.write_text(json.dumps({
            'schema_version': 1, 'max_length': 6, 'min_loss_tokens': 2,
            'loss_mask_filter': 'has_consecutive_supervised_tokens',
            'records': len(rows), 'source_bytes': len(data),
            'source_sha256': hashlib.sha256(data).hexdigest(),
            'offsets_file': 'offsets.u64', 'offsets_sha256': hashlib.sha256(packed).hexdigest(),
        }))

    def tearDown(self):
        self.directory.cleanup()

    def prepare(self, indexed=True, **kwargs):
        options = dict(tokenizer=None, chat_template=None, max_length=6,
                       is_preformatted=False, train_only_last_turn=False, cache_dir=None,
                       cache_key=None, num_proc=1, min_loss_tokens=2,
                       loss_mask_filter=has_consecutive_supervised_tokens)
        options.update(kwargs)
        return prepare_prompt_tasks(self.source, index_path=str(self.manifest) if indexed else None,
                                    **options)

    def test_rows_and_three_sampler_epochs_match_eager_path(self):
        indexed, eager = self.prepare(), self.prepare(False)
        self.assertEqual(len(indexed), len(eager))
        for epoch in range(3):
            order = list(range(len(eager)))
            random.Random(42 + epoch).shuffle(order)
            self.assertEqual([indexed[i] for i in order], [eager[i] for i in order])
        self.assertEqual(indexed[-1], eager[-1])
        self.assertEqual(indexed[2:7:2], eager[2:7:2])
        self.assertEqual(len(self.prepare(max_prompts=3)), 3)

    def test_source_digest_and_live_mutation_fail_closed(self):
        indexed = self.prepare()
        self.source.write_bytes(self.source.read_bytes().replace(b'19', b'18', 1))
        with self.assertRaisesRegex(ValueError, 'changed'):
            indexed[0]
        with self.assertRaisesRegex(ValueError, 'digest'):
            self.prepare()

    def test_index_digest_and_admission_contract_are_enforced(self):
        with self.assertRaisesRegex(ValueError, 'contract'):
            self.prepare(max_length=5)
        (self.root / 'offsets.u64').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'digest'):
            self.prepare()

    def test_invalid_admitted_row_is_rejected_not_silently_dropped(self):
        # Valid bytes and offsets do not excuse a false admission assertion.
        self.source.write_bytes(self.source.read_bytes().replace(b'[0, 0, 1, 1, 1, 1, 1]', b'[0, 0, 0, 0, 0, 0, 0]', 1))
        manifest = json.loads(self.manifest.read_text())
        manifest['source_sha256'] = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.manifest.write_text(json.dumps(manifest))
        indexed = self.prepare()
        with self.assertRaisesRegex(ValueError, 'frozen admission'):
            indexed[0]


if __name__ == '__main__':
    unittest.main()
