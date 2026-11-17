"""Observations stay bounded and cannot turn an I/O failure into a native failure."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from specforge.runtime.data_plane.pool_diagnostics import Diagnostics


class TestDiagnosticTrace(unittest.TestCase):
    def test_disabled_writes_nothing(self):
        d = Diagnostics()
        with d.batch(9), d.component('x').span('work'):
            self.assertIsNone(d.begin_block({'namespace': 'teacherA'}))
        self.assertEqual(d.snapshot()['metrics'], {})

    def test_full_identity_distinguishes_teachers_and_step(self):
        with tempfile.TemporaryDirectory() as path:
            d = Diagnostics(path)
            a = {'namespace': 'A', 'load_epoch': 'epoch-A', 'feature': 'aux',
                 'logical_key': 'A/digest/aux', 'physical_key': 'physical1', 'nbytes': 8}
            b = {**a, 'namespace': 'B', 'load_epoch': 'epoch-B', 'logical_key': 'B/digest/aux'}
            with d.batch(9):
                x = d.begin_block(a); d.event('pin', x); d.event('copy_complete', x)
                d.event('finish', x, {'pin_released': True})
                d.begin_block(a); d.begin_block(b)
            d.flush()
            rows = [json.loads(x) for x in d.path.read_text().splitlines()]
            keys = [x for x in rows if isinstance(x, dict) and x['type'] == 'key']
            self.assertEqual(len(keys), 2)
            self.assertEqual(keys[0]['load_epoch'], 'epoch-A')
            self.assertTrue(all(x[4] == 9 for x in rows if isinstance(x, list)))
            self.assertEqual(d.snapshot()['trace_bytes_written'], d.path.stat().st_size)
            self.assertTrue(d.snapshot()['trace_complete'])

    def test_bound_exhaustion_is_explicit_not_exception(self):
        with tempfile.TemporaryDirectory() as path:
            d = Diagnostics(path, trace_limit_bytes=2048, trace_limit_keys=2, buffer_bytes=256)
            for i in range(50):
                d.begin_block({'namespace': 'x', 'logical_key': str(i)})
            d.flush()
            self.assertFalse(d.snapshot()['trace_complete'])
            self.assertLessEqual(d.path.stat().st_size, 2048)
            self.assertLessEqual(d.snapshot()['trace_keys'], 2)

    def test_write_error_never_masks_body_exception(self):
        with tempfile.TemporaryDirectory() as path:
            d = Diagnostics(path, buffer_bytes=1)
            d.file.close()
            with self.assertRaisesRegex(ValueError, 'native original'):
                with d.component('store').span('get'):
                    d.begin_block({'namespace': 'x'})
                    raise ValueError('native original')
            self.assertFalse(d.snapshot()['trace_complete'])
            self.assertEqual(d.snapshot()['metrics']['store.get_calls'], 1)

    def test_no_per_event_fsync(self):
        with tempfile.TemporaryDirectory() as path, patch('os.fsync', side_effect=AssertionError('fsync')):
            d = Diagnostics(path, buffer_bytes=128)
            for i in range(20):
                d.event('lookup', d.begin_block({'key': i}))
            d.flush()
            self.assertTrue(d.snapshot()['trace_complete'])


if __name__ == '__main__':
    unittest.main()
