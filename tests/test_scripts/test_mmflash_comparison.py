import unittest

from examples.mmflash.benchmark import compare_runs, measure_stream


def event(count, verifies=0):
    meta = {"completion_tokens": count}
    if verifies:
        meta["spec_verify_ct"] = verifies
    return {"meta_info": meta, "output_ids": list(range(count)), "text": "result"}


class MMFlashComparisonTest(unittest.TestCase):
    def test_decode_clock_excludes_prefill_and_first_block(self):
        row = measure_stream(
            iter(
                [
                    (0.1, event(0)),
                    (4.0, event(3)),
                    (5.0, event(8, 4)),
                    (9.0, event(8, 4)),
                ]
            )
        )
        self.assertEqual((row["decode_tokens"], row["decode_seconds"]), (5, 1))
        self.assertEqual(row["accept_length"], 2)

    def test_requires_two_token_emissions_and_verification(self):
        for events in [[(1, event(1)), (2, event(3))], [(1, event(8, 4))]]:
            with self.assertRaises(ValueError):
                measure_stream(iter(events))

    def test_no_silent_unpaired_or_mismatching_outputs(self):
        row = dict(
            measure_stream(iter([(1, event(1)), (2, event(8, 4))])),
            id="a",
            request_sha256="same",
        )
        with self.assertRaises(ValueError):
            compare_runs([row], [dict(row, id="b")])
        with self.assertRaises(ValueError):
            compare_runs([row], [dict(row, request_sha256="changed")])
        report = compare_runs([row], [dict(row, output_ids=[99])])
        self.assertEqual(report["different_greedy_outputs"], ["a"])
