import importlib.util
import unittest
from pathlib import Path

path = (
    Path(__file__).resolve().parents[2] / ".deployment/l20n/eval_dflash_acceptance.py"
)
spec = importlib.util.spec_from_file_location("acceptance_eval", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AcceptanceAggregationTest(unittest.TestCase):
    def test_dspark_width_includes_anchor(self):
        for width in (8, 16):
            self.assertEqual(
                module.speculative_server_args("DSPARK", 16, width),
                [
                    "--speculative-algorithm",
                    "DSPARK",
                    "--speculative-dspark-block-size",
                    "16",
                ],
            )
            self.assertEqual(
                module.speculative_server_args("DSPARK", width),
                [
                    "--speculative-algorithm",
                    "DSPARK",
                    "--speculative-dspark-block-size",
                    str(width - 1),
                ],
            )
            self.assertEqual(
                module.speculative_server_args("DFLASH", width),
                [
                    "--speculative-algorithm",
                    "DFLASH",
                    "--speculative-dflash-block-size",
                    str(width),
                ],
            )

    def test_verify_budget_includes_one_prefill_token(self):
        module.validate_verify_budget(
            [{"completion_tokens": 169, "verify_count": 21}], 8
        )
        for tokens, verifies in [(170, 21), (0, 21), (1, 0)]:
            with self.assertRaises(ValueError):
                module.validate_verify_budget(
                    [{"completion_tokens": tokens, "verify_count": verifies}], 8
                )

    def test_macro_and_micro_differ(self):
        records = [
            {
                "completion_tokens": 10,
                "verify_count": 2,
                "accept_length": 5.0,
                "finish_reason": {},
            },
            {
                "completion_tokens": 9,
                "verify_count": 3,
                "accept_length": 3.0,
                "finish_reason": {"type": "length"},
            },
        ]
        result = module.aggregate(records)
        self.assertEqual(result["macro_accept_length"], 4.0)
        self.assertEqual(result["micro_accept_length"], 3.8)
        self.assertEqual(result["length_limited_requests"], 1)

    def test_missing_or_inconsistent_counters_fail(self):
        for count, reported in [(0, 4.0), (2, 4.0)]:
            with self.assertRaises(ValueError):
                module.aggregate(
                    [
                        {
                            "completion_tokens": 10,
                            "verify_count": count,
                            "accept_length": reported,
                        }
                    ]
                )


if __name__ == "__main__":
    unittest.main()
