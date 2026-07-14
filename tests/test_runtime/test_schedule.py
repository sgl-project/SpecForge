import unittest

from specforge.training.schedule import resolve_total_steps


class TestResolveTotalSteps(unittest.TestCase):
    def test_finite_data_horizon_counts_optimizer_steps(self):
        self.assertEqual(
            resolve_total_steps(
                total_steps=None,
                max_steps=None,
                num_samples=25,
                batch_size=4,
                accumulation_steps=3,
                num_epochs=2,
            ),
            4,
        )

    def test_explicit_horizon_wins(self):
        self.assertEqual(
            resolve_total_steps(
                total_steps=17,
                max_steps=5,
                num_samples=None,
                batch_size=1,
                accumulation_steps=1,
                num_epochs=1,
            ),
            17,
        )

    def test_stream_requires_explicit_horizon(self):
        with self.assertRaisesRegex(ValueError, "streaming training run"):
            resolve_total_steps(
                total_steps=None,
                max_steps=None,
                num_samples=None,
                batch_size=1,
                accumulation_steps=1,
                num_epochs=1,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
