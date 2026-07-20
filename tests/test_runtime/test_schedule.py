import json
import tempfile
import unittest
from types import SimpleNamespace

from specforge.training.disaggregated import (
    _ONLINE_SCHEDULE_SUFFIX,
    _online_schedule_payload,
    _read_online_total_steps,
    _write_control,
)
from specforge.training.schedule import (
    resolve_online_total_steps,
    resolve_total_steps,
    validate_fixed_accumulation_plan,
)


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

    def test_max_steps_is_only_the_fallback_horizon(self):
        self.assertEqual(
            resolve_total_steps(
                total_steps=None,
                max_steps=5,
                num_samples=100,
                batch_size=2,
                accumulation_steps=1,
                num_epochs=3,
            ),
            5,
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

    def test_online_horizon_uses_complete_global_optimizer_windows(self):
        self.assertEqual(
            resolve_online_total_steps(
                num_prompts=25,
                prompt_epochs=3,
                dp_size=2,
                batch_size=3,
                accumulation_steps=2,
            ),
            6,
        )

    def test_online_horizon_rejects_a_plan_without_one_step(self):
        with self.assertRaisesRegex(ValueError, "produces no optimizer step"):
            resolve_online_total_steps(
                num_prompts=3,
                prompt_epochs=1,
                dp_size=4,
                batch_size=2,
                accumulation_steps=1,
            )

    def test_online_schedule_sidecar_round_trips_the_producer_horizon(self):
        cfg = SimpleNamespace(
            training=SimpleNamespace(
                num_epochs=3,
                seed=17,
                batch_size=2,
                accumulation_steps=4,
            ),
            deployment=SimpleNamespace(
                trainer=SimpleNamespace(nnodes=2, nproc_per_node=2)
            ),
        )
        payload = _online_schedule_payload(cfg, num_prompts=100)
        self.assertEqual(payload["total_steps"], 9)
        self.assertEqual(payload["prompt_seed"], 17)

        with tempfile.TemporaryDirectory() as directory:
            channel_path = f"{directory}/refs.jsonl"
            _write_control(
                channel_path + _ONLINE_SCHEDULE_SUFFIX,
                json.dumps(payload),
            )
            self.assertEqual(_read_online_total_steps(cfg, channel_path), 9)

            cfg.training.seed = 18
            with self.assertRaisesRegex(ValueError, "does not match"):
                _read_online_total_steps(cfg, channel_path)

    def test_fixed_plan_rejects_partial_accumulation_before_training(self):
        with self.assertRaisesRegex(
            ValueError, "ends with incomplete gradient accumulation"
        ):
            validate_fixed_accumulation_plan(
                num_samples=14,
                batch_size=2,
                accumulation_steps=3,
                num_epochs=1,
                max_steps=None,
            )

    def test_fixed_plan_allows_a_cap_before_the_partial_tail(self):
        validate_fixed_accumulation_plan(
            num_samples=14,
            batch_size=2,
            accumulation_steps=3,
            num_epochs=1,
            max_steps=2,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
