import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from specforge.training.disaggregated import (
    _ONLINE_SCHEDULE_SUFFIX,
    _online_flow_window,
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
    @staticmethod
    def _online_flow_config(*, high=1152, low=1024):
        return SimpleNamespace(
            runtime=SimpleNamespace(
                in_flight_high_watermark=high,
                in_flight_low_watermark=low,
                producer_lease=8,
            ),
            training=SimpleNamespace(batch_size=8, accumulation_steps=32),
            deployment=SimpleNamespace(
                trainer=SimpleNamespace(nnodes=1, nproc_per_node=4)
            ),
        )

    def test_online_flow_window_accepts_one_global_optimizer_window(self):
        self.assertEqual(
            _online_flow_window(self._online_flow_config()),
            (1152, 1024),
        )

    def test_online_flow_window_rejects_small_watermarks_before_data_build(self):
        cfg = self._online_flow_config(high=64, low=32)
        with self.assertRaisesRegex(ValueError, "high watermark 64.*quantum 1024"):
            _online_flow_window(cfg)

        cfg = self._online_flow_config(high=1152, low=32)
        with self.assertRaisesRegex(ValueError, "low watermark 32.*quantum 1024"):
            _online_flow_window(cfg)

    def test_online_flow_window_preserves_high_only_environment_override(self):
        cfg = self._online_flow_config(high=64, low=32)
        environment = {
            "DISAGG_IN_FLIGHT_HIGH_WATERMARK": "1024",
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            os.environ.pop("DISAGG_IN_FLIGHT_LOW_WATERMARK", None)
            self.assertEqual(_online_flow_window(cfg), (1024, None))

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
                prompt_seed=None,
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

    def test_online_prompt_seed_is_independent_from_model_seed(self):
        cfg = SimpleNamespace(
            training=SimpleNamespace(
                num_epochs=3,
                seed=17,
                prompt_seed=5,
                batch_size=2,
                accumulation_steps=4,
            ),
            deployment=SimpleNamespace(
                trainer=SimpleNamespace(nnodes=2, nproc_per_node=2)
            ),
        )
        payload = _online_schedule_payload(cfg, num_prompts=100)
        self.assertEqual(payload["prompt_seed"], 5)

        with tempfile.TemporaryDirectory() as directory:
            channel_path = f"{directory}/refs.jsonl"
            _write_control(
                channel_path + _ONLINE_SCHEDULE_SUFFIX,
                json.dumps(payload),
            )
            cfg.training.seed = 18
            self.assertEqual(_read_online_total_steps(cfg, channel_path), 9)

            cfg.training.prompt_seed = 6
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
