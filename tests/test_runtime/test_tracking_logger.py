# coding=utf-8
"""Backend-neutral experiment tracking at the Trainer logger seam."""

import unittest
from types import SimpleNamespace
from unittest import mock

from specforge.tracker import _public_config
from specforge.training.tracking import (
    TrackerLogger,
    create_tracker_logger,
    scalar_metrics,
    training_metric_names,
)
from specforge.training.trainer import Trainer


class _TensorDouble:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def numel(self):
        return len(self.values)

    def item(self):
        return self.values[0]

    def reshape(self, _):
        return self

    def tolist(self):
        return list(self.values)


class _Tracker:
    def __init__(self):
        self.logged = []
        self.close_calls = 0

    def log(self, metrics, step=None):
        self.logged.append((dict(metrics), step))

    def close(self):
        self.close_calls += 1


class TrackingLoggerTest(unittest.TestCase):
    @staticmethod
    def _trainer(*, fit_error=None):
        trainer = Trainer.__new__(Trainer)
        trainer._controller = mock.Mock(last_checkpoint_step=1)
        if fit_error is None:
            trainer._controller.fit.return_value = 1
        else:
            trainer._controller.fit.side_effect = fit_error
        trainer._loader = object()
        trainer._fit_context = None
        trainer._on_fit_success = None
        trainer._on_fit_failure = None
        trainer._on_fit_finally = None
        trainer._logger = mock.Mock()
        return trainer

    def test_tracker_metadata_redacts_credentials(self):
        config = _public_config(
            SimpleNamespace(
                wandb_key="secret",
                auth_token="also-secret",
                wandb_project="specforge",
            )
        )
        self.assertEqual(config["wandb_key"], "<redacted>")
        self.assertEqual(config["auth_token"], "<redacted>")
        self.assertEqual(config["wandb_project"], "specforge")

    def test_normalizes_scalars_and_expands_vectors(self):
        self.assertEqual(
            scalar_metrics(
                {
                    "loss": 1,
                    "acceptance": _TensorDouble([0.25]),
                    "ploss": _TensorDouble([2.0, 3.0]),
                    "ignored": "text",
                }
            ),
            {
                "loss": 1.0,
                "acceptance": 0.25,
                "ploss/0": 2.0,
                "ploss/1": 3.0,
            },
        )

    def test_callable_fans_out_to_console_and_tracker(self):
        tracker = _Tracker()
        console = mock.Mock()
        logger = TrackerLogger(tracker, console_logger=console)

        logger({"loss": 1.5}, 7)

        self.assertEqual(tracker.logged, [({"train/loss": 1.5}, 7)])
        console.assert_called_once_with({"train/loss": 1.5}, 7)

    def test_training_namespace_keeps_eval_metrics_unchanged(self):
        self.assertEqual(
            training_metric_names({"loss": 1.0, "eval/loss": 2.0}),
            {"train/loss": 1.0, "eval/loss": 2.0},
        )

    def test_close_is_idempotent_and_logging_after_close_fails(self):
        tracker = _Tracker()
        logger = TrackerLogger(tracker)
        logger.close()
        logger.close()
        self.assertEqual(tracker.close_calls, 1)
        with self.assertRaisesRegex(RuntimeError, "after"):
            logger({"loss": 1.0}, 1)

    def test_factory_adapts_existing_tracker_registry(self):
        tracker = _Tracker()
        tracker_class = mock.Mock(return_value=tracker)
        with mock.patch(
            "specforge.tracker.get_tracker_class", return_value=tracker_class
        ) as make:
            logger = create_tracker_logger(
                mock.Mock(report_to="tensorboard"), "/tmp/output"
            )
        make.assert_called_once_with("tensorboard")
        tracker_class.validate_args.assert_called_once()
        logger({"loss": 2.0}, 3)
        self.assertEqual(tracker.logged, [({"train/loss": 2.0}, 3)])

    def test_noop_tracker_does_not_require_initialized_distributed(self):
        logger = create_tracker_logger(
            SimpleNamespace(report_to="none"), "/tmp/output"
        )
        logger({"loss": 1.0}, 1)
        logger.close()

    def test_trainer_closes_tracker_after_a_successful_fit(self):
        trainer = self._trainer()

        self.assertEqual(trainer.fit(), 1)

        trainer._logger.close.assert_called_once_with()

    def test_trainer_closes_tracker_after_a_failed_fit(self):
        error = RuntimeError("training failed")
        trainer = self._trainer(fit_error=error)

        with self.assertRaises(RuntimeError) as raised:
            trainer.fit()

        self.assertIs(raised.exception, error)
        trainer._logger.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main(verbosity=2)
