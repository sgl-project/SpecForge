"""Cleanup must report its own failures without replacing the original failure."""
import unittest
from types import SimpleNamespace
from unittest import mock

from specforge.training.assembly import TrainingRun
from specforge.training.trainer import Trainer


def raising(error):
    def fail(*_args, **_kwargs):
        raise error
    return fail


class TestPrimaryShutdownError(unittest.TestCase):
    def test_training_run_keeps_prefetch_timeout_when_store_close_refuses(self):
        primary = RuntimeError("prefetch drain timed out with live read intent")
        close = RuntimeError("cannot close capture transport with live readers")
        cleanup = mock.Mock(side_effect=close)
        run = TrainingRun(trainer=SimpleNamespace(fit=raising(primary)), on_finally=cleanup)
        with self.assertRaises(RuntimeError) as caught:
            run.run()
        self.assertIs(caught.exception, primary)
        cleanup.assert_called_once_with()
        self.assertIn(str(close), "\n".join(primary.__notes__))

    def test_failure_hook_and_store_close_are_both_retained(self):
        primary = ValueError("training failed first")
        hook = RuntimeError("failed marker unavailable")
        close = RuntimeError("readers have not drained")
        run = TrainingRun(trainer=SimpleNamespace(fit=raising(primary)),
                          on_failure=raising(hook), on_finally=raising(close))
        with self.assertRaises(ValueError) as caught:
            run.run()
        self.assertIs(caught.exception, primary)
        for error in (hook, close):
            self.assertIn(str(error), "\n".join(primary.__notes__))

    def test_interruption_is_not_replaced_by_close_failure(self):
        primary = KeyboardInterrupt("interrupted")
        run = TrainingRun(trainer=SimpleNamespace(fit=raising(primary)),
                          on_finally=raising(RuntimeError("still reading")))
        with self.assertRaises(KeyboardInterrupt) as caught:
            run.run()
        self.assertIs(caught.exception, primary)
        self.assertIn("still reading", "\n".join(primary.__notes__))

    def test_cleanup_error_without_primary_still_fails(self):
        cleanup = RuntimeError("store close failed")
        run = TrainingRun(trainer=SimpleNamespace(fit=lambda: 2), on_finally=raising(cleanup))
        with self.assertRaises(RuntimeError) as caught:
            run.run()
        self.assertIs(caught.exception, cleanup)

    def test_trainer_failure_hook_does_not_replace_primary_or_cleanup_notes(self):
        primary = RuntimeError("optimizer failure")
        trainer = object.__new__(Trainer)
        trainer._loader = SimpleNamespace(close=mock.Mock(side_effect=RuntimeError("prefetch timeout")))
        trainer._controller = SimpleNamespace(fit=raising(primary), close_profiler=lambda: None)
        trainer._fit_context = None
        trainer._on_fit_success = None
        trainer._on_fit_failure = raising(RuntimeError("consumer failed marker failed"))
        trainer._on_fit_finally = raising(RuntimeError("distributor close failed"))
        trainer._logger = None
        with self.assertRaises(RuntimeError) as caught:
            trainer.fit()
        self.assertIs(caught.exception, primary)
        trainer._loader.close.assert_called_once_with()
        for message in ("prefetch timeout", "consumer failed marker failed", "distributor close failed"):
            self.assertIn(message, "\n".join(primary.__notes__))


if __name__ == '__main__':
    unittest.main()
