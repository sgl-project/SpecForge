from __future__ import annotations

import os
import unittest
from unittest.mock import call, patch

from specforge.training.disaggregated import (
    _hold_mooncake_producer,
    _optional_timeout_s,
    _wait_for,
)


class DisaggregatedTimeoutTest(unittest.TestCase):
    def test_wait_is_unbounded_by_default(self):
        with (
            patch(
                "specforge.training.disaggregated.os.path.exists",
                side_effect=(False, True),
            ),
            patch("specforge.training.disaggregated.time.sleep") as sleep,
            patch("specforge.training.disaggregated.time.monotonic") as monotonic,
        ):
            _wait_for("/control/ready")

        sleep.assert_called_once_with(0.25)
        monotonic.assert_not_called()

    def test_explicit_wait_timeout_is_terminal(self):
        with (
            patch(
                "specforge.training.disaggregated.os.path.exists",
                return_value=False,
            ),
            patch("specforge.training.disaggregated.time.sleep"),
            patch(
                "specforge.training.disaggregated.time.monotonic",
                side_effect=(10.0, 12.0),
            ),
        ):
            with self.assertRaisesRegex(TimeoutError, "/control/ready"):
                _wait_for("/control/ready", timeout_s=1.0)

    def test_mooncake_hold_has_no_implicit_one_hour_limit(self):
        environment = {"DISAGG_BACKEND": "mooncake"}
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("specforge.training.disaggregated._wait_for") as wait_for,
        ):
            _hold_mooncake_producer("/control/manifest.json")

        self.assertEqual(
            call(
                "/control/manifest.json.consumed",
                timeout_s=None,
                failure_path="/control/manifest.json.consumer_failed",
            ),
            wait_for.call_args,
        )

    def test_configured_timeouts_must_be_positive_numbers(self):
        with patch.dict(
            os.environ, {"DISAGG_PEER_WAIT_TIMEOUT": "12.5"}, clear=True
        ):
            self.assertEqual(12.5, _optional_timeout_s("DISAGG_PEER_WAIT_TIMEOUT"))

        for value in ("0", "-1", "not-a-number"):
            with (
                self.subTest(value=value),
                patch.dict(
                    os.environ,
                    {"DISAGG_PEER_WAIT_TIMEOUT": value},
                    clear=True,
                ),
                self.assertRaisesRegex(ValueError, "DISAGG_PEER_WAIT_TIMEOUT"),
            ):
                _optional_timeout_s("DISAGG_PEER_WAIT_TIMEOUT")


if __name__ == "__main__":
    unittest.main()
