"""Readiness probes allow slow healthy servers without overrunning startup."""

import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock
from urllib.error import HTTPError

from pydantic import ValidationError

from specforge.config import Config
from specforge.config.schema import (
    ManagedLocalCaptureServerConfig,
    ManagedLocalMooncakeConfig,
)
from specforge.launch_plan import (
    CommandSpec,
    ReadinessSpec,
    ServiceSpec,
    _http_ready,
    _readiness_satisfied,
    _wait_for_service,
)
from tests.test_runtime.test_launch_plan import (
    CAPTURE_CONTRACT,
    _managed_config,
    build_launch_plan,
)


class ServiceReadinessTest(unittest.TestCase):
    def test_http_readiness_accepts_healthy_response_after_one_second(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                time.sleep(1.2)
                try:
                    self.send_response(200)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                except ConnectionError:
                    pass  # An old one-second probe has already disconnected.

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            readiness = ReadinessSpec(
                "http", f"http://127.0.0.1:{server.server_port}/health", 10
            )
            with mock.patch.dict("os.environ", {"no_proxy": "127.0.0.1"}):
                self.assertTrue(_http_ready(readiness))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_probe_timeout_defaults_and_validation(self):
        for cls, kwargs in (
            (ManagedLocalMooncakeConfig, {}),
            (
                ManagedLocalCaptureServerConfig,
                dict(port=31000, cuda_visible_devices=["0"]),
            ),
        ):
            with self.subTest(config=cls.__name__):
                self.assertEqual(cls(**kwargs).probe_timeout_s, 5.0)
                for invalid in (0, -1, float("inf"), float("nan")):
                    with (
                        self.subTest(value=invalid),
                        self.assertRaises(ValidationError),
                    ):
                        cls(**kwargs, probe_timeout_s=invalid)

    def test_plan_preserves_per_service_probe_timeouts(self):
        cfg = _managed_config("/fresh/health-probe-test")
        raw = cfg.model_dump()
        managed = raw["deployment"]["disaggregated"]["managed_local"]
        managed["mooncake"]["probe_timeout_s"] = 2.5
        managed["capture_servers"][0]["probe_timeout_s"] = 7.5
        cfg = Config.model_validate(raw)
        with mock.patch(
            "specforge.training.capture_contract.resolve_server_capture_contract",
            return_value=CAPTURE_CONTRACT,
        ):
            plan = build_launch_plan(cfg, config_path="train.yaml", env={})
        self.assertEqual(
            [s.readiness.probe_timeout_s for s in plan.services], [2.5, 7.5]
        )
        self.assertEqual(
            [s.as_dict()["readiness"]["probe_timeout_s"] for s in plan.services],
            [2.5, 7.5],
        )

    def test_http_probe_uses_configured_timeout(self):
        readiness = ReadinessSpec(
            "http", "http://capture/health", 60, probe_timeout_s=8
        )
        response = mock.MagicMock()
        response.__enter__.return_value.status = 200
        with mock.patch(
            "specforge.launch_plan.urllib_request.urlopen", return_value=response
        ) as open_url:
            self.assertTrue(_http_ready(readiness))
        open_url.assert_called_once_with(readiness.url, timeout=8)

    def test_http_errors_and_socket_timeouts_are_not_ready(self):
        readiness = ReadinessSpec("http", "http://capture/health", 60)
        for error in (
            HTTPError(readiness.url, 503, "unhealthy", None, None),
            TimeoutError(),
        ):
            with (
                self.subTest(error=error),
                mock.patch(
                    "specforge.launch_plan.urllib_request.urlopen", side_effect=error
                ),
            ):
                self.assertFalse(_http_ready(readiness))

    def test_probe_is_capped_by_remaining_startup_time(self):
        readiness = ReadinessSpec("http", "http://capture/health", 60)
        with (
            mock.patch("specforge.launch_plan.time.monotonic", return_value=10),
            mock.patch("specforge.launch_plan._http_ready", return_value=True) as http,
        ):
            self.assertTrue(_readiness_satisfied(readiness, deadline=12))
        http.assert_called_once_with(readiness, timeout_s=2)

    def test_expired_deadline_does_not_issue_a_probe(self):
        readiness = ReadinessSpec("http", "http://capture/health", 60)
        with (
            mock.patch("specforge.launch_plan.time.monotonic", return_value=12),
            mock.patch("specforge.launch_plan._http_ready") as http,
        ):
            self.assertFalse(_readiness_satisfied(readiness, deadline=12))
        http.assert_not_called()

    def test_late_healthy_response_does_not_pass_startup_deadline(self):
        readiness = ReadinessSpec("http", "http://capture/health", 60)
        with (
            mock.patch("specforge.launch_plan.time.monotonic", side_effect=[10, 12]),
            mock.patch("specforge.launch_plan._http_ready", return_value=True),
        ):
            self.assertFalse(_readiness_satisfied(readiness, deadline=12))

    def test_mooncake_http_and_tcp_share_probe_budget(self):
        readiness = ReadinessSpec(
            "mooncake",
            "http://master/metadata",
            60,
            tcp_host="127.0.0.1",
            tcp_port=35551,
        )
        with (
            mock.patch(
                "specforge.launch_plan.time.monotonic", side_effect=[10, 11.5, 11.75]
            ),
            mock.patch("specforge.launch_plan._http_ready", return_value=True) as http,
            mock.patch("specforge.launch_plan.socket.create_connection") as tcp,
        ):
            self.assertTrue(_readiness_satisfied(readiness, deadline=12))
        http.assert_called_once_with(readiness, timeout_s=2)
        tcp.assert_called_once_with(("127.0.0.1", 35551), timeout=0.5)

    def test_mooncake_skips_tcp_after_http_exhausts_probe_budget(self):
        readiness = ReadinessSpec(
            "mooncake",
            "http://master/metadata",
            60,
            tcp_host="127.0.0.1",
            tcp_port=35551,
        )
        with (
            mock.patch("specforge.launch_plan.time.monotonic", side_effect=[10, 15]),
            mock.patch("specforge.launch_plan._http_ready", return_value=True),
            mock.patch("specforge.launch_plan.socket.create_connection") as tcp,
        ):
            self.assertFalse(_readiness_satisfied(readiness, deadline=60))
        tcp.assert_not_called()

    def test_unresponsive_service_stops_at_total_startup_deadline(self):
        now = [0.0]
        service = ServiceSpec(
            CommandSpec("capture", ("capture",)),
            ReadinessSpec("http", "http://capture/health", 2),
            "capture.log",
            1,
        )
        process = mock.Mock()
        process.poll.return_value = None

        def timed_out(_readiness, *, timeout_s):
            now[0] += timeout_s
            return False

        with (
            mock.patch(
                "specforge.launch_plan.time.monotonic", side_effect=lambda: now[0]
            ),
            mock.patch("specforge.launch_plan._http_ready", side_effect=timed_out),
            mock.patch("specforge.launch_plan.time.sleep") as sleep,
        ):
            with self.assertRaisesRegex(TimeoutError, "after 2.0s"):
                _wait_for_service(service, process, [(service, process)])
        self.assertEqual(now[0], 2)
        sleep.assert_not_called()

    def test_poll_sleep_is_capped_by_remaining_startup_time(self):
        now = [0.0]
        service = ServiceSpec(
            CommandSpec("capture", ("capture",)),
            ReadinessSpec("http", "http://capture/health", 1),
            "capture.log",
            1,
        )
        process = mock.Mock()
        process.poll.return_value = None

        def not_ready(_readiness, *, deadline):
            now[0] = 0.9
            return False

        def advance(duration):
            now[0] += duration

        with (
            mock.patch(
                "specforge.launch_plan.time.monotonic", side_effect=lambda: now[0]
            ),
            mock.patch(
                "specforge.launch_plan._readiness_satisfied", side_effect=not_ready
            ),
            mock.patch(
                "specforge.launch_plan.time.sleep", side_effect=advance
            ) as sleep,
        ):
            with self.assertRaises(TimeoutError):
                _wait_for_service(service, process, [(service, process)])
        self.assertAlmostEqual(sleep.call_args.args[0], 0.1)
        self.assertEqual(now[0], 1)
