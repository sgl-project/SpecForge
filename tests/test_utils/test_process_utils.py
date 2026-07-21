import signal
import subprocess
import unittest
from unittest import mock

from tests.utils import (
    execute_shell_command,
    terminate_process_group,
    wait_for_server,
)


class ProcessUtilsTest(unittest.TestCase):
    @mock.patch("tests.utils.subprocess.Popen")
    def test_execute_shell_command_can_start_a_new_session(self, popen):
        """Pass the requested session isolation to subprocess.Popen."""
        execute_shell_command("python -V", start_new_session=True)

        self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_wait_for_server_fails_when_process_exits(self):
        """Report an early server exit without waiting for the timeout."""
        process = mock.Mock()
        process.poll.return_value = 7
        process.returncode = 7

        with self.assertRaisesRegex(RuntimeError, "exited with code 7"):
            wait_for_server("http://localhost:1", process=process)

    @mock.patch("tests.utils.os.killpg")
    def test_terminate_process_group_sends_sigterm(self, killpg):
        """Terminate the full session and wait for its parent process."""
        process = mock.Mock(pid=1234)

        terminate_process_group(process)

        killpg.assert_called_once_with(1234, signal.SIGTERM)
        process.wait.assert_called_once_with(timeout=30)

    @mock.patch("tests.utils.os.killpg")
    def test_terminate_process_group_escalates_after_timeout(self, killpg):
        """Escalate to SIGKILL when graceful shutdown times out."""
        process = mock.Mock(pid=1234)
        process.wait.side_effect = [subprocess.TimeoutExpired("server", 30), None]

        terminate_process_group(process)

        self.assertEqual(
            killpg.call_args_list,
            [mock.call(1234, signal.SIGTERM), mock.call(1234, signal.SIGKILL)],
        )
        self.assertEqual(process.wait.call_count, 2)


if __name__ == "__main__":
    unittest.main()
