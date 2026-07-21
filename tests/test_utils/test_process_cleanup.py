import os
import signal
import sys
import tempfile
import time
import unittest

from tests.utils import execute_shell_command, terminate_process_trees


@unittest.skipUnless(
    hasattr(os, "killpg") and hasattr(os, "getpgid"),
    "requires POSIX process groups",
)
class ProcessCleanupTest(unittest.TestCase):
    def test_terminates_descendants_after_the_group_leader_exits(self):
        child_code = """
import os
import subprocess
import sys
import time

grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    stream.write(f"{os.getpid()} {grandchild.pid}")
time.sleep(60)
"""
        with tempfile.TemporaryDirectory() as root:
            script = os.path.join(root, "child.py")
            marker = os.path.join(root, "pids.txt")
            with open(script, "w", encoding="utf-8") as stream:
                stream.write(child_code)

            process = execute_shell_command(
                f"{sys.executable} {script} {marker}",
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and not os.path.exists(marker):
                    if process.poll() is not None:
                        break
                    time.sleep(0.02)
                self.assertTrue(os.path.exists(marker), "child never became ready")
                with open(marker, encoding="utf-8") as stream:
                    leader_pid, grandchild_pid = map(int, stream.read().split())

                self.assertEqual(leader_pid, process.pid)
                self.assertEqual(os.getpgid(leader_pid), leader_pid)
                self.assertEqual(os.getpgid(grandchild_pid), leader_pid)

                # Reproduce an SGLang launcher exiting before its GPU workers.
                os.kill(leader_pid, signal.SIGTERM)
                process.wait(timeout=5)
                terminate_process_trees(process, grace_s=1)

                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    try:
                        os.killpg(leader_pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.02)
                else:
                    self.fail(f"process group {leader_pid} survived cleanup")
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
