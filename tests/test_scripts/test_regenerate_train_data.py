import os
import subprocess
import unittest
from pathlib import Path

from sglang.utils import execute_shell_command, wait_for_server

CACHE_DIR = Path(__file__).parent.parent.parent.joinpath("cache")


def execute_shell_command_without_proxy(command: str) -> subprocess.Popen:
    """
    Execute a shell command and return its process handle.
    """
    command = command.replace("\\\n", " ").replace("\\", " ")
    parts = command.split()
    env = os.environ.copy()
    env.pop("http_proxy", None)
    env.pop("https_proxy", None)
    env.pop("no_proxy", None)
    env.pop("HTTP_PROXY", None)
    env.pop("HTTPS_PROXY", None)
    env.pop("NO_PROXY", None)
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    return subprocess.Popen(parts, text=True, stderr=subprocess.STDOUT, env=env)


class TestRegenerateTrainData(unittest.TestCase):

    def test_regenerate_sharegpt(self):
        # prepare data
        data_process = execute_shell_command(
            "python scripts/prepare_data.py --dataset sharegpt"
        )
        data_process.wait()

        # launch sglang
        sglang_process = execute_shell_command_without_proxy(
            """python3 -m sglang.launch_server \
    --model unsloth/Llama-3.2-1B-Instruct \
    --tp 1 \
    --cuda-graph-bs 4 \
    --dtype bfloat16 \
    --mem-frac=0.8 \
    --port 30000
        """
        )
        wait_for_server(f"http://localhost:30000", timeout=60)

        regeneration_process = execute_shell_command_without_proxy(
            """python scripts/regenerate_train_data.py \
    --model unsloth/Llama-3.2-1B-Instruct \
    --concurrency 128 \
    --max-tokens 128 \
    --server-address localhost:30000 \
    --temperature 0.8 \
    --input-file-path ./cache/dataset/sharegpt_train.jsonl \
    --output-file-path ./cache/dataset/sharegpt_train_regen.jsonl \
    --num-samples 10
        """
        )
        regeneration_process.wait()
        self.assertEqual(regeneration_process.returncode, 0)
        self.assertTrue(
            CACHE_DIR.joinpath("dataset", "sharegpt_train_regen.jsonl").exists()
        )
        sglang_process.terminate()
        sglang_process.wait()


if __name__ == "__main__":
    unittest.main(verbosity=2)
