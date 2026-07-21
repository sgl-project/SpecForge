import json
import tempfile
import unittest
from pathlib import Path

from tests.utils import (
    execute_shell_command,
    get_available_port,
    terminate_process_group,
    wait_for_server,
)

class TestRegenerateTrainData(unittest.TestCase):
    def test_regenerate_sharegpt(self):
        """Regenerate one local ShareGPT-format row through a live server."""
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        input_path = Path(temporary_directory.name) / "sharegpt_input.jsonl"
        output_path = Path(temporary_directory.name) / "sharegpt_output.jsonl"
        input_path.write_text(
            json.dumps(
                {
                    "id": "ci-sample",
                    "conversations": [
                        {"role": "user", "content": "Reply with a short greeting."},
                        {"role": "assistant", "content": "Hello!"},
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        port = get_available_port()
        sglang_process = execute_shell_command(
            f"""python3 -m sglang.launch_server \
    --model unsloth/Llama-3.2-1B-Instruct \
    --tp 1 \
    --cuda-graph-bs 4 \
    --dtype bfloat16 \
    --mem-frac=0.8 \
    --port {port}
        """,
            disable_proxy=True,
            enable_hf_mirror=False,
            sglang_use_modelscope=True,
            start_new_session=True,
        )
        try:
            wait_for_server(
                f"http://localhost:{port}",
                timeout=300,
                disable_proxy=True,
                process=sglang_process,
            )
            regeneration_process = execute_shell_command(
                f"""python scripts/regenerate_train_data.py \
    --model unsloth/Llama-3.2-1B-Instruct \
    --concurrency 1 \
    --max-tokens 128 \
    --server-address localhost:{port} \
    --temperature 0.8 \
    --input-file-path {input_path} \
    --output-file-path {output_path} \
    --num-samples 1
        """,
                disable_proxy=True,
                enable_hf_mirror=False,
            )
            regeneration_process.wait()
            self.assertEqual(regeneration_process.returncode, 0)
            regenerated_rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(regenerated_rows), 1)
            self.assertEqual(regenerated_rows[0]["status"], "success")
        finally:
            terminate_process_group(sglang_process)


if __name__ == "__main__":
    unittest.main(verbosity=2)
