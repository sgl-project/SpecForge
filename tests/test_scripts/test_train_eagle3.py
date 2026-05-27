import os
import shutil
import subprocess
import unittest
from pathlib import Path

from tests.utils import execute_shell_command

CACHE_DIR = Path(__file__).parent.parent.parent.joinpath("cache")
SOURCE_TARGET_MODEL = "nreHieW/Llama-3.1-8B-Instruct"
ONLINE_SCRIPT_PATH = Path(__file__).parent.parent.parent.joinpath(
    "examples", "run_llama3.1_8b_eagle3_online.sh"
)
OFFLINE_SCRIPT_PATH = Path(__file__).parent.parent.parent.joinpath(
    "examples", "run_llama3.1_8b_eagle3_offline.sh"
)
ONLINE_SCRIPT_TEMPLATE = ONLINE_SCRIPT_PATH.read_text()
OFFLINE_SCRIPT_TEMPLATE = OFFLINE_SCRIPT_PATH.read_text()


def replace_in_script(script_path: Path, pattern: str, replacement: str):
    script_path.write_text(script_path.read_text().replace(pattern, replacement))


def build_online_script() -> str:
    return ONLINE_SCRIPT_TEMPLATE.replace(
        "meta-llama/Llama-3.1-8B-Instruct",
        SOURCE_TARGET_MODEL,
    ).replace(
        "$ROOT_DIR/scripts/train_eagle3.py",
        "$ROOT_DIR/scripts/train_eagle3.py --max-num-steps 10",
    )


def build_offline_script() -> str:
    return (
        OFFLINE_SCRIPT_TEMPLATE.replace(
            "meta-llama/Llama-3.1-8B-Instruct",
            SOURCE_TARGET_MODEL,
        )
        .replace("--batch-size 32", "--batch-size 5")
        .replace(
            "scripts/prepare_hidden_states.py",
            "scripts/prepare_hidden_states.py --num-samples 10",
        )
        .replace(
            "$ROOT_DIR/scripts/train_eagle3.py",
            "$ROOT_DIR/scripts/train_eagle3.py --max-num-steps 2",
        )
    )


def print_gpu_memory_usage(label: str):
    print(f"\n===== GPU memory usage before {label} =====", flush=True)
    subprocess.run(["nvidia-smi"], check=False)
    print("===== End GPU memory usage =====\n", flush=True)


class TestTrainEagle3(unittest.TestCase):

    def setUp(self) -> None:
        self.addCleanup(ONLINE_SCRIPT_PATH.write_text, ONLINE_SCRIPT_TEMPLATE)
        self.addCleanup(OFFLINE_SCRIPT_PATH.write_text, OFFLINE_SCRIPT_TEMPLATE)

        # prepare data
        data_process = execute_shell_command(
            "python scripts/prepare_data.py --dataset sharegpt"
        )
        data_process.wait()

        ONLINE_SCRIPT_PATH.write_text(build_online_script())

    def test_online_train_eagle3_with_sglang_backend(self):
        print_gpu_memory_usage("test_online_train_eagle3_with_sglang_backend")

        # run training
        old_memory_debug = os.environ.get("SPECFORGE_CI_MEMORY_DEBUG")
        os.environ["SPECFORGE_CI_MEMORY_DEBUG"] = "1"
        try:
            train_process = execute_shell_command(
                "bash examples/run_llama3.1_8b_eagle3_online.sh 2"
            )
            train_process.wait()
        finally:
            if old_memory_debug is None:
                os.environ.pop("SPECFORGE_CI_MEMORY_DEBUG", None)
            else:
                os.environ["SPECFORGE_CI_MEMORY_DEBUG"] = old_memory_debug
        self.assertEqual(train_process.returncode, 0)

    def test_online_train_eagle3_with_hf_backend(self):
        # replace --target-model-backend sglang with --target-model-backend hf
        script_path = ONLINE_SCRIPT_PATH
        replace_in_script(
            script_path, "--target-model-backend sglang", "--target-model-backend hf"
        )

        # run training
        train_process = execute_shell_command(
            "bash examples/run_llama3.1_8b_eagle3_online.sh 2"
        )
        train_process.wait()
        self.assertEqual(train_process.returncode, 0)

    def test_online_train_eagle3_with_custom_backend(self):
        # replace --target-model-backend sglang with --target-model-backend custom
        script_path = ONLINE_SCRIPT_PATH
        replace_in_script(
            script_path,
            "--target-model-backend sglang",
            "--target-model-backend custom",
        )

        # run training
        train_process = execute_shell_command(
            "bash examples/run_llama3.1_8b_eagle3_online.sh 2"
        )
        train_process.wait()
        self.assertEqual(train_process.returncode, 0)

    def test_offline_train_eagle3(self):
        # remove the hidden states if they exist
        script_path = OFFLINE_SCRIPT_PATH
        script_path.write_text(build_offline_script())

        hidden_states_path = Path(__file__).parent.parent.parent.joinpath(
            "cache", "hidden_states", "sharegpt_train_Llama-3.1-8B-Instruct"
        )
        if hidden_states_path.exists():
            # delete the directory
            shutil.rmtree(hidden_states_path)

        training_process = execute_shell_command(
            "bash examples/run_llama3.1_8b_eagle3_offline.sh 2",
        )
        training_process.wait()
        self.assertEqual(training_process.returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
