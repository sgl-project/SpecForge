import shutil
import unittest
from contextlib import contextmanager
from pathlib import Path

from tests.utils import execute_shell_command

CACHE_DIR = Path(__file__).parent.parent.parent.joinpath("cache")


@contextmanager
def replace_in_script(script_path: Path, *pattern_replacement_pairs):
    assert len(pattern_replacement_pairs) % 2 == 0
    with open(script_path, "r") as f:
        script = f.readlines()
    replaced_script = script
    for pattern, replacement in zip(
        pattern_replacement_pairs[::2], pattern_replacement_pairs[1::2]
    ):
        replaced_script = [
            line.replace(pattern, replacement) for line in replaced_script
        ]
    with open(script_path, "w") as f:
        for line in replaced_script:
            f.write(line)

    yield

    with open(script_path, "w") as f:
        for line in script:
            f.write(line)


class TestTrainEagle3(unittest.TestCase):
    def setUp(self) -> None:
        # prepare data
        data_process = execute_shell_command(
            "python scripts/prepare_data.py --dataset sharegpt"
        )
        data_process.wait()

        # modify the sccript to only train for 10 steps
        # add --max-num-steps 10 to the launch command
        script_path = Path(__file__).parent.parent.parent.joinpath(
            "examples", "run_llama3.1_8b_eagle3_online.sh"
        )
        with open(script_path, "r") as f:
            script = f.readlines()

        # remove empty lines
        script = [line for line in script if line.strip()]
        script[-1] = script[-1].rstrip() + " --max-num-steps 10"

        # replace meta-llama/Llama-3.1-8B-Instruct with unsloth/Llama-3.2-1B-Instruct
        # so that we don't need HF token for gated repo
        script = [
            line.replace(
                "meta-llama/Llama-3.1-8B-Instruct", "nreHieW/Llama-3.1-8B-Instruct"
            )
            for line in script
        ]

        # write the script back to the file
        with open(script_path, "w") as f:
            for line in script:
                f.write(line)

    def test_online_train_eagle3_with_sglang_backend(self):
        # run training
        train_process = execute_shell_command(
            "bash examples/run_llama3.1_8b_eagle3_online.sh 2"
        )
        train_process.wait()
        self.assertEqual(train_process.returncode, 0)

    def test_online_train_eagle3_with_hf_backend(self):
        # replace --target-model-backend sglang with --target-model-backend hf
        script_path = Path(__file__).parent.parent.parent.joinpath(
            "examples", "run_llama3.1_8b_eagle3_online.sh"
        )
        with replace_in_script(
            script_path, "--target-model-backend sglang", "--target-model-backend hf"
        ):
            # run training
            train_process = execute_shell_command(
                "bash examples/run_llama3.1_8b_eagle3_online.sh 2 2"
            )
            train_process.wait()
        self.assertEqual(train_process.returncode, 0)

    def test_online_train_eagle3_with_custom_backend(self):
        # replace --target-model-backend sglang with --target-model-backend custom
        script_path = Path(__file__).parent.parent.parent.joinpath(
            "examples", "run_llama3.1_8b_eagle3_online.sh"
        )
        with replace_in_script(
            script_path,
            "--target-model-backend sglang",
            "--target-model-backend custom",
        ):
            # run training
            train_process = execute_shell_command(
                "bash examples/run_llama3.1_8b_eagle3_online.sh 2 2"
            )
            train_process.wait()
        self.assertEqual(train_process.returncode, 0)

    def test_offline_train_eagle3(self):
        # remove the hidden states if they exist
        hidden_states_path = Path(__file__).parent.parent.parent.joinpath(
            "cache", "hidden_states", "sharegpt_train_Llama-3.1-8B-Instruct"
        )
        if hidden_states_path.exists():
            # delete the directory
            shutil.rmtree(hidden_states_path)

        script_path = Path(__file__).parent.parent.parent.joinpath(
            "examples", "run_llama3.1_8b_eagle3_offline.sh"
        )
        with replace_in_script(
            script_path,
            "meta-llama/Llama-3.1-8B-Instruct",
            "nreHieW/Llama-3.1-8B-Instruct",
            "--batch-size 32",
            "--batch-size 5",
            "scripts/prepare_hidden_states.py",
            "scripts/prepare_hidden_states.py --num-samples 10",
            "$ROOT_DIR/scripts/train_eagle3.py",
            "$ROOT_DIR/scripts/train_eagle3.py --max-num-steps 2",
        ):
            training_process = execute_shell_command(
                "bash examples/run_llama3.1_8b_eagle3_offline.sh 2",
            )
            training_process.wait()
        self.assertEqual(training_process.returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
