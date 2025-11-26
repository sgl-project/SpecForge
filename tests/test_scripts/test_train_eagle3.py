import unittest
from pathlib import Path

from sglang.utils import execute_shell_command

CACHE_DIR = Path(__file__).parent.parent.parent.joinpath("cache")


class TestTrainEagle3(unittest.TestCase):

    def test_online_train_eagle3(self):
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

        with open(script_path, "w") as f:
            for line in script:
                f.write(line)

        # run training
        train_process = execute_shell_command(
            "bash examples/run_llama3.1_8b_eagle3_online.sh 2"
        )
        train_process.wait()
        self.assertEqual(train_process.returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
