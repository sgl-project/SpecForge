import shutil
import unittest
from pathlib import Path

from sglang.utils import execute_shell_command


class TestPrepareHiddenStates(unittest.TestCase):

    def test_prepare_sharegpt_hidden_states(self):
        # prepare data
        sharegpt_process = execute_shell_command(
            "python scripts/prepare_data.py --dataset sharegpt"
        )
        sharegpt_process.wait()

        # remove the hidden states if they exist
        hidden_states_path = Path(__file__).parent.parent.parent.joinpath(
            "cache", "hidden_states"
        )
        if hidden_states_path.exists():
            # delete the directory
            shutil.rmtree(hidden_states_path)

        # generate hidden states
        hidden_states_generation_process = execute_shell_command(
            """torchrun \
    --standalone \
    --nproc_per_node 2 \
    scripts/prepare_hidden_states.py \
    --target-model-path unsloth/Llama-3.2-1B-Instruct \
    --enable-aux-hidden-states \
    --data-path ./cache/dataset/sharegpt_train.jsonl \
    --output-path ./cache/hidden_states/sharegpt_train_Llama-3.2-1B-Instruct \
    --chat-template llama3 \
    --max-length 4096 \
    --tp-size 1 \
    --batch-size 2 \
    --num-samples 10
        """
        )
        hidden_states_generation_process.wait()
        self.assertEqual(hidden_states_generation_process.returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
