import gc
import os
import shutil
import subprocess
import unittest
from pathlib import Path

from tests.utils import execute_shell_command

CACHE_DIR = Path(__file__).parent.parent.parent.joinpath("cache")
SOURCE_TARGET_MODEL = "nreHieW/Llama-3.1-8B-Instruct"
RANDOM_TARGET_MODEL_DIR = CACHE_DIR.joinpath("models", "random_Llama-3.1-8B-Instruct")
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


def prepare_random_target_model():
    has_config = RANDOM_TARGET_MODEL_DIR.joinpath("config.json").exists()
    has_weights = (
        RANDOM_TARGET_MODEL_DIR.joinpath("model.safetensors").exists()
        or RANDOM_TARGET_MODEL_DIR.joinpath("pytorch_model.bin").exists()
        or len(list(RANDOM_TARGET_MODEL_DIR.glob("*.index.json"))) > 0
    )
    if has_config and has_weights:
        return

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tmp_dir = RANDOM_TARGET_MODEL_DIR.with_name(f"{RANDOM_TARGET_MODEL_DIR.name}.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(SOURCE_TARGET_MODEL)
    tokenizer.save_pretrained(tmp_dir)

    config = AutoConfig.from_pretrained(SOURCE_TARGET_MODEL)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    model.save_pretrained(
        tmp_dir,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    del model
    gc.collect()

    if RANDOM_TARGET_MODEL_DIR.exists():
        shutil.rmtree(RANDOM_TARGET_MODEL_DIR)
    tmp_dir.rename(RANDOM_TARGET_MODEL_DIR)


def build_online_script() -> str:
    return (
        ONLINE_SCRIPT_TEMPLATE.replace(
            "meta-llama/Llama-3.1-8B-Instruct",
            str(RANDOM_TARGET_MODEL_DIR),
        )
        .replace("--max-length 4096", "--max-length 512")
        .replace(
            "$ROOT_DIR/scripts/train_eagle3.py",
            "$ROOT_DIR/scripts/train_eagle3.py --max-num-steps 10",
        )
    )


def build_offline_script() -> str:
    return (
        OFFLINE_SCRIPT_TEMPLATE.replace(
            "meta-llama/Llama-3.1-8B-Instruct",
            str(RANDOM_TARGET_MODEL_DIR),
        )
        .replace("--max-length 4096", "--max-length 512")
        .replace("--batch-size 32", "--batch-size 1")
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
        prepare_random_target_model()

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
        print_gpu_memory_usage("test_online_train_eagle3_with_hf_backend")
        train_process = execute_shell_command(
            "bash examples/run_llama3.1_8b_eagle3_online.sh 1"
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
            "bash examples/run_llama3.1_8b_eagle3_online.sh 1"
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

        print_gpu_memory_usage("test_offline_train_eagle3")
        training_process = execute_shell_command(
            "bash examples/run_llama3.1_8b_eagle3_offline.sh 2",
        )
        training_process.wait()
        self.assertEqual(training_process.returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
