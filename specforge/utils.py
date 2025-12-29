import json
import logging
import os
import re
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, Shard, distribute_tensor
from transformers import AutoConfig, PretrainedConfig

logger = logging.getLogger(__name__)


@contextmanager
def rank_0_priority():
    rank = dist.get_rank()

    if rank == 0:
        yield
        dist.barrier()
    else:
        dist.barrier()
        yield


@contextmanager
def default_torch_dtype(dtype: torch.dtype):
    current_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(current_dtype)


@torch.no_grad()
def padding(tensor, left=True):
    zeropadding = torch.zeros_like(tensor[:, -1:])
    if left:
        tensor = torch.cat((zeropadding, tensor[:, :-1]), dim=1)
    else:
        tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
    return tensor


def load_config_from_file(config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)

    return PretrainedConfig.from_dict(config)


def print_with_rank(message):
    if dist.is_available() and dist.is_initialized():
        logger.info(f"rank {dist.get_rank()}: {message}")
    else:
        logger.info(f"non-distributed: {message}")


def print_args_with_dots(args):
    if dist.get_rank() == 0:
        args_dict = vars(args)
        max_key_length = max(len(key) for key in args_dict.keys())
        total_width = 50

        print("\n -----------【args】-----------")
        for key, value in args_dict.items():
            key_str = f"{key:<{max_key_length}}"
            value_str = str(value)
            dot_count = total_width - len(key_str) - len(value_str)
            dot_fill = "·" * dot_count
            print(f"{key_str} {dot_fill} {value_str}")


def print_on_rank0(message):
    if dist.get_rank() == 0:
        logger.info(message)


def get_last_checkpoint(folder, prefix="epoch"):
    content = os.listdir(folder)
    _re_checkpoint = re.compile(r"^" + prefix + r"_(\d+)$")
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None
        and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(
        folder,
        max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])),
    )


def generate_draft_model_config(
    target_model_path: str, template_config_path: str = None, cache_dir: str = None
):
    """
    Auto-generate draft model config based on target model parameters aligned with template config

    Args:
        target_model_path (str): Path to the target model
        template_config_path (str, optional): Template config file path, defaults to llama3-8B-eagle3.json
        cache_dir (str, optional): Cache directory

    Returns:
        dict: Generated draft model config dictionary
    """
    # Get target model config
    target_config = AutoConfig.from_pretrained(target_model_path, cache_dir=cache_dir)

    # If no template specified, use default llama3-8B-eagle3.json
    if template_config_path is None:
        # Use the script execution directory as base
        import sys

        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        project_root = os.path.dirname(script_dir)  # Go up one level from scripts/
        template_config_path = os.path.join(
            project_root, "configs", "llama3-8B-eagle3.json"
        )

    # Read template config
    with open(template_config_path, "r") as f:
        draft_config = json.load(f)

    # Adjust architecture config based on target model type
    if hasattr(target_config, "model_type"):
        # Default to llama architecture
        draft_config["model_type"] = "llama"

    # Align key parameters
    param_mappings = {
        "vocab_size": "vocab_size",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_key_value_heads": "num_key_value_heads",
        "intermediate_size": "intermediate_size",
        "max_position_embeddings": "max_position_embeddings",
        "rms_norm_eps": "rms_norm_eps",
        "hidden_act": "hidden_act",
        "bos_token_id": "bos_token_id",
        "eos_token_id": "eos_token_id",
        "torch_dtype": "torch_dtype",
    }

    # Copy parameters from target model to draft config
    for target_param, draft_param in param_mappings.items():
        if hasattr(target_config, target_param):
            value = getattr(target_config, target_param)
            # Special handling for torch_dtype to make it JSON serializable
            if target_param == "torch_dtype" and isinstance(value, torch.dtype):
                value = str(value).replace("torch.", "")
            draft_config[draft_param] = value

    # Special handling for some parameters
    # Ensure num_hidden_layers is always 1 (EAGLE3 feature)
    draft_config["num_hidden_layers"] = 1

    # Keep some fixed draft model specific parameters
    draft_config["tie_word_embeddings"] = False
    draft_config["use_cache"] = True

    # If template doesn't have draft_vocab_size, set default
    if "draft_vocab_size" not in draft_config:
        draft_config["draft_vocab_size"] = 32000  # Default value

    return draft_config


def save_draft_model_config(config_dict: dict, output_path: str):
    """
    Save draft model config to file

    Args:
        config_dict (dict): Config dictionary
        output_path (str): Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    print(f"Draft model config saved to: {output_path}")


def create_draft_config_from_target(
    target_model_path: str,
    output_dir: str = None,
    template_config_path: str = None,
    cache_dir: str = None,
):
    """
    Convenient function to create draft model config file from target model

    Args:
        target_model_path (str): Target model path
        output_dir (str, optional): Output directory, defaults to configs folder in current directory
        template_config_path (str, optional): Template config path
        cache_dir (str, optional): Cache directory

    Returns:
        str: Generated config file path
    """
    # Generate config
    rank = dist.get_rank()

    if rank == 0:
        print_with_rank(
            "No draft model config provided, auto-generating from target model..."
        )
        config_dict = generate_draft_model_config(
            target_model_path, template_config_path, cache_dir
        )
    dist.barrier()

    # Determine output path
    if output_dir is None:
        # Use the script execution directory as base
        import sys

        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        project_root = os.path.dirname(script_dir)  # Go up one level from scripts/
        output_dir = os.path.join(project_root, "configs")

    # Extract model name from model path
    model_name = target_model_path.split("/")[-1].lower()
    output_filename = f"{model_name}-eagle3-auto.json"
    output_path = os.path.join(output_dir, output_filename)

    # Save config
    if rank == 0:
        save_draft_model_config(config_dict, output_path)
        print_with_rank(f"Auto-generated draft model config saved to: {output_path}")
    dist.barrier()

    return output_path


def get_full_optimizer_state(optimizer_state_dict: dict):
    """
    Convert optimizer state dict with DTensor to full tensors for saving

    Args:
        optimizer_state_dict (dict): Optimizer state dict possibly containing DTensors
    Returns:
        dict: Optimizer state dict with full tensors
    """
    full_optimizer_state_dict = {
        k: v for k, v in optimizer_state_dict.items() if k != "state"
    }
    if "state" in optimizer_state_dict:
        full_optimizer_state_dict["state"] = {
            param_id: {
                state_key: (
                    state_tensor.full_tensor()
                    if isinstance(state_tensor, torch.distributed.tensor.DTensor)
                    else state_tensor
                )
                for state_key, state_tensor in param_state.items()
            }
            for param_id, param_state in optimizer_state_dict["state"].items()
        }
    return full_optimizer_state_dict


def shard_optimizer_state_with_dtensor(bf16_optimizer, device_mesh):
    """
    Shards the optimizer state tensors of a BF16Optimizer instance using DTensor.

    Args:
        bf16_optimizer (BF16Optimizer): An instance of BF16Optimizer, which contains
            the actual optimizer (e.g., torch.optim.Adam) as its `.optimizer` attribute.
    """

    optim = bf16_optimizer.optimizer

    for group in optim.param_groups:
        for p in group["params"]:
            if not isinstance(p, DTensor):
                continue

            state = optim.state.get(p, None)
            if state is None:
                continue

            mesh = device_mesh
            placements = (Shard(dim=0),)

            for k, v in list(state.items()):
                if k == "step":
                    continue

                if isinstance(v, DTensor):
                    continue

                if not isinstance(v, torch.Tensor):
                    continue

                state[k] = distribute_tensor(
                    v.to(p.device), device_mesh=mesh, placements=placements
                )
