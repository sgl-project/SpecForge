import json
import netrc
import os
import re
from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.distributed as dist
from transformers import AutoConfig, PretrainedConfig


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
    print(f"rank {dist.get_rank()}: {message}")


PREFIX_CHECKPOINT_DIR = "epoch"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"_(\d+)$")


def get_last_checkpoint(folder):
    content = os.listdir(folder)
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
        # Get current file directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
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
        "rope_theta": "rope_theta",
        "hidden_act": "hidden_act",
        "bos_token_id": "bos_token_id",
        "eos_token_id": "eos_token_id",
        "torch_dtype": "torch_dtype",
    }

    # Copy parameters from target model to draft config
    for target_param, draft_param in param_mappings.items():
        if hasattr(target_config, target_param):
            value = getattr(target_config, target_param)
            draft_config[draft_param] = value

    # Special handling for some parameters
    # Ensure num_hidden_layers is always 1 (EAGLE3 feature)
    draft_config["num_hidden_layers"] = 1

    # Copy attention_dropout if target model has it
    if hasattr(target_config, "attention_dropout"):
        draft_config["attention_dropout"] = target_config.attention_dropout

    # Copy head_dim if target model has it
    if hasattr(target_config, "head_dim"):
        draft_config["head_dim"] = target_config.head_dim

    # Copy attention_bias if target model has it
    if hasattr(target_config, "attention_bias"):
        draft_config["attention_bias"] = target_config.attention_bias

    # Copy rope_scaling if target model has it
    if hasattr(target_config, "rope_scaling"):
        draft_config["rope_scaling"] = target_config.rope_scaling

    # Copy sliding_window related configs if target model has them
    if hasattr(target_config, "sliding_window"):
        draft_config["sliding_window"] = target_config.sliding_window
    if hasattr(target_config, "use_sliding_window"):
        draft_config["use_sliding_window"] = target_config.use_sliding_window
    if hasattr(target_config, "max_window_layers"):
        draft_config["max_window_layers"] = target_config.max_window_layers

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
    config_dict = generate_draft_model_config(
        target_model_path, template_config_path, cache_dir
    )

    # Determine output path
    if output_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        output_dir = os.path.join(project_root, "configs")

    # Extract model name from model path
    model_name = target_model_path.split("/")[-1].lower()
    output_filename = f"{model_name}-eagle3-auto.json"
    output_path = os.path.join(output_dir, output_filename)

    # Save config
    save_draft_model_config(config_dict, output_path)

    return output_path
