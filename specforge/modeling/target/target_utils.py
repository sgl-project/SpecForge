import glob
import json
import os
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig


class TargetEmbeddingsAndHead(nn.Module):
    """
    Efficiently loads only the embedding layer and lm_head from a pretrained model.
    Avoids loading the full model into memory.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        embed_key: str = "model.embed_tokens.weight",
        lm_head_key: str = "lm_head.weight",
        cache_dir: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "TargetEmbeddingsAndHead":

        # 1. Load Config
        config = AutoConfig.from_pretrained(model_path, cache_dir=cache_dir)
        instance = cls(config)

        # 2. Resolve Model Path (Handle Hub)
        local_model_path = model_path
        if not os.path.exists(local_model_path):
            try:
                local_model_path = snapshot_download(
                    repo_id=model_path, cache_dir=cache_dir
                )
            except:
                pass  # Maybe it's a local path that looks like a repo ID but doesn't exist?

        # 3. Load Weights Efficiently
        instance._load_weights(local_model_path, embed_key, lm_head_key)

        # 4. Move to Device & Freeze
        instance.to(device=device, dtype=dtype)
        instance.eval()
        instance.requires_grad_(False)

        return instance

    def _load_weights(self, model_path: str, embed_key: str, lm_head_key: str):
        # Locate index.json
        index_files = glob.glob(os.path.join(model_path, "*.index.json"))

        weight_map = {}
        if index_files:
            # Sharded Checkpoint
            with open(index_files[0], "r") as f:
                index = json.load(f)

            # Find which file contains our keys
            weight_map = index.get("weight_map", {})
            files_to_load = {}

            if embed_key in weight_map:
                files_to_load[embed_key] = weight_map[embed_key]
            else:
                # Fallback: sometimes keys are prefixed differently?
                print(
                    f"Warning: {embed_key} not found in weight_map. Keys available: {list(weight_map.keys())[:5]}..."
                )

            if lm_head_key in weight_map:
                files_to_load[lm_head_key] = weight_map[lm_head_key]

            # Load specific files
            for key, filename in files_to_load.items():
                file_path = os.path.join(model_path, filename)
                self._load_key_from_file(file_path, key)

        else:
            # Non-sharded Checkpoint (single file)
            # Try finding .safetensors or .bin
            safetensors = glob.glob(os.path.join(model_path, "*.safetensors"))
            bins = glob.glob(os.path.join(model_path, "*.bin"))

            target_file = None
            if safetensors:
                target_file = safetensors[0]
            elif bins:
                target_file = bins[0]

            if target_file:
                self._load_key_from_file(target_file, embed_key)
                self._load_key_from_file(target_file, lm_head_key)
            else:
                raise FileNotFoundError(f"No checkpoint file found in {model_path}")

    def _load_key_from_file(self, file_path: str, key: str):
        tensor = None
        if file_path.endswith(".safetensors"):
            with safe_open(file_path, framework="pt") as f:
                if key in f.keys():
                    tensor = f.get_tensor(key)
        else:
            # torch.load loads full dict, less efficient but works
            state_dict = torch.load(file_path, map_location="cpu")
            if key in state_dict:
                tensor = state_dict[key]
                del state_dict  # Free immediately

        if tensor is not None:
            if key.endswith("embed_tokens.weight"):
                self.embed_tokens.weight.data.copy_(tensor)
                print(f"Loaded embedding weights from {file_path}")
            elif key.endswith("lm_head.weight"):
                self.lm_head.weight.data.copy_(tensor)
                print(f"Loaded lm_head weights from {file_path}")
        else:
            print(f"Warning: Key {key} not found in {file_path}")
