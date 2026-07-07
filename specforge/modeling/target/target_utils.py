import gc
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
    Handles safetensors slicing and Weight Tying correctly.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        # Support for MLLMs with separate text_config
        if hasattr(config, "text_config"):
            self.embed_tokens = nn.Embedding(
                config.text_config.vocab_size,
                config.text_config.hidden_size,
                padding_idx=config.text_config.pad_token_id,
            )
            self.lm_head = nn.Linear(
                config.text_config.hidden_size,
                config.text_config.vocab_size,
                bias=False,
            )
        else:
            self.embed_tokens = nn.Embedding(
                config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
            )
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        embed_key: Optional[str] = None,
        lm_head_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = False,
    ) -> "TargetEmbeddingsAndHead":

        # 1. Load Config
        config = AutoConfig.from_pretrained(
            model_path, cache_dir=cache_dir, trust_remote_code=trust_remote_code
        )
        instance = cls(config)

        if embed_key is None:
            embed_key = "model.embed_tokens.weight"
        if lm_head_key is None:
            lm_head_key = "lm_head.weight"

        # 2. Resolve Model Path
        local_model_path = model_path
        if not os.path.exists(local_model_path):
            try:
                local_model_path = snapshot_download(
                    repo_id=model_path,
                    cache_dir=cache_dir,
                    allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model"],
                )
            except Exception as e:
                print(f"Warning: Snapshot download failed or path check failed: {e}")

        # 3. Handle Weight Tying
        tie_weights = getattr(config, "tie_word_embeddings", False)

        # 4. Load Weights
        instance._load_weights(local_model_path, embed_key, lm_head_key, tie_weights)

        # 5. Move to Device & Freeze
        instance.to(device=device, dtype=dtype)
        instance.eval()
        instance.requires_grad_(False)

        return instance

    def _load_weights(
        self, model_path: str, embed_key: str, lm_head_key: str, tie_weights: bool
    ):
        index_files = glob.glob(os.path.join(model_path, "*.index.json"))
        weight_map = {}
        files_to_load = {}

        if index_files:
            with open(index_files[0], "r") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})

            # Auto-detect the embedding key if the supplied/default key is missing.
            # This handles plain LLMs (model.embed_tokens.weight), MLLMs
            # (model.language_model.model.embed_tokens.weight), and renamed checkpoints.
            candidate_embed_keys = [
                embed_key,
                "model.embed_tokens.weight",
                "embed_tokens.weight",
                "model.language_model.model.embed_tokens.weight",
                "model.language_model.embed_tokens.weight",
                "language_model.model.embed_tokens.weight",
                "language_model.embed_tokens.weight",
                "model.model.embed_tokens.weight",
            ]
            resolved_embed_key = None
            for key in candidate_embed_keys:
                if key in weight_map:
                    resolved_embed_key = key
                    break
            if resolved_embed_key is None:
                raise ValueError(
                    f"Embedding key '{embed_key}' not found in weight map and no "
                    f"candidate embed key matched. Available keys (first 20): "
                    f"{list(weight_map.keys())[:20]}"
                )
            if resolved_embed_key != embed_key:
                print(
                    f"Resolved embedding key '{embed_key}' -> '{resolved_embed_key}'"
                )
            files_to_load[resolved_embed_key] = weight_map[resolved_embed_key]
            embed_key = resolved_embed_key

            if not tie_weights:
                candidate_head_keys = [
                    lm_head_key,
                    "lm_head.weight",
                    "model.lm_head.weight",
                    "model.language_model.lm_head.weight",
                    "language_model.lm_head.weight",
                ]
                resolved_head_key = None
                for key in candidate_head_keys:
                    if key in weight_map:
                        resolved_head_key = key
                        break
                if resolved_head_key is None:
                    print(
                        f"Warning: {lm_head_key} not found. Ensure model doesn't use tied weights manually."
                    )
                else:
                    if resolved_head_key != lm_head_key:
                        print(
                            f"Resolved lm_head key '{lm_head_key}' -> '{resolved_head_key}'"
                        )
                    files_to_load[resolved_head_key] = weight_map[resolved_head_key]
                    lm_head_key = resolved_head_key
        else:
            safetensors = glob.glob(os.path.join(model_path, "*.safetensors"))
            bins = glob.glob(os.path.join(model_path, "*.bin"))
            target_file = safetensors[0] if safetensors else (bins[0] if bins else None)

            if not target_file:
                raise FileNotFoundError("No checkpoint found.")

            # Read the available keys so we can auto-detect embed/lm_head names
            # in single-file checkpoints too.
            if target_file.endswith(".safetensors"):
                with safe_open(target_file, framework="np") as f:
                    available_keys = set(f.keys())
            else:
                # For .bin files we fall back to the provided keys; auto-detection
                # would require loading the whole state dict up front.
                available_keys = None

            candidate_embed_keys = [
                embed_key,
                "model.embed_tokens.weight",
                "embed_tokens.weight",
                "model.language_model.model.embed_tokens.weight",
                "model.language_model.embed_tokens.weight",
                "language_model.model.embed_tokens.weight",
                "language_model.embed_tokens.weight",
                "model.model.embed_tokens.weight",
            ]
            resolved_embed_key = None
            if available_keys is not None:
                for key in candidate_embed_keys:
                    if key in available_keys:
                        resolved_embed_key = key
                        break
            else:
                resolved_embed_key = embed_key

            if resolved_embed_key is None:
                raise ValueError(
                    f"Embedding key '{embed_key}' not found in checkpoint and no "
                    f"candidate embed key matched."
                )
            if resolved_embed_key != embed_key:
                print(
                    f"Resolved embedding key '{embed_key}' -> '{resolved_embed_key}'"
                )
            files_to_load[resolved_embed_key] = os.path.basename(target_file)
            embed_key = resolved_embed_key

            if not tie_weights:
                candidate_head_keys = [
                    lm_head_key,
                    "lm_head.weight",
                    "model.lm_head.weight",
                    "model.language_model.lm_head.weight",
                    "language_model.lm_head.weight",
                ]
                resolved_head_key = None
                if available_keys is not None:
                    for key in candidate_head_keys:
                        if key in available_keys:
                            resolved_head_key = key
                            break
                else:
                    resolved_head_key = lm_head_key

                if resolved_head_key is None:
                    print(
                        f"Warning: {lm_head_key} not found. Ensure model doesn't use tied weights manually."
                    )
                else:
                    if resolved_head_key != lm_head_key:
                        print(
                            f"Resolved lm_head key '{lm_head_key}' -> '{resolved_head_key}'"
                        )
                    files_to_load[resolved_head_key] = os.path.basename(target_file)
                    lm_head_key = resolved_head_key

        loaded_keys = set()

        file_to_keys_map = {}
        for key, filename in files_to_load.items():
            full_path = os.path.join(model_path, filename)
            if full_path not in file_to_keys_map:
                file_to_keys_map[full_path] = []
            file_to_keys_map[full_path].append(key)

        for file_path, keys in file_to_keys_map.items():
            self._load_file_content(file_path, keys, embed_key, lm_head_key)
            loaded_keys.update(keys)

        if tie_weights:
            print(
                "Weight tying detected: Sharing weights between Embeddings and LM Head."
            )
            self.lm_head.weight = self.embed_tokens.weight

        if embed_key not in loaded_keys:
            raise RuntimeError("Failed to load embeddings.")
        if not tie_weights and lm_head_key not in loaded_keys:
            print(
                "Warning: LM Head weights were not found (and tie_weights is False). Head is random."
            )

    def _load_file_content(
        self,
        file_path: str,
        keys_to_extract: list,
        target_embed_key: str,
        target_head_key: str,
    ):
        """Helper to load specific keys from a file"""
        print(f"Loading {keys_to_extract} from {os.path.basename(file_path)}...")

        state_dict_part = {}

        if file_path.endswith(".safetensors"):
            with safe_open(file_path, framework="pt") as f:
                for k in keys_to_extract:
                    if k in f.keys():
                        state_dict_part[k] = f.get_tensor(k)
        else:
            print(
                f"Warning: Loading .bin file {os.path.basename(file_path)} into RAM. Convert to safetensors for efficiency."
            )
            full_state = torch.load(file_path, map_location="cpu")
            for k in keys_to_extract:
                if k in full_state:
                    state_dict_part[k] = full_state[k]
            del full_state
            gc.collect()

        for k, tensor in state_dict_part.items():
            if k == target_embed_key:
                self.embed_tokens.weight.data.copy_(tensor)
                print(" -> Loaded Embeddings")
            elif k == target_head_key:
                if tensor.shape == self.lm_head.weight.data.shape:
                    self.lm_head.weight.data.copy_(tensor)
                    print(" -> Loaded LM Head")
                else:
                    raise RuntimeError(
                        f"Shape mismatch for {k}. Expected {self.lm_head.weight.shape}, got {tensor.shape}"
                    )
