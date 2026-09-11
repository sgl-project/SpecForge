import glob
import json
import os
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open

from specforge.modeling.target.target_utils import (
    load_target_config,
    target_hidden_size,
    target_vocab_size,
)
from specforge.utils import get_local_device, padding


class TargetHead(nn.Module):
    def __init__(
        self,
        model_path,
        trust_remote_code: bool = False,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.config = load_target_config(
            model_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        self.hidden_size = target_hidden_size(self.config)
        self.vocab_size = target_vocab_size(self.config)

        self.fc = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        lm_head_key: str = "lm_head.weight",
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> "TargetHead":
        target_head = cls(
            model_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        target_head.load_weights(
            model_path=model_path,
            lm_head_key=lm_head_key,
            cache_dir=cache_dir,
        )
        target_head.freeze_weights()
        target_head = target_head.eval().to(
            device=get_local_device(), dtype=torch.bfloat16
        )
        return target_head

    @torch.no_grad()
    def load_weights(
        self,
        model_path,
        lm_head_key: str = "lm_head.weight",
        cache_dir: Optional[str] = None,
    ):
        if os.path.exists(model_path):
            self.model_path = model_path
        else:
            self.model_path = snapshot_download(repo_id=model_path, cache_dir=cache_dir)

        index_json_paths = glob.glob(
            os.path.join(self.model_path, "*.index.json")
        )
        if len(index_json_paths) > 1:
            raise FileNotFoundError(
                f"Multiple index.json files found in {self.model_path}"
            )

        if index_json_paths:
            index_json_path = index_json_paths[0]
            with open(index_json_path, encoding="utf-8") as config_file:
                weight_map = json.load(config_file).get("weight_map", {})
            if lm_head_key not in weight_map:
                raise RuntimeError(
                    f"Target head key {lm_head_key!r} is missing from "
                    f"{index_json_path}"
                )
            checkpoint_path = os.path.join(
                self.model_path, weight_map[lm_head_key]
            )
        else:
            candidates = (
                os.path.join(self.model_path, "model.safetensors"),
                os.path.join(self.model_path, "pytorch_model.bin"),
            )
            checkpoint_path = next(
                (path for path in candidates if os.path.isfile(path)),
                None,
            )
            if checkpoint_path is None:
                raise FileNotFoundError(
                    "No index.json, model.safetensors, or pytorch_model.bin "
                    f"found in {self.model_path}"
                )

        if checkpoint_path.endswith(".safetensors"):
            with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
                if lm_head_key not in checkpoint.keys():
                    raise RuntimeError(
                        f"Target head key {lm_head_key!r} is missing from "
                        f"{checkpoint_path}"
                    )
                lm_head = checkpoint.get_tensor(lm_head_key)
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if lm_head_key not in state_dict:
                raise RuntimeError(
                    f"Target head key {lm_head_key!r} is missing from "
                    f"{checkpoint_path}"
                )
            lm_head = state_dict[lm_head_key]
        if tuple(lm_head.shape) != tuple(self.fc.weight.shape):
            raise RuntimeError(
                f"Target head {lm_head_key!r} has shape {tuple(lm_head.shape)}, "
                f"expected {tuple(self.fc.weight.shape)}"
            )
        self.fc.weight.copy_(lm_head)

    def freeze_weights(self):
        for param in self.fc.parameters():
            param.requires_grad = False

    def forward(self, hidden_states):
        return self.fc(hidden_states)

    def preprocess(self, input_ids, target, loss_mask):
        # apply pading
        target = padding(target, left=False)
        input_ids = padding(input_ids, left=False)
        loss_mask = loss_mask[..., None]
        return input_ids, target, loss_mask
