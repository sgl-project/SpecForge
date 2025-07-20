import os
import json
import glob
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig


class TargetHead(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_path)
        self.fc = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    @torch.no_grad()
    def load_weights(self, model_path, lm_head_key: str = "lm_head.weight"):
        if os.path.exists(model_path):
            self.model_path = model_path
        else:
            self.model_path = snapshot_download(repo_id=model_path)

        # model_path is a local directory
        # check if there is file ending with index.json
        glob_path = os.path.join(self.model_path, "*.index.json")
        index_json_path = glob.glob(glob_path)

        if len(index_json_path) == 0:
            raise FileNotFoundError(f"No index.json file found in {self.model_path}")
        if len(index_json_path) > 1:
            raise FileNotFoundError(
                f"Multiple index.json files found in {self.model_path}"
            )
        index_json_path = index_json_path[0]

        with open(index_json_path, "r") as f:
            index_json = json.load(f)
        ckpt_file = index_json["weight_map"][lm_head_key]

        if ckpt_file.endswith(".safetensors"):
            with safe_open(
                os.path.join(self.model_path, ckpt_file), framework="pt"
            ) as f:
                lm_head = f.get_tensor(lm_head_key)
        else:
            state_dict = torch.load(os.path.join(self.model_path, ckpt_file))
            lm_head = state_dict[lm_head_key]
        self.fc.weight.copy_(lm_head)
            

    def freeze_weights(self):
        for param in self.fc.parameters():
            param.requires_grad = False

    def forward(self, hidden_states):
        return self.fc(hidden_states)