import glob
import json
import os
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig

from specforge.utils import padding


class TargetHead(nn.Module):
    def __init__(self, model_path, trust_remote_code: bool = False):
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        self.text_config = getattr(self.config, "text_config", self.config)

        self.hidden_size = self.text_config.hidden_size
        self.vocab_size = self.text_config.vocab_size

        self.fc = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        # Early vocab projection support (target-to-draft mapping)
        self.t2d_mapping = None
        self.logits_chunk_size = 2048

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        lm_head_key: str = "lm_head.weight",
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> "TargetHead":
        target_head = cls(model_path, trust_remote_code=trust_remote_code)
        target_head.load_weights(
            model_path=model_path,
            lm_head_key=lm_head_key,
            cache_dir=cache_dir,
        )
        target_head.freeze_weights()
        target_head = target_head.eval().cuda().to(torch.bfloat16)
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

    def set_vocab_mapping(self, t2d_mapping):
        """Set target-to-draft vocab mapping for early projection.

        When set, forward() will project hidden_states to draft_vocab_size
        instead of full vocab_size, reducing peak memory from
        (batch, seq_len, 248320) = ~7.7 GB to (batch, seq_len, 32000) = ~1 GB.
        """
        self.t2d_mapping = t2d_mapping

    def set_logits_chunk_size(self, chunk_size):
        """Set chunk size for chunked logits computation.

        Computes logits in chunks along the sequence dimension to limit peak
        memory to (batch, chunk_size, vocab_size) instead of (batch, seq_len, vocab_size).
        """
        self.logits_chunk_size = chunk_size

    def forward(self, hidden_states):
        if self.t2d_mapping is not None:
            return self._forward_chunked_projected(hidden_states)
        return self.fc(hidden_states), None

    @torch.no_grad()
    def _forward_chunked_projected(self, hidden_states):
        """Project hidden_states to draft vocab using chunked computation.

        Also computes target_in_draft_mask: for each position, whether the
        argmax token (in full vocab) falls within the draft vocab.

        Without this optimization:
          fc output = (batch, 16384, 248320) ≈ 7.7 GB  →  OOM
        With chunked projection (chunk_size=2048):
          per-chunk peak = (batch, 2048, 248320) ≈ 0.96 GB
          final output   = (batch, 16384, 32000)  ≈ 1.0 GB

        Returns:
            tuple: (projected_logits, target_in_draft_mask)
              - projected_logits: (batch, seq_len, draft_vocab_size)
              - target_in_draft_mask: (batch, seq_len) bool tensor
        """
        t2d = self.t2d_mapping
        if t2d.device != hidden_states.device:
            t2d = t2d.to(hidden_states.device)
            self.t2d_mapping = t2d  # cache on correct device

        seq_len = hidden_states.shape[1]
        chunk_size = self.logits_chunk_size if self.logits_chunk_size > 0 else seq_len

        projected_chunks = []
        mask_chunks = []
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            chunk = hidden_states[:, start:end, :]
            full_logits = self.fc(chunk)              # (batch, chunk_len, full_vocab)
            # Compute mask: is argmax token in draft vocab?
            argmax_tokens = full_logits.argmax(dim=-1)  # (batch, chunk_len)
            chunk_mask = t2d[argmax_tokens]              # (batch, chunk_len) bool
            mask_chunks.append(chunk_mask)
            # Project to draft vocab
            draft_logits = full_logits[:, :, t2d]     # (batch, chunk_len, draft_vocab)
            projected_chunks.append(draft_logits)
            del full_logits, argmax_tokens

        projected = torch.cat(projected_chunks, dim=1)  # (batch, seq_len, draft_vocab)
        target_in_draft_mask = torch.cat(mask_chunks, dim=1)  # (batch, seq_len)
        return projected, target_in_draft_mask

    def preprocess(self, input_ids, target, loss_mask):
        # apply pading
        target = padding(target, left=False)
        input_ids = padding(input_ids, left=False)
        loss_mask = loss_mask[..., None]
        return input_ids, target, loss_mask
